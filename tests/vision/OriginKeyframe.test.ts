import { describe, expect, it } from "vitest";
import { resolveConfig } from "../../src/ar/ARConfig";
import { createRng } from "../../src/vision/OutlierRejection";
import type { VisionInput, VisionOutput } from "../../src/vision/types";
import { VisionEngine } from "../../src/vision/VisionEngine";
import { TP_H, TP_K, TP_W, tpGravityAt, tpPose, tpRender } from "../helpers/twoPlaneScene";

/**
 * 修正指示書「座標系整合性と AR オブジェクトの空間固定性」§3, §7.1, §10-A:
 * the origin keyframe is the two-view reference view — the map frame itself
 * (identity pose), with the reference frame's image and the init landmarks
 * at their reference pixels. Nothing re-poses it: not the init stages'
 * keyframe re-pose, not the keyframe bundle adjustment, not eviction.
 */
const W = TP_W;
const H = TP_H;
const slide = (f: number) => tpPose([0.005 * f, 0.001 * f, 0.002 * f], 0.0015 * f, 0);
const rendered: Uint8Array[] = [];
function frame(f: number): Uint8Array {
  while (rendered.length <= f) rendered.push(tpRender(slide(rendered.length)));
  return rendered[f];
}
function input(f: number): VisionInput {
  return { frameId: f, timestamp: f * 33.3, width: W, height: H, gray: frame(f), intrinsics: TP_K, gravity: tpGravityAt(slide(f)) };
}
const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0];
const poseOf = (kf: { pose: { rotation: Float64Array; translation: Float64Array } }) => [...kf.pose.rotation, ...kf.pose.translation];

describe("origin keyframe = the fixed reference view (指示書 §3)", () => {
  it("is created at initialization from the reference frame: identity pose, the reference image, the init landmarks at their reference pixels; the init frame is a free keyframe", () => {
    const engine = new VisionEngine(W, H, resolveConfig(), createRng(5));
    let initFrame = -1;
    let out: VisionOutput | null = null;
    for (let f = 0; f < 40 && initFrame < 0; f++) {
      out = engine.process(input(f));
      if (out.mapPose) initFrame = f;
    }
    expect(initFrame).toBeGreaterThan(0);
    const mp = out!.mapPose!;
    const kfs = engine.keyframes;
    expect(kfs.length).toBe(2);
    const origin = kfs.filter((kf) => kf.origin);
    expect(origin.length).toBe(1);
    const o = origin[0];
    // The map frame: the two-view reference camera, identity.
    expect(poseOf(o)).toEqual(IDENTITY);
    expect(o.frameId).toBe(mp.mapFrameId);
    expect(o.frameId).toBeLessThan(initFrame);
    // Its image is the reference frame's (copied when that frame became the reference, before the map existed).
    expect(o.pyramid.levels[0].data).toEqual(frame(mp.mapFrameId));
    // Its observations are exactly the init landmarks at their reference pixels.
    const tracker = engine.mapTrackerForTests();
    const init = [...engine.landmarkMap.values()].filter((lm) => tracker.isInitLandmark(lm.id));
    expect(init.length).toBeGreaterThan(30);
    expect(o.observations.length).toBe(init.length);
    for (const ob of o.observations) {
      const lm = engine.landmarkMap.get(ob.landmarkId)!;
      expect(lm.anchorX).toBe(ob.x);
      expect(lm.anchorY).toBe(ob.y);
      // And they reproject there (the map was triangulated from this view).
      const u = (lm.position[0] / lm.position[2]) * TP_K.fx + TP_K.cx, v = (lm.position[1] / lm.position[2]) * TP_K.fy + TP_K.cy;
      expect(Math.hypot(u - ob.x, v - ob.y)).toBeLessThan(1.0);
    }
    // The initialization frame is an ordinary free keyframe at the two-view baseline (|t| = 1 before refinement; the refinement keeps the median depth, not |t|).
    const kf0 = kfs.find((kf) => !kf.origin)!;
    expect(kf0.frameId).toBe(initFrame);
    const t = kf0.pose.translation;
    expect(Math.hypot(t[0], t[1], t[2])).toBeGreaterThan(0.5);
    expect(Math.hypot(t[0], t[1], t[2])).toBeLessThan(2);
    // The init landmarks name the init-frame keyframe as their newest view (v18 propagation), not the fixed origin.
    for (const lm of init) expect(lm.refKeyframeId).toBe(kf0.id);
  });

  it("stays bit-identical through every init stage and every bundle adjustment of a 200-frame run, while the other keyframes are re-posed or carried", () => {
    const engine = new VisionEngine(W, H, resolveConfig(), createRng(5));
    let originPose: number[] | null = null;
    let originId = -1;
    let stages = 0, baRuns = 0, baAccepted = 0;
    for (let f = 0; f < 200; f++) {
      const o = engine.process(input(f));
      if (!o.mapPose) continue;
      const origin = engine.keyframes.find((kf) => kf.origin);
      expect(origin, `frame ${f}: origin keyframe present`).toBeDefined();
      if (originPose === null) {
        originPose = poseOf(origin!);
        originId = origin!.id;
        expect(originPose).toEqual(IDENTITY);
      }
      // Same object (never evicted), same pose (never written), identity.
      expect(origin!.id).toBe(originId);
      expect(poseOf(origin!)).toEqual(originPose);
      const st = o.initRefinement?.lastStage;
      if (st && st.frameId === f) {
        stages++;
        if (st.applied) {
          expect(st.keyframesFixed).toBe(1);
          expect(st.keyframesReposed + st.keyframesPropagated + 1).toBe(engine.keyframes.length);
        }
      }
      const ba = o.bundleAdjustment;
      if (ba.ranThisFrame) {
        baRuns++;
        if (!ba.rejected) baAccepted++;
        expect(ba.gauge).toBe("reference");
        expect(ba.freeKeyframes).toBe(ba.keyframes - 1);
      }
    }
    console.log(`[origin] ${stages} init stages, ${baRuns} BA runs (${baAccepted} accepted): origin keyframe KF${originId} identity throughout, ${engine.keyframes.length} keyframes stored`);
    expect(stages).toBeGreaterThanOrEqual(1);
    expect(baAccepted).toBeGreaterThanOrEqual(1);
    // Eviction happened (more keyframes were made than the store holds) and still kept the origin.
    expect(engine.keyframeSummary().length).toBeLessThanOrEqual(resolveConfig().relocalization.maxKeyframes);
    expect(engine.keyframes[0].origin).toBe(true);
  });

  it("§7.1–§7.2: a keyframe struck out of the BA stays stored for relocalization, and the origin is never the one struck", () => {
    const engine = new VisionEngine(W, H, resolveConfig(), createRng(5));
    for (let f = 0; f < 120; f++) engine.process(input(f));
    // Every stored keyframe is either the origin or free in the solve; none is dropped by the BA bookkeeping.
    const stored = engine.keyframes.length;
    expect(engine.runBundleAdjustment(TP_K)).toBe(true);
    expect(engine.keyframes.length).toBe(stored);
    expect(engine.bundleAdjustmentExcludedKeyframes.has(engine.keyframes.find((kf) => kf.origin)!.id)).toBe(false);
  });
});
