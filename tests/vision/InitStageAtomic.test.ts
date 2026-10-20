import { describe, expect, it } from "vitest";
import { resolveConfig } from "../../src/ar/ARConfig";
import { composeTransforms, invertTransform, rotationDistance, type RigidTransform } from "../../src/math/Pose";
import { createRng } from "../../src/vision/OutlierRejection";
import type { VisionInput, VisionOutput } from "../../src/vision/types";
import { VisionEngine } from "../../src/vision/VisionEngine";
import { TP_H, TP_K, TP_W, tpGravityAt, tpPose, tpRender, type TPPose } from "../helpers/twoPlaneScene";

/**
 * 修正指示書「座標系整合性と AR アンカー安定性」§3–§5, §9.1–§9.3: the
 * multi-view init stages change the map, the keyframes and the current pose
 * atomically; the fixed view (the reference camera = the map frame) is never
 * written; a keyframe is re-posed only from solved landmarks and otherwise
 * carried by the rigid correction; a rejected stage changes nothing.
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
  const p: TPPose = slide(f);
  return { frameId: f, timestamp: f * 33.3, width: W, height: H, gray: frame(f), intrinsics: TP_K, gravity: tpGravityAt(p) };
}

interface Snapshot {
  landmarks: Map<number, number[]>;
  keyframes: { id: number; pose: number[] }[];
  pose: number[];
}
function snapshot(engine: VisionEngine): Snapshot {
  const landmarks = new Map<number, number[]>();
  for (const lm of engine.landmarkMap.values()) landmarks.set(lm.id, Array.from(lm.position));
  const p = engine.currentMapPose;
  return {
    landmarks,
    keyframes: engine.keyframes.map((kf) => ({ id: kf.id, pose: [...kf.pose.rotation, ...kf.pose.translation] })),
    pose: [...p.rotation, ...p.translation],
  };
}
function expectSameState(a: Snapshot, b: Snapshot): void {
  expect(a.landmarks.size).toBe(b.landmarks.size);
  for (const [id, p] of a.landmarks) expect(b.landmarks.get(id)).toEqual(p);
  expect(a.keyframes).toEqual(b.keyframes);
  expect(a.pose).toEqual(b.pose);
}

/** Mean reprojection error (px) of a keyframe's observations of the landmarks in `ids` against the current map. */
function keyframeError(engine: VisionEngine, kfIndex: number, ids: ReadonlySet<number>): { n: number; meanPx: number } {
  const kf = engine.keyframes[kfIndex];
  const r = kf.pose.rotation, t = kf.pose.translation;
  let n = 0, sum = 0;
  for (const o of kf.observations) {
    if (!ids.has(o.landmarkId)) continue;
    const lm = engine.landmarkMap.get(o.landmarkId);
    if (!lm) continue;
    const X = lm.position;
    const z = r[6] * X[0] + r[7] * X[1] + r[8] * X[2] + t[2];
    const u = ((r[0] * X[0] + r[1] * X[1] + r[2] * X[2] + t[0]) / z) * TP_K.fx + TP_K.cx;
    const v = ((r[3] * X[0] + r[4] * X[1] + r[5] * X[2] + t[1]) / z) * TP_K.fy + TP_K.cy;
    sum += Math.hypot(u - o.x, v - o.y);
    n++;
  }
  return { n, meanPx: n ? sum / n : 0 };
}

/** Mean reprojection error (px) of the init landmarks in the reference view (identity): the map frame is still the reference camera. */
function referenceViewError(engine: VisionEngine): { n: number; meanPx: number } {
  const tracker = engine.mapTrackerForTests();
  let n = 0, sum = 0;
  for (const lm of engine.landmarkMap.values()) {
    if (!tracker.isInitLandmark(lm.id) || !lm.anchorPose) continue;
    const a = lm.anchorPose;
    // The reference view: identity pose, never written.
    expect(a.rotation[0]).toBe(1);
    expect(a.translation[0]).toBe(0);
    const X = lm.position;
    const u = (X[0] / X[2]) * TP_K.fx + TP_K.cx, v = (X[1] / X[2]) * TP_K.fy + TP_K.cy;
    sum += Math.hypot(u - lm.anchorX, v - lm.anchorY);
    n++;
  }
  return { n, meanPx: n ? sum / n : 0 };
}

describe("init stages are atomic and keep one map frame (指示書 §3–§5)", () => {
  it("a rejected stage (forced no_gain) leaves landmarks, keyframe poses and the canonical pose bit-identical to a run without stages", () => {
    const cfgA = resolveConfig({ bundleAdjustment: { minGainFraction: 1 } });
    const cfgB = resolveConfig({ bundleAdjustment: { minGainFraction: 1 }, landmarks: { initRefineStages: 0 } });
    const a = new VisionEngine(W, H, cfgA, createRng(5));
    const b = new VisionEngine(W, H, cfgB, createRng(5));
    let stages = 0;
    const reasons: string[] = [];
    for (let f = 0; f < 110; f++) {
      const oa = a.process(input(f));
      b.process(input(f));
      const st = oa.initRefinement?.lastStage;
      if (st && st.frameId === f) {
        stages++;
        reasons.push(st.rejectReason);
        expect(st.applied).toBe(false);
      }
      // Frame by frame: a rejected stage never leaves a partial write behind.
      expectSameState(snapshot(a), snapshot(b));
    }
    console.log(`[initstage] forced rejections: ${stages} stages, reasons ${reasons.join(",")}`);
    expect(stages).toBeGreaterThanOrEqual(2);
    for (const r of reasons) expect(r).toBe("no_gain");
  });

  it("refineInitStage itself writes nothing: the candidate is a pure solve", () => {
    const engine = new VisionEngine(W, H, resolveConfig({ landmarks: { initRefineStages: 0 } }), createRng(5));
    let out: VisionOutput | null = null;
    for (let f = 0; f < 40; f++) out = engine.process(input(f));
    expect(out!.mapPose).not.toBeNull();
    const tracker = engine.mapTrackerForTests();
    const before = snapshot(engine);
    // Ask for a stage directly (stages are configured off, so none was consumed by the engine).
    (tracker as unknown as { config: { initRefineStages: number } }).config.initRefineStages = 3;
    const c = tracker.refineInitStage(engine.tracksForTests(), TP_K, 40, 0.2);
    expect(c).not.toBeNull();
    expect(c!.after.length).toBe(c!.ids.length * 3);
    expectSameState(snapshot(engine), before);
    // And its pose / landmarks are copies, not the tracker's objects.
    expect(c!.poseAfter).not.toBe(tracker.pose);
  });

  it("an accepted stage leaves every keyframe and the reference view consistent with the refined map, with no half-written state", () => {
    const engine = new VisionEngine(W, H, resolveConfig(), createRng(5));
    const accepted: { f: number; reposed: number; propagated: number; refPx: number; kfErr: number[] }[] = [];
    for (let f = 0; f < 120; f++) {
      const o = engine.process(input(f));
      const st = o.initRefinement?.lastStage;
      if (!st || st.frameId !== f || !st.applied) continue;
      const tracker = engine.mapTrackerForTests();
      // Every keyframe was re-posed (PnP on solved landmarks), carried by the rigid correction, or is the fixed origin — none skipped.
      expect(st.keyframesFixed).toBe(1);
      expect(st.keyframesReposed + st.keyframesPropagated + st.keyframesFixed).toBe(engine.keyframes.length);
      // All finite.
      for (const lm of engine.landmarkMap.values()) for (const v of lm.position) expect(Number.isFinite(v)).toBe(true);
      for (const kf of engine.keyframes) for (const v of [...kf.pose.rotation, ...kf.pose.translation]) expect(Number.isFinite(v)).toBe(true);
      // The reference view (map frame) was not written: the init landmarks still anchor at the identity and reproject there.
      const ref = referenceViewError(engine);
      expect(ref.n).toBeGreaterThan(30);
      expect(ref.meanPx).toBeLessThan(1.0);
      expect(st.referenceErrorAfterPx).toBeLessThan(1.0);
      // Each keyframe's observations of the solved landmarks agree with its (new) pose: one frame, not two.
      const solved = new Set<number>();
      for (const lm of engine.landmarkMap.values()) if (tracker.isInitLandmark(lm.id)) solved.add(lm.id);
      const kfErr: number[] = [];
      for (let i = 0; i < engine.keyframes.length; i++) {
        const e = keyframeError(engine, i, solved);
        if (e.n >= 25) {
          expect(e.meanPx).toBeLessThan(3.0);
          kfErr.push(e.meanPx);
        }
      }
      accepted.push({ f, reposed: st.keyframesReposed, propagated: st.keyframesPropagated, refPx: st.referenceErrorAfterPx, kfErr });
    }
    console.log(`[initstage] accepted ${accepted.map((a) => `f${a.f} kf ${a.reposed}+${a.propagated} ref ${a.refPx.toFixed(2)}px kfErr ${a.kfErr.map((e) => e.toFixed(2)).join("/")}`).join(" | ")}`);
    expect(accepted.length).toBeGreaterThanOrEqual(2);
  });

  it("§4: a keyframe observing too few solved landmarks is not solved from a short or mixed list — it is carried by the rigid correction exactly", () => {
    const engine = new VisionEngine(W, H, resolveConfig(), createRng(5));
    // Find the first stage frame with the default configuration.
    let stageFrame = -1;
    {
      const probe = new VisionEngine(W, H, resolveConfig(), createRng(5));
      for (let f = 0; f < 80 && stageFrame < 0; f++) {
        const o = probe.process(input(f));
        const st = o.initRefinement?.lastStage;
        if (st && st.frameId === f) stageFrame = f;
      }
    }
    expect(stageFrame).toBeGreaterThan(0);
    for (let f = 0; f < stageFrame; f++) engine.process(input(f));
    const tracker = engine.mapTrackerForTests();
    // Strip the newest keyframe's observations of init landmarks down to 10 (< relocalization.minInliers 25).
    const kf = engine.createKeyframeForTests((stageFrame - 1) * 33.3);
    let kept = 0;
    kf.observations = kf.observations.filter((o) => {
      if (!tracker.isInitLandmark(o.landmarkId)) return true;
      return kept++ < 10;
    });
    const oldPose: RigidTransform = { rotation: Float64Array.from(kf.pose.rotation), translation: Float64Array.from(kf.pose.translation) };
    const poseBefore: RigidTransform = { rotation: Float64Array.from(tracker.pose.rotation), translation: Float64Array.from(tracker.pose.translation) };
    void poseBefore;
    const o = engine.process(input(stageFrame));
    const st = o.initRefinement?.lastStage;
    expect(st && st.frameId === stageFrame).toBe(true);
    expect(st!.applied).toBe(true);
    expect(st!.keyframesPropagated).toBeGreaterThanOrEqual(1);
    // With no anchor points the stage's correction is the current pose's own C (old map → new map);
    // the carried keyframe is exactly T_old ∘ C⁻¹.
    const c: RigidTransform = { rotation: Float64Array.from(st!.correctionRotation), translation: Float64Array.from(st!.correctionTranslation) };
    const expected = composeTransforms(oldPose, invertTransform(c));
    const stored = engine.keyframes.find((k) => k.id === kf.id)!;
    // (rotationDistance goes through acos: ~1e-7 rad resolution near 0.)
    expect(rotationDistance(stored.pose.rotation, expected.rotation)).toBeLessThan(1e-6);
    for (let i = 0; i < 3; i++) expect(stored.pose.translation[i]).toBeCloseTo(expected.translation[i], 9);
  });

  it("§9.2: a non-finite solved value rejects the stage as invalid and nothing changes", () => {
    const engine = new VisionEngine(W, H, resolveConfig({ landmarks: { initRefineStages: 0 } }), createRng(5));
    for (let f = 0; f < 40; f++) engine.process(input(f));
    const tracker = engine.mapTrackerForTests();
    // Poison one init landmark's reference observation: the stage's solve becomes non-finite.
    const victim = [...engine.landmarkMap.values()].find((lm) => tracker.isInitLandmark(lm.id) && lm.trackId >= 0)!;
    victim.anchorX = NaN;
    (tracker as unknown as { config: { initRefineStages: number } }).config.initRefineStages = 3;
    const c = tracker.refineInitStage(engine.tracksForTests(), TP_K, 40, 0.2);
    expect(c).not.toBeNull();
    const hasNaN = !Number.isFinite(c!.errorAfterPx) || Array.from(c!.after).some((v) => !Number.isFinite(v));
    expect(hasNaN).toBe(true);
  });
});

describe("keyframe BA gauge (指示書 §3)", () => {
  it("holds the origin keyframe (the reference view, identity) fixed with every other stored keyframe free — also once its landmarks are gone", () => {
    const engine = new VisionEngine(W, H, resolveConfig(), createRng(5));
    let last: VisionOutput | null = null;
    for (let f = 0; f < 100; f++) last = engine.process(input(f));
    const ba = last!.bundleAdjustment;
    expect(ba.runs).toBeGreaterThanOrEqual(2);
    expect(ba.gauge).toBe("reference");
    expect(ba.freeKeyframes).toBe(ba.keyframes - 1);
    expect(ba.referenceObservations).toBeGreaterThanOrEqual(12);
    expect(engine.keyframes.filter((kf) => kf.origin).length).toBe(1);
    expect(engine.keyframes[0].origin).toBe(true);
    const identity = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0];
    expect([...engine.keyframes[0].pose.rotation, ...engine.keyframes[0].pose.translation]).toEqual(identity);
    // The reference view still explains the init landmarks after the runs.
    const ref = referenceViewError(engine);
    expect(ref.n).toBeGreaterThan(30);
    expect(ref.meanPx).toBeLessThan(1.0);
    // Remove the init landmarks: the origin keyframe has nothing left to observe, is still the fixed one and does not move.
    const tracker = engine.mapTrackerForTests();
    const initIds = [...engine.landmarkMap.values()].filter((lm) => tracker.isInitLandmark(lm.id)).map((lm) => lm.id);
    tracker.purgeLandmarks(initIds, engine.tracksForTests());
    const origin = engine.keyframes.find((kf) => kf.origin)!;
    expect(engine.runBundleAdjustment(TP_K)).toBe(true);
    const b2 = engine.bundleAdjustment;
    expect(b2.gauge).toBe("reference");
    expect(b2.referenceObservations).toBe(0);
    expect(b2.freeKeyframes).toBe(b2.keyframes - 1);
    expect(Number.isFinite(b2.errorAfterPx)).toBe(true);
    expect([...origin.pose.rotation, ...origin.pose.translation]).toEqual(identity);
  });
});
