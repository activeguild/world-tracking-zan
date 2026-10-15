import { describe, expect, it } from "vitest";
import { resolveConfig } from "../../src/ar/ARConfig";
import { cameraCenterOf, rotationDistance, type RigidTransform } from "../../src/math/Pose";
import { createRng } from "../../src/vision/OutlierRejection";
import type { MapPoseOutput, VisionInput, VisionOutput } from "../../src/vision/types";
import { VisionEngine } from "../../src/vision/VisionEngine";
import { TP_H, TP_K, TP_W, tpFloorPoint, tpGravityAt, tpPose, tpRender, type TPPose } from "../helpers/twoPlaneScene";

/**
 * 修正指示書 §6: the scale-drift harness showed the map scale, as seen by
 * PnP, growing 1.4 → 3.9 % along the initial slide even with the map frozen
 * at the world (no new landmarks). A frozen map that PnP reads at a
 * different scale from different viewpoints is not uniformly scaled: it is
 * distorted. The map frame is camera 0 (the two-view reference, identity
 * pose) with |t| = 1 to the init frame, so every init landmark has a known
 * true position along its own reference ray. This measures the distortion.
 */
const W = TP_W;
const H = TP_H;

function rigid(mp: MapPoseOutput): RigidTransform {
  return { rotation: Float64Array.from(mp.rotation), translation: Float64Array.from(mp.translation) };
}

export function initMap(poseAt: (f: number) => TPPose, config: Parameters<typeof resolveConfig>[0] = {}, through = 20) {
  const engine = new VisionEngine(W, H, resolveConfig(config), createRng(5));
  let init: { f: number; out: VisionOutput } | null = null;
  const outs: VisionOutput[] = [];
  for (let f = 0; f <= through; f++) {
    const input: VisionInput = { frameId: f, timestamp: f * 33.3, width: W, height: H, gray: tpRender(poseAt(f)), intrinsics: TP_K, gravity: tpGravityAt(poseAt(f)) };
    const o = engine.process(input);
    outs.push(o);
    if (!init && o.mapPose) init = { f, out: o };
  }
  expect(init).not.toBeNull();
  const refFrame = init!.out.mapPose!.mapFrameId;
  expect(refFrame).toBe(0);
  const refPose = poseAt(refFrame), initPose = poseAt(init!.f);
  const baseline = Math.hypot(initPose.C[0] - refPose.C[0], initPose.C[1] - refPose.C[1], initPose.C[2] - refPose.C[2]);
  // Two-view pose at init vs truth: rotation and translation direction.
  const est = rigid(init!.out.mapPose!);
  const cEst = cameraCenterOf(est);
  // Map units per camera-0 unit: the estimated baseline over the true one
  // (|t| = 1 from the essential matrix; the two-view refinement keeps the
  // median depth instead, so |t| may differ slightly from 1).
  const scale = Math.hypot(cEst[0], cEst[1], cEst[2]) / baseline;
  const cTru = [initPose.C[0] - refPose.C[0], initPose.C[1] - refPose.C[1], initPose.C[2] - refPose.C[2]];
  const nE = Math.hypot(cEst[0], cEst[1], cEst[2]), nT = Math.hypot(cTru[0], cTru[1], cTru[2]);
  const dirErrDeg = (Math.acos(Math.max(-1, Math.min(1, (cEst[0] * cTru[0] + cEst[1] * cTru[1] + cEst[2] * cTru[2]) / (nE * nT)))) * 180) / Math.PI;
  const rotErrDeg = (rotationDistance(est.rotation, Float64Array.from(initPose.R)) * 180) / Math.PI;
  // Landmark depth ratio along the reference ray, floor only.
  const samples: { ratio: number; depth: number; x: number; y: number }[] = [];
  for (const lm of engine.landmarkMap.values()) {
    if (!lm.anchorPose || lm.anchorY < 260) continue;
    // Init landmarks anchor at the identity pose (camera 0).
    const a = lm.anchorPose;
    if (Math.abs(a.rotation[0] - 1) > 1e-9 || Math.abs(a.translation[0]) > 1e-9 || Math.abs(a.translation[2]) > 1e-9) continue;
    const X = tpFloorPoint(refPose, lm.anchorX, lm.anchorY);
    if (!X) continue;
    const tru = Math.hypot(X[0], X[1], X[2]) * scale;
    const e = Math.hypot(lm.position[0], lm.position[1], lm.position[2]);
    samples.push({ ratio: e / tru, depth: tru, x: lm.anchorX, y: lm.anchorY });
  }
  const stats = (s: { ratio: number }[]) => {
    if (!s.length) return "n0";
    const m = s.reduce((a, c) => a + c.ratio, 0) / s.length;
    const sd = Math.sqrt(s.reduce((a, c) => a + (c.ratio - m) ** 2, 0) / s.length);
    return `n${s.length} ${m.toFixed(4)}±${sd.toFixed(4)}`;
  };
  const byDepth = [0, 14, 20, 28, 40, 1e9];
  const depthBins = byDepth.slice(0, -1).map((lo, i) => `${lo}–${byDepth[i + 1] === 1e9 ? "∞" : byDepth[i + 1]}u ${stats(samples.filter((s) => s.depth >= lo && s.depth < byDepth[i + 1]))}`);
  const byX = [0, 160, 320, 480, 640];
  const xBins = byX.slice(0, -1).map((lo, i) => `x${lo}–${byX[i + 1]} ${stats(samples.filter((s) => s.x >= lo && s.x < byX[i + 1]))}`);
  const byY = [260, 330, 400, 480];
  const yBins = byY.slice(0, -1).map((lo, i) => `y${lo}–${byY[i + 1]} ${stats(samples.filter((s) => s.y >= lo && s.y < byY[i + 1]))}`);
  return {
    initFrame: init!.f,
    refinement: engine.mapTrackerForTests().lastInitRefinement,
    model: init!.out.pose?.model ?? "?",
    baselinePx: init!.out.pose?.parallaxPx ?? NaN,
    dirErrDeg,
    rotErrDeg,
    all: stats(samples),
    depthBins,
    xBins,
    yBins,
    landmarks: engine.landmarkMap.size,
  };
}

describe("distortion of the two-view initial map (指示書 §6)", () => {
  it("diagnostic: init landmark depth ratio (estimated / true) along the reference rays, by depth and image position", () => {
    const slide = (f: number) => tpPose([0.005 * f, 0.001 * f, 0.002 * f], 0.0015 * f, 0);
    const lateral = (f: number) => tpPose([0.005 * f, 0, 0], 0, 0);
    const results: Record<string, ReturnType<typeof initMap>> = {};
    for (const [name, traj, cfg] of [
      ["slide + yaw, no refinement", slide, { landmarks: { initRefineWithBA: false } }],
      ["slide + yaw, refined", slide, {}],
      ["pure lateral, no refinement", lateral, { landmarks: { initRefineWithBA: false } }],
      ["pure lateral, refined", lateral, {}],
    ] as const) {
      const r = initMap(traj, cfg);
      results[name] = r;
      const ref = r.refinement;
      console.log(
        `[initmap] ${name}: init frame ${r.initFrame} model ${r.model} parallax ${r.baselinePx.toFixed(1)} px; t-direction error ${r.dirErrDeg.toFixed(3)}°, rotation error ${r.rotErrDeg.toFixed(3)}°; lm ${r.landmarks}` +
          (ref ? `; refinement ${ref.errorBeforePx.toFixed(3)} → ${ref.errorAfterPx.toFixed(3)} px in ${ref.iterations} it, moved ${ref.rotationDeg.toFixed(3)}° / dir ${ref.directionDeg.toFixed(3)}°, ${ref.applied ? "applied" : "NOT applied"}` : "") +
          `\n[initmap]   depth ratio all ${r.all}\n[initmap]   by depth: ${r.depthBins.join(" | ")}\n[initmap]   by x: ${r.xBins.join(" | ")}\n[initmap]   by y: ${r.yBins.join(" | ")}`,
      );
    }
    // The unrefined essential-matrix init is 2.3° off on the slide (a
    // laterally graded 1–6 % depth distortion); the refinement brings the
    // direction to the lateral case's noise floor and the ratios flat.
    const raw = results["slide + yaw, no refinement"], refined = results["slide + yaw, refined"];
    expect(raw.dirErrDeg).toBeGreaterThan(1.5);
    expect(refined.dirErrDeg).toBeLessThan(0.5);
    expect(refined.rotErrDeg).toBeLessThan(0.1);
    expect(refined.refinement?.applied).toBe(true);
    expect(results["pure lateral, refined"].dirErrDeg).toBeLessThan(0.3);
  });
});
