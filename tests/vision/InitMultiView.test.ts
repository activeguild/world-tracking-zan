import { describe, expect, it } from "vitest";
import { resolveConfig } from "../../src/ar/ARConfig";
import { TrackingState } from "../../src/ar/ARState";
import { cameraCenterOf, type RigidTransform } from "../../src/math/Pose";
import { createRng } from "../../src/vision/OutlierRejection";
import type { MapPoseOutput, VisionInput, VisionOutput } from "../../src/vision/types";
import { VisionEngine } from "../../src/vision/VisionEngine";
import { TP_H, TP_K, TP_W, tpFloorPoint, tpGravityAt, tpPose, tpRender, type TPPose } from "../helpers/twoPlaneScene";

/**
 * v25: multi-view refinement stages of the initial map (Android recordings
 * 39/40). The two-view initialization is degenerate in the translation
 * direction on a floor-dominant view with a 30 px baseline: on device the
 * refinement moved the direction 3.5° without changing the residual, and the
 * fresh map (0.5 px) jumped to ~2 px as soon as the camera had moved on. The
 * stages re-solve the init landmarks and the current pose against the fixed
 * reference view once the parallax from the reference has grown (60 / 120 /
 * 240 px). The synthetic scene has a wall, so its two-view init is already
 * well constrained; the degeneracy is emulated by distorting the init map
 * right after the initialization the way the device's would be (depth
 * graded laterally, i.e. a wrong translation direction), and the stages
 * must restore it — which the two-view refinement no longer can.
 */
const W = TP_W;
const H = TP_H;

function rigid(mp: MapPoseOutput): RigidTransform {
  return { rotation: Float64Array.from(mp.rotation), translation: Float64Array.from(mp.translation) };
}

const rendered = new Map<(f: number) => TPPose, Uint8Array[]>();
function frameOf(poseAt: (f: number) => TPPose, f: number): Uint8Array {
  let seq = rendered.get(poseAt);
  if (!seq) {
    seq = [];
    rendered.set(poseAt, seq);
  }
  while (seq.length <= f) seq.push(tpRender(poseAt(seq.length)));
  return seq[f];
}

interface Run {
  initFrame: number;
  outs: VisionOutput[];
  engine: VisionEngine;
  /** Depth ratio (estimated / true) of the init landmarks along their reference rays, floor only. */
  ratios: { ratio: number; x: number }[];
  ratioMean: number;
  ratioSd: number;
  /** Lateral gradient of the ratio: mean over the right third minus the left third. */
  ratioGradient: number;
  /** Direction error (deg) of the final camera centre from the reference against the truth. */
  dirErrDeg: number;
  pnp: { f: number; err: number; inliers: number }[];
  stages: number;
  /** Stage records snapshotted on the frame they ran (the engine's output object is shared across frames in-process). */
  stageRuns: { f: number; stage: NonNullable<NonNullable<VisionOutput["initRefinement"]>["lastStage"]> }[];
  worldAt: number;
}

function run(poseAt: (f: number) => TPPose, frames: number, config: Parameters<typeof resolveConfig>[0] = {}, distortAfterInit?: (engine: VisionEngine) => void): Run {
  const engine = new VisionEngine(W, H, resolveConfig(config), createRng(5));
  const outs: VisionOutput[] = [];
  let initFrame = -1;
  let worldAt = -1;
  const pnp: Run["pnp"] = [];
  const stageRuns: Run["stageRuns"] = [];
  for (let f = 0; f < frames; f++) {
    const input: VisionInput = { frameId: f, timestamp: f * 33.3, width: W, height: H, gray: frameOf(poseAt, f), intrinsics: TP_K, gravity: tpGravityAt(poseAt(f)) };
    const o = engine.process(input);
    outs.push(o);
    const st = o.initRefinement?.lastStage;
    if (st && st.frameId === f) stageRuns.push({ f, stage: { ...st } });
    if (initFrame < 0 && o.mapPose) {
      initFrame = f;
      distortAfterInit?.(engine);
    }
    if (worldAt < 0 && o.worldEstablished) worldAt = f;
    if (o.mapPose && o.mapPose.framesSinceTracked === 0) pnp.push({ f, err: o.mapPose.meanReprojectionErrorPx, inliers: o.mapPose.inlierCount });
  }
  expect(initFrame).toBeGreaterThanOrEqual(0);
  const last = outs[outs.length - 1].mapPose!;
  const refPose = poseAt(last.mapFrameId);
  const cEst = cameraCenterOf(rigid(last));
  const lastPose = poseAt(frames - 1);
  const cTru = [lastPose.C[0] - refPose.C[0], lastPose.C[1] - refPose.C[1], lastPose.C[2] - refPose.C[2]];
  const nE = Math.hypot(cEst[0], cEst[1], cEst[2]), nT = Math.hypot(cTru[0], cTru[1], cTru[2]);
  const scale = nE / nT;
  const dirErrDeg = (Math.acos(Math.max(-1, Math.min(1, (cEst[0] * cTru[0] + cEst[1] * cTru[1] + cEst[2] * cTru[2]) / (nE * nT)))) * 180) / Math.PI;
  const ratios: Run["ratios"] = [];
  for (const lm of engine.landmarkMap.values()) {
    if (!lm.anchorPose || lm.anchorY < 260) continue;
    const a = lm.anchorPose;
    if (Math.abs(a.rotation[0] - 1) > 1e-9 || Math.abs(a.translation[0]) > 1e-9 || Math.abs(a.translation[2]) > 1e-9) continue;
    const X = tpFloorPoint(refPose, lm.anchorX, lm.anchorY);
    if (!X) continue;
    const tru = Math.hypot(X[0], X[1], X[2]) * scale;
    ratios.push({ ratio: Math.hypot(lm.position[0], lm.position[1], lm.position[2]) / tru, x: lm.anchorX });
  }
  const mean = ratios.reduce((s, r) => s + r.ratio, 0) / Math.max(1, ratios.length);
  const sd = Math.sqrt(ratios.reduce((s, r) => s + (r.ratio - mean) ** 2, 0) / Math.max(1, ratios.length));
  const third = (lo: number, hi: number) => {
    const s = ratios.filter((r) => r.x >= lo && r.x < hi);
    return s.reduce((a, r) => a + r.ratio, 0) / Math.max(1, s.length);
  };
  return {
    initFrame,
    outs,
    engine,
    ratios,
    ratioMean: mean,
    ratioSd: sd,
    ratioGradient: third(W * (2 / 3), W) - third(0, W / 3),
    dirErrDeg,
    pnp,
    stages: outs[outs.length - 1].initRefinement?.stages ?? 0,
    stageRuns,
    worldAt,
  };
}

/** The device-like distortion: init landmark depths graded 3 % → −3 % across the image (a wrong translation direction). */
function distort(engine: VisionEngine): void {
  for (const lm of engine.landmarkMap.values()) {
    const s = 1 + 0.06 * (0.5 - lm.anchorX / W);
    lm.position[0] *= s;
    lm.position[1] *= s;
    lm.position[2] *= s;
  }
}

const slide = (f: number) => tpPose([0.005 * f, 0.001 * f, 0.002 * f], 0.0015 * f, 0);

function describeRun(name: string, r: Run): void {
  const stages = r.outs[r.outs.length - 1].initRefinement;
  const pnpAt = (f: number) => r.pnp.find((p) => p.f >= f);
  console.log(
    `[initmv] ${name}: init ${r.initFrame}, world ${r.worldAt}, stages ${r.stages}/${stages?.maxStages ?? 0}` +
      r.stageRuns.map(({ f, stage: s }) => `\n[initmv]   stage ${s.stage} @ frame ${f} par ${s.parallaxPx.toFixed(0)}px ${s.landmarks}lm ${s.errorBeforePx.toFixed(3)} → ${s.errorAfterPx.toFixed(3)}px it ${s.iterations} moved ${s.rotationDeg.toFixed(2)}° / dir ${s.directionDeg.toFixed(2)}° Δlm ${s.maxLandmarkShift.toFixed(3)} rebuilt ${s.rebuilt} removed ${s.removed} kf ${s.keyframesReposed} corr ${Math.hypot(s.correctionTranslation[0], s.correctionTranslation[1], s.correctionTranslation[2]).toFixed(4)} (fit n${s.anchorFitLandmarks}) ${s.applied ? "applied" : "NOT applied"}`).join("") +
      `\n[initmv]   final dir err ${r.dirErrDeg.toFixed(3)}°; init-landmark depth ratio n${r.ratios.length} ${r.ratioMean.toFixed(4)}±${r.ratioSd.toFixed(4)} gradient ${r.ratioGradient.toFixed(4)}; PnP ${[20, 40, 60, 80, 100, 119].map((f) => { const p = pnpAt(f); return p ? `${p.f}:${p.err.toFixed(2)}/${p.inliers}` : `${f}:-`; }).join(" ")}`,
  );
}

describe("multi-view refinement of the initial map (v25)", () => {
  it("restores a direction-distorted init map once the parallax from the reference has grown; the two-view refinement alone cannot", () => {
    const stagesRun = run(slide, 120, {}, distort);
    const noStages = run(slide, 120, { landmarks: { initRefineStages: 0 } }, distort);
    const clean = run(slide, 120, {});
    describeRun("distorted, stages", stagesRun);
    describeRun("distorted, no stages", noStages);
    describeRun("clean, stages", clean);
    // The distortion was injected: without stages the lateral gradient persists.
    expect(noStages.stages).toBe(0);
    expect(Math.abs(noStages.ratioGradient)).toBeGreaterThan(0.015);
    // With stages the init landmarks are back on their true depths (flat ratio).
    expect(stagesRun.stages).toBeGreaterThanOrEqual(1);
    expect(stagesRun.outs[stagesRun.outs.length - 1].initRefinement?.lastStage?.applied).toBe(true);
    expect(Math.abs(stagesRun.ratioGradient)).toBeLessThan(0.006);
    expect(stagesRun.ratioSd).toBeLessThan(0.01);
    expect(stagesRun.dirErrDeg).toBeLessThan(0.5);
    // The clean run is not harmed by the stages.
    expect(clean.ratioSd).toBeLessThan(0.01);
    expect(clean.dirErrDeg).toBeLessThan(0.5);
    const lateErr = (r: Run) => r.pnp.filter((p) => p.f > 80).map((p) => p.err);
    const med = (a: number[]) => [...a].sort((x, y) => x - y)[a.length >> 1];
    expect(med(lateErr(stagesRun))).toBeLessThan(med(lateErr(noStages)));
    expect(med(lateErr(clean))).toBeLessThan(0.5);
    // Map id unchanged, no loss.
    for (const r of [stagesRun, clean]) {
      const id = r.outs[r.initFrame].mapPose!.mapFrameId;
      for (let f = r.initFrame; f < r.outs.length; f++) {
        expect(r.outs[f].mapPose?.mapFrameId).toBe(id);
        expect(r.outs[f].state).not.toBe(TrackingState.RELOCALIZING);
      }
    }
  });

  it("stages run in order at their parallax thresholds, before and after the world is established, with enough init landmarks", () => {
    const r = run(slide, 120, {});
    const stageFrames = r.stageRuns.map((s) => s.f);
    console.log(`[initmv] stage frames ${stageFrames.join(" ")}, world at ${r.worldAt}`);
    expect(r.worldAt).toBeGreaterThan(0);
    expect(stageFrames.length).toBeGreaterThanOrEqual(2);
    expect(stageFrames.some((f) => f > r.worldAt)).toBe(true);
    // Stage order and thresholds: each stage ran at or above its parallax threshold.
    let prev = 0;
    for (const { f, stage: st } of r.stageRuns) {
      expect(st.frameId).toBe(f);
      expect(st.stage).toBe(prev + 1);
      expect(st.parallaxPx).toBeGreaterThanOrEqual(30 * Math.pow(2, st.stage) - 1e-9);
      expect(st.landmarks).toBeGreaterThanOrEqual(30);
      prev = st.stage;
    }
    // The stage records survive on the output after the world exists.
    const last = r.outs[r.outs.length - 1].initRefinement!;
    expect(last.stages).toBe(prev);
    expect(last.maxStages).toBe(3);
  });
});
