import { describe, expect, it } from "vitest";
import { resolveConfig } from "../../src/ar/ARConfig";
import { cameraCenterOf, type RigidTransform } from "../../src/math/Pose";
import { createRng } from "../../src/vision/OutlierRejection";
import type { MapPoseOutput, VisionInput } from "../../src/vision/types";
import { VisionEngine } from "../../src/vision/VisionEngine";
import { TP_H, TP_K, TP_W, tpGravityAt, tpPose, tpRender, type TPPose } from "../helpers/twoPlaneScene";

/**
 * 修正指示書 §6: the map scale along a trajectory. The map frame is the
 * two-view reference camera with |t| = 1 to the initialization frame, so
 * the ratio (estimated camera-centre move since init) / (true move × init
 * scale) is 1 for a map whose scale is conserved. The anchor-invariance
 * scenario measured 1.014 → 1.045 over the initial slide: the map grows
 * ~3 % larger than its own first generation while new landmarks are
 * triangulated. This harness isolates what the growth depends on.
 */
const W = TP_W;
const H = TP_H;

function rigid(mp: MapPoseOutput): RigidTransform {
  return { rotation: Float64Array.from(mp.rotation), translation: Float64Array.from(mp.translation) };
}

const rendered = new Map<(f: number) => TPPose, Uint8Array[]>();
function framesOf(poseAt: (f: number) => TPPose, frames: number): Uint8Array[] {
  let seq = rendered.get(poseAt);
  if (!seq) {
    seq = [];
    rendered.set(poseAt, seq);
  }
  while (seq.length < frames) seq.push(tpRender(poseAt(seq.length)));
  return seq;
}

export function scaleSeries(poseAt: (f: number) => TPPose, frames: number, config: Parameters<typeof resolveConfig>[0] = {}, sample = [20, 30, 40, 50, 60, 70, 80, 89]) {
  const engine = new VisionEngine(W, H, resolveConfig(config), createRng(5));
  const outs: (MapPoseOutput | null)[] = [];
  const seq = framesOf(poseAt, frames);
  let added = 0;
  for (let f = 0; f < frames; f++) {
    const input: VisionInput = { frameId: f, timestamp: f * 33.3, width: W, height: H, gray: seq[f], intrinsics: TP_K, gravity: tpGravityAt(poseAt(f)) };
    const o = engine.process(input);
    outs.push(o.mapPose);
    added += o.mapPose?.triangulation.added ?? 0;
  }
  const initFrame = outs.findIndex((m) => m !== null);
  const refFrame = outs[initFrame]!.mapFrameId;
  const refPose = poseAt(refFrame), initPose = poseAt(initFrame);
  const baseline = Math.hypot(initPose.C[0] - refPose.C[0], initPose.C[1] - refPose.C[1], initPose.C[2] - refPose.C[2]);
  const cInit = cameraCenterOf(rigid(outs[initFrame]!));
  // Map units per camera-0 unit at init: estimated baseline / true baseline.
  const initScale = Math.hypot(cInit[0], cInit[1], cInit[2]) / baseline;
  const at = (f: number): number => {
    const m = outs[f];
    if (!m || m.framesSinceTracked !== 0) return NaN;
    const c = cameraCenterOf(rigid(m));
    const tc = poseAt(f).C;
    const est = Math.hypot(c[0] - cInit[0], c[1] - cInit[1], c[2] - cInit[2]);
    const tru = Math.hypot(tc[0] - initPose.C[0], tc[1] - initPose.C[1], tc[2] - initPose.C[2]) * initScale;
    return est / tru;
  };
  const series = sample.filter((f) => f > initFrame + 3 && f < frames).map((f) => ({ f, s: at(f) }));
  return { initFrame, refFrame, series, landmarks: engine.landmarkMap.size, added, last: series[series.length - 1].s, first: series[0].s };
}

const fmt = (r: ReturnType<typeof scaleSeries>) => `init ${r.initFrame} (ref ${r.refFrame}) lm ${r.landmarks} added ${r.added}: ${r.series.map((x) => `${x.f}:${x.s.toFixed(4)}`).join(" ")}`;

const slide = (f: number) => tpPose([0.005 * f, 0.001 * f, 0.002 * f], 0.0015 * f, 0);
const lateral = (f: number) => tpPose([0.005 * f, 0, 0], 0, 0);

describe("map scale along a trajectory (指示書 §6)", () => {
  it("the two-view init refinement keeps the map scale conserved along the initial slide (A/B against the raw essential-matrix init)", { timeout: 120000 }, () => {
    const base = scaleSeries(slide, 90);
    const raw = scaleSeries(slide, 90, { landmarks: { initRefineWithBA: false } });
    console.log(`[scale] slide + yaw (default):          ${fmt(base)}`);
    console.log(`[scale] slide + yaw, no init refine:    ${fmt(raw)}`);
    console.log(`[scale] slide + yaw, no BA:             ${fmt(scaleSeries(slide, 90, { bundleAdjustment: { enabled: false } }))}`);
    console.log(`[scale] slide, no init refine, no BA:   ${fmt(scaleSeries(slide, 90, { landmarks: { initRefineWithBA: false }, bundleAdjustment: { enabled: false } }))}`);
    console.log(`[scale] slide, no init refine, frozen:  ${fmt(scaleSeries(slide, 90, { landmarks: { initRefineWithBA: false, freezeAfterWorld: true } }))}`);
    console.log(`[scale] pure lateral (default):         ${fmt(scaleSeries(lateral, 90))}`);
    console.log(`[scale] pure lateral, no init refine:   ${fmt(scaleSeries(lateral, 90, { landmarks: { initRefineWithBA: false } }))}`);
    expect(base.series.length).toBeGreaterThan(3);
    // Without the refinement the scale read by PnP grows ~4 % along the slide
    // (1.014 at frame 20 → 1.044 at 89); with it the map is uniformly scaled
    // and the drift stays within 1 %.
    expect(Math.abs(raw.last - 1)).toBeGreaterThan(0.03);
    for (const x of base.series) expect(Math.abs(x.s - 1)).toBeLessThan(0.01);
  });
});
