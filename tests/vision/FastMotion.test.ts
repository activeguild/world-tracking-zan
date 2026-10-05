import { describe, expect, it } from "vitest";
import { resolveConfig, type PartialARConfig } from "../../src/ar/ARConfig";
import { approximateIntrinsics } from "../../src/camera/CameraIntrinsics";
import { createRng } from "../../src/vision/OutlierRejection";
import type { VisionInput } from "../../src/vision/types";
import { VisionEngine } from "../../src/vision/VisionEngine";
import { makeTexture } from "../helpers/synthetic";

/**
 * Fast camera motion: on device the cube held still for slow motion but
 * tracking broke as soon as the phone moved a bit faster. Per-frame
 * displacements of 30–45 px (at 640 px wide, 30 fps ≈ a brisk hand motion)
 * must stay trackable: 4 pyramid levels + constant-velocity LK seeding.
 */
const W = 640;
const H = 480;
const K = approximateIntrinsics(W, H);

function sequence(frames: number, dxPerFrame: number, dyPerFrame: number, accel = 0): Uint8Array[] {
  const BW = W + 800;
  const BH = H + 400;
  const big = makeTexture(BW, BH, createRng(4242), [12, 30, 70, 160]);
  const out: Uint8Array[] = [];
  let ox = 40;
  let oy = 100;
  for (let f = 0; f < frames; f++) {
    const crop = new Uint8Array(W * H);
    const x0 = Math.round(ox);
    const y0 = Math.round(oy);
    for (let y = 0; y < H; y++) crop.set(big.subarray((y0 + y) * BW + x0, (y0 + y) * BW + x0 + W), y * W);
    out.push(crop);
    ox += dxPerFrame + accel * f;
    oy += dyPerFrame;
  }
  return out;
}

function trackedRatios(config: PartialARConfig, frames: Uint8Array[]): number[] {
  const engine = new VisionEngine(W, H, resolveConfig(config), createRng(5));
  const ratios: number[] = [];
  let prevCount = 0;
  for (let f = 0; f < frames.length; f++) {
    const input: VisionInput = { frameId: f, timestamp: f * 33.3, width: W, height: H, gray: frames[f], intrinsics: K };
    const o = engine.process(input);
    if (f > 0) ratios.push(prevCount ? o.quality.trackedCount / prevCount : 0);
    prevCount = o.quality.featureCount;
  }
  return ratios;
}

describe("fast motion tracking", () => {
  it("keeps most features through a 32 px/frame pan (default config) and beats the 3-level / no-prediction tracker", () => {
    const frames = sequence(8, 32, 6);
    const now = trackedRatios({}, frames);
    const legacy = trackedRatios({ tracker: { pyramidLevels: 3, predictMotion: false } }, frames);
    console.log(`[fast] 32 px/frame: default ${now.map((r) => r.toFixed(2)).join(" ")} | legacy ${legacy.map((r) => r.toFixed(2)).join(" ")}`);
    for (let i = 1; i < now.length; i++) expect(now[i], `frame ${i + 1}`).toBeGreaterThan(0.7);
    const meanNow = now.reduce((a, b) => a + b, 0) / now.length;
    const meanLegacy = legacy.reduce((a, b) => a + b, 0) / legacy.length;
    expect(meanNow).toBeGreaterThan(meanLegacy);
  });

  it("follows an accelerating pan up to ~45 px/frame thanks to the velocity prediction", () => {
    const frames = sequence(9, 20, 4, 3.5); // 20, 23.5, 27, … ≈ 48 px at the end
    const now = trackedRatios({}, frames);
    const noPredict = trackedRatios({ tracker: { predictMotion: false } }, frames);
    console.log(`[fast] accelerating: predict ${now.map((r) => r.toFixed(2)).join(" ")} | no-predict ${noPredict.map((r) => r.toFixed(2)).join(" ")}`);
    for (let i = 1; i < now.length; i++) expect(now[i], `frame ${i + 1}`).toBeGreaterThan(0.6);
  });

  // ---- v7: motion level, homography prediction, adaptive search gate ----
  function run(config: PartialARConfig, frames: Uint8Array[]) {
    const engine = new VisionEngine(W, H, resolveConfig(config), createRng(5));
    const outs = [];
    for (let f = 0; f < frames.length; f++) {
      outs.push(engine.process({ frameId: f, timestamp: f * 33.3, width: W, height: H, gray: frames[f], intrinsics: K }));
    }
    return outs;
  }

  it("reports the motion level from the previous frame's median displacement: normal / medium / fast (v7 §3)", () => {
    const cfg = resolveConfig().tracker;
    const slow = run({}, sequence(5, 3, 1));
    const medium = run({}, sequence(5, 12, 2));
    const fast = run({}, sequence(5, 30, 4));
    // Frame 1 measures the displacement; frame 2 is classified from it.
    expect(slow[2].motion.level).toBe("normal");
    expect(slow[2].motion.medianDisplacementPx).toBeLessThan(cfg.mediumMotionPx);
    expect(medium[2].motion.level).toBe("medium");
    expect(medium[2].motion.medianDisplacementPx).toBeGreaterThanOrEqual(cfg.mediumMotionPx);
    expect(medium[2].motion.searchScale).toBe(cfg.mediumMotionSearchScale);
    expect(fast[2].motion.level).toBe("fast");
    expect(fast[2].motion.medianDisplacementPx).toBeGreaterThanOrEqual(cfg.fastMotionPx);
    expect(fast[2].motion.searchScale).toBe(cfg.fastMotionSearchScale);
    // The first frame has nothing to measure and starts at normal.
    expect(slow[0].motion.level).toBe("normal");
    expect(slow[0].motion.trackedAfter).toBe(0);
    console.log(`[motion] slow ${slow[2].motion.medianDisplacementPx.toFixed(1)}px  medium ${medium[2].motion.medianDisplacementPx.toFixed(1)}px  fast ${fast[2].motion.medianDisplacementPx.toFixed(1)}px`);
  });

  it("seeds LK with the previous homography when it was well supported, and keeps the LK bookkeeping consistent (v7 §6)", () => {
    const outs = run({}, sequence(6, 24, 5));
    // From the 3rd frame on the previous frame had a RANSAC homography with
    // plenty of inliers on this planar texture → homography prediction.
    for (let f = 2; f < outs.length; f++) {
      expect(outs[f].motion.predictionMode, `frame ${f}`).toBe("homography");
      expect(outs[f].motion.trackedAfter).toBeLessThanOrEqual(outs[f].motion.trackedBefore);
      expect(outs[f].motion.trackedAfter).toBe(outs[f].quality.trackedCount);
      expect(outs[f].motion.trackedBefore).toBe(outs[f - 1].quality.featureCount);
      expect(outs[f].motion.maxDisplacementPx).toBeGreaterThanOrEqual(outs[f].motion.medianDisplacementPx);
    }
    // Frame 1 has no previous homography: constant velocity (zero for fresh tracks).
    expect(outs[1].motion.predictionMode).toBe("velocity");
    const withH = trackedRatios({}, sequence(8, 32, 6));
    const noH = trackedRatios({ tracker: { homographyPrediction: false } }, sequence(8, 32, 6));
    const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length;
    console.log(`[motion] 32 px/frame: homography ${withH.map((r) => r.toFixed(2)).join(" ")} | velocity ${noH.map((r) => r.toFixed(2)).join(" ")}`);
    // The global model is at least as good as per-track velocity on a pure pan.
    expect(mean(withH)).toBeGreaterThanOrEqual(mean(noH) - 0.02);
  });

  it("falls back to per-track velocity when the previous homography was poorly supported (v7 §7)", () => {
    // Demand an impossible support so the homography is never trusted.
    const outs = run({ tracker: { predictionMinInliers: 100000 } }, sequence(6, 24, 5));
    for (let f = 1; f < outs.length; f++) expect(outs[f].motion.predictionMode, `frame ${f}`).toBe("velocity");
    const off = run({ tracker: { predictMotion: false } }, sequence(4, 10, 2));
    for (let f = 1; f < off.length; f++) expect(off[f].motion.predictionMode).toBe("none");
    expect(off[2].motion.searchScale).toBe(resolveConfig().tracker.mediumMotionSearchScale);
  });

  it("slow motion is unchanged by the v7 gate: level normal, scale 1, every feature kept", () => {
    const outs = run({}, sequence(8, 3, 1));
    for (let f = 2; f < outs.length; f++) {
      expect(outs[f].motion.level, `frame ${f}`).toBe("normal");
      expect(outs[f].motion.searchScale).toBe(1);
      expect(outs[f].quality.trackedCount / outs[f - 1].quality.featureCount).toBeGreaterThan(0.95);
    }
  });
});
