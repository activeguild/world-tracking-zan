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
});
