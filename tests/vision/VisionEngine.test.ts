import { describe, expect, it } from "vitest";
import { resolveConfig } from "../../src/ar/ARConfig";
import { TrackingState } from "../../src/ar/ARState";
import { approximateIntrinsics } from "../../src/camera/CameraIntrinsics";
import { createRng } from "../../src/vision/OutlierRejection";
import { TRACK_STRIDE, type VisionInput, type VisionOutput } from "../../src/vision/types";
import { VisionEngine } from "../../src/vision/VisionEngine";
import { makeTexture, similarityH, warpImage } from "../helpers/synthetic";

const W = 640;
const H = 480;
const K = approximateIntrinsics(W, H);

function input(frameId: number, gray: Uint8Array): VisionInput {
  return { frameId, timestamp: frameId * 33.3, width: W, height: H, gray, intrinsics: K };
}

/** Big texture cropped by a sliding window: content stays valid while "the camera" pans. */
function makeSequence(frames: number, step: { dx: number; dy: number; rot?: number; scale?: number }) {
  const BW = W + 400;
  const BH = H + 300;
  const big = makeTexture(BW, BH, createRng(2024), [12, 30, 70, 160]);
  const out: Uint8Array[] = [];
  for (let f = 0; f < frames; f++) {
    const ox = 200 + f * step.dx;
    const oy = 150 + f * step.dy;
    const crop = new Uint8Array(W * H);
    for (let y = 0; y < H; y++) {
      const sy = Math.round(oy) + y;
      crop.set(big.subarray(sy * BW + Math.round(ox), sy * BW + Math.round(ox) + W), y * W);
    }
    if (step.rot || step.scale) {
      const Hm = similarityH(W / 2, H / 2, (step.rot ?? 0) * f, Math.pow(step.scale ?? 1, f), 0, 0);
      out.push(warpImage(crop, W, H, Hm, 128));
    } else {
      out.push(crop);
    }
  }
  return out;
}

function run(engine: VisionEngine, frames: Uint8Array[]): VisionOutput[] {
  return frames.map((g, i) => engine.process(input(i, g)));
}

describe("VisionEngine (Phase 1 pipeline)", () => {
  const config = resolveConfig();

  it("detects ~maxFeatures on the first frame and enters TRACKING", () => {
    const frames = makeSequence(3, { dx: 1, dy: 0 });
    const engine = new VisionEngine(W, H, config, createRng(1));
    const outs = run(engine, frames);

    expect(outs[0].quality.featureCount).toBeGreaterThanOrEqual(config.features.maxFeatures * 0.9);
    expect(outs[0].quality.trackedCount).toBe(0);
    expect(outs[0].state).toBe(TrackingState.SEARCHING_FEATURES);
    expect(outs[1].quality.trackedCount).toBeGreaterThan(100);
    expect(outs[1].state).toBe(TrackingState.TRACKING);
    expect(outs[0].tracks.length).toBe(outs[0].trackCount * TRACK_STRIDE);
  });

  it("Phase 1 criterion: ≥100 features tracked over a 30-frame pan", () => {
    const frames = makeSequence(30, { dx: 3, dy: 2 });
    const engine = new VisionEngine(W, H, config, createRng(1));
    const outs = run(engine, frames);

    for (let i = 1; i < outs.length; i++) {
      const q = outs[i].quality;
      expect(q.trackedCount, `frame ${i}`).toBeGreaterThanOrEqual(100);
      expect(q.inlierCount / q.trackedCount, `frame ${i} inlier ratio`).toBeGreaterThan(0.85);
      expect(outs[i].state).toBe(TrackingState.TRACKING);
      expect(q.lowFeature).toBe(false);
      expect(q.reprojectionError).toBeLessThan(1.0);
    }

    // Tracks really follow the motion: median displacement ≈ (3, 2).
    const last = outs[outs.length - 1];
    const dxs: number[] = [];
    const dys: number[] = [];
    for (let i = 0; i < last.trackCount; i++) {
      const o = i * TRACK_STRIDE;
      if (last.tracks[o + 5] < 1) continue; // new feature, no motion yet
      dxs.push(last.tracks[o + 1] - last.tracks[o + 3]);
      dys.push(last.tracks[o + 2] - last.tracks[o + 4]);
    }
    dxs.sort((a, b) => a - b);
    dys.sort((a, b) => a - b);
    expect(dxs[dxs.length >> 1]).toBeCloseTo(-3, 0);
    expect(dys[dys.length >> 1]).toBeCloseTo(-2, 0);

    // Long-lived tracks exist (ages accumulate across frames).
    let maxAge = 0;
    for (let i = 0; i < last.trackCount; i++) maxAge = Math.max(maxAge, last.tracks[i * TRACK_STRIDE + 5]);
    expect(maxAge).toBeGreaterThanOrEqual(20);
  });

  it("tracks through rotation + zoom", () => {
    const frames = makeSequence(20, { dx: 0, dy: 0, rot: 0.01, scale: 1.005 });
    const engine = new VisionEngine(W, H, config, createRng(3));
    const outs = run(engine, frames);
    for (let i = 1; i < outs.length; i++) {
      expect(outs[i].quality.trackedCount, `frame ${i}`).toBeGreaterThanOrEqual(100);
      expect(outs[i].state).toBe(TrackingState.TRACKING);
    }
  });

  it("static scene: all features tracked with near-zero jitter", () => {
    const frames = makeSequence(1, { dx: 0, dy: 0 });
    const engine = new VisionEngine(W, H, config, createRng(1));
    const g = frames[0];
    const outs = [engine.process(input(0, g)), engine.process(input(1, g)), engine.process(input(2, g))];
    const q = outs[2].quality;
    expect(q.trackedCount / outs[1].quality.featureCount).toBeGreaterThan(0.95);
    let maxJitter = 0;
    for (let i = 0; i < outs[2].trackCount; i++) {
      const o = i * TRACK_STRIDE;
      if (outs[2].tracks[o + 5] < 1) continue;
      maxJitter = Math.max(
        maxJitter,
        Math.hypot(outs[2].tracks[o + 1] - outs[2].tracks[o + 3], outs[2].tracks[o + 2] - outs[2].tracks[o + 4]),
      );
    }
    expect(maxJitter).toBeLessThan(0.1);
  });

  it("declares TRACKING_LOST on a blank frame and recovers afterwards", () => {
    const frames = makeSequence(4, { dx: 1, dy: 0 });
    const blank = new Uint8Array(W * H).fill(90);
    const engine = new VisionEngine(W, H, config, createRng(1));
    const seq = [frames[0], frames[1], frames[2], blank, blank, blank, blank, frames[3], frames[3]];
    const outs = run(engine, seq);

    expect(outs[2].state).toBe(TrackingState.TRACKING);
    expect(outs[3].quality.trackedCount).toBe(0);
    expect(outs[3].quality.lowFeature).toBe(true);
    const lostIdx = outs.findIndex((o) => o.state === TrackingState.TRACKING_LOST);
    expect(lostIdx).toBeGreaterThanOrEqual(3);
    expect(lostIdx).toBeLessThanOrEqual(3 + config.state.lostFrameTolerance);
    expect(outs[outs.length - 1].state).toBe(TrackingState.TRACKING);
    expect(outs[outs.length - 1].quality.trackedCount).toBeGreaterThan(100);
  });

  it("keeps the feature count bounded and ids unique", () => {
    const frames = makeSequence(15, { dx: 5, dy: -3 });
    const engine = new VisionEngine(W, H, config, createRng(2));
    const outs = run(engine, frames);
    for (const o of outs) {
      expect(o.trackCount).toBeLessThanOrEqual(config.features.maxFeatures);
      const ids = new Set<number>();
      for (let i = 0; i < o.trackCount; i++) ids.add(o.tracks[i * TRACK_STRIDE]);
      expect(ids.size).toBe(o.trackCount);
    }
  });

  it("processes a 640x480 frame within budget", () => {
    const frames = makeSequence(12, { dx: 2, dy: 1 });
    const engine = new VisionEngine(W, H, config, createRng(1));
    run(engine, frames.slice(0, 2)); // warm-up
    const times: number[] = [];
    for (let i = 2; i < frames.length; i++) {
      const out = engine.process(input(i, frames[i]));
      times.push(out.processingMs);
    }
    times.sort((a, b) => a - b);
    const medianMs = times[times.length >> 1];
    // Node on a CI box; browsers are comparable. Spec target is <33 ms, ideally <20 ms.
    expect(medianMs).toBeLessThan(120);
    // Surface the number in the test output for the Phase 1 report.
    console.log(`[perf] VisionEngine median ${medianMs.toFixed(1)} ms/frame`, engine.timing);
  });

  it("rejects frames of a different size", () => {
    const engine = new VisionEngine(W, H, config);
    expect(() =>
      engine.process({ frameId: 0, timestamp: 0, width: 320, height: 240, gray: new Uint8Array(320 * 240), intrinsics: K }),
    ).toThrow();
  });
});
