import { describe, expect, it } from "vitest";
import { resolveConfig } from "../../src/ar/ARConfig";
import { TrackingState } from "../../src/ar/ARState";
import { approximateIntrinsics } from "../../src/camera/CameraIntrinsics";
import { type Mat3, mat3Invert, mat3Multiply } from "../../src/math/Matrix";
import { createRng } from "../../src/vision/OutlierRejection";
import type { VisionInput, VisionOutput } from "../../src/vision/types";
import { VisionEngine } from "../../src/vision/VisionEngine";
import { makeTexture, warpImage } from "../helpers/synthetic";

/**
 * v16 A/B: `landmarks.freezeAfterWorld` stops triangulating new landmarks
 * once the world is established, so the map stays the one the world was
 * anchored to. Tracking (PnP, classification, pruning) is unchanged.
 */
const W = 640;
const H = 480;
const K = approximateIntrinsics(W, H, 66);
const Kmat = new Float64Array([K.fx, 0, K.cx, 0, K.fy, K.cy, 0, 0, 1]);
const KInv = mat3Invert(Kmat)!;
const I: Mat3 = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
const base = makeTexture(W, H, createRng(777), [12, 30, 70, 160]);
const n = [0, Math.SQRT1_2, Math.SQRT1_2];
const gravity = [0, Math.SQRT1_2, Math.SQRT1_2];
const FRAMES = 90;

function pixelHomography(r: Mat3, t: ArrayLike<number>, nn: ArrayLike<number>, d: number): Mat3 {
  const h = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) h[i * 3 + j] = r[i * 3 + j] + (t[i] * nn[j]) / d;
  return mat3Multiply(mat3Multiply(Kmat, h), KInv);
}

/** A slow slide over the floor; new texture keeps entering the view, so new tracks appear. */
const sequence: Uint8Array[] = [];
for (let f = 0; f < FRAMES; f++) {
  const t = [-0.004 * f, -0.001 * f, 0];
  sequence.push(f === 0 ? base : warpImage(base, W, H, pixelHomography(I, t, n, 1), 128));
}

function run(freeze: boolean): { outs: VisionOutput[]; foundIdx: number; counts: number[] } {
  const engine = new VisionEngine(W, H, resolveConfig({ landmarks: { freezeAfterWorld: freeze } }), createRng(5));
  const outs: VisionOutput[] = [];
  const counts: number[] = [];
  for (let f = 0; f < FRAMES; f++) {
    const input: VisionInput = { frameId: f, timestamp: f * 33.3, width: W, height: H, gray: sequence[f], intrinsics: K, gravity };
    outs.push(engine.process(input));
    counts.push(engine.landmarkCount);
  }
  const foundIdx = outs.findIndex((o) => o.state === TrackingState.PLANE_FOUND);
  expect(foundIdx, `freeze ${freeze}: plane found`).toBeGreaterThan(0);
  return { outs, foundIdx, counts };
}

describe("freezeAfterWorld (v16 A/B)", () => {
  it("adds no landmarks once the world is established, while the default map keeps growing on the same frames", () => {
    const live = run(false);
    const frozen = run(true);
    // Identical up to the frame the world is established (the flag is inert before it).
    expect(frozen.foundIdx).toBe(live.foundIdx);
    for (let f = 0; f <= frozen.foundIdx; f++) expect(frozen.counts[f]).toBe(live.counts[f]);
    // Afterwards the frozen map never gains a landmark (pruning may still remove some)...
    for (let f = frozen.foundIdx + 1; f < FRAMES; f++) {
      expect(frozen.counts[f], `frame ${f}`).toBeLessThanOrEqual(frozen.counts[f - 1]);
      expect(frozen.outs[f].mapPose?.triangulation.added ?? 0).toBe(0);
    }
    // ...while the live map grows on the same sequence.
    expect(live.counts[FRAMES - 1]).toBeGreaterThan(live.counts[live.foundIdx]);
    // Tracking itself is unaffected while the first view stays in frame.
    const frozenTracked = frozen.outs.slice(frozen.foundIdx).filter((o) => o.mapPose?.framesSinceTracked === 0).length;
    expect(frozenTracked).toBe(FRAMES - frozen.foundIdx);
    console.log(
      `[freeze] world at frame ${live.foundIdx}: live ${live.counts[live.foundIdx]} → ${live.counts[FRAMES - 1]} landmarks, frozen ${frozen.counts[frozen.foundIdx]} → ${frozen.counts[FRAMES - 1]}`,
    );
  });
});
