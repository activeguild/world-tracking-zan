import { describe, expect, it } from "vitest";
import { resolveConfig } from "../../src/ar/ARConfig";
import { TrackingState } from "../../src/ar/ARState";
import { approximateIntrinsics } from "../../src/camera/CameraIntrinsics";
import { type Mat3, mat3Invert, mat3Multiply, mat3TransformPoint } from "../../src/math/Matrix";
import { createRng } from "../../src/vision/OutlierRejection";
import type { ReprojectionBreakdown, VisionInput, VisionOutput } from "../../src/vision/types";
import { VisionEngine } from "../../src/vision/VisionEngine";
import { makeTexture, sampleBilinear } from "../helpers/synthetic";

/**
 * v16 diagnostics: the PnP inlier error split by image region (center disc
 * vs edge ring) and landmark kind. On device (Android) placed objects
 * shifted while the phone was tilted and came back when tilted back, with a
 * mean error of 2.4–3.0 px: a camera-model mismatch. The breakdown must
 * separate a pinhole-consistent scene (center ≈ edge) from a lens with
 * radial distortion (edge ≫ center).
 */
const W = 640;
const H = 480;
const K = approximateIntrinsics(W, H, 66);
const Kmat = new Float64Array([K.fx, 0, K.cx, 0, K.fy, K.cy, 0, 0, 1]);
const KInv = mat3Invert(Kmat)!;
const I: Mat3 = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
const base = makeTexture(W, H, createRng(4141), [12, 30, 70, 160]);
const n = [0, Math.SQRT1_2, Math.SQRT1_2];
const gravity = [0, Math.SQRT1_2, Math.SQRT1_2];
const FRAMES = 60;
const HALF_DIAG = Math.hypot(W, H) / 2;

function pixelHomography(r: Mat3, t: ArrayLike<number>, nn: ArrayLike<number>, d: number): Mat3 {
  const h = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) h[i * 3 + j] = r[i * 3 + j] + (t[i] * nn[j]) / d;
  return mat3Multiply(mat3Multiply(Kmat, h), KInv);
}

/**
 * Render frame f of a slow slide over the floor, then apply a radial lens
 * distortion to the *observed* image: an observed pixel p maps to the ideal
 * pinhole pixel c + (p − c)·(1 + k1·r²), r = |p − c| / half-diagonal. k1 = 0
 * is the pinhole the engine assumes.
 */
function render(f: number, k1: number): Uint8Array {
  const t = [-0.004 * f, -0.001 * f, 0];
  const Hf = f === 0 ? I : pixelHomography(I, t, n, 1);
  const inv = mat3Invert(Hf)!;
  const out = new Uint8Array(W * H);
  const pt = new Float64Array(2);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const dx = x - K.cx;
      const dy = y - K.cy;
      const r2 = (dx * dx + dy * dy) / (HALF_DIAG * HALF_DIAG);
      const s = 1 + k1 * r2;
      const ux = K.cx + dx * s;
      const uy = K.cy + dy * s;
      if (!mat3TransformPoint(inv, ux, uy, pt)) {
        out[y * W + x] = 128;
        continue;
      }
      out[y * W + x] = Math.round(sampleBilinear(base, W, H, pt[0], pt[1], 128));
    }
  }
  return out;
}

function run(k1: number): { breakdown: ReprojectionBreakdown; meanErrPx: number; frames: number } {
  const engine = new VisionEngine(W, H, resolveConfig(), createRng(9));
  const outs: VisionOutput[] = [];
  for (let f = 0; f < FRAMES; f++) {
    const input: VisionInput = { frameId: f, timestamp: f * 33.3, width: W, height: H, gray: render(f, k1), intrinsics: K, gravity };
    outs.push(engine.process(input));
  }
  const foundIdx = outs.findIndex((o) => o.state === TrackingState.PLANE_FOUND);
  expect(foundIdx, `k1 ${k1}: plane found`).toBeGreaterThan(0);
  // Average the breakdown over the tracked frames after the plane was found.
  const acc: ReprojectionBreakdown = { centerCount: 0, centerErrorPx: 0, edgeCount: 0, edgeErrorPx: 0, planeCount: 0, planeErrorPx: 0, otherCount: 0, otherErrorPx: 0, centerRadius: 0 };
  let frames = 0;
  let meanErr = 0;
  for (let f = foundIdx; f < FRAMES; f++) {
    const mp = outs[f].mapPose;
    if (!mp || mp.framesSinceTracked !== 0) continue;
    const b = mp.reprojection;
    acc.centerCount += b.centerCount;
    acc.centerErrorPx += b.centerErrorPx * b.centerCount;
    acc.edgeCount += b.edgeCount;
    acc.edgeErrorPx += b.edgeErrorPx * b.edgeCount;
    acc.planeCount += b.planeCount;
    acc.planeErrorPx += b.planeErrorPx * b.planeCount;
    acc.otherCount += b.otherCount;
    acc.otherErrorPx += b.otherErrorPx * b.otherCount;
    acc.centerRadius = b.centerRadius;
    meanErr += mp.meanReprojectionErrorPx;
    frames++;
  }
  acc.centerErrorPx /= Math.max(1, acc.centerCount);
  acc.edgeErrorPx /= Math.max(1, acc.edgeCount);
  acc.planeErrorPx /= Math.max(1, acc.planeCount);
  acc.otherErrorPx /= Math.max(1, acc.otherCount);
  return { breakdown: acc, meanErrPx: meanErr / Math.max(1, frames), frames };
}

describe("PnP reprojection breakdown (v16 diagnostics)", () => {
  it("a pinhole-consistent scene has similar center and edge errors; a radially distorted lens shows edge ≫ center", () => {
    const pinhole = run(0);
    const distorted = run(-0.06); // ≈ 24 px of barrel at the corners, ≈ 6 px at the center/edge boundary
    const fmt = (r: ReturnType<typeof run>) =>
      `mean ${r.meanErrPx.toFixed(2)} px | center ${r.breakdown.centerErrorPx.toFixed(2)} (${r.breakdown.centerCount}) edge ${r.breakdown.edgeErrorPx.toFixed(2)} (${r.breakdown.edgeCount}) | plane ${r.breakdown.planeErrorPx.toFixed(2)} (${r.breakdown.planeCount}) other ${r.breakdown.otherErrorPx.toFixed(2)} (${r.breakdown.otherCount}) over ${r.frames} frames`;
    console.log(`[reproj] pinhole:   ${fmt(pinhole)}\n[reproj] distorted: ${fmt(distorted)}`);
    // Both regions populated, the split radius reported.
    expect(pinhole.breakdown.centerCount).toBeGreaterThan(0);
    expect(pinhole.breakdown.edgeCount).toBeGreaterThan(0);
    expect(pinhole.breakdown.centerRadius).toBeCloseTo(0.5, 5);
    // Pinhole: the edge ring is not systematically worse than the center.
    const pinholeRatio = pinhole.breakdown.edgeErrorPx / Math.max(1e-6, pinhole.breakdown.centerErrorPx);
    expect(pinholeRatio).toBeLessThan(1.5);
    // Distorted lens: the edge ring carries the model error.
    const distortedRatio = distorted.breakdown.edgeErrorPx / Math.max(1e-6, distorted.breakdown.centerErrorPx);
    expect(distortedRatio).toBeGreaterThan(1.5);
    expect(distorted.breakdown.edgeErrorPx).toBeGreaterThan(pinhole.breakdown.edgeErrorPx);
    // The per-kind counts add up to the per-region counts (same inlier set).
    expect(pinhole.breakdown.planeCount + pinhole.breakdown.otherCount).toBe(pinhole.breakdown.centerCount + pinhole.breakdown.edgeCount);
  });
});
