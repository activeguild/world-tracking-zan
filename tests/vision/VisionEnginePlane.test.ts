import { describe, expect, it } from "vitest";
import { resolveConfig } from "../../src/ar/ARConfig";
import { TrackingState } from "../../src/ar/ARState";
import { approximateIntrinsics } from "../../src/camera/CameraIntrinsics";
import { type Mat3, mat3Invert, mat3Multiply } from "../../src/math/Matrix";
import { angleBetween, rotationDistance } from "../../src/math/Pose";
import { createRng } from "../../src/vision/OutlierRejection";
import type { VisionInput, VisionOutput } from "../../src/vision/types";
import { VisionEngine } from "../../src/vision/VisionEngine";
import { makeTexture, warpImage } from "../helpers/synthetic";

/**
 * Phase 3 completion criterion: on a floor / desk the detected plane has a
 * normal ≈ world up (gravity), reaching PLANE_FOUND; a wall is not accepted.
 *
 * Image sequences: homography warps of a textured plane under camera
 * motion, H_pix = K (R + t nᵀ/d) K⁻¹, with the plane normal `n` (pointing
 * away from the camera, nᵀX = d) chosen per scenario.
 */
const W = 640;
const H = 480;
const K = approximateIntrinsics(W, H);
const Kmat = new Float64Array([K.fx, 0, K.cx, 0, K.fy, K.cy, 0, 0, 1]);
const KInv = mat3Invert(Kmat)!;
const I: Mat3 = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
const deg = (rad: number) => (rad * 180) / Math.PI;

function pixelHomography(r: Mat3, t: ArrayLike<number>, n: ArrayLike<number>, d: number): Mat3 {
  const h = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) h[i * 3 + j] = r[i * 3 + j] + (t[i] * n[j]) / d;
  return mat3Multiply(mat3Multiply(Kmat, h), KInv);
}

function input(frameId: number, gray: Uint8Array, gravity: number[] | null): VisionInput {
  return { frameId, timestamp: frameId * 33.3, width: W, height: H, gray, intrinsics: K, gravity };
}

const base = makeTexture(W, H, createRng(2025), [12, 30, 70, 160]);

/** Run a lateral camera translation over a plane with normal n (unit, nᵀX = d). */
function runSequence(n: number[], gravity: number[] | null, frames: number, seed: number, step = 0.004) {
  const engine = new VisionEngine(W, H, resolveConfig(), createRng(seed));
  const outs: VisionOutput[] = [];
  for (let f = 0; f < frames; f++) {
    // Camera center C = (step·f, 0.3·step·f, 0) → t = −C (R = I).
    const t = [-step * f, -0.3 * step * f, 0];
    const img = f === 0 ? base : warpImage(base, W, H, pixelHomography(I, t, n, 1), 128);
    outs.push(engine.process(input(f, img, gravity)));
  }
  return { engine, outs };
}

describe("VisionEngine plane detection (Phase 3)", () => {
  it("desk seen from 45° above: PLANE_FOUND with normal ≈ up (gravity)", () => {
    // Plane normal pointing away from the camera: (0, 0.707, 0.707); gravity
    // (down) in the camera frame for a phone pitched 45° down: (0, 0.707, 0.707).
    const n = [0, Math.SQRT1_2, Math.SQRT1_2];
    const gravity = [0, Math.SQRT1_2, Math.SQRT1_2];
    const { outs } = runSequence(n, gravity, 45, 1);

    const initIdx = outs.findIndex((o) => o.mapPose !== null);
    expect(initIdx).toBeGreaterThan(0);
    expect(initIdx).toBeLessThan(30);

    const foundIdx = outs.findIndex((o) => o.state === TrackingState.PLANE_FOUND);
    expect(foundIdx, "PLANE_FOUND reached").toBeGreaterThan(initIdx);

    const last = outs[outs.length - 1];
    expect(last.state).toBe(TrackingState.PLANE_FOUND);
    expect(last.plane).not.toBeNull();
    expect(last.plane!.found).toBe(true);
    expect(last.plane!.horizontal).toBe(true);
    expect(last.plane!.usedGravity).toBe(true);
    expect(last.plane!.inlierCount).toBeGreaterThan(40);
    // Normal is oriented toward the camera (up for a floor) and parallel to gravity.
    expect(deg(angleBetween(last.plane!.normal, gravity, true))).toBeLessThan(5);
    expect(last.plane!.normal[1] + last.plane!.normal[2]).toBeLessThan(0);
    expect(last.quality.planeConfidence).toBeGreaterThan(0.4);

    // Map pose: PnP tracks every frame, camera translates with consistent
    // scale (constant per-frame step) and no rotation.
    const steps: number[] = [];
    for (let f = initIdx + 1; f < outs.length; f++) {
      const mp = outs[f].mapPose!;
      expect(mp).not.toBeNull();
      expect(mp.inlierCount, `frame ${f} PnP inliers`).toBeGreaterThanOrEqual(resolveConfig().landmarks.minPnPInliers);
      expect(mp.framesSinceTracked).toBe(0);
      expect(mp.meanReprojectionErrorPx).toBeLessThan(1.0);
      expect(deg(rotationDistance(Float64Array.from(mp.rotation), I)), `frame ${f} R`).toBeLessThan(1.5);
      expect(mp.landmarkCount).toBeGreaterThan(30);
      const prev = outs[f - 1].mapPose!.translation;
      steps.push(Math.hypot(mp.translation[0] - prev[0], mp.translation[1] - prev[1], mp.translation[2] - prev[2]));
    }
    const medianStep = [...steps].sort((a, b) => a - b)[steps.length >> 1];
    for (const s of steps) expect(Math.abs(s - medianStep) / medianStep, "constant per-frame step").toBeLessThan(0.2);
    const a = outs[initIdx + 5].mapPose!.translation;
    const b = outs[outs.length - 1].mapPose!.translation;
    // Translation direction ≈ (−1, −0.3, 0) normalized and growing in magnitude.
    expect(deg(angleBetween(b, [-1, -0.3, 0]))).toBeLessThan(10);
    expect(Math.hypot(...b)).toBeGreaterThan(Math.hypot(...a));
  });

  it("wall in front of an upright phone: plane detected but never PLANE_FOUND", () => {
    const n = [0, 0, 1];
    const gravity = [0, 1, 0];
    const { outs } = runSequence(n, gravity, 40, 2);
    const initIdx = outs.findIndex((o) => o.mapPose !== null);
    expect(initIdx).toBeGreaterThan(0);
    expect(outs.some((o) => o.state === TrackingState.PLANE_FOUND)).toBe(false);
    // Gravity-constrained fitting reports no horizontal plane on a wall
    // (no dense height cluster with a 2D extent); nothing is ever `found`.
    for (const o of outs) {
      if (o.plane) expect(o.plane.found).toBe(false);
    }
    expect(outs[outs.length - 1].state).toBe(TrackingState.PLANE_DETECTING);
    // The map itself keeps tracking on the wall.
    expect(outs[outs.length - 1].mapPose!.landmarkCount).toBeGreaterThan(30);
  });

  it("floor straight below (phone looking down): PLANE_FOUND with gravity along +Z", () => {
    const n = [0, 0, 1];
    const gravity = [0, 0, 1];
    const { outs } = runSequence(n, gravity, 45, 3);
    expect(outs.some((o) => o.state === TrackingState.PLANE_FOUND)).toBe(true);
    const last = outs[outs.length - 1];
    expect(last.plane!.found).toBe(true);
    expect(deg(angleBetween(last.plane!.normal, [0, 0, -1]))).toBeLessThan(5);
  });

  it("without gravity the fallback up axis is used (desk at 45° still passes)", () => {
    const n = [0, Math.SQRT1_2, Math.SQRT1_2];
    const { outs } = runSequence(n, null, 45, 4);
    const last = outs[outs.length - 1];
    expect(last.plane).not.toBeNull();
    expect(last.plane!.usedGravity).toBe(false);
    expect(last.plane!.horizontalness).toBeCloseTo(Math.SQRT1_2, 1);
    expect(last.state).toBe(TrackingState.PLANE_FOUND);
  });

  it("static camera never initializes the map (no parallax)", () => {
    const engine = new VisionEngine(W, H, resolveConfig(), createRng(5));
    let last: VisionOutput | null = null;
    for (let f = 0; f < 10; f++) last = engine.process(input(f, base, null));
    expect(last!.mapPose).toBeNull();
    expect(last!.plane).toBeNull();
    expect(last!.state).toBe(TrackingState.TRACKING);
  });

  it("map + plane stay within the frame budget", () => {
    const n = [0, Math.SQRT1_2, Math.SQRT1_2];
    const { engine, outs } = runSequence(n, [0, Math.SQRT1_2, Math.SQRT1_2], 30, 6);
    const times = outs.slice(10).map((o) => o.processingMs).sort((a, b) => a - b);
    const median = times[times.length >> 1];
    console.log(`[perf] Phase 3 total median ${median.toFixed(1)} ms`, engine.timing);
    expect(median).toBeLessThan(150);
  });
});
