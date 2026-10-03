import { describe, expect, it } from "vitest";
import { resolveConfig } from "../../src/ar/ARConfig";
import { TrackingState } from "../../src/ar/ARState";
import { WorldAnchor } from "../../src/ar/WorldAnchor";
import { approximateIntrinsics } from "../../src/camera/CameraIntrinsics";
import { worldToMap } from "../../src/math/CoordinateSystem";
import { type Mat3, mat3Invert, mat3Multiply, mat3TransformPoint } from "../../src/math/Matrix";
import { createRng } from "../../src/vision/OutlierRejection";
import type { VisionInput, VisionOutput } from "../../src/vision/types";
import { VisionEngine } from "../../src/vision/VisionEngine";
import { makeTexture, warpImage } from "../helpers/synthetic";

/**
 * Phase 4 completion criterion: an object placed by hit test stays on the
 * same physical spot of the desk while the camera moves.
 *
 * Frame f of the sequence is the base texture warped by H_f (pixels). A
 * physical desk point seen at pixel pA in frame A is seen at
 * H_B · H_A⁻¹ · pA in frame B. We place a world point by hit test at pA in
 * frame A and check that its projection through the estimated map pose of
 * frame B lands on that pixel.
 */
const W = 640;
const H = 480;
const K = approximateIntrinsics(W, H);
const Kmat = new Float64Array([K.fx, 0, K.cx, 0, K.fy, K.cy, 0, 0, 1]);
const KInv = mat3Invert(Kmat)!;
const I: Mat3 = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);

function pixelHomography(r: Mat3, t: ArrayLike<number>, n: ArrayLike<number>, d: number): Mat3 {
  const h = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) h[i * 3 + j] = r[i * 3 + j] + (t[i] * n[j]) / d;
  return mat3Multiply(mat3Multiply(Kmat, h), KInv);
}

function input(frameId: number, gray: Uint8Array, gravity: number[] | null): VisionInput {
  return { frameId, timestamp: frameId * 33.3, width: W, height: H, gray, intrinsics: K, gravity };
}

function projectWorld(anchor: WorldAnchor, mp: NonNullable<VisionOutput["mapPose"]>, pw: ArrayLike<number>): [number, number] {
  const pm = worldToMap(anchor.frame!, pw);
  const r = mp.rotation, t = mp.translation;
  const x = r[0] * pm[0] + r[1] * pm[1] + r[2] * pm[2] + t[0];
  const y = r[3] * pm[0] + r[4] * pm[1] + r[5] * pm[2] + t[1];
  const z = r[6] * pm[0] + r[7] * pm[1] + r[8] * pm[2] + t[2];
  return [(x / z) * K.fx + K.cx, (y / z) * K.fy + K.cy];
}

const base = makeTexture(W, H, createRng(2026), [12, 30, 70, 160]);

describe("VisionEngine + WorldAnchor: placed object stays fixed (Phase 4)", () => {
  it("hit-tested point reprojects onto the same desk pixel across camera motion", () => {
    const n = [0, Math.SQRT1_2, Math.SQRT1_2];
    const gravity = [0, Math.SQRT1_2, Math.SQRT1_2];
    const engine = new VisionEngine(W, H, resolveConfig(), createRng(7));
    const anchor = new WorldAnchor({ assumedPlaneDistanceMeters: 0.5 });
    const frames = 60;
    const Hs: Mat3[] = [];
    const outs: VisionOutput[] = [];
    for (let f = 0; f < frames; f++) {
      // Lateral + slight forward motion with a small yaw: t = −R C.
      const yaw = 0.002 * f;
      const R: Mat3 = new Float64Array([Math.cos(yaw), 0, Math.sin(yaw), 0, 1, 0, -Math.sin(yaw), 0, Math.cos(yaw)]);
      const C = [0.004 * f, 0.001 * f, 0.002 * f];
      const t = [
        -(R[0] * C[0] + R[1] * C[1] + R[2] * C[2]),
        -(R[3] * C[0] + R[4] * C[1] + R[5] * C[2]),
        -(R[6] * C[0] + R[7] * C[1] + R[8] * C[2]),
      ];
      const Hf = f === 0 ? I : pixelHomography(R, t, n, 1);
      Hs.push(Hf);
      const img = f === 0 ? base : warpImage(base, W, H, Hf, 128);
      outs.push(engine.process(input(f, img, gravity)));
    }

    const foundIdx = outs.findIndex((o) => o.state === TrackingState.PLANE_FOUND);
    expect(foundIdx).toBeGreaterThan(0);
    const A = foundIdx + 2;
    expect(anchor.create(outs[A].plane!, outs[A].mapPose!)).toBe(true);

    // Place three "cubes" by hit test in frame A.
    const taps: [number, number][] = [[320, 260], [220, 300], [430, 220]];
    const placed = taps.map(([u, v]) => anchor.hitTest(u, v, K, outs[A].mapPose!)!);
    for (const p of placed) {
      expect(p).not.toBeNull();
      expect(p.position[1]).toBeCloseTo(0, 9);
    }
    const HAinv = mat3Invert(Hs[A])!;

    let maxErr = 0;
    const errs: number[] = [];
    const pt = new Float64Array(2);
    for (let B = A + 1; B < frames; B++) {
      const mp = outs[B].mapPose!;
      expect(mp.framesSinceTracked, `frame ${B}`).toBe(0);
      const HBA = mat3Multiply(Hs[B], HAinv);
      for (let k = 0; k < taps.length; k++) {
        mat3TransformPoint(HBA, taps[k][0], taps[k][1], pt);
        const [pu, pv] = projectWorld(anchor, mp, placed[k].position);
        const e = Math.hypot(pu - pt[0], pv - pt[1]);
        errs.push(e);
        maxErr = Math.max(maxErr, e);
      }
    }
    errs.sort((a, b) => a - b);
    console.log(`[world] reprojection drift over ${frames - A - 1} frames: median ${errs[errs.length >> 1].toFixed(2)} px, max ${maxErr.toFixed(2)} px`);
    expect(errs[errs.length >> 1]).toBeLessThan(1.5);
    expect(maxErr).toBeLessThan(4);

    // Camera in world: 0.5 m above the plane at creation, height nearly constant
    // (the simulated motion is parallel-ish to the plane).
    const camA = anchor.cameraPose(outs[A].mapPose!)!;
    expect(camA.position[1]).toBeCloseTo(0.5, 3);
    for (let B = A + 1; B < frames; B++) {
      const cb = anchor.cameraPose(outs[B].mapPose!)!;
      expect(Math.abs(cb.position[1] - 0.5)).toBeLessThan(0.03);
    }
  });
});
