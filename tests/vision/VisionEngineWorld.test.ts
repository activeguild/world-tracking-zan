import { describe, expect, it } from "vitest";
import { resolveConfig, type PartialARConfig } from "../../src/ar/ARConfig";
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
 * Phase 4 completion criterion (and 修正指示書 §22, Test A–C): an object placed
 * by hit test stays on the same physical spot of the desk while the camera
 * moves.
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
const n = [0, Math.SQRT1_2, Math.SQRT1_2];
const gravity = [0, Math.SQRT1_2, Math.SQRT1_2];

interface RunResult {
  outs: VisionOutput[];
  anchor: WorldAnchor;
  A: number;
  errs: number[];
  maxErr: number;
  heights: number[];
  expectedHeights: number[];
}

/** Run the synthetic desk sequence; place three objects at frame A; measure drift. */
function run(config: PartialARConfig, frames = 60): RunResult {
  const engine = new VisionEngine(W, H, resolveConfig(config), createRng(7));
  const anchor = new WorldAnchor({ assumedPlaneDistanceMeters: 0.5 });
  const Hs: Mat3[] = [];
  const outs: VisionOutput[] = [];
  const centers: number[][] = [];
  for (let f = 0; f < frames; f++) {
    // Lateral + slight forward motion with a small yaw: t = −R C.
    const yaw = 0.002 * f;
    const R: Mat3 = new Float64Array([Math.cos(yaw), 0, Math.sin(yaw), 0, 1, 0, -Math.sin(yaw), 0, Math.cos(yaw)]);
    const C = [0.008 * f, 0.002 * f, 0.004 * f];
    centers.push(C);
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
  const plane = outs[A].planeAnchor ?? outs[A].plane!;
  expect(anchor.create(plane, outs[A].mapPose!)).toBe(true);

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

  // Camera height above the plane: 0.5 m at creation, then following the
  // simulated motion (distance to the plane nᵀX = 1 is 1 − n·C), scaled by
  // the world scale fixed at frame A.
  const distAt = (f: number) => 1 - (n[0] * centers[f][0] + n[1] * centers[f][1] + n[2] * centers[f][2]);
  const heights: number[] = [];
  const expectedHeights: number[] = [];
  for (let B = A; B < frames; B++) {
    heights.push(anchor.cameraPose(outs[B].mapPose!)!.position[1]);
    expectedHeights.push((0.5 * distAt(B)) / distAt(A));
  }
  return { outs, anchor, A, errs, maxErr, heights, expectedHeights };
}

describe("VisionEngine + WorldAnchor: placed object stays fixed (Phase 4, 修正指示書 §22)", () => {
  it("fixed map + PnP (default): hit-tested points reproject onto the same desk pixel across camera motion", () => {
    const r = run({});
    const median = r.errs[r.errs.length >> 1];
    console.log(`[world] fixed-map PnP drift over ${r.outs.length - r.A - 1} frames: median ${median.toFixed(2)} px, max ${r.maxErr.toFixed(2)} px`);
    expect(median).toBeLessThan(1.0);
    expect(r.maxErr).toBeLessThan(3);
    const last = r.outs[r.outs.length - 1];
    expect(last.mapPose!.source).toBe("map");
    expect(last.planeAnchor).toBeNull();
    // PnP telemetry (v2 §7): per-frame camera motion is small and smooth.
    for (let B = r.A + 1; B < r.outs.length; B++) {
      const mp = r.outs[B].mapPose!;
      expect(mp.translationHeld, `frame ${B}`).toBe(false);
      expect(mp.deltaRotationDeg, `frame ${B} Δrot`).toBeLessThan(1);
      expect(mp.deltaTranslation * r.anchor.frame!.scale, `frame ${B} Δt`).toBeLessThan(0.02);
    }
    expect(r.heights[0]).toBeCloseTo(0.5, 3);
    for (let i = 1; i < r.heights.length; i++) {
      expect(Math.abs(r.heights[i] - r.expectedHeights[i]), `height ${i}`).toBeLessThan(0.02);
    }
  });

  it("with landmark depth refinement on (A/B, v2 §26) the synthetic desk also stays fixed", () => {
    const r = run({ landmarks: { enableLandmarkDepthRefinement: true } });
    const median = r.errs[r.errs.length >> 1];
    console.log(`[world] refinement-on drift over ${r.outs.length - r.A - 1} frames: median ${median.toFixed(2)} px, max ${r.maxErr.toFixed(2)} px`);
    expect(median).toBeLessThan(1.5);
    expect(r.maxErr).toBeLessThan(4);
  });

  it("plane-anchored pose (experimental, planeTracking.enabled): hit-tested points reproject onto the same desk pixel", () => {
    const r = run({ planeTracking: { enabled: true } });
    const median = r.errs[r.errs.length >> 1];
    console.log(`[world] plane-anchored drift over ${r.outs.length - r.A - 1} frames: median ${median.toFixed(2)} px, max ${r.maxErr.toFixed(2)} px`);
    expect(median).toBeLessThan(1.0);
    expect(r.maxErr).toBeLessThan(3);

    // The world plane is fixed once and the pose is plane-relative from then on.
    const anchorFrame = r.outs[r.A].planeAnchor!.frameId;
    for (let B = r.A; B < r.outs.length; B++) {
      const o = r.outs[B];
      expect(o.planeAnchor?.frameId, `frame ${B} anchor`).toBe(anchorFrame);
      expect(o.state, `frame ${B} state`).toBe(TrackingState.PLANE_FOUND);
    }
    const planeFrames = r.outs.slice(r.A + 1).filter((o) => o.mapPose!.source === "plane").length;
    expect(planeFrames).toBeGreaterThanOrEqual((r.outs.length - r.A - 1) * 0.9);
    const last = r.outs[r.outs.length - 1];
    expect(last.planePose!.tracked).toBe(true);
    expect(last.planePose!.inlierCount).toBeGreaterThanOrEqual(40);
    expect(last.planePose!.inlierRatio).toBeGreaterThan(0.7);
    expect(last.planePose!.reprojectionErrorPx).toBeLessThan(1.5);

    expect(r.heights[0]).toBeCloseTo(0.5, 3);
    for (let i = 1; i < r.heights.length; i++) {
      expect(Math.abs(r.heights[i] - r.expectedHeights[i]), `height ${i}`).toBeLessThan(0.02);
    }
  });

});
