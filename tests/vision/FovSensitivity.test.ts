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
 * 修正指示書 v2 §15: FOV sensitivity. The same synthetic camera motion
 * (rendered with a 66° camera) is tracked with the engine configured for
 * 60 / 63 / 66 / 69 / 72°. For each: camera X / Z displacement in world,
 * object screen drift, PnP reprojection error. A wrong focal length must
 * show up as drift, and the correct one must be the best.
 */
const W = 640;
const H = 480;
const TRUE_FOV = 66;
const K = approximateIntrinsics(W, H, TRUE_FOV);
const Kmat = new Float64Array([K.fx, 0, K.cx, 0, K.fy, K.cy, 0, 0, 1]);
const KInv = mat3Invert(Kmat)!;
const I: Mat3 = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
const base = makeTexture(W, H, createRng(2026), [12, 30, 70, 160]);
const n = [0, Math.SQRT1_2, Math.SQRT1_2];
const gravity = [0, Math.SQRT1_2, Math.SQRT1_2];
const FRAMES = 60;

function pixelHomography(r: Mat3, t: ArrayLike<number>, nn: ArrayLike<number>, d: number): Mat3 {
  const h = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) h[i * 3 + j] = r[i * 3 + j] + (t[i] * nn[j]) / d;
  return mat3Multiply(mat3Multiply(Kmat, h), KInv);
}

/** Frames rendered once with the true camera. */
const sequence: { img: Uint8Array; H: Mat3 }[] = [];
for (let f = 0; f < FRAMES; f++) {
  const yaw = 0.002 * f;
  const R: Mat3 = new Float64Array([Math.cos(yaw), 0, Math.sin(yaw), 0, 1, 0, -Math.sin(yaw), 0, Math.cos(yaw)]);
  const C = [0.008 * f, 0.002 * f, 0.004 * f];
  const t = [
    -(R[0] * C[0] + R[1] * C[1] + R[2] * C[2]),
    -(R[3] * C[0] + R[4] * C[1] + R[5] * C[2]),
    -(R[6] * C[0] + R[7] * C[1] + R[8] * C[2]),
  ];
  const Hf = f === 0 ? I : pixelHomography(R, t, n, 1);
  sequence.push({ img: f === 0 ? base : warpImage(base, W, H, Hf, 128), H: Hf });
}

interface Row {
  fov: number;
  cameraDx: number;
  cameraDz: number;
  driftMedianPx: number;
  driftMaxPx: number;
  reprojPx: number;
}

function runWithFov(fov: number): Row {
  const k = approximateIntrinsics(W, H, fov);
  const engine = new VisionEngine(W, H, resolveConfig(), createRng(7));
  const anchor = new WorldAnchor({ assumedPlaneDistanceMeters: 0.5 });
  const outs: VisionOutput[] = [];
  for (let f = 0; f < FRAMES; f++) {
    const input: VisionInput = { frameId: f, timestamp: f * 33.3, width: W, height: H, gray: sequence[f].img, intrinsics: k, gravity };
    outs.push(engine.process(input));
  }
  const foundIdx = outs.findIndex((o) => o.state === TrackingState.PLANE_FOUND);
  expect(foundIdx, `fov ${fov}: plane found`).toBeGreaterThan(0);
  const A = foundIdx + 2;
  expect(anchor.create(outs[A].plane!, outs[A].mapPose!)).toBe(true);
  const taps: [number, number][] = [[320, 260], [220, 300], [430, 220]];
  const placed = taps.map(([u, v]) => anchor.hitTest(u, v, k, outs[A].mapPose!)!);
  const HAinv = mat3Invert(sequence[A].H)!;
  const errs: number[] = [];
  let reproj = 0;
  let count = 0;
  const pt = new Float64Array(2);
  for (let B = A + 1; B < FRAMES; B++) {
    const mp = outs[B].mapPose!;
    if (mp.framesSinceTracked !== 0) continue;
    reproj += mp.meanReprojectionErrorPx;
    count++;
    const HBA = mat3Multiply(sequence[B].H, HAinv);
    for (let j = 0; j < taps.length; j++) {
      mat3TransformPoint(HBA, taps[j][0], taps[j][1], pt);
      const pm = worldToMap(anchor.frame!, placed[j].position);
      const r = mp.rotation, t = mp.translation;
      const x = r[0] * pm[0] + r[1] * pm[1] + r[2] * pm[2] + t[0];
      const y = r[3] * pm[0] + r[4] * pm[1] + r[5] * pm[2] + t[1];
      const z = r[6] * pm[0] + r[7] * pm[1] + r[8] * pm[2] + t[2];
      errs.push(Math.hypot((x / z) * k.fx + k.cx - pt[0], (y / z) * k.fy + k.cy - pt[1]));
    }
  }
  errs.sort((a, b) => a - b);
  const camA = anchor.cameraPose(outs[A].mapPose!)!.position;
  const camZ = anchor.cameraPose(outs[FRAMES - 1].mapPose!)!.position;
  return {
    fov,
    cameraDx: camZ[0] - camA[0],
    cameraDz: camZ[2] - camA[2],
    driftMedianPx: errs[errs.length >> 1],
    driftMaxPx: errs[errs.length - 1],
    reprojPx: count ? reproj / count : NaN,
  };
}

describe("FOV sensitivity (v2 §15)", () => {
  it("drift of placed objects and recovered camera motion vs assumed field of view", () => {
    const rows = [60, 63, 66, 69, 72].map(runWithFov);
    console.log(
      "[fov] " +
        rows
          .map(
            (r) =>
              `fov ${r.fov}: camera Δx ${r.cameraDx.toFixed(3)} m Δz ${r.cameraDz.toFixed(3)} m | drift median ${r.driftMedianPx.toFixed(2)} px max ${r.driftMaxPx.toFixed(2)} px | PnP ${r.reprojPx.toFixed(2)} px`,
          )
          .join("\n[fov] "),
    );
    // Finding: on a *planar* scene a wrong focal length does not make placed
    // objects drift — the reconstruction is consistently distorted and the
    // plane stays a plane — it only rescales the recovered camera motion
    // (a wider assumed FOV → larger displacement for the same pixel motion).
    // So FOV error can explain on-device drift only through the non-planar
    // part of a real scene; the synthetic test cannot rank FOVs by drift.
    for (const r of rows) {
      expect(r.driftMedianPx, `fov ${r.fov} drift`).toBeLessThan(0.5);
      expect(r.reprojPx, `fov ${r.fov} PnP`).toBeLessThan(1);
    }
    for (let i = 1; i < rows.length; i++) {
      expect(rows[i].cameraDx, `Δx grows with the assumed FOV (${rows[i].fov}°)`).toBeGreaterThan(rows[i - 1].cameraDx);
    }
    const trueRow = rows.find((r) => r.fov === TRUE_FOV)!;
    // Simulated lateral motion at frame 59 vs frame A ≈ 0.008·(59−A) map units × 0.5 m/unit
    // scaled by the camera–plane distance (1 map unit): ≈ 0.15–0.17 m.
    expect(trueRow.cameraDx).toBeGreaterThan(0.12);
    expect(trueRow.cameraDx).toBeLessThan(0.2);
  });
});
