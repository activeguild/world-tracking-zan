import { describe, expect, it } from "vitest";
import { resolveConfig } from "../../src/ar/ARConfig";
import { approximateIntrinsics } from "../../src/camera/CameraIntrinsics";
import type { Mat3 } from "../../src/math/Matrix";
import { rotationAxisAngle } from "../../src/math/Pose";
import { createRng } from "../../src/vision/OutlierRejection";
import type { VisionInput } from "../../src/vision/types";
import { VisionEngine } from "../../src/vision/VisionEngine";
import { makeTexture, sampleBilinear } from "../helpers/synthetic";

/**
 * v16: what a wrong focal length does to the PnP error on a scene with depth
 * variation (floor + back wall), rendered with a 66° camera and tracked
 * with 60 / 66 / 72° assumed.
 *
 * Findings this test pins down:
 *  - During the sideways slide the map was built from, a 10% focal error
 *    changes the PnP error by well under 0.2 px: the map absorbs the error.
 *    The 0.98 → 2.40 px growth seen on Android during its scan is therefore
 *    not a focal-length problem.
 *  - The error of a wrong focal length appears only under large tilts
 *    (±25° here, ≈ +0.5 px) — the tilt test is what exposes a camera-model
 *    error, and even then the effect is modest.
 *  - A per-frame "PnP + f" re-solve on the existing map cannot recover the
 *    true focal length (the ratio f_est / f stays ≈ 1.00 with a 10% error,
 *    measured while developing this test): the map, built with the assumed f, is already a
 *    consistent reconstruction for it. Such a HUD row would be misleading,
 *    so it was not added.
 */
const W = 640;
const H = 480;
const TRUE_FOV = 66;
const K = approximateIntrinsics(W, H, TRUE_FOV);

const nFloor = [0, Math.SQRT1_2, Math.SQRT1_2];
const dFloor = 1;
const gravity = [0, Math.SQRT1_2, Math.SQRT1_2];
const uAxis = [1, 0, 0];
const vAxis = [0, -Math.SQRT1_2, Math.SQRT1_2]; // floor forward direction (n × u)
// Back wall: perpendicular to the floor, facing the camera, 1.6 m ahead of the optical-axis hit.
const nWall = [0, Math.SQRT1_2, -Math.SQRT1_2]; // points towards the camera
const PPM = 640;
const TEX = 3200;
const floorTex = makeTexture(TEX, TEX, createRng(9090), [20, 48, 120, 280]);
const wallTex = makeTexture(TEX, TEX, createRng(4242), [16, 40, 100, 240]);
const floorCenter = [nFloor[0] * dFloor, nFloor[1] * dFloor, nFloor[2] * dFloor];
const wallPoint = [floorCenter[0] + 1.6 * vAxis[0], floorCenter[1] + 1.6 * vAxis[1], floorCenter[2] + 1.6 * vAxis[2]];
const cWall = nWall[0] * wallPoint[0] + nWall[1] * wallPoint[1] + nWall[2] * wallPoint[2];

interface Pose {
  R: Mat3;
  C: number[];
}

function hit(nn: number[], d: number, C: number[], w: number[]): number {
  const denom = nn[0] * w[0] + nn[1] * w[1] + nn[2] * w[2];
  if (Math.abs(denom) <= 1e-6) return -1;
  const s = (d - (nn[0] * C[0] + nn[1] * C[1] + nn[2] * C[2])) / denom;
  return s > 0 ? s : -1;
}

function render(p: Pose): Uint8Array {
  const out = new Uint8Array(W * H);
  const { R, C } = p;
  const w = [0, 0, 0];
  for (let y = 0; y < H; y++) {
    const dy = (y - K.cy) / K.fy;
    for (let x = 0; x < W; x++) {
      const dx = (x - K.cx) / K.fx;
      w[0] = R[0] * dx + R[3] * dy + R[6];
      w[1] = R[1] * dx + R[4] * dy + R[7];
      w[2] = R[2] * dx + R[5] * dy + R[8];
      const sF = hit(nFloor, dFloor, C, w);
      const sW = hit(nWall, cWall, C, w);
      let value = 128;
      if (sF > 0 && (sW <= 0 || sF <= sW)) {
        const px = C[0] + sF * w[0] - floorCenter[0];
        const py = C[1] + sF * w[1] - floorCenter[1];
        const pz = C[2] + sF * w[2] - floorCenter[2];
        const a = px * uAxis[0] + py * uAxis[1] + pz * uAxis[2];
        const b = px * vAxis[0] + py * vAxis[1] + pz * vAxis[2];
        value = sampleBilinear(floorTex, TEX, TEX, a * PPM + TEX / 2, b * PPM + TEX / 2, 128);
      } else if (sW > 0) {
        const px = C[0] + sW * w[0] - wallPoint[0];
        const py = C[1] + sW * w[1] - wallPoint[1];
        const pz = C[2] + sW * w[2] - wallPoint[2];
        const a = px * uAxis[0] + py * uAxis[1] + pz * uAxis[2];
        const b = px * nFloor[0] + py * nFloor[1] + pz * nFloor[2]; // height on the wall
        value = sampleBilinear(wallTex, TEX, TEX, a * PPM + TEX / 2, b * PPM + TEX / 2, 128);
      }
      out[y * W + x] = Math.round(value);
    }
  }
  return out;
}

const I: Mat3 = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
const SLIDE = 90;
const FRAMES = 210;
/**
 * Frames 0–89: a sideways slide with a slow yaw (landmarks on both surfaces
 * get triangulated). Frames 90–209: the on-device tilt test — pitch swings of
 * ±25° about the camera x axis from a fixed position, back to level at the
 * end.
 */
function poseAt(f: number): Pose {
  const s = Math.min(f, SLIDE);
  const yaw = 0.0015 * s;
  const pitch = f > SLIDE ? ((25 * Math.PI) / 180) * Math.sin(((f - SLIDE) / (FRAMES - SLIDE)) * 2 * Math.PI) : 0;
  const ry = rotationAxisAngle([0, 1, 0], yaw);
  const rx = rotationAxisAngle([1, 0, 0], pitch);
  // World-from-camera = Ry · Rx; the camera matrix is its transpose.
  const rot = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) rot[i * 3 + j] = ry[i * 3] * rx[j] + ry[i * 3 + 1] * rx[3 + j] + ry[i * 3 + 2] * rx[6 + j];
  const R: Mat3 = new Float64Array([rot[0], rot[3], rot[6], rot[1], rot[4], rot[7], rot[2], rot[5], rot[8]]);
  return { R: f === 0 ? I : R, C: [0.005 * s, 0.001 * s, 0.002 * s] };
}
const sequence: Uint8Array[] = [];
for (let f = 0; f < FRAMES; f++) sequence.push(render(poseAt(f)));

interface Run {
  fov: number;
  tracked: number;
  /** Mean PnP error (px) over the slide / over the tilt phase. */
  slideErrPx: number;
  tiltErrPx: number;
}

function run(assumedFov: number): Run {
  const k = approximateIntrinsics(W, H, assumedFov);
  const engine = new VisionEngine(W, H, resolveConfig(), createRng(11));
  let tracked = 0;
  let slideErr = 0, slideN = 0, tiltErr = 0, tiltN = 0;
  for (let f = 0; f < FRAMES; f++) {
    const input: VisionInput = { frameId: f, timestamp: f * 33.3, width: W, height: H, gray: sequence[f], intrinsics: k, gravity };
    const mp = engine.process(input).mapPose;
    if (mp?.framesSinceTracked !== 0) continue;
    tracked++;
    if (f < SLIDE) {
      slideErr += mp.meanReprojectionErrorPx;
      slideN++;
    } else {
      tiltErr += mp.meanReprojectionErrorPx;
      tiltN++;
    }
  }
  expect(slideN, `fov ${assumedFov}: tracked during the slide`).toBeGreaterThan(30);
  expect(tiltN, `fov ${assumedFov}: tracked during the tilt`).toBeGreaterThan(60);
  return { fov: assumedFov, tracked, slideErrPx: slideErr / slideN, tiltErrPx: tiltErr / tiltN };
}

describe("focal length error signature (v16)", () => {
  it("a 10% focal error leaves the slide-phase PnP error unchanged and shows only under ±25° tilts", () => {
    const rows = [60, 66, 72].map(run);
    console.log(
      "[focal] " +
        rows.map((r) => `assumed ${r.fov}°: PnP err slide ${r.slideErrPx.toFixed(2)} px  tilt ${r.tiltErrPx.toFixed(2)} px  (tracked ${r.tracked}/${FRAMES})`).join("\n[focal] "),
    );
    const right = rows.find((r) => r.fov === 66)!;
    for (const r of rows) {
      // The map absorbs the focal error for the views it was built from.
      expect(Math.abs(r.slideErrPx - right.slideErrPx), `fov ${r.fov} slide`).toBeLessThan(0.2);
      expect(r.slideErrPx).toBeLessThan(0.6);
    }
    // Tilting exposes it: both wrong assumptions are measurably worse than the right one.
    for (const r of rows) {
      if (r.fov === TRUE_FOV) continue;
      expect(r.tiltErrPx - right.tiltErrPx, `fov ${r.fov} tilt`).toBeGreaterThan(0.3);
    }
    // Even with the right focal length the tilt phase is costlier than the slide
    // (views the map was not built from), but it stays near 1 px on this synthetic scene.
    expect(right.tiltErrPx).toBeLessThan(1.2);
  });
});
