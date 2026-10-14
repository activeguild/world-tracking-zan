import { approximateIntrinsics } from "../../src/camera/CameraIntrinsics";
import type { Mat3 } from "../../src/math/Matrix";
import { rotationAxisAngle } from "../../src/math/Pose";
import { createRng } from "../../src/vision/OutlierRejection";
import { makeTexture, sampleBilinear } from "./synthetic";

/**
 * A textured floor (45° below the first camera, 1 m along its normal) plus a
 * textured back wall perpendicular to it 1.6 m ahead, rendered exactly for
 * an arbitrary camera pose with a 66° camera. Non-planar on purpose: a wrong
 * camera model or an inconsistent map shows up here, while a single plane
 * absorbs both.
 */
export const TP_W = 640;
export const TP_H = 480;
export const TP_FOV = 66;
export const TP_K = approximateIntrinsics(TP_W, TP_H, TP_FOV);

export const tpFloorNormal = [0, Math.SQRT1_2, Math.SQRT1_2];
export const tpFloorD = 1;
export const tpGravity = [0, Math.SQRT1_2, Math.SQRT1_2];
const uAxis = [1, 0, 0];
const vAxis = [0, -Math.SQRT1_2, Math.SQRT1_2]; // floor forward direction (n × u)
const nWall = [0, Math.SQRT1_2, -Math.SQRT1_2]; // points towards the camera
const PPM = 640;
const TEX = 3200;
const floorTex = makeTexture(TEX, TEX, createRng(9090), [20, 48, 120, 280]);
const wallTex = makeTexture(TEX, TEX, createRng(4242), [16, 40, 100, 240]);
const floorCenter = [tpFloorNormal[0] * tpFloorD, tpFloorNormal[1] * tpFloorD, tpFloorNormal[2] * tpFloorD];
const wallPoint = [floorCenter[0] + 1.6 * vAxis[0], floorCenter[1] + 1.6 * vAxis[1], floorCenter[2] + 1.6 * vAxis[2]];
const cWall = nWall[0] * wallPoint[0] + nWall[1] * wallPoint[1] + nWall[2] * wallPoint[2];

export interface TPPose {
  /** World → camera rotation (X_cam = R (X − C)). */
  R: Mat3;
  /** Camera centre in the camera-0 frame. */
  C: number[];
}

function hit(nn: number[], d: number, C: number[], w: number[]): number {
  const denom = nn[0] * w[0] + nn[1] * w[1] + nn[2] * w[2];
  if (Math.abs(denom) <= 1e-6) return -1;
  const s = (d - (nn[0] * C[0] + nn[1] * C[1] + nn[2] * C[2])) / denom;
  return s > 0 ? s : -1;
}

/** World direction of the ray through pixel (x, y) for pose p. */
export function tpRay(p: TPPose, x: number, y: number): number[] {
  const dx = (x - TP_K.cx) / TP_K.fx;
  const dy = (y - TP_K.cy) / TP_K.fy;
  const { R } = p;
  return [R[0] * dx + R[3] * dy + R[6], R[1] * dx + R[4] * dy + R[7], R[2] * dx + R[5] * dy + R[8]];
}

/** The floor point seen at pixel (x, y) of pose p (camera-0 frame), or null when the ray misses the floor. */
export function tpFloorPoint(p: TPPose, x: number, y: number): number[] | null {
  const w = tpRay(p, x, y);
  const s = hit(tpFloorNormal, tpFloorD, p.C, w);
  if (s <= 0) return null;
  return [p.C[0] + s * w[0], p.C[1] + s * w[1], p.C[2] + s * w[2]];
}

/** Project a camera-0-frame point with pose p (pixels), or null when behind the camera. */
export function tpProject(p: TPPose, X: number[]): [number, number] | null {
  const { R, C } = p;
  const d = [X[0] - C[0], X[1] - C[1], X[2] - C[2]];
  const xc = R[0] * d[0] + R[1] * d[1] + R[2] * d[2];
  const yc = R[3] * d[0] + R[4] * d[1] + R[5] * d[2];
  const zc = R[6] * d[0] + R[7] * d[1] + R[8] * d[2];
  if (zc <= 1e-6) return null;
  return [(xc / zc) * TP_K.fx + TP_K.cx, (yc / zc) * TP_K.fy + TP_K.cy];
}

export function tpRender(p: TPPose): Uint8Array {
  const out = new Uint8Array(TP_W * TP_H);
  const { R, C } = p;
  const w = [0, 0, 0];
  for (let y = 0; y < TP_H; y++) {
    const dy = (y - TP_K.cy) / TP_K.fy;
    for (let x = 0; x < TP_W; x++) {
      const dx = (x - TP_K.cx) / TP_K.fx;
      w[0] = R[0] * dx + R[3] * dy + R[6];
      w[1] = R[1] * dx + R[4] * dy + R[7];
      w[2] = R[2] * dx + R[5] * dy + R[8];
      const sF = hit(tpFloorNormal, tpFloorD, C, w);
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
        const b = px * tpFloorNormal[0] + py * tpFloorNormal[1] + pz * tpFloorNormal[2]; // height on the wall
        value = sampleBilinear(wallTex, TEX, TEX, a * PPM + TEX / 2, b * PPM + TEX / 2, 128);
      }
      out[y * TP_W + x] = Math.round(value);
    }
  }
  return out;
}

const I: Mat3 = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);

/** Camera pose from a world-from-camera yaw (about y) then pitch (about x) and a centre. */
export function tpPose(C: number[], yawRad: number, pitchRad: number): TPPose {
  if (yawRad === 0 && pitchRad === 0) return { R: I, C };
  const ry = rotationAxisAngle([0, 1, 0], yawRad);
  const rx = rotationAxisAngle([1, 0, 0], pitchRad);
  const rot = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) rot[i * 3 + j] = ry[i * 3] * rx[j] + ry[i * 3 + 1] * rx[3 + j] + ry[i * 3 + 2] * rx[6 + j];
  const R: Mat3 = new Float64Array([rot[0], rot[3], rot[6], rot[1], rot[4], rot[7], rot[2], rot[5], rot[8]]);
  return { R, C };
}
