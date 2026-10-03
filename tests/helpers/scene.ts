import type { CameraIntrinsics } from "../../src/camera/CameraIntrinsics";
import { mulMatVec3 } from "../../src/math/Decomposition";
import type { Mat3 } from "../../src/math/Matrix";
import type { RigidTransform } from "../../src/math/Pose";
import type { Rng } from "../../src/vision/OutlierRejection";

/**
 * Synthetic 3D scenes and projections for two-view geometry tests.
 * Camera frame: X right, Y down, Z forward. Camera 1 is at the origin.
 */

export interface Correspondences {
  x1: Float64Array;
  y1: Float64Array;
  x2: Float64Array;
  y2: Float64Array;
  n: number;
  /** 3D points in camera-1 coordinates. */
  points: Float64Array;
}

/** Random points in a box in front of camera 1. */
export function randomBoxPoints(rng: Rng, n: number, zMin = 1, zMax = 4, halfWidth = 1.5): Float64Array {
  const pts = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) {
    pts[i * 3] = (rng() * 2 - 1) * halfWidth;
    pts[i * 3 + 1] = (rng() * 2 - 1) * halfWidth * 0.75;
    pts[i * 3 + 2] = zMin + rng() * (zMax - zMin);
  }
  return pts;
}

/** Random points on the plane nᵀX = d (n unit), inside the camera-1 frustum. */
export function randomPlanePoints(rng: Rng, n: number, normal: ArrayLike<number>, d: number, spread = 1.5): Float64Array {
  const pts = new Float64Array(n * 3);
  // Build a basis on the plane.
  const nn = normalize(normal);
  const ref = Math.abs(nn[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const u = normalize(cross(nn, ref));
  const v = cross(nn, u);
  const center = [nn[0] * d, nn[1] * d, nn[2] * d];
  for (let i = 0; i < n; i++) {
    const a = (rng() * 2 - 1) * spread;
    const b = (rng() * 2 - 1) * spread;
    pts[i * 3] = center[0] + u[0] * a + v[0] * b;
    pts[i * 3 + 1] = center[1] + u[1] * a + v[1] * b;
    pts[i * 3 + 2] = center[2] + u[2] * a + v[2] * b;
  }
  return pts;
}

/**
 * Project 3D points (camera-1 frame) into both cameras. Returns pixel
 * coordinates when `k` is given, normalized coordinates otherwise.
 * Points behind either camera are dropped.
 */
export function projectTwoViews(
  points: Float64Array,
  pose: RigidTransform,
  k: CameraIntrinsics | null,
  noisePx = 0,
  rng: Rng = Math.random,
): Correspondences {
  const n = points.length / 3;
  const x1 = new Float64Array(n);
  const y1 = new Float64Array(n);
  const x2 = new Float64Array(n);
  const y2 = new Float64Array(n);
  const kept = new Float64Array(n * 3);
  let m = 0;
  const p2 = new Float64Array(3);
  for (let i = 0; i < n; i++) {
    const X = points[i * 3];
    const Y = points[i * 3 + 1];
    const Z = points[i * 3 + 2];
    if (Z <= 0.05) continue;
    mulMatVec3(pose.rotation, [X, Y, Z], p2);
    p2[0] += pose.translation[0];
    p2[1] += pose.translation[1];
    p2[2] += pose.translation[2];
    if (p2[2] <= 0.05) continue;
    let u1 = X / Z;
    let v1 = Y / Z;
    let u2 = p2[0] / p2[2];
    let v2 = p2[1] / p2[2];
    if (k) {
      u1 = u1 * k.fx + k.cx;
      v1 = v1 * k.fy + k.cy;
      u2 = u2 * k.fx + k.cx;
      v2 = v2 * k.fy + k.cy;
      if (u1 < 0 || u1 >= k.width || v1 < 0 || v1 >= k.height) continue;
      if (u2 < 0 || u2 >= k.width || v2 < 0 || v2 >= k.height) continue;
      if (noisePx > 0) {
        u1 += gauss(rng) * noisePx;
        v1 += gauss(rng) * noisePx;
        u2 += gauss(rng) * noisePx;
        v2 += gauss(rng) * noisePx;
      }
    } else if (noisePx > 0) {
      // interpret noise as normalized units
      u1 += gauss(rng) * noisePx;
      v1 += gauss(rng) * noisePx;
      u2 += gauss(rng) * noisePx;
      v2 += gauss(rng) * noisePx;
    }
    x1[m] = u1;
    y1[m] = v1;
    x2[m] = u2;
    y2[m] = v2;
    kept[m * 3] = X;
    kept[m * 3 + 1] = Y;
    kept[m * 3 + 2] = Z;
    m++;
  }
  return {
    x1: x1.subarray(0, m),
    y1: y1.subarray(0, m),
    x2: x2.subarray(0, m),
    y2: y2.subarray(0, m),
    n: m,
    points: kept.subarray(0, m * 3),
  };
}

/** Replace a fraction of correspondences in view 2 with random positions. */
export function corrupt(c: Correspondences, fraction: number, k: CameraIntrinsics | null, rng: Rng): Uint8Array {
  const truth = new Uint8Array(c.n).fill(1);
  for (let i = 0; i < c.n; i++) {
    if (rng() < fraction) {
      if (k) {
        c.x2[i] = rng() * k.width;
        c.y2[i] = rng() * k.height;
      } else {
        c.x2[i] = (rng() * 2 - 1) * 0.6;
        c.y2[i] = (rng() * 2 - 1) * 0.45;
      }
      truth[i] = 0;
    }
  }
  return truth;
}

/** Pose from a camera-2 center C (in camera-1 frame) and rotation R: t = −R C. */
export function poseFromCenter(rotation: Mat3, center: ArrayLike<number>): RigidTransform {
  const rc = mulMatVec3(rotation, center);
  return { rotation, translation: new Float64Array([-rc[0], -rc[1], -rc[2]]) };
}

export function gauss(rng: Rng): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function normalize(v: ArrayLike<number>): number[] {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}

function cross(a: ArrayLike<number>, b: ArrayLike<number>): number[] {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

export const TEST_K: CameraIntrinsics = { fx: 640, fy: 640, cx: 320, cy: 240, width: 640, height: 480 };
