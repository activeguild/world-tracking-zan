import { symmetricEigen } from "./Decomposition";
import type { RigidTransform } from "./Pose";

/**
 * Linear two-view triangulation in normalized camera coordinates.
 *
 * Camera 1 is P1 = [I | 0]; camera 2 is P2 = [R | t] with X2 = R X1 + t.
 * Points are (x, y) with z = 1 (already divided by focal length and
 * centered). The result is in camera-1 coordinates, at the (unknown) scale
 * of t.
 */
export interface TriangulationResult {
  /** Point in camera-1 frame. */
  point: Float64Array;
  /** Depth in camera 1 (z). */
  depth1: number;
  /** Depth in camera 2. */
  depth2: number;
  /** Parallax angle between the two viewing rays, radians. */
  parallax: number;
  /** Reprojection error in normalized coords (sum of both views). */
  error: number;
}

const a = new Float64Array(16);
const row = new Float64Array(4);

export function triangulatePoint(
  pose: RigidTransform,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  out?: TriangulationResult,
): TriangulationResult {
  const r = pose.rotation;
  const t = pose.translation;
  // P1 rows: [1 0 0 0], [0 1 0 0], [0 0 1 0]
  // P2 rows: [r0 r1 r2 t0], [r3 r4 r5 t1], [r6 r7 r8 t2]
  // Equations: x·P3 − P1 = 0, y·P3 − P2 = 0 for each camera.
  a.fill(0);
  // Camera 1
  accumulateRow(a, setRow(row, -1, 0, x1, 0));
  accumulateRow(a, setRow(row, 0, -1, y1, 0));
  // Camera 2
  accumulateRow(
    a,
    setRow(row, x2 * r[6] - r[0], x2 * r[7] - r[1], x2 * r[8] - r[2], x2 * t[2] - t[0]),
  );
  accumulateRow(
    a,
    setRow(row, y2 * r[6] - r[3], y2 * r[7] - r[4], y2 * r[8] - r[5], y2 * t[2] - t[1]),
  );
  const eig = symmetricEigen(a, 4);
  // Smallest eigenvector = column 0.
  const w = eig.vectors[12];
  const res = out ?? {
    point: new Float64Array(3),
    depth1: 0,
    depth2: 0,
    parallax: 0,
    error: Number.POSITIVE_INFINITY,
  };
  if (Math.abs(w) < 1e-12) {
    res.depth1 = res.depth2 = 0;
    res.parallax = 0;
    res.error = Number.POSITIVE_INFINITY;
    return res;
  }
  const X = eig.vectors[0] / w;
  const Y = eig.vectors[4] / w;
  const Z = eig.vectors[8] / w;
  res.point[0] = X;
  res.point[1] = Y;
  res.point[2] = Z;
  res.depth1 = Z;
  const X2 = r[0] * X + r[1] * Y + r[2] * Z + t[0];
  const Y2 = r[3] * X + r[4] * Y + r[5] * Z + t[1];
  const Z2 = r[6] * X + r[7] * Y + r[8] * Z + t[2];
  res.depth2 = Z2;

  // Parallax between rays: ray1 = X (from cam1 center), ray2 = X − C2, where C2 = −Rᵀ t.
  const c2x = -(r[0] * t[0] + r[3] * t[1] + r[6] * t[2]);
  const c2y = -(r[1] * t[0] + r[4] * t[1] + r[7] * t[2]);
  const c2z = -(r[2] * t[0] + r[5] * t[1] + r[8] * t[2]);
  const r2x = X - c2x;
  const r2y = Y - c2y;
  const r2z = Z - c2z;
  const n1 = Math.hypot(X, Y, Z);
  const n2 = Math.hypot(r2x, r2y, r2z);
  const cos = n1 > 0 && n2 > 0 ? (X * r2x + Y * r2y + Z * r2z) / (n1 * n2) : 1;
  res.parallax = Math.acos(Math.max(-1, Math.min(1, cos)));

  let err = Number.POSITIVE_INFINITY;
  if (Z > 0 && Z2 > 0) {
    const e1 = Math.hypot(X / Z - x1, Y / Z - y1);
    const e2 = Math.hypot(X2 / Z2 - x2, Y2 / Z2 - y2);
    err = e1 + e2;
  }
  res.error = err;
  return res;
}

function setRow(r: Float64Array, a0: number, a1: number, a2: number, a3: number): Float64Array {
  r[0] = a0;
  r[1] = a1;
  r[2] = a2;
  r[3] = a3;
  return r;
}

function accumulateRow(ata: Float64Array, r: Float64Array): void {
  for (let i = 0; i < 4; i++) {
    const ri = r[i];
    if (ri === 0) continue;
    for (let j = 0; j < 4; j++) ata[i * 4 + j] += ri * r[j];
  }
}

/**
 * Count correspondences that triangulate in front of both cameras with
 * acceptable reprojection error. Used for cheirality-based disambiguation.
 */
export function countCheirality(
  pose: RigidTransform,
  x1: Float64Array,
  y1: Float64Array,
  x2: Float64Array,
  y2: Float64Array,
  idx: ArrayLike<number> | null,
  n: number,
  maxError: number,
): { good: number; parallaxSum: number } {
  let good = 0;
  let parallaxSum = 0;
  const res: TriangulationResult = {
    point: new Float64Array(3),
    depth1: 0,
    depth2: 0,
    parallax: 0,
    error: 0,
  };
  for (let k = 0; k < n; k++) {
    const i = idx ? idx[k] : k;
    triangulatePoint(pose, x1[i], y1[i], x2[i], y2[i], res);
    if (res.depth1 > 0 && res.depth2 > 0 && res.error < maxError) {
      good++;
      parallaxSum += res.parallax;
    }
  }
  return { good, parallaxSum };
}
