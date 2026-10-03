import { symmetricEigen } from "./Decomposition";
import type { Rng } from "../vision/OutlierRejection";

/**
 * Planes and RANSAC plane fitting (spec §21–§22).
 *
 * Plane equation:  n · X + d = 0,  |n| = 1.
 * `center` is the centroid of the inliers used for the fit.
 */
export interface PlaneModel {
  normal: Float64Array;
  d: number;
  center: Float64Array;
}

export function planeFromPoints(
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number,
  cx: number, cy: number, cz: number,
): PlaneModel | null {
  const ux = bx - ax, uy = by - ay, uz = bz - az;
  const vx = cx - ax, vy = cy - ay, vz = cz - az;
  const nx = uy * vz - uz * vy;
  const ny = uz * vx - ux * vz;
  const nz = ux * vy - uy * vx;
  const len = Math.hypot(nx, ny, nz);
  if (len < 1e-12) return null;
  const normal = new Float64Array([nx / len, ny / len, nz / len]);
  const d = -(normal[0] * ax + normal[1] * ay + normal[2] * az);
  return { normal, d, center: new Float64Array([(ax + bx + cx) / 3, (ay + by + cy) / 3, (az + bz + cz) / 3]) };
}

export function planeDistance(p: PlaneModel, x: number, y: number, z: number): number {
  return p.normal[0] * x + p.normal[1] * y + p.normal[2] * z + p.d;
}

/**
 * Least-squares plane through the selected points (PCA: normal = eigenvector
 * of the smallest covariance eigenvalue). Returns null for < 3 points or a
 * degenerate (collinear) configuration.
 */
export function fitPlaneLeastSquares(points: Float64Array | Float32Array, idx: ArrayLike<number> | null, n: number): PlaneModel | null {
  if (n < 3) return null;
  let mx = 0, my = 0, mz = 0;
  for (let k = 0; k < n; k++) {
    const i = (idx ? idx[k] : k) * 3;
    mx += points[i];
    my += points[i + 1];
    mz += points[i + 2];
  }
  mx /= n; my /= n; mz /= n;
  const cov = new Float64Array(9);
  for (let k = 0; k < n; k++) {
    const i = (idx ? idx[k] : k) * 3;
    const x = points[i] - mx;
    const y = points[i + 1] - my;
    const z = points[i + 2] - mz;
    cov[0] += x * x; cov[1] += x * y; cov[2] += x * z;
    cov[4] += y * y; cov[5] += y * z;
    cov[8] += z * z;
  }
  cov[3] = cov[1]; cov[6] = cov[2]; cov[7] = cov[5];
  const eig = symmetricEigen(cov, 3);
  // Degenerate when the two smallest eigenvalues are both ~0 (collinear).
  if (eig.values[1] <= 1e-12 * Math.max(eig.values[2], 1e-300)) return null;
  const normal = new Float64Array([eig.vectors[0], eig.vectors[3], eig.vectors[6]]);
  const len = Math.hypot(normal[0], normal[1], normal[2]);
  if (len < 1e-12) return null;
  normal[0] /= len; normal[1] /= len; normal[2] /= len;
  const d = -(normal[0] * mx + normal[1] * my + normal[2] * mz);
  return { normal, d, center: new Float64Array([mx, my, mz]) };
}

export interface PlaneRansacConfig {
  /** Inlier distance threshold (same units as the points). */
  threshold: number;
  confidence: number;
  maxIterations: number;
  minInliers: number;
}

export interface PlaneRansacResult {
  plane: PlaneModel | null;
  inliers: Uint8Array;
  inlierCount: number;
  /** RMS distance of inliers to the plane. */
  rmsResidual: number;
  iterations: number;
  /** Best sample support seen, even when below `minInliers` (diagnostics). */
  bestInlierCount: number;
}

export function ransacPlane(
  points: Float64Array | Float32Array,
  n: number,
  config: PlaneRansacConfig,
  rng: Rng = Math.random,
): PlaneRansacResult {
  const inliers = new Uint8Array(n);
  if (n < Math.max(3, config.minInliers)) {
    return { plane: null, inliers, inlierCount: 0, rmsResidual: 0, iterations: 0, bestInlierCount: 0 };
  }
  const thr = config.threshold;
  const best = new Uint8Array(n);
  let bestCount = 0;
  let bestErr = Number.POSITIVE_INFINITY;
  let bestPlane: PlaneModel | null = null;
  let maxIter = config.maxIterations;
  let iter = 0;
  for (; iter < maxIter; iter++) {
    const a = (rng() * n) | 0;
    let b = (rng() * n) | 0;
    let c = (rng() * n) | 0;
    if (b === a) b = (b + 1) % n;
    if (c === a || c === b) c = (c + 1) % n;
    if (c === a || c === b) c = (c + 1) % n;
    const p = planeFromPoints(
      points[a * 3], points[a * 3 + 1], points[a * 3 + 2],
      points[b * 3], points[b * 3 + 1], points[b * 3 + 2],
      points[c * 3], points[c * 3 + 1], points[c * 3 + 2],
    );
    if (!p) continue;
    let count = 0;
    let errSum = 0;
    for (let i = 0; i < n; i++) {
      const dist = Math.abs(planeDistance(p, points[i * 3], points[i * 3 + 1], points[i * 3 + 2]));
      if (dist < thr) {
        inliers[i] = 1;
        count++;
        errSum += dist * dist;
      } else {
        inliers[i] = 0;
      }
    }
    if (count > bestCount || (count === bestCount && errSum < bestErr)) {
      bestCount = count;
      bestErr = errSum;
      bestPlane = p;
      best.set(inliers);
      const w = count / n;
      const pIn = w * w * w;
      if (pIn >= 1 - 1e-9) maxIter = iter + 1;
      else if (pIn > 0) {
        const needed = Math.ceil(Math.log(1 - config.confidence) / Math.log(1 - pIn));
        maxIter = Math.min(config.maxIterations, Math.max(iter + 1, needed));
      }
    }
  }
  if (!bestPlane || bestCount < config.minInliers) {
    inliers.fill(0);
    return { plane: null, inliers, inlierCount: 0, rmsResidual: 0, iterations: iter, bestInlierCount: bestCount };
  }

  // Refit on inliers (PCA), re-classify, and refit once more.
  let plane = bestPlane;
  let count = bestCount;
  inliers.set(best);
  for (let pass = 0; pass < 2; pass++) {
    const idx = new Int32Array(count);
    let k = 0;
    for (let i = 0; i < n; i++) if (inliers[i]) idx[k++] = i;
    const refit = fitPlaneLeastSquares(points, idx, count);
    if (!refit) break;
    let c2 = 0;
    for (let i = 0; i < n; i++) {
      const dist = Math.abs(planeDistance(refit, points[i * 3], points[i * 3 + 1], points[i * 3 + 2]));
      inliers[i] = dist < thr ? 1 : 0;
      if (inliers[i]) c2++;
    }
    if (c2 < bestCount * 0.8) {
      // Refit lost too many points; keep the previous model.
      inliers.set(best);
      count = bestCount;
      plane = bestPlane;
      break;
    }
    plane = refit;
    count = c2;
    best.set(inliers);
    bestCount = c2;
    bestPlane = refit;
  }
  let errSum = 0;
  for (let i = 0; i < n; i++) {
    if (!inliers[i]) continue;
    const dist = planeDistance(plane, points[i * 3], points[i * 3 + 1], points[i * 3 + 2]);
    errSum += dist * dist;
  }
  return {
    plane,
    inliers,
    inlierCount: count,
    rmsResidual: count ? Math.sqrt(errSum / count) : 0,
    iterations: iter,
    bestInlierCount: Math.max(bestCount, count),
  };
}

/**
 * |cos| of the angle between the plane normal and the up direction:
 * 1 = perfectly horizontal, 0 = vertical (wall).
 */
export function horizontalness(normal: ArrayLike<number>, up: ArrayLike<number>): number {
  const lu = Math.hypot(up[0], up[1], up[2]);
  if (lu < 1e-12) return 0;
  return Math.abs(normal[0] * up[0] + normal[1] * up[1] + normal[2] * up[2]) / lu;
}

/** Signed distance of a point to the plane along the normal. */
export function pointToPlaneSigned(p: PlaneModel, x: ArrayLike<number>): number {
  return planeDistance(p, x[0], x[1], x[2]);
}
