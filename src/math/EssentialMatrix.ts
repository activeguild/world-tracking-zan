import { column3, det3, svd3, transpose3 } from "./Decomposition";
import { symmetricEigen } from "./Decomposition";
import { type Mat3, mat3Multiply } from "./Matrix";
import type { RigidTransform } from "./Pose";
import { countCheirality } from "./Triangulation";

/**
 * Essential matrix estimation and decomposition (spec §16, §17).
 *
 * Convention: points are in normalized camera coordinates (K⁻¹ applied),
 * x2ᵀ E x1 = 0, and E = [t]× R with X2 = R X1 + t.
 */

/**
 * Normalized 8-point algorithm on the correspondences selected by `idx`
 * (or the first n). Returns E with singular values (1, 1, 0), or null.
 */
export function estimateEssential8pt(
  x1: ArrayLike<number>,
  y1: ArrayLike<number>,
  x2: ArrayLike<number>,
  y2: ArrayLike<number>,
  idx: ArrayLike<number> | null,
  n: number,
): Mat3 | null {
  if (n < 8) return null;
  // Isotropic normalization of both point sets for conditioning.
  const n1 = normalization(x1, y1, idx, n);
  const n2 = normalization(x2, y2, idx, n);
  if (!n1 || !n2) return null;

  const ata = new Float64Array(81);
  const row = new Float64Array(9);
  for (let k = 0; k < n; k++) {
    const i = idx ? idx[k] : k;
    const a = (x1[i] - n1.mx) * n1.s;
    const b = (y1[i] - n1.my) * n1.s;
    const c = (x2[i] - n2.mx) * n2.s;
    const d = (y2[i] - n2.my) * n2.s;
    // x2ᵀ E x1 = 0 → coefficients of E (row-major) are x2_j · x1_k
    row[0] = c * a; row[1] = c * b; row[2] = c;
    row[3] = d * a; row[4] = d * b; row[5] = d;
    row[6] = a;     row[7] = b;     row[8] = 1;
    for (let r = 0; r < 9; r++) {
      const rr = row[r];
      for (let cc = 0; cc < 9; cc++) ata[r * 9 + cc] += rr * row[cc];
    }
  }
  const eig = symmetricEigen(ata, 9);
  const en = new Float64Array(9);
  for (let r = 0; r < 9; r++) en[r] = eig.vectors[r * 9]; // smallest eigenvector (column 0)

  // De-normalize: E = T2ᵀ En T1
  const t1 = new Float64Array([n1.s, 0, -n1.mx * n1.s, 0, n1.s, -n1.my * n1.s, 0, 0, 1]);
  const t2 = new Float64Array([n2.s, 0, -n2.mx * n2.s, 0, n2.s, -n2.my * n2.s, 0, 0, 1]);
  const e = mat3Multiply(mat3Multiply(transpose3(t2), en), t1);
  return enforceEssentialConstraint(e);
}

/** Project onto the essential manifold: singular values (1, 1, 0). */
export function enforceEssentialConstraint(e: Mat3): Mat3 | null {
  const { u, v } = svd3(e);
  const d = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 0]);
  const out = mat3Multiply(mat3Multiply(u, d), transpose3(v));
  if (!Number.isFinite(out[0])) return null;
  return out;
}

function normalization(xs: ArrayLike<number>, ys: ArrayLike<number>, idx: ArrayLike<number> | null, n: number) {
  let mx = 0;
  let my = 0;
  for (let k = 0; k < n; k++) {
    const i = idx ? idx[k] : k;
    mx += xs[i];
    my += ys[i];
  }
  mx /= n;
  my /= n;
  let mean = 0;
  for (let k = 0; k < n; k++) {
    const i = idx ? idx[k] : k;
    mean += Math.hypot(xs[i] - mx, ys[i] - my);
  }
  mean /= n;
  if (mean < 1e-12) return null;
  return { mx, my, s: Math.SQRT2 / mean };
}

/** Squared Sampson distance of a correspondence under E (normalized coords). */
export function sampsonErrorSq(e: Mat3, x1: number, y1: number, x2: number, y2: number): number {
  // Ex1
  const ex0 = e[0] * x1 + e[1] * y1 + e[2];
  const ex1 = e[3] * x1 + e[4] * y1 + e[5];
  const ex2 = e[6] * x1 + e[7] * y1 + e[8];
  // Eᵀx2
  const etx0 = e[0] * x2 + e[3] * y2 + e[6];
  const etx1 = e[1] * x2 + e[4] * y2 + e[7];
  const num = x2 * ex0 + y2 * ex1 + ex2;
  const den = ex0 * ex0 + ex1 * ex1 + etx0 * etx0 + etx1 * etx1;
  if (den < 1e-20) return Number.POSITIVE_INFINITY;
  return (num * num) / den;
}

export interface EssentialRansacConfig {
  /** Sampson threshold in normalized coordinates (pixels / focal length). */
  threshold: number;
  confidence: number;
  maxIterations: number;
  minCorrespondences: number;
}

export interface EssentialRansacResult {
  essential: Mat3 | null;
  inliers: Uint8Array;
  inlierCount: number;
  /** Mean Sampson distance of inliers (normalized coords). */
  meanError: number;
  iterations: number;
}

export function ransacEssential(
  x1: Float64Array,
  y1: Float64Array,
  x2: Float64Array,
  y2: Float64Array,
  n: number,
  config: EssentialRansacConfig,
  rng: () => number = Math.random,
): EssentialRansacResult {
  const inliers = new Uint8Array(n);
  if (n < Math.max(8, config.minCorrespondences)) {
    return { essential: null, inliers, inlierCount: 0, meanError: 0, iterations: 0 };
  }
  const threshSq = config.threshold * config.threshold;
  const sample = new Int32Array(8);
  const best = new Uint8Array(n);
  let bestCount = 0;
  let bestErr = Number.POSITIVE_INFINITY;
  let bestE: Mat3 | null = null;
  let maxIter = config.maxIterations;
  let iter = 0;

  for (; iter < maxIter; iter++) {
    drawDistinct(sample, 8, n, rng);
    const e = estimateEssential8pt(x1, y1, x2, y2, sample, 8);
    if (!e) continue;
    let count = 0;
    let errSum = 0;
    for (let i = 0; i < n; i++) {
      const d = sampsonErrorSq(e, x1[i], y1[i], x2[i], y2[i]);
      if (d < threshSq) {
        inliers[i] = 1;
        count++;
        errSum += d;
      } else {
        inliers[i] = 0;
      }
    }
    if (count > bestCount || (count === bestCount && errSum < bestErr)) {
      bestCount = count;
      bestErr = errSum;
      bestE = e;
      best.set(inliers);
      const w = count / n;
      const p = Math.pow(w, 8);
      if (p >= 1 - 1e-9) maxIter = iter + 1;
      else if (p > 0) {
        const needed = Math.ceil(Math.log(1 - config.confidence) / Math.log(1 - p));
        maxIter = Math.min(config.maxIterations, Math.max(iter + 1, needed));
      }
    }
  }
  if (!bestE || bestCount < 8) {
    inliers.fill(0);
    return { essential: null, inliers, inlierCount: 0, meanError: 0, iterations: iter };
  }

  // Refit on all inliers and re-classify.
  const idx = new Int32Array(bestCount);
  let k = 0;
  for (let i = 0; i < n; i++) if (best[i]) idx[k++] = i;
  const refined = estimateEssential8pt(x1, y1, x2, y2, idx, bestCount) ?? bestE;
  let count = 0;
  let errSum = 0;
  for (let i = 0; i < n; i++) {
    const d = sampsonErrorSq(refined, x1[i], y1[i], x2[i], y2[i]);
    if (d < threshSq) {
      inliers[i] = 1;
      count++;
      errSum += Math.sqrt(d);
    } else {
      inliers[i] = 0;
    }
  }
  if (count < bestCount) {
    inliers.set(best);
    return { essential: bestE, inliers, inlierCount: bestCount, meanError: Math.sqrt(bestErr / bestCount), iterations: iter };
  }
  return { essential: refined, inliers, inlierCount: count, meanError: count ? errSum / count : 0, iterations: iter };
}

function drawDistinct(out: Int32Array, k: number, n: number, rng: () => number): void {
  for (let i = 0; i < k; i++) {
    let v: number;
    let dup: boolean;
    do {
      v = Math.min(n - 1, (rng() * n) | 0);
      dup = false;
      for (let j = 0; j < i; j++) if (out[j] === v) dup = true;
    } while (dup);
    out[i] = v;
  }
}

/** The four (R, t) candidates of an essential matrix. */
export function decomposeEssential(e: Mat3): { rotations: [Mat3, Mat3]; translation: Float64Array } {
  const { u, v } = svd3(e);
  // Rz(π/2) and Rz(−π/2)ᵀ... W = [[0,-1,0],[1,0,0],[0,0,1]]
  const w = new Float64Array([0, -1, 0, 1, 0, 0, 0, 0, 1]);
  const wt = transpose3(w);
  const vt = transpose3(v);
  let r1 = mat3Multiply(mat3Multiply(u, w), vt);
  let r2 = mat3Multiply(mat3Multiply(u, wt), vt);
  if (det3(r1) < 0) r1 = negate(r1);
  if (det3(r2) < 0) r2 = negate(r2);
  const t = column3(u, 2);
  return { rotations: [r1, r2], translation: t };
}

function negate(m: Mat3): Mat3 {
  const o = new Float64Array(9);
  for (let i = 0; i < 9; i++) o[i] = -m[i];
  return o;
}

export interface RecoveredPose {
  pose: RigidTransform;
  /** Correspondences in front of both cameras under the chosen pose. */
  good: number;
  /** Second-best candidate's count (ambiguity indicator). */
  secondBest: number;
  /** Mean parallax (radians) of the good points. */
  meanParallax: number;
}

/**
 * recoverPose: pick the (R, t) candidate with the most points in front of
 * both cameras. `maxError` is the triangulation reprojection tolerance in
 * normalized coordinates.
 */
export function recoverPose(
  e: Mat3,
  x1: Float64Array,
  y1: Float64Array,
  x2: Float64Array,
  y2: Float64Array,
  inliers: Uint8Array | null,
  n: number,
  maxError: number,
): RecoveredPose | null {
  const { rotations, translation } = decomposeEssential(e);
  const idxArr: number[] = [];
  for (let i = 0; i < n; i++) if (!inliers || inliers[i]) idxArr.push(i);
  if (idxArr.length === 0) return null;
  const idx = Int32Array.from(idxArr);
  const negT = new Float64Array([-translation[0], -translation[1], -translation[2]]);
  const candidates: RigidTransform[] = [
    { rotation: rotations[0], translation },
    { rotation: rotations[0], translation: negT },
    { rotation: rotations[1], translation },
    { rotation: rotations[1], translation: negT },
  ];
  let best: RecoveredPose | null = null;
  let secondBest = 0;
  for (const c of candidates) {
    const { good, parallaxSum } = countCheirality(c, x1, y1, x2, y2, idx, idx.length, maxError);
    if (!best || good > best.good) {
      if (best) secondBest = best.good;
      best = { pose: c, good, secondBest: 0, meanParallax: good ? parallaxSum / good : 0 };
    } else if (good > secondBest) {
      secondBest = good;
    }
  }
  if (best) best.secondBest = secondBest;
  return best;
}
