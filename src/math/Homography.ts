import { type Mat3, mat3Multiply, mat3TransformPoint, solveLinearSystem } from "./Matrix";

/**
 * Homography estimation (spec §15).
 *
 *   p' ≃ H p
 *
 * Implemented with the normalized DLT: both point sets are translated to the
 * origin and scaled to RMS distance √2 (Hartley normalization), the 8 unknowns
 * (h33 = 1) are solved from the normal equations, and the result is
 * de-normalized. Works for exactly 4 points (RANSAC sample) and for the
 * over-determined refit on all inliers.
 */

interface Normalization {
  /** Transform T such that x_norm = T x. */
  t: Mat3;
  /** Inverse of t. */
  tInv: Mat3;
  pts: Float64Array;
}

function normalizePoints(xs: ArrayLike<number>, ys: ArrayLike<number>, idx: ArrayLike<number> | null, n: number): Normalization | null {
  let mx = 0;
  let my = 0;
  for (let k = 0; k < n; k++) {
    const i = idx ? idx[k] : k;
    mx += xs[i];
    my += ys[i];
  }
  mx /= n;
  my /= n;
  let meanDist = 0;
  for (let k = 0; k < n; k++) {
    const i = idx ? idx[k] : k;
    const dx = xs[i] - mx;
    const dy = ys[i] - my;
    meanDist += Math.sqrt(dx * dx + dy * dy);
  }
  meanDist /= n;
  if (meanDist < 1e-9) return null;
  const s = Math.SQRT2 / meanDist;

  const pts = new Float64Array(n * 2);
  for (let k = 0; k < n; k++) {
    const i = idx ? idx[k] : k;
    pts[k * 2] = (xs[i] - mx) * s;
    pts[k * 2 + 1] = (ys[i] - my) * s;
  }
  const t = new Float64Array([s, 0, -mx * s, 0, s, -my * s, 0, 0, 1]);
  const tInv = new Float64Array([1 / s, 0, mx, 0, 1 / s, my, 0, 0, 1]);
  return { t, tInv, pts };
}

/**
 * Estimate H mapping (x1,y1) → (x2,y2) from the correspondences selected by
 * `idx` (or all `n` if idx is null). Requires n ≥ 4.
 * Returns null when the configuration is degenerate.
 */
export function estimateHomography(
  x1: ArrayLike<number>,
  y1: ArrayLike<number>,
  x2: ArrayLike<number>,
  y2: ArrayLike<number>,
  idx: ArrayLike<number> | null,
  n: number,
): Mat3 | null {
  if (n < 4) return null;
  const n1 = normalizePoints(x1, y1, idx, n);
  const n2 = normalizePoints(x2, y2, idx, n);
  if (!n1 || !n2) return null;

  // Build normal equations AᵀA h = Aᵀb for the 8-parameter DLT with h33 = 1.
  // Each correspondence contributes two rows:
  //   [x y 1 0 0 0 -x'x -x'y] h = x'
  //   [0 0 0 x y 1 -y'x -y'y] h = y'
  const ata = new Float64Array(64);
  const atb = new Float64Array(8);
  const row = new Float64Array(8);
  for (let k = 0; k < n; k++) {
    const x = n1.pts[k * 2];
    const y = n1.pts[k * 2 + 1];
    const xp = n2.pts[k * 2];
    const yp = n2.pts[k * 2 + 1];

    row[0] = x; row[1] = y; row[2] = 1; row[3] = 0; row[4] = 0; row[5] = 0; row[6] = -xp * x; row[7] = -xp * y;
    accumulate(ata, atb, row, xp);
    row[0] = 0; row[1] = 0; row[2] = 0; row[3] = x; row[4] = y; row[5] = 1; row[6] = -yp * x; row[7] = -yp * y;
    accumulate(ata, atb, row, yp);
  }

  if (!solveLinearSystem(ata, atb, 8)) return null;
  const hn = new Float64Array([atb[0], atb[1], atb[2], atb[3], atb[4], atb[5], atb[6], atb[7], 1]);

  // De-normalize: H = T2⁻¹ · Hn · T1
  const tmp = mat3Multiply(n2.tInv, hn);
  const h = mat3Multiply(tmp, n1.t);
  // Normalize so h33 = 1 when possible for numerical comparability.
  if (Math.abs(h[8]) > 1e-12) {
    const inv = 1 / h[8];
    for (let i = 0; i < 9; i++) h[i] *= inv;
  }
  if (!Number.isFinite(h[0])) return null;
  return h;
}

function accumulate(ata: Float64Array, atb: Float64Array, row: Float64Array, rhs: number): void {
  for (let i = 0; i < 8; i++) {
    const ri = row[i];
    if (ri === 0) continue;
    atb[i] += ri * rhs;
    const base = i * 8;
    for (let j = 0; j < 8; j++) ata[base + j] += ri * row[j];
  }
}

const tmpPt = new Float64Array(2);

/** Squared reprojection error of correspondence i under H (p1 → p2). */
export function homographyTransferErrorSq(
  h: Mat3,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
): number {
  if (!mat3TransformPoint(h, x1, y1, tmpPt)) return Number.POSITIVE_INFINITY;
  const dx = tmpPt[0] - x2;
  const dy = tmpPt[1] - y2;
  return dx * dx + dy * dy;
}

/** True when any three of the four points are (nearly) collinear. */
export function isDegenerateQuad(
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  idx: ArrayLike<number>,
  minArea = 1.0,
): boolean {
  for (let a = 0; a < 4; a++) {
    for (let b = a + 1; b < 4; b++) {
      for (let c = b + 1; c < 4; c++) {
        const ia = idx[a];
        const ib = idx[b];
        const ic = idx[c];
        const area = Math.abs(
          (xs[ib] - xs[ia]) * (ys[ic] - ys[ia]) - (xs[ic] - xs[ia]) * (ys[ib] - ys[ia]),
        );
        if (area < minArea) return true;
      }
    }
  }
  return false;
}
