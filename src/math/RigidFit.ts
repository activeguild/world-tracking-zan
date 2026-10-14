import { det3, svd3 } from "./Decomposition";
import type { RigidTransform } from "./Pose";

/**
 * Rigid (rotation + translation, no scale) least-squares fit of two point
 * sets: after_i ≈ R before_i + t (Kabsch / Umeyama without scale), made
 * robust with a few reweighted passes so that points that moved on their
 * own (a refined landmark, a mis-associated one) do not drag the fit.
 *
 * Used after a bundle adjustment to read the rigid part of the map motion
 * off the adjusted landmarks (改善指示書 Phase 2): the world anchor must
 * follow the landmarks, not the newest keyframe's own pose correction.
 */
export interface RigidFitResult {
  transform: RigidTransform;
  /** Points that carried weight ≥ 0.5 in the last pass. */
  inliers: number;
  /** RMS residual (same units as the points) over the inliers. */
  rms: number;
}

export function rigidFit(
  before: ArrayLike<number>,
  after: ArrayLike<number>,
  n: number,
  passes = 3,
  /** Optional per-point prior weights (e.g. locality); multiplied with the robust weights. */
  prior: ArrayLike<number> | null = null,
): RigidFitResult | null {
  if (n < 3) return null;
  const w = new Float64Array(n).fill(1);
  const base = new Float64Array(n).fill(1);
  if (prior) for (let i = 0; i < n; i++) base[i] = Math.max(0, prior[i]);
  const r = new Float64Array(9);
  const t = new Float64Array(3);
  const resid = new Float64Array(n);
  let inliers = n;
  let rms = 0;
  for (let pass = 0; pass < passes; pass++) {
    if (pass === 0) w.set(base);
    // Weighted centroids.
    let sw = 0;
    const ca = [0, 0, 0], cb = [0, 0, 0];
    for (let i = 0; i < n; i++) {
      const wi = w[i];
      sw += wi;
      for (let a = 0; a < 3; a++) {
        ca[a] += wi * before[i * 3 + a];
        cb[a] += wi * after[i * 3 + a];
      }
    }
    if (sw <= 1e-12) return null;
    for (let a = 0; a < 3; a++) {
      ca[a] /= sw;
      cb[a] /= sw;
    }
    // Cross-covariance H = Σ w (after − cb)(before − ca)ᵀ.
    const h = new Float64Array(9);
    for (let i = 0; i < n; i++) {
      const wi = w[i];
      if (wi === 0) continue;
      const bx = before[i * 3] - ca[0], by = before[i * 3 + 1] - ca[1], bz = before[i * 3 + 2] - ca[2];
      const ax = after[i * 3] - cb[0], ay = after[i * 3 + 1] - cb[1], az = after[i * 3 + 2] - cb[2];
      h[0] += wi * ax * bx; h[1] += wi * ax * by; h[2] += wi * ax * bz;
      h[3] += wi * ay * bx; h[4] += wi * ay * by; h[5] += wi * ay * bz;
      h[6] += wi * az * bx; h[7] += wi * az * by; h[8] += wi * az * bz;
    }
    // R = U diag(1, 1, det(U Vᵀ)) Vᵀ.
    const { u, v } = svd3(h);
    const uvt = new Float64Array(9);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) uvt[i * 3 + j] = u[i * 3] * v[j * 3] + u[i * 3 + 1] * v[j * 3 + 1] + u[i * 3 + 2] * v[j * 3 + 2];
    const d = det3(uvt) < 0 ? -1 : 1;
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) r[i * 3 + j] = u[i * 3] * v[j * 3] + u[i * 3 + 1] * v[j * 3 + 1] + d * u[i * 3 + 2] * v[j * 3 + 2];
    if (!Number.isFinite(r[0])) return null;
    for (let a = 0; a < 3; a++) t[a] = cb[a] - (r[a * 3] * ca[0] + r[a * 3 + 1] * ca[1] + r[a * 3 + 2] * ca[2]);
    // Residuals → weights for the next pass (Huber at 1.5 × the weighted median).
    let med = 0;
    for (let i = 0; i < n; i++) {
      const bx = before[i * 3], by = before[i * 3 + 1], bz = before[i * 3 + 2];
      const px = r[0] * bx + r[1] * by + r[2] * bz + t[0];
      const py = r[3] * bx + r[4] * by + r[5] * bz + t[1];
      const pz = r[6] * bx + r[7] * by + r[8] * bz + t[2];
      resid[i] = Math.hypot(px - after[i * 3], py - after[i * 3 + 1], pz - after[i * 3 + 2]);
    }
    // Weighted median of the residuals over the prior weights.
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => resid[a] - resid[b]);
    let acc = 0;
    let total = 0;
    for (let i = 0; i < n; i++) total += base[i];
    med = resid[order[n - 1]];
    for (const i of order) {
      acc += base[i];
      if (acc >= total / 2) {
        med = resid[i];
        break;
      }
    }
    const sigma = Math.max(1.5 * med, 1e-9);
    inliers = 0;
    let ss = 0, sw2 = 0;
    for (let i = 0; i < n; i++) {
      const robust = resid[i] <= sigma ? 1 : sigma / resid[i];
      w[i] = base[i] * robust;
      if (robust >= 0.5 && base[i] > 0) {
        inliers++;
        ss += base[i] * resid[i] * resid[i];
        sw2 += base[i];
      }
    }
    rms = sw2 > 0 ? Math.sqrt(ss / sw2) : 0;
  }
  return { transform: { rotation: r, translation: t }, inliers, rms };
}
