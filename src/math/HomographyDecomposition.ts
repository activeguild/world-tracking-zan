import { cross3, det3, mulMatVec3, svd3, transpose3 } from "./Decomposition";
import { type Mat3, mat3Multiply } from "./Matrix";
import { normalize3, projectToRotation } from "./Pose";

/**
 * Decomposition of a calibrated homography into (R, t/d, n)
 * (Faugeras & Lustman 1988, as in Ma–Soatto–Košecká–Sastry, Alg. 5.2).
 *
 * Input: H in normalized camera coordinates with x2 ≃ H x1, where
 *   H = R + (1/d) t nᵀ,   X2 = R X1 + t,   nᵀ X1 = d (plane in camera 1).
 *
 * Returns up to four physically distinct solutions (before the positive-depth
 * test), or a single pure-rotation solution when H is (numerically) a
 * rotation.
 */
export interface HomographySolution {
  rotation: Mat3;
  /** t / d — translation at plane-distance scale. Zero for pure rotation. */
  translation: Float64Array;
  /** Unit plane normal in camera-1 frame. Zero for pure rotation. */
  normal: Float64Array;
  pureRotation: boolean;
}

export function decomposeHomography(hIn: Mat3, pureRotationTolerance = 0.01): HomographySolution[] {
  // Scale so that the middle singular value is 1.
  const svd = svd3(hIn);
  const s2 = svd.s[1];
  if (!(s2 > 1e-12)) return [];
  const h = new Float64Array(9);
  for (let i = 0; i < 9; i++) h[i] = hIn[i] / s2;
  // Fix the sign so that det(H) > 0 (H is close to a rotation up to the plane term).
  if (det3(h) < 0) for (let i = 0; i < 9; i++) h[i] = -h[i];

  const s1 = svd.s[0] / s2;
  const s3 = svd.s[2] / s2;

  if (s1 - s3 < pureRotationTolerance) {
    return [
      {
        rotation: projectToRotation(h),
        translation: new Float64Array(3),
        normal: new Float64Array(3),
        pureRotation: true,
      },
    ];
  }

  // Eigenvectors of HᵀH (right singular vectors) — recompute from the scaled H
  // to keep signs consistent with `h`.
  const { v } = svd3(h);
  const v1 = [v[0], v[3], v[6]];
  const v2 = [v[1], v[4], v[7]];
  const v3 = [v[2], v[5], v[8]];

  const a = Math.sqrt(Math.max(0, 1 - s3 * s3));
  const b = Math.sqrt(Math.max(0, s1 * s1 - 1));
  const denom = Math.sqrt(Math.max(1e-24, s1 * s1 - s3 * s3));

  const u1 = normalize3([
    (a * v1[0] + b * v3[0]) / denom,
    (a * v1[1] + b * v3[1]) / denom,
    (a * v1[2] + b * v3[2]) / denom,
  ]);
  const u2 = normalize3([
    (a * v1[0] - b * v3[0]) / denom,
    (a * v1[1] - b * v3[1]) / denom,
    (a * v1[2] - b * v3[2]) / denom,
  ]);

  const solutions: HomographySolution[] = [];
  for (const u of [u1, u2]) {
    const n = normalize3(cross3(v2, u));
    const hv2 = mulMatVec3(h, v2);
    const hu = mulMatVec3(h, u);
    const hn = normalize3(cross3(hv2, hu));
    // U = [v2 u v2×u], W = [Hv2 Hu Hv2×Hu], R = W Uᵀ
    const U = new Float64Array([v2[0], u[0], n[0], v2[1], u[1], n[1], v2[2], u[2], n[2]]);
    const W = new Float64Array([hv2[0], hu[0], hn[0], hv2[1], hu[1], hn[1], hv2[2], hu[2], hn[2]]);
    let r = mat3Multiply(W, transpose3(U));
    r = projectToRotation(r);
    // t/d = (H − R) n
    const hr = new Float64Array(9);
    for (let i = 0; i < 9; i++) hr[i] = h[i] - r[i];
    const t = mulMatVec3(hr, n);
    solutions.push({ rotation: r, translation: t, normal: n, pureRotation: false });
    // Mirror solution: (R, −t, −n)
    solutions.push({
      rotation: r,
      translation: new Float64Array([-t[0], -t[1], -t[2]]),
      normal: new Float64Array([-n[0], -n[1], -n[2]]),
      pureRotation: false,
    });
  }
  return solutions;
}

/**
 * Positive-depth support of a homography solution over correspondences in
 * normalized coordinates: a point lies in front of camera 1 iff nᵀx1 > 0
 * (with d > 0), and in front of camera 2 iff the z of R·X1/d + t/d is > 0.
 */
export function homographySolutionSupport(
  sol: HomographySolution,
  x1: Float64Array,
  y1: Float64Array,
  idx: ArrayLike<number> | null,
  n: number,
): number {
  if (sol.pureRotation) return n;
  const r = sol.rotation;
  const t = sol.translation;
  const nn = sol.normal;
  let good = 0;
  for (let k = 0; k < n; k++) {
    const i = idx ? idx[k] : k;
    const x = x1[i];
    const y = y1[i];
    const dn = nn[0] * x + nn[1] * y + nn[2];
    if (dn <= 1e-9) continue;
    const inv = 1 / dn; // X1/d = x1 / (nᵀx1)
    const z2 = (r[6] * x + r[7] * y + r[8]) * inv + t[2];
    if (z2 > 0) good++;
  }
  return good;
}
