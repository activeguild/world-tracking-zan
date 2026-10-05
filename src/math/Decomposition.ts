/**
 * Small dense matrix decompositions for two-view geometry.
 *
 * All matrices are Float64Array, row-major. Sizes are tiny (≤ 9×9), so a
 * cyclic Jacobi scheme is both simple and numerically robust.
 */

export interface SymmetricEigen {
  /** Eigenvalues in ascending order. */
  values: Float64Array;
  /** Eigenvectors as columns of an n×n row-major matrix (column i ↔ values[i]). */
  vectors: Float64Array;
}

/**
 * Eigen-decomposition of a symmetric n×n matrix by cyclic Jacobi rotations.
 * The input is copied, not modified.
 */
export function symmetricEigen(input: Float64Array, n: number, maxSweeps = 60): SymmetricEigen {
  const a = Float64Array.from(input);
  const v = new Float64Array(n * n);
  for (let i = 0; i < n; i++) v[i * n + i] = 1;

  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p * n + q] * a[p * n + q];
    if (off < 1e-30) break;

    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p * n + q];
        if (Math.abs(apq) < 1e-300) continue;
        const app = a[p * n + p];
        const aqq = a[q * n + q];
        const theta = (aqq - app) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;

        // A ← Jᵀ A J
        for (let k = 0; k < n; k++) {
          const akp = a[k * n + p];
          const akq = a[k * n + q];
          a[k * n + p] = c * akp - s * akq;
          a[k * n + q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p * n + k];
          const aqk = a[q * n + k];
          a[p * n + k] = c * apk - s * aqk;
          a[q * n + k] = s * apk + c * aqk;
        }
        // V ← V J
        for (let k = 0; k < n; k++) {
          const vkp = v[k * n + p];
          const vkq = v[k * n + q];
          v[k * n + p] = c * vkp - s * vkq;
          v[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
  }

  // Sort ascending.
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => a[i * n + i] - a[j * n + j]);
  const values = new Float64Array(n);
  const vectors = new Float64Array(n * n);
  for (let c = 0; c < n; c++) {
    const src = order[c];
    values[c] = a[src * n + src];
    for (let r = 0; r < n; r++) vectors[r * n + c] = v[r * n + src];
  }
  return { values, vectors };
}

export interface Svd3 {
  /** 3×3, columns are left singular vectors. */
  u: Float64Array;
  /** Singular values, descending. */
  s: Float64Array;
  /** 3×3, columns are right singular vectors. */
  v: Float64Array;
}

/**
 * SVD of a 3×3 matrix via the eigen-decomposition of MᵀM.
 * Adequate for essential matrices and homographies (well-conditioned, tiny).
 */
export function svd3(m: Float64Array): Svd3 {
  const mtm = new Float64Array(9);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      mtm[i * 3 + j] = m[i] * m[j] + m[3 + i] * m[3 + j] + m[6 + i] * m[6 + j];
    }
  }
  const eig = symmetricEigen(mtm, 3);
  // Descending order of singular values.
  const s = new Float64Array(3);
  const v = new Float64Array(9);
  for (let c = 0; c < 3; c++) {
    const src = 2 - c;
    s[c] = Math.sqrt(Math.max(0, eig.values[src]));
    for (let r = 0; r < 3; r++) v[r * 3 + c] = eig.vectors[r * 3 + src];
  }
  const u = new Float64Array(9);
  const col = new Float64Array(3);
  const tol = 1e-9 * Math.max(s[0], 1e-300);
  for (let c = 0; c < 3; c++) {
    // u_c = M v_c / s_c ; degenerate columns are completed to an orthonormal basis.
    let ok = s[c] > tol;
    if (ok) {
      for (let r = 0; r < 3; r++) {
        col[r] = m[r * 3] * v[c] + m[r * 3 + 1] * v[3 + c] + m[r * 3 + 2] * v[6 + c];
      }
      const len = Math.hypot(col[0], col[1], col[2]);
      ok = len > 1e-12;
      if (ok) for (let r = 0; r < 3; r++) col[r] /= len;
    }
    if (!ok) {
      if (c === 0) {
        col[0] = 1; col[1] = 0; col[2] = 0;
      } else if (c === 1) {
        const o = orthogonalTo([u[0], u[3], u[6]]);
        col[0] = o[0]; col[1] = o[1]; col[2] = o[2];
      } else {
        const cr = cross3([u[0], u[3], u[6]], [u[1], u[4], u[7]]);
        col[0] = cr[0]; col[1] = cr[1]; col[2] = cr[2];
      }
    }
    u[c] = col[0];
    u[3 + c] = col[1];
    u[6 + c] = col[2];
  }
  return { u, s, v };
}

function orthogonalTo(a: number[]): number[] {
  const ref = Math.abs(a[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const c = cross3(a, ref);
  const l = Math.hypot(c[0], c[1], c[2]);
  return [c[0] / l, c[1] / l, c[2] / l];
}

export function cross3(a: ArrayLike<number>, b: ArrayLike<number>): number[] {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

export function det3(m: Float64Array): number {
  return (
    m[0] * (m[4] * m[8] - m[5] * m[7]) -
    m[1] * (m[3] * m[8] - m[5] * m[6]) +
    m[2] * (m[3] * m[7] - m[4] * m[6])
  );
}

export function transpose3(m: Float64Array, out = new Float64Array(9)): Float64Array {
  const r = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) r[j * 3 + i] = m[i * 3 + j];
  out.set(r);
  return out;
}

export function mulMatVec3(m: Float64Array, v: ArrayLike<number>, out = new Float64Array(3)): Float64Array {
  const x = v[0];
  const y = v[1];
  const z = v[2];
  out[0] = m[0] * x + m[1] * y + m[2] * z;
  out[1] = m[3] * x + m[4] * y + m[5] * z;
  out[2] = m[6] * x + m[7] * y + m[8] * z;
  return out;
}

/** Matrix from three column vectors. */
export function fromColumns3(c0: ArrayLike<number>, c1: ArrayLike<number>, c2: ArrayLike<number>): Float64Array {
  return new Float64Array([c0[0], c1[0], c2[0], c0[1], c1[1], c2[1], c0[2], c1[2], c2[2]]);
}

export function column3(m: Float64Array, c: number): Float64Array {
  return new Float64Array([m[c], m[3 + c], m[6 + c]]);
}
