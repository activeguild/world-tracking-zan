/**
 * Minimal dense linear algebra used by the vision pipeline.
 *
 * Everything operates on Float64Array in row-major order so it can later be
 * swapped for a WASM implementation without API changes.
 */

/** 3×3 matrix, row-major. */
export type Mat3 = Float64Array;

export function mat3Identity(): Mat3 {
  const m = new Float64Array(9);
  m[0] = m[4] = m[8] = 1;
  return m;
}

export function mat3Multiply(a: Mat3, b: Mat3, out: Mat3 = new Float64Array(9)): Mat3 {
  const r = new Float64Array(9);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      r[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
    }
  }
  out.set(r);
  return out;
}

export function mat3Invert(m: Mat3, out: Mat3 = new Float64Array(9)): Mat3 | null {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h;
  const B = -(d * i - f * g);
  const C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-14) return null;
  const inv = 1 / det;
  out[0] = A * inv;
  out[1] = -(b * i - c * h) * inv;
  out[2] = (b * f - c * e) * inv;
  out[3] = B * inv;
  out[4] = (a * i - c * g) * inv;
  out[5] = -(a * f - c * d) * inv;
  out[6] = C * inv;
  out[7] = -(a * h - b * g) * inv;
  out[8] = (a * e - b * d) * inv;
  return out;
}

/** Apply a 3×3 projective transform to (x, y). Returns null if w ≈ 0. */
export function mat3TransformPoint(m: Mat3, x: number, y: number, out: Float64Array): boolean {
  const w = m[6] * x + m[7] * y + m[8];
  if (Math.abs(w) < 1e-12) return false;
  const iw = 1 / w;
  out[0] = (m[0] * x + m[1] * y + m[2]) * iw;
  out[1] = (m[3] * x + m[4] * y + m[5]) * iw;
  return true;
}

/**
 * Solve the dense linear system A x = b in place using Gaussian elimination
 * with partial pivoting. `a` is n×n row-major, `b` has length n.
 * Returns false if the matrix is singular.
 */
export function solveLinearSystem(a: Float64Array, b: Float64Array, n: number): boolean {
  for (let col = 0; col < n; col++) {
    // Pivot
    let pivot = col;
    let max = Math.abs(a[col * n + col]);
    for (let r = col + 1; r < n; r++) {
      const v = Math.abs(a[r * n + col]);
      if (v > max) {
        max = v;
        pivot = r;
      }
    }
    if (max < 1e-12) return false;
    if (pivot !== col) {
      for (let k = 0; k < n; k++) {
        const t = a[col * n + k];
        a[col * n + k] = a[pivot * n + k];
        a[pivot * n + k] = t;
      }
      const tb = b[col];
      b[col] = b[pivot];
      b[pivot] = tb;
    }
    // Eliminate
    const inv = 1 / a[col * n + col];
    for (let r = col + 1; r < n; r++) {
      const factor = a[r * n + col] * inv;
      if (factor === 0) continue;
      for (let k = col; k < n; k++) a[r * n + k] -= factor * a[col * n + k];
      b[r] -= factor * b[col];
    }
  }
  // Back substitution
  for (let r = n - 1; r >= 0; r--) {
    let s = b[r];
    for (let k = r + 1; k < n; k++) s -= a[r * n + k] * b[k];
    b[r] = s / a[r * n + r];
  }
  return true;
}
