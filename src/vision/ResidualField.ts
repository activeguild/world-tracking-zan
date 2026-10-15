/**
 * Residual vector field of one PnP frame (v22 diagnostics).
 *
 * Every breakdown so far split the inlier reprojection error by *magnitude*
 * (image region, landmark kind, parallax, age, BA mode, link source) and on
 * Android every split came out uniform at 2–3 px after the first fast move
 * (recordings 15–32). The *direction* of the residuals was never looked at.
 * This fits the residual vectors d_i = observed − projected (px) over the
 * inlier positions p_i (px from the principal point) with an affine model
 *
 *     d = a + B p
 *
 * and reads the model off: `a` is a common shift (a pose bias, or the frame
 * and the pose out of step), the symmetric part of B is a radial scale
 * (focal length or depth scale off) and a shear (fx ≠ fy, aspect / principal
 * point), the antisymmetric part is a rotation about the principal point
 * (roll bias). What the model does not explain is the random part: LK
 * precision on this device. The per-group mean vectors say whether two
 * halves of the map pull the pose in opposite directions (map split).
 *
 * Nothing here feeds back into tracking: it is read on the HUD only.
 */
export interface ResidualField {
  /** Inliers that entered the fit. */
  count: number;
  /** Common shift of the observations against the projections (px). */
  meanX: number;
  meanY: number;
  /** Radial scale: fraction of the distance from the principal point (positive = observations farther out than projections). */
  scale: number;
  /** The same scale as a displacement at half the image diagonal (px). */
  scaleEdgePx: number;
  /** Rotation about the principal point (degrees, image axes: positive = observations rotated clockwise on screen). */
  rotationDeg: number;
  /** Anisotropic part: (B11 − B22) / 2 and (B12 + B21) / 2, fractions. */
  shearX: number;
  shearY: number;
  /** RMS of the raw residual vectors (px). */
  rmsPx: number;
  /** RMS left after the affine model is removed (px): the random part. */
  randomPx: number;
  /** Share of the residual energy the affine model explains, 0–1. */
  systematic: number;
  /** Mean residual vectors of the landmark groups (px), count 0 when absent. */
  adjustedCount: number;
  adjustedX: number;
  adjustedY: number;
  untouchedCount: number;
  untouchedX: number;
  untouchedY: number;
  planeCount: number;
  planeX: number;
  planeY: number;
  otherCount: number;
  otherX: number;
  otherY: number;
}

export function emptyResidualField(): ResidualField {
  return {
    count: 0,
    meanX: 0,
    meanY: 0,
    scale: 0,
    scaleEdgePx: 0,
    rotationDeg: 0,
    shearX: 0,
    shearY: 0,
    rmsPx: 0,
    randomPx: 0,
    systematic: 0,
    adjustedCount: 0,
    adjustedX: 0,
    adjustedY: 0,
    untouchedCount: 0,
    untouchedX: 0,
    untouchedY: 0,
    planeCount: 0,
    planeX: 0,
    planeY: 0,
    otherCount: 0,
    otherX: 0,
    otherY: 0,
  };
}

/** Group flags of one inlier for the per-group means. */
export const RESIDUAL_GROUP_ADJUSTED = 1;
export const RESIDUAL_GROUP_PLANE = 2;

/**
 * Fit the affine residual model to `n` inliers. `px/py` are positions
 * relative to the principal point (px), `dx/dy` the residuals observed −
 * projected (px), `groups` the flags above per inlier.
 */
export function fitResidualField(
  px: ArrayLike<number>,
  py: ArrayLike<number>,
  dx: ArrayLike<number>,
  dy: ArrayLike<number>,
  groups: ArrayLike<number>,
  n: number,
  /** Half the image diagonal (px), for `scaleEdgePx`. */
  halfDiagonalPx = 0,
): ResidualField {
  const out = emptyResidualField();
  out.count = n;
  if (n === 0) return out;
  // Group means and the raw RMS.
  let ss = 0;
  for (let i = 0; i < n; i++) {
    const x = dx[i], y = dy[i];
    ss += x * x + y * y;
    out.meanX += x;
    out.meanY += y;
    if (groups[i] & RESIDUAL_GROUP_ADJUSTED) {
      out.adjustedCount++;
      out.adjustedX += x;
      out.adjustedY += y;
    } else {
      out.untouchedCount++;
      out.untouchedX += x;
      out.untouchedY += y;
    }
    if (groups[i] & RESIDUAL_GROUP_PLANE) {
      out.planeCount++;
      out.planeX += x;
      out.planeY += y;
    } else {
      out.otherCount++;
      out.otherX += x;
      out.otherY += y;
    }
  }
  out.meanX /= n;
  out.meanY /= n;
  if (out.adjustedCount) {
    out.adjustedX /= out.adjustedCount;
    out.adjustedY /= out.adjustedCount;
  }
  if (out.untouchedCount) {
    out.untouchedX /= out.untouchedCount;
    out.untouchedY /= out.untouchedCount;
  }
  if (out.planeCount) {
    out.planeX /= out.planeCount;
    out.planeY /= out.planeCount;
  }
  if (out.otherCount) {
    out.otherX /= out.otherCount;
    out.otherY /= out.otherCount;
  }
  out.rmsPx = Math.sqrt(ss / n);
  if (n < 6) {
    // Too few for six parameters: report the shift only.
    out.randomPx = out.rmsPx;
    return out;
  }
  // Normal equations of d = a + B p, one 3-parameter system per component
  // with the shared design row [1, px, py].
  let s1 = n, sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
  const bx = [0, 0, 0], by = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const x = px[i], y = py[i];
    sx += x;
    sy += y;
    sxx += x * x;
    sxy += x * y;
    syy += y * y;
    bx[0] += dx[i];
    bx[1] += dx[i] * x;
    bx[2] += dx[i] * y;
    by[0] += dy[i];
    by[1] += dy[i] * x;
    by[2] += dy[i] * y;
  }
  const m = [s1, sx, sy, sx, sxx, sxy, sy, sxy, syy];
  const inv = invert3(m);
  if (!inv) {
    out.randomPx = out.rmsPx;
    return out;
  }
  const cx = mul3(inv, bx), cy = mul3(inv, by);
  // d_x = cx0 + cx1 px + cx2 py, d_y = cy0 + cy1 px + cy2 py.
  const b11 = cx[1], b12 = cx[2], b21 = cy[1], b22 = cy[2];
  out.scale = (b11 + b22) / 2;
  out.scaleEdgePx = out.scale * halfDiagonalPx;
  const omega = (b21 - b12) / 2;
  out.rotationDeg = (Math.atan(omega) * 180) / Math.PI;
  out.shearX = (b11 - b22) / 2;
  out.shearY = (b12 + b21) / 2;
  let sr = 0;
  for (let i = 0; i < n; i++) {
    const ex = dx[i] - (cx[0] + b11 * px[i] + b12 * py[i]);
    const ey = dy[i] - (cy[0] + b21 * px[i] + b22 * py[i]);
    sr += ex * ex + ey * ey;
  }
  out.randomPx = Math.sqrt(sr / n);
  out.systematic = ss > 0 ? Math.max(0, 1 - sr / ss) : 0;
  return out;
}

function invert3(m: number[]): number[] | null {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (!(Math.abs(det) > 1e-12)) return null;
  const inv = 1 / det;
  return [
    A * inv, -(b * i - c * h) * inv, (b * f - c * e) * inv,
    B * inv, (a * i - c * g) * inv, -(a * f - c * d) * inv,
    C * inv, -(a * h - b * g) * inv, (a * e - b * d) * inv,
  ];
}

function mul3(m: number[], v: number[]): number[] {
  return [m[0] * v[0] + m[1] * v[1] + m[2] * v[2], m[3] * v[0] + m[4] * v[1] + m[5] * v[2], m[6] * v[0] + m[7] * v[1] + m[8] * v[2]];
}
