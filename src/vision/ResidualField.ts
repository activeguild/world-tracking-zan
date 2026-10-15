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
  /**
   * v23: the random part split along / across each landmark's parallax
   * direction (where its projection moves when its depth changes, i.e. the
   * epipolar direction from the view that fixed the depth). Isotropic noise
   * gives `parallaxRatio` ≈ 1; depth errors put the residual along the
   * parallax direction, ratio ≫ 1. Over the landmarks with enough depth
   * sensitivity (≥ `PARALLAX_MIN_SENSITIVITY_PX`): `depthErrorMedian` =
   * median |implied relative depth error|, `depthErrorBias` = its signed mean
   * (positive = the map puts the landmark too close), per group as well.
   */
  parallelRmsPx: number;
  perpRmsPx: number;
  parallaxRatio: number;
  /** Landmarks with a usable parallax direction (sensitivity ≥ the minimum). */
  parallaxCount: number;
  /** Median projection sensitivity to a 100 % depth change (px). */
  depthSensitivityMedianPx: number;
  depthErrorMedian: number;
  depthErrorBias: number;
  planeDepthBias: number;
  otherDepthBias: number;
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
    parallelRmsPx: 0,
    perpRmsPx: 0,
    parallaxRatio: 0,
    parallaxCount: 0,
    depthSensitivityMedianPx: 0,
    depthErrorMedian: 0,
    depthErrorBias: 0,
    planeDepthBias: 0,
    otherDepthBias: 0,
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
 * Minimum depth sensitivity (px per 100 % depth change) for a landmark to
 * enter the depth-error statistics: below it the current view is too close
 * to the view that fixed the depth and the direction says nothing.
 */
export const PARALLAX_MIN_SENSITIVITY_PX = 5;

/** Optional per-inlier parallax geometry for the v23 split. */
export interface ParallaxInput {
  /** Unit parallax direction per inlier (px space), zero when undefined. */
  ex: ArrayLike<number>;
  ey: ArrayLike<number>;
  /** Projection displacement for a 100 % depth change (px). */
  sensitivity: ArrayLike<number>;
}

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
  parallax: ParallaxInput | null = null,
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
    if (parallax) splitParallax(out, dx, dy, groups, n, parallax, null);
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
  const rx = new Float64Array(n), ry = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    rx[i] = dx[i] - (cx[0] + b11 * px[i] + b12 * py[i]);
    ry[i] = dy[i] - (cy[0] + b21 * px[i] + b22 * py[i]);
    sr += rx[i] * rx[i] + ry[i] * ry[i];
  }
  out.randomPx = Math.sqrt(sr / n);
  out.systematic = ss > 0 ? Math.max(0, 1 - sr / ss) : 0;
  // v23: the random part along / across the parallax directions.
  if (parallax) splitParallax(out, rx, ry, groups, n, parallax, null);
  return out;
}

/**
 * Split the (affine-removed) residuals into the component along each
 * landmark's parallax direction and the one across it, and read the implied
 * relative depth error `(r · e) / sensitivity` off the landmarks whose
 * sensitivity is large enough.
 */
function splitParallax(out: ResidualField, rx: ArrayLike<number>, ry: ArrayLike<number>, groups: ArrayLike<number>, n: number, p: ParallaxInput, _unused: null): void {
  let sPar = 0, sPerp = 0, m = 0;
  const errs: number[] = [];
  const sens: number[] = [];
  let bias = 0, planeBias = 0, planeN = 0, otherBias = 0, otherN = 0;
  for (let i = 0; i < n; i++) {
    const s = p.sensitivity[i];
    if (!(s >= PARALLAX_MIN_SENSITIVITY_PX)) continue;
    const ex = p.ex[i], ey = p.ey[i];
    const along = rx[i] * ex + ry[i] * ey;
    const across = -rx[i] * ey + ry[i] * ex;
    sPar += along * along;
    sPerp += across * across;
    const depthErr = along / s;
    errs.push(Math.abs(depthErr));
    sens.push(s);
    bias += depthErr;
    if (groups[i] & RESIDUAL_GROUP_PLANE) {
      planeBias += depthErr;
      planeN++;
    } else {
      otherBias += depthErr;
      otherN++;
    }
    m++;
  }
  out.parallaxCount = m;
  if (m === 0) return;
  out.parallelRmsPx = Math.sqrt(sPar / m);
  out.perpRmsPx = Math.sqrt(sPerp / m);
  out.parallaxRatio = out.perpRmsPx > 1e-9 ? out.parallelRmsPx / out.perpRmsPx : 0;
  errs.sort((a, b) => a - b);
  sens.sort((a, b) => a - b);
  out.depthErrorMedian = errs[m >> 1];
  out.depthSensitivityMedianPx = sens[m >> 1];
  out.depthErrorBias = bias / m;
  out.planeDepthBias = planeN ? planeBias / planeN : 0;
  out.otherDepthBias = otherN ? otherBias / otherN : 0;
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
