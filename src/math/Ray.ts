/**
 * Rays and ray/plane intersection (spec §27–§28).
 *
 *   plane:  n · p + d = 0
 *   ray:    p = o + t r
 *   →       t = −(n · o + d) / (n · r),   no hit when n · r ≈ 0 or t < 0
 */
export interface Ray {
  origin: Float64Array;
  /** Unit direction. */
  direction: Float64Array;
}

export interface RayPlaneHit {
  point: Float64Array;
  /** Distance along the ray. */
  t: number;
}

export function intersectRayPlane(
  ray: Ray,
  normal: ArrayLike<number>,
  d: number,
  epsilon = 1e-6,
): RayPlaneHit | null {
  const o = ray.origin;
  const r = ray.direction;
  const denom = normal[0] * r[0] + normal[1] * r[1] + normal[2] * r[2];
  if (Math.abs(denom) < epsilon) return null;
  const t = -(normal[0] * o[0] + normal[1] * o[1] + normal[2] * o[2] + d) / denom;
  if (t < 0) return null;
  return {
    point: new Float64Array([o[0] + r[0] * t, o[1] + r[1] * t, o[2] + r[2] * t]),
    t,
  };
}

/**
 * Camera ray through a pixel, in the camera's own frame (CV convention:
 * X right, Y down, Z forward). `fx, fy, cx, cy` are the intrinsics in the
 * same pixel units as (u, v).
 */
export function pixelRay(u: number, v: number, fx: number, fy: number, cx: number, cy: number): Ray {
  const x = (u - cx) / fx;
  const y = (v - cy) / fy;
  const len = Math.hypot(x, y, 1);
  return {
    origin: new Float64Array(3),
    direction: new Float64Array([x / len, y / len, 1 / len]),
  };
}

/** Transform a ray by X' = R X + t. */
export function transformRay(ray: Ray, r: ArrayLike<number>, t: ArrayLike<number>): Ray {
  const o = ray.origin;
  const dir = ray.direction;
  return {
    origin: new Float64Array([
      r[0] * o[0] + r[1] * o[1] + r[2] * o[2] + t[0],
      r[3] * o[0] + r[4] * o[1] + r[5] * o[2] + t[1],
      r[6] * o[0] + r[7] * o[1] + r[8] * o[2] + t[2],
    ]),
    direction: new Float64Array([
      r[0] * dir[0] + r[1] * dir[1] + r[2] * dir[2],
      r[3] * dir[0] + r[4] * dir[1] + r[5] * dir[2],
      r[6] * dir[0] + r[7] * dir[1] + r[8] * dir[2],
    ]),
  };
}
