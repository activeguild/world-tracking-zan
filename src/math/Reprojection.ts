import type { Mat3 } from "./Matrix";

/**
 * Pinhole reprojection in normalized camera coordinates and its Jacobian,
 * shared by the motion-only PnP and the bundle adjustment so that both use
 * one derivation (改善指示書 §4.1).
 *
 * Convention:  X_cam = R · X + t,   π(X_cam) = (x/z, y/z).
 *
 * Update applied by the solvers:  R ← exp(δω) R,  t ← t + δt,  X ← X + δX.
 * Under that update the rotation acts on R X only, so
 *
 *   ∂X_cam/∂δω = −[R X]×,   ∂X_cam/∂δt = I,   ∂X_cam/∂X = R,
 *   ∂π/∂X_cam  = [[1/z, 0, −x/z²], [0, 1/z, −y/z²]].
 *
 * (−[X_cam]× would be the derivative of the SE(3) left perturbation, which
 * also rotates t; it coincides with the above only when t = 0. The engine
 * used it before and the mismatch δω × t grew with the camera's distance
 * from the map origin.)
 */

/** out = (x/z, y/z); false when the point is at or behind the camera plane. */
export function projectNormalized(r: Mat3, t: ArrayLike<number>, X: number, Y: number, Z: number, out: Float64Array): boolean {
  const z = r[6] * X + r[7] * Y + r[8] * Z + t[2];
  if (z <= 1e-6) return false;
  out[0] = (r[0] * X + r[1] * Y + r[2] * Z + t[0]) / z;
  out[1] = (r[3] * X + r[4] * Y + r[5] * Z + t[1]) / z;
  return true;
}

/**
 * Jacobian of (u, v) = π(R X + t) w.r.t. [δω (3), δt (3), δX (3)].
 * `ju` / `jv` receive the 9 partial derivatives of u and v. Returns false
 * (and leaves the outputs untouched) when the point is behind the camera.
 */
export function reprojectionJacobian(r: Mat3, t: ArrayLike<number>, X: number, Y: number, Z: number, ju: Float64Array, jv: Float64Array): boolean {
  // p = R X (the part the rotation update acts on), X_cam = p + t.
  const px = r[0] * X + r[1] * Y + r[2] * Z;
  const py = r[3] * X + r[4] * Y + r[5] * Z;
  const pz = r[6] * X + r[7] * Y + r[8] * Z;
  const zc = pz + t[2];
  if (zc <= 1e-6) return false;
  const iz = 1 / zc;
  const u = (px + t[0]) * iz;
  const v = (py + t[1]) * iz;
  // Rotation: [iz, 0, −u iz] · (−[p]×) and [0, iz, −v iz] · (−[p]×),
  //   −[p]× = [[0, pz, −py], [−pz, 0, px], [py, −px, 0]].
  ju[0] = -u * iz * py;
  ju[1] = iz * pz + u * iz * px;
  ju[2] = -iz * py;
  jv[0] = -iz * pz - v * iz * py;
  jv[1] = v * iz * px;
  jv[2] = iz * px;
  // Translation.
  ju[3] = iz; ju[4] = 0; ju[5] = -u * iz;
  jv[3] = 0; jv[4] = iz; jv[5] = -v * iz;
  // Landmark: [iz, 0, −u iz] · R and [0, iz, −v iz] · R.
  ju[6] = iz * r[0] - u * iz * r[6]; ju[7] = iz * r[1] - u * iz * r[7]; ju[8] = iz * r[2] - u * iz * r[8];
  jv[6] = iz * r[3] - v * iz * r[6]; jv[7] = iz * r[4] - v * iz * r[7]; jv[8] = iz * r[5] - v * iz * r[8];
  return true;
}
