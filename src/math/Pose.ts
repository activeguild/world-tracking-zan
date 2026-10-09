import { det3, svd3, transpose3 } from "./Decomposition";
import { type Mat3, mat3Identity, mat3Multiply } from "./Matrix";

/**
 * Rigid camera pose utilities in the computer-vision camera frame
 * (X right, Y down, Z forward). Conversion to Three.js happens only in
 * CoordinateSystem.ts (Phase 4).
 *
 * Convention: a pose (R, t) maps points from frame A to frame B:
 *   X_B = R · X_A + t
 */
export interface RigidTransform {
  /** 3×3 rotation, row-major. */
  rotation: Mat3;
  /** Translation (unit length when the scale is unknown). */
  translation: Float64Array;
}

/** Quaternion as [x, y, z, w]. */
export type Quaternion = Float64Array;

export function identityTransform(): RigidTransform {
  return { rotation: mat3Identity(), translation: new Float64Array(3) };
}

/** Compose: (b ∘ a)(X) = b(a(X)). */
export function composeTransforms(b: RigidTransform, a: RigidTransform): RigidTransform {
  const rotation = mat3Multiply(b.rotation, a.rotation);
  const r = b.rotation;
  const t = a.translation;
  const translation = new Float64Array([
    r[0] * t[0] + r[1] * t[1] + r[2] * t[2] + b.translation[0],
    r[3] * t[0] + r[4] * t[1] + r[5] * t[2] + b.translation[1],
    r[6] * t[0] + r[7] * t[1] + r[8] * t[2] + b.translation[2],
  ]);
  return { rotation, translation };
}

/** out = R p + t. */
export function applyTransform(T: RigidTransform, p: ArrayLike<number>, out = new Float64Array(3)): Float64Array {
  const r = T.rotation, t = T.translation;
  const x = p[0], y = p[1], z = p[2];
  out[0] = r[0] * x + r[1] * y + r[2] * z + t[0];
  out[1] = r[3] * x + r[4] * y + r[5] * z + t[1];
  out[2] = r[6] * x + r[7] * y + r[8] * z + t[2];
  return out;
}

export function invertTransform(p: RigidTransform): RigidTransform {
  const rt = transpose3(p.rotation);
  const t = p.translation;
  return {
    rotation: rt,
    translation: new Float64Array([
      -(rt[0] * t[0] + rt[1] * t[1] + rt[2] * t[2]),
      -(rt[3] * t[0] + rt[4] * t[1] + rt[5] * t[2]),
      -(rt[6] * t[0] + rt[7] * t[1] + rt[8] * t[2]),
    ]),
  };
}

/** Rotation angle in radians of a rotation matrix. */
export function rotationAngle(r: Mat3): number {
  const tr = r[0] + r[4] + r[8];
  return Math.acos(Math.max(-1, Math.min(1, (tr - 1) / 2)));
}

/** Angle between two rotations: angle(Rᵀa · Rb). */
export function rotationDistance(a: Mat3, b: Mat3): number {
  return rotationAngle(mat3Multiply(transpose3(a), b));
}

/** Rotation about the X axis (right-handed). */
export function rotationX(rad: number): Mat3 {
  const c = Math.cos(rad);
  const s = Math.sin(rad);
  return new Float64Array([1, 0, 0, 0, c, -s, 0, s, c]);
}

export function rotationY(rad: number): Mat3 {
  const c = Math.cos(rad);
  const s = Math.sin(rad);
  return new Float64Array([c, 0, s, 0, 1, 0, -s, 0, c]);
}

export function rotationZ(rad: number): Mat3 {
  const c = Math.cos(rad);
  const s = Math.sin(rad);
  return new Float64Array([c, -s, 0, s, c, 0, 0, 0, 1]);
}

/** Rodrigues: rotation about a unit axis. */
export function rotationAxisAngle(axis: ArrayLike<number>, rad: number): Mat3 {
  const l = Math.hypot(axis[0], axis[1], axis[2]) || 1;
  const x = axis[0] / l;
  const y = axis[1] / l;
  const z = axis[2] / l;
  const c = Math.cos(rad);
  const s = Math.sin(rad);
  const t = 1 - c;
  return new Float64Array([
    t * x * x + c, t * x * y - s * z, t * x * z + s * y,
    t * x * y + s * z, t * y * y + c, t * y * z - s * x,
    t * x * z - s * y, t * y * z + s * x, t * z * z + c,
  ]);
}

/** Closest rotation matrix (Frobenius) to an arbitrary 3×3 matrix. */
export function projectToRotation(m: Mat3): Mat3 {
  const { u, v } = svd3(m);
  let r = mat3Multiply(u, transpose3(v));
  if (det3(r) < 0) {
    // Flip the last column of U.
    const u2 = Float64Array.from(u);
    u2[2] = -u2[2];
    u2[5] = -u2[5];
    u2[8] = -u2[8];
    r = mat3Multiply(u2, transpose3(v));
  }
  return r;
}

export function rotationToQuaternion(m: Mat3): Quaternion {
  const q = new Float64Array(4);
  const tr = m[0] + m[4] + m[8];
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    q[3] = 0.25 * s;
    q[0] = (m[7] - m[5]) / s;
    q[1] = (m[2] - m[6]) / s;
    q[2] = (m[3] - m[1]) / s;
  } else if (m[0] > m[4] && m[0] > m[8]) {
    const s = Math.sqrt(1 + m[0] - m[4] - m[8]) * 2;
    q[3] = (m[7] - m[5]) / s;
    q[0] = 0.25 * s;
    q[1] = (m[1] + m[3]) / s;
    q[2] = (m[2] + m[6]) / s;
  } else if (m[4] > m[8]) {
    const s = Math.sqrt(1 + m[4] - m[0] - m[8]) * 2;
    q[3] = (m[2] - m[6]) / s;
    q[0] = (m[1] + m[3]) / s;
    q[1] = 0.25 * s;
    q[2] = (m[5] + m[7]) / s;
  } else {
    const s = Math.sqrt(1 + m[8] - m[0] - m[4]) * 2;
    q[3] = (m[3] - m[1]) / s;
    q[0] = (m[2] + m[6]) / s;
    q[1] = (m[5] + m[7]) / s;
    q[2] = 0.25 * s;
  }
  return q;
}

/**
 * Euler angles (degrees) of a camera-frame rotation for the debug HUD:
 * yaw about Y, pitch about X, roll about Z (ZYX order: R = Ry · Rx · Rz? no —
 * we use R = Ry(yaw) · Rx(pitch) · Rz(roll)).
 */
export function rotationToEulerDeg(m: Mat3): { yaw: number; pitch: number; roll: number } {
  // R = Ry · Rx · Rz
  const pitch = Math.asin(Math.max(-1, Math.min(1, -m[5])));
  let yaw: number;
  let roll: number;
  if (Math.abs(Math.cos(pitch)) > 1e-6) {
    yaw = Math.atan2(m[2], m[8]);
    roll = Math.atan2(m[3], m[4]);
  } else {
    yaw = Math.atan2(-m[6], m[0]);
    roll = 0;
  }
  const d = 180 / Math.PI;
  return { yaw: yaw * d, pitch: pitch * d, roll: roll * d };
}

export function normalize3(v: ArrayLike<number>, out = new Float64Array(3)): Float64Array {
  const l = Math.hypot(v[0], v[1], v[2]);
  if (l < 1e-12) {
    out[0] = out[1] = out[2] = 0;
    return out;
  }
  out[0] = v[0] / l;
  out[1] = v[1] / l;
  out[2] = v[2] / l;
  return out;
}

export function dot3(a: ArrayLike<number>, b: ArrayLike<number>): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/** Angle in radians between two direction vectors (sign-insensitive when `axial`). */
export function angleBetween(a: ArrayLike<number>, b: ArrayLike<number>, axial = false): number {
  const la = Math.hypot(a[0], a[1], a[2]);
  const lb = Math.hypot(b[0], b[1], b[2]);
  if (la < 1e-12 || lb < 1e-12) return Math.PI;
  let c = dot3(a, b) / (la * lb);
  if (axial) c = Math.abs(c);
  return Math.acos(Math.max(-1, Math.min(1, c)));
}
