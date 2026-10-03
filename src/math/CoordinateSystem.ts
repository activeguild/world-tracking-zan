import { cross3, transpose3 } from "./Decomposition";
import { type Mat3, mat3Multiply } from "./Matrix";
import { normalize3, type RigidTransform, rotationToQuaternion } from "./Pose";
import type { CameraIntrinsics } from "../camera/CameraIntrinsics";

/**
 * The single place where frames are converted (spec §6–§7, §25–§26):
 *
 *   map frame    CV camera frame of the frame the landmark map was
 *                initialized in (X right, Y down, Z forward), scale-free.
 *   world frame  Three.js convention, Y up. Origin = center of the first
 *                found plane, +Y = plane normal (toward the camera), −Z ≈
 *                the camera's viewing direction projected onto the plane.
 *   CV camera    X right, Y down, Z forward (what the vision engine outputs)
 *   Three camera X right, Y up, looks along −Z
 *
 * Nothing outside this file is allowed to know these sign conventions.
 */

/** Plane coordinate system (spec §26). All vectors in the map frame. */
export interface PlaneCoordinateSystem {
  origin: Float64Array;
  right: Float64Array;
  up: Float64Array;
  forward: Float64Array;
  /** 4×4 column-major (Three.js layout) matrix: world → map. */
  matrix: Float64Array;
}

/** World frame definition derived from a plane in the map frame. */
export interface WorldFrame {
  plane: PlaneCoordinateSystem;
  /** X_world = R_wm · (X_map − origin) · scale */
  rotation: Mat3;
  /** Map units → world meters. */
  scale: number;
}

/**
 * Build the world frame from a plane (normal oriented toward the camera,
 * n·X + d = 0, center on the plane) and the map-frame camera pose used to
 * choose the horizontal "forward" direction.
 *
 * @param scale map-unit → meter factor
 */
export function worldFromPlane(
  normal: ArrayLike<number>,
  center: ArrayLike<number>,
  cameraPose: RigidTransform,
  scale: number,
): WorldFrame {
  const up = normalize3(normal);
  // Camera forward (map frame) = Rᵀ · (0,0,1) = third row of R.
  const r = cameraPose.rotation;
  const camFwd = [r[6], r[7], r[8]];
  // Project onto the plane.
  const dot = camFwd[0] * up[0] + camFwd[1] * up[1] + camFwd[2] * up[2];
  let fwdMap = [camFwd[0] - dot * up[0], camFwd[1] - dot * up[1], camFwd[2] - dot * up[2]];
  if (Math.hypot(fwdMap[0], fwdMap[1], fwdMap[2]) < 1e-6) {
    // Looking straight down: use the camera's "up" (−Y) projected instead.
    const camUp = [-r[3], -r[4], -r[5]];
    const d2 = camUp[0] * up[0] + camUp[1] * up[1] + camUp[2] * up[2];
    fwdMap = [camUp[0] - d2 * up[0], camUp[1] - d2 * up[1], camUp[2] - d2 * up[2]];
  }
  const fwd = normalize3(fwdMap); // world −Z direction
  // Right-handed: X = Y × Z with Z = −fwd  →  X = fwd × Y... check: Y × (−fwd) = fwd × Y? No: Y × (−fwd) = −(Y × fwd) = fwd × Y. ✓
  const right = normalize3(cross3(fwd, up));
  const zAxis = new Float64Array([-fwd[0], -fwd[1], -fwd[2]]);

  // R_wm rows = world axes in map coordinates.
  const rotation = new Float64Array([
    right[0], right[1], right[2],
    up[0], up[1], up[2],
    zAxis[0], zAxis[1], zAxis[2],
  ]);
  const origin = new Float64Array([center[0], center[1], center[2]]);
  // world → map 4×4 (column-major): columns = right, up, zAxis (scaled), origin
  const matrix = new Float64Array([
    right[0] / scale, right[1] / scale, right[2] / scale, 0,
    up[0] / scale, up[1] / scale, up[2] / scale, 0,
    zAxis[0] / scale, zAxis[1] / scale, zAxis[2] / scale, 0,
    origin[0], origin[1], origin[2], 1,
  ]);
  return {
    plane: { origin, right, up, forward: fwd, matrix },
    rotation,
    scale,
  };
}

/** Map → world point. */
export function mapToWorld(w: WorldFrame, p: ArrayLike<number>, outArr?: Float64Array): Float64Array {
  const out = outArr ?? new Float64Array(3);
  const o = w.plane.origin;
  const r = w.rotation;
  const x = p[0] - o[0];
  const y = p[1] - o[1];
  const z = p[2] - o[2];
  out[0] = (r[0] * x + r[1] * y + r[2] * z) * w.scale;
  out[1] = (r[3] * x + r[4] * y + r[5] * z) * w.scale;
  out[2] = (r[6] * x + r[7] * y + r[8] * z) * w.scale;
  return out;
}

/** World → map point. */
export function worldToMap(w: WorldFrame, p: ArrayLike<number>, outArr?: Float64Array): Float64Array {
  const out = outArr ?? new Float64Array(3);
  const o = w.plane.origin;
  const r = w.rotation;
  const x = p[0] / w.scale;
  const y = p[1] / w.scale;
  const z = p[2] / w.scale;
  // Rᵀ · p
  out[0] = r[0] * x + r[3] * y + r[6] * z + o[0];
  out[1] = r[1] * x + r[4] * y + r[7] * z + o[1];
  out[2] = r[2] * x + r[5] * y + r[8] * z + o[2];
  return out;
}

/** Map-frame direction → world direction (no translation / scale). */
export function mapDirToWorld(w: WorldFrame, v: ArrayLike<number>, outArr?: Float64Array): Float64Array {
  const out = outArr ?? new Float64Array(3);
  const r = w.rotation;
  out[0] = r[0] * v[0] + r[1] * v[1] + r[2] * v[2];
  out[1] = r[3] * v[0] + r[4] * v[1] + r[5] * v[2];
  out[2] = r[6] * v[0] + r[7] * v[1] + r[8] * v[2];
  return out;
}

/** Three.js camera pose in the world frame. */
export interface ThreeCameraPose {
  position: Float64Array;
  /** [x, y, z, w] */
  quaternion: Float64Array;
}

/**
 * Convert the CV camera pose in the map frame (X_cam = R X_map + t) into a
 * Three.js camera pose in the world frame.
 */
export function cameraPoseToThree(w: WorldFrame, cam: RigidTransform): ThreeCameraPose {
  const r = cam.rotation;
  const t = cam.translation;
  // Camera center in map: C = −Rᵀ t
  const cMap = [
    -(r[0] * t[0] + r[3] * t[1] + r[6] * t[2]),
    -(r[1] * t[0] + r[4] * t[1] + r[7] * t[2]),
    -(r[2] * t[0] + r[5] * t[1] + r[8] * t[2]),
  ];
  const position = mapToWorld(w, cMap);
  // CV camera axes in map coordinates = columns of Rᵀ = rows of R.
  const xCv = [r[0], r[1], r[2]];
  const yCv = [r[3], r[4], r[5]];
  const zCv = [r[6], r[7], r[8]];
  // Three.js camera axes: X = X_cv, Y = −Y_cv, Z = −Z_cv (in map), then to world.
  const xW = mapDirToWorld(w, xCv);
  const yW = mapDirToWorld(w, [-yCv[0], -yCv[1], -yCv[2]]);
  const zW = mapDirToWorld(w, [-zCv[0], -zCv[1], -zCv[2]]);
  // Rotation matrix with these as columns (world ← three-camera).
  const m: Mat3 = new Float64Array([
    xW[0], yW[0], zW[0],
    xW[1], yW[1], zW[1],
    xW[2], yW[2], zW[2],
  ]);
  return { position, quaternion: rotationToQuaternion(m) };
}

/**
 * Three.js camera pose (world) → CV camera pose in the map frame.
 * Inverse of `cameraPoseToThree`; used by tests and by hit-testing when
 * only the rendered camera is known.
 */
export function threePoseToCamera(w: WorldFrame, position: ArrayLike<number>, quaternion: ArrayLike<number>): RigidTransform {
  const [x, y, z, q] = [quaternion[0], quaternion[1], quaternion[2], quaternion[3]];
  // world ← three-camera rotation from the quaternion
  const m = new Float64Array([
    1 - 2 * (y * y + z * z), 2 * (x * y - z * q), 2 * (x * z + y * q),
    2 * (x * y + z * q), 1 - 2 * (x * x + z * z), 2 * (y * z - x * q),
    2 * (x * z - y * q), 2 * (y * z + x * q), 1 - 2 * (x * x + y * y),
  ]);
  // Columns are three-camera axes in world. Convert to map and flip Y, Z to CV.
  const rwT = transpose3(w.rotation); // world → map direction = Rᵀ_wm
  const axesMap = mat3Multiply(rwT, m); // columns = three-camera axes in map
  // CV axes in map: X = col0, Y = −col1, Z = −col2  → rows of R (cam ← map)
  const rotation: Mat3 = new Float64Array([
    axesMap[0], axesMap[3], axesMap[6],
    -axesMap[1], -axesMap[4], -axesMap[7],
    -axesMap[2], -axesMap[5], -axesMap[8],
  ]);
  const cMap = worldToMap(w, position);
  // t = −R C
  const translation = new Float64Array([
    -(rotation[0] * cMap[0] + rotation[1] * cMap[1] + rotation[2] * cMap[2]),
    -(rotation[3] * cMap[0] + rotation[4] * cMap[1] + rotation[5] * cMap[2]),
    -(rotation[6] * cMap[0] + rotation[7] * cMap[1] + rotation[8] * cMap[2]),
  ]);
  return { rotation, translation };
}

/**
 * OpenGL/Three.js projection matrix (column-major 4×4) from pinhole
 * intrinsics expressed in *viewport* pixels (after the object-fit: cover
 * mapping), for a viewport of `width × height`.
 */
export function projectionMatrixFromIntrinsics(
  fx: number,
  fy: number,
  cx: number,
  cy: number,
  width: number,
  height: number,
  near: number,
  far: number,
): Float64Array {
  const m = new Float64Array(16);
  // column-major: m[col*4 + row]
  m[0] = (2 * fx) / width; // row0 col0
  m[5] = (2 * fy) / height; // row1 col1
  m[8] = 1 - (2 * cx) / width; // row0 col2
  m[9] = (2 * cy) / height - 1; // row1 col2
  m[10] = -(far + near) / (far - near); // row2 col2
  m[11] = -1; // row3 col2
  m[14] = (-2 * far * near) / (far - near); // row2 col3
  return m;
}

/**
 * Intrinsics of the processing image rescaled to the displayed viewport
 * (object-fit: cover): display = scale · processing + offset.
 */
export function viewportIntrinsics(
  k: CameraIntrinsics,
  viewportWidth: number,
  viewportHeight: number,
): { fx: number; fy: number; cx: number; cy: number; scale: number; offsetX: number; offsetY: number } {
  const scale = Math.max(viewportWidth / k.width, viewportHeight / k.height);
  const offsetX = (viewportWidth - k.width * scale) / 2;
  const offsetY = (viewportHeight - k.height * scale) / 2;
  return {
    fx: k.fx * scale,
    fy: k.fy * scale,
    cx: k.cx * scale + offsetX,
    cy: k.cy * scale + offsetY,
    scale,
    offsetX,
    offsetY,
  };
}
