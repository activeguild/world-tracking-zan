/**
 * Camera intrinsics (spec §8).
 *
 * Phase 1 does not use intrinsics for any computation, but the type and its
 * approximation are defined now so that `VisionInput` already carries them and
 * later phases (Essential Matrix, triangulation) do not need protocol changes.
 */
export interface CameraIntrinsics {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  width: number;
  height: number;
}

/**
 * Default horizontal field of view along the image's long side, degrees.
 * Smartphone rear cameras in 16:9 video mode cover roughly 63–70° along
 * the long side (26–28 mm equivalent); 66° is a middle-of-the-road guess.
 * Spec §8's "fx ≈ width" corresponds to ~53°, too narrow for current phones,
 * and a wrong focal length distorts the map and makes objects drift.
 */
export const DEFAULT_LONG_SIDE_FOV_DEG = 66;

/**
 * Approximate pinhole intrinsics for a typical smartphone rear camera from
 * the field of view along the long image side; principal point at the
 * center. Pass a device-specific FOV when known.
 */
export function approximateIntrinsics(
  width: number,
  height: number,
  longSideFovDeg = DEFAULT_LONG_SIDE_FOV_DEG,
): CameraIntrinsics {
  const long = Math.max(width, height);
  const f = long / 2 / Math.tan((longSideFovDeg * Math.PI) / 360);
  return {
    fx: f,
    fy: f,
    cx: width / 2,
    cy: height / 2,
    width,
    height,
  };
}

/** Rescale intrinsics when the image is resized (e.g. 1280×720 → 640×360). */
export function scaleIntrinsics(k: CameraIntrinsics, newWidth: number, newHeight: number): CameraIntrinsics {
  const sx = newWidth / k.width;
  const sy = newHeight / k.height;
  return {
    fx: k.fx * sx,
    fy: k.fy * sy,
    cx: k.cx * sx,
    cy: k.cy * sy,
    width: newWidth,
    height: newHeight,
  };
}
