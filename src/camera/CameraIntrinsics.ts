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
 * Approximate pinhole intrinsics for a typical smartphone rear camera.
 *
 * Spec initial approximation: fx ≈ fy ≈ width, principal point at the center.
 * `focalScale` lets device-specific calibration adjust the focal length
 * without touching call sites.
 */
export function approximateIntrinsics(
  width: number,
  height: number,
  focalScale = 1.0,
): CameraIntrinsics {
  const f = Math.max(width, height) * focalScale;
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
