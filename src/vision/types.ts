import type { CameraIntrinsics } from "../camera/CameraIntrinsics";
import type { TrackingState } from "../ar/ARState";
import type { TrackingQuality } from "./TrackingQuality";

/** A detected corner (before it becomes a track). */
export interface Corner {
  x: number;
  y: number;
  score: number;
}

/**
 * A feature tracked across frames (Phase 1 unit of state).
 *
 * Positions are in level-0 processing-image pixels.
 */
export interface Track {
  id: number;
  /** Position in the current frame. */
  x: number;
  y: number;
  /** Position in the previous frame (for motion-vector display). */
  prevX: number;
  prevY: number;
  /** Number of consecutive frames the track has survived. */
  age: number;
  /** FAST score at detection time. */
  score: number;
  /** Survived RANSAC in the current frame. New tracks start as inliers. */
  inlier: boolean;
}

/** Input to the vision engine for one frame (spec §40). */
export interface VisionInput {
  frameId: number;
  timestamp: number;
  width: number;
  height: number;
  /** Grayscale pixels, row-major, width*height bytes. */
  gray: Uint8Array;
  intrinsics: CameraIntrinsics;
}

/**
 * Output of the vision engine for one frame (spec §40).
 *
 * Phase 1 produces tracks + quality + state. `pose` / `plane` are added in
 * later phases without changing the shape of the existing fields.
 */
export interface VisionOutput {
  frameId: number;
  timestamp: number;
  state: TrackingState;
  quality: TrackingQuality;
  /** Packed track data for cheap transfer: see `TRACK_STRIDE`. */
  tracks: Float32Array;
  trackCount: number;
  /** Wall-clock milliseconds spent inside the engine for this frame. */
  processingMs: number;
}

/**
 * Packed track layout (Float32):
 *   [id, x, y, prevX, prevY, age, inlier]
 */
export const TRACK_STRIDE = 7;

export function packTracks(tracks: readonly Track[], out?: Float32Array): Float32Array {
  const n = tracks.length;
  const arr = out && out.length >= n * TRACK_STRIDE ? out : new Float32Array(n * TRACK_STRIDE);
  for (let i = 0; i < n; i++) {
    const t = tracks[i];
    const o = i * TRACK_STRIDE;
    arr[o] = t.id;
    arr[o + 1] = t.x;
    arr[o + 2] = t.y;
    arr[o + 3] = t.prevX;
    arr[o + 4] = t.prevY;
    arr[o + 5] = t.age;
    arr[o + 6] = t.inlier ? 1 : 0;
  }
  return arr;
}
