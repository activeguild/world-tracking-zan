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
  /**
   * Position in the current reference frame (Phase 2 two-view geometry).
   * Only meaningful when `refFrame` equals the engine's reference frame id;
   * tracks born after the reference was set have refFrame = -1 until the
   * next renewal.
   */
  refX: number;
  refY: number;
  refFrame: number;
  /** Landmark observed by this track (Phase 3), -1 when none. */
  landmarkId: number;
  /**
   * Anchor observation for triangulation: the map-frame camera pose and the
   * pixel position at the frame where the track started being watched by
   * the map tracker. anchorFrame = -1 when not set.
   */
  anchorFrame: number;
  anchorX: number;
  anchorY: number;
  anchorPose: { rotation: Float64Array; translation: Float64Array } | null;
}

/** Detected plane (spec §21, Phase 3). All geometry in the map frame. */
export interface PlaneOutput {
  /** Unit normal, oriented toward the map-frame camera (i.e. "up" for a floor seen from above). */
  normal: number[];
  /** Plane equation n·X + d = 0. */
  d: number;
  /** Centroid of the inlier landmarks. */
  center: number[];
  inlierCount: number;
  /** RMS distance of inliers to the plane (map units). */
  rmsResidual: number;
  /** Rough area covered by the inliers (map units²). */
  areaEstimate: number;
  /** |cos| between the normal and the up direction (1 = horizontal). */
  horizontalness: number;
  horizontal: boolean;
  /** 0–1 (spec §23). */
  confidence: number;
  /** Consecutive frames the same plane has been detected. */
  stableFrames: number;
  /** PLANE_FOUND criterion satisfied (spec §24). */
  found: boolean;
  /** True when the horizontality test used a gravity reading. */
  usedGravity: boolean;
}

/** Camera pose in the map frame with consistent (but arbitrary) scale. */
export interface MapPoseOutput {
  /** X_cam = R · X_map + t (row-major R). */
  rotation: number[];
  translation: number[];
  /** PnP inliers in this frame (0 when the pose was propagated without PnP). */
  inlierCount: number;
  /** Mean PnP reprojection error of inliers (px). */
  meanReprojectionErrorPx: number;
  landmarkCount: number;
  /** Frame id of the map origin. */
  mapFrameId: number;
  /** Frames since the last successful PnP. */
  framesSinceTracked: number;
}

/** Packed landmark layout (Float32): [x, y, z, planeInlier] in the map frame. */
export const LANDMARK_STRIDE = 4;

/** Keyframe / relocalization status (Phase 5). */
/** Plane search diagnostics (why no plane yet). */
export interface PlaneSearchOutput {
  points: number;
  bestInliers: number;
  minInliers: number;
  threshold: number;
  horizontalness: number;
}

export interface RelocalizationOutput {
  keyframes: number;
  /** Result of the attempt made in this frame. */
  attempt: "none" | "success" | "fail";
  inlierCount: number;
  candidatesTried: number;
  /** Frame id of the last successful relocalization (-1 when none). */
  lastSuccessFrame: number;
  /** Total successful relocalizations in this session. */
  successCount: number;
}

/**
 * Camera pose output (Phase 2, spec §17–§18). Scale-free: the translation is
 * a unit direction. All quantities are in the CV camera frame
 * (X right, Y down, Z forward); conversion to Three.js happens in Phase 4.
 */
export interface PoseOutput {
  /**
   * Accumulated rotation R_cur←origin (row-major 3×3): maps directions in
   * the first reference frame of the session into the current camera frame.
   */
  rotation: number[];
  /** Same rotation as a quaternion [x, y, z, w]. */
  quaternion: number[];
  /**
   * Unit translation direction of the current camera relative to the
   * current reference frame, expressed in the current camera frame
   * (X_cur = R·X_ref + t). Zero when not observable.
   */
  translationDirection: number[];
  /** Rotation relative to the current reference frame only (row-major). */
  relativeRotation: number[];
  /** Median parallax (px) between reference and current frame. */
  parallaxPx: number;
  /** Geometric model that produced the estimate. */
  model: "essential" | "homography" | "rotation" | "none";
  /** 0–1 confidence of the estimate. */
  confidence: number;
  /** 0–1 confidence of the translation direction. */
  translationConfidence: number;
  /** Correspondences used (reference ↔ current). */
  correspondences: number;
  /** Inliers of the chosen model. */
  inlierCount: number;
  /** Frame id of the current reference frame. */
  referenceFrameId: number;
  /** Plane normal in the current reference camera frame when known. */
  planeNormal: number[] | null;
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
  /**
   * Optional gravity direction in the camera frame (any scale, sign may be
   * device dependent). Used only for the horizontality test of planes.
   */
  gravity?: number[] | null;
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
  /** Relative camera pose (Phase 2). Null until enough parallax / tracks exist. */
  pose: PoseOutput | null;
  /** Camera pose in the landmark map frame (Phase 3). Null until the map is initialized. */
  mapPose: MapPoseOutput | null;
  /** Current plane candidate (Phase 3). Null when none. */
  plane: PlaneOutput | null;
  /** Plane search diagnostics (null until the map exists). */
  planeSearch: PlaneSearchOutput | null;
  /** Keyframe / relocalization status (Phase 5). */
  relocalization: RelocalizationOutput;
  /** Packed landmarks for debug rendering: see `LANDMARK_STRIDE`. */
  landmarks: Float32Array;
  landmarkCount: number;
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
