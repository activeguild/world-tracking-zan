import type { CameraIntrinsics } from "../camera/CameraIntrinsics";
import type { TrackingState } from "../ar/ARState";
import type { TrackingQuality } from "./TrackingQuality";
import type { PoseRejection } from "./PoseValidation";
import type { RelocalizationDiagnostics, RelocalizationRejectCode } from "./Relocalizer";

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
  /** Consecutive frames the track was a frame-to-frame RANSAC outlier. */
  outlierStreak: number;
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
  /**
   * Plane-anchored tracking: the 3D point on the anchored plane (map frame)
   * this track is assumed to observe (its pixel ray intersected with the
   * plane). Null until lifted.
   */
  planePoint: Float64Array | null;
  /** Consecutive frames the plane point reprojected within the gate. */
  planeStreak: number;
  /** Consecutive frames it did not. */
  planeOutliers: number;
  /** Decided to be off the plane: excluded from the plane-relative pose. */
  offPlane: boolean;
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

/**
 * Diagnostics of one pose candidate (修正指示書 v4 §8, §25): PnP quality,
 * continuity against the previous canonical pose, and why it was not used.
 */
export interface PoseCandidateReport {
  inlierCount: number;
  reprojectionErrorPx: number;
  /** Camera-center / rotation difference to the previous canonical pose (map units / deg). */
  deltaTranslation: number;
  deltaRotationDeg: number;
  /** PnP quality only ("trusted", v4 §2): it never skips the temporal gate. */
  trusted: boolean;
  /** Structured rejection, null when the candidate was accepted (it may still not be the chosen source). */
  reject: PoseRejection | null;
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
  /** Camera center in the map frame (C = −Rᵀt). */
  cameraCenter: number[];
  /** Camera center displacement since the previous frame (map units, 修正指示書 v2 §7). */
  deltaTranslation: number;
  /** Camera rotation since the previous frame (degrees). */
  deltaRotationDeg: number;
  /** PnP failed this frame and the translation was held (v2 §6). */
  translationHeld: boolean;
  /** PnP failed this frame and the camera center was predicted from the last tracked velocity. */
  translationPredicted: boolean;
  /** PnP found a pose but the jump gate rejected it (v2 §8). */
  jumpRejected: boolean;
  /** Unlinked landmarks re-linked to tracks this frame (v3 §15). */
  reassociated: number;
  /** Inliers of the landmark-PnP candidate / the plane candidate this frame (0 when absent). */
  mapInlierCount: number;
  planeInlierCount: number;
  /** Why a candidate was not adopted this frame (plane first, then map), null when nothing was rejected. */
  rejectReason: string | null;
  /** Per-candidate diagnostics (null when the estimator produced nothing this frame). */
  mapCandidate: PoseCandidateReport | null;
  planeCandidate: PoseCandidateReport | null;
  /** Temporal-gate limits used this frame (map units / deg). */
  gateMaxTranslation: number;
  gateMaxRotationDeg: number;
  /**
   * Map PnP recovery status (v6 §10): landmark observations available, the
   * inliers the candidate needed and which rule set that requirement
   * (tracking: minPnPInliers, recovery: minRecoveryInliers, long: minRecoveryInliersLong).
   */
  observations: number;
  requiredInliers: number;
  recoveryMode: "tracking" | "recovery" | "long";
  /** Map vs plane candidate difference when both existed (map units / degrees). */
  sourceDeltaTranslation: number;
  sourceDeltaRotationDeg: number;
  /** Last ~50 frames' pose sources, newest last: M = map, P = plane, · = propagated, R = relocalized. */
  sourceHistory: string;
  /** The pose was re-seeded by a relocalization in this frame or is still within its monitoring window (v5 §20). */
  relocalized: boolean;
  /** New-landmark triangulation of this frame (v11 diagnostics). */
  triangulation: TriangulationStats;
  /**
   * Where this frame's pose came from: "plane" = plane-relative estimate
   * (depth-free), "map" = PnP on triangulated landmarks, "propagated" = no
   * estimate this frame (previous pose rotated by the frame-to-frame rotation).
   */
  source: "plane" | "map" | "propagated";
}

/** The plane the AR world is anchored to (fixed once; map frame). */
export interface PlaneAnchorOutput {
  normal: number[];
  /** n·X + d = 0 */
  d: number;
  center: number[];
  /** Frame in which the plane was fixed. */
  frameId: number;
}

/** Quality of the plane-relative pose (修正指示書 §9 HomographyQuality). */
export interface PlanePoseOutput {
  /** The plane PnP produced a candidate this frame. */
  tracked: boolean;
  /** The candidate passed validation and the plane bookkeeping was updated with the canonical pose (v4 §12–§13). */
  accepted: boolean;
  inlierCount: number;
  candidateCount: number;
  confirmedCount: number;
  inlierRatio: number;
  reprojectionErrorPx: number;
  confidence: number;
}

/** Packed landmark layout (Float32): [x, y, z, planeInlier] in the map frame. */
export const LANDMARK_STRIDE = 4;

/**
 * Where the per-frame plane search stopped (修正指示書 v11 §21–§22, §37–§38):
 *   points      – fewer landmarks than `plane.minInliers` were offered
 *   support     – the densest height window (gravity) / best RANSAC sample had too few inliers
 *   reclassify  – the window had enough support but re-classifying around its mean plane lost it
 *   extent      – enough inliers, but they do not cover a 2D patch (thin strip / compact cluster)
 *   candidate   – a plane candidate was produced (it may still be non-horizontal or unstable)
 */
export type PlaneSearchStage = "points" | "support" | "reclassify" | "extent" | "candidate";

/** Diagnostics of the last plane search (why no plane yet). */
export interface PlaneSearchInfo {
  /** Landmarks offered to RANSAC. */
  points: number;
  /** Best sample support found (even when below minInliers). */
  bestInliers: number;
  /** Inlier distance threshold used (map units). */
  threshold: number;
  /** Horizontalness of the best plane when one was fitted. */
  horizontalness: number;
  /** Stage the search reached this frame (v11 §22). */
  stage: PlaneSearchStage;
  /** Inliers of the fitted plane after re-classification (0 when none was fitted). */
  inliers: number;
  /** Std. dev. of the inliers along the two in-plane axes and the minimum the second one needs (map units). */
  extentMajor: number;
  extentMinor: number;
  extentRequired: number;
}

export interface PlaneSearchOutput extends PlaneSearchInfo {
  minInliers: number;
}

/**
 * Why a plane recovery was started (v11 §23, v11.1 §10). `fast_motion`
 * (v7 motion level), `two_view_motion` (confident reference↔current parallax
 * crossing) and `manual` are triggered today; `insufficient_plane_points`
 * is reserved.
 */
export type PlaneRecoveryReason = "fast_motion" | "two_view_motion" | "insufficient_plane_points" | "manual" | null;

/**
 * Plane recovery state (v11.1 §27): `starting` in the start frame and while
 * the camera still moves fast, `warmup` while plane points are re-collected
 * at the new view (no candidate in the current frame), `candidate` when the
 * current frame's search produced one, `stable` once its stability streak
 * has begun, `inactive` otherwise.
 */
export type PlaneRecoveryState = "inactive" | "starting" | "warmup" | "candidate" | "stable";

/**
 * Plane recovery diagnostics (v11 §23, v11.1 §21–§24, §29). After a
 * significant motion with a healthy map the plane detector alone is
 * re-seeded; the map, the camera pose and the world are untouched. Numbers
 * only: formatting happens in the HUD, debug on. `candidateFound` /
 * `candidateCommitted` describe the *current frame's* search; a candidate
 * the detector still holds from earlier frames shows as `previousCandidateHeld`.
 */
export interface PlaneRecoveryDiagnostics {
  active: boolean;
  reason: PlaneRecoveryReason;
  state: PlaneRecoveryState;
  /** Map PnP located the camera this frame with enough inliers and a finite pose. */
  mapHealthy: boolean;
  mapInliers: number;
  trackedFeatures: number;
  /** Landmarks observed within the seed window, and how many of them were handed to the plane search. */
  seedCandidates: number;
  seededPoints: number;
  /** Plane search of this frame (same values as `planeSearch`). */
  searchPoints: number;
  bestInliers: number;
  requiredInliers: number;
  searchStage: PlaneSearchStage;
  /** This frame's search produced a plane candidate (`searchStage === "candidate"`). */
  candidateFound: boolean;
  /** This frame's candidate is horizontal, so it counts toward the stability streak. */
  candidateCommitted: boolean;
  /** The detector still holds a candidate from an earlier frame (grace period) although this frame's search produced none. */
  previousCandidateHeld: boolean;
  stableFrames: number;
  requiredStableFrames: number;
  /** Time since the recovery started (0 when inactive). */
  recoveryElapsedMs: number;
  /** Recoveries started for the current map. */
  recoveries: number;
}

/** New-landmark triangulation of one frame (v11 diagnostics: why the map does / does not grow). */
export interface TriangulationStats {
  /** Landmark-less tracks with an anchor from an earlier frame. */
  candidates: number;
  /** Rejected: displacement from the anchor below `triangulateMinParallaxPx`. */
  parallaxRejected: number;
  /** Rejected: negative depth in either view. */
  cheiralityRejected: number;
  /** Rejected: ray angle below `minTriangulationAngleDeg`. */
  angleRejected: number;
  /** Rejected: two-view residual above `maxTriangulationErrorPx`. */
  errorRejected: number;
  /** Rejected: depth outside `maxDepthRatio` of the median landmark depth. */
  depthRejected: number;
  added: number;
}

/** Keyframe / relocalization status (Phase 5). */

export interface RelocalizationOutput {
  keyframes: number;
  /**
   * Result of the attempt made in this frame (v5 §18): `candidate` = passed
   * the global validation, held for confirmation; `success` = applied to the
   * canonical pose; `fail` = rejected (see `rejectCode`).
   */
  attempt: "none" | "candidate" | "success" | "fail";
  inlierCount: number;
  candidatesTried: number;
  /** Frame id of the last successful relocalization (-1 when none). */
  lastSuccessFrame: number;
  /** Total successful (applied) relocalizations in this session. */
  successCount: number;
  /** Why the last attempt failed (best candidate's stage), null when none / success. */
  reason: string | null;
  rejectCode: RelocalizationRejectCode | null;
  /** Quality of the candidate of this attempt (v5 §5–§6). */
  meanReprojectionErrorPx: number;
  /**
   * Reprojection-error tier of this attempt's candidate (v12): `strict` ≤
   * `maxMeanErrorPx`, `acceptable` ≤ `acceptableMeanErrorPx` with every other
   * condition passing (always confirmed before apply), `rejected` otherwise;
   * null when no candidate reached PnP.
   */
  errorTier: "strict" | "acceptable" | "rejected" | null;
  matchScore: number;
  inlierRatio: number;
  spatialCells: number;
  keyframeId: number;
  /** On candidate / success: how far the candidate pose is from the held / predicted pose (map units / deg). */
  jumpTranslation: number;
  jumpRotationDeg: number;
  /**
   * Post-relocalization consistency (v5 §16–§17): largest difference between
   * the map PnP pose and the relocalized pose over the monitored frames after
   * the last relocalization (map units / deg), and whether the map PnP
   * contradicted it (jump-rejected or far off).
   */
  postDeltaTranslation: number;
  postDeltaRotationDeg: number;
  postInconsistent: boolean;
  /**
   * Stage counters / best trial of the most recent attempt (v6 §1–§5). Kept
   * while the camera stays lost (attempts run every few frames), cleared
   * when tracking resumes. `framesSinceAttempt` says how old it is.
   */
  diagnostics: RelocalizationDiagnostics | null;
  framesSinceAttempt: number;
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

/**
 * Frame-to-frame motion and LK diagnostics (修正指示書 v7 §3, §12–§13).
 * `level` is decided from the *previous* frame's median displacement and
 * drives this frame's prediction / search scale.
 */
export type MotionLevel = "normal" | "medium" | "fast";

export interface MotionDiagnostics {
  level: MotionLevel;
  /** Median / max displacement of the tracks that survived LK this frame (px). */
  medianDisplacementPx: number;
  maxDisplacementPx: number;
  /** Tracks offered to LK and tracks it kept. */
  trackedBefore: number;
  trackedAfter: number;
  forwardBackwardRejects: number;
  tooFarRejects: number;
  /** Mean LK residual of the kept tracks (0–255). */
  meanResidual: number;
  /** How LK was seeded this frame. */
  predictionMode: "homography" | "velocity" | "none";
  /** Displacement-gate scale applied this frame (1 = normal). */
  searchScale: number;
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
  /** Plane recovery after fast motion (v11 §23); always present, `active` false when idle. */
  planeRecovery: PlaneRecoveryDiagnostics;
  /** The fixed plane the world is anchored to (null until a plane was found). */
  planeAnchor: PlaneAnchorOutput | null;
  /** Plane-relative pose quality (null until anchored). */
  planePose: PlanePoseOutput | null;
  /** Keyframe / relocalization status (Phase 5). */
  relocalization: RelocalizationOutput;
  /** Frame-to-frame motion level and LK diagnostics (v7). */
  motion: MotionDiagnostics;
  /**
   * World tracking established for the current map (v10 §5–§7): a plane was
   * found and the world anchored to it. Only then does a lost map lead to
   * RELOCALIZING; before, the scan continues wherever the camera looks.
   */
  worldEstablished: boolean;
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
