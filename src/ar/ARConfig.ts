/**
 * Central configuration for the AR engine.
 *
 * Nothing in the vision pipeline embeds magic numbers; every tunable lives
 * here so that on-device tuning (iPhone Safari / Android Chrome) only touches
 * this file or the object passed to `ARSession`.
 */

/** Resolution used for vision processing (not the camera capture resolution). */
export interface ProcessingConfig {
  /** Target processing width in pixels (spec: 640). */
  width: number;
  /** Target processing height in pixels (spec: 480 or 360; aspect follows camera). */
  height: number;
  /**
   * Vision frames per second cap. The camera may run at 30fps+, vision runs at
   * 15〜30fps. 0 means "as fast as frames arrive".
   */
  maxVisionFps: number;
  /**
   * Camera field of view along the long image side (degrees), used to
   * approximate the focal length (spec §8). A wrong value distorts the map
   * and makes placed objects drift while the camera moves.
   */
  longSideFovDeg: number;
  /**
   * Show the camera frame that the pose was computed on instead of the live
   * video, so rendering and video are time-aligned (the live video would be
   * ~2 frames ahead of the pose and objects would lag during motion).
   */
  syncVideoToPose: boolean;
}

/** Feature detection configuration (spec §11, §49). */
export interface FeatureConfig {
  /** Upper bound on simultaneously tracked features. */
  maxFeatures: number;
  /** Below this count the engine reports INSUFFICIENT_FEATURES / LOW_FEATURE. */
  minFeatures: number;
  /** Replenish (re-detect) when tracked count drops under this value. */
  replenishBelow: number;
  /**
   * FAST corner threshold (0–255). Adaptive: lowered when too few corners are
   * found, raised when too many.
   */
  fastThreshold: number;
  /** Minimum FAST threshold reachable by adaptation. */
  fastThresholdMin: number;
  /** Maximum FAST threshold reachable by adaptation. */
  fastThresholdMax: number;
  /** Minimum distance (pixels) between any two features. */
  minDistance: number;
  /** Grid used for spatial distribution of features (columns). */
  gridCols: number;
  /** Grid used for spatial distribution of features (rows). */
  gridRows: number;
  /** Maximum features accepted per grid cell. */
  maxPerCell: number;
  /** Ignore corners closer than this to the image border. */
  borderMargin: number;
}

/** Pyramidal Lucas-Kanade configuration (spec §12, §13). */
export interface TrackerConfig {
  /** Number of pyramid levels (3 → 640, 320, 160). */
  pyramidLevels: number;
  /** LK window size in pixels (odd). */
  windowSize: number;
  /** Max Gauss-Newton iterations per level. */
  maxIterations: number;
  /** Convergence threshold on the update step (pixels). */
  epsilon: number;
  /** Reject tracks whose mean absolute residual exceeds this (0–255). */
  maxResidual: number;
  /** Reject when the structure tensor's min eigenvalue is below this (per pixel). */
  minEigenvalue: number;
  /** Forward-backward error threshold in pixels (spec §13). */
  forwardBackwardThreshold: number;
  /** Maximum displacement a track may move between frames (pixels, at level 0). */
  maxDisplacement: number;
  /**
   * Seed LK with a motion prediction so fast motion stays inside the
   * pyramid's capture range (修正指示書 v7 §6–§7): the previous frame's
   * frame-to-frame homography applied to each track when that homography is
   * well supported (≥ predictionMinInliers inliers, inlier ratio ≥
   * predictionMinInlierRatio), else each track's own last displacement
   * (constant velocity). Off → LK starts at the previous position.
   */
  predictMotion: boolean;
  homographyPrediction: boolean;
  predictionMinInliers: number;
  predictionMinInlierRatio: number;
  /**
   * Motion level from the previous frame's median track displacement (px,
   * v7 §3): below mediumMotionPx = normal, below fastMotionPx = medium,
   * else fast. The LK displacement gate (maxDisplacement, measured from the
   * prediction) is scaled by the level's search scale (v7 §4) so the gate
   * only opens up while the camera actually moves fast.
   */
  mediumMotionPx: number;
  fastMotionPx: number;
  mediumMotionSearchScale: number;
  fastMotionSearchScale: number;
}

/** RANSAC configuration (spec §14, §15). */
export interface RansacConfig {
  /** Inlier threshold in pixels (symmetric transfer error, level-0 pixels). */
  inlierThreshold: number;
  /** Desired probability that at least one sample is outlier-free. */
  confidence: number;
  /** Hard cap on iterations. */
  maxIterations: number;
  /** Minimum correspondences needed to run RANSAC at all. */
  minCorrespondences: number;
  /** Consecutive outlier frames before a track is dropped (1 = immediately). */
  outlierFramesToDrop: number;
}

/** Two-view relative pose estimation (spec §15–§18, Phase 2). */
export interface PoseConfig {
  /** Minimum reference↔current correspondences to attempt a pose. */
  minCorrespondences: number;
  /** RANSAC inlier threshold in pixels (Sampson for E, transfer error for H). */
  ransacThresholdPx: number;
  /** Hard cap on RANSAC iterations for E. */
  maxIterations: number;
  /**
   * Choose the homography model when H_inliers / (H_inliers + E_inliers)
   * exceeds this (planar scene or pure rotation; ORB-SLAM uses 0.45).
   */
  homographyRatioThreshold: number;
  /** Below this median parallax (px) only the rotation is estimated. */
  minParallaxPx: number;
  /** Parallax (px) at which the translation direction gets full confidence. */
  fullConfidenceParallaxPx: number;
  /** Triangulation reprojection tolerance (px) for the cheirality test. */
  maxTriangulationErrorPx: number;
  /** Minimum fraction of inliers that must pass the positive-depth test. */
  minCheiralityRatio: number;
  /** Inlier count that saturates the confidence score. */
  goodInlierCount: number;
  /**
   * Renew the reference frame when fewer than this many tracks still link
   * to it. The accumulated rotation is composed across renewals.
   */
  minReferenceTracks: number;
  /**
   * Renew the reference frame when the median parallax exceeds this (px),
   * keeping the two-view problem well conditioned and bounded in time.
   */
  maxReferenceParallaxPx: number;
}

/** Landmark map: two-view initialization, PnP tracking, triangulation (spec §19–§20, §48, Phase 3). */
export interface LandmarkConfig {
  /** Median parallax (px) between reference and current frame needed to initialize the map. */
  initMinParallaxPx: number;
  /** Translation confidence of the two-view pose needed to initialize. */
  initMinTranslationConfidence: number;
  /** Landmarks that must triangulate well for the initialization to be accepted. */
  initMinLandmarks: number;
  /** Minimum ray parallax angle (degrees) for a triangulated point. */
  minTriangulationAngleDeg: number;
  /** Maximum reprojection error (px, both views summed) for a triangulated point. */
  maxTriangulationErrorPx: number;
  /** Pixel displacement from the anchor observation before a track is triangulated. */
  triangulateMinParallaxPx: number;
  /** Huber threshold for PnP (px). */
  pnpHuberPx: number;
  /** PnP inlier gate (px). */
  pnpInlierPx: number;
  pnpMaxIterations: number;
  /** PnP inliers needed to accept a map-frame pose. */
  minPnPInliers: number;
  /** Consecutive PnP-outlier frames before a landmark is removed. */
  maxOutlierCount: number;
  /** A young landmark (≤ 2 observations) is removed after this many outlier frames. */
  youngOutlierFrames: number;
  /**
   * Landmarks take part in the pose solve only once they have been PnP
   * inliers this many times (v2 §27: a fresh triangulation is a candidate,
   * not yet part of the fixed map). Younger ones are classified against the
   * solved pose and mature or die.
   */
  minObservationsForPose: number;
  /** Use the mature-only solve when at least this many mature landmarks are observed. */
  minMaturePnPPoints: number;
  /** Reject new triangulations deeper than this × (or shallower than 1/this ×) the median landmark depth. */
  maxDepthRatio: number;
  /** Upper bound on stored landmarks (spec §48: 500–2000). */
  maxLandmarks: number;
  /** Landmarks unseen for this many frames are removed. */
  maxLandmarkAgeFrames: number;
  /** Frames without a map pose before the map is reset (Phase 5 relocalization replaces this). */
  lostResetFrames: number;
  /**
   * Re-triangulate a landmark from its anchor observation when the ray
   * parallax has grown by this factor since its last triangulation
   * (depth refinement as the baseline grows).
   */
  refineParallaxGrowth: number;
  /**
   * Re-triangulate landmarks as the baseline grows (修正指示書 v2 §3). Off by
   * default: the map is the coordinate system the world is anchored to, and
   * rewriting landmark positions from the current pose feeds pose error back
   * into the map (pose → landmark → pose). `?refine=1` in the demo for A/B.
   */
  enableLandmarkDepthRefinement: boolean;
  /**
   * Temporal jump gate (修正指示書 v2 §8, v4 §2–§5). Every pose candidate (map
   * PnP and plane PnP alike) is rejected (frame counted as lost, pose
   * propagated) when its camera center is more than
   * max(jumpRejectDepthRatio × median landmark depth, jumpRejectSpeedFactor ×
   * previous frame's displacement) away from the current pose or rotated
   * more than jumpRejectRotationDeg from it. There is no quality bypass:
   * jumpRejectTrustedInliers / jumpRejectTrustedErrorPx only *label* a
   * candidate as trusted (good PnP) on the HUD — trusted ≠ continuous (v4 §9).
   * While lost, the reference is a predicted / held pose whose uncertainty
   * grows, so both limits grow by jumpRejectLostGrowthPerFrame × lost frames
   * (recovery from a long loss otherwise could never pass; the recovery
   * inlier threshold still applies).
   */
  jumpRejectDepthRatio: number;
  jumpRejectSpeedFactor: number;
  jumpRejectRotationDeg: number;
  jumpRejectTrustedInliers: number;
  jumpRejectTrustedErrorPx: number;
  jumpRejectLostGrowthPerFrame: number;
  /** Cap of the lost-time growth factor (v5 §13): beyond it only relocalization can bring the camera back. */
  jumpRejectMaxLostGrowth: number;
  /**
   * Before a world is established (v10 §8–§12) a lost map is not worth
   * keeping: after this many lost frames the map and keyframes are dropped
   * and the scan re-initializes where the camera looks now. Short enough
   * that moving to another desk feels immediate, long enough for a fast
   * swing over the same surface to recover through re-association.
   */
  preWorldLostResetFrames: number;
  /**
   * Long loss (v5 §12–§14): after this many lost frames the normal PnP
   * recovery needs minRecoveryInliersLong inliers (instead of
   * minRecoveryInliers), so a stale pose, a widened gate and a few links do
   * not re-seed the camera; relocalization is the preferred way back.
   */
  longLostFrames: number;
  minRecoveryInliersLong: number;
  /**
   * Hysteresis between pose sources (v3 §7): after the canonical pose
   * switched source (map ↔ plane), switching back waits this many frames
   * unless the current source has no valid candidate.
   */
  sourceSwitchCooldownFrames: number;
  /**
   * Re-association of unlinked landmarks (v3 §15, map kept alive): mature
   * landmarks whose track died are projected with the current pose and
   * linked to an unlinked track within this radius (px). FAST re-detects the
   * same corners, so replenished tracks pick their landmarks back up instead
   * of leaving PnP without observations.
   */
  reassociateRadiusPx: number;
  /** Also re-associate with the propagated pose while lost for at most this many frames. */
  reassociateMaxLostFrames: number;
  /**
   * PnP inliers required to come back from a lost frame (higher than
   * minPnPInliers): re-associated links are unverified, and a dozen chance
   * matches on an unrelated scene must not look like a recovered pose.
   */
  minRecoveryInliers: number;
  /**
   * Guided recovery: when lost and the PnP on the few re-associated links has
   * at least this many inliers with at most recoverySeedErrorPx mean error,
   * every unlinked landmark is projected with that seed pose and
   * re-associated within recoveryReassociateRadiusPx, then PnP runs again
   * on the enlarged set in the same frame.
   */
  recoverySeedInliers: number;
  recoverySeedErrorPx: number;
  recoveryReassociateRadiusPx: number;
  /**
   * While lost, predict the camera center with the last tracked velocity for
   * this many frames (then hold it). The prediction is what lets landmarks
   * be re-associated during a short burst of fast motion.
   */
  velocityPropagationFrames: number;
}

/** Plane detection (spec §21–§24, §51, Phase 3). */
export interface PlaneConfig {
  /** RANSAC inlier distance as a fraction of the median landmark distance from the map origin. */
  inlierThresholdRatio: number;
  minInliers: number;
  maxIterations: number;
  confidence: number;
  /** |cos(normal, up)| above which a plane is horizontal when gravity is known (spec §22: 0.90). */
  horizontalThreshold: number;
  /**
   * Threshold used when no gravity reading is available. The camera −Y axis of
   * the map frame is then assumed to point up (phone held upright), which
   * cannot distinguish a wall from a floor when the phone looks straight down.
   */
  fallbackHorizontalThreshold: number;
  /** Normal change (degrees) tolerated between frames for the plane to count as stable. */
  stableAngleDeg: number;
  /** Center shift tolerated between frames, as a fraction of the median landmark distance. */
  stableCenterRatio: number;
  /** Consecutive stable frames before PLANE_FOUND (spec §24). */
  stableFramesRequired: number;
  /** Frames a found plane may go undetected before it is dropped. */
  lostFrames: number;
  /** Inlier count that saturates the confidence score. */
  goodInlierCount: number;
  /** Landmarks need this many observations to take part in plane fitting. */
  minLandmarkObservations: number;
  /**
   * Plane recovery after fast motion (v11 §9–§10, §17): while a recovery is
   * active the plane search is seeded only with landmarks seen as a PnP
   * inlier within this many frames, i.e. the part of the map the camera is
   * looking at *now*; landmarks of the view before the motion stay in the
   * map (for tracking) but do not vote for the plane. Outside a recovery the
   * seed set is unchanged (every mature landmark).
   */
  recoverySeedMaxAgeFrames: number;
}

/** Keyframes and relocalization (spec §35–§36, Phase 5). */
export interface RelocalizationConfig {
  /** Keyframes kept (the first one is always kept). */
  maxKeyframes: number;
  /** PnP inliers required in a frame to become a keyframe. */
  keyframeMinInliers: number;
  /** Minimum frames between keyframes. */
  keyframeMinFrameGap: number;
  /** Create a keyframe at the latest after this many frames. */
  keyframeMaxFrameGap: number;
  /** Rotation since the last keyframe (degrees) that triggers a new one. */
  keyframeRotationDeg: number;
  /** Median landmark displacement since the last keyframe (px) that triggers a new one. */
  keyframeParallaxPx: number;
  /**
   * Keyframe ranking (v14 §8–§12): every attempt ranks *all* keyframes by a
   * cheap coarse similarity (zero-mean NCC on a 1/16 image, shift search of
   * this radius in its pixels — the same ±level-0 range as
   * `coarseSearchRadius` on the 1/8 image) and only the best
   * `maxLkCandidatesPerFrame` go on to the 1/8 refinement, LK and PnP. No
   * keyframe is excluded by its age or by the lost pose (§13–§15, §35).
   */
  rankSearchRadius: number;
  /**
   * Refinement of the ranked shift on the 1/8 coarse image (radius in its
   * pixels around 2 × the rank shift); its score is the coarse NCC that
   * `coarseMinScore` gates and that v12's relaxed range requires.
   */
  coarseRefineRadius: number;
  /** Coarse search radius (pixels of the coarse image ≈ level-0 / 8); kept for the full search used in tests / fallback. */
  coarseSearchRadius: number;
  /** Minimum NCC score of the coarse alignment to proceed. */
  coarseMinScore: number;
  /**
   * Search budget per attempt (v14 §16–§17): keyframes sent to LK (the top
   * of the ranking) and to PnP; the first attempt of a lost episode may try
   * more (`lkCandidatesFirstAttempt`). Never every keyframe (§11).
   */
  maxLkCandidatesPerFrame: number;
  lkCandidatesFirstAttempt: number;
  maxPnpCandidatesPerFrame: number;
  /** LK displacement gate around the coarse guess (px). */
  lkMaxDisplacementPx: number;
  /**
   * Relocalization-specific LK thresholds (v14 §28–§30): keyframe → current
   * frame LK bridges seconds and a viewpoint change, so its forward-backward
   * tolerance and residual bound are its own; the normal tracker's values
   * are untouched. The checks themselves (FB, residual, texture, bounds,
   * displacement) all stay in force, and PnP + validation remain the
   * verifiers. Initial candidate values.
   */
  lkForwardBackwardPx: number;
  lkMaxResidual: number;
  /**
   * Retry scheduling (v14 §21–§22, §34): a keyframe that failed is not sent
   * to LK again for this many frames *unless the image changed* — the
   * current 1/16 image vs the one it failed on has a zero-shift NCC below
   * `retryImageChangeScore`. It stays ranked and visible in the diagnostics.
   */
  retryCooldownFrames: number;
  retryImageChangeScore: number;
  /**
   * A pending (validated, unconfirmed) candidate older than this many frames
   * is dropped and recomputed (v14 §43–§44): a stale candidate is never
   * applied as the current pose.
   */
  candidateMaxAgeFrames: number;
  pnpHuberPx: number;
  pnpInlierPx: number;
  /** Inliers needed to accept a relocalization. */
  minInliers: number;
  /** Strict reprojection bound: a candidate at or below this is `strong` (v12 §3.1). */
  maxMeanErrorPx: number;
  /**
   * Relaxed reprojection bound (v12 §4–§5): a candidate above `maxMeanErrorPx`
   * but at or below this is `acceptable` when *every other* condition passes
   * — minInliers, minInlierRatio, minSpatialCells, minSpatialCoverage,
   * `relaxedMinMatchScore`, finite pose and the relocalization jump limits —
   * and is then always held for a confirmation frame, never applied at once.
   * The strict bound is not moved; the on-device 50i / 2.44 px / ratio 0.85
   * / 7 cells / NCC 0.67 return passes through the other conditions. Values
   * ≤ `maxMeanErrorPx` disable the relaxed range. Initial candidate value
   * (v12 §19); never ≥ 5 px (§24).
   */
  relaxedMeanErrorPx: number;
  /**
   * Coarse NCC score the relaxed error range requires (v12 §5–§6). The
   * coarse gate `coarseMinScore` (0.25) only saves work; a candidate that
   * needs the relaxed range must also look like the keyframe. Initial value.
   */
  relaxedMinMatchScore: number;
  /**
   * Relocalization-specific jump limits vs the held pose (v12 §9), applied
   * to every candidate: translation ≤ this × scene depth (map units),
   * rotation ≤ `maxRotationJumpDeg`. Generous on purpose — a correct return
   * after a long loss can be far from the held pose (v5 §9) — they only cut
   * off a candidate that would put the camera somewhere else entirely. 0
   * disables.
   */
  maxTranslationJumpDepthRatio: number;
  maxRotationJumpDeg: number;
  /** Frames the map may stay lost before a relocalization attempt starts. */
  startAfterLostFrames: number;
  /** Failed relocalization attempts required (besides the lost time) before the map / world are reset. */
  minAttemptsBeforeReset: number;
  /** Try to relocalize every N lost frames (cost control). */
  attemptEveryNFrames: number;
  /**
   * Global validation of a relocalization candidate (修正指示書 v5 §5–§7):
   * besides minInliers / maxMeanErrorPx / coarseMinScore, the PnP inliers
   * must be at least this fraction of the LK-tracked observations …
   */
  minInlierRatio: number;
  /** … and occupy at least this many cells of a 3×3 grid over the image (spatial distribution, v5 §6) … */
  minSpatialCells: number;
  /**
   * … and their bounding box must cover at least this fraction of the image
   * (v9 §13: 35 inliers in 4 cells can span half the image or sit in one
   * patch). 0 = diagnostic only; raised only after on-device logs (v9 §15, §30).
   */
  minSpatialCoverage: number;
  /**
   * Confirmation (v5 §10–§11): a validated candidate is applied at once only
   * when it is clearly high quality (≥ immediateInliers inliers and mean
   * error ≤ immediateMaxErrorPx); otherwise it is held and applied when the
   * next frame's relocalization lands within confirmTranslationDepthRatio ×
   * scene depth / confirmRotationDeg of it. 0 confirmation frames = always
   * apply at once.
   */
  confirmationFrames: number;
  immediateInliers: number;
  immediateMaxErrorPx: number;
  confirmTranslationDepthRatio: number;
  confirmRotationDeg: number;
  /** Frames after a relocalization during which the map PnP is compared with the relocalized pose (v5 §16–§17). */
  postRelocMonitorFrames: number;
}

/** World anchoring, hit test and rendering (spec §25–§30, §33–§34, Phase 4). */
export interface WorldConfig {
  /** Assumed camera→plane distance (m) when the world is created; fixes the monocular scale. */
  assumedPlaneDistanceMeters: number;
  /**
   * No longer used for object visibility (v13 §1–§2, §5): objects are hidden
   * as soon as the state machine declares the loss (its own
   * `state.mapLostFrameTolerance` / `lostFrameTolerance` hysteresis) and
   * shown again only on a fresh, confirmed map pose. Kept for configuration
   * compatibility.
   */
  holdPoseOnLostMs: number;
  /**
   * After the world was lost, show generic recovery guidance first; the
   * "return to where you were" guidance appears only once relocalization
   * has been failing for this long (ms, v10 §14–§15).
   */
  relocGuidanceDelayMs: number;
  /**
   * Apply the One Euro filters to the rendered camera pose. Off while the
   * raw pose is being validated (v3 §22): filter lag and real drift must not
   * be confused. `?smooth=1` in the demo.
   */
  smoothing: boolean;
  /** Pose smoothing (One Euro) for the rendered camera position. */
  positionSmoothing: { minCutoff: number; beta: number; dCutoff: number };
  /** Pose smoothing for the rendered camera rotation. */
  rotationSmoothing: { minCutoff: number; beta: number; dCutoff: number };
  /** Three.js camera near / far planes (m). */
  near: number;
  far: number;
  /** Edge length of the demo cube (m). */
  cubeSize: number;
  /** Show the plane grid in the Three.js scene. */
  showPlaneGrid: boolean;
}

/** Tracking state transition thresholds (spec §31, §33). */
export interface StateConfig {
  /** Tracked inliers needed to enter / remain in TRACKING. */
  minTrackedForTracking: number;
  /** Below this tracked inlier count → TRACKING_LOST. */
  lostBelow: number;
  /** Consecutive bad frames tolerated before declaring TRACKING_LOST. */
  lostFrameTolerance: number;
  /**
   * Consecutive frames without a map pose (PnP failed) tolerated before
   * declaring RELOCALIZING. Relocalization attempts start earlier
   * (`relocalization.startAfterLostFrames`); this only keeps a single bad
   * PnP frame from flipping the state and hiding the content.
   */
  mapLostFrameTolerance: number;
}

export interface DebugConfig {
  /** Enable `[AR]` console logs (disabled in production builds by default). */
  log: boolean;
  /** Interval between log lines in milliseconds. */
  logIntervalMs: number;
  /** Draw features / motion vectors / HUD. */
  overlay: boolean;
  /** Pose age (render time − frame capture time) above which the HUD shows POSE STALE (ms). */
  poseStaleMs: number;
}

/**
 * Plane-anchored camera tracking (修正指示書 §5–§9): once the plane is fixed,
 * the pose is solved against features lifted onto that plane instead of
 * triangulated landmark depths.
 */
export interface PlaneTrackingConfig {
  /**
   * Use the plane-relative pose as the main pose once a plane is anchored.
   * Experimental alternative to the landmark PnP; off by default.
   */
  enabled: boolean;
  /** Plane inliers required to accept the plane-relative pose (else fall back to the map pose). */
  minInliers: number;
  /** Inlier count that yields full confidence. */
  goodInliers: number;
  /** Huber threshold of the plane PnP (px). */
  pnpHuberPx: number;
  /** Inlier gate of the plane PnP (px). */
  pnpInlierPx: number;
  pnpMaxIterations: number;
  /** Consecutive inlier frames before a lifted feature is trusted for the solve. */
  probationFrames: number;
  /** Consecutive outlier frames after which a feature is declared off-plane. */
  maxOutlierStreak: number;
  /** Lift new features only in frames whose pose had at least this many inliers … */
  liftMinInliers: number;
  /** … and at most this mean reprojection error (px). */
  liftMaxMeanErrorPx: number;
  /** Rays closer than this angle to the plane are not lifted (unstable intersection). */
  minRayAngleDeg: number;
  /** Do not lift points farther than this × the anchoring camera–plane distance. */
  maxLiftDistanceRatio: number;
  /**
   * New plane points are lifted only from a trusted *map* pose (never from
   * the plane pose itself, v3 §8–§11) and only this many frames after the
   * pose source last switched.
   */
  liftCooldownFrames: number;
}

export interface ARConfig {
  processing: ProcessingConfig;
  features: FeatureConfig;
  tracker: TrackerConfig;
  ransac: RansacConfig;
  pose: PoseConfig;
  landmarks: LandmarkConfig;
  plane: PlaneConfig;
  planeTracking: PlaneTrackingConfig;
  relocalization: RelocalizationConfig;
  world: WorldConfig;
  state: StateConfig;
  debug: DebugConfig;
  /** Run the vision engine in a Web Worker (false → main thread fallback). */
  useWorker: boolean;
}

export const DEFAULT_CONFIG: ARConfig = {
  processing: {
    width: 640,
    height: 480,
    maxVisionFps: 30,
    longSideFovDeg: 66,
    syncVideoToPose: true,
  },
  features: {
    maxFeatures: 300,
    minFeatures: 40,
    replenishBelow: 240,
    fastThreshold: 20,
    fastThresholdMin: 8,
    fastThresholdMax: 60,
    minDistance: 10,
    gridCols: 8,
    gridRows: 6,
    maxPerCell: 12,
    borderMargin: 12,
  },
  tracker: {
    // 4 levels (360×640 → 45×80) roughly double the per-frame displacement
    // LK can follow (~56 px at level 0 with a 15 px window).
    pyramidLevels: 4,
    windowSize: 15,
    maxIterations: 20,
    epsilon: 0.03,
    maxResidual: 30,
    minEigenvalue: 0.5,
    forwardBackwardThreshold: 1.0,
    maxDisplacement: 60,
    predictMotion: true,
    homographyPrediction: true,
    predictionMinInliers: 30,
    predictionMinInlierRatio: 0.6,
    mediumMotionPx: 8,
    fastMotionPx: 20,
    mediumMotionSearchScale: 1.5,
    fastMotionSearchScale: 2.0,
  },
  ransac: {
    // Frame-to-frame homography gate. Real rooms are not planar: with the
    // camera 1 m from the floor and furniture 3 m away, correct tracks on the
    // background legitimately deviate from the floor homography by several
    // pixels per frame. The gate therefore only catches gross LK failures;
    // tracks must be outliers in `outlierFramesToDrop` consecutive frames.
    inlierThreshold: 6.0,
    confidence: 0.99,
    maxIterations: 200,
    minCorrespondences: 12,
    outlierFramesToDrop: 2,
  },
  pose: {
    minCorrespondences: 20,
    ransacThresholdPx: 1.5,
    maxIterations: 300,
    homographyRatioThreshold: 0.45,
    minParallaxPx: 2.0,
    fullConfidenceParallaxPx: 25,
    maxTriangulationErrorPx: 4.0,
    minCheiralityRatio: 0.7,
    goodInlierCount: 60,
    minReferenceTracks: 40,
    maxReferenceParallaxPx: 120,
  },
  landmarks: {
    initMinParallaxPx: 30,
    initMinTranslationConfidence: 0.5,
    initMinLandmarks: 30,
    minTriangulationAngleDeg: 1.0,
    // The map is fixed once built (no depth refinement), so landmarks must
    // be well conditioned when created: 20 px of parallax from the anchor
    // (6 px with later refinement before) and a tight two-view residual.
    maxTriangulationErrorPx: 2.5,
    triangulateMinParallaxPx: 20,
    pnpHuberPx: 4.0,
    pnpInlierPx: 6.0,
    pnpMaxIterations: 10,
    minPnPInliers: 12,
    maxOutlierCount: 5,
    youngOutlierFrames: 2,
    minObservationsForPose: 3,
    minMaturePnPPoints: 24,
    maxDepthRatio: 3,
    maxLandmarks: 1000,
    maxLandmarkAgeFrames: 150,
    // World reset is the last resort (v3 §16): 10 s lost AND enough failed
    // relocalization attempts (relocalization.minAttemptsBeforeReset).
    lostResetFrames: 300,
    refineParallaxGrowth: 1.3,
    enableLandmarkDepthRefinement: false,
    jumpRejectDepthRatio: 0.08,
    jumpRejectSpeedFactor: 3,
    jumpRejectRotationDeg: 20,
    jumpRejectTrustedInliers: 40,
    jumpRejectTrustedErrorPx: 1.5,
    // 10 lost frames double the gate; 90 frames (the re-association horizon)
    // make it 10× — about 40 cm at the desk scale.
    jumpRejectLostGrowthPerFrame: 0.1,
    // Capped at 3× (20 lost frames): about 12 cm at the desk scale.
    jumpRejectMaxLostGrowth: 3,
    preWorldLostResetFrames: 30,
    longLostFrames: 30,
    minRecoveryInliersLong: 40,
    sourceSwitchCooldownFrames: 15,
    // FAST re-detects a corner within ~1 px; a tight radius keeps chance
    // matches on unrelated texture rare.
    reassociateRadiusPx: 2.5,
    reassociateMaxLostFrames: 90,
    minRecoveryInliers: 24,
    recoverySeedInliers: 4,
    recoverySeedErrorPx: 2.0,
    recoveryReassociateRadiusPx: 4,
    velocityPropagationFrames: 10,
  },
  relocalization: {
    maxKeyframes: 8,
    keyframeMinInliers: 30,
    keyframeMinFrameGap: 10,
    keyframeMaxFrameGap: 90,
    keyframeRotationDeg: 10,
    keyframeParallaxPx: 40,
    // v14 §10–§11: rank all keyframes on the 1/16 image (±12 px there = the
    // ±192 level-0 px the old ±24 px search on the 1/8 image covered, at
    // ~1/8 of its cost per keyframe), refine the top ones on the 1/8 image.
    rankSearchRadius: 12,
    coarseRefineRadius: 2,
    coarseSearchRadius: 24,
    // The PnP acceptance (≥ 25 inliers, ≤ 1.5 px) is the real verifier; the
    // coarse score only saves work. 0.45 refused views that came back with a
    // few degrees of rotation.
    coarseMinScore: 0.25,
    // v14 §16–§17: top 3 keyframes per attempt (4 on the first attempt of an
    // episode), never all 8 through LK / PnP.
    maxLkCandidatesPerFrame: 3,
    lkCandidatesFirstAttempt: 4,
    maxPnpCandidatesPerFrame: 3,
    lkMaxDisplacementPx: 40,
    // v14 §28–§30: keyframe → current LK after seconds and a viewpoint change
    // is not frame-to-frame LK; its forward-backward tolerance is its own
    // (the tracker keeps 1.0 px). PnP (6 px) and validation still decide.
    lkForwardBackwardPx: 2.0,
    lkMaxResidual: 30,
    // v14 §21–§22: a failed keyframe waits two attempt periods (3 frames
    // each) unless the image changed (NCC vs the frame it failed on < 0.9).
    retryCooldownFrames: 6,
    retryImageChangeScore: 0.9,
    // v14 §43–§44: a candidate is confirmed in the next frame or dropped.
    candidateMaxAgeFrames: 2,
    pnpHuberPx: 4,
    pnpInlierPx: 6,
    minInliers: 25,
    maxMeanErrorPx: 1.5,
    // Keyframe→current LK after a loss is naturally noisier than the
    // per-frame LK; between 1.5 and 3 px the other conditions decide (v12).
    relaxedMeanErrorPx: 3.0,
    relaxedMinMatchScore: 0.5,
    maxTranslationJumpDepthRatio: 1.0,
    maxRotationJumpDeg: 90,
    startAfterLostFrames: 1,
    minAttemptsBeforeReset: 10,
    // Every 3rd frame: relocalization attempts dominated the lost-frame cost
    // (67 ms on iPhone); guided re-association now carries the quick recoveries.
    attemptEveryNFrames: 3,
    minInlierRatio: 0.5,
    minSpatialCells: 4,
    minSpatialCoverage: 0,
    confirmationFrames: 1,
    immediateInliers: 60,
    immediateMaxErrorPx: 1.0,
    confirmTranslationDepthRatio: 0.05,
    confirmRotationDeg: 5,
    postRelocMonitorFrames: 3,
  },
  plane: {
    inlierThresholdRatio: 0.05,
    minInliers: 20,
    maxIterations: 200,
    confidence: 0.99,
    horizontalThreshold: 0.9,
    fallbackHorizontalThreshold: 0.5,
    stableAngleDeg: 5,
    stableCenterRatio: 0.1,
    stableFramesRequired: 5,
    lostFrames: 15,
    goodInlierCount: 80,
    minLandmarkObservations: 3,
    // Same window as `lostFrames`: a landmark that has not been a PnP inlier
    // for this long is not part of the current view.
    recoverySeedMaxAgeFrames: 15,
  },
  world: {
    assumedPlaneDistanceMeters: 0.5,
    // Unused since v13 (objects hide on the engine's own lost hysteresis).
    holdPoseOnLostMs: 10000,
    relocGuidanceDelayMs: 2000,
    smoothing: false,
    // Light smoothing: with the displayed frame synchronized to the pose,
    // any filter lag shows up as the object sliding during motion.
    positionSmoothing: { minCutoff: 4.0, beta: 1.5, dCutoff: 1.0 },
    rotationSmoothing: { minCutoff: 4.0, beta: 2.0, dCutoff: 1.0 },
    near: 0.01,
    far: 50,
    cubeSize: 0.1,
    showPlaneGrid: true,
  },
  planeTracking: {
    // Experimental (修正指示書 v2 §30 keeps PnP + fixed map as the main path);
    // `?planetrack=1` in the demo for comparison.
    enabled: false,
    minInliers: 12,
    goodInliers: 40,
    pnpHuberPx: 3,
    pnpInlierPx: 5,
    pnpMaxIterations: 10,
    probationFrames: 3,
    maxOutlierStreak: 3,
    liftMinInliers: 20,
    liftMaxMeanErrorPx: 1.5,
    minRayAngleDeg: 5,
    maxLiftDistanceRatio: 4,
    liftCooldownFrames: 10,
  },
  state: {
    minTrackedForTracking: 40,
    lostBelow: 20,
    lostFrameTolerance: 3,
    mapLostFrameTolerance: 3,
  },
  debug: {
    log: false,
    logIntervalMs: 1000,
    overlay: true,
    poseStaleMs: 100,
  },
  useWorker: true,
};

type DeepPartial<T> = { [K in keyof T]?: T[K] extends object ? DeepPartial<T[K]> : T[K] };

export type PartialARConfig = DeepPartial<ARConfig>;

/** Merge a partial override on top of the defaults (one level deep per section). */
export function resolveConfig(overrides?: PartialARConfig): ARConfig {
  if (!overrides) return structuredCloneConfig(DEFAULT_CONFIG);
  const base = structuredCloneConfig(DEFAULT_CONFIG);
  return {
    processing: { ...base.processing, ...overrides.processing },
    features: { ...base.features, ...overrides.features },
    tracker: { ...base.tracker, ...overrides.tracker },
    ransac: { ...base.ransac, ...overrides.ransac },
    pose: { ...base.pose, ...overrides.pose },
    landmarks: { ...base.landmarks, ...overrides.landmarks },
    plane: { ...base.plane, ...overrides.plane },
    planeTracking: { ...base.planeTracking, ...overrides.planeTracking },
    relocalization: { ...base.relocalization, ...overrides.relocalization },
    world: {
      ...base.world,
      ...overrides.world,
      positionSmoothing: { ...base.world.positionSmoothing, ...overrides.world?.positionSmoothing },
      rotationSmoothing: { ...base.world.rotationSmoothing, ...overrides.world?.rotationSmoothing },
    },
    state: { ...base.state, ...overrides.state },
    debug: { ...base.debug, ...overrides.debug },
    useWorker: overrides.useWorker ?? base.useWorker,
  };
}

function structuredCloneConfig(c: ARConfig): ARConfig {
  return {
    processing: { ...c.processing },
    features: { ...c.features },
    tracker: { ...c.tracker },
    ransac: { ...c.ransac },
    pose: { ...c.pose },
    landmarks: { ...c.landmarks },
    plane: { ...c.plane },
    planeTracking: { ...c.planeTracking },
    relocalization: { ...c.relocalization },
    world: { ...c.world, positionSmoothing: { ...c.world.positionSmoothing }, rotationSmoothing: { ...c.world.rotationSmoothing } },
    state: { ...c.state },
    debug: { ...c.debug },
    useWorker: c.useWorker,
  };
}
