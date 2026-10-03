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
   * Seed LK with a constant-velocity prediction (previous frame's
   * displacement) so fast motion stays inside the pyramid's capture range.
   */
  predictMotion: boolean;
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
   * Pose jump gate (修正指示書 v2 §8). A PnP result is rejected (pose held, frame
   * counted as lost) when the camera center moves more than
   * max(jumpRejectDepthRatio × median landmark depth, jumpRejectSpeedFactor ×
   * previous frame's displacement) or rotates more than jumpRejectRotationDeg,
   * unless the solve is trusted (≥ jumpRejectTrustedInliers inliers and mean
   * error ≤ jumpRejectTrustedErrorPx): a well-supported pose is believed even
   * when the motion is fast.
   */
  jumpRejectDepthRatio: number;
  jumpRejectSpeedFactor: number;
  jumpRejectRotationDeg: number;
  jumpRejectTrustedInliers: number;
  jumpRejectTrustedErrorPx: number;
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
  /** Keyframe candidates tried per lost frame. */
  candidatesPerFrame: number;
  /** Coarse search radius (pixels of the coarse image ≈ level-0 / 8). */
  coarseSearchRadius: number;
  /** Minimum NCC score of the coarse alignment to proceed. */
  coarseMinScore: number;
  /** LK displacement gate around the coarse guess (px). */
  lkMaxDisplacementPx: number;
  pnpHuberPx: number;
  pnpInlierPx: number;
  /** Inliers needed to accept a relocalization. */
  minInliers: number;
  /** Stop trying more candidates once this many inliers are found. */
  goodInliers: number;
  /** Accept only when the mean reprojection error is below this (px). */
  maxMeanErrorPx: number;
  /** Frames the map may stay lost before a relocalization attempt starts. */
  startAfterLostFrames: number;
  /** Try to relocalize every N lost frames (cost control). */
  attemptEveryNFrames: number;
}

/** World anchoring, hit test and rendering (spec §25–§30, §33–§34, Phase 4). */
export interface WorldConfig {
  /** Assumed camera→plane distance (m) when the world is created; fixes the monocular scale. */
  assumedPlaneDistanceMeters: number;
  /** Keep rendering objects at the last pose for this long after tracking is lost (ms, spec §33). */
  holdPoseOnLostMs: number;
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
    maxTriangulationErrorPx: 4.0,
    triangulateMinParallaxPx: 6,
    pnpHuberPx: 4.0,
    pnpInlierPx: 6.0,
    pnpMaxIterations: 10,
    minPnPInliers: 12,
    maxOutlierCount: 5,
    maxLandmarks: 1000,
    maxLandmarkAgeFrames: 150,
    lostResetFrames: 150,
    refineParallaxGrowth: 1.3,
    enableLandmarkDepthRefinement: false,
    jumpRejectDepthRatio: 0.08,
    jumpRejectSpeedFactor: 3,
    jumpRejectRotationDeg: 20,
    jumpRejectTrustedInliers: 40,
    jumpRejectTrustedErrorPx: 1.5,
  },
  relocalization: {
    maxKeyframes: 8,
    keyframeMinInliers: 30,
    keyframeMinFrameGap: 10,
    keyframeMaxFrameGap: 90,
    keyframeRotationDeg: 10,
    keyframeParallaxPx: 40,
    candidatesPerFrame: 2,
    coarseSearchRadius: 24,
    coarseMinScore: 0.45,
    lkMaxDisplacementPx: 40,
    pnpHuberPx: 4,
    pnpInlierPx: 6,
    minInliers: 25,
    goodInliers: 60,
    maxMeanErrorPx: 1.5,
    startAfterLostFrames: 1,
    attemptEveryNFrames: 2,
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
  },
  world: {
    assumedPlaneDistanceMeters: 0.5,
    holdPoseOnLostMs: 1500,
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
