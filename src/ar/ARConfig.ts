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
}

export interface DebugConfig {
  /** Enable `[AR]` console logs (disabled in production builds by default). */
  log: boolean;
  /** Interval between log lines in milliseconds. */
  logIntervalMs: number;
  /** Draw features / motion vectors / HUD. */
  overlay: boolean;
}

export interface ARConfig {
  processing: ProcessingConfig;
  features: FeatureConfig;
  tracker: TrackerConfig;
  ransac: RansacConfig;
  pose: PoseConfig;
  landmarks: LandmarkConfig;
  plane: PlaneConfig;
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
    pyramidLevels: 3,
    windowSize: 15,
    maxIterations: 20,
    epsilon: 0.03,
    maxResidual: 30,
    minEigenvalue: 0.5,
    forwardBackwardThreshold: 1.0,
    maxDisplacement: 60,
  },
  ransac: {
    inlierThreshold: 3.0,
    confidence: 0.99,
    maxIterations: 200,
    minCorrespondences: 12,
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
    initMinParallaxPx: 20,
    initMinTranslationConfidence: 0.5,
    initMinLandmarks: 30,
    minTriangulationAngleDeg: 1.0,
    maxTriangulationErrorPx: 3.0,
    triangulateMinParallaxPx: 8,
    pnpHuberPx: 3.0,
    pnpInlierPx: 4.0,
    pnpMaxIterations: 10,
    minPnPInliers: 15,
    maxOutlierCount: 3,
    maxLandmarks: 1000,
    maxLandmarkAgeFrames: 150,
    lostResetFrames: 150,
  },
  relocalization: {
    maxKeyframes: 8,
    keyframeMinInliers: 30,
    keyframeMinFrameGap: 10,
    keyframeMaxFrameGap: 90,
    keyframeRotationDeg: 10,
    keyframeParallaxPx: 40,
    candidatesPerFrame: 3,
    coarseSearchRadius: 24,
    coarseMinScore: 0.3,
    lkMaxDisplacementPx: 40,
    pnpHuberPx: 3,
    pnpInlierPx: 4,
    minInliers: 20,
    goodInliers: 60,
    maxMeanErrorPx: 2.0,
    startAfterLostFrames: 1,
    attemptEveryNFrames: 1,
  },
  plane: {
    inlierThresholdRatio: 0.02,
    minInliers: 30,
    maxIterations: 200,
    confidence: 0.99,
    horizontalThreshold: 0.9,
    fallbackHorizontalThreshold: 0.5,
    stableAngleDeg: 5,
    stableCenterRatio: 0.1,
    stableFramesRequired: 5,
    lostFrames: 15,
    goodInlierCount: 80,
    minLandmarkObservations: 2,
  },
  world: {
    assumedPlaneDistanceMeters: 0.5,
    holdPoseOnLostMs: 1500,
    positionSmoothing: { minCutoff: 1.5, beta: 0.3, dCutoff: 1.0 },
    rotationSmoothing: { minCutoff: 1.5, beta: 0.5, dCutoff: 1.0 },
    near: 0.01,
    far: 50,
    cubeSize: 0.1,
    showPlaneGrid: true,
  },
  state: {
    minTrackedForTracking: 40,
    lostBelow: 20,
    lostFrameTolerance: 3,
  },
  debug: {
    log: false,
    logIntervalMs: 1000,
    overlay: true,
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
    relocalization: { ...c.relocalization },
    world: { ...c.world, positionSmoothing: { ...c.world.positionSmoothing }, rotationSmoothing: { ...c.world.rotationSmoothing } },
    state: { ...c.state },
    debug: { ...c.debug },
    useWorker: c.useWorker,
  };
}
