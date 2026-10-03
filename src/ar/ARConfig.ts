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
    state: { ...c.state },
    debug: { ...c.debug },
    useWorker: c.useWorker,
  };
}
