/**
 * Tracking state machine (spec §31, 修正指示書 v10 §3–§4).
 *
 *   INITIALIZING → SEARCHING_FEATURES → TRACKING → PLANE_DETECTING → PLANE_FOUND → AR_ACTIVE
 *                                          ↑            │                  │
 *                                          └────────────┘ (map lost, no    ├→ TRACKING_LOST → RELOCALIZING → PLANE_FOUND
 *                                            world yet: keep scanning)     └→ RELOCALIZING ──────────────────┘
 *
 * In v10 terms: INITIALIZING / SEARCHING_FEATURES = INITIAL_SCAN, TRACKING /
 * PLANE_DETECTING = SURFACE_SCAN (PLANE_CANDIDATE while a plane candidate
 * exists), PLANE_FOUND / AR_ACTIVE = WORLD_TRACKING, TRACKING_LOST /
 * RELOCALIZING with an established world = WORLD_LOST / RELOCALIZING.
 * RELOCALIZING is reachable only once `worldEstablished` is true.
 */
export enum TrackingState {
  INITIALIZING = "INITIALIZING",
  SEARCHING_FEATURES = "SEARCHING_FEATURES",
  TRACKING = "TRACKING",
  PLANE_DETECTING = "PLANE_DETECTING",
  PLANE_FOUND = "PLANE_FOUND",
  AR_ACTIVE = "AR_ACTIVE",
  TRACKING_LOST = "TRACKING_LOST",
  RELOCALIZING = "RELOCALIZING",
}

/** Error / status codes exposed to the application (spec §53). */
export enum ARErrorCode {
  CAMERA_PERMISSION_DENIED = "CAMERA_PERMISSION_DENIED",
  CAMERA_UNAVAILABLE = "CAMERA_UNAVAILABLE",
  INSUFFICIENT_FEATURES = "INSUFFICIENT_FEATURES",
  TRACKING_LOST = "TRACKING_LOST",
  PLANE_NOT_FOUND = "PLANE_NOT_FOUND",
}

export class ARError extends Error {
  constructor(
    public readonly code: ARErrorCode,
    message?: string,
    options?: { cause?: unknown },
  ) {
    super(message ?? code, options);
    this.name = "ARError";
  }
}

export interface StateMachineThresholds {
  minTrackedForTracking: number;
  lostBelow: number;
  lostFrameTolerance: number;
}

/** Per-frame observation fed to the state machine. */
export interface FrameObservation {
  /** Features tracked into the current frame and surviving RANSAC. */
  inlierCount: number;
  /** Features present in the current frame (tracked + newly detected). */
  featureCount: number;
  /** Landmark map initialized and plane search running (Phase 3). */
  mapInitialized?: boolean;
  /** A stable horizontal plane is available (Phase 3). */
  planeFound?: boolean;
  /** Map initialized but the camera pose could not be computed this frame (Phase 5). */
  mapLost?: boolean;
  /**
   * World tracking has been established for the current map (a plane was
   * found and the world anchored to it, 修正指示書 v10 §5–§7). Before that
   * a lost map is not something to return to: the user is still scanning
   * and may freely move elsewhere, so RELOCALIZING is never entered.
   */
  worldEstablished?: boolean;
}

/**
 * Deterministic state machine. Pure logic, no DOM, so it is unit-testable.
 */
export class TrackingStateMachine {
  private _state: TrackingState = TrackingState.INITIALIZING;
  private badFrames = 0;

  constructor(private readonly thresholds: StateMachineThresholds) {}

  get state(): TrackingState {
    return this._state;
  }

  reset(): void {
    this._state = TrackingState.INITIALIZING;
    this.badFrames = 0;
  }

  /**
   * Advance the machine with the observation of one processed frame.
   * Returns the new state (may be unchanged).
   */
  update(obs: FrameObservation): TrackingState {
    const t = this.thresholds;
    switch (this._state) {
      case TrackingState.INITIALIZING:
        this._state = TrackingState.SEARCHING_FEATURES;
        this.badFrames = 0;
        // fall through intentionally handled by next frame
        return this._state;

      case TrackingState.SEARCHING_FEATURES:
      case TrackingState.TRACKING_LOST:
      case TrackingState.RELOCALIZING: {
        // Relocalization exists only to return to an *established* world
        // (v10 §2, §9): a lost map without a world is just scanning.
        const relocatable = !!obs.mapInitialized && !!obs.worldEstablished;
        if (obs.inlierCount >= t.minTrackedForTracking) {
          // Features track again. If an established world exists but the
          // camera is not yet located in it, we are relocalizing; otherwise
          // plain tracking (surface scan).
          this._state = relocatable && obs.mapLost ? TrackingState.RELOCALIZING : TrackingState.TRACKING;
          this.badFrames = 0;
        } else if (this._state === TrackingState.TRACKING_LOST) {
          // Nothing tracked for a while → searching (or relocalizing when an established world exists).
          this._state = relocatable ? TrackingState.RELOCALIZING : TrackingState.SEARCHING_FEATURES;
        } else if (this._state === TrackingState.RELOCALIZING && !relocatable) {
          this._state = TrackingState.SEARCHING_FEATURES;
        }
        return this._state;
      }

      case TrackingState.TRACKING:
      case TrackingState.PLANE_DETECTING:
      case TrackingState.PLANE_FOUND:
      case TrackingState.AR_ACTIVE:
        if (obs.inlierCount < t.lostBelow) {
          this.badFrames++;
          if (this.badFrames >= t.lostFrameTolerance) {
            this._state = TrackingState.TRACKING_LOST;
            this.badFrames = 0;
          }
          return this._state;
        }
        this.badFrames = 0;
        // Phase 5: features track but the camera is not located in the map.
        // Only an established world is worth relocalizing into (v10 §8–§10);
        // before that the scan simply continues where the camera looks now
        // (the engine re-initializes the map there).
        if (obs.mapInitialized && obs.mapLost) {
          this._state = obs.worldEstablished ? TrackingState.RELOCALIZING : TrackingState.TRACKING;
          return this._state;
        }
        // Phase 3 transitions driven by the landmark map / plane detector.
        if (this._state === TrackingState.AR_ACTIVE) {
          if (!obs.mapInitialized) this._state = TrackingState.TRACKING;
        } else if (obs.planeFound) {
          this._state = TrackingState.PLANE_FOUND;
        } else if (obs.mapInitialized) {
          if (this._state !== TrackingState.PLANE_FOUND) this._state = TrackingState.PLANE_DETECTING;
        } else {
          this._state = TrackingState.TRACKING;
        }
        return this._state;
    }
  }
}
