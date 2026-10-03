/**
 * Tracking state machine (spec §31).
 *
 * Phase 1 only exercises the first part of the machine:
 *
 *   INITIALIZING → SEARCHING_FEATURES → TRACKING → TRACKING_LOST → SEARCHING_FEATURES
 *
 * The later states are declared so that the public API is stable, but no
 * transition targets them yet (PLANE_DETECTING etc. arrive in Phase 3/4).
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
      case TrackingState.RELOCALIZING:
        if (obs.inlierCount >= t.minTrackedForTracking) {
          this._state = TrackingState.TRACKING;
          this.badFrames = 0;
        } else if (
          this._state === TrackingState.RELOCALIZING ||
          this._state === TrackingState.TRACKING_LOST
        ) {
          // Nothing tracked for a while → go back to searching.
          this._state = TrackingState.SEARCHING_FEATURES;
        }
        return this._state;

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
        } else {
          this.badFrames = 0;
        }
        return this._state;
    }
  }
}
