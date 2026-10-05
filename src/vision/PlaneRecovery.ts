import type { MotionLevel, PlaneRecoveryDiagnostics, PlaneRecoveryReason, PlaneRecoveryState, PoseOutput } from "./types";

/**
 * Plane recovery after a significant camera motion (修正指示書 v11 §1,
 * §5–§8, §13–§14; v11.1 §3–§10, §30–§32).
 *
 * On device, after a fast move the camera kept tracking against the map
 * (LK 284/285, PnP 42 inliers / 0.38 px, source MAP, lost 0 ms) while the
 * plane detector sat on an old candidate / too few re-seeded points and the
 * world was never established. Nothing about the camera is wrong in that
 * situation, so the fix is deliberately narrow:
 *
 *   significant motion && map healthy && no world yet
 *     → recovery START: PlaneDetector.resetForRecovery()   (once, §3–§5)
 *     → plane search re-seeded from the landmarks observed *now*
 *     → warm-up: the existing stability rule decides PLANE_FOUND
 *
 * "Significant motion" (§6–§9) is either the v7 motion level `fast` or a
 * strong two-view motion event: the reference↔current parallax crossing
 * `pose.fullConfidenceParallaxPx` with a confident estimate (the on-device
 * `Motion MED 6.3px / 2view par 45px conf 1.00/1.00 n116` case). Parallax
 * alone never triggers (§8).
 *
 * While a recovery is active, further fast / two-view motion does *not*
 * restart it (§5, §31): the detector is reset exactly once per recovery so
 * the stability streak can accumulate through a shaky scan. The map, the
 * canonical pose, the keyframes and the world are never touched (§11–§12),
 * and relocalization is never entered because of a motion (§13).
 *
 * This class is the pure decision part (unit-testable without images); the
 * engine applies its result. A recovery is a *pre-world* concept: once a
 * world exists the found plane is the fixed reference and a fast motion is
 * ridden out by the map tracking alone (v11 §30).
 */
export interface PlaneRecoveryObservation {
  /** This frame's motion level (v7 classification; the fast-motion trigger). */
  motionLevel: MotionLevel;
  /** Significant two-view motion event this frame (see `significantTwoViewMotion`). */
  twoViewMotion: boolean;
  mapInitialized: boolean;
  /** Map PnP produced the canonical pose this frame (source map or plane, not propagated). */
  mapTracked: boolean;
  /** The camera has been lost in the map past the state machine's tolerance (ends a recovery, §32). */
  mapLost: boolean;
  /** Map PnP inliers this frame. */
  mapInliers: number;
  /** Inliers the map PnP needed this frame (`landmarks.minPnPInliers` while tracking). */
  requiredInliers: number;
  poseFinite: boolean;
  worldEstablished: boolean;
  /** Explicit request (tests, debugging). */
  manual?: boolean;
  timestamp: number;
}

/** Existing thresholds reused for the two-view trigger (v11.1 §7: no new constants). */
export interface TwoViewMotionThresholds {
  /** `pose.fullConfidenceParallaxPx`: parallax at which the two-view translation is fully trusted. */
  parallaxPx: number;
  /** `landmarks.initMinTranslationConfidence`: the confidence the map initialization itself requires. */
  minConfidence: number;
  /** `pose.minCorrespondences`: inliers the two-view estimate must rest on. */
  minInliers: number;
}

/**
 * A strong two-view motion *event* (v11.1 §7–§9): the reference↔current
 * parallax crosses the threshold in this frame (it was below it in the
 * previous frame of the same reference, or the reference was just renewed)
 * and the estimate is confident in both rotation and translation with
 * enough inliers. A level test would fire on every slow scan as soon as
 * the parallax has grown; the crossing marks the moment the viewpoint
 * changed a lot.
 */
export function significantTwoViewMotion(
  pose: Pick<PoseOutput, "parallaxPx" | "confidence" | "translationConfidence" | "inlierCount" | "referenceFrameId" | "model"> | null,
  previous: Pick<PoseOutput, "parallaxPx" | "referenceFrameId"> | null,
  t: TwoViewMotionThresholds,
): boolean {
  if (!pose || pose.model === "none" || pose.model === "rotation") return false;
  if (pose.parallaxPx < t.parallaxPx) return false;
  if (pose.confidence < t.minConfidence || pose.translationConfidence < t.minConfidence) return false;
  if (pose.inlierCount < t.minInliers) return false;
  const previousParallax = previous && previous.referenceFrameId === pose.referenceFrameId ? previous.parallaxPx : 0;
  return previousParallax < t.parallaxPx;
}

export class PlaneRecovery {
  private _active = false;
  private _reason: PlaneRecoveryReason = null;
  private startedAt = 0;
  private _count = 0;
  private _lastMotionTimestamp = 0;

  get active(): boolean {
    return this._active;
  }

  get reason(): PlaneRecoveryReason {
    return this._reason;
  }

  /** Recoveries started for the current map (= detector resets, one per start). */
  get count(): number {
    return this._count;
  }

  /** Timestamp of the last significant motion seen while active (diagnostic only). */
  get lastMotionTimestamp(): number {
    return this._lastMotionTimestamp;
  }

  /** Map health as the recovery sees it (v11 §6, §32): located, enough inliers, finite pose. */
  static mapHealthy(obs: Pick<PlaneRecoveryObservation, "mapInitialized" | "mapTracked" | "mapInliers" | "requiredInliers" | "poseFinite">): boolean {
    return obs.mapInitialized && obs.mapTracked && obs.mapInliers >= obs.requiredInliers && obs.poseFinite;
  }

  /** Trigger of this frame in priority order (v11.1 §10), null when none. */
  static trigger(obs: Pick<PlaneRecoveryObservation, "manual" | "motionLevel" | "twoViewMotion">): PlaneRecoveryReason {
    if (obs.manual) return "manual";
    if (obs.motionLevel === "fast") return "fast_motion";
    if (obs.twoViewMotion) return "two_view_motion";
    return null;
  }

  /**
   * Decide for one frame. Returns true only in the frame a recovery
   * STARTS (inactive → active, v11.1 §4): the caller resets the plane
   * detector then, and never again while the recovery is active (§5, §31).
   * Without a healthy map nothing starts: a lost map is the map tracker's
   * business (v11 §5, §31).
   */
  update(obs: PlaneRecoveryObservation): boolean {
    if (obs.worldEstablished) {
      // v11 §30: an established world keeps its plane; the map tracking
      // alone rides out the motion. Any pre-world recovery still active is over.
      this.finish();
      return false;
    }
    const trigger = PlaneRecovery.trigger(obs);
    if (this._active) {
      if (obs.mapLost) {
        // §32: a real loss is handled by the map / state machine, not here.
        this.finish();
        return false;
      }
      if (trigger !== null) this._lastMotionTimestamp = obs.timestamp;
      return false;
    }
    if (trigger === null || !PlaneRecovery.mapHealthy(obs)) return false;
    this._active = true;
    this._reason = trigger;
    this.startedAt = obs.timestamp;
    this._lastMotionTimestamp = obs.timestamp;
    this._count++;
    return true;
  }

  /** The plane was found: the recovery did its job. */
  finish(): void {
    this._active = false;
    this._reason = null;
  }

  /** Map reset / engine reset: nothing to recover. */
  reset(): void {
    this.finish();
    this._count = 0;
    this._lastMotionTimestamp = 0;
  }

  elapsedMs(timestamp: number): number {
    return this._active ? Math.max(0, timestamp - this.startedAt) : 0;
  }

  /**
   * Recovery state for the HUD / guidance (v11.1 §27): `starting` in the
   * start frame and while the camera still moves fast, `warmup` until the
   * current frame's search yields a candidate, `candidate` when it does,
   * `stable` once the stability streak has begun.
   */
  state(startedThisFrame: boolean, motionLevel: MotionLevel, candidateThisFrame: boolean, stableFrames: number): PlaneRecoveryState {
    if (!this._active) return "inactive";
    if (startedThisFrame || motionLevel === "fast") return "starting";
    if (!candidateThisFrame) return "warmup";
    return stableFrames > 0 ? "stable" : "candidate";
  }
}

export function emptyPlaneRecovery(): PlaneRecoveryDiagnostics {
  return {
    active: false,
    reason: null,
    state: "inactive",
    mapHealthy: false,
    mapInliers: 0,
    trackedFeatures: 0,
    seedCandidates: 0,
    seededPoints: 0,
    searchPoints: 0,
    bestInliers: 0,
    requiredInliers: 0,
    searchStage: "points",
    candidateFound: false,
    candidateCommitted: false,
    previousCandidateHeld: false,
    stableFrames: 0,
    requiredStableFrames: 0,
    recoveryElapsedMs: 0,
    recoveries: 0,
  };
}
