import type { MotionLevel, PlaneRecoveryDiagnostics, PlaneRecoveryPhase, PlaneRecoveryReason } from "./types";

/**
 * Plane recovery after fast motion (修正指示書 v11 §1, §5–§8, §13–§14).
 *
 * On device, after a fast move the camera kept tracking against the map
 * (LK 284/285, PnP 42 inliers / 0.38 px, source MAP, lost 0 ms) while the
 * plane detector sat on an old candidate / too few re-seeded points and the
 * world was never established. Nothing about the camera is wrong in that
 * situation, so the fix is deliberately narrow:
 *
 *   fast motion && map healthy && no world yet
 *     → PlaneDetector.resetForRecovery()      (plane state only)
 *     → plane search re-seeded from the landmarks observed *now*
 *     → warm-up: the existing stability rule decides PLANE_FOUND
 *
 * The map, the canonical pose, the keyframes and the world are never touched
 * here (§7), relocalization is never entered because of a fast motion (§29,
 * §56-2), and no plane threshold changes (§20).
 *
 * This class is the pure decision part (unit-testable without images); the
 * engine applies its result. A recovery is a *pre-world* concept: once a
 * world exists the found plane is the fixed reference and a fast motion is
 * handled by the map tracking alone (§30).
 */
export interface PlaneRecoveryObservation {
  /** This frame's motion level (v7 classification; the fast-motion trigger). */
  motionLevel: MotionLevel;
  mapInitialized: boolean;
  /** Map PnP produced the canonical pose this frame (source map or plane, not propagated). */
  mapTracked: boolean;
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

export class PlaneRecovery {
  private _active = false;
  private _reason: PlaneRecoveryReason = null;
  private startedAt = 0;
  private _count = 0;

  get active(): boolean {
    return this._active;
  }

  get reason(): PlaneRecoveryReason {
    return this._reason;
  }

  /** Recoveries started for the current map. */
  get count(): number {
    return this._count;
  }

  /** Map health as the recovery sees it (v11 §6, §32): located, enough inliers, finite pose. */
  static mapHealthy(obs: Pick<PlaneRecoveryObservation, "mapInitialized" | "mapTracked" | "mapInliers" | "requiredInliers" | "poseFinite">): boolean {
    return obs.mapInitialized && obs.mapTracked && obs.mapInliers >= obs.requiredInliers && obs.poseFinite;
  }

  /**
   * Decide for one frame. Returns true when a recovery (re)starts in this
   * frame — the caller then resets the plane detector's state. A fast
   * motion while already recovering restarts it (nothing stable can have
   * formed during the motion anyway). Without a healthy map nothing starts:
   * a lost map is the map tracker's business (§5, §31).
   */
  update(obs: PlaneRecoveryObservation): boolean {
    if (obs.worldEstablished) {
      // §30: an established world keeps its plane; the map tracking alone
      // rides out the motion. Any pre-world recovery still active is over.
      this._active = false;
      this._reason = null;
      return false;
    }
    const trigger: PlaneRecoveryReason = obs.manual ? "manual" : obs.motionLevel === "fast" ? "fast_motion" : null;
    if (trigger === null || !PlaneRecovery.mapHealthy(obs)) return false;
    this._active = true;
    this._reason = trigger;
    this.startedAt = obs.timestamp;
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
  }

  elapsedMs(timestamp: number): number {
    return this._active ? Math.max(0, timestamp - this.startedAt) : 0;
  }

  /** Phase for the HUD / guidance (v11 §28, §45). */
  phase(motionLevel: MotionLevel, candidate: boolean): PlaneRecoveryPhase {
    if (!this._active) return "none";
    if (motionLevel === "fast") return "recovery";
    return candidate ? "candidate" : "warmup";
  }
}

export function emptyPlaneRecovery(): PlaneRecoveryDiagnostics {
  return {
    active: false,
    reason: null,
    phase: "none",
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
    stableFrames: 0,
    requiredStableFrames: 0,
    recoveryElapsedMs: 0,
    recoveries: 0,
  };
}
