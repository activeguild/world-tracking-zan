/**
 * Relocalization bookkeeping of one map (v5 §10–§11, §16–§17, v3 §16, v10
 * §8–§12, v14 §43–§44), as a pure class: which candidate is waiting for its
 * confirmation, how many attempts failed since the camera was last located,
 * whether the post-relocalization monitor window is open, and when the lost
 * map is given up.
 *
 * It decides nothing about *what* a candidate is (the Relocalizer validates
 * it) nor where the camera is (the MapTracker counts the frames since it was
 * located): the engine feeds those in and does what the ledger says. One
 * frame is
 *
 *   begin → [settle, when begin says to attempt] → (map PnP) → afterTracking
 *         → [shouldResetMap → clear, when the PnP did not locate the camera]
 *
 * A loss is measured by the map PnP alone: applying a relocalization re-seeds
 * the tracker's pose (and its frames-since-tracked count), but the loss goes
 * on until the PnP follows, and an applied relocalization the PnP does not
 * follow in the same frame counts as a failed attempt. Otherwise a map the
 * PnP cannot use would never be given up while relocalizations keep being
 * applied to it, frame after frame.
 *
 * quint/tracking.qnt models exactly this sequence (`mapFrame`);
 * tests/quint/RelocalizationLedgerConformance.test.ts replays the model's
 * walks on this class.
 */
import { candidateStale, shouldAttemptRelocalization } from "./RelocalizationSchedule";

export interface RelocalizationLedgerConfig {
  startAfterLostFrames: number;
  attemptEveryNFrames: number;
  candidateMaxAgeFrames: number;
  postRelocMonitorFrames: number;
  minAttemptsBeforeReset: number;
  /** landmarks.lostResetFrames / preWorldLostResetFrames */
  lostResetFrames: number;
  preWorldLostResetFrames: number;
}

/** A validated candidate waiting to be reproduced. */
export interface PendingCandidate<T> {
  candidate: T;
  /** Frame it was found in. */
  frameId: number;
  confirmations: number;
}

/** What `begin` found at the start of a frame. */
export interface LedgerFrameStart<T> {
  /** Run a relocalization attempt this frame. */
  attempt: boolean;
  /** The candidate still waiting for its confirmation (after the stale check). */
  pending: PendingCandidate<T> | null;
  /** A pending candidate dropped because it outlived its age limit (counted as a failed attempt). */
  stale: PendingCandidate<T> | null;
}

/** What an attempt returned, as far as the bookkeeping cares. */
export interface LedgerCandidate<T> {
  candidate: T;
  /** Confirmations a candidate of this level needs (`requiredConfirmations`). */
  requiredConfirmations: number;
  /** May be applied without a confirmation (`candidateAction` = "apply"). */
  immediate: boolean;
  /** Lands where the pending candidate did (ignored without one). */
  nearPending: boolean;
}

export type LedgerOutcome =
  /** Re-seed the pose with this attempt's candidate now. */
  | "apply"
  /** The candidate waits (new, or one confirmation closer). */
  | "pending"
  /** There was a pending candidate and this attempt did not reproduce it: it produced none. */
  | "confirmation_lost"
  /** … or it produced one somewhere else (which starts its own confirmation). */
  | "confirmation_moved"
  /** No candidate, none was pending. */
  | "none";

export class RelocalizationLedger<T> {
  private _pending: PendingCandidate<T> | null = null;
  private _failedAttempts = 0;
  private _lostFrames = 0;
  /** Frames the monitor window stays open; null = no window. */
  private _monitorLeft: number | null = null;
  private appliedThisFrame = false;

  constructor(private readonly cfg: RelocalizationLedgerConfig) {}

  get pending(): PendingCandidate<T> | null {
    return this._pending;
  }

  /** Failed attempts since the camera was last located. */
  get failedAttempts(): number {
    return this._failedAttempts;
  }

  /** Frames since the map PnP located the camera (an applied relocalization does not reset it). */
  get lostFrames(): number {
    return this._lostFrames;
  }

  /** The post-relocalization monitor window is open. */
  get monitoring(): boolean {
    return this._monitorLeft !== null;
  }

  /**
   * Start of a frame with a map. `framesSinceTracked` is the tracker's count
   * before this frame's PnP (0 = located in the previous frame).
   */
  begin(frameId: number, framesSinceTracked: number, worldEstablished: boolean): LedgerFrameStart<T> {
    this.appliedThisFrame = false;
    // A candidate only lives while the camera is lost in an established world.
    if (framesSinceTracked === 0 || !worldEstablished) this._pending = null;
    let stale: PendingCandidate<T> | null = null;
    if (this._pending && candidateStale(this._pending.frameId, frameId, this.cfg.candidateMaxAgeFrames)) {
      stale = this._pending;
      this._pending = null;
      this._failedAttempts++;
    }
    const attempt = shouldAttemptRelocalization({
      worldEstablished,
      framesSinceTracked,
      startAfterLostFrames: this.cfg.startAfterLostFrames,
      attemptEveryNFrames: this.cfg.attemptEveryNFrames,
      pending: this._pending !== null,
    });
    return { attempt, pending: this._pending, stale };
  }

  /** The result of this frame's attempt (`null` = no validated candidate). */
  settle(frameId: number, c: LedgerCandidate<T> | null): LedgerOutcome {
    const pending = this._pending;
    if (!c) {
      this._failedAttempts++;
      this._pending = null;
      return pending ? "confirmation_lost" : "none";
    }
    let apply = false;
    let outcome: LedgerOutcome = "pending";
    if (pending) {
      if (c.nearPending) {
        pending.confirmations++;
        apply = pending.confirmations >= c.requiredConfirmations;
      } else {
        this._failedAttempts++;
        // The newer candidate starts its own confirmation.
        this._pending = { candidate: c.candidate, frameId, confirmations: 0 };
        outcome = "confirmation_moved";
      }
    } else if (c.immediate) {
      apply = true;
    } else {
      this._pending = { candidate: c.candidate, frameId, confirmations: 0 };
    }
    if (!apply) return outcome;
    this._pending = null;
    this._monitorLeft = this.cfg.postRelocMonitorFrames;
    this.appliedThisFrame = true;
    return "apply";
  }

  /**
   * After this frame's map PnP. `monitored`: the pose of this frame is a
   * relocalized one still being watched (the apply frame and the frames the
   * window is open); `watch`: compare this frame's PnP with the relocalized
   * pose (every monitored frame but the apply frame).
   */
  afterTracking(located: boolean): { monitored: boolean; watch: boolean } {
    const monitored = this._monitorLeft !== null;
    const watch = monitored && !this.appliedThisFrame;
    if (watch) {
      this._monitorLeft = this._monitorLeft! - 1;
      if (this._monitorLeft <= 0) this._monitorLeft = null;
    }
    if (located) {
      this._failedAttempts = 0;
      this._lostFrames = 0;
    } else {
      this._lostFrames++;
      // The relocalization was applied and the PnP did not follow it.
      if (this.appliedThisFrame) this._failedAttempts++;
    }
    return { monitored, watch };
  }

  /**
   * The map is given up (asked after `afterTracking`, when the PnP did not
   * locate the camera). In an established world it is the last resort: a
   * long loss and enough failed attempts. Before a world exists the lost map
   * is dropped quickly.
   */
  shouldResetMap(worldEstablished: boolean): boolean {
    return worldEstablished
      ? this._lostFrames > this.cfg.lostResetFrames && this._failedAttempts >= this.cfg.minAttemptsBeforeReset
      : this._lostFrames > this.cfg.preWorldLostResetFrames;
  }

  /** The map is gone (or the engine was reset). */
  clear(): void {
    this._pending = null;
    this._failedAttempts = 0;
    this._lostFrames = 0;
    this._monitorLeft = null;
  }
}
