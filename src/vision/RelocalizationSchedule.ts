import type { RelocalizationConfig } from "../ar/ARConfig";
import type { MotionLevel } from "./types";

/**
 * Relocalization scheduling and bookkeeping (修正指示書 v14): pure functions
 * the engine uses to decide *when* to prepare, *when* to attempt, how to
 * treat a validated candidate and how to read an LK failure. Nothing here
 * touches the normal tracking, the PnP / validation thresholds or the
 * confirmation rules (v14 §1, §52).
 */

/**
 * Where the relocalization search of the current frame stands (v14 §24):
 * `idle` while tracking normally, `prepare` while tracking is weak and the
 * keyframe ranking is kept warm (§3–§6), `coarse` / `lk` / `pnp` /
 * `validation` = the furthest stage any keyframe reached in this attempt,
 * `confirming` while a validated candidate waits for its confirmation frame,
 * `applied` on the frame the pose was re-seeded.
 */
export type RelocalizationSearchStage = "idle" | "prepare" | "coarse" | "lk" | "pnp" | "validation" | "confirming" | "applied";

/**
 * Why keyframe → current LK failed to keep enough observations (v14 §49):
 * the dominant per-point status among the rejected points. `insufficient_tracks`
 * when no status dominates (or nothing was rejected but the count is short).
 */
export type LkFailureReason = "insufficient_tracks" | "high_fb_error" | "high_lk_error" | "out_of_bounds" | "diverged" | "low_texture" | "too_far";

/**
 * Per-status counts of one LK run (FeatureTracker statuses). `diverged` is
 * an out-of-bounds result from a start *inside* the image: the iterations
 * ran away, which is what a motion-blurred current frame does to a sharp
 * keyframe template; `outOfBounds` is a start already outside the image
 * (the coarse shift moved it out).
 */
export interface LkStatusCounts {
  ok: number;
  outOfBounds: number;
  diverged: number;
  lowTexture: number;
  highResidual: number;
  fbError: number;
  tooFar: number;
}

export function emptyLkStatusCounts(): LkStatusCounts {
  return { ok: 0, outOfBounds: 0, diverged: 0, lowTexture: 0, highResidual: 0, fbError: 0, tooFar: 0 };
}

/** Dominant failure of an LK run that kept too few points (v14 §49). */
export function lkFailureReason(c: LkStatusCounts): LkFailureReason {
  const entries: [LkFailureReason, number][] = [
    ["high_fb_error", c.fbError],
    ["high_lk_error", c.highResidual],
    ["out_of_bounds", c.outOfBounds],
    ["diverged", c.diverged],
    ["low_texture", c.lowTexture],
    ["too_far", c.tooFar],
  ];
  let best: [LkFailureReason, number] = ["insufficient_tracks", 0];
  let second = 0;
  for (const e of entries) {
    if (e[1] > best[1]) {
      second = best[1];
      best = e;
    } else if (e[1] > second) {
      second = e[1];
    }
  }
  // A clear majority among the rejections names the reason; otherwise the
  // run simply did not keep enough points.
  const rejected = c.fbError + c.highResidual + c.outOfBounds + c.diverged + c.lowTexture + c.tooFar;
  if (best[1] === 0 || best[1] * 2 < rejected) return "insufficient_tracks";
  return best[0];
}

/** What the engine knows about the current tracked frame (v14 §5). */
export interface RelocalizationPrepareInput {
  worldEstablished: boolean;
  /** Map PnP located the camera this frame. */
  tracked: boolean;
  /** PnP inliers and mean reprojection error (px) of the accepted candidate. */
  inliers: number;
  errorPx: number;
  motionLevel: MotionLevel;
}

/** Below / above these the located PnP counts as weak for preparation (v16). */
export interface RelocalizationPrepareThresholds {
  minInliers: number;
  maxErrorPx: number;
}

/**
 * Keep the relocalization warm while tracking is weak (v14 §3–§6): an
 * established world, a camera that *is* located this frame, and either a
 * weak PnP solve or fast motion. Preparation never replaces tracking and
 * never hides objects (v13 decides that); it only ranks the keyframes so the
 * first lost frame starts with a ranking.
 *
 * "Weak" (v16) is fewer than `minInliers` inliers or a mean error above
 * `maxErrorPx` — its own thresholds, not the `trusted` label: on Android
 * Chrome the reprojection error sits at 2.6–3.3 px during healthy tracking
 * (1.5 px is never reached, the trusted label is never set) and the
 * preparation ran every 10 frames for the whole session, 6–8 ms each.
 */
export function shouldPrepareRelocalization(i: RelocalizationPrepareInput, t: RelocalizationPrepareThresholds): boolean {
  if (!i.worldEstablished || !i.tracked) return false;
  const weak = i.inliers < t.minInliers || i.errorPx > t.maxErrorPx;
  return weak || i.motionLevel === "fast";
}

export interface RelocalizationAttemptInput {
  worldEstablished: boolean;
  /** Frames since the map PnP last located the camera (0 = located now). */
  framesSinceTracked: number;
  startAfterLostFrames: number;
  attemptEveryNFrames: number;
  /** A validated candidate is waiting for confirmation (attempt every frame). */
  pending: boolean;
}

/**
 * Attempt a relocalization this frame (v14 §7): only into an established
 * world, only while the map PnP is not locating the camera, from
 * `startAfterLostFrames` on and then every `attemptEveryNFrames` — or every
 * frame while a candidate awaits confirmation. The state machine's own lost
 * hysteresis is untouched; fast motion with a healthy PnP never attempts.
 */
export function shouldAttemptRelocalization(i: RelocalizationAttemptInput): boolean {
  if (!i.worldEstablished || i.framesSinceTracked <= 0) return false;
  if (i.pending) return true;
  if (i.framesSinceTracked < i.startAfterLostFrames) return false;
  const every = Math.max(1, i.attemptEveryNFrames);
  return (i.framesSinceTracked - i.startAfterLostFrames) % every === 0;
}

/** A pending candidate older than `maxAgeFrames` frames is stale (v14 §43–§44). */
export function candidateStale(candidateFrameId: number, frameId: number, maxAgeFrames: number): boolean {
  return frameId - candidateFrameId > Math.max(0, maxAgeFrames);
}

/**
 * What to do with a validated candidate (v5 §10–§11, v12 §10–§11, v14 §18–§20):
 * apply at once only when it is strong and clearly high quality, or when no
 * confirmation is configured for its level; otherwise hold it for
 * confirmation. An acceptable candidate is always confirmed.
 */
export function candidateAction(
  level: "strong" | "acceptable",
  inliers: number,
  meanErrorPx: number,
  cfg: Pick<RelocalizationConfig, "confirmationFrames" | "immediateInliers" | "immediateMaxErrorPx">,
  requiredConfirmations: number,
): "apply" | "confirm" {
  if (requiredConfirmations <= 0) return "apply";
  if (level === "strong" && inliers >= cfg.immediateInliers && meanErrorPx <= cfg.immediateMaxErrorPx) return "apply";
  return "confirm";
}

/**
 * Timing of one lost episode (v14 §45–§46, §55): frame ids and milliseconds
 * since the loss at which each stage first succeeded; −1 until it happens.
 * Measured, not targeted (§56).
 */
export interface RelocalizationTimeline {
  lostFrame: number;
  lostTimestamp: number;
  firstAttemptFrame: number;
  firstAttemptMs: number;
  firstCoarseMatchFrame: number;
  firstCoarseMatchMs: number;
  firstLkSuccessFrame: number;
  firstLkSuccessMs: number;
  firstPnpSuccessFrame: number;
  firstPnpSuccessMs: number;
  validationSuccessFrame: number;
  validationSuccessMs: number;
  confirmationSuccessFrame: number;
  confirmationSuccessMs: number;
  /** Attempts made in this episode. */
  attempts: number;
}

export function emptyRelocalizationTimeline(lostFrame = -1, lostTimestamp = -1): RelocalizationTimeline {
  return {
    lostFrame,
    lostTimestamp,
    firstAttemptFrame: -1,
    firstAttemptMs: -1,
    firstCoarseMatchFrame: -1,
    firstCoarseMatchMs: -1,
    firstLkSuccessFrame: -1,
    firstLkSuccessMs: -1,
    firstPnpSuccessFrame: -1,
    firstPnpSuccessMs: -1,
    validationSuccessFrame: -1,
    validationSuccessMs: -1,
    confirmationSuccessFrame: -1,
    confirmationSuccessMs: -1,
    attempts: 0,
  };
}

/** Stage counters an attempt reports (subset of RelocalizationDiagnostics). */
export interface TimelineAttempt {
  coarsePassed: number;
  lkPassed: number;
  pnpPassed: number;
  validated: number;
}

/** Record the first success of each stage of an attempt made at `frameId` / `timestamp`. */
export function advanceTimeline(t: RelocalizationTimeline, a: TimelineAttempt, frameId: number, timestamp: number): void {
  const ms = t.lostTimestamp >= 0 ? Math.max(0, timestamp - t.lostTimestamp) : -1;
  t.attempts++;
  if (t.firstAttemptFrame < 0) {
    t.firstAttemptFrame = frameId;
    t.firstAttemptMs = ms;
  }
  if (a.coarsePassed > 0 && t.firstCoarseMatchFrame < 0) {
    t.firstCoarseMatchFrame = frameId;
    t.firstCoarseMatchMs = ms;
  }
  if (a.lkPassed > 0 && t.firstLkSuccessFrame < 0) {
    t.firstLkSuccessFrame = frameId;
    t.firstLkSuccessMs = ms;
  }
  if (a.pnpPassed > 0 && t.firstPnpSuccessFrame < 0) {
    t.firstPnpSuccessFrame = frameId;
    t.firstPnpSuccessMs = ms;
  }
  if (a.validated > 0 && t.validationSuccessFrame < 0) {
    t.validationSuccessFrame = frameId;
    t.validationSuccessMs = ms;
  }
}

/** Record the frame the relocalized pose was applied (confirmation success). */
export function completeTimeline(t: RelocalizationTimeline, frameId: number, timestamp: number): void {
  if (t.confirmationSuccessFrame >= 0) return;
  t.confirmationSuccessFrame = frameId;
  t.confirmationSuccessMs = t.lostTimestamp >= 0 ? Math.max(0, timestamp - t.lostTimestamp) : -1;
}
