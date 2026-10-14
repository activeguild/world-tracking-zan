import type { RelocalizationConfig } from "../ar/ARConfig";
import type { CameraIntrinsics } from "../camera/CameraIntrinsics";
import { refinePosePnP } from "../math/PnP";
import { rotationDistance, type RigidTransform } from "../math/Pose";
import { FeatureTracker, TrackStatus } from "./FeatureTracker";
import type { ImagePyramid } from "./ImagePyramid";
import { clonePyramid, type CoarseImage, downsampleToCoarse, type Keyframe, type KeyframeObservation } from "./Keyframe";
import type { LandmarkMap } from "./LandmarkMap";
import { poseDelta } from "./PoseValidation";
import { emptyLkStatusCounts, type LkFailureReason, lkFailureReason, type LkStatusCounts, type RelocalizationSearchStage } from "./RelocalizationSchedule";
import type { Track } from "./types";

/**
 * Keyframe store + relocalization (spec §35–§36, Phase 5).
 *
 * Relocalization strategy (short-term loss, the target use case):
 *
 *   1. rank *all* keyframes by coarse visual similarity (zero-mean NCC shift
 *      search on a 1/16 image, v14 §8–§12); keyframes in a retry cooldown
 *      whose view has not changed are skipped (v14 §21–§22)
 *   2. for the top K only: refine the shift on the 1/8 image (the coarse
 *      score `coarseMinScore` gates)
 *   3. pyramidal LK from the keyframe image to the current image for the
 *      keyframe's landmark observations, initialized with the coarse shift
 *      (relocalization-specific LK thresholds, v14 §28–§30)
 *   4. PnP (LM + Huber) from the keyframe pose on the surviving 2D–3D pairs
 *   5. global validation (inliers, error, inlier ratio, spatial distribution,
 *      NCC, jump) → candidate; the search stops at the first strong one
 *      (v14 §18); the caller confirms and applies it
 *
 * No descriptors are needed because the keyframe image itself is matched;
 * this covers blur, brief occlusion and the camera coming back to a view
 * it has seen, but not large viewpoint changes (deferred, spec §2).
 */

/** Structured reasons a relocalization candidate was not accepted (修正指示書 v5 §19). */
export type RelocalizationRejectCode =
  | "no_keyframes"
  | "low_match_score"
  | "insufficient_landmarks"
  | "lk_failed"
  | "insufficient_inliers"
  | "high_reprojection_error"
  | "low_inlier_ratio"
  | "poor_spatial_distribution"
  | "pose_jump"
  | "confirmation_failed"
  | "stale_candidate"
  | "retry_cooldown"
  | "invalid_pose";

/**
 * How far a keyframe candidate got in one attempt (v6 §1–§2, names made
 * unambiguous in v9 §29: no bare "error"). Validation order follows v12
 * §17: pose → inliers → ratio → spatial → ncc → jump → reprojection.
 */
export type RelocalizationStage =
  | "coarse" // rejected by the coarse NCC match
  | "landmarks" // too few of its landmarks are still in the map
  | "lk" // LK from the keyframe image failed
  | "pnp_inliers" // PnP ran but had too few inliers
  | "ratio" // inliers / tracked observations too low
  | "spatial" // inliers concentrated in too few image cells
  | "ncc" // coarse match score too low for the relaxed error range (v12)
  | "jump" // too far from the held pose (relocalization-specific limits, v12)
  | "reprojection" // every other condition fine, mean reprojection error above the relaxed bound
  | "invalid" // non-finite pose
  | "ok"; // passed the global validation

/** Which validation condition rejected a candidate that reached PnP (v9 §3, v12 §8). */
export type RelocValidationRejectReason =
  | "inliers"
  | "reprojection_error"
  | "inlier_ratio"
  | "spatial_distribution"
  | "ncc"
  | "pose_invalid"
  | "translation_jump"
  | "rotation_jump"
  | "confirmation"
  | "unknown";

/**
 * Validation level of a candidate (v12 §3–§7): `strong` = reprojection
 * error within the strict bound (`maxMeanErrorPx`, the classic acceptance),
 * `acceptable` = above it but within the relaxed bound
 * (`relaxedMeanErrorPx`) while every other condition — inliers, inlier
 * ratio, spatial distribution, coverage, coarse NCC, finite pose and the
 * relocalization-specific jump limits — passes; such a candidate is never
 * applied without a confirmation frame. `reject` otherwise.
 */
export type RelocValidationLevel = "strong" | "acceptable" | "reject";

/**
 * Every validation condition of one PnP candidate, evaluated independently
 * (v9 §4–§6): values, the thresholds they were measured against and a
 * PASS / FAIL per condition, so several failing conditions are all visible
 * rather than only the first one. Translation / rotation jump vs the held
 * pose are *diagnostic only* (v9 §16–§17): a correct return after a loss
 * can be far from the held pose, so they never fail a relocalization
 * candidate; only a non-finite pose does.
 */
export interface RelocValidationDiagnostics {
  inliers: number;
  requiredInliers: number;
  reprojectionErrorPx: number;
  /** Strict bound (`maxMeanErrorPx`). */
  maxReprojectionErrorPx: number;
  /** Relaxed bound (`relaxedMeanErrorPx`, v12 §4); equal to the strict bound when the relaxed range is disabled. */
  relaxedReprojectionErrorPx: number;
  /** Validation level (v12 §7). */
  level: RelocValidationLevel;
  inlierRatio: number;
  minInlierRatio: number;
  coveredCells: number;
  totalCells: number;
  minSpatialCells: number;
  /** Bounding box of the inliers / image area (0…1). */
  spatialCoverage: number;
  minSpatialCoverage: number;
  /** Coarse NCC score of the keyframe match and the score the relaxed range requires (v12 §5). */
  nccScore: number;
  requiredNccScore: number;
  /** Distance of the candidate pose from the held pose (map units / deg); NaN when unknown. */
  translationJump: number;
  rotationJumpDeg: number;
  /** Relocalization-specific jump limits (v12 §9); Infinity when disabled. */
  maxTranslationJump: number;
  maxRotationJumpDeg: number;
  inliersPassed: boolean;
  /** Strict reprojection condition (≤ `maxMeanErrorPx`). */
  reprojectionPassed: boolean;
  /** Same as `reprojectionPassed` under the v12 name. */
  reprojectionStrictOk: boolean;
  /** Error within the relaxed bound (a bound check only; the level says whether the other conditions carried it). */
  reprojectionRelaxedOk: boolean;
  ratioPassed: boolean;
  spatialPassed: boolean;
  coveragePassed: boolean;
  /** Coarse NCC condition; always true for a strong-range error (the coarse gate already applied), checked against `requiredNccScore` in the relaxed range. */
  nccPassed: boolean;
  translationJumpPassed: boolean;
  rotationJumpPassed: boolean;
  posePassed: boolean;
  passed: boolean;
  rejectReason: RelocValidationRejectReason | null;
}

/** Per-keyframe outcome of one attempt (internal diagnostics, v6 §2). */
export interface RelocalizationKeyframeTrial {
  keyframeId: number;
  stage: RelocalizationStage;
  /** Ranking score on the 1/16 image (v14 §10). */
  rankScore: number;
  /** Refined coarse NCC on the 1/8 image (−1 when the refinement did not run). */
  coarseScore: number;
  /** LK-tracked / observations (0 when LK did not run). */
  lkRatio: number;
  /** LK breakdown (v14 §49): observations tried, kept, per-status rejects and the dominant failure. */
  lkObservations: number;
  lkTracked: number;
  lkStatus: LkStatusCounts;
  lkFailureReason: LkFailureReason | null;
  inlierCount: number;
  meanReprojectionErrorPx: number;
  inlierRatio: number;
  spatialCells: number;
  spatialCoverage: number;
  /** Full validation breakdown when the candidate reached PnP, null before that. */
  validation: RelocValidationDiagnostics | null;
}

/** Thresholds of the relocalization validation (v9 §31); taken from RelocalizationConfig. */
export interface RelocValidationThresholds {
  minInliers: number;
  /** Strict reprojection bound. */
  maxMeanErrorPx: number;
  /**
   * Relaxed reprojection bound (v12 §4): a candidate above `maxMeanErrorPx`
   * but within this bound still passes — as `acceptable` — when every other
   * condition does, and then always goes through a confirmation frame.
   * Values ≤ `maxMeanErrorPx` (or undefined) disable the relaxed range.
   */
  relaxedMeanErrorPx?: number;
  /** Coarse NCC score the relaxed range requires (v12 §5); undefined = not checked. */
  relaxedMinMatchScore?: number;
  minInlierRatio: number;
  minSpatialCells: number;
  minSpatialCoverage: number;
  /**
   * Relocalization-specific jump limits (v12 §9) vs the held pose, map units
   * / degrees; undefined, 0 or Infinity = not checked. Generous by design:
   * a correct return after a long loss can be far from the held pose.
   */
  maxTranslationJump?: number;
  maxRotationJumpDeg?: number;
}

/** Measured values of one candidate that reached PnP. */
export interface RelocCandidateMeasures {
  inliers: number;
  reprojectionErrorPx: number;
  inlierRatio: number;
  coveredCells: number;
  spatialCoverage: number;
  poseFinite: boolean;
  /** Coarse NCC score of the keyframe match (v12 §5); undefined = unknown (treated as passing). */
  matchScore?: number;
  /** Candidate pose vs the held pose (map units / deg); undefined / NaN = unknown (treated as passing). */
  translationJump?: number;
  rotationJumpDeg?: number;
}

function limitOf(v: number | undefined): number {
  return v === undefined || !(v > 0) ? Number.POSITIVE_INFINITY : v;
}

/**
 * Relocalization validation (v9 §5–§6, v12 §3–§9, §17): every condition is
 * evaluated, none short-circuits, and the reject reason is the first
 * failing one in the order pose → inliers → inlier ratio → spatial /
 * coverage → NCC → translation jump → rotation jump → reprojection error.
 * The reprojection error decides the *level*: ≤ strict = `strong`, within
 * the relaxed bound with everything else passing = `acceptable`, otherwise
 * `reject`. The jump limits here are the relocalization-specific ones (v12
 * §9), not the normal-tracking gate.
 */
export function validateRelocalizationCandidate(m: RelocCandidateMeasures, t: RelocValidationThresholds): RelocValidationDiagnostics {
  const relaxedBound = Math.max(t.maxMeanErrorPx, t.relaxedMeanErrorPx ?? t.maxMeanErrorPx);
  const requiredNcc = t.relaxedMinMatchScore ?? Number.NEGATIVE_INFINITY;
  const maxTranslationJump = limitOf(t.maxTranslationJump);
  const maxRotationJumpDeg = limitOf(t.maxRotationJumpDeg);
  const translationJump = m.translationJump ?? NaN;
  const rotationJumpDeg = m.rotationJumpDeg ?? NaN;
  const nccScore = m.matchScore ?? NaN;

  const posePassed = m.poseFinite && Number.isFinite(m.reprojectionErrorPx);
  const inliersPassed = m.inliers >= t.minInliers;
  const ratioPassed = m.inlierRatio >= t.minInlierRatio;
  const spatialPassed = m.coveredCells >= t.minSpatialCells;
  const coveragePassed = m.spatialCoverage >= t.minSpatialCoverage;
  const translationJumpPassed = !Number.isFinite(translationJump) || translationJump <= maxTranslationJump;
  const rotationJumpPassed = !Number.isFinite(rotationJumpDeg) || rotationJumpDeg <= maxRotationJumpDeg;
  const reprojectionStrictOk = posePassed && m.reprojectionErrorPx <= t.maxMeanErrorPx;
  const reprojectionRelaxedOk = posePassed && m.reprojectionErrorPx <= relaxedBound;
  // The coarse gate (`coarseMinScore`) already admitted every candidate that
  // reached PnP; the relaxed error range asks for a better match on top.
  const nccPassed = reprojectionStrictOk || !Number.isFinite(nccScore) || nccScore >= requiredNcc;

  const othersPassed = posePassed && inliersPassed && ratioPassed && spatialPassed && coveragePassed && nccPassed && translationJumpPassed && rotationJumpPassed;
  const level: RelocValidationLevel = !othersPassed ? "reject" : reprojectionStrictOk ? "strong" : reprojectionRelaxedOk ? "acceptable" : "reject";
  const passed = level !== "reject";
  let rejectReason: RelocValidationRejectReason | null = null;
  if (!passed) {
    rejectReason = !posePassed
      ? "pose_invalid"
      : !inliersPassed
        ? "inliers"
        : !ratioPassed
          ? "inlier_ratio"
          : !spatialPassed || !coveragePassed
            ? "spatial_distribution"
            : !nccPassed
              ? "ncc"
              : !translationJumpPassed
                ? "translation_jump"
                : !rotationJumpPassed
                  ? "rotation_jump"
                  : !reprojectionRelaxedOk
                    ? "reprojection_error"
                    : "unknown";
  }
  return {
    inliers: m.inliers,
    requiredInliers: t.minInliers,
    reprojectionErrorPx: m.reprojectionErrorPx,
    maxReprojectionErrorPx: t.maxMeanErrorPx,
    relaxedReprojectionErrorPx: relaxedBound,
    level,
    inlierRatio: m.inlierRatio,
    minInlierRatio: t.minInlierRatio,
    coveredCells: m.coveredCells,
    totalCells: 9,
    minSpatialCells: t.minSpatialCells,
    spatialCoverage: m.spatialCoverage,
    minSpatialCoverage: t.minSpatialCoverage,
    nccScore,
    requiredNccScore: requiredNcc,
    translationJump,
    rotationJumpDeg,
    maxTranslationJump,
    maxRotationJumpDeg,
    inliersPassed,
    reprojectionPassed: reprojectionStrictOk,
    reprojectionStrictOk,
    reprojectionRelaxedOk,
    ratioPassed,
    spatialPassed,
    coveragePassed,
    nccPassed,
    translationJumpPassed,
    rotationJumpPassed,
    posePassed,
    passed,
    rejectReason,
  };
}

/**
 * Confirmation frames a validated candidate needs before it is applied
 * (v12 §10–§11): the configured count, but never fewer than one for an
 * `acceptable` candidate — its error is above the strict bound, so it must
 * be reproduced in the next frame whatever the immediate-apply rule or
 * `confirmationFrames` say.
 */
export function requiredConfirmations(level: RelocValidationLevel, configuredFrames: number): number {
  return level === "acceptable" ? Math.max(1, configuredFrames) : Math.max(0, configuredFrames);
}

/** Stage a candidate stopped at, from its validation (first failing condition). */
export function stageOfValidation(v: RelocValidationDiagnostics): RelocalizationStage {
  switch (v.rejectReason) {
    case null:
      return "ok";
    case "pose_invalid":
      return "invalid";
    case "inliers":
      return "pnp_inliers";
    case "reprojection_error":
      return "reprojection";
    case "inlier_ratio":
      return "ratio";
    case "spatial_distribution":
      return "spatial";
    case "ncc":
      return "ncc";
    case "translation_jump":
    case "rotation_jump":
      return "jump";
    default:
      return "invalid";
  }
}

/** Bounding box of the points / image area (0…1). */
export function spatialCoverageOf(xs: ArrayLike<number>, ys: ArrayLike<number>, n: number, width: number, height: number): number {
  if (n === 0) return 0;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = xs[i], y = ys[i];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  return Math.max(0, Math.min(1, ((maxX - minX) * (maxY - minY)) / (width * height)));
}

/**
 * Where the candidates of one attempt dropped out (v6 §1, §5), and the best
 * one — the candidate that progressed furthest (ties: more inliers) — even
 * when the attempt failed (v6 §3).
 */
export interface RelocalizationDiagnostics {
  /** Keyframes in the store at the time of the attempt. */
  keyframes: number;
  /**
   * Ranking of every keyframe by coarse similarity (v14 §8–§12, §46), best
   * first: which were selected for LK and which were held back by the
   * retry cooldown (§21–§22).
   */
  ranked: RankedKeyframe[];
  /** Keyframes sent to LK / PnP in this attempt (the search budget, §16). */
  lkCandidates: number;
  pnpCandidates: number;
  /** Keyframes skipped because they failed recently on an unchanged view (§21). */
  retrySuppressed: number;
  /** Keyframes with fewer live landmarks than `minInliers` (v15): ranked, never tried. */
  unusableKeyframes: number;
  /** Furthest stage any keyframe reached in this attempt (§24). */
  searchStage: RelocalizationSearchStage;
  /** The ranking came from a preparation made while still tracking (§3–§6, §33). */
  usedPreparedRanking: boolean;
  /**
   * The jump limits vs the held pose were in force for this attempt (v16):
   * false after `jumpLimitMaxLostFrames` lost frames (the jump is then
   * diagnostic only) or without a held pose.
   */
  jumpLimitsActive: boolean;
  candidatesTried: number;
  coarseTested: number;
  coarsePassed: number;
  lkTested: number;
  lkPassed: number;
  pnpTested: number;
  /** PnP with ≥ minInliers inliers. */
  pnpPassed: number;
  /** Passed every validation check. */
  validated: number;
  /** Of the validated ones, how many passed at the `acceptable` level only (v12). */
  validatedAcceptable: number;
  /** Rejected by the relaxed-range NCC requirement / the relocalization jump limits (v12). */
  nccRejected: number;
  jumpRejected: number;
  errorRejected: number;
  ratioRejected: number;
  spatialRejected: number;
  bestCoarseScore: number;
  /** Best candidate of the attempt (null when no keyframe was tried). */
  best: RelocalizationKeyframeTrial | null;
  trials: RelocalizationKeyframeTrial[];
  rejectCode: RelocalizationRejectCode | null;
  rejectReason: string | null;
}

/**
 * A relocalization *candidate* (v5 §4): the best keyframe match of this
 * attempt with every quality measure the validation looked at. `success`
 * means it passed the global validation (v5 §5–§7); the caller still decides
 * when to apply it (confirmation, v5 §10–§11).
 */
export interface RelocalizationResult {
  success: boolean;
  keyframeId: number;
  /** Pose in the map frame (also for a rejected candidate that got as far as PnP). */
  pose: RigidTransform | null;
  inlierCount: number;
  meanReprojectionErrorPx: number;
  /** Coarse NCC score of the keyframe match (−1…1). */
  matchScore: number;
  /** LK-tracked observations / keyframe observations. */
  lkRatio: number;
  /** PnP inliers / LK-tracked observations. */
  inlierRatio: number;
  /** Cells of a 3×3 grid over the image that contain PnP inliers (1…9). */
  spatialCells: number;
  /** Bounding box of the inliers / image area (v9 §13). */
  spatialCoverage: number;
  /** Full validation breakdown (null when the candidate did not reach PnP). */
  validation: RelocValidationDiagnostics | null;
  /** Validation level of the candidate (v12 §7), null before PnP. */
  level: RelocValidationLevel | null;
  /** Coarse shift found (level-0 pixels). */
  shiftX: number;
  shiftY: number;
  /** Relocalized observations to turn into live tracks (current-frame positions). */
  tracks: { landmarkId: number; x: number; y: number }[];
  candidatesTried: number;
  /** Why the attempt failed (best candidate's stage), null on success. */
  reason: string | null;
  rejectCode: RelocalizationRejectCode | null;
  /** Stage counters and per-keyframe outcomes of this attempt (v6). */
  diagnostics: RelocalizationDiagnostics;
}

// v12 §17 order: the later a candidate fails, the further it got.
const STAGE_RANK: Record<RelocalizationStage, number> = {
  coarse: 0,
  landmarks: 1,
  lk: 2,
  pnp_inliers: 3,
  invalid: 3,
  ratio: 4,
  spatial: 5,
  ncc: 6,
  jump: 7,
  reprojection: 8,
  ok: 9,
};

const STAGE_CODE: Record<RelocalizationStage, RelocalizationRejectCode | null> = {
  coarse: "low_match_score",
  landmarks: "insufficient_landmarks",
  lk: "lk_failed",
  pnp_inliers: "insufficient_inliers",
  invalid: "invalid_pose",
  ratio: "low_inlier_ratio",
  spatial: "poor_spatial_distribution",
  ncc: "low_match_score",
  jump: "pose_jump",
  reprojection: "high_reprojection_error",
  ok: null,
};

/** Number of occupied cells of a 3×3 grid over a width×height image. */
export function spatialCellCount(xs: ArrayLike<number>, ys: ArrayLike<number>, n: number, width: number, height: number): number {
  let mask = 0;
  for (let i = 0; i < n; i++) {
    const cx = Math.min(2, Math.max(0, Math.floor((xs[i] * 3) / width)));
    const cy = Math.min(2, Math.max(0, Math.floor((ys[i] * 3) / height)));
    mask |= 1 << (cy * 3 + cx);
  }
  let count = 0;
  for (let b = 0; b < 9; b++) if (mask & (1 << b)) count++;
  return count;
}

/** One keyframe in the ranking of an attempt (v14 §9–§12). */
export interface RankedKeyframe {
  keyframeId: number;
  /** Zero-mean NCC on the 1/16 image at the best shift (−1…1). */
  rankScore: number;
  /** Best 1/16 shift (its pixels). */
  dx: number;
  dy: number;
  /** Sent to the 1/8 refinement / LK in this attempt. */
  selected: boolean;
  /** Held back by the retry cooldown (failed recently, view unchanged). */
  suppressed: boolean;
  /** Observations whose landmarks are still in the map (v15); −1 when not evaluated. */
  alive: number;
  /** Fewer live landmarks than `minInliers`: cannot relocalize, never spends an LK slot (v15). */
  unusable: boolean;
}

export function emptyRelocalizationDiagnostics(keyframes = 0): RelocalizationDiagnostics {
  return {
    keyframes,
    ranked: [],
    lkCandidates: 0,
    pnpCandidates: 0,
    retrySuppressed: 0,
    unusableKeyframes: 0,
    searchStage: "coarse",
    usedPreparedRanking: false,
    jumpLimitsActive: false,
    candidatesTried: 0,
    coarseTested: 0,
    coarsePassed: 0,
    lkTested: 0,
    lkPassed: 0,
    pnpTested: 0,
    pnpPassed: 0,
    validated: 0,
    validatedAcceptable: 0,
    nccRejected: 0,
    jumpRejected: 0,
    errorRejected: 0,
    ratioRejected: 0,
    spatialRejected: 0,
    bestCoarseScore: -1,
    best: null,
    trials: [],
    rejectCode: null,
    rejectReason: null,
  };
}

/**
 * Rank trials (v9 §8): a validated candidate beats any rejected one; among
 * rejected ones the furthest stage, then more inliers, then lower
 * reprojection error, then wider coverage, then higher coarse score.
 */
function betterTrial(a: RelocalizationKeyframeTrial, b: RelocalizationKeyframeTrial | null): boolean {
  if (!b) return true;
  const ra = STAGE_RANK[a.stage];
  const rb = STAGE_RANK[b.stage];
  if (ra !== rb) return ra > rb;
  if (a.inlierCount !== b.inlierCount) return a.inlierCount > b.inlierCount;
  if (a.validation && b.validation && a.meanReprojectionErrorPx !== b.meanReprojectionErrorPx) {
    return a.meanReprojectionErrorPx < b.meanReprojectionErrorPx;
  }
  if (a.spatialCoverage !== b.spatialCoverage) return a.spatialCoverage > b.spatialCoverage;
  return a.coarseScore > b.coarseScore;
}

/** Options of one relocalization attempt (v14). */
export interface RelocalizeOptions {
  /** Current frame id (retry cooldowns, prepared-ranking age); omit to disable both. */
  frameId?: number;
  /** First attempt of a lost episode: may try `lkCandidatesFirstAttempt` keyframes (v14 §17). */
  firstAttempt?: boolean;
  /** Keyframe to rank first whatever its score — the pending candidate's keyframe (confirmation, v14 §33). */
  preferKeyframeId?: number;
  /**
   * Frames the map has been lost (v16): beyond `jumpLimitMaxLostFrames` the
   * held pose is stale and the jump limits vs it are diagnostic only.
   * Omitted = the held pose is treated as fresh.
   */
  lostFrames?: number;
}

/** Retry bookkeeping per keyframe (v14 §21–§22, §34). */
interface KeyframeRetryState {
  /** Frame of the last failed trial and the 1/16 image it failed on. */
  failedFrame: number;
  failedRank: CoarseImage;
  cooldownUntil: number;
  lastStage: RelocalizationStage;
  lastSuccessFrame: number;
}

/** A ranking kept from a tracked frame (v14 §3–§6, §33). */
export interface RelocalizationPreparation {
  frameId: number;
  ranked: RankedKeyframe[];
  rank: CoarseImage;
}

export class Relocalizer {
  readonly keyframes: Keyframe[] = [];
  private nextId = 1;
  private lastKeyframePose: RigidTransform | null = null;
  private lastKeyframeFrame = -Infinity;
  private readonly tracker: FeatureTracker;
  /** Half LK window (level-0 px) for the in-bounds check of the shifted observations. */
  private readonly lkHalf: number;
  private scratchCoarse: CoarseImage | null = null;
  private readonly retry = new Map<number, KeyframeRetryState>();
  private prepared: RelocalizationPreparation | null = null;

  constructor(
    private readonly config: RelocalizationConfig,
    trackerConfig: ConstructorParameters<typeof FeatureTracker>[0],
  ) {
    // Relocalization-specific LK (v14 §28–§30): keyframe → current frame
    // bridges seconds and a viewpoint change, so it gets its own FB / residual
    // bounds and more iterations; the engine's frame-to-frame tracker keeps
    // its configuration. Every check (FB, residual, texture, bounds, gate)
    // still runs.
    this.tracker = new FeatureTracker({
      ...trackerConfig,
      maxIterations: trackerConfig.maxIterations + 10,
      forwardBackwardThreshold: config.lkForwardBackwardPx > 0 ? config.lkForwardBackwardPx : trackerConfig.forwardBackwardThreshold,
      maxResidual: config.lkMaxResidual > 0 ? config.lkMaxResidual : trackerConfig.maxResidual,
    });
    this.lkHalf = ((trackerConfig.windowSize | 1) - 1) / 2;
  }

  reset(): void {
    this.keyframes.length = 0;
    this.nextId = 1;
    this.evictedCount = 0;
    this.lastKeyframePose = null;
    this.lastKeyframeFrame = -Infinity;
    this.retry.clear();
    this.prepared = null;
    this.referencedCache = null;
  }

  /** Ranking prepared on a tracked frame, if any (diagnostics). */
  get preparation(): RelocalizationPreparation | null {
    return this.prepared;
  }

  /** Retry state of a keyframe (tests / diagnostics). */
  retryStateOf(keyframeId: number): { cooldownUntil: number; lastStage: RelocalizationStage; lastSuccessFrame: number } | null {
    const s = this.retry.get(keyframeId);
    return s ? { cooldownUntil: s.cooldownUntil, lastStage: s.lastStage, lastSuccessFrame: s.lastSuccessFrame } : null;
  }

  get count(): number {
    return this.keyframes.length;
  }

  /**
   * Decide whether the current (tracked) frame should become a keyframe
   * (spec §35: movement / rotation / time since the last one).
   */
  shouldCreate(pose: RigidTransform, frameId: number, inliers: number, medianParallaxPx: number, sceneDepth = 0): boolean {
    const cfg = this.config;
    if (inliers < cfg.keyframeMinInliers) return false;
    if (!this.lastKeyframePose) return true;
    if (frameId - this.lastKeyframeFrame < cfg.keyframeMinFrameGap) return false;
    // Time-based refresh of the current view (lighting / exposure drift).
    if (frameId - this.lastKeyframeFrame >= cfg.keyframeMaxFrameGap) return true;
    const rot = (rotationDistance(this.lastKeyframePose.rotation, pose.rotation) * 180) / Math.PI;
    if (rot <= cfg.keyframeRotationDeg && medianParallaxPx <= cfg.keyframeParallaxPx) return false;
    // View coverage (v15): the view must also be new with respect to *every*
    // stored keyframe, not only the last one — a camera swinging back and
    // forth otherwise creates a near-duplicate every `keyframeMinFrameGap`
    // frames and the 8 slots churn through the same two views (keyframe ids
    // reached 56 on device while 8 were kept).
    return this.nearestKeyframeDistance(pose, sceneDepth) >= 1;
  }

  /**
   * Distance of a pose to the nearest stored keyframe in units of the
   * keyframe thresholds: max(rotation / keyframeRotationDeg, camera-center
   * translation / (keyframeTranslationDepthRatio × sceneDepth)); ≥ 1 means
   * "a different view". Infinity with no keyframes; the translation term is
   * skipped while the scene depth is unknown.
   */
  nearestKeyframeDistance(pose: RigidTransform, sceneDepth: number, except?: Keyframe): number {
    let best = Number.POSITIVE_INFINITY;
    for (const kf of this.keyframes) {
      if (kf === except) continue;
      const d = this.keyframeDistance(kf.pose, pose, sceneDepth);
      if (d < best) best = d;
    }
    return best;
  }

  private keyframeDistance(a: RigidTransform, b: RigidTransform, sceneDepth: number): number {
    const cfg = this.config;
    const d = poseDelta(a, b);
    const rot = d.rotationDeg / Math.max(1e-9, cfg.keyframeRotationDeg);
    const transLimit = sceneDepth > 0 && cfg.keyframeTranslationDepthRatio > 0 ? cfg.keyframeTranslationDepthRatio * sceneDepth : 0;
    const trans = transLimit > 0 ? d.translation / transLimit : 0;
    return Math.max(rot, trans);
  }

  /**
   * Make room for a new keyframe (v15): instead of the oldest, drop the most
   * *redundant* keyframe — the one closest to another stored keyframe (ties:
   * fewer observations whose landmarks are still in the map). The first
   * keyframe (map origin view) and the newest are never dropped, so the
   * store keeps covering distinct viewpoints rather than the last few
   * seconds of motion.
   */
  private evictRedundant(map: LandmarkMap | null, sceneDepth: number): Keyframe | null {
    const n = this.keyframes.length;
    if (n <= this.config.maxKeyframes) return null;
    let victim: Keyframe | null = null;
    let victimKey = Number.POSITIVE_INFINITY;
    let victimAlive = Number.POSITIVE_INFINITY;
    for (let i = 1; i < n - 1; i++) {
      const kf = this.keyframes[i];
      const alive = map ? this.aliveObservations(kf, map) : kf.observations.length;
      // A keyframe with fewer live landmarks than a relocalization needs can
      // never succeed: it is the most redundant whatever its viewpoint.
      const d = alive < this.config.minInliers ? -1 : this.nearestKeyframeDistance(kf.pose, sceneDepth, kf);
      if (d < victimKey || (d === victimKey && alive < victimAlive)) {
        victim = kf;
        victimKey = d;
        victimAlive = alive;
      }
    }
    if (!victim) victim = this.keyframes[1];
    const idx = this.keyframes.indexOf(victim);
    this.keyframes.splice(idx, 1);
    this.retry.delete(victim.id);
    this.evictedCount++;
    this.referencedCache = null;
    return victim;
  }

  /** Keyframes created / evicted in this map (diagnostics). */
  get createdCount(): number {
    return this.nextId - 1;
  }
  private evictedCount = 0;
  get evictions(): number {
    return this.evictedCount;
  }

  private referencedCache: Set<number> | null = null;

  /**
   * Landmark ids any stored keyframe observes (v15): the map keeps these
   * alive past the age limit so that the keyframes stay usable for
   * relocalization. Cached until the keyframe set changes.
   */
  referencedLandmarkIds(): ReadonlySet<number> {
    if (!this.referencedCache) {
      const s = new Set<number>();
      for (const kf of this.keyframes) for (const o of kf.observations) s.add(o.landmarkId);
      this.referencedCache = s;
    }
    return this.referencedCache;
  }

  /** Observations of a keyframe whose landmarks are still in the map. */
  aliveObservations(kf: Keyframe, map: LandmarkMap): number {
    let n = 0;
    for (const o of kf.observations) if (map.get(o.landmarkId)) n++;
    return n;
  }

  /** Per-keyframe summary (diagnostics / tests): id, observations, observations still in the map. */
  summary(map: LandmarkMap): { id: number; frameId: number; observations: number; alive: number }[] {
    return this.keyframes.map((kf) => ({ id: kf.id, frameId: kf.frameId, observations: kf.observations.length, alive: this.aliveObservations(kf, map) }));
  }

  /**
   * Store a keyframe from the current pyramid, pose and landmark-linked
   * tracks. `map` / `sceneDepth` drive the redundancy-based eviction (v15);
   * without them the most redundant keyframe is judged by pose alone.
   */
  create(
    pyramid: ImagePyramid,
    pose: RigidTransform,
    tracks: readonly Track[],
    frameId: number,
    timestamp: number,
    map: LandmarkMap | null = null,
    sceneDepth = 0,
  ): Keyframe {
    const observations: KeyframeObservation[] = [];
    for (const t of tracks) {
      if (t.landmarkId >= 0) observations.push({ landmarkId: t.landmarkId, x: t.x, y: t.y });
    }
    const coarseSrc = pyramid.levels[Math.min(2, pyramid.levels.length - 1)];
    const coarse = downsampleToCoarse(coarseSrc);
    const kf: Keyframe = {
      id: this.nextId++,
      frameId,
      timestamp,
      pose: { rotation: Float64Array.from(pose.rotation), translation: Float64Array.from(pose.translation) },
      pyramid: clonePyramid(pyramid),
      coarse,
      rank: downsampleToCoarse(coarse),
      observations,
    };
    this.keyframes.push(kf);
    this.referencedCache = null;
    this.evictRedundant(map, sceneDepth);
    this.lastKeyframePose = kf.pose;
    this.lastKeyframeFrame = frameId;
    this.prepared = null;
    return kf;
  }

  /** 1/8 and 1/16 images of the current frame. */
  private coarseImagesOf(current: ImagePyramid): { coarse: CoarseImage; rank: CoarseImage } {
    const coarseSrc = current.levels[Math.min(2, current.levels.length - 1)];
    const coarse = downsampleToCoarse(coarseSrc);
    return { coarse, rank: downsampleToCoarse(coarse) };
  }

  /**
   * Rank every keyframe by coarse similarity to `rank` (the current 1/16
   * image), best first (v14 §8–§12). Nothing is excluded here: age, the lost
   * pose and earlier failures only show up as `selected` / `suppressed`
   * later. `preferKeyframeId` is put first regardless of its score.
   */
  private rankKeyframes(rank: CoarseImage, preferKeyframeId?: number): RankedKeyframe[] {
    const radius = this.config.rankSearchRadius;
    const ranked: RankedKeyframe[] = this.keyframes.map((kf) => {
      const s = coarseShift(kf.rank, rank, radius, this.config.coarseMinOverlap);
      return { keyframeId: kf.id, rankScore: s.score, dx: s.dx, dy: s.dy, selected: false, suppressed: false, alive: -1, unusable: false };
    });
    const recency = new Map<number, number>();
    this.keyframes.forEach((kf, i) => recency.set(kf.id, i));
    ranked.sort((a, b) => {
      if (a.keyframeId === preferKeyframeId) return -1;
      if (b.keyframeId === preferKeyframeId) return 1;
      if (a.rankScore !== b.rankScore) return b.rankScore - a.rankScore;
      return (recency.get(b.keyframeId) ?? 0) - (recency.get(a.keyframeId) ?? 0);
    });
    return ranked;
  }

  /**
   * Preparation while still tracking (v14 §3–§6, §33): rank the keyframes
   * against the current frame so that the first attempt after a loss starts
   * from a ranking instead of computing one. Cheap (1/16 images), no LK, no
   * PnP, no pose change.
   */
  prepare(current: ImagePyramid, frameId: number): RelocalizationPreparation | null {
    if (this.keyframes.length === 0) {
      this.prepared = null;
      return null;
    }
    const { rank } = this.coarseImagesOf(current);
    this.prepared = { frameId, ranked: this.rankKeyframes(rank), rank };
    return this.prepared;
  }

  /**
   * Whether the ranking prepared earlier still describes this frame: made
   * within the attempt period and on a view that has not changed (zero-shift
   * NCC between the two 1/16 images, v14 §33).
   */
  private preparedRankingFor(rank: CoarseImage, frameId: number | undefined): RankedKeyframe[] | null {
    const p = this.prepared;
    if (!p || frameId === undefined) return null;
    const age = frameId - p.frameId;
    if (age < 0 || age > Math.max(1, this.config.prepareEveryNFrames)) return null;
    if (coarseShift(p.rank, rank, 0).score < this.config.retryImageChangeScore) return null;
    // Copy with fresh selection flags; the cached scores order the candidates.
    return p.ranked.map((r) => ({ ...r, selected: false, suppressed: false, alive: -1, unusable: false }));
  }

  /**
   * Retry cooldown (v14 §21–§22, §34): a keyframe that failed within
   * `retryCooldownFrames` is skipped only while the view it failed on is
   * still what the camera sees; a changed image (or camera motion that
   * changes it) makes it eligible again at once.
   */
  private suppressed(kf: Keyframe, rank: CoarseImage, frameId: number | undefined): boolean {
    if (frameId === undefined) return false;
    const s = this.retry.get(kf.id);
    if (!s || frameId >= s.cooldownUntil) return false;
    return coarseShift(s.failedRank, rank, 0).score >= this.config.retryImageChangeScore;
  }

  private noteFailure(kf: Keyframe, rank: CoarseImage, frameId: number | undefined, stage: RelocalizationStage): void {
    if (frameId === undefined) return;
    const prev = this.retry.get(kf.id);
    this.retry.set(kf.id, {
      failedFrame: frameId,
      failedRank: rank,
      cooldownUntil: frameId + Math.max(0, this.config.retryCooldownFrames),
      lastStage: stage,
      lastSuccessFrame: prev?.lastSuccessFrame ?? -1,
    });
  }

  private noteSuccess(kf: Keyframe, frameId: number | undefined): void {
    if (frameId === undefined) return;
    const prev = this.retry.get(kf.id);
    this.retry.set(kf.id, {
      failedFrame: prev?.failedFrame ?? -1,
      failedRank: prev?.failedRank ?? kf.rank,
      cooldownUntil: -1,
      lastStage: "ok",
      lastSuccessFrame: frameId,
    });
  }

  /**
   * Try to relocalize the current frame against the stored keyframes.
   *
   * Every keyframe tried gets a trial record saying where it dropped out
   * (coarse → landmarks → lk → pnp → error → ratio → spatial → ok); the
   * stage counters and the best trial are returned with the result so that
   * a failing relocalization can be diagnosed without loosening anything
   * (v6 §1–§6).
   *
   * `heldPose` / `sceneDepth` (v12 §9) feed the relocalization-specific
   * jump limits: translation ≤ `maxTranslationJumpDepthRatio` × depth,
   * rotation ≤ `maxRotationJumpDeg`. Without them the jump is diagnostic.
   */
  relocalize(
    current: ImagePyramid,
    map: LandmarkMap,
    k: CameraIntrinsics,
    heldPose: RigidTransform | null = null,
    sceneDepth = 0,
    opts: RelocalizeOptions = {},
  ): RelocalizationResult {
    const cfg = this.config;
    const diag = emptyRelocalizationDiagnostics(this.keyframes.length);
    const fail: RelocalizationResult = {
      success: false, keyframeId: -1, pose: null, inlierCount: 0, meanReprojectionErrorPx: 0,
      matchScore: -1, lkRatio: 0, inlierRatio: 0, spatialCells: 0, spatialCoverage: 0, validation: null, level: null,
      shiftX: 0, shiftY: 0, tracks: [], candidatesTried: 0, reason: "no keyframes", rejectCode: "no_keyframes",
      diagnostics: diag,
    };
    if (this.keyframes.length === 0) {
      diag.rejectCode = "no_keyframes";
      diag.rejectReason = "no keyframes";
      return fail;
    }

    const { coarse: curCoarse, rank: curRank } = this.coarseImagesOf(current);
    this.scratchCoarse = curCoarse;
    const coarseScale = current.width / curCoarse.width;
    const f = (k.fx + k.fy) / 2;
    const frameId = opts.frameId;

    // ---- stage 0: rank every keyframe by coarse similarity (v14 §8–§12) ----
    // The ranking prepared on the last tracked frames is reused when the
    // view has not changed (§33); otherwise it is computed now. The lost /
    // held pose plays no part (§13–§15): visual similarity orders the
    // candidates, the pending candidate's keyframe is kept first so that its
    // confirmation is tried.
    const preparedRanking = this.preparedRankingFor(curRank, frameId);
    const ranked = preparedRanking ?? this.rankKeyframes(curRank, opts.preferKeyframeId);
    if (preparedRanking && opts.preferKeyframeId !== undefined) {
      const i = ranked.findIndex((r) => r.keyframeId === opts.preferKeyframeId);
      if (i > 0) ranked.unshift(...ranked.splice(i, 1));
    }
    diag.usedPreparedRanking = preparedRanking !== null;
    diag.ranked = ranked;
    // `bestCoarseScore` stays the best *refined* (1/8) NCC among the keyframes
    // tried, as before (v6); the ranking scores live in `ranked`.
    // Search budget (§16–§17): the top K not in a retry cooldown go to LK.
    const budget = Math.max(1, opts.firstAttempt ? cfg.lkCandidatesFirstAttempt : cfg.maxLkCandidatesPerFrame);
    const byId = new Map<number, Keyframe>();
    for (const kf of this.keyframes) byId.set(kf.id, kf);
    const candidates: { kf: Keyframe; rank: RankedKeyframe }[] = [];
    for (const r of ranked) {
      if (candidates.length >= budget) break;
      const kf = byId.get(r.keyframeId);
      if (!kf) continue;
      // v15: a keyframe whose landmarks have left the map cannot produce
      // `minInliers` correspondences — it stays visible in the ranking but
      // does not spend an LK slot (on device the NCC-0.91 top candidate had
      // 26 live observations and the attempt was wasted on it).
      r.alive = this.aliveObservations(kf, map);
      if (r.alive < cfg.minInliers) {
        r.unusable = true;
        diag.unusableKeyframes++;
        continue;
      }
      if (this.suppressed(kf, curRank, frameId)) {
        r.suppressed = true;
        diag.retrySuppressed++;
        continue;
      }
      r.selected = true;
      candidates.push({ kf, rank: r });
    }
    diag.lkCandidates = candidates.length;
    if (candidates.length === 0) {
      if (diag.unusableKeyframes === ranked.length) {
        diag.rejectCode = "insufficient_landmarks";
        diag.rejectReason = `all ${ranked.length} keyframes have fewer than ${cfg.minInliers} landmarks left in the map`;
        return { ...fail, candidatesTried: 0, reason: diag.rejectReason, rejectCode: "insufficient_landmarks", diagnostics: diag };
      }
      diag.rejectCode = "retry_cooldown";
      diag.rejectReason = `all ${ranked.length} keyframes in retry cooldown (view unchanged)`;
      return { ...fail, candidatesTried: 0, reason: diag.rejectReason, rejectCode: "retry_cooldown", diagnostics: diag };
    }

    let best: RelocalizationResult | null = null;
    // The rejected candidate that got furthest, with its reason.
    let rejected: { result: RelocalizationResult; trial: RelocalizationKeyframeTrial } | null = null;

    const record = (trial: RelocalizationKeyframeTrial, partial: Partial<RelocalizationResult>, reason: string | null) => {
      diag.trials.push(trial);
      if (betterTrial(trial, diag.best)) diag.best = trial;
      if (trial.stage !== "ok" && (!rejected || betterTrial(trial, rejected.trial))) {
        rejected = {
          trial,
          result: { ...fail, ...partial, success: false, keyframeId: trial.keyframeId, reason, rejectCode: STAGE_CODE[trial.stage] },
        };
      }
    };

    for (const { kf, rank } of candidates) {
      diag.candidatesTried++;
      const trial: RelocalizationKeyframeTrial = {
        keyframeId: kf.id,
        stage: "coarse",
        rankScore: rank.rankScore,
        coarseScore: -1,
        lkRatio: 0,
        lkObservations: 0,
        lkTracked: 0,
        lkStatus: emptyLkStatusCounts(),
        lkFailureReason: null,
        inlierCount: 0,
        meanReprojectionErrorPx: 0,
        inlierRatio: 0,
        spatialCells: 0,
        spatialCoverage: 0,
        validation: null,
      };

      // ---- stage 1: coarse NCC alignment, refined around the ranked shift (v14 §10) ----
      // A prepared ranking orders the candidates, but the shift it found is
      // a frame or two old: re-measure it for the selected keyframes so the
      // 1/8 refinement starts from this frame's alignment.
      if (preparedRanking) {
        const fresh = coarseShift(kf.rank, curRank, cfg.rankSearchRadius, cfg.coarseMinOverlap);
        rank.dx = fresh.dx;
        rank.dy = fresh.dy;
        rank.rankScore = fresh.score;
        trial.rankScore = fresh.score;
      }
      diag.coarseTested++;
      const shift = coarseShiftAround(kf.coarse, curCoarse, rank.dx * 2, rank.dy * 2, cfg.coarseRefineRadius, cfg.coarseMinOverlap);
      trial.coarseScore = shift.score;
      diag.bestCoarseScore = Math.max(diag.bestCoarseScore, shift.score);
      if (shift.score < cfg.coarseMinScore) {
        this.noteFailure(kf, curRank, frameId, "coarse");
        record(trial, { matchScore: shift.score }, `score ${shift.score.toFixed(2)} < ${cfg.coarseMinScore} (kf ${kf.id})`);
        continue;
      }
      diag.coarsePassed++;
      const sx = shift.dx * coarseScale;
      const sy = shift.dy * coarseScale;

      // ---- stage 2: LK from keyframe → current with the coarse shift as the initial guess (v14 §26–§27) ----
      // The keyframe observation positions are *not* used as the start in
      // the current frame: each starts at its position plus the global
      // coarse shift, and the displacement gate is measured from there.
      const obs = kf.observations.filter((o) => map.get(o.landmarkId) !== undefined);
      const n = obs.length;
      if (n < cfg.minInliers) {
        trial.stage = "landmarks";
        this.noteFailure(kf, curRank, frameId, "landmarks");
        record(trial, { matchScore: shift.score }, `kf ${kf.id}: ${n} landmarks left < ${cfg.minInliers}`);
        continue;
      }
      const pts = new Float32Array(n * 2);
      const guesses = new Float32Array(n * 2);
      let inBounds = 0;
      const margin = this.lkHalf + 2;
      for (let i = 0; i < n; i++) {
        pts[i * 2] = obs[i].x;
        pts[i * 2 + 1] = obs[i].y;
        const gx = obs[i].x + sx;
        const gy = obs[i].y + sy;
        guesses[i * 2] = gx;
        guesses[i * 2 + 1] = gy;
        if (gx >= margin && gy >= margin && gx < current.width - margin && gy < current.height - margin) inBounds++;
      }
      trial.lkObservations = n;
      if (inBounds < cfg.minInliers) {
        // v14 follow-up: the shift leaves too few observations inside the
        // image for a validation to be possible — do not spend the LK. When
        // most observations are still inside, the shortfall is the keyframe's
        // thin landmark set, not the shift (v15: `landmarks` stage).
        const outside = n - inBounds;
        trial.lkStatus.outOfBounds = outside;
        if (outside * 2 < n) {
          trial.stage = "landmarks";
          this.noteFailure(kf, curRank, frameId, "landmarks");
          record(trial, { matchScore: shift.score }, `kf ${kf.id}: only ${inBounds} usable observations (${n} with landmarks, ${outside} outside the image) < ${cfg.minInliers}`);
          continue;
        }
        trial.stage = "lk";
        trial.lkFailureReason = "out_of_bounds";
        this.noteFailure(kf, curRank, frameId, "lk");
        record(
          trial,
          { matchScore: shift.score },
          `kf ${kf.id}: only ${inBounds}/${n} observations inside the image after the coarse shift (${sx.toFixed(0)}, ${sy.toFixed(0)}) px < ${cfg.minInliers} (score ${shift.score.toFixed(2)}, overlap ${shift.overlap.toFixed(2)})`,
        );
        continue;
      }
      diag.lkTested++;
      const res = this.tracker.track(kf.pyramid, current, pts, n, undefined, guesses, cfg.lkMaxDisplacementPx);
      trial.lkObservations = n;
      trial.lkTracked = res.okCount;
      trial.lkRatio = res.okCount / n;
      // Per-status breakdown (v14 §49): says whether the keyframe, the initial
      // guess or the search window is what fails. A point whose start was
      // inside the image but that ended out of bounds *diverged* — the LK
      // iterations ran away, typically on a motion-blurred frame (on device
      // 154 of 179 points on a 36–80 px/frame blur) — and is counted apart
      // from a start outside the image.
      const st = trial.lkStatus;
      for (let i = 0; i < n; i++) {
        switch (res.status[i]) {
          case TrackStatus.OK: st.ok++; break;
          case TrackStatus.OUT_OF_BOUNDS: {
            const gx = guesses[i * 2];
            const gy = guesses[i * 2 + 1];
            const inside = gx >= margin && gy >= margin && gx < current.width - margin && gy < current.height - margin;
            if (inside) st.diverged++;
            else st.outOfBounds++;
            break;
          }
          case TrackStatus.LOW_TEXTURE: st.lowTexture++; break;
          case TrackStatus.HIGH_RESIDUAL: st.highResidual++; break;
          case TrackStatus.FB_ERROR: st.fbError++; break;
          case TrackStatus.TOO_FAR: st.tooFar++; break;
        }
      }
      if (res.okCount < cfg.minInliers) {
        trial.stage = "lk";
        trial.lkFailureReason = lkFailureReason(st);
        this.noteFailure(kf, curRank, frameId, "lk");
        record(
          trial,
          { matchScore: shift.score, lkRatio: trial.lkRatio },
          `kf ${kf.id}: lk ${res.okCount}/${n} < ${cfg.minInliers} (${trial.lkFailureReason}: fb ${st.fbError} res ${st.highResidual} far ${st.tooFar} div ${st.diverged} oob ${st.outOfBounds} tex ${st.lowTexture}; score ${shift.score.toFixed(2)})`,
        );
        continue;
      }
      diag.lkPassed++;

      // ---- stage 3: PnP from the keyframe pose (budget §16) ----
      if (diag.pnpCandidates >= Math.max(1, cfg.maxPnpCandidatesPerFrame)) {
        // Over budget for this frame: the keyframe stays eligible (no cooldown).
        trial.stage = "lk";
        record(trial, { matchScore: shift.score, lkRatio: trial.lkRatio }, `kf ${kf.id}: PnP budget (${cfg.maxPnpCandidatesPerFrame}) spent this frame`);
        continue;
      }
      diag.pnpCandidates++;
      diag.pnpTested++;
      const m = res.okCount;
      const pts3 = new Float64Array(m * 3);
      const ox = new Float64Array(m);
      const oy = new Float64Array(m);
      const idx: number[] = [];
      let j = 0;
      for (let i = 0; i < n; i++) {
        if (res.status[i] !== TrackStatus.OK) continue;
        const lm = map.get(obs[i].landmarkId)!;
        pts3[j * 3] = lm.position[0];
        pts3[j * 3 + 1] = lm.position[1];
        pts3[j * 3 + 2] = lm.position[2];
        ox[j] = (res.positions[i * 2] - k.cx) / k.fx;
        oy[j] = (res.positions[i * 2 + 1] - k.cy) / k.fy;
        idx.push(i);
        j++;
      }
      const pnp = refinePosePnP(kf.pose, pts3, ox, oy, m, {
        huber: cfg.pnpHuberPx / f,
        inlierThreshold: cfg.pnpInlierPx / f,
        maxIterations: 20,
        epsilon: 1e-7,
      });

      // ---- stage 4: global validation of the candidate (v5 §5–§7) ----
      // No single measure decides: PnP support, reprojection error, the
      // fraction of tracked observations the pose explains, and where the
      // inliers sit in the image (30 inliers in one corner pin the pose badly
      // and are typical of a repeated-pattern false match).
      const tracks: RelocalizationResult["tracks"] = [];
      const inX: number[] = [];
      const inY: number[] = [];
      for (let q = 0; q < m; q++) {
        if (!pnp.inliers[q]) continue;
        const i = idx[q];
        const x = res.positions[i * 2];
        const y = res.positions[i * 2 + 1];
        tracks.push({ landmarkId: obs[i].landmarkId, x, y });
        inX.push(x);
        inY.push(y);
      }
      const errPx = pnp.meanError * f;
      trial.inlierCount = pnp.inlierCount;
      trial.meanReprojectionErrorPx = errPx;
      trial.inlierRatio = pnp.inlierCount / m;
      trial.spatialCells = spatialCellCount(inX, inY, inX.length, current.width, current.height);
      trial.spatialCoverage = spatialCoverageOf(inX, inY, inX.length, current.width, current.height);
      // Every condition is evaluated (v9 §5–§6); the trial's stage is the
      // first failing one, the breakdown keeps all of them.
      const poseFinite =
        Array.from(pnp.pose.translation).every((v) => Number.isFinite(v)) && Array.from(pnp.pose.rotation).every((v) => Number.isFinite(v));
      const jump = heldPose && poseFinite ? poseDelta(pnp.pose, heldPose) : null;
      // v16: the held pose is dead-reckoned while lost; after a long loss it
      // says nothing about where the camera is (a full turn lost half-way
      // left it ~150° off on device), so the jump limits apply only while it
      // is fresh. The jump itself stays in the diagnostics.
      const heldFresh = opts.lostFrames === undefined || opts.lostFrames <= cfg.jumpLimitMaxLostFrames;
      diag.jumpLimitsActive = heldPose !== null && heldFresh;
      const validation = validateRelocalizationCandidate(
        {
          inliers: pnp.inlierCount,
          reprojectionErrorPx: errPx,
          inlierRatio: trial.inlierRatio,
          coveredCells: trial.spatialCells,
          spatialCoverage: trial.spatialCoverage,
          poseFinite,
          matchScore: shift.score,
          translationJump: jump?.translation,
          rotationJumpDeg: jump?.rotationDeg,
        },
        {
          minInliers: cfg.minInliers,
          maxMeanErrorPx: cfg.maxMeanErrorPx,
          relaxedMeanErrorPx: cfg.relaxedMeanErrorPx,
          relaxedMinMatchScore: cfg.relaxedMinMatchScore,
          minInlierRatio: cfg.minInlierRatio,
          minSpatialCells: cfg.minSpatialCells,
          minSpatialCoverage: cfg.minSpatialCoverage,
          maxTranslationJump: heldFresh && sceneDepth > 0 ? cfg.maxTranslationJumpDepthRatio * sceneDepth : undefined,
          maxRotationJumpDeg: heldFresh ? cfg.maxRotationJumpDeg : undefined,
        },
      );
      trial.validation = validation;
      trial.stage = stageOfValidation(validation);
      const candidate: RelocalizationResult = {
        success: false,
        keyframeId: kf.id,
        pose: pnp.pose,
        inlierCount: pnp.inlierCount,
        meanReprojectionErrorPx: errPx,
        matchScore: shift.score,
        lkRatio: trial.lkRatio,
        inlierRatio: trial.inlierRatio,
        spatialCells: trial.spatialCells,
        spatialCoverage: trial.spatialCoverage,
        validation,
        level: validation.level,
        shiftX: sx,
        shiftY: sy,
        tracks,
        candidatesTried: diag.candidatesTried,
        reason: null,
        rejectCode: null,
        diagnostics: diag,
      };
      if (validation.inliersPassed && validation.posePassed) diag.pnpPassed++;
      // "Rejected by the error" means beyond the relaxed bound (v12): an
      // acceptable candidate is not counted here.
      if (validation.posePassed && validation.inliersPassed && !validation.reprojectionRelaxedOk) diag.errorRejected++;
      if (!validation.ratioPassed) diag.ratioRejected++;
      if (!validation.spatialPassed || !validation.coveragePassed) diag.spatialRejected++;
      if (!validation.nccPassed) diag.nccRejected++;
      if (!validation.translationJumpPassed || !validation.rotationJumpPassed) diag.jumpRejected++;
      if (!validation.passed) {
        const relaxedNote =
          !validation.reprojectionStrictOk && validation.reprojectionRelaxedOk
            ? ` (err ${errPx.toFixed(2)} px in the relaxed range ≤ ${validation.relaxedReprojectionErrorPx}: every other condition required)`
            : "";
        const reason =
          validation.rejectReason === "pose_invalid"
            ? `kf ${kf.id}: invalid pose`
            : validation.rejectReason === "inliers"
              ? `kf ${kf.id}: pnp ${pnp.inlierCount}/${m} < ${cfg.minInliers}`
              : validation.rejectReason === "reprojection_error"
                ? `kf ${kf.id}: err ${errPx.toFixed(2)} > ${validation.relaxedReprojectionErrorPx} px relaxed (strict ${cfg.maxMeanErrorPx})`
                : validation.rejectReason === "inlier_ratio"
                  ? `kf ${kf.id}: inlier ratio ${trial.inlierRatio.toFixed(2)} < ${cfg.minInlierRatio}${relaxedNote}`
                  : validation.rejectReason === "ncc"
                    ? `kf ${kf.id}: ncc ${shift.score.toFixed(2)} < ${validation.requiredNccScore} for the relaxed error range${relaxedNote}`
                    : validation.rejectReason === "translation_jump"
                      ? `kf ${kf.id}: translation jump ${validation.translationJump.toFixed(3)} > ${validation.maxTranslationJump.toFixed(3)} (reloc limit)${relaxedNote}`
                      : validation.rejectReason === "rotation_jump"
                        ? `kf ${kf.id}: rotation jump ${validation.rotationJumpDeg.toFixed(1)} > ${validation.maxRotationJumpDeg}° (reloc limit)${relaxedNote}`
                        : `kf ${kf.id}: inliers in ${trial.spatialCells}/9 cells < ${cfg.minSpatialCells}, coverage ${trial.spatialCoverage.toFixed(2)} < ${cfg.minSpatialCoverage}${relaxedNote}`;
        this.noteFailure(kf, curRank, frameId, trial.stage);
        record(trial, candidate, reason);
        continue;
      }
      diag.validated++;
      if (validation.level === "acceptable") diag.validatedAcceptable++;
      this.noteSuccess(kf, frameId);
      record(trial, candidate, null);
      candidate.success = true;
      // Among validated candidates a strong one beats an acceptable one
      // (v12 §12); otherwise more inliers win, as before.
      const strongBeatsAcceptable = best !== null && best.level === "strong" && candidate.level === "acceptable";
      const acceptableLosesToStrong = best !== null && best.level === "acceptable" && candidate.level === "strong";
      if (!best || acceptableLosesToStrong || (!strongBeatsAcceptable && candidate.inlierCount > best.inlierCount)) best = candidate;
      // Early success (v14 §18): a strong candidate ends the search; an
      // acceptable one lets the remaining selected keyframes try for a
      // strong one (the confirmation still follows either way).
      if (candidate.level === "strong") break;
    }
    diag.searchStage = diag.validated > 0 ? "validation" : diag.pnpTested > 0 ? "pnp" : diag.lkTested > 0 ? "lk" : "coarse";
    if (best) {
      return { ...best, candidatesTried: diag.candidatesTried, diagnostics: diag };
    }
    const r = rejected as { result: RelocalizationResult; trial: RelocalizationKeyframeTrial } | null;
    if (r) {
      diag.rejectCode = r.result.rejectCode;
      diag.rejectReason = r.result.reason;
      return { ...r.result, candidatesTried: diag.candidatesTried, diagnostics: diag };
    }
    return { ...fail, candidatesTried: diag.candidatesTried, reason: "no candidates", diagnostics: diag };
  }

  /** Last coarse image of the current frame (debug). */
  get lastCoarse() {
    return this.scratchCoarse;
  }
}

/**
 * Best integer shift (dx, dy) such that b(x, y) ≈ a(x − dx, y − dy), by
 * zero-mean normalized cross-correlation over the overlapping region.
 * Images must have equal size.
 */
export function coarseShift(
  a: { width: number; height: number; data: Uint8Array; mean: number },
  b: { width: number; height: number; data: Uint8Array; mean: number },
  radius: number,
  minOverlap = 0.5,
): CoarseShiftResult {
  return coarseShiftAround(a, b, 0, 0, radius, minOverlap);
}

/** Result of a coarse shift search: the raw NCC at the chosen shift and the overlap it was measured on. */
export interface CoarseShiftResult {
  dx: number;
  dy: number;
  /** Zero-mean NCC over the overlapping region (−1…1). */
  score: number;
  /** Overlapping area / image area at that shift (0…1). */
  overlap: number;
  /** score × √overlap — what the search maximizes (v14 follow-up). */
  weightedScore: number;
}

/**
 * Same search restricted to shifts within `radius` of (cx, cy) — the 1/8
 * refinement around the shift found on the 1/16 ranking image (v14 §10).
 * Radius 0 evaluates the single shift (cx, cy).
 *
 * The shift is chosen by `score × √overlap`, not by the raw NCC: the NCC of
 * two unrelated patches has a standard deviation ∝ 1/√N, so a large shift
 * with a small overlap produces high *chance* maxima — on device that sent
 * 83 of 89 keyframe observations out of the image (`out_of_bounds`). The
 * weight normalizes the noise level across shifts; the raw NCC at the
 * chosen shift is still what `coarseMinScore` and v12's relaxed-range NCC
 * gate on. Shifts with less than `minOverlap` of the image overlapping per
 * axis are not considered.
 */
export function coarseShiftAround(
  a: { width: number; height: number; data: Uint8Array; mean: number },
  b: { width: number; height: number; data: Uint8Array; mean: number },
  cx: number,
  cy: number,
  radius: number,
  minOverlap = 0.5,
): CoarseShiftResult {
  const w = a.width;
  const h = a.height;
  let best: CoarseShiftResult = { dx: 0, dy: 0, score: -1, overlap: 0, weightedScore: -1 };
  const am = a.mean;
  const bm = b.mean;
  const minW = w * minOverlap;
  const minH = h * minOverlap;
  for (let dy = cy - radius; dy <= cy + radius; dy++) {
    const y0 = Math.max(0, dy);
    const y1 = Math.min(h, h + dy);
    for (let dx = cx - radius; dx <= cx + radius; dx++) {
      const x0 = Math.max(0, dx);
      const x1 = Math.min(w, w + dx);
      if (x1 - x0 < minW || y1 - y0 < minH) continue;
      let sab = 0;
      let saa = 0;
      let sbb = 0;
      for (let y = y0; y < y1; y++) {
        let ib = y * w + x0;
        let ia = (y - dy) * w + (x0 - dx);
        for (let x = x0; x < x1; x++, ia++, ib++) {
          const va = a.data[ia] - am;
          const vb = b.data[ib] - bm;
          sab += va * vb;
          saa += va * va;
          sbb += vb * vb;
        }
      }
      const denom = Math.sqrt(saa * sbb);
      if (denom < 1e-9) continue;
      const score = sab / denom;
      const overlap = ((x1 - x0) * (y1 - y0)) / (w * h);
      const weightedScore = score * Math.sqrt(overlap);
      if (weightedScore > best.weightedScore) best = { dx, dy, score, overlap, weightedScore };
    }
  }
  return best;
}
