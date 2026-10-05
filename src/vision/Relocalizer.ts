import type { RelocalizationConfig } from "../ar/ARConfig";
import type { CameraIntrinsics } from "../camera/CameraIntrinsics";
import { refinePosePnP } from "../math/PnP";
import { rotationDistance, type RigidTransform } from "../math/Pose";
import { FeatureTracker, TrackStatus } from "./FeatureTracker";
import type { ImagePyramid } from "./ImagePyramid";
import { clonePyramid, downsampleToCoarse, type Keyframe, type KeyframeObservation } from "./Keyframe";
import type { LandmarkMap } from "./LandmarkMap";
import type { Track } from "./types";

/**
 * Keyframe store + relocalization (spec §35–§36, Phase 5).
 *
 * Relocalization strategy (short-term loss, the target use case):
 *
 *   1. candidate keyframes, most recent first
 *   2. coarse global alignment: best integer shift between the keyframe's
 *      and the current frame's low-resolution images (zero-mean NCC)
 *   3. pyramidal LK from the keyframe image to the current image for the
 *      keyframe's landmark observations, initialized with the coarse shift
 *   4. PnP (LM + Huber) from the keyframe pose on the surviving 2D–3D pairs
 *   5. global validation (inliers, error, inlier ratio, spatial distribution)
 *      → candidate; the caller confirms and applies it
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
  | "confirmation_failed"
  | "invalid_pose";

/**
 * How far a keyframe candidate got in one attempt (v6 §1–§2, names made
 * unambiguous in v9 §29: no bare "error").
 */
export type RelocalizationStage =
  | "coarse" // rejected by the coarse NCC match
  | "landmarks" // too few of its landmarks are still in the map
  | "lk" // LK from the keyframe image failed
  | "pnp_inliers" // PnP ran but had too few inliers
  | "reprojection" // enough inliers, mean reprojection error too high
  | "ratio" // inliers / tracked observations too low
  | "spatial" // inliers concentrated in too few image cells
  | "invalid" // non-finite pose
  | "ok"; // passed the global validation

/** Which validation condition rejected a candidate that reached PnP (v9 §3). */
export type RelocValidationRejectReason =
  | "inliers"
  | "reprojection_error"
  | "inlier_ratio"
  | "spatial_distribution"
  | "pose_invalid"
  | "confirmation"
  | "unknown";

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
  maxReprojectionErrorPx: number;
  inlierRatio: number;
  minInlierRatio: number;
  coveredCells: number;
  totalCells: number;
  minSpatialCells: number;
  /** Bounding box of the inliers / image area (0…1). */
  spatialCoverage: number;
  minSpatialCoverage: number;
  /** Distance of the candidate pose from the held pose (map units / deg); NaN when unknown. */
  translationJump: number;
  rotationJumpDeg: number;
  inliersPassed: boolean;
  reprojectionPassed: boolean;
  ratioPassed: boolean;
  spatialPassed: boolean;
  coveragePassed: boolean;
  posePassed: boolean;
  passed: boolean;
  rejectReason: RelocValidationRejectReason | null;
}

/** Per-keyframe outcome of one attempt (internal diagnostics, v6 §2). */
export interface RelocalizationKeyframeTrial {
  keyframeId: number;
  stage: RelocalizationStage;
  coarseScore: number;
  /** LK-tracked / observations (0 when LK did not run). */
  lkRatio: number;
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
  maxMeanErrorPx: number;
  minInlierRatio: number;
  minSpatialCells: number;
  minSpatialCoverage: number;
}

/** Measured values of one candidate that reached PnP. */
export interface RelocCandidateMeasures {
  inliers: number;
  reprojectionErrorPx: number;
  inlierRatio: number;
  coveredCells: number;
  spatialCoverage: number;
  poseFinite: boolean;
  /** Optional, diagnostic only. */
  translationJump?: number;
  rotationJumpDeg?: number;
}

/**
 * Relocalization validation (v9 §5–§6): every condition is evaluated, none
 * short-circuits. The reject reason is the first failing condition in the
 * order pose → inliers → reprojection → ratio → spatial / coverage; the
 * per-condition flags keep the rest. The normal-tracking jump gate is *not*
 * part of this (v9 §16–§17).
 */
export function validateRelocalizationCandidate(m: RelocCandidateMeasures, t: RelocValidationThresholds): RelocValidationDiagnostics {
  const posePassed = m.poseFinite && Number.isFinite(m.reprojectionErrorPx);
  const inliersPassed = m.inliers >= t.minInliers;
  const reprojectionPassed = posePassed && m.reprojectionErrorPx <= t.maxMeanErrorPx;
  const ratioPassed = m.inlierRatio >= t.minInlierRatio;
  const spatialPassed = m.coveredCells >= t.minSpatialCells;
  const coveragePassed = m.spatialCoverage >= t.minSpatialCoverage;
  const passed = posePassed && inliersPassed && reprojectionPassed && ratioPassed && spatialPassed && coveragePassed;
  let rejectReason: RelocValidationRejectReason | null = null;
  if (!passed) {
    rejectReason = !posePassed
      ? "pose_invalid"
      : !inliersPassed
        ? "inliers"
        : !reprojectionPassed
          ? "reprojection_error"
          : !ratioPassed
            ? "inlier_ratio"
            : !spatialPassed || !coveragePassed
              ? "spatial_distribution"
              : "unknown";
  }
  return {
    inliers: m.inliers,
    requiredInliers: t.minInliers,
    reprojectionErrorPx: m.reprojectionErrorPx,
    maxReprojectionErrorPx: t.maxMeanErrorPx,
    inlierRatio: m.inlierRatio,
    minInlierRatio: t.minInlierRatio,
    coveredCells: m.coveredCells,
    totalCells: 9,
    minSpatialCells: t.minSpatialCells,
    spatialCoverage: m.spatialCoverage,
    minSpatialCoverage: t.minSpatialCoverage,
    translationJump: m.translationJump ?? NaN,
    rotationJumpDeg: m.rotationJumpDeg ?? NaN,
    inliersPassed,
    reprojectionPassed,
    ratioPassed,
    spatialPassed,
    coveragePassed,
    posePassed,
    passed,
    rejectReason,
  };
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

const STAGE_RANK: Record<RelocalizationStage, number> = {
  coarse: 0,
  landmarks: 1,
  lk: 2,
  pnp_inliers: 3,
  invalid: 3,
  reprojection: 4,
  ratio: 5,
  spatial: 6,
  ok: 7,
};

const STAGE_CODE: Record<RelocalizationStage, RelocalizationRejectCode | null> = {
  coarse: "low_match_score",
  landmarks: "insufficient_landmarks",
  lk: "lk_failed",
  pnp_inliers: "insufficient_inliers",
  invalid: "invalid_pose",
  reprojection: "high_reprojection_error",
  ratio: "low_inlier_ratio",
  spatial: "poor_spatial_distribution",
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

export function emptyRelocalizationDiagnostics(keyframes = 0): RelocalizationDiagnostics {
  return {
    keyframes,
    candidatesTried: 0,
    coarseTested: 0,
    coarsePassed: 0,
    lkTested: 0,
    lkPassed: 0,
    pnpTested: 0,
    pnpPassed: 0,
    validated: 0,
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

export class Relocalizer {
  readonly keyframes: Keyframe[] = [];
  private nextId = 1;
  private lastKeyframePose: RigidTransform | null = null;
  private lastKeyframeFrame = -Infinity;
  private readonly tracker: FeatureTracker;
  /** Round-robin cursor so that successive attempts cover every keyframe, not only the most recent ones. */
  private candidateCursor = 0;
  private scratchCoarse: { width: number; height: number; data: Uint8Array; mean: number; norm: number } | null = null;

  constructor(
    private readonly config: RelocalizationConfig,
    trackerConfig: ConstructorParameters<typeof FeatureTracker>[0],
  ) {
    this.tracker = new FeatureTracker({ ...trackerConfig, maxIterations: trackerConfig.maxIterations + 10 });
  }

  reset(): void {
    this.keyframes.length = 0;
    this.lastKeyframePose = null;
    this.lastKeyframeFrame = -Infinity;
    this.candidateCursor = 0;
  }

  get count(): number {
    return this.keyframes.length;
  }

  /**
   * Decide whether the current (tracked) frame should become a keyframe
   * (spec §35: movement / rotation / time since the last one).
   */
  shouldCreate(pose: RigidTransform, frameId: number, inliers: number, medianParallaxPx: number): boolean {
    const cfg = this.config;
    if (inliers < cfg.keyframeMinInliers) return false;
    if (!this.lastKeyframePose) return true;
    if (frameId - this.lastKeyframeFrame < cfg.keyframeMinFrameGap) return false;
    const rot = (rotationDistance(this.lastKeyframePose.rotation, pose.rotation) * 180) / Math.PI;
    if (rot > cfg.keyframeRotationDeg) return true;
    if (medianParallaxPx > cfg.keyframeParallaxPx) return true;
    return frameId - this.lastKeyframeFrame >= cfg.keyframeMaxFrameGap;
  }

  /** Store a keyframe from the current pyramid, pose and landmark-linked tracks. */
  create(pyramid: ImagePyramid, pose: RigidTransform, tracks: readonly Track[], frameId: number, timestamp: number): Keyframe {
    const observations: KeyframeObservation[] = [];
    for (const t of tracks) {
      if (t.landmarkId >= 0) observations.push({ landmarkId: t.landmarkId, x: t.x, y: t.y });
    }
    const coarseSrc = pyramid.levels[Math.min(2, pyramid.levels.length - 1)];
    const kf: Keyframe = {
      id: this.nextId++,
      frameId,
      timestamp,
      pose: { rotation: Float64Array.from(pose.rotation), translation: Float64Array.from(pose.translation) },
      pyramid: clonePyramid(pyramid),
      coarse: downsampleToCoarse(coarseSrc),
      observations,
    };
    this.keyframes.push(kf);
    if (this.keyframes.length > this.config.maxKeyframes) {
      // Drop the oldest, but always keep the first (map origin view).
      this.keyframes.splice(1, 1);
    }
    this.lastKeyframePose = kf.pose;
    this.lastKeyframeFrame = frameId;
    return kf;
  }

  /**
   * Try to relocalize the current frame against the stored keyframes.
   *
   * Every keyframe tried gets a trial record saying where it dropped out
   * (coarse → landmarks → lk → pnp → error → ratio → spatial → ok); the
   * stage counters and the best trial are returned with the result so that
   * a failing relocalization can be diagnosed without loosening anything
   * (v6 §1–§6).
   */
  relocalize(current: ImagePyramid, map: LandmarkMap, k: CameraIntrinsics): RelocalizationResult {
    const cfg = this.config;
    const diag = emptyRelocalizationDiagnostics(this.keyframes.length);
    const fail: RelocalizationResult = {
      success: false, keyframeId: -1, pose: null, inlierCount: 0, meanReprojectionErrorPx: 0,
      matchScore: -1, lkRatio: 0, inlierRatio: 0, spatialCells: 0, spatialCoverage: 0, validation: null,
      shiftX: 0, shiftY: 0, tracks: [], candidatesTried: 0, reason: "no keyframes", rejectCode: "no_keyframes",
      diagnostics: diag,
    };
    if (this.keyframes.length === 0) {
      diag.rejectCode = "no_keyframes";
      diag.rejectReason = "no keyframes";
      return fail;
    }

    const coarseSrc = current.levels[Math.min(2, current.levels.length - 1)];
    const curCoarse = downsampleToCoarse(coarseSrc);
    this.scratchCoarse = curCoarse;
    const coarseScale = current.width / curCoarse.width;
    const f = (k.fx + k.fy) / 2;

    // Candidates: most recent first, then round-robin through the rest on
    // the following attempts so that the view the camera returns to is
    // eventually tried even when it is an old keyframe.
    const ordered = [...this.keyframes].reverse();
    const candidates: Keyframe[] = [];
    for (let i = 0; i < Math.min(cfg.candidatesPerFrame, ordered.length); i++) {
      candidates.push(ordered[(this.candidateCursor + i) % ordered.length]);
    }
    this.candidateCursor = (this.candidateCursor + candidates.length) % Math.max(1, ordered.length);

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

    for (const kf of candidates) {
      diag.candidatesTried++;
      const trial: RelocalizationKeyframeTrial = {
        keyframeId: kf.id,
        stage: "coarse",
        coarseScore: -1,
        lkRatio: 0,
        inlierCount: 0,
        meanReprojectionErrorPx: 0,
        inlierRatio: 0,
        spatialCells: 0,
        spatialCoverage: 0,
        validation: null,
      };

      // ---- stage 1: coarse NCC alignment ----
      diag.coarseTested++;
      const shift = coarseShift(kf.coarse, curCoarse, cfg.coarseSearchRadius);
      trial.coarseScore = shift.score;
      diag.bestCoarseScore = Math.max(diag.bestCoarseScore, shift.score);
      if (shift.score < cfg.coarseMinScore) {
        record(trial, { matchScore: shift.score }, `score ${shift.score.toFixed(2)} < ${cfg.coarseMinScore} (kf ${kf.id})`);
        continue;
      }
      diag.coarsePassed++;
      const sx = shift.dx * coarseScale;
      const sy = shift.dy * coarseScale;

      // ---- stage 2: LK from keyframe → current with the coarse shift as the initial guess ----
      const obs = kf.observations.filter((o) => map.get(o.landmarkId) !== undefined);
      const n = obs.length;
      if (n < cfg.minInliers) {
        trial.stage = "landmarks";
        record(trial, { matchScore: shift.score }, `kf ${kf.id}: ${n} landmarks left < ${cfg.minInliers}`);
        continue;
      }
      diag.lkTested++;
      const pts = new Float32Array(n * 2);
      const guesses = new Float32Array(n * 2);
      for (let i = 0; i < n; i++) {
        pts[i * 2] = obs[i].x;
        pts[i * 2 + 1] = obs[i].y;
        guesses[i * 2] = obs[i].x + sx;
        guesses[i * 2 + 1] = obs[i].y + sy;
      }
      const res = this.tracker.track(kf.pyramid, current, pts, n, undefined, guesses, cfg.lkMaxDisplacementPx);
      trial.lkRatio = res.okCount / n;
      if (res.okCount < cfg.minInliers) {
        trial.stage = "lk";
        record(
          trial,
          { matchScore: shift.score, lkRatio: trial.lkRatio },
          `kf ${kf.id}: lk ${res.okCount}/${n} < ${cfg.minInliers} (score ${shift.score.toFixed(2)})`,
        );
        continue;
      }
      diag.lkPassed++;

      // ---- stage 3: PnP from the keyframe pose ----
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
      const validation = validateRelocalizationCandidate(
        {
          inliers: pnp.inlierCount,
          reprojectionErrorPx: errPx,
          inlierRatio: trial.inlierRatio,
          coveredCells: trial.spatialCells,
          spatialCoverage: trial.spatialCoverage,
          poseFinite,
        },
        {
          minInliers: cfg.minInliers,
          maxMeanErrorPx: cfg.maxMeanErrorPx,
          minInlierRatio: cfg.minInlierRatio,
          minSpatialCells: cfg.minSpatialCells,
          minSpatialCoverage: cfg.minSpatialCoverage,
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
        shiftX: sx,
        shiftY: sy,
        tracks,
        candidatesTried: diag.candidatesTried,
        reason: null,
        rejectCode: null,
        diagnostics: diag,
      };
      if (validation.inliersPassed && validation.posePassed) diag.pnpPassed++;
      if (!validation.reprojectionPassed && validation.posePassed) diag.errorRejected++;
      if (!validation.ratioPassed) diag.ratioRejected++;
      if (!validation.spatialPassed || !validation.coveragePassed) diag.spatialRejected++;
      if (!validation.passed) {
        const reason =
          validation.rejectReason === "pose_invalid"
            ? `kf ${kf.id}: invalid pose`
            : validation.rejectReason === "inliers"
              ? `kf ${kf.id}: pnp ${pnp.inlierCount}/${m} < ${cfg.minInliers}`
              : validation.rejectReason === "reprojection_error"
                ? `kf ${kf.id}: err ${errPx.toFixed(2)} > ${cfg.maxMeanErrorPx} px`
                : validation.rejectReason === "inlier_ratio"
                  ? `kf ${kf.id}: inlier ratio ${trial.inlierRatio.toFixed(2)} < ${cfg.minInlierRatio}`
                  : `kf ${kf.id}: inliers in ${trial.spatialCells}/9 cells < ${cfg.minSpatialCells}, coverage ${trial.spatialCoverage.toFixed(2)} < ${cfg.minSpatialCoverage}`;
        record(trial, candidate, reason);
        continue;
      }
      diag.validated++;
      record(trial, candidate, null);
      candidate.success = true;
      if (!best || candidate.inlierCount > best.inlierCount) best = candidate;
      if (candidate.inlierCount >= cfg.goodInliers) break;
    }
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
): { dx: number; dy: number; score: number } {
  const w = a.width;
  const h = a.height;
  let best = { dx: 0, dy: 0, score: -1 };
  const am = a.mean;
  const bm = b.mean;
  for (let dy = -radius; dy <= radius; dy++) {
    const y0 = Math.max(0, dy);
    const y1 = Math.min(h, h + dy);
    for (let dx = -radius; dx <= radius; dx++) {
      const x0 = Math.max(0, dx);
      const x1 = Math.min(w, w + dx);
      if (x1 - x0 < w / 2 || y1 - y0 < h / 2) continue;
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
      if (score > best.score) best = { dx, dy, score };
    }
  }
  return best;
}
