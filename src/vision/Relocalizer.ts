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
 *   5. accept when enough inliers → camera pose in the existing map, and
 *      the observations become live tracks again
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
  /** Coarse shift found (level-0 pixels). */
  shiftX: number;
  shiftY: number;
  /** Relocalized observations to turn into live tracks (current-frame positions). */
  tracks: { landmarkId: number; x: number; y: number }[];
  candidatesTried: number;
  /** Why the attempt failed (best candidate's stage), null on success. */
  reason: string | null;
  rejectCode: RelocalizationRejectCode | null;
}

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
   */
  relocalize(current: ImagePyramid, map: LandmarkMap, k: CameraIntrinsics): RelocalizationResult {
    const cfg = this.config;
    const fail: RelocalizationResult = {
      success: false, keyframeId: -1, pose: null, inlierCount: 0, meanReprojectionErrorPx: 0,
      matchScore: -1, lkRatio: 0, inlierRatio: 0, spatialCells: 0,
      shiftX: 0, shiftY: 0, tracks: [], candidatesTried: 0, reason: "no keyframes", rejectCode: "no_keyframes",
    };
    if (this.keyframes.length === 0) return fail;

    const coarseSrc = current.levels[Math.min(2, current.levels.length - 1)];
    const curCoarse = downsampleToCoarse(coarseSrc);
    this.scratchCoarse = curCoarse;
    const coarseScale = current.width / curCoarse.width;

    // Candidates: most recent first, then round-robin through the rest on
    // the following attempts so that the view the camera returns to is
    // eventually tried even when it is an old keyframe.
    const ordered = [...this.keyframes].reverse();
    const candidates: Keyframe[] = [];
    for (let i = 0; i < Math.min(cfg.candidatesPerFrame, ordered.length); i++) {
      candidates.push(ordered[(this.candidateCursor + i) % ordered.length]);
    }
    this.candidateCursor = (this.candidateCursor + candidates.length) % Math.max(1, ordered.length);
    let tried = 0;
    let best: RelocalizationResult | null = null;
    // The rejected candidate that got furthest (for the reason / diagnostics).
    let rejected: RelocalizationResult = { ...fail, reason: "" };
    let rejectedStage = -1;
    const reject = (stage: number, partial: Partial<RelocalizationResult>, code: RelocalizationRejectCode, reason: string) => {
      if (stage < rejectedStage) return;
      rejectedStage = stage;
      rejected = { ...fail, ...partial, success: false, reason, rejectCode: code };
    };
    for (const kf of candidates) {
      tried++;
      const shift = coarseShift(kf.coarse, curCoarse, cfg.coarseSearchRadius);
      if (shift.score < cfg.coarseMinScore) {
        if (rejectedStage <= 0 && shift.score > rejected.matchScore) {
          reject(0, { keyframeId: kf.id, matchScore: shift.score }, "low_match_score", `score ${shift.score.toFixed(2)} < ${cfg.coarseMinScore} (kf ${kf.id})`);
        }
        continue;
      }
      const sx = shift.dx * coarseScale;
      const sy = shift.dy * coarseScale;

      // LK from keyframe → current with the coarse shift as the initial guess.
      const obs = kf.observations.filter((o) => map.get(o.landmarkId) !== undefined);
      const n = obs.length;
      if (n < cfg.minInliers) {
        reject(1, { keyframeId: kf.id, matchScore: shift.score }, "insufficient_landmarks", `kf ${kf.id}: ${n} landmarks left < ${cfg.minInliers}`);
        continue;
      }
      const pts = new Float32Array(n * 2);
      const guesses = new Float32Array(n * 2);
      for (let i = 0; i < n; i++) {
        pts[i * 2] = obs[i].x;
        pts[i * 2 + 1] = obs[i].y;
        guesses[i * 2] = obs[i].x + sx;
        guesses[i * 2 + 1] = obs[i].y + sy;
      }
      const res = this.tracker.track(kf.pyramid, current, pts, n, undefined, guesses, cfg.lkMaxDisplacementPx);
      const lkRatio = res.okCount / n;
      if (res.okCount < cfg.minInliers) {
        reject(
          2,
          { keyframeId: kf.id, matchScore: shift.score, lkRatio },
          "lk_failed",
          `kf ${kf.id}: lk ${res.okCount}/${n} < ${cfg.minInliers} (score ${shift.score.toFixed(2)})`,
        );
        continue;
      }

      // PnP from the keyframe pose.
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
      const f = (k.fx + k.fy) / 2;
      const pnp = refinePosePnP(kf.pose, pts3, ox, oy, m, {
        huber: cfg.pnpHuberPx / f,
        inlierThreshold: cfg.pnpInlierPx / f,
        maxIterations: 20,
        epsilon: 1e-7,
      });
      // ---- Global validation of the candidate (v5 §5–§7) ----
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
      const inlierRatio = pnp.inlierCount / m;
      const spatialCells = spatialCellCount(inX, inY, inX.length, current.width, current.height);
      const candidate: RelocalizationResult = {
        success: false,
        keyframeId: kf.id,
        pose: pnp.pose,
        inlierCount: pnp.inlierCount,
        meanReprojectionErrorPx: errPx,
        matchScore: shift.score,
        lkRatio,
        inlierRatio,
        spatialCells,
        shiftX: sx,
        shiftY: sy,
        tracks,
        candidatesTried: tried,
        reason: null,
        rejectCode: null,
      };
      if (!Number.isFinite(errPx) || pnp.pose.translation.some((v) => !Number.isFinite(v))) {
        reject(3, candidate, "invalid_pose", `kf ${kf.id}: invalid pose`);
        continue;
      }
      if (pnp.inlierCount < cfg.minInliers) {
        reject(3, candidate, "insufficient_inliers", `kf ${kf.id}: pnp ${pnp.inlierCount}/${m} < ${cfg.minInliers}`);
        continue;
      }
      if (errPx > cfg.maxMeanErrorPx) {
        reject(4, candidate, "high_reprojection_error", `kf ${kf.id}: err ${errPx.toFixed(2)} > ${cfg.maxMeanErrorPx} px`);
        continue;
      }
      if (inlierRatio < cfg.minInlierRatio) {
        reject(5, candidate, "low_inlier_ratio", `kf ${kf.id}: inlier ratio ${inlierRatio.toFixed(2)} < ${cfg.minInlierRatio}`);
        continue;
      }
      if (spatialCells < cfg.minSpatialCells) {
        reject(6, candidate, "poor_spatial_distribution", `kf ${kf.id}: inliers in ${spatialCells}/9 cells < ${cfg.minSpatialCells}`);
        continue;
      }
      candidate.success = true;
      if (!best || candidate.inlierCount > best.inlierCount) best = candidate;
      if (candidate.inlierCount >= cfg.goodInliers) break;
    }
    if (best) return { ...best, candidatesTried: tried };
    return { ...rejected, candidatesTried: tried };
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
