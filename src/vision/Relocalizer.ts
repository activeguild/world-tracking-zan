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
export interface RelocalizationResult {
  success: boolean;
  keyframeId: number;
  /** Pose in the map frame when successful. */
  pose: RigidTransform | null;
  inlierCount: number;
  meanReprojectionErrorPx: number;
  /** Coarse shift found (level-0 pixels). */
  shiftX: number;
  shiftY: number;
  /** Relocalized observations to turn into live tracks (current-frame positions). */
  tracks: { landmarkId: number; x: number; y: number }[];
  candidatesTried: number;
  /** Why the attempt failed (best candidate's stage), null on success. */
  reason: string | null;
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
      shiftX: 0, shiftY: 0, tracks: [], candidatesTried: 0, reason: "no keyframes",
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
    let reason = "";
    let bestScore = -1;
    for (const kf of candidates) {
      tried++;
      const shift = coarseShift(kf.coarse, curCoarse, cfg.coarseSearchRadius);
      if (shift.score > bestScore) {
        bestScore = shift.score;
        reason = `score ${shift.score.toFixed(2)} < ${cfg.coarseMinScore} (kf ${kf.id})`;
      }
      if (shift.score < cfg.coarseMinScore) continue;
      const sx = shift.dx * coarseScale;
      const sy = shift.dy * coarseScale;

      // LK from keyframe → current with the coarse shift as the initial guess.
      const obs = kf.observations.filter((o) => map.get(o.landmarkId) !== undefined);
      const n = obs.length;
      if (n < cfg.minInliers) {
        reason = `kf ${kf.id}: ${n} landmarks left < ${cfg.minInliers}`;
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
      if (res.okCount < cfg.minInliers) {
        reason = `kf ${kf.id}: lk ${res.okCount}/${n} < ${cfg.minInliers} (score ${shift.score.toFixed(2)})`;
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
      if (pnp.inlierCount < cfg.minInliers) {
        reason = `kf ${kf.id}: pnp ${pnp.inlierCount}/${m} < ${cfg.minInliers}`;
        continue;
      }
      if (pnp.meanError * f > cfg.maxMeanErrorPx) {
        reason = `kf ${kf.id}: err ${(pnp.meanError * f).toFixed(2)} > ${cfg.maxMeanErrorPx} px`;
        continue;
      }

      const tracks: RelocalizationResult["tracks"] = [];
      for (let q = 0; q < m; q++) {
        if (!pnp.inliers[q]) continue;
        const i = idx[q];
        tracks.push({ landmarkId: obs[i].landmarkId, x: res.positions[i * 2], y: res.positions[i * 2 + 1] });
      }
      const result: RelocalizationResult = {
        success: true,
        keyframeId: kf.id,
        pose: pnp.pose,
        inlierCount: pnp.inlierCount,
        meanReprojectionErrorPx: pnp.meanError * f,
        shiftX: sx,
        shiftY: sy,
        tracks,
        candidatesTried: tried,
        reason: null,
      };
      if (!best || result.inlierCount > best.inlierCount) best = result;
      if (result.inlierCount >= cfg.goodInliers) break;
    }
    if (best) return best;
    return { ...fail, candidatesTried: tried, reason };
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
