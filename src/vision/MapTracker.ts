import type { LandmarkConfig } from "../ar/ARConfig";
import type { CameraIntrinsics } from "../camera/CameraIntrinsics";
import { type Mat3, mat3Identity, mat3Multiply } from "../math/Matrix";
import { refinePosePnP } from "../math/PnP";
import { composeTransforms, invertTransform, type RigidTransform } from "../math/Pose";
import { triangulatePoint, type TriangulationResult } from "../math/Triangulation";
import { LandmarkMap } from "./LandmarkMap";
import type { RelativePose } from "./PoseEstimator";
import type { Track } from "./types";

/**
 * SLAM-lite front end (spec §19–§20, Phase 3):
 *
 *   1. two-view initialization from the Phase 2 reference↔current pose
 *      (map frame = reference camera, |t| = 1)
 *   2. per-frame camera pose in the map frame by PnP against the landmarks
 *      observed by the live tracks (motion-only refinement from a prior)
 *   3. triangulation of new landmarks from each track's anchor observation
 *      once it has enough parallax
 *   4. landmark culling (outliers, age, cap)
 *
 * Scale stays consistent because every new landmark is triangulated from
 * poses that were themselves computed against the existing map.
 */
export interface MapTrackingResult {
  /** True when PnP produced a pose this frame. */
  tracked: boolean;
  inlierCount: number;
  meanReprojectionErrorPx: number;
  newLandmarks: number;
}

export class MapTracker {
  readonly map = new LandmarkMap();
  private _initialized = false;
  private _mapFrameId = -1;
  /** X_cam = R X_map + t for the current frame. */
  private _pose: RigidTransform = identity();
  private _framesSinceTracked = 0;
  private lastResult: MapTrackingResult = { tracked: false, inlierCount: 0, meanReprojectionErrorPx: 0, newLandmarks: 0 };

  // scratch
  private pts3 = new Float64Array(0);
  private obsX = new Float64Array(0);
  private obsY = new Float64Array(0);
  private tri: TriangulationResult = { point: new Float64Array(3), depth1: 0, depth2: 0, parallax: 0, error: Infinity };

  constructor(private readonly config: LandmarkConfig) {}

  get initialized(): boolean {
    return this._initialized;
  }

  get mapFrameId(): number {
    return this._mapFrameId;
  }

  get pose(): RigidTransform {
    return this._pose;
  }

  get framesSinceTracked(): number {
    return this._framesSinceTracked;
  }

  get result(): MapTrackingResult {
    return this.lastResult;
  }

  /**
   * Phase 5: a relocalization found the camera in the existing map. The
   * pose becomes the PnP prior of the following `update()`.
   */
  applyRelocalization(pose: RigidTransform): void {
    this._pose = { rotation: Float64Array.from(pose.rotation), translation: Float64Array.from(pose.translation) };
    this._framesSinceTracked = 0;
  }

  reset(tracks: readonly Track[]): void {
    this.map.clear();
    this._initialized = false;
    this._mapFrameId = -1;
    this._pose = identity();
    this._framesSinceTracked = 0;
    this.lastResult = { tracked: false, inlierCount: 0, meanReprojectionErrorPx: 0, newLandmarks: 0 };
    for (const t of tracks) {
      t.landmarkId = -1;
      t.anchorFrame = -1;
      t.anchorPose = null;
    }
  }

  /**
   * Try to initialize the map from the reference↔current two-view pose.
   * `rel` must be the estimate for the tracks whose refFrame === refFrameId,
   * in the same order the engine used (tracks filtered by refFrame).
   */
  tryInitialize(tracks: readonly Track[], rel: RelativePose, refFrameId: number, frameId: number, k: CameraIntrinsics): boolean {
    const cfg = this.config;
    if (rel.model === "none" || rel.model === "rotation") return false;
    if (rel.parallaxPx < cfg.initMinParallaxPx) return false;
    if (rel.translationConfidence < cfg.initMinTranslationConfidence) return false;

    const pose: RigidTransform = { rotation: rel.rotation, translation: rel.translationDirection };
    const refTracks = tracks.filter((t) => t.refFrame === refFrameId);
    if (refTracks.length !== rel.inlierMask.length) return false;
    const f = (k.fx + k.fy) / 2;
    const maxErr = cfg.maxTriangulationErrorPx / f;
    const minAngle = (cfg.minTriangulationAngleDeg * Math.PI) / 180;

    const created: { track: Track; p: Float64Array }[] = [];
    for (let i = 0; i < refTracks.length; i++) {
      if (!rel.inlierMask[i]) continue;
      const t = refTracks[i];
      const x1 = (t.refX - k.cx) / k.fx;
      const y1 = (t.refY - k.cy) / k.fy;
      const x2 = (t.x - k.cx) / k.fx;
      const y2 = (t.y - k.cy) / k.fy;
      triangulatePoint(pose, x1, y1, x2, y2, this.tri);
      if (this.tri.depth1 <= 0 || this.tri.depth2 <= 0) continue;
      if (this.tri.error > maxErr || this.tri.parallax < minAngle) continue;
      created.push({ track: t, p: Float64Array.from(this.tri.point) });
    }
    if (created.length < cfg.initMinLandmarks) return false;

    this.map.clear();
    this._initialized = true;
    this._mapFrameId = refFrameId;
    this._pose = { rotation: Float64Array.from(rel.rotation), translation: Float64Array.from(rel.translationDirection) };
    this._framesSinceTracked = 0;
    for (const { track, p } of created) {
      const lm = this.map.add(p, track.id, frameId);
      lm.observations = 2;
      track.landmarkId = lm.id;
    }
    // Anchors: reference tracks anchor at the map origin; the others at the
    // current pose.
    const refPose = identity();
    for (const t of tracks) {
      if (t.refFrame === refFrameId) {
        t.anchorFrame = refFrameId;
        t.anchorX = t.refX;
        t.anchorY = t.refY;
        t.anchorPose = refPose;
      } else {
        t.anchorFrame = frameId;
        t.anchorX = t.x;
        t.anchorY = t.y;
        t.anchorPose = this._pose;
      }
    }
    this.lastResult = { tracked: true, inlierCount: created.length, meanReprojectionErrorPx: 0, newLandmarks: created.length };
    return true;
  }

  /**
   * Per-frame update once initialized: PnP, landmark bookkeeping, new
   * triangulations, pruning.
   * @param rotationPrior R_cur←prev from the two-view estimator (may be null)
   */
  update(tracks: Track[], frameId: number, k: CameraIntrinsics, rotationPrior: Mat3 | null): MapTrackingResult {
    const cfg = this.config;
    const f = (k.fx + k.fy) / 2;

    // Drop landmark links of tracks that died (track list only has survivors).
    const alive = new Set<number>();
    for (const t of tracks) if (t.landmarkId >= 0) alive.add(t.landmarkId);
    for (const lm of this.map.values()) if (!alive.has(lm.trackId) && lm.trackId >= 0) lm.trackId = -1;

    // ---- PnP ----
    let n = 0;
    const obsTracks: Track[] = [];
    for (const t of tracks) {
      if (t.landmarkId < 0) continue;
      const lm = this.map.get(t.landmarkId);
      if (!lm) {
        t.landmarkId = -1;
        continue;
      }
      obsTracks.push(t);
      n++;
    }
    this.ensure(n);
    for (let i = 0; i < n; i++) {
      const t = obsTracks[i];
      const lm = this.map.get(t.landmarkId)!;
      this.pts3[i * 3] = lm.position[0];
      this.pts3[i * 3 + 1] = lm.position[1];
      this.pts3[i * 3 + 2] = lm.position[2];
      this.obsX[i] = (t.x - k.cx) / k.fx;
      this.obsY[i] = (t.y - k.cy) / k.fy;
    }

    let tracked = false;
    let inlierCount = 0;
    let meanErrPx = 0;
    if (n >= 6) {
      const prior: RigidTransform = rotationPrior
        ? { rotation: mat3Multiply(rotationPrior, this._pose.rotation), translation: this._pose.translation }
        : this._pose;
      const res = refinePosePnP(prior, this.pts3, this.obsX, this.obsY, n, {
        huber: cfg.pnpHuberPx / f,
        inlierThreshold: cfg.pnpInlierPx / f,
        maxIterations: cfg.pnpMaxIterations,
        epsilon: 1e-6,
      });
      if (res.inlierCount >= cfg.minPnPInliers) {
        tracked = true;
        inlierCount = res.inlierCount;
        meanErrPx = res.meanError * f;
        this._pose = res.pose;
        this._framesSinceTracked = 0;
        for (let i = 0; i < n; i++) {
          const lm = this.map.get(obsTracks[i].landmarkId)!;
          if (res.inliers[i]) {
            lm.observations++;
            lm.lastSeenFrame = frameId;
            lm.outlierCount = 0;
          } else {
            lm.outlierCount++;
            if (lm.outlierCount > cfg.maxOutlierCount) {
              this.map.remove(lm.id);
              obsTracks[i].landmarkId = -1;
            }
          }
        }
      }
    }
    if (!tracked) {
      this._framesSinceTracked++;
      if (rotationPrior) {
        // Propagate the rotation so the pose does not freeze during short dropouts.
        this._pose = { rotation: mat3Multiply(rotationPrior, this._pose.rotation), translation: this._pose.translation };
      }
    }

    // ---- Anchors for tracks the map has not seen yet ----
    for (const t of tracks) {
      if (t.anchorFrame < 0) {
        t.anchorFrame = frameId;
        t.anchorX = t.x;
        t.anchorY = t.y;
        t.anchorPose = this._pose;
      }
    }

    // ---- Triangulate new landmarks ----
    let created = 0;
    if (tracked) {
      const maxErr = cfg.maxTriangulationErrorPx / f;
      const minAngle = (cfg.minTriangulationAngleDeg * Math.PI) / 180;
      const minPar = cfg.triangulateMinParallaxPx;
      for (const t of tracks) {
        if (t.landmarkId >= 0 || !t.anchorPose || t.anchorFrame === frameId) continue;
        if (Math.hypot(t.x - t.anchorX, t.y - t.anchorY) < minPar) continue;
        if (this.map.size >= cfg.maxLandmarks) break;
        // Relative pose anchor → current, triangulate in the anchor frame.
        const anchorInv = invertTransform(t.anchorPose);
        const rel = composeTransforms(this._pose, anchorInv);
        const x1 = (t.anchorX - k.cx) / k.fx;
        const y1 = (t.anchorY - k.cy) / k.fy;
        const x2 = (t.x - k.cx) / k.fx;
        const y2 = (t.y - k.cy) / k.fy;
        triangulatePoint(rel, x1, y1, x2, y2, this.tri);
        if (this.tri.depth1 <= 0 || this.tri.depth2 <= 0) continue;
        if (this.tri.error > maxErr || this.tri.parallax < minAngle) continue;
        // To map frame: X_map = anchorInv(X_anchor)
        const pa = this.tri.point;
        const r = anchorInv.rotation;
        const tt = anchorInv.translation;
        const X = r[0] * pa[0] + r[1] * pa[1] + r[2] * pa[2] + tt[0];
        const Y = r[3] * pa[0] + r[4] * pa[1] + r[5] * pa[2] + tt[1];
        const Z = r[6] * pa[0] + r[7] * pa[1] + r[8] * pa[2] + tt[2];
        const lm = this.map.add([X, Y, Z], t.id, frameId);
        lm.observations = 2;
        t.landmarkId = lm.id;
        created++;
      }
    }

    this.map.prune(frameId, cfg.maxLandmarkAgeFrames, cfg.maxLandmarks);
    this.lastResult = { tracked, inlierCount, meanReprojectionErrorPx: meanErrPx, newLandmarks: created };
    return this.lastResult;
  }

  private ensure(n: number): void {
    if (this.pts3.length < n * 3) {
      this.pts3 = new Float64Array(n * 3);
      this.obsX = new Float64Array(n);
      this.obsY = new Float64Array(n);
    }
  }
}

function identity(): RigidTransform {
  return { rotation: mat3Identity(), translation: new Float64Array(3) };
}
