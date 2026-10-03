import type { LandmarkConfig } from "../ar/ARConfig";
import type { CameraIntrinsics } from "../camera/CameraIntrinsics";
import { type Mat3, mat3Identity, mat3Multiply } from "../math/Matrix";
import { refinePosePnP } from "../math/PnP";
import { composeTransforms, invertTransform, rotationDistance, type RigidTransform } from "../math/Pose";
import {
  poseDelta,
  validatePoseCandidate,
  type PoseCandidate,
  type PoseSource,
  type PoseValidationLimits,
} from "./PoseValidation";
import { triangulatePoint, type TriangulationResult } from "../math/Triangulation";
import { LandmarkMap, type Landmark } from "./LandmarkMap";
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
  /** Camera center moved this much since the previous frame (map units, 修正指示書 v2 §7–§8). */
  poseDeltaTranslation: number;
  /** Camera rotated this much since the previous frame (degrees). */
  poseDeltaRotationDeg: number;
  /** PnP failed: the translation was held and only the rotation prior applied (v2 §6). */
  translationHeld: boolean;
  /** PnP produced a pose but it was rejected by the jump gate (v2 §8). */
  jumpRejected: boolean;
}

/** Pose candidate handed in by an external estimator (plane-relative PnP). */
export interface ExternalPoseCandidate {
  pose: RigidTransform;
  inlierCount: number;
  meanErrorPx: number;
}

/** How the canonical pose of the last frame was chosen (v3 §19–§21). */
export interface PoseSelection {
  source: PoseSource;
  mapInlierCount: number;
  planeInlierCount: number;
  /** Why the map candidate was not used (null when used or absent). */
  mapReject: string | null;
  /** Why the plane candidate was not used (null when used or absent). */
  planeReject: string | null;
  /** Map vs plane candidate difference when both existed (map units / deg). */
  sourceDeltaTranslation: number;
  sourceDeltaRotationDeg: number;
}

const EMPTY_SELECTION: PoseSelection = {
  source: "propagated",
  mapInlierCount: 0,
  planeInlierCount: 0,
  mapReject: null,
  planeReject: null,
  sourceDeltaTranslation: 0,
  sourceDeltaRotationDeg: 0,
};

const EMPTY_RESULT: MapTrackingResult = {
  tracked: false,
  inlierCount: 0,
  meanReprojectionErrorPx: 0,
  newLandmarks: 0,
  poseDeltaTranslation: 0,
  poseDeltaRotationDeg: 0,
  translationHeld: false,
  jumpRejected: false,
};

export class MapTracker {
  readonly map = new LandmarkMap();
  private _initialized = false;
  private _mapFrameId = -1;
  /** X_cam = R X_map + t for the current frame. */
  private _pose: RigidTransform = identity();
  private _framesSinceTracked = 0;
  private lastResult: MapTrackingResult = EMPTY_RESULT;
  private lastSelection: PoseSelection = EMPTY_SELECTION;
  /** Source of the current canonical pose and frames since it last changed (hysteresis, v3 §7). */
  private currentSource: PoseSource = "map";
  private framesSinceSwitch = 0;

  // scratch
  private pts3 = new Float64Array(0);
  private obsX = new Float64Array(0);
  private obsY = new Float64Array(0);
  private maskBuf = new Uint8Array(0);
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

  /** How the last frame's pose was chosen. */
  get selection(): PoseSelection {
    return this.lastSelection;
  }

  /** Frames since the pose source last changed. */
  get framesSinceSourceSwitch(): number {
    return this.framesSinceSwitch;
  }

  /**
   * Phase 5: a relocalization found the camera in the existing map. The
   * pose becomes the PnP prior of the following `update()`.
   */
  applyRelocalization(pose: RigidTransform): void {
    this._pose = { rotation: Float64Array.from(pose.rotation), translation: Float64Array.from(pose.translation) };
    this._framesSinceTracked = 0;
  }

  /**
   * Phase 2 of 修正指示書: an external (plane-relative) estimator provides the
   * pose for the current frame; the map bookkeeping below uses it as is.
   */
  setPose(pose: RigidTransform): void {
    this._pose = { rotation: Float64Array.from(pose.rotation), translation: Float64Array.from(pose.translation) };
    this._framesSinceTracked = 0;
  }

  reset(tracks: readonly Track[]): void {
    this.map.clear();
    this._initialized = false;
    this._mapFrameId = -1;
    this._pose = identity();
    this._framesSinceTracked = 0;
    this.lastResult = EMPTY_RESULT;
    this.lastSelection = EMPTY_SELECTION;
    this.currentSource = "map";
    this.framesSinceSwitch = 0;
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
    const refPose = identity();
    for (const { track, p } of created) {
      const lm = this.map.add(p, track.id, frameId);
      lm.observations = 2;
      lm.anchorPose = refPose;
      lm.anchorX = track.refX;
      lm.anchorY = track.refY;
      lm.parallax = this.tri.parallax; // last computed; refined below
      track.landmarkId = lm.id;
    }
    // Record each landmark's own parallax (the loop above reused the scratch).
    for (const { track, p } of created) {
      void p;
      const lm = this.map.get(track.landmarkId)!;
      const x1 = (track.refX - k.cx) / k.fx, y1 = (track.refY - k.cy) / k.fy;
      const x2 = (track.x - k.cx) / k.fx, y2 = (track.y - k.cy) / k.fy;
      triangulatePoint(pose, x1, y1, x2, y2, this.tri);
      lm.parallax = this.tri.parallax;
    }
    // Anchors: reference tracks anchor at the map origin; the others at the
    // current pose.
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
    this.lastResult = { ...EMPTY_RESULT, tracked: true, inlierCount: created.length, newLandmarks: created.length };
    return true;
  }

  /** Camera center in the map frame: C = −Rᵀ t. */
  cameraCenter(out = new Float64Array(3)): Float64Array {
    const r = this._pose.rotation;
    const t = this._pose.translation;
    out[0] = -(r[0] * t[0] + r[3] * t[1] + r[6] * t[2]);
    out[1] = -(r[1] * t[0] + r[4] * t[1] + r[7] * t[2]);
    out[2] = -(r[2] * t[0] + r[5] * t[1] + r[8] * t[2]);
    return out;
  }

  /**
   * Per-frame update once initialized: PnP, landmark bookkeeping, new
   * triangulations, pruning.
   * @param rotationPrior R_cur←prev from the two-view estimator (may be null)
   * @param external      pose candidate from the plane-relative estimator for
   *                      this frame (with its quality). It is validated like
   *                      the landmark PnP candidate and never adopted
   *                      unconditionally (v3 §3–§6)
   */
  update(
    tracks: Track[],
    frameId: number,
    k: CameraIntrinsics,
    rotationPrior: Mat3 | null,
    external: ExternalPoseCandidate | null = null,
  ): MapTrackingResult {
    const cfg = this.config;
    const f = (k.fx + k.fy) / 2;
    const prevCenter = this.cameraCenter();
    const prevRotation: Mat3 = Float64Array.from(this._pose.rotation);

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

    // ---- Pose candidates → validation → canonical pose (v3 §1–§7) ----
    //
    //   landmark PnP  ─┐
    //   plane PnP     ─┼─→ candidates → validatePoseCandidate() → this._pose
    //   propagation   ─┘
    //
    // Neither estimator writes the pose directly. Both are checked against
    // the previous accepted pose with the same limits (jump gate), the plane
    // candidate additionally against the map candidate when both exist, and
    // switching between sources is damped by a cooldown.
    const prior: RigidTransform = rotationPrior
      ? { rotation: mat3Multiply(rotationPrior, this._pose.rotation), translation: this._pose.translation }
      : this._pose;
    const depth = n > 0 ? this.medianDepth(prior, n) : 0;
    const limits: PoseValidationLimits = {
      maxTranslation: Math.max(cfg.jumpRejectDepthRatio * Math.max(depth, 1e-9), cfg.jumpRejectSpeedFactor * this.lastResult.poseDeltaTranslation),
      maxRotationDeg: cfg.jumpRejectRotationDeg,
    };
    const trusted = (c: { inlierCount: number; reprojectionErrorPx: number }) =>
      c.inlierCount >= cfg.jumpRejectTrustedInliers && c.reprojectionErrorPx <= cfg.jumpRejectTrustedErrorPx;

    // Candidate 1: landmark PnP (mature landmarks only when enough of them, v2 §27).
    let mapCandidate: PoseCandidate | null = null;
    let mapReject: string | null = null;
    let mask: Uint8Array | null = null;
    if (n >= 6) {
      let mature = 0;
      this.ensureMask(n);
      for (let i = 0; i < n; i++) {
        const m = this.map.get(obsTracks[i].landmarkId)!.observations >= cfg.minObservationsForPose ? 1 : 0;
        this.maskBuf[i] = m;
        mature += m;
      }
      if (mature >= cfg.minMaturePnPPoints) mask = this.maskBuf;
      const res = refinePosePnP(
        prior,
        this.pts3,
        this.obsX,
        this.obsY,
        n,
        { huber: cfg.pnpHuberPx / f, inlierThreshold: cfg.pnpInlierPx / f, maxIterations: cfg.pnpMaxIterations, epsilon: 1e-6 },
        mask,
      );
      if (res.inlierCount >= cfg.minPnPInliers) {
        mapCandidate = { pose: res.pose, source: "map", inlierCount: res.inlierCount, reprojectionErrorPx: res.meanError * f };
        // Jump gate (v2 §8): a weakly supported solve that moves the camera
        // implausibly far in one frame is a wrong pose, not fast motion.
        if (!trusted(mapCandidate)) {
          const v = validatePoseCandidate(res.pose, this._pose, limits, "map");
          if (!v.accepted) {
            mapReject = v.reason;
            mapCandidate = null;
          }
        }
      } else {
        mapReject = `map inliers ${res.inlierCount} < ${cfg.minPnPInliers}`;
      }
    } else if (n > 0) {
      mapReject = `map observations ${n} < 6`;
    }

    // Candidate 2: plane-relative PnP (external). Same gate, plus agreement
    // with the map candidate (v3 §3–§6); never an unconditional override.
    let planeCandidate: PoseCandidate | null = null;
    let planeReject: string | null = null;
    let sourceDelta = { translation: 0, rotationDeg: 0 };
    if (external) {
      planeCandidate = { pose: external.pose, source: "plane", inlierCount: external.inlierCount, reprojectionErrorPx: external.meanErrorPx };
      if (!trusted(planeCandidate)) {
        const v = validatePoseCandidate(external.pose, this._pose, limits, "plane");
        if (!v.accepted) {
          planeReject = v.reason;
          planeCandidate = null;
        }
      }
      if (planeCandidate && mapCandidate) {
        sourceDelta = poseDelta(mapCandidate.pose, planeCandidate.pose);
        const agree = validatePoseCandidate(planeCandidate.pose, mapCandidate.pose, limits, "plane vs map");
        if (!agree.accepted) {
          planeReject = agree.reason;
          planeCandidate = null;
        }
      }
    }

    // Selection with hysteresis (v3 §7): the plane estimator is preferred when
    // it produced a valid candidate, but switching back to it from the map
    // waits for the cooldown unless the map has nothing.
    let chosen: PoseCandidate | null = null;
    if (planeCandidate && (this.currentSource === "plane" || !mapCandidate || this.framesSinceSwitch >= cfg.sourceSwitchCooldownFrames)) {
      chosen = planeCandidate;
    } else if (mapCandidate) {
      chosen = mapCandidate;
    }
    if (chosen && planeCandidate && chosen !== planeCandidate && !planeReject) {
      planeReject = `cooldown ${this.framesSinceSwitch}/${cfg.sourceSwitchCooldownFrames}`;
    }

    let tracked = false;
    let inlierCount = 0;
    let meanErrPx = 0;
    const jumpRejected = mapReject !== null && mapReject.includes("jump");
    if (chosen) {
      tracked = true;
      if (chosen.source !== this.currentSource) {
        this.currentSource = chosen.source;
        this.framesSinceSwitch = 0;
      } else {
        this.framesSinceSwitch++;
      }
      this._pose = { rotation: Float64Array.from(chosen.pose.rotation), translation: Float64Array.from(chosen.pose.translation) };
      this._framesSinceTracked = 0;
      if (n > 0) {
        // Classify every observed landmark (mature and young) against the canonical pose.
        const cls = this.classify(this._pose, n, cfg.pnpInlierPx / f);
        inlierCount = cls.inlierCount;
        meanErrPx = cls.meanError * f;
        const maxErr = cfg.maxTriangulationErrorPx / f;
        for (let i = 0; i < n; i++) {
          const lm = this.map.get(obsTracks[i].landmarkId)!;
          if (cls.inliers[i]) {
            lm.observations++;
            lm.lastSeenFrame = frameId;
            lm.outlierCount = 0;
            // The map is the fixed reference the world is anchored to; the
            // pose must not rewrite it unless explicitly enabled (v2 §3–§5).
            if (cfg.enableLandmarkDepthRefinement) this.refineLandmark(lm, obsTracks[i], k, maxErr);
          } else {
            lm.outlierCount++;
            if (lm.observations <= 2 && lm.outlierCount >= cfg.youngOutlierFrames) {
              // A candidate that disagrees with the pose right away was a bad
              // triangulation: drop it before it can bias anything.
              obsTracks[i].landmarkId = -1;
              this.map.remove(lm.id);
            } else if (lm.outlierCount > cfg.maxOutlierCount) {
              // The track drifted away from the landmark (LK drift) more
              // often than the landmark is wrong: unlink the track and keep
              // the landmark for the map / plane; young landmarks (never
              // confirmed by PnP) are removed instead.
              obsTracks[i].landmarkId = -1;
              lm.trackId = -1;
              lm.outlierCount = 0;
              if (lm.observations <= 2) this.map.remove(lm.id);
            }
          }
        }
      }
    }
    let translationHeld = false;
    if (!tracked) {
      this._framesSinceTracked++;
      this.framesSinceSwitch++;
      if (rotationPrior) {
        // Propagate the rotation so the pose does not freeze during short
        // dropouts; the translation is held (a two-view translation is
        // scale-free and must not enter the map-frame pose, v2 §6).
        this._pose = { rotation: mat3Multiply(rotationPrior, this._pose.rotation), translation: this._pose.translation };
        translationHeld = true;
      }
    }
    this.lastSelection = {
      source: chosen ? chosen.source : "propagated",
      mapInlierCount: mapCandidate ? mapCandidate.inlierCount : 0,
      planeInlierCount: external ? external.inlierCount : 0,
      mapReject,
      planeReject,
      sourceDeltaTranslation: sourceDelta.translation,
      sourceDeltaRotationDeg: sourceDelta.rotationDeg,
    };

    // ---- Anchors for tracks the map has not seen yet ----
    // Anchors need a trustworthy pose: only assign them in tracked frames.
    // While the camera is lost, anchors of landmark-less tracks are cleared
    // so they re-anchor once the pose is known again.
    for (const t of tracks) {
      if (t.landmarkId >= 0) continue;
      if (tracked) {
        if (t.anchorFrame < 0) {
          t.anchorFrame = frameId;
          t.anchorX = t.x;
          t.anchorY = t.y;
          t.anchorPose = this._pose;
        }
      } else {
        t.anchorFrame = -1;
        t.anchorPose = null;
      }
    }

    // ---- Triangulate new landmarks ----
    let created = 0;
    if (tracked) {
      const maxErr = cfg.maxTriangulationErrorPx / f;
      const minAngle = (cfg.minTriangulationAngleDeg * Math.PI) / 180;
      const minPar = cfg.triangulateMinParallaxPx;
      // Depth sanity against the existing map (a point "at infinity" or in
      // front of the lens is a triangulation failure, not a landmark).
      const medDepth = n > 0 ? this.medianDepth(this._pose, n) : 0;
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
        if (medDepth > 0 && (this.tri.depth2 > cfg.maxDepthRatio * medDepth || this.tri.depth2 < medDepth / cfg.maxDepthRatio)) continue;
        // To map frame: X_map = anchorInv(X_anchor)
        const pa = this.tri.point;
        const r = anchorInv.rotation;
        const tt = anchorInv.translation;
        const X = r[0] * pa[0] + r[1] * pa[1] + r[2] * pa[2] + tt[0];
        const Y = r[3] * pa[0] + r[4] * pa[1] + r[5] * pa[2] + tt[1];
        const Z = r[6] * pa[0] + r[7] * pa[1] + r[8] * pa[2] + tt[2];
        const lm = this.map.add([X, Y, Z], t.id, frameId);
        lm.observations = 2;
        lm.anchorPose = t.anchorPose;
        lm.anchorX = t.anchorX;
        lm.anchorY = t.anchorY;
        lm.parallax = this.tri.parallax;
        t.landmarkId = lm.id;
        created++;
      }
    }

    this.map.prune(frameId, cfg.maxLandmarkAgeFrames, cfg.maxLandmarks);
    const center = this.cameraCenter();
    this.lastResult = {
      tracked,
      inlierCount,
      meanReprojectionErrorPx: meanErrPx,
      newLandmarks: created,
      poseDeltaTranslation: Math.hypot(center[0] - prevCenter[0], center[1] - prevCenter[1], center[2] - prevCenter[2]),
      poseDeltaRotationDeg: (rotationDistance(prevRotation, this._pose.rotation) * 180) / Math.PI,
      translationHeld,
      jumpRejected,
    };
    return this.lastResult;
  }

  /**
   * Depth refinement: once the baseline between a landmark's anchor view and
   * the current view has grown enough, re-triangulate it. Two-view depth
   * error scales with 1/parallax, so the first estimate (from the small
   * initialization baseline) is replaced by progressively better ones.
   */
  private refineLandmark(lm: Landmark, track: Track, k: CameraIntrinsics, maxErr: number): void {
    if (!lm.anchorPose) return;
    const cfg = this.config;
    const anchorInv = invertTransform(lm.anchorPose);
    const rel = composeTransforms(this._pose, anchorInv);
    const x1 = (lm.anchorX - k.cx) / k.fx;
    const y1 = (lm.anchorY - k.cy) / k.fy;
    const x2 = (track.x - k.cx) / k.fx;
    const y2 = (track.y - k.cy) / k.fy;
    triangulatePoint(rel, x1, y1, x2, y2, this.tri);
    if (this.tri.depth1 <= 0 || this.tri.depth2 <= 0) return;
    if (this.tri.error > maxErr) return;
    if (this.tri.parallax < lm.parallax * cfg.refineParallaxGrowth) return;
    const pa = this.tri.point;
    const r = anchorInv.rotation;
    const tt = anchorInv.translation;
    lm.position[0] = r[0] * pa[0] + r[1] * pa[1] + r[2] * pa[2] + tt[0];
    lm.position[1] = r[3] * pa[0] + r[4] * pa[1] + r[5] * pa[2] + tt[1];
    lm.position[2] = r[6] * pa[0] + r[7] * pa[1] + r[8] * pa[2] + tt[2];
    lm.parallax = this.tri.parallax;
  }

  /** Inlier classification of the first `n` scratch observations under `pose`. */
  private classify(pose: RigidTransform, n: number, thr: number): { inliers: Uint8Array; inlierCount: number; meanError: number } {
    const r = pose.rotation;
    const t = pose.translation;
    const inliers = new Uint8Array(n);
    const thrSq = thr * thr;
    let count = 0;
    let errSum = 0;
    for (let i = 0; i < n; i++) {
      const X = this.pts3[i * 3], Y = this.pts3[i * 3 + 1], Z = this.pts3[i * 3 + 2];
      const z = r[6] * X + r[7] * Y + r[8] * Z + t[2];
      if (z <= 1e-6) continue;
      const u = (r[0] * X + r[1] * Y + r[2] * Z + t[0]) / z;
      const v = (r[3] * X + r[4] * Y + r[5] * Z + t[1]) / z;
      const e2 = (u - this.obsX[i]) ** 2 + (v - this.obsY[i]) ** 2;
      if (e2 < thrSq) {
        inliers[i] = 1;
        count++;
        errSum += Math.sqrt(e2);
      }
    }
    return { inliers, inlierCount: count, meanError: count ? errSum / count : 0 };
  }

  private ensureMask(n: number): void {
    if (this.maskBuf.length < n) this.maskBuf = new Uint8Array(n);
  }

  /** Median depth of the first `n` scratch points under `pose` (map units). */
  private medianDepth(pose: RigidTransform, n: number): number {
    const r = pose.rotation;
    const t = pose.translation;
    const zs: number[] = [];
    for (let i = 0; i < n; i++) {
      const z = r[6] * this.pts3[i * 3] + r[7] * this.pts3[i * 3 + 1] + r[8] * this.pts3[i * 3 + 2] + t[2];
      if (z > 0) zs.push(z);
    }
    if (zs.length === 0) return 1;
    zs.sort((a, b) => a - b);
    return zs[zs.length >> 1];
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
