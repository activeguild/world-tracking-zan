import type { ARConfig } from "../ar/ARConfig";
import { TrackingState, TrackingStateMachine } from "../ar/ARState";
import { FeatureDetector } from "./FeatureDetector";
import { FeatureTracker, TrackStatus, allocResult, type TrackResult } from "./FeatureTracker";
import { ImagePyramid } from "./ImagePyramid";
import { ransacHomography, type Rng } from "./OutlierRejection";
import { MapTracker } from "./MapTracker";
import { PlaneDetector } from "./PlaneDetector";
import { PlaneTracker } from "./PlaneTracker";
import { PoseEstimator, type RelativePose } from "./PoseEstimator";
import { Relocalizer } from "./Relocalizer";
import { computeTrackingConfidence, emptyQuality, type TrackingQuality } from "./TrackingQuality";
import {
  LANDMARK_STRIDE,
  packTracks,
  type MapPoseOutput,
  type PlaneAnchorOutput,
  type PlaneOutput,
  type PlanePoseOutput,
  type PoseOutput,
  type RelocalizationOutput,
  type Track,
  type VisionInput,
  type VisionOutput,
} from "./types";
import { type Mat3, mat3Identity, mat3Multiply } from "../math/Matrix";
import { transpose3 } from "../math/Decomposition";
import { rotationDistance, rotationToQuaternion } from "../math/Pose";

/**
 * Phase 1 vision pipeline (spec §55):
 *
 *   gray → pyramid → LK (prev→cur) → forward-backward check
 *        → Homography RANSAC → drop outliers
 *        → FAST replenishment (grid distributed, masked around live tracks)
 *        → TrackingQuality + state machine
 *
 * The engine is DOM-free and deterministic given a seeded RNG, so it runs
 * identically inside the Web Worker, on the main thread, and in Vitest.
 */
export class VisionEngine {
  readonly width: number;
  readonly height: number;

  private readonly detector: FeatureDetector;
  private readonly tracker: FeatureTracker;
  private prevPyramid: ImagePyramid;
  private curPyramid: ImagePyramid;
  private hasPrev = false;

  private tracks: Track[] = [];
  private nextTrackId = 1;

  private readonly mask: Uint8Array;
  private readonly cellCounts: Uint16Array;
  private pointBuf: Float32Array;
  private guessBuf = new Float32Array(0);
  private trackResult: TrackResult;
  private readonly c1x: Float32Array;
  private readonly c1y: Float32Array;
  private readonly c2x: Float32Array;
  private readonly c2y: Float32Array;

  private readonly stateMachine: TrackingStateMachine;
  private lastQuality: TrackingQuality = emptyQuality();
  private lastHomographyInliers = 0;
  private lastRansacError = 0;

  // Phase 2: two-view pose relative to a reference frame.
  private readonly poseEstimator: PoseEstimator;
  private referenceFrameId = -1;
  /** R_ref←origin: rotation accumulated across reference renewals. */
  private referenceRotation: Mat3 = mat3Identity();
  /** Last R_cur←ref, used when the reference is renewed. */
  private lastRelativeRotation: Mat3 = mat3Identity();
  private lastAccumulatedRotation: Mat3 = mat3Identity();
  private lastPlaneNormal: Float64Array | null = null;
  private lastPose: PoseOutput | null = null;
  private readonly r1x: Float32Array;
  private readonly r1y: Float32Array;
  private readonly r2x: Float32Array;
  private readonly r2y: Float32Array;
  /** Last reference↔current relative pose (for map initialization). */
  private lastRelative: RelativePose | null = null;
  private lastRelativeRefFrame = -1;
  /** Rotation of the previous frame relative to its reference (for the PnP prior). */
  private prevFrameRotation: Mat3 | null = null;

  // Phase 3: landmark map + plane.
  private readonly mapTracker: MapTracker;
  private readonly planeDetector: PlaneDetector;
  private lastMapPose: MapPoseOutput | null = null;
  private lastPlane: PlaneOutput | null = null;
  private packedLandmarks = new Float32Array(0);
  private packedLandmarkCount = 0;

  // Plane-anchored tracking (修正指示書): the first found plane is fixed and
  // the camera pose is solved relative to it (depth-free).
  private readonly planeTracker: PlaneTracker;
  private lastPlaneAnchor: PlaneAnchorOutput | null = null;
  private lastPlanePose: PlanePoseOutput | null = null;
  private lastPoseSource: MapPoseOutput["source"] = "propagated";

  // Phase 5: keyframes + relocalization.
  private readonly relocalizer: Relocalizer;
  private relocStatus: RelocalizationOutput = emptyReloc();
  private relocSuccessCount = 0;
  private relocLastSuccessFrame = -1;

  /** Timing breakdown of the last frame (ms). */
  readonly timing = { pyramid: 0, track: 0, ransac: 0, detect: 0, pose: 0, map: 0, plane: 0, reloc: 0, total: 0 };

  constructor(
    width: number,
    height: number,
    private readonly config: ARConfig,
    private readonly rng: Rng = Math.random,
  ) {
    this.width = width;
    this.height = height;
    this.detector = new FeatureDetector(width, height, config.features);
    this.tracker = new FeatureTracker(config.tracker);
    this.prevPyramid = new ImagePyramid(width, height, config.tracker.pyramidLevels);
    this.curPyramid = new ImagePyramid(width, height, config.tracker.pyramidLevels);
    this.mask = new Uint8Array(width * height);
    this.cellCounts = new Uint16Array(config.features.gridCols * config.features.gridRows);
    const cap = config.features.maxFeatures;
    this.pointBuf = new Float32Array(cap * 2);
    this.trackResult = allocResult(cap);
    this.c1x = new Float32Array(cap);
    this.c1y = new Float32Array(cap);
    this.c2x = new Float32Array(cap);
    this.c2y = new Float32Array(cap);
    this.r1x = new Float32Array(cap);
    this.r1y = new Float32Array(cap);
    this.r2x = new Float32Array(cap);
    this.r2y = new Float32Array(cap);
    this.stateMachine = new TrackingStateMachine(config.state);
    this.poseEstimator = new PoseEstimator(config.pose, config.ransac, rng);
    this.mapTracker = new MapTracker(config.landmarks);
    this.planeDetector = new PlaneDetector(config.plane, rng);
    this.planeTracker = new PlaneTracker(config.planeTracking);
    this.relocalizer = new Relocalizer(config.relocalization, config.tracker);
  }

  /** The fixed plane the world is anchored to (null until a plane was found). */
  get planeAnchor(): PlaneAnchorOutput | null {
    return this.lastPlaneAnchor;
  }

  get keyframeCount(): number {
    return this.relocalizer.count;
  }

  /** Last camera pose in the map frame (null until the map is initialized). */
  get mapPose(): MapPoseOutput | null {
    return this.lastMapPose;
  }

  /** Last plane candidate. */
  get plane(): PlaneOutput | null {
    return this.lastPlane;
  }

  get landmarkCount(): number {
    return this.mapTracker.map.size;
  }

  get state(): TrackingState {
    return this.stateMachine.state;
  }

  get currentTracks(): readonly Track[] {
    return this.tracks;
  }

  get fastThreshold(): number {
    return this.detector.threshold;
  }

  reset(): void {
    this.tracks = [];
    this.hasPrev = false;
    this.stateMachine.reset();
    this.lastQuality = emptyQuality();
    this.referenceFrameId = -1;
    this.referenceRotation = mat3Identity();
    this.lastRelativeRotation = mat3Identity();
    this.lastAccumulatedRotation = mat3Identity();
    this.lastPlaneNormal = null;
    this.lastPose = null;
    this.lastRelative = null;
    this.lastRelativeRefFrame = -1;
    this.prevFrameRotation = null;
    this.mapTracker.reset(this.tracks);
    this.planeDetector.reset();
    this.planeTracker.reset(this.tracks);
    this.lastPlaneAnchor = null;
    this.lastPlanePose = null;
    this.lastPoseSource = "propagated";
    this.lastMapPose = null;
    this.lastPlane = null;
    this.packedLandmarkCount = 0;
    this.relocalizer.reset();
    this.relocStatus = emptyReloc();
    this.relocSuccessCount = 0;
    this.relocLastSuccessFrame = -1;
  }

  /** Last pose output (null until a reference frame and enough tracks exist). */
  get pose(): PoseOutput | null {
    return this.lastPose;
  }

  process(input: VisionInput): VisionOutput {
    const t0 = now();
    if (input.width !== this.width || input.height !== this.height) {
      throw new Error(
        `VisionEngine: frame size ${input.width}x${input.height} does not match engine ${this.width}x${this.height}`,
      );
    }

    // 1. Pyramid
    this.curPyramid.build(input.gray);
    const t1 = now();

    // 2. Track previous features into this frame
    const previousCount = this.tracks.length;
    let trackedCount = 0;
    if (this.hasPrev && previousCount > 0) {
      trackedCount = this.trackExisting();
    } else {
      this.tracks = [];
    }
    const t2 = now();

    // 3. RANSAC on the surviving correspondences
    let inlierCount = trackedCount;
    if (trackedCount > 0) {
      inlierCount = this.rejectOutliers();
    } else {
      this.lastHomographyInliers = 0;
      this.lastRansacError = 0;
    }
    const t3 = now();

    // 4. Replenish features when we are short
    const cfg = this.config.features;
    if (this.tracks.length < cfg.replenishBelow) {
      this.replenish(input.gray);
    }
    const t4 = now();

    // 5. Two-view pose against the reference frame (Phase 2)
    const pose = this.estimatePose(input);
    const t4b = now();

    // 6. Landmark map (Phase 3): initialization, PnP, triangulation
    this.updateMap(input);
    const t4c = now();

    // 7. Plane detection (Phase 3)
    this.updatePlane(input);
    const t4d = now();

    // 8. Quality + state
    const featureCount = this.tracks.length;
    const quality: TrackingQuality = {
      featureCount,
      trackedCount,
      inlierCount,
      reprojectionError: this.lastMapPose ? this.lastMapPose.meanReprojectionErrorPx : this.lastRansacError,
      poseDelta: this.lastPoseDelta,
      planeConfidence: this.lastPlane ? this.lastPlane.confidence : 0,
      trackingConfidence: computeTrackingConfidence(
        inlierCount,
        previousCount,
        this.config.state.minTrackedForTracking,
      ),
      lowFeature: featureCount < cfg.minFeatures,
    };
    this.lastQuality = quality;
    const state = this.stateMachine.update({
      inlierCount,
      featureCount,
      mapInitialized: this.mapTracker.initialized,
      // Once anchored, the plane is the fixed world reference: it stays
      // "found" even when the per-frame detector does not re-detect it.
      planeFound: this.planeTracker.anchored || (this.lastPlane?.found ?? false),
      mapLost:
        this.mapTracker.initialized && this.mapTracker.framesSinceTracked > this.config.state.mapLostFrameTolerance,
    });

    // Swap pyramids for the next frame.
    const tmp = this.prevPyramid;
    this.prevPyramid = this.curPyramid;
    this.curPyramid = tmp;
    this.hasPrev = true;

    const t5 = now();
    this.timing.pyramid = t1 - t0;
    this.timing.track = t2 - t1;
    this.timing.ransac = t3 - t2;
    this.timing.detect = t4 - t3;
    this.timing.pose = t4b - t4;
    this.timing.map = t4c - t4b - this.timing.reloc;
    this.timing.plane = t4d - t4c;
    this.timing.total = t5 - t0;

    return {
      frameId: input.frameId,
      timestamp: input.timestamp,
      state,
      quality,
      pose,
      mapPose: this.lastMapPose,
      plane: this.lastPlane,
      planeSearch: this.mapTracker.initialized
        ? { ...this.planeDetector.lastSearch, minInliers: this.config.plane.minInliers }
        : null,
      planeAnchor: this.lastPlaneAnchor,
      planePose: this.lastPlanePose,
      relocalization: this.relocStatus,
      landmarks: this.packedLandmarks.slice(0, this.packedLandmarkCount * LANDMARK_STRIDE),
      landmarkCount: this.packedLandmarkCount,
      tracks: packTracks(this.tracks),
      trackCount: this.tracks.length,
      processingMs: t5 - t0,
    };
  }

  /**
   * Phase 3: initialize the landmark map from the two-view pose, then track
   * the camera against the map with PnP and triangulate new landmarks.
   */
  private updateMap(input: VisionInput): void {
    const cfg = this.config.landmarks;
    const k = input.intrinsics;
    const frameId = input.frameId;
    const tracker = this.mapTracker;

    // Frame-to-frame rotation prior from the two-view estimator:
    // R_cur←prev = R_cur←ref · R_prev←refᵀ when both share the reference.
    let rotationPrior: Mat3 | null = null;
    if (this.lastRelative && this.prevFrameRotation && this.lastRelative.model !== "none") {
      rotationPrior = mat3Multiply(this.lastRelative.rotation, transpose3(this.prevFrameRotation));
    }

    this.timing.reloc = 0;
    this.relocStatus = {
      keyframes: this.relocalizer.count,
      attempt: "none",
      inlierCount: 0,
      candidatesTried: 0,
      lastSuccessFrame: this.relocLastSuccessFrame,
      successCount: this.relocSuccessCount,
    };

    if (!tracker.initialized) {
      if (this.lastRelative && this.lastRelativeRefFrame >= 0) {
        if (tracker.tryInitialize(this.tracks, this.lastRelative, this.lastRelativeRefFrame, frameId, k)) {
          this.planeDetector.reset();
          this.planeTracker.reset(this.tracks);
          this.lastPlaneAnchor = null;
          this.relocalizer.reset();
          // The initialization frame is the first keyframe.
          this.relocalizer.create(this.curPyramid, tracker.pose, this.tracks, frameId, input.timestamp);
          this.relocStatus.keyframes = this.relocalizer.count;
        }
      }
    } else {
      // Phase 5: relocalize when the camera was not located in the previous frame.
      const rc = this.config.relocalization;
      if (
        tracker.framesSinceTracked >= rc.startAfterLostFrames &&
        (tracker.framesSinceTracked - rc.startAfterLostFrames) % rc.attemptEveryNFrames === 0
      ) {
        const tr0 = now();
        const r = this.relocalizer.relocalize(this.curPyramid, tracker.map, k);
        this.timing.reloc = now() - tr0;
        this.relocStatus.attempt = r.success ? "success" : "fail";
        this.relocStatus.inlierCount = r.inlierCount;
        this.relocStatus.candidatesTried = r.candidatesTried;
        if (r.success && r.pose) {
          tracker.applyRelocalization(r.pose);
          this.injectRelocalizedTracks(r.tracks, frameId, r.pose);
          rotationPrior = null; // the relocalized pose is the prior
          this.relocSuccessCount++;
          this.relocLastSuccessFrame = frameId;
          this.relocStatus.lastSuccessFrame = frameId;
          this.relocStatus.successCount = this.relocSuccessCount;
        }
      }

      // Plane-relative pose (修正指示書 §7): main path once a plane is anchored.
      // The landmark PnP stays as the fallback when the plane is out of view.
      const pt = this.config.planeTracking;
      let planeRes = this.planeTracker.result;
      let poseOverride = null;
      if (pt.enabled && this.planeTracker.anchored) {
        const prior = rotationPrior
          ? { rotation: mat3Multiply(rotationPrior, tracker.pose.rotation), translation: tracker.pose.translation }
          : tracker.pose;
        planeRes = this.planeTracker.update(this.tracks, prior, k);
        if (planeRes.tracked) poseOverride = planeRes.pose;
      }

      const res = tracker.update(this.tracks, frameId, k, rotationPrior, poseOverride);
      this.lastPoseSource = !res.tracked ? "propagated" : poseOverride ? "plane" : "map";
      if (res.tracked) {
        if (this.planeTracker.anchored) {
          // Lift features that are new since the anchor, only from frames
          // whose pose is well supported (a bad pose would bake its error
          // into the lifted points).
          const good = poseOverride
            ? planeRes.inlierCount >= pt.liftMinInliers && planeRes.meanErrorPx <= pt.liftMaxMeanErrorPx
            : res.inlierCount >= pt.liftMinInliers && res.meanReprojectionErrorPx <= pt.liftMaxMeanErrorPx;
          if (good) this.planeTracker.lift(this.tracks, tracker.pose, k, (t) => this.observesPlaneLandmark(t));
        }
        // Keyframe policy (spec §35).
        const par = this.medianParallaxSinceLastKeyframe();
        if (this.relocalizer.shouldCreate(tracker.pose, frameId, res.inlierCount, par)) {
          this.relocalizer.create(this.curPyramid, tracker.pose, this.tracks, frameId, input.timestamp);
          this.relocStatus.keyframes = this.relocalizer.count;
        }
      } else if (tracker.framesSinceTracked > cfg.lostResetFrames) {
        // Lost for too long and relocalization did not succeed: start over.
        tracker.reset(this.tracks);
        this.planeDetector.reset();
        this.planeTracker.reset(this.tracks);
        this.lastPlaneAnchor = null;
        this.relocalizer.reset();
      }
    }
    this.lastPlanePose = this.planeTracker.anchored ? toPlanePoseOutput(this.planeTracker.result) : null;

    if (tracker.initialized) {
      const p = tracker.pose;
      const r = tracker.result;
      this.lastMapPose = {
        rotation: Array.from(p.rotation),
        translation: Array.from(p.translation),
        inlierCount: r.inlierCount,
        meanReprojectionErrorPx: r.meanReprojectionErrorPx,
        landmarkCount: tracker.map.size,
        mapFrameId: tracker.mapFrameId,
        framesSinceTracked: tracker.framesSinceTracked,
        cameraCenter: Array.from(tracker.cameraCenter()),
        deltaTranslation: r.poseDeltaTranslation,
        deltaRotationDeg: r.poseDeltaRotationDeg,
        translationHeld: r.translationHeld,
        jumpRejected: r.jumpRejected,
        source: this.lastPoseSource,
      };
    } else {
      this.lastMapPose = null;
    }
    // Remember this frame's relative rotation for the next prior.
    this.prevFrameRotation = this.lastRelative && this.lastRelative.model !== "none" ? this.lastRelative.rotation : null;
  }

  /**
   * Phase 5: turn relocalized keyframe observations into live tracks so the
   * following PnP / LK continue from them. Existing tracks observing the
   * same landmark or sitting on the same spot are merged.
   */
  private injectRelocalizedTracks(
    obs: { landmarkId: number; x: number; y: number }[],
    frameId: number,
    pose: { rotation: Float64Array; translation: Float64Array },
  ): void {
    const minDist = this.config.features.minDistance;
    const minDistSq = minDist * minDist;
    const byLandmark = new Map<number, Track>();
    for (const t of this.tracks) if (t.landmarkId >= 0) byLandmark.set(t.landmarkId, t);
    const anchorPose = { rotation: Float64Array.from(pose.rotation), translation: Float64Array.from(pose.translation) };
    let added = 0;
    for (const o of obs) {
      const existing = byLandmark.get(o.landmarkId);
      if (existing) {
        existing.x = o.x;
        existing.y = o.y;
        continue;
      }
      // Reuse a nearby landmark-less track if there is one.
      let near: Track | null = null;
      for (const t of this.tracks) {
        if (t.landmarkId >= 0) continue;
        const dx = t.x - o.x;
        const dy = t.y - o.y;
        if (dx * dx + dy * dy < minDistSq) {
          near = t;
          break;
        }
      }
      if (near) {
        near.x = o.x;
        near.y = o.y;
        near.landmarkId = o.landmarkId;
        near.anchorFrame = frameId;
        near.anchorX = o.x;
        near.anchorY = o.y;
        near.anchorPose = anchorPose;
        byLandmark.set(o.landmarkId, near);
        continue;
      }
      if (this.tracks.length >= this.config.features.maxFeatures) continue;
      this.tracks.push({
        id: this.nextTrackId++,
        x: o.x,
        y: o.y,
        prevX: o.x,
        prevY: o.y,
        age: 1,
        score: 0,
        inlier: true,
        outlierStreak: 0,
        refX: o.x,
        refY: o.y,
        refFrame: -1,
        landmarkId: o.landmarkId,
        anchorFrame: frameId,
        anchorX: o.x,
        anchorY: o.y,
        anchorPose,
        planePoint: null,
        planeStreak: 0,
        planeOutliers: 0,
        offPlane: false,
      });
      added++;
    }
    void added;
  }

  /** True when the track observes a landmark that the plane detector counted as a plane inlier. */
  private observesPlaneLandmark(t: Track): boolean {
    if (t.landmarkId < 0) return false;
    const lm = this.mapTracker.map.get(t.landmarkId);
    return !!lm && lm.planeInlier;
  }

  /** Median pixel displacement of landmark tracks relative to the last keyframe. */
  private medianParallaxSinceLastKeyframe(): number {
    const kfs = this.relocalizer.keyframes;
    if (kfs.length === 0) return Number.POSITIVE_INFINITY;
    const last = kfs[kfs.length - 1];
    const pos = new Map<number, { x: number; y: number }>();
    for (const o of last.observations) pos.set(o.landmarkId, o);
    const d: number[] = [];
    for (const t of this.tracks) {
      if (t.landmarkId < 0) continue;
      const p = pos.get(t.landmarkId);
      if (p) d.push(Math.hypot(t.x - p.x, t.y - p.y));
    }
    if (d.length === 0) return Number.POSITIVE_INFINITY;
    d.sort((a, b) => a - b);
    return d[d.length >> 1];
  }

  /** Phase 3: RANSAC plane on the landmarks, horizontality via gravity when available. */
  private updatePlane(input: VisionInput): void {
    const tracker = this.mapTracker;
    if (!tracker.initialized) {
      this.lastPlane = null;
      this.packedLandmarkCount = 0;
      return;
    }
    const cfg = this.config.plane;
    const { points, ids } = tracker.map.collect(cfg.minLandmarkObservations);
    const n = ids.length;

    // Gravity (camera frame of the current frame) → map frame: g_map = Rᵀ g_cam.
    let up: Float64Array | null = null;
    if (input.gravity && input.gravity.length === 3) {
      const g = input.gravity;
      const r = tracker.pose.rotation;
      up = new Float64Array([
        r[0] * g[0] + r[3] * g[1] + r[6] * g[2],
        r[1] * g[0] + r[4] * g[1] + r[7] * g[2],
        r[2] * g[0] + r[5] * g[1] + r[8] * g[2],
      ]);
    }
    const candidate = this.planeDetector.update(points, ids, n, up);
    this.lastPlane = candidate
      ? {
          normal: candidate.normal,
          d: candidate.d,
          center: candidate.center,
          inlierCount: candidate.inlierCount,
          rmsResidual: candidate.rmsResidual,
          areaEstimate: candidate.areaEstimate,
          horizontalness: candidate.horizontalness,
          horizontal: candidate.horizontal,
          confidence: candidate.confidence,
          stableFrames: candidate.stableFrames,
          found: candidate.found,
          usedGravity: candidate.usedGravity,
        }
      : null;

    // Flag plane inliers (debug rendering + "confirmed on plane" for lifting).
    const inlierSet = candidate ? new Set(candidate.inlierIds) : null;
    for (const lm of tracker.map.values()) lm.planeInlier = inlierSet ? inlierSet.has(lm.id) : false;

    // Anchor the world plane the first time a plane is found (修正指示書 §6):
    // from here on it is fixed and the camera pose is solved relative to it.
    if (
      candidate?.found &&
      !this.planeTracker.anchored &&
      this.config.planeTracking.enabled &&
      tracker.result.tracked
    ) {
      this.planeTracker.setAnchor(candidate, tracker.pose, input.frameId);
      this.planeTracker.lift(this.tracks, tracker.pose, input.intrinsics, (t) => this.observesPlaneLandmark(t));
      this.lastPlaneAnchor = {
        normal: Array.from(candidate.normal),
        d: candidate.d,
        center: Array.from(candidate.center),
        frameId: input.frameId,
      };
      this.lastPlanePose = toPlanePoseOutput(this.planeTracker.result);
    }
    const total = tracker.map.size;
    if (this.packedLandmarks.length < total * LANDMARK_STRIDE) {
      this.packedLandmarks = new Float32Array(Math.max(total, 256) * LANDMARK_STRIDE);
    }
    let i = 0;
    for (const lm of tracker.map.values()) {
      const o = i * LANDMARK_STRIDE;
      this.packedLandmarks[o] = lm.position[0];
      this.packedLandmarks[o + 1] = lm.position[1];
      this.packedLandmarks[o + 2] = lm.position[2];
      this.packedLandmarks[o + 3] = lm.planeInlier ? 1 : 0;
      i++;
    }
    this.packedLandmarkCount = i;
  }

  /** LK + forward-backward. Mutates `this.tracks` (drops failures). Returns surviving count. */
  private trackExisting(): number {
    const tracks = this.tracks;
    const n = tracks.length;
    if (this.pointBuf.length < n * 2) this.pointBuf = new Float32Array(n * 2);
    if (this.trackResult.status.length < n) this.trackResult = allocResult(n);
    const pts = this.pointBuf;
    for (let i = 0; i < n; i++) {
      pts[i * 2] = tracks[i].x;
      pts[i * 2 + 1] = tracks[i].y;
    }
    // Constant-velocity prediction as the LK starting point: a track that
    // moved (dx, dy) last frame is searched around x + dx first, which keeps
    // fast motion within the pyramid's capture range.
    let guesses: Float32Array | null = null;
    if (this.config.tracker.predictMotion) {
      if (this.guessBuf.length < n * 2) this.guessBuf = new Float32Array(n * 2);
      guesses = this.guessBuf;
      const maxD = this.config.tracker.maxDisplacement;
      for (let i = 0; i < n; i++) {
        const t = tracks[i];
        let vx = t.age > 0 ? t.x - t.prevX : 0;
        let vy = t.age > 0 ? t.y - t.prevY : 0;
        const v = Math.hypot(vx, vy);
        if (v > maxD) {
          vx *= maxD / v;
          vy *= maxD / v;
        }
        guesses[i * 2] = Math.min(this.width - 1, Math.max(0, t.x + vx));
        guesses[i * 2 + 1] = Math.min(this.height - 1, Math.max(0, t.y + vy));
      }
    }
    const res = this.tracker.track(this.prevPyramid, this.curPyramid, pts, n, this.trackResult, guesses);

    const survivors: Track[] = [];
    for (let i = 0; i < n; i++) {
      if (res.status[i] !== TrackStatus.OK) continue;
      const t = tracks[i];
      t.prevX = t.x;
      t.prevY = t.y;
      t.x = res.positions[i * 2];
      t.y = res.positions[i * 2 + 1];
      t.age++;
      t.inlier = true;
      survivors.push(t);
    }
    this.tracks = survivors;
    return survivors.length;
  }

  /** Homography RANSAC on prev→cur positions. Drops outliers. Returns inlier count. */
  private rejectOutliers(): number {
    const tracks = this.tracks;
    const n = tracks.length;
    const { c1x, c1y, c2x, c2y } = this;
    for (let i = 0; i < n; i++) {
      c1x[i] = tracks[i].prevX;
      c1y[i] = tracks[i].prevY;
      c2x[i] = tracks[i].x;
      c2y[i] = tracks[i].y;
    }
    const r = ransacHomography(c1x, c1y, c2x, c2y, n, this.config.ransac, this.rng);
    this.lastHomographyInliers = r.inlierCount;
    this.lastRansacError = r.meanError;
    if (r.homography === null && r.inlierCount === n) {
      // Too few correspondences to run RANSAC: keep all.
      return n;
    }
    // Outliers are dropped only after `outlierFramesToDrop` consecutive
    // frames: one-off deviations (parallax in non-planar scenes) survive,
    // repeated ones (a track that jumped) do not.
    const dropAfter = this.config.ransac.outlierFramesToDrop;
    const kept: Track[] = [];
    let inlierCount = 0;
    for (let i = 0; i < n; i++) {
      const t = tracks[i];
      if (r.inliers[i]) {
        t.outlierStreak = 0;
        t.inlier = true;
        inlierCount++;
        kept.push(t);
      } else {
        t.outlierStreak++;
        t.inlier = false;
        if (t.outlierStreak < dropAfter) kept.push(t);
      }
    }
    this.tracks = kept;
    return inlierCount;
  }

  /** Detect new FAST corners away from existing tracks and add them. */
  private replenish(gray: Uint8Array): void {
    const cfg = this.config.features;
    const wanted = cfg.maxFeatures - this.tracks.length;
    if (wanted <= 0) return;

    this.buildMask();
    const corners = this.detector.detect(gray, {
      mask: this.mask,
      wanted,
      cellCounts: this.cellCounts,
    });
    for (const c of corners) {
      this.tracks.push({
        id: this.nextTrackId++,
        x: c.x,
        y: c.y,
        prevX: c.x,
        prevY: c.y,
        age: 0,
        score: c.score,
        inlier: true,
        outlierStreak: 0,
        refX: c.x,
        refY: c.y,
        refFrame: -1,
        landmarkId: -1,
        anchorFrame: -1,
        anchorX: c.x,
        anchorY: c.y,
        anchorPose: null,
        planePoint: null,
        planeStreak: 0,
        planeOutliers: 0,
        offPlane: false,
      });
    }
  }

  private lastPoseDelta = 0;

  /**
   * Phase 2: estimate the camera pose relative to the reference frame from
   * the tracks that still link to it, and renew the reference when needed.
   */
  private estimatePose(input: VisionInput): PoseOutput | null {
    const cfg = this.config.pose;
    const frameId = input.frameId;

    this.lastRelative = null;
    if (this.referenceFrameId < 0) {
      // First frame with features becomes the reference.
      if (this.tracks.length >= cfg.minCorrespondences) this.renewReference(frameId, mat3Identity());
      this.lastPose = null;
      return null;
    }

    // Gather reference ↔ current correspondences.
    const { r1x, r1y, r2x, r2y } = this;
    let n = 0;
    for (const t of this.tracks) {
      if (t.refFrame !== this.referenceFrameId) continue;
      if (n >= r1x.length) break;
      r1x[n] = t.refX;
      r1y[n] = t.refY;
      r2x[n] = t.x;
      r2y[n] = t.y;
      n++;
    }

    if (n < cfg.minReferenceTracks || n < cfg.minCorrespondences) {
      // Lost the link to the reference: start a new reference, carrying the
      // last relative rotation into the accumulated rotation.
      if (this.tracks.length >= cfg.minCorrespondences) {
        this.renewReference(frameId, this.lastRelativeRotation);
      } else {
        this.lastPose = null;
      }
      return this.lastPose;
    }

    const rel: RelativePose = this.poseEstimator.estimate(
      r1x, r1y, r2x, r2y, n, input.intrinsics, this.lastPlaneNormal,
    );
    this.lastRelative = rel;
    this.lastRelativeRefFrame = this.referenceFrameId;

    if (rel.model === "none") {
      // Keep the previous pose (if any) rather than flickering to identity.
      return this.lastPose;
    }

    const accumulated = mat3Multiply(rel.rotation, this.referenceRotation);
    this.lastPoseDelta = (rotationDistance(this.lastAccumulatedRotation, accumulated) * 180) / Math.PI;
    this.lastAccumulatedRotation = accumulated;
    this.lastRelativeRotation = rel.rotation;
    if (rel.planeNormal) this.lastPlaneNormal = rel.planeNormal;

    const q = rotationToQuaternion(accumulated);
    this.lastPose = {
      rotation: Array.from(accumulated),
      quaternion: Array.from(q),
      translationDirection: Array.from(rel.translationDirection),
      relativeRotation: Array.from(rel.rotation),
      parallaxPx: rel.parallaxPx,
      model: rel.model,
      confidence: rel.confidence,
      translationConfidence: rel.translationConfidence,
      correspondences: n,
      inlierCount: rel.inlierCount,
      referenceFrameId: this.referenceFrameId,
      planeNormal: rel.planeNormal ? Array.from(rel.planeNormal) : null,
    };

    // Renew the reference once the baseline is large (keeps the two-view
    // problem bounded; Phase 3 will turn these into keyframes).
    if (rel.parallaxPx > cfg.maxReferenceParallaxPx && rel.confidence > 0.5) {
      this.renewReference(frameId, rel.rotation);
    }
    return this.lastPose;
  }

  /** Make the current frame the reference for all live tracks. */
  private renewReference(frameId: number, relativeRotation: Mat3): void {
    this.referenceRotation = mat3Multiply(relativeRotation, this.referenceRotation);
    this.lastRelativeRotation = mat3Identity();
    this.referenceFrameId = frameId;
    for (const t of this.tracks) {
      t.refX = t.x;
      t.refY = t.y;
      t.refFrame = frameId;
    }
    // The plane normal is expressed in the reference frame: rotate it.
    if (this.lastPlaneNormal) {
      const r = relativeRotation;
      const nrm = this.lastPlaneNormal;
      this.lastPlaneNormal = new Float64Array([
        r[0] * nrm[0] + r[1] * nrm[1] + r[2] * nrm[2],
        r[3] * nrm[0] + r[4] * nrm[1] + r[5] * nrm[2],
        r[6] * nrm[0] + r[7] * nrm[1] + r[8] * nrm[2],
      ]);
    }
  }

  /** Stamp a disc of radius minDistance around each live track into the mask; count per cell. */
  private buildMask(): void {
    const cfg = this.config.features;
    const { width: w, height: h, mask, cellCounts } = this;
    mask.fill(0);
    cellCounts.fill(0);
    const r = cfg.minDistance;
    const r2 = r * r;
    const cellW = w / cfg.gridCols;
    const cellH = h / cfg.gridRows;
    for (const t of this.tracks) {
      const cx = Math.round(t.x);
      const cy = Math.round(t.y);
      const gx = Math.min(cfg.gridCols - 1, Math.max(0, (t.x / cellW) | 0));
      const gy = Math.min(cfg.gridRows - 1, Math.max(0, (t.y / cellH) | 0));
      cellCounts[gy * cfg.gridCols + gx]++;
      const y0 = Math.max(0, cy - r);
      const y1 = Math.min(h - 1, cy + r);
      for (let y = y0; y <= y1; y++) {
        const dy = y - cy;
        const span = Math.floor(Math.sqrt(Math.max(0, r2 - dy * dy)));
        const x0 = Math.max(0, cx - span);
        const x1 = Math.min(w - 1, cx + span);
        mask.fill(1, y * w + x0, y * w + x1 + 1);
      }
    }
  }

  /** Last computed quality (for debugging / logging). */
  get quality(): TrackingQuality {
    return this.lastQuality;
  }

  /** Inliers reported by the last RANSAC run. */
  get homographyInliers(): number {
    return this.lastHomographyInliers;
  }
}

function now(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

function toPlanePoseOutput(r: PlaneTracker["result"]): PlanePoseOutput {
  return {
    tracked: r.tracked,
    inlierCount: r.inlierCount,
    candidateCount: r.candidateCount,
    confirmedCount: r.confirmedCount,
    inlierRatio: r.inlierRatio,
    reprojectionErrorPx: r.meanErrorPx,
    confidence: r.confidence,
  };
}

function emptyReloc(): RelocalizationOutput {
  return { keyframes: 0, attempt: "none", inlierCount: 0, candidatesTried: 0, lastSuccessFrame: -1, successCount: 0 };
}
