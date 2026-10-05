import type { ARConfig } from "../ar/ARConfig";
import { TrackingState, TrackingStateMachine } from "../ar/ARState";
import { FeatureDetector } from "./FeatureDetector";
import { FeatureTracker, TrackStatus, allocResult, type TrackResult } from "./FeatureTracker";
import { ImagePyramid } from "./ImagePyramid";
import { ransacHomography, type Rng } from "./OutlierRejection";
import { MapTracker } from "./MapTracker";
import { PlaneDetector } from "./PlaneDetector";
import { PlaneRecovery, emptyPlaneRecovery, significantTwoViewMotion } from "./PlaneRecovery";
import { PlaneTracker } from "./PlaneTracker";
import { PoseEstimator, type RelativePose } from "./PoseEstimator";
import { Relocalizer, type RelocalizationDiagnostics, type RelocalizationResult } from "./Relocalizer";
import { isJumpRejection, poseDelta } from "./PoseValidation";
import { computeTrackingConfidence, emptyQuality, type TrackingQuality } from "./TrackingQuality";
import {
  LANDMARK_STRIDE,
  packTracks,
  type MapPoseOutput,
  type MotionDiagnostics,
  type MotionLevel,
  type PlaneAnchorOutput,
  type PlaneOutput,
  type PlanePoseOutput,
  type PlaneRecoveryDiagnostics,
  type PoseOutput,
  type RelocalizationOutput,
  type Track,
  type VisionInput,
  type VisionOutput,
} from "./types";
import { type Mat3, mat3Identity, mat3Multiply, mat3TransformPoint } from "../math/Matrix";
import { transpose3 } from "../math/Decomposition";
import { rotationDistance, rotationToQuaternion, type RigidTransform } from "../math/Pose";

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
  private lastHomographyInlierRatio = 0;
  private lastRansacError = 0;
  /** Fast-motion state (v7): previous frame's median displacement decides this frame's level. */
  private lastMedianDisplacementPx = 0;
  private lastMotion: MotionDiagnostics = emptyMotion();
  private readonly predictScratch = new Float64Array(2);
  private dispScratch = new Float32Array(0);
  /** Frame-to-frame pixel homography (prev → cur) of the last frame, for landmark re-association while lost. */
  private lastImageMotion: Mat3 | null = null;

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
  /**
   * Plane recovery after fast motion (v11): re-seeds the plane detector from
   * the landmarks in view while the map, pose and world stay untouched.
   */
  private readonly planeRecovery = new PlaneRecovery();
  private lastPlaneRecovery: PlaneRecoveryDiagnostics = emptyPlaneRecovery();
  private planeRecoveryRequested = false;
  /** Previous frame's two-view estimate, for the parallax-crossing trigger (v11.1 §7). */
  private prevTwoView: { parallaxPx: number; referenceFrameId: number } | null = null;

  // Plane-anchored tracking (修正指示書): the first found plane is fixed and
  // the camera pose is solved relative to it (depth-free).
  private readonly planeTracker: PlaneTracker;
  private lastPlaneAnchor: PlaneAnchorOutput | null = null;
  private lastPlanePose: PlanePoseOutput | null = null;
  private lastPoseSource: MapPoseOutput["source"] = "propagated";
  /** Pose source of the last ~50 frames (debug, v3 §21). */
  private sourceHistory = "";
  /** Failed relocalization attempts since the camera was last located (world-reset condition, v3 §16). */
  private relocAttemptsSinceLost = 0;

  // Phase 5: keyframes + relocalization.
  private readonly relocalizer: Relocalizer;
  private relocStatus: RelocalizationOutput = emptyReloc();
  private relocSuccessCount = 0;
  private relocLastSuccessFrame = -1;
  /** Validated candidate waiting to be reproduced in the next frame (v5 §10–§11). */
  private pendingReloc: { result: RelocalizationResult; frameId: number; confirmations: number } | null = null;
  /** Map-PnP-vs-relocalized-pose watch for the frames after a relocalization (v5 §16–§17). */
  private relocMonitor: {
    pose: RigidTransform;
    framesLeft: number;
    maxDeltaTranslation: number;
    maxDeltaRotationDeg: number;
    inconsistent: boolean;
  } | null = null;
  private lastRelocalized = false;
  /**
   * World tracking established for the current map (first PLANE_FOUND, v10
   * §5–§7). Gates RELOCALIZING and the relocalizer: before it, a lost map is
   * dropped and the scan restarts where the camera looks now.
   */
  private worldEstablished = false;
  /** Stage counters of the most recent relocalization attempt of the current lost episode (v6). */
  private relocDiagnostics: RelocalizationDiagnostics | null = null;
  private relocDiagnosticsAge = 0;

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
    this.lastMedianDisplacementPx = 0;
    this.lastMotion = emptyMotion();
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
    this.planeRecovery.reset();
    this.lastPlaneRecovery = emptyPlaneRecovery();
    this.planeRecoveryRequested = false;
    this.prevTwoView = null;
    this.planeTracker.reset(this.tracks);
    this.lastPlaneAnchor = null;
    this.lastPlanePose = null;
    this.lastPoseSource = "propagated";
    this.sourceHistory = "";
    this.relocAttemptsSinceLost = 0;
    this.lastMapPose = null;
    this.lastPlane = null;
    this.packedLandmarkCount = 0;
    this.relocalizer.reset();
    this.relocStatus = emptyReloc();
    this.relocSuccessCount = 0;
    this.relocLastSuccessFrame = -1;
    this.pendingReloc = null;
    this.relocMonitor = null;
    this.lastRelocalized = false;
    this.relocDiagnostics = null;
    this.worldEstablished = false;
  }

  /** World tracking established for the current map (v10). */
  get isWorldEstablished(): boolean {
    return this.worldEstablished;
  }

  /** Plane recovery diagnostics of the last frame (v11 §23). */
  get planeRecoveryDiagnostics(): PlaneRecoveryDiagnostics {
    return this.lastPlaneRecovery;
  }

  /**
   * Ask for a plane recovery on the next frame (reason `manual`, v11 §23).
   * Honoured only under the same conditions as the automatic trigger: a
   * healthy map and no established world.
   */
  requestPlaneRecovery(): void {
    this.planeRecoveryRequested = true;
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
      this.lastHomographyInlierRatio = 0;
      this.lastRansacError = 0;
      this.lastImageMotion = null;
    }
    if (trackedCount === 0) {
      // Nothing tracked (first frame or a total loss): no motion measurement.
      this.lastMotion = { ...emptyMotion(), level: this.motionLevel(), trackedBefore: previousCount };
      this.lastMedianDisplacementPx = 0;
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

    // 6b. Plane recovery decision (v11 §5–§6, v11.1 §3–§10): a significant
    // motion (fast motion level, or a confident two-view parallax crossing)
    // with a healthy map and no world yet → forget the plane-specific state
    // once, so the plane is searched in the view the camera has now. Map /
    // pose untouched; an active recovery is never restarted.
    const mapHealthy = this.mapHealthyThisFrame();
    const twoViewMotion = significantTwoViewMotion(pose, this.prevTwoView, {
      parallaxPx: this.config.pose.fullConfidenceParallaxPx,
      minConfidence: this.config.landmarks.initMinTranslationConfidence,
      minInliers: this.config.pose.minCorrespondences,
    });
    this.prevTwoView = pose ? { parallaxPx: pose.parallaxPx, referenceFrameId: pose.referenceFrameId } : null;
    const recoveryStarted = this.planeRecovery.update({
      motionLevel: this.lastMotion.level,
      twoViewMotion,
      mapInitialized: this.mapTracker.initialized,
      mapTracked: this.mapTracker.initialized && this.mapTracker.result.tracked && this.mapTracker.framesSinceTracked === 0,
      mapLost: this.mapTracker.initialized && this.mapTracker.framesSinceTracked > this.config.state.mapLostFrameTolerance,
      mapInliers: this.mapTracker.selection.mapInlierCount,
      requiredInliers: this.config.landmarks.minPnPInliers,
      poseFinite: mapHealthy.poseFinite,
      worldEstablished: this.worldEstablished,
      manual: this.planeRecoveryRequested,
      timestamp: input.timestamp,
    });
    this.planeRecoveryRequested = false;
    // Exactly once per recovery (v11.1 §3–§5): `update` returns true only
    // in the frame the recovery starts.
    if (recoveryStarted) this.planeDetector.resetForRecovery();

    // 7. Plane detection (Phase 3)
    this.updatePlane(input);
    this.lastPlaneRecovery = this.planeRecoveryDiagnosticsFor(input, mapHealthy.healthy, recoveryStarted);
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
      worldEstablished: this.worldEstablished,
    });
    // World tracking is established the first time a plane is found for this
    // map (the session anchors the world in that same frame, v10 §5–§7); from
    // here on a lost map is something to relocalize into.
    if (state === TrackingState.PLANE_FOUND && this.mapTracker.initialized) this.worldEstablished = true;

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
      motion: this.lastMotion,
      worldEstablished: this.worldEstablished,
      planeRecovery: this.lastPlaneRecovery,
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
      ...emptyReloc(),
      keyframes: this.relocalizer.count,
      lastSuccessFrame: this.relocLastSuccessFrame,
      successCount: this.relocSuccessCount,
      postDeltaTranslation: this.relocMonitor?.maxDeltaTranslation ?? 0,
      postDeltaRotationDeg: this.relocMonitor?.maxDeltaRotationDeg ?? 0,
      postInconsistent: this.relocMonitor?.inconsistent ?? false,
      // The last attempt's stage counters stay visible between attempts (v6).
      diagnostics: this.relocDiagnostics,
      framesSinceAttempt: this.relocDiagnostics ? ++this.relocDiagnosticsAge : -1,
    };
    let relocalizedNow = false;
    let monitoredThisFrame = false;

    if (!tracker.initialized) {
      if (this.lastRelative && this.lastRelativeRefFrame >= 0) {
        if (tracker.tryInitialize(this.tracks, this.lastRelative, this.lastRelativeRefFrame, frameId, k)) {
          this.planeDetector.reset();
          this.planeRecovery.reset();
          this.planeTracker.reset(this.tracks);
          this.lastPlaneAnchor = null;
          this.relocalizer.reset();
          this.worldEstablished = false;
          // The initialization frame is the first keyframe.
          this.relocalizer.create(this.curPyramid, tracker.pose, this.tracks, frameId, input.timestamp);
          this.relocStatus.keyframes = this.relocalizer.count;
        }
      }
    } else {
      // Phase 5: relocalize when the camera was not located in the previous frame.
      //
      //   keyframe match → PnP → global validation (Relocalizer) → candidate
      //     → confirmation (v5 §10–§11) → applyRelocalization → canonical pose
      //
      // A candidate is never written to the pose unvalidated (v5 §3). The
      // temporal gate is not used here (a correct return after a loss can be
      // far from the held pose, v5 §9); the delta is diagnostic only.
      const rc = this.config.relocalization;
      const pending = this.pendingReloc;
      // Relocalization only returns to an *established* world (v10 §2, §9):
      // before a plane fixed the world there is nothing to go back to, the
      // user may be scanning somewhere else on purpose, and the map is
      // re-initialized there instead (reset below).
      const scheduled =
        this.worldEstablished &&
        tracker.framesSinceTracked >= rc.startAfterLostFrames &&
        (tracker.framesSinceTracked - rc.startAfterLostFrames) % rc.attemptEveryNFrames === 0;
      if (tracker.framesSinceTracked === 0 || !this.worldEstablished) {
        this.pendingReloc = null;
      }
      if (tracker.framesSinceTracked > 0 && this.worldEstablished && (pending !== null || scheduled)) {
        const tr0 = now();
        const r = this.relocalizer.relocalize(this.curPyramid, tracker.map, k);
        this.timing.reloc = now() - tr0;
        const d = r.pose ? poseDelta(r.pose, tracker.pose) : { translation: 0, rotationDeg: 0 };
        this.relocDiagnostics = r.diagnostics;
        this.relocDiagnosticsAge = 0;
        this.relocStatus = {
          ...this.relocStatus,
          diagnostics: r.diagnostics,
          framesSinceAttempt: 0,
          attempt: r.success ? "candidate" : "fail",
          inlierCount: r.inlierCount,
          candidatesTried: r.candidatesTried,
          reason: r.reason,
          rejectCode: r.rejectCode,
          meanReprojectionErrorPx: r.meanReprojectionErrorPx,
          matchScore: r.matchScore,
          inlierRatio: r.inlierRatio,
          spatialCells: r.spatialCells,
          keyframeId: r.keyframeId,
          jumpTranslation: d.translation,
          jumpRotationDeg: d.rotationDeg,
        };
        let apply = false;
        if (r.success && r.pose) {
          const immediate =
            rc.confirmationFrames <= 0 || (r.inlierCount >= rc.immediateInliers && r.meanReprojectionErrorPx <= rc.immediateMaxErrorPx);
          if (pending) {
            // Confirmation: the new candidate must land where the pending one did.
            const c = poseDelta(r.pose, pending.result.pose!);
            const depth = tracker.sceneDepth;
            const near =
              c.translation <= rc.confirmTranslationDepthRatio * Math.max(depth, 1e-9) && c.rotationDeg <= rc.confirmRotationDeg;
            if (near) {
              pending.confirmations++;
              apply = pending.confirmations >= rc.confirmationFrames;
            } else {
              this.relocStatus.reason = `confirmation failed: Δ ${c.translation.toFixed(3)} / ${c.rotationDeg.toFixed(1)}° vs frame ${pending.frameId}`;
              this.relocStatus.rejectCode = "confirmation_failed";
              this.relocStatus.attempt = "fail";
              this.relocAttemptsSinceLost++;
              // The newer candidate starts its own confirmation.
              this.pendingReloc = { result: r, frameId, confirmations: 0 };
            }
          } else if (immediate) {
            apply = true;
          } else {
            this.pendingReloc = { result: r, frameId, confirmations: 0 };
          }
        } else {
          this.relocAttemptsSinceLost++;
          if (pending) {
            // A candidate that cannot be reproduced in the next frame is dropped.
            this.relocStatus.rejectCode = "confirmation_failed";
            this.relocStatus.reason = `confirmation failed: ${r.reason ?? "no candidate"}`;
            this.pendingReloc = null;
          }
        }
        if (apply && r.pose) {
          // Re-seeds the canonical pose; from the next frame on the map PnP is
          // gated against it again and compared with it (monitor below).
          tracker.applyRelocalization(r.pose);
          this.injectRelocalizedTracks(r.tracks, frameId, r.pose);
          rotationPrior = null; // the relocalized pose is the prior
          this.pendingReloc = null;
          relocalizedNow = true;
          this.relocSuccessCount++;
          this.relocLastSuccessFrame = frameId;
          this.relocStatus.attempt = "success";
          this.relocStatus.lastSuccessFrame = frameId;
          this.relocStatus.successCount = this.relocSuccessCount;
          this.relocMonitor = {
            pose: { rotation: Float64Array.from(r.pose.rotation), translation: Float64Array.from(r.pose.translation) },
            framesLeft: rc.postRelocMonitorFrames,
            maxDeltaTranslation: 0,
            maxDeltaRotationDeg: 0,
            inconsistent: false,
          };
          this.relocStatus.postDeltaTranslation = 0;
          this.relocStatus.postDeltaRotationDeg = 0;
          this.relocStatus.postInconsistent = false;
        }
      }

      // Plane-relative pose (experimental): a *candidate* for the canonical
      // pose, validated inside MapTracker like the landmark PnP (v3 §1–§6).
      const pt = this.config.planeTracking;
      let planeRes = this.planeTracker.result;
      let external = null;
      if (pt.enabled && this.planeTracker.anchored) {
        const prior = rotationPrior
          ? { rotation: mat3Multiply(rotationPrior, tracker.pose.rotation), translation: tracker.pose.translation }
          : tracker.pose;
        planeRes = this.planeTracker.update(this.tracks, prior, k);
        if (planeRes.tracked && planeRes.pose) {
          external = { pose: planeRes.pose, inlierCount: planeRes.inlierCount, meanErrorPx: planeRes.meanErrorPx };
        }
      }

      const res = tracker.update(this.tracks, frameId, k, rotationPrior, external, this.lastImageMotion);
      const sel = tracker.selection;
      this.lastPoseSource = res.tracked ? sel.source : "propagated";
      // Post-relocalization consistency (v5 §16–§17): the map PnP of the
      // following frames must agree with the relocalized pose up to the
      // camera's own motion; a jump-rejected or far-off solve means the
      // relocalization itself is suspect.
      const mon = this.relocMonitor;
      monitoredThisFrame = mon !== null;
      if (mon && !relocalizedNow) {
        mon.framesLeft--;
        if (sel.map) {
          const dm = poseDelta(tracker.pose, mon.pose);
          mon.maxDeltaTranslation = Math.max(mon.maxDeltaTranslation, dm.translation);
          mon.maxDeltaRotationDeg = Math.max(mon.maxDeltaRotationDeg, dm.rotationDeg);
          if (isJumpRejection(sel.mapRejection)) mon.inconsistent = true;
        }
        this.relocStatus.postDeltaTranslation = mon.maxDeltaTranslation;
        this.relocStatus.postDeltaRotationDeg = mon.maxDeltaRotationDeg;
        this.relocStatus.postInconsistent = mon.inconsistent;
        if (mon.framesLeft <= 0) this.relocMonitor = null;
      }
      if (res.tracked) {
        this.relocAttemptsSinceLost = 0;
        if (!relocalizedNow) {
          // Tracking again: the lost-episode diagnostics are over.
          this.relocDiagnostics = null;
          this.relocStatus.diagnostics = null;
          this.relocStatus.framesSinceAttempt = -1;
        }
        if (external) {
          // Plane bookkeeping (streaks, off-plane flags, confidence) is
          // updated only when the plane candidate passed validation, and
          // always against the *canonical* pose: a rejected plane pose must
          // not strengthen the plane state (v4 §12–§13). A candidate held
          // back only by the source cooldown is valid.
          const rej = sel.planeRejection;
          if (!rej || rej.code === "source_cooldown") this.planeTracker.commit(this.tracks, tracker.pose, k);
        }
        if (this.planeTracker.anchored) {
          // New plane points only from a trusted *map* pose, never from the
          // plane pose itself (no pose → point → pose feedback, v3 §8–§11),
          // and not right after a source switch.
          const good =
            sel.source === "map" &&
            sel.mapInlierCount >= pt.liftMinInliers &&
            res.meanReprojectionErrorPx <= pt.liftMaxMeanErrorPx &&
            tracker.framesSinceSourceSwitch >= pt.liftCooldownFrames;
          if (good) this.planeTracker.lift(this.tracks, tracker.pose, k, (t) => this.observesPlaneLandmark(t));
        }
        // Keyframe policy (spec §35).
        const par = this.medianParallaxSinceLastKeyframe();
        if (this.relocalizer.shouldCreate(tracker.pose, frameId, res.inlierCount, par)) {
          this.relocalizer.create(this.curPyramid, tracker.pose, this.tracks, frameId, input.timestamp);
          this.relocStatus.keyframes = this.relocalizer.count;
        }
      } else if (
        this.worldEstablished
          ? tracker.framesSinceTracked > cfg.lostResetFrames &&
            this.relocAttemptsSinceLost >= this.config.relocalization.minAttemptsBeforeReset
          : tracker.framesSinceTracked > cfg.preWorldLostResetFrames
      ) {
        // Established world: reset is the last resort (v3 §12, §16) — long
        // loss AND enough failed relocalization attempts; a short loss keeps
        // map, world and objects while the camera holds its last good pose.
        // No world yet (v10 §8–§12): the lost map is dropped quickly and the
        // scan re-initializes where the camera looks now.
        tracker.reset(this.tracks);
        this.planeDetector.reset();
        this.planeRecovery.reset();
        this.planeTracker.reset(this.tracks);
        this.lastPlaneAnchor = null;
        this.relocalizer.reset();
        this.relocAttemptsSinceLost = 0;
        this.pendingReloc = null;
        this.relocMonitor = null;
        this.relocDiagnostics = null;
        this.worldEstablished = false;
      }
      this.sourceHistory = (
        this.sourceHistory + (relocalizedNow ? "R" : res.tracked ? (sel.source === "plane" ? "P" : "M") : "·")
      ).slice(-50);
    }
    this.lastRelocalized = relocalizedNow || monitoredThisFrame;
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
        translationPredicted: r.translationPredicted,
        jumpRejected: r.jumpRejected,
        reassociated: r.reassociated,
        mapInlierCount: tracker.selection.mapInlierCount,
        planeInlierCount: tracker.selection.planeInlierCount,
        rejectReason: tracker.selection.planeReject ?? tracker.selection.mapReject,
        mapCandidate: tracker.selection.map,
        planeCandidate: tracker.selection.plane,
        gateMaxTranslation: tracker.selection.limits.maxTranslation,
        gateMaxRotationDeg: tracker.selection.limits.maxRotationDeg,
        observations: tracker.selection.observations,
        requiredInliers: tracker.selection.requiredInliers,
        recoveryMode: tracker.selection.recoveryMode,
        sourceDeltaTranslation: tracker.selection.sourceDeltaTranslation,
        sourceDeltaRotationDeg: tracker.selection.sourceDeltaRotationDeg,
        sourceHistory: this.sourceHistory,
        relocalized: this.lastRelocalized,
        triangulation: r.triangulation,
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

  /** Map health for the plane recovery (v11 §6, §32): located this frame with a finite pose. */
  private mapHealthyThisFrame(): { healthy: boolean; poseFinite: boolean } {
    const tracker = this.mapTracker;
    if (!tracker.initialized) return { healthy: false, poseFinite: false };
    const p = tracker.pose;
    let poseFinite = true;
    for (let i = 0; i < 9; i++) if (!Number.isFinite(p.rotation[i])) poseFinite = false;
    for (let i = 0; i < 3; i++) if (!Number.isFinite(p.translation[i])) poseFinite = false;
    const healthy = PlaneRecovery.mapHealthy({
      mapInitialized: true,
      mapTracked: tracker.result.tracked && tracker.framesSinceTracked === 0,
      mapInliers: tracker.selection.mapInlierCount,
      requiredInliers: this.config.landmarks.minPnPInliers,
      poseFinite,
    });
    return { healthy, poseFinite };
  }

  /**
   * Numbers for the HUD / tests (v11 §23, v11.1 §21–§24); formatted only
   * when the debug HUD is shown. Candidate facts are about *this frame's*
   * search: a candidate the detector still holds through its grace period
   * is reported separately (`previousCandidateHeld`), never as "found".
   */
  private planeRecoveryDiagnosticsFor(input: VisionInput, mapHealthy: boolean, startedThisFrame: boolean): PlaneRecoveryDiagnostics {
    const tracker = this.mapTracker;
    const cfg = this.config.plane;
    const search = this.planeDetector.lastSearch;
    const held = this.planeDetector.current;
    const candidateNow = search.stage === "candidate" && held !== null;
    const rec = this.planeRecovery;
    const stableFrames = this.planeDetector.stableFrameCount;
    return {
      active: rec.active,
      reason: rec.reason,
      state: rec.state(startedThisFrame, this.lastMotion.level, candidateNow, stableFrames),
      mapHealthy,
      mapInliers: tracker.initialized ? tracker.selection.mapInlierCount : 0,
      trackedFeatures: this.tracks.length,
      seedCandidates: tracker.initialized ? tracker.map.countRecent(input.frameId, cfg.recoverySeedMaxAgeFrames) : 0,
      seededPoints: search.points,
      searchPoints: search.points,
      bestInliers: search.bestInliers,
      requiredInliers: cfg.minInliers,
      searchStage: search.stage,
      candidateFound: candidateNow,
      // "Committed" = this frame's candidate is horizontal, so the stability
      // streak counts toward PLANE_FOUND (a non-horizontal candidate never commits).
      candidateCommitted: candidateNow && held!.horizontal,
      previousCandidateHeld: !candidateNow && held !== null,
      stableFrames,
      requiredStableFrames: cfg.stableFramesRequired,
      recoveryElapsedMs: rec.elapsedMs(input.timestamp),
      recoveries: rec.count,
    };
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
    // Plane seeds (v11 §9–§10, §40): every mature landmark normally; during a
    // recovery only those seen as a PnP inlier within the seed window, i.e.
    // the landmarks of the view the camera has *now*. They come from the map
    // and the map pose only — never from a plane pose (§11, AC-4).
    const { points, ids } = this.planeRecovery.active
      ? tracker.map.collect(cfg.minLandmarkObservations, cfg.recoverySeedMaxAgeFrames, input.frameId)
      : tracker.map.collect(cfg.minLandmarkObservations);
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
    // A found plane ends the recovery (v11 §44): the world is established from it.
    if (candidate?.found) this.planeRecovery.finish();

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

  /** Motion level for this frame from the previous frame's median displacement (v7 §3). */
  private motionLevel(): MotionLevel {
    const cfg = this.config.tracker;
    const d = this.lastMedianDisplacementPx;
    return d >= cfg.fastMotionPx ? "fast" : d >= cfg.mediumMotionPx ? "medium" : "normal";
  }

  /**
   * LK + forward-backward. Mutates `this.tracks` (drops failures). Returns
   * surviving count.
   *
   * Fast motion (v7 §4–§8): the LK starting point is a *prediction* of where
   * each track is now — the previous frame's frame-to-frame homography
   * applied to the track when that homography was well supported (a global
   * model: pan, rotation, zoom), else the track's own last displacement —
   * and the displacement gate around that prediction is widened only while
   * the camera is measured to move fast. The pyramid itself (coarse-to-fine)
   * is unchanged; PnP / RANSAC are not touched.
   */
  private trackExisting(): number {
    const tracks = this.tracks;
    const n = tracks.length;
    const cfg = this.config.tracker;
    if (this.pointBuf.length < n * 2) this.pointBuf = new Float32Array(n * 2);
    if (this.trackResult.status.length < n) this.trackResult = allocResult(n);
    const pts = this.pointBuf;
    for (let i = 0; i < n; i++) {
      pts[i * 2] = tracks[i].x;
      pts[i * 2 + 1] = tracks[i].y;
    }

    const level = this.motionLevel();
    const searchScale = level === "fast" ? cfg.fastMotionSearchScale : level === "medium" ? cfg.mediumMotionSearchScale : 1;
    const maxD = cfg.maxDisplacement * searchScale;

    // Prediction (v7 §6–§7): the homography only when it was trustworthy in
    // the previous frame — a poorly supported one would move every track
    // wrongly at once — else per-track constant velocity.
    let guesses: Float32Array | null = null;
    let predictionMode: MotionDiagnostics["predictionMode"] = "none";
    if (cfg.predictMotion) {
      if (this.guessBuf.length < n * 2) this.guessBuf = new Float32Array(n * 2);
      guesses = this.guessBuf;
      const H = this.lastImageMotion;
      const useH =
        cfg.homographyPrediction &&
        H !== null &&
        this.lastHomographyInliers >= cfg.predictionMinInliers &&
        this.lastHomographyInlierRatio >= cfg.predictionMinInlierRatio;
      predictionMode = useH ? "homography" : "velocity";
      const out = this.predictScratch;
      for (let i = 0; i < n; i++) {
        const t = tracks[i];
        let vx = t.age > 0 ? t.x - t.prevX : 0;
        let vy = t.age > 0 ? t.y - t.prevY : 0;
        if (useH && mat3TransformPoint(H!, t.x, t.y, out)) {
          vx = out[0] - t.x;
          vy = out[1] - t.y;
        }
        const v = Math.hypot(vx, vy);
        if (v > maxD) {
          vx *= maxD / v;
          vy *= maxD / v;
        }
        guesses[i * 2] = Math.min(this.width - 1, Math.max(0, t.x + vx));
        guesses[i * 2 + 1] = Math.min(this.height - 1, Math.max(0, t.y + vy));
      }
    }
    const res = this.tracker.track(this.prevPyramid, this.curPyramid, pts, n, this.trackResult, guesses, maxD);

    const survivors: Track[] = [];
    let fbRejects = 0;
    let tooFar = 0;
    let residualSum = 0;
    let maxDisp = 0;
    const disps = this.dispScratch.length >= n ? this.dispScratch : (this.dispScratch = new Float32Array(Math.max(n, 64)));
    for (let i = 0; i < n; i++) {
      const st = res.status[i];
      if (st !== TrackStatus.OK) {
        if (st === TrackStatus.FB_ERROR) fbRejects++;
        else if (st === TrackStatus.TOO_FAR) tooFar++;
        continue;
      }
      const t = tracks[i];
      t.prevX = t.x;
      t.prevY = t.y;
      t.x = res.positions[i * 2];
      t.y = res.positions[i * 2 + 1];
      t.age++;
      t.inlier = true;
      const d = Math.hypot(t.x - t.prevX, t.y - t.prevY);
      disps[survivors.length] = d;
      if (d > maxDisp) maxDisp = d;
      residualSum += res.residual[i];
      survivors.push(t);
    }
    const m = survivors.length;
    const median = m ? medianOf(disps, m) : 0;
    this.lastMotion = {
      level,
      medianDisplacementPx: median,
      maxDisplacementPx: maxDisp,
      trackedBefore: n,
      trackedAfter: m,
      forwardBackwardRejects: fbRejects,
      tooFarRejects: tooFar,
      meanResidual: m ? residualSum / m : 0,
      predictionMode,
      searchScale,
    };
    // The level of the *next* frame is decided from what was measured now.
    this.lastMedianDisplacementPx = median;
    this.tracks = survivors;
    return m;
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
    this.lastHomographyInlierRatio = n > 0 ? r.inlierCount / n : 0;
    this.lastRansacError = r.meanError;
    this.lastImageMotion = r.homography;
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
    accepted: r.accepted,
    inlierCount: r.inlierCount,
    candidateCount: r.candidateCount,
    confirmedCount: r.confirmedCount,
    inlierRatio: r.inlierRatio,
    reprojectionErrorPx: r.meanErrorPx,
    confidence: r.confidence,
  };
}

function emptyMotion(): MotionDiagnostics {
  return {
    level: "normal",
    medianDisplacementPx: 0,
    maxDisplacementPx: 0,
    trackedBefore: 0,
    trackedAfter: 0,
    forwardBackwardRejects: 0,
    tooFarRejects: 0,
    meanResidual: 0,
    predictionMode: "none",
    searchScale: 1,
  };
}

/** Median of the first `n` values (sorts a copy). */
function medianOf(values: Float32Array, n: number): number {
  const a = Array.from(values.subarray(0, n)).sort((x, y) => x - y);
  return n % 2 ? a[n >> 1] : (a[(n >> 1) - 1] + a[n >> 1]) / 2;
}

function emptyReloc(): RelocalizationOutput {
  return {
    keyframes: 0,
    attempt: "none",
    inlierCount: 0,
    candidatesTried: 0,
    lastSuccessFrame: -1,
    successCount: 0,
    reason: null,
    rejectCode: null,
    meanReprojectionErrorPx: 0,
    matchScore: 0,
    inlierRatio: 0,
    spatialCells: 0,
    keyframeId: -1,
    jumpTranslation: 0,
    jumpRotationDeg: 0,
    postDeltaTranslation: 0,
    postDeltaRotationDeg: 0,
    postInconsistent: false,
    diagnostics: null,
    framesSinceAttempt: -1,
  };
}
