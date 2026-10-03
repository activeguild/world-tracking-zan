import type { ARConfig } from "../ar/ARConfig";
import { TrackingState, TrackingStateMachine } from "../ar/ARState";
import { FeatureDetector } from "./FeatureDetector";
import { FeatureTracker, TrackStatus, allocResult, type TrackResult } from "./FeatureTracker";
import { ImagePyramid } from "./ImagePyramid";
import { ransacHomography, type Rng } from "./OutlierRejection";
import { PoseEstimator, type RelativePose } from "./PoseEstimator";
import { computeTrackingConfidence, emptyQuality, type TrackingQuality } from "./TrackingQuality";
import { packTracks, type PoseOutput, type Track, type VisionInput, type VisionOutput } from "./types";
import { type Mat3, mat3Identity, mat3Multiply } from "../math/Matrix";
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

  /** Timing breakdown of the last frame (ms). */
  readonly timing = { pyramid: 0, track: 0, ransac: 0, detect: 0, pose: 0, total: 0 };

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

    // 6. Quality + state
    const featureCount = this.tracks.length;
    const quality: TrackingQuality = {
      featureCount,
      trackedCount,
      inlierCount,
      reprojectionError: this.lastRansacError,
      poseDelta: this.lastPoseDelta,
      planeConfidence: 0,
      trackingConfidence: computeTrackingConfidence(
        inlierCount,
        previousCount,
        this.config.state.minTrackedForTracking,
      ),
      lowFeature: featureCount < cfg.minFeatures,
    };
    this.lastQuality = quality;
    const state = this.stateMachine.update({ inlierCount, featureCount });

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
    this.timing.total = t5 - t0;

    return {
      frameId: input.frameId,
      timestamp: input.timestamp,
      state,
      quality,
      pose,
      tracks: packTracks(this.tracks),
      trackCount: this.tracks.length,
      processingMs: t5 - t0,
    };
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
    const res = this.tracker.track(this.prevPyramid, this.curPyramid, pts, n, this.trackResult);

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
    const kept: Track[] = [];
    for (let i = 0; i < n; i++) {
      if (r.inliers[i]) kept.push(tracks[i]);
    }
    this.tracks = kept;
    return kept.length;
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
        refX: c.x,
        refY: c.y,
        refFrame: -1,
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
