import { CameraManager, type CameraOptions } from "../camera/CameraManager";
import { FrameGrabber } from "../camera/CameraFrame";
import { approximateIntrinsics, type CameraIntrinsics } from "../camera/CameraIntrinsics";
import { ARLogger } from "../debug/Logger";
import { FeatureRenderer } from "../debug/FeatureRenderer";
import { PlaneRenderer } from "../debug/PlaneRenderer";
import type { TrackingQuality } from "../vision/TrackingQuality";
import { emptyQuality } from "../vision/TrackingQuality";
import type { MapPoseOutput, PlaneOutput, PoseOutput } from "../vision/types";
import {
  MainThreadVisionBackend,
  VisionWorkerClient,
  type VisionBackend,
  type VisionResult,
} from "../worker/VisionWorkerClient";
import { resolveConfig, type ARConfig, type PartialARConfig } from "./ARConfig";
import { ARError, ARErrorCode, TrackingState } from "./ARState";

/**
 * Public entry point (spec §54). Phase 1 surface:
 *
 *   const ar = new ARSession({ video, overlayCanvas });
 *   await ar.start();
 *   ar.on("trackingStateChanged", state => ...);
 *   ar.on("frame", result => ...);
 *
 * `hitTest`, `planeFound` and Three.js integration arrive in later phases.
 * Internals (FAST/LK/RANSAC) are never exposed here.
 */
export interface ARSessionOptions {
  video: HTMLVideoElement;
  /** Optional canvas laid over the video for feature debug drawing. */
  overlayCanvas?: HTMLCanvasElement;
  config?: PartialARConfig;
  camera?: CameraOptions;
  /**
   * Optional gravity source: returns the gravity direction in the camera
   * frame (any scale) or null. Used for the plane horizontality test.
   */
  gravitySource?: () => ArrayLike<number> | null;
}

export interface ARSessionEvents {
  trackingStateChanged: (state: TrackingState, previous: TrackingState) => void;
  /** Fired when a stable horizontal plane is first found (spec §54). */
  planeFound: (plane: PlaneOutput) => void;
  /** Fired when the found plane is lost (map reset / plane dropped). */
  planeLost: () => void;
  frame: (result: VisionResult) => void;
  error: (error: ARError) => void;
  started: () => void;
  stopped: () => void;
}

export interface ARStats {
  renderFps: number;
  visionFps: number;
  visionMs: number;
  quality: TrackingQuality;
  pose: PoseOutput | null;
  mapPose: MapPoseOutput | null;
  plane: PlaneOutput | null;
  landmarkCount: number;
  gravityAvailable: boolean;
  state: TrackingState;
  fastThreshold: number;
  framesProcessed: number;
  framesDropped: number;
  processingWidth: number;
  processingHeight: number;
  backend: "worker" | "main";
}

type Listener<K extends keyof ARSessionEvents> = ARSessionEvents[K];

export class ARSession {
  readonly config: ARConfig;
  readonly camera: CameraManager;

  private readonly video: HTMLVideoElement;
  private readonly renderer: FeatureRenderer | null;
  private readonly planeRenderer: PlaneRenderer | null;
  private readonly gravitySource: (() => ArrayLike<number> | null) | null;
  private readonly logger: ARLogger;
  private backend: VisionBackend | null = null;
  private backendKind: "worker" | "main" = "worker";
  private grabber: FrameGrabber | null = null;
  private intrinsics: CameraIntrinsics | null = null;

  private running = false;
  private rafHandle = 0;
  private vfcHandle = 0;
  private lastSentTime = 0;

  private _state: TrackingState = TrackingState.INITIALIZING;
  private readonly listeners: { [K in keyof ARSessionEvents]: Set<Listener<K>> } = {
    trackingStateChanged: new Set(),
    planeFound: new Set(),
    planeLost: new Set(),
    frame: new Set(),
    error: new Set(),
    started: new Set(),
    stopped: new Set(),
  };

  // Stats
  private quality: TrackingQuality = emptyQuality();
  private pose: PoseOutput | null = null;
  private mapPose: MapPoseOutput | null = null;
  private plane: PlaneOutput | null = null;
  private landmarkCount = 0;
  private planeWasFound = false;
  private lastGravity: number[] | null = null;
  private visionMs = 0;
  private fastThreshold = 0;
  private framesProcessed = 0;
  private framesDropped = 0;
  private renderFpsCounter = new FpsCounter();
  private visionFpsCounter = new FpsCounter();

  constructor(options: ARSessionOptions) {
    this.config = resolveConfig(options.config);
    this.video = options.video;
    this.camera = new CameraManager(options.video, options.camera);
    this.renderer = options.overlayCanvas ? new FeatureRenderer(options.overlayCanvas) : null;
    this.planeRenderer = options.overlayCanvas ? new PlaneRenderer(options.overlayCanvas) : null;
    this.gravitySource = options.gravitySource ?? null;
    this.logger = new ARLogger(this.config.debug.log, this.config.debug.logIntervalMs);
  }

  /** Current plane (null when none / not yet found). */
  get currentPlane(): PlaneOutput | null {
    return this.plane;
  }

  get state(): TrackingState {
    return this._state;
  }

  get isRunning(): boolean {
    return this.running;
  }

  on<K extends keyof ARSessionEvents>(event: K, listener: Listener<K>): () => void {
    this.listeners[event].add(listener);
    return () => this.listeners[event].delete(listener);
  }

  off<K extends keyof ARSessionEvents>(event: K, listener: Listener<K>): void {
    this.listeners[event].delete(listener);
  }

  private emit<K extends keyof ARSessionEvents>(event: K, ...args: Parameters<ARSessionEvents[K]>): void {
    for (const l of this.listeners[event]) {
      (l as (...a: Parameters<ARSessionEvents[K]>) => void)(...args);
    }
  }

  async start(): Promise<void> {
    if (this.running) return;
    try {
      await this.camera.start();
    } catch (e) {
      const err = e instanceof ARError ? e : new ARError(ARErrorCode.CAMERA_UNAVAILABLE, String(e), { cause: e });
      this.emit("error", err);
      throw err;
    }

    const size = FrameGrabber.fitProcessingSize(
      this.camera.width,
      this.camera.height,
      this.config.processing.width,
      this.config.processing.height,
    );
    this.grabber = new FrameGrabber(size.width, size.height);
    this.intrinsics = approximateIntrinsics(size.width, size.height);

    this.backend = await this.createBackend(size.width, size.height);
    this.backend.onResult = (r) => this.handleResult(r);
    this.backend.onError = (message, gray) => {
      if (gray) this.grabber?.release(gray);
      this.logger.warn(`vision backend error: ${message}`);
    };

    this.running = true;
    this.setState(TrackingState.INITIALIZING);
    this.logger.info(
      `started camera=${this.camera.width}x${this.camera.height} processing=${size.width}x${size.height} backend=${this.backendKind}`,
    );
    this.renderer?.resize();
    this.scheduleFrame();
    this.emit("started");
  }

  stop(): void {
    if (!this.running) return;
    this.running = false;
    if (this.rafHandle) cancelAnimationFrame(this.rafHandle);
    if (this.vfcHandle && "cancelVideoFrameCallback" in this.video) {
      (this.video as HTMLVideoElement & { cancelVideoFrameCallback(h: number): void }).cancelVideoFrameCallback(
        this.vfcHandle,
      );
    }
    this.rafHandle = 0;
    this.vfcHandle = 0;
    this.backend?.dispose();
    this.backend = null;
    this.camera.stop();
    this.renderer?.clear();
    this.emit("stopped");
  }

  /** Forget all tracks and restart from INITIALIZING (keeps the camera running). */
  reset(): void {
    this.backend?.reset();
    this.setState(TrackingState.INITIALIZING);
  }

  getStats(): ARStats {
    return {
      renderFps: this.renderFpsCounter.fps,
      visionFps: this.visionFpsCounter.fps,
      visionMs: this.visionMs,
      quality: this.quality,
      pose: this.pose,
      mapPose: this.mapPose,
      plane: this.plane,
      landmarkCount: this.landmarkCount,
      gravityAvailable: this.lastGravity !== null,
      state: this._state,
      fastThreshold: this.fastThreshold,
      framesProcessed: this.framesProcessed,
      framesDropped: this.framesDropped,
      processingWidth: this.grabber?.width ?? 0,
      processingHeight: this.grabber?.height ?? 0,
      backend: this.backendKind,
    };
  }

  private async createBackend(width: number, height: number): Promise<VisionBackend> {
    if (this.config.useWorker && typeof Worker !== "undefined") {
      const client = new VisionWorkerClient();
      try {
        await client.init(width, height, this.config);
        this.backendKind = "worker";
        return client;
      } catch (e) {
        this.logger.warn(`worker unavailable, falling back to main thread: ${String(e)}`);
        client.dispose();
      }
    }
    const main = new MainThreadVisionBackend();
    await main.init(width, height, this.config);
    this.backendKind = "main";
    return main;
  }

  private scheduleFrame(): void {
    if (!this.running) return;
    const video = this.video as HTMLVideoElement & {
      requestVideoFrameCallback?: (cb: (now: number) => void) => number;
    };
    if (typeof video.requestVideoFrameCallback === "function") {
      // Fires once per new camera frame (Safari 15.4+, Chrome 83+).
      this.vfcHandle = video.requestVideoFrameCallback((now) => {
        this.onVideoFrame(now);
        this.scheduleFrame();
      });
    } else {
      this.rafHandle = requestAnimationFrame((now) => {
        this.onVideoFrame(now);
        this.scheduleFrame();
      });
    }
  }

  private onVideoFrame(now: number): void {
    if (!this.running || !this.backend || !this.grabber || !this.intrinsics) return;
    this.renderFpsCounter.tick(now);
    if (this.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;

    // Backpressure: one frame in flight at a time.
    if (this.backend.busy) {
      this.framesDropped++;
      return;
    }
    const maxFps = this.config.processing.maxVisionFps;
    if (maxFps > 0 && now - this.lastSentTime < 1000 / maxFps - 1) return;
    this.lastSentTime = now;

    let gravity: number[] | null = null;
    if (this.gravitySource) {
      const g = this.gravitySource();
      if (g && g.length === 3) gravity = [g[0], g[1], g[2]];
    }
    this.lastGravity = gravity;
    const frame = this.grabber.grab(this.video, now, this.intrinsics, gravity);
    this.backend.processFrame(frame);
  }

  private handleResult(r: VisionResult): void {
    if (!this.running) return;
    this.grabber?.release(r.grayBuffer);
    this.framesProcessed++;
    this.visionFpsCounter.tick(performance.now());
    this.visionMs = r.processingMs;
    this.quality = r.quality;
    this.pose = r.pose;
    this.mapPose = r.mapPose;
    this.plane = r.plane;
    this.landmarkCount = r.landmarkCount;
    this.fastThreshold = r.fastThreshold;
    this.setState(r.state);

    const found = !!r.plane?.found;
    if (found && !this.planeWasFound) {
      this.planeWasFound = true;
      this.logger.info("plane found");
      this.emit("planeFound", r.plane!);
    } else if (!found && this.planeWasFound) {
      this.planeWasFound = false;
      this.logger.info("plane lost");
      this.emit("planeLost");
    }

    if (this.renderer && this.config.debug.overlay && this.grabber && this.intrinsics) {
      this.renderer.resize();
      this.renderer.draw(r.tracks, r.trackCount, this.grabber.width, this.grabber.height);
      this.planeRenderer?.draw(r.mapPose, r.plane, r.landmarks, r.landmarkCount, this.intrinsics);
    }
    this.logger.periodic(performance.now(), {
      state: r.state,
      features: r.quality.featureCount,
      tracked: r.quality.trackedCount,
      inliers: r.quality.inlierCount,
      planeConfidence: r.quality.planeConfidence,
      visionFPS: this.visionFpsCounter.fps,
      visionMs: r.processingMs,
    });
    this.emit("frame", r);
  }

  private setState(next: TrackingState): void {
    if (next === this._state) return;
    const prev = this._state;
    this._state = next;
    this.logger.info(`state ${prev} → ${next}`);
    this.emit("trackingStateChanged", next, prev);
  }
}

/** Rolling frames-per-second counter over a one second window. */
class FpsCounter {
  private times: number[] = [];
  fps = 0;

  tick(now: number): void {
    this.times.push(now);
    const cutoff = now - 1000;
    while (this.times.length && this.times[0] < cutoff) this.times.shift();
    this.fps = this.times.length;
  }
}
