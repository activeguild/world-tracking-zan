import * as THREE from "three";
import { CameraManager, type CameraOptions } from "../camera/CameraManager";
import { FrameGrabber } from "../camera/CameraFrame";
import { approximateIntrinsics, type CameraIntrinsics } from "../camera/CameraIntrinsics";
import { ARLogger } from "../debug/Logger";
import { FeatureRenderer } from "../debug/FeatureRenderer";
import { PlaneRenderer } from "../debug/PlaneRenderer";
import { viewportIntrinsics } from "../math/CoordinateSystem";
import { ARCamera } from "../rendering/ARCamera";
import type { ARObject } from "../rendering/ARObject";
import { ARRenderer } from "../rendering/ARRenderer";
import { ARWorld } from "../rendering/ARWorld";
import { FramePresenter } from "../rendering/FramePresenter";
import type { TrackingQuality } from "../vision/TrackingQuality";
import { emptyQuality } from "../vision/TrackingQuality";
import type { MapPoseOutput, PlaneOutput, PlaneSearchOutput, PoseOutput, RelocalizationOutput } from "../vision/types";
import { WorldAnchor } from "./WorldAnchor";
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
  /**
   * Canvas that shows the camera frame the current pose was computed on
   * (time-aligned with the rendering). When given and
   * `processing.syncVideoToPose` is on, the live <video> is hidden behind it.
   */
  frameCanvas?: HTMLCanvasElement;
  /**
   * Three.js integration (Phase 4). Either give a transparent `canvas`
   * (the session owns renderer + scene), or an existing `scene` / `camera`
   * that the session drives while the application renders.
   */
  threeCanvas?: HTMLCanvasElement;
  threeScene?: THREE.Scene;
  threeCamera?: THREE.PerspectiveCamera;
}

/** Result of `ARSession.hitTest` (spec §27, §54). World frame, meters. */
export interface ARHitResult {
  position: THREE.Vector3;
  normal: THREE.Vector3;
  distance: number;
}

export interface ARSessionEvents {
  trackingStateChanged: (state: TrackingState, previous: TrackingState) => void;
  /** Fired when a stable horizontal plane is first found (spec §54). */
  planeFound: (plane: PlaneOutput) => void;
  /** Fired when the found plane is lost (map reset / plane dropped). */
  planeLost: () => void;
  /** World frame created on the first found plane; hit tests work from now on. */
  worldReady: () => void;
  /** World frame dropped (map reset); placed objects were removed. */
  worldLost: () => void;
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
  worldReady: boolean;
  /** Map units → meters (0 when no world). */
  worldScale: number;
  placedObjects: number;
  relocalization: RelocalizationOutput | null;
  planeSearch: PlaneSearchOutput | null;
  state: TrackingState;
  fastThreshold: number;
  framesProcessed: number;
  framesDropped: number;
  processingWidth: number;
  processingHeight: number;
  /** Focal length in processing pixels. */
  focalPx: number;
  /** Frame-synchronized display active. */
  syncVideo: boolean;
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

  // Phase 4: world anchor + Three.js
  private readonly worldAnchor: WorldAnchor;
  private readonly arCamera: ARCamera;
  readonly world: ARWorld;
  private readonly arRenderer: ARRenderer | null;
  private readonly externalScene: THREE.Scene | null;
  private lastMapPose: MapPoseOutput | null = null;
  private viewportSize: { width: number; height: number } | null = null;

  // Frame-synchronized display (ring of canvases, see FramePresenter).
  private readonly frameCanvas: HTMLCanvasElement | null;
  private presenter: FramePresenter | null = null;
  private syncActive = false;
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
    worldReady: new Set(),
    worldLost: new Set(),
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
  private relocalization: RelocalizationOutput | null = null;
  private planeSearch: PlaneSearchOutput | null = null;
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
    this.frameCanvas = options.frameCanvas ?? null;

    const w = this.config.world;
    this.worldAnchor = new WorldAnchor({ assumedPlaneDistanceMeters: w.assumedPlaneDistanceMeters });
    this.arCamera = new ARCamera(w.near, w.far, w.positionSmoothing, w.rotationSmoothing, options.threeCamera);
    this.world = new ARWorld(w.holdPoseOnLostMs);
    this.externalScene = options.threeScene ?? null;
    this.arRenderer = options.threeCanvas ? new ARRenderer(options.threeCanvas, options.threeScene) : null;
    const scene = this.arRenderer?.scene ?? this.externalScene;
    if (scene) this.world.attach(scene);
  }

  /** The Three.js camera driven by the session. */
  get threeCamera(): THREE.PerspectiveCamera {
    return this.arCamera.camera;
  }

  /** True once the world frame exists (plane found). */
  get isWorldReady(): boolean {
    return this.worldAnchor.isReady;
  }

  /** Current plane (null when none / not yet found). */
  get currentPlane(): PlaneOutput | null {
    return this.plane;
  }

  /** Engine state, promoted to AR_ACTIVE once an object is placed on a found plane. */
  get state(): TrackingState {
    if (this._state === TrackingState.PLANE_FOUND && this.world.placedCount > 0) return TrackingState.AR_ACTIVE;
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
    this.intrinsics = approximateIntrinsics(size.width, size.height, this.config.processing.longSideFovDeg);

    // Frame-synchronized display.
    this.syncActive = false;
    if (this.config.processing.syncVideoToPose && this.frameCanvas) {
      this.presenter ??= new FramePresenter(this.frameCanvas);
      this.syncActive = this.presenter.available;
    }
    if (this.frameCanvas) this.frameCanvas.style.display = this.syncActive ? "block" : "none";

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
    this.worldAnchor.reset();
    this.world.setWorldReady(false);
    this.presenter?.clear();
    this.video.style.opacity = "";
    this.emit("stopped");
  }

  /**
   * Hit test (spec §27–§28, §54): `x, y` in CSS pixels relative to the video
   * element. Returns the point on the detected plane, or null when there is
   * no world yet / the ray misses the plane.
   */
  hitTest(x: number, y: number): ARHitResult | null {
    if (!this.worldAnchor.isReady || !this.lastMapPose || !this.intrinsics) return null;
    const rect = this.video.getBoundingClientRect();
    const vp = viewportIntrinsics(this.intrinsics, rect.width, rect.height);
    // CSS → processing pixels (inverse of the object-fit: cover mapping).
    const u = (x - vp.offsetX) / vp.scale;
    const v = (y - vp.offsetY) / vp.scale;
    const hit = this.worldAnchor.hitTest(u, v, this.intrinsics, this.lastMapPose);
    if (!hit) return null;
    return {
      position: new THREE.Vector3(hit.position[0], hit.position[1], hit.position[2]),
      normal: new THREE.Vector3(hit.normal[0], hit.normal[1], hit.normal[2]),
      distance: hit.distance,
    };
  }

  /** Place the demo cube at a hit (creates it on first use, moves it afterwards). */
  placeCube(hit: ARHitResult, existing?: ARObject): ARObject {
    const obj = existing ?? this.world.createCube(this.config.world.cubeSize);
    obj.place([hit.position.x, hit.position.y, hit.position.z]);
    this.renderNow();
    return obj;
  }

  /**
   * Place an arbitrary Three.js object (e.g. a loaded GLB scene) at a hit.
   * With `targetSize` > 0 the model is normalized to that footprint (m) and
   * stood on the plane; otherwise it is used as is.
   */
  placeObject(object3d: THREE.Object3D, hit: ARHitResult, targetSize = 0): ARObject {
    const obj = targetSize > 0 ? this.world.addModel(object3d, targetSize) : this.world.add(object3d);
    obj.place([hit.position.x, hit.position.y, hit.position.z]);
    this.renderNow();
    return obj;
  }

  /** Move an already placed object to a new hit. */
  moveObject(obj: ARObject, hit: ARHitResult): void {
    obj.place([hit.position.x, hit.position.y, hit.position.z]);
    this.renderNow();
  }

  private renderNow(): void {
    if (!this.arRenderer) return;
    this.arRenderer.render(this.arCamera.camera);
  }

  /** Keep the Three.js projection in sync with the video viewport. */
  private updateViewport(): void {
    if (!this.intrinsics) return;
    const rect = this.video.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width));
    const h = Math.max(1, Math.round(rect.height));
    if (this.viewportSize && this.viewportSize.width === w && this.viewportSize.height === h) return;
    this.viewportSize = { width: w, height: h };
    this.arRenderer?.resize();
    this.arCamera.updateProjection(this.intrinsics, w, h);
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
      worldReady: this.worldAnchor.isReady,
      worldScale: this.worldAnchor.frame?.scale ?? 0,
      placedObjects: this.world.placedCount,
      relocalization: this.relocalization,
      planeSearch: this.planeSearch,
      state: this.state,
      fastThreshold: this.fastThreshold,
      framesProcessed: this.framesProcessed,
      framesDropped: this.framesDropped,
      processingWidth: this.grabber?.width ?? 0,
      processingHeight: this.grabber?.height ?? 0,
      focalPx: this.intrinsics?.fx ?? 0,
      syncVideo: this.syncActive,
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
    // Keep a display copy of this frame for when its pose arrives (GPU blit).
    if (this.syncActive) this.presenter!.capture(this.video, frame.frameId);
    this.backend.processFrame(frame);
  }

  /** Show the frame the result belongs to, time-aligned with the pose. */
  private presentFrame(frameId: number): void {
    if (!this.presenter?.present(frameId)) return;
    // Hide the live video once a synchronized frame is on screen.
    if (this.video.style.opacity !== "0") this.video.style.opacity = "0";
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
    this.relocalization = r.relocalization;
    this.planeSearch = r.planeSearch;
    this.fastThreshold = r.fastThreshold;
    this.lastMapPose = r.mapPose;
    this.setState(r.state);
    this.updateWorld(r);
    if (this.syncActive) this.presentFrame(r.frameId);

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
      this.planeRenderer?.draw(r.mapPose, r.plane, r.landmarks, r.landmarkCount, this.intrinsics, !this.worldAnchor.isReady);
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

  /**
   * Phase 4: create / drop the world frame, drive the Three.js camera,
   * apply the lost-tracking hold, render.
   */
  private updateWorld(r: VisionResult): void {
    // World invalid when the landmark map was reset.
    if (this.worldAnchor.checkMap(r.mapPose)) {
      this.world.setWorldReady(false);
      this.arCamera.resetSmoothing();
      this.logger.info("world lost (map reset)");
      this.emit("worldLost");
    }
    // Create the world on the first found plane.
    if (!this.worldAnchor.isReady && r.plane?.found && r.mapPose) {
      if (this.worldAnchor.create(r.plane, r.mapPose)) {
        const scale = this.worldAnchor.frame!.scale;
        const extent = Math.max(0.4, Math.min(3, Math.sqrt(Math.max(r.plane.areaEstimate, 1e-6)) * scale * 1.5));
        this.world.setWorldReady(true);
        this.world.showPlaneGrid(extent, this.config.debug.overlay && this.config.world.showPlaneGrid);
        this.arCamera.resetSmoothing();
        this.logger.info(`world ready scale=${scale.toFixed(3)} m/unit grid=${extent.toFixed(2)} m`);
        this.emit("worldReady");
      }
    }
    if (!this.worldAnchor.isReady || !r.mapPose) return;

    this.updateViewport();
    const pose = this.worldAnchor.cameraPose(r.mapPose);
    const tracking =
      r.mapPose.framesSinceTracked === 0 &&
      this._state !== TrackingState.TRACKING_LOST &&
      this._state !== TrackingState.RELOCALIZING &&
      this._state !== TrackingState.SEARCHING_FEATURES;
    if (pose && tracking) this.arCamera.setPose(pose, r.timestamp / 1000);
    this.world.updateTracking(tracking, performance.now());
    this.renderNow();
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
