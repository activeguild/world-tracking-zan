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
import type {
  MapPoseOutput,
  PlaneOutput,
  PlanePoseOutput,
  PlaneSearchOutput,
  PoseCandidateReport,
  PoseOutput,
  RelocalizationOutput,
  MotionDiagnostics,
} from "../vision/types";
import type { PoseRejectCode } from "../vision/PoseValidation";
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
  /** Frame-to-frame motion level and LK diagnostics (v7). */
  motion: MotionDiagnostics | null;
  planeSearch: PlaneSearchOutput | null;
  state: TrackingState;
  fastThreshold: number;
  framesProcessed: number;
  framesDropped: number;
  processingWidth: number;
  processingHeight: number;
  /** Plane-relative pose quality (null until a plane is anchored). */
  planePose: PlanePoseOutput | null;
  /** The world plane has been fixed. */
  planeAnchored: boolean;
  /** Capture time (performance.now ms) of the frame the current pose belongs to. */
  frameTimestampMs: number;
  /** Time the pose for that frame arrived from the vision backend. */
  poseTimestampMs: number;
  /** Render time − capture time of the frame the current pose belongs to (ms). */
  poseAgeMs: number;
  /** poseAgeMs above `debug.poseStaleMs`. */
  poseStale: boolean;
  /** Three.js camera position in world meters (null before the world exists). */
  cameraWorldPosition: number[] | null;
  /** How long tracking has been lost (ms), 0 while tracking; objects hold their pose meanwhile. */
  lostMs: number;
  /** Placed objects' world poses — must not change when only the camera moves. */
  objects: { id: number; position: number[]; yaw: number }[];
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
  private motion: MotionDiagnostics | null = null;
  /** Debug visualization (feature overlay, plane grid) on/off; the engine runs either way (v7 §17–§18). */
  private debugVisualization = true;
  private planeSearch: PlaneSearchOutput | null = null;
  private planeWasFound = false;
  private lastGravity: number[] | null = null;
  private visionMs = 0;
  private poseAgeMs = 0;
  private frameTimestampMs = 0;
  private poseTimestampMs = 0;
  private translationHeldLogged = false;
  private loggedSource: MapPoseOutput["source"] | null = null;
  private loggedReject: { map: PoseRejectCode | null; plane: PoseRejectCode | null } = { map: null, plane: null };
  private loggedRelocReject: string | null = null;
  private loggedRelocInconsistent = false;
  private lastWorldUpdateMs = 0;
  private planePose: PlanePoseOutput | null = null;
  private planeAnchored = false;
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
    this.arCamera.setSmoothing(w.smoothing);
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

  /**
   * Show / hide the debug visualization (feature overlay, landmark / plane
   * drawing, plane grid). UI state only: tracking and diagnostics continue
   * unchanged (v7 §17–§18).
   */
  setDebugVisualization(visible: boolean): void {
    this.debugVisualization = visible;
    if (!visible) this.renderer?.clear();
    this.world.setPlaneGridVisible(visible && this.config.debug.overlay && this.config.world.showPlaneGrid);
  }

  get debugVisualizationVisible(): boolean {
    return this.debugVisualization;
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
      motion: this.motion,
      planeSearch: this.planeSearch,
      planePose: this.planePose,
      planeAnchored: this.planeAnchored,
      frameTimestampMs: this.frameTimestampMs,
      poseTimestampMs: this.poseTimestampMs,
      poseAgeMs: this.poseAgeMs,
      poseStale: this.poseAgeMs > this.config.debug.poseStaleMs,
      cameraWorldPosition: this.worldAnchor.isReady ? this.arCamera.camera.position.toArray() : null,
      lostMs: this.world.lostDurationMs(performance.now()),
      objects: this.world.objects
        .filter((o) => o.placed)
        .map((o) => ({ id: o.id, position: o.root.position.toArray(), yaw: o.root.rotation.y })),
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
    const arrived = performance.now();
    this.visionFpsCounter.tick(arrived);
    this.visionMs = r.processingMs;
    this.frameTimestampMs = r.timestamp;
    this.poseTimestampMs = arrived;
    // v2 §6: make the "PnP failed, translation held" episodes visible.
    if (r.mapPose?.translationHeld) {
      if (!this.translationHeldLogged) {
        this.translationHeldLogged = true;
        this.logger.info(`PnP failed at frame ${r.frameId}: translation held, rotation propagated (lost ${r.mapPose.framesSinceTracked})`);
      }
    } else if (this.translationHeldLogged) {
      this.translationHeldLogged = false;
      this.logger.info(`PnP recovered at frame ${r.frameId} (${r.mapPose?.source ?? "none"})`);
    }
    this.logPoseSelection(r);
    this.quality = r.quality;
    this.pose = r.pose;
    this.mapPose = r.mapPose;
    this.plane = r.plane;
    this.landmarkCount = r.landmarkCount;
    this.relocalization = r.relocalization;
    this.motion = r.motion;
    this.planeSearch = r.planeSearch;
    this.planePose = r.planePose;
    this.planeAnchored = r.planeAnchor !== null;
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

    if (this.renderer && this.config.debug.overlay && this.debugVisualization && this.grabber && this.intrinsics) {
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
      ...(r.mapPose ? { poseSource: r.mapPose.source } : {}),
      ...(r.planePose ? { planeInliers: r.planePose.inlierCount, planeErrorPx: r.planePose.reprojectionErrorPx } : {}),
      ...(this.worldAnchor.isReady ? { poseAgeMs: this.poseAgeMs } : {}),
    });
    this.emit("frame", r);
  }

  /**
   * v4 §26–§27: source changes and candidate rejections, with the measured
   * delta and the limit in force (world meters once the world exists, map
   * units before), once per episode rather than every frame.
   */
  private logPoseSelection(r: VisionResult): void {
    const m = r.mapPose;
    if (!m || !this.logger.enabled) return;
    const scale = this.worldAnchor.frame?.scale ?? 0;
    const len = (mapUnits: number) => (scale > 0 ? `${(mapUnits * scale).toFixed(3)}m` : `${mapUnits.toFixed(3)}u`);
    // v5 §8, §18: every relocalization candidate, what was applied, what was
    // rejected (once per reject code), and how the map PnP agreed afterwards.
    const rl = r.relocalization;
    if (rl.attempt === "candidate" || rl.attempt === "success") {
      this.logger.info(
        `RELOCALIZATION ${rl.attempt === "success" ? "APPLIED" : "CANDIDATE"} at frame ${r.frameId} (kf ${rl.keyframeId})\n` +
          `translation = ${len(rl.jumpTranslation)}\nrotation = ${rl.jumpRotationDeg.toFixed(1)}deg\n` +
          `inliers = ${rl.inlierCount}\nerror = ${rl.meanReprojectionErrorPx.toFixed(2)}px\nmatch = ${rl.matchScore.toFixed(2)}\n` +
          `inlier ratio = ${rl.inlierRatio.toFixed(2)}\ncells = ${rl.spatialCells}/9`,
      );
      this.loggedRelocReject = null;
    } else if (rl.attempt === "fail") {
      // v6 §1–§5: where the keyframes dropped out (NCC → LK → PnP → VAL) and
      // the best of them, so a failing relocalization can be diagnosed from
      // the console without the HUD.
      const d = rl.diagnostics;
      const key = `${rl.rejectCode}:${d?.best?.stage ?? ""}`;
      if (rl.rejectCode && key !== this.loggedRelocReject) {
        const stages = d
          ? `\nkeyframes = ${d.keyframes}  tried = ${d.candidatesTried}\nNCC ${d.coarsePassed}/${d.coarseTested} (best ${d.bestCoarseScore.toFixed(2)})  LK ${d.lkPassed}/${d.lkTested}  PnP ${d.pnpPassed}/${d.pnpTested}  VAL ${d.validated}` +
            (d.best
              ? `\nbest = KF${d.best.keyframeId} stage ${d.best.stage}  ${d.best.inlierCount}i  ${d.best.meanReprojectionErrorPx.toFixed(2)}px  ratio ${d.best.inlierRatio.toFixed(2)}  cells ${d.best.spatialCells}/9  ncc ${d.best.coarseScore.toFixed(2)}`
              : "") +
            `\ntrials = ${d.trials.map((t) => `KF${t.keyframeId}:${t.stage}${t.inlierCount ? `(${t.inlierCount}i)` : ""}`).join(" ")}`
          : "";
        this.logger.info(`RELOC REJECT at frame ${r.frameId}\nreason = ${rl.rejectCode}${rl.reason ? `\ndetail = ${rl.reason}` : ""}${stages}`);
      }
      this.loggedRelocReject = key;
    }
    if (rl.postInconsistent && !this.loggedRelocInconsistent) {
      this.logger.warn(
        `RELOC INCONSISTENT at frame ${r.frameId}: map PnP after relocalization Δ ${len(rl.postDeltaTranslation)} / ${rl.postDeltaRotationDeg.toFixed(1)}deg`,
      );
    }
    this.loggedRelocInconsistent = rl.postInconsistent;
    if (m.framesSinceTracked === 0 && m.source !== "propagated") {
      if (this.loggedSource !== null && this.loggedSource !== m.source) {
        this.logger.info(
          `SOURCE ${this.loggedSource.toUpperCase()} -> ${m.source.toUpperCase()} at frame ${r.frameId}\n` +
            `map/plane delta = ${len(m.sourceDeltaTranslation)}\nrotation delta = ${m.sourceDeltaRotationDeg.toFixed(1)}deg`,
        );
      }
      this.loggedSource = m.source;
    }
    const logReject = (label: "MAP" | "PLANE", c: PoseCandidateReport | null, key: "map" | "plane") => {
      const rej = c?.reject ?? null;
      const code = rej?.code ?? null;
      // Cooldown and "nothing to solve" are not rejections worth a line.
      const notable = code !== null && code !== "source_cooldown" && code !== "insufficient_observations";
      if (notable && this.loggedReject[key] !== code) {
        const isLength = code === "translation_jump" || (code === "map_plane_disagreement" && rej!.reason.includes("translation"));
        const isAngle = code === "rotation_jump" || (code === "map_plane_disagreement" && rej!.reason.includes("rotation"));
        const fmt = (v: number) => (isLength ? len(v) : isAngle ? `${v.toFixed(1)}deg` : v.toFixed(2));
        this.logger.info(
          `${label} REJECT at frame ${r.frameId}\nreason = ${code}\ndelta = ${fmt(rej!.delta)}\nlimit = ${fmt(rej!.limit)}` +
            (c ? `\ninliers = ${c.inlierCount}  error = ${c.reprojectionErrorPx.toFixed(2)}px${c.trusted ? "  (trusted)" : ""}` : ""),
        );
      }
      this.loggedReject[key] = notable ? code : null;
    };
    logReject("MAP", m.mapCandidate, "map");
    logReject("PLANE", m.planeCandidate, "plane");
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
    // Create the world on the anchored plane (the engine fixes it on the
    // first PLANE_FOUND; the same plane is the reference of the plane-relative
    // pose, so world and pose share one definition). Without plane tracking
    // the first found plane is used directly.
    const anchorPlane = r.planeAnchor ?? (r.plane?.found ? r.plane : null);
    if (!this.worldAnchor.isReady && anchorPlane && r.mapPose) {
      if (this.worldAnchor.create(anchorPlane, r.mapPose)) {
        const scale = this.worldAnchor.frame!.scale;
        const area = r.plane?.areaEstimate ?? 0;
        const extent = Math.max(0.4, Math.min(3, Math.sqrt(Math.max(area, 1e-6)) * scale * 1.5));
        this.world.setWorldReady(true);
        // Build the grid whenever debug drawing is configured; whether it is
        // shown right now follows the debug HUD toggle.
        this.world.showPlaneGrid(extent, this.config.debug.overlay && this.config.world.showPlaneGrid);
        this.world.setPlaneGridVisible(this.debugVisualization);
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
    // Camera pose → Three.js camera. Objects are not touched here (修正指示書 §11, §16).
    if (pose && tracking) this.arCamera.setPose(pose, r.timestamp / 1000);
    const nowMs = performance.now();
    this.world.updateTracking(tracking, nowMs);
    // Objects' own animation, world space, independent of the camera (§14, §29).
    if (this.lastWorldUpdateMs > 0) this.world.update(Math.min(0.1, (nowMs - this.lastWorldUpdateMs) / 1000));
    this.lastWorldUpdateMs = nowMs;
    // Pose age (§17–§18): render time − capture time of the frame the pose belongs to.
    this.poseAgeMs = nowMs - r.timestamp;
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
