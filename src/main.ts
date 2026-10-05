import { ARSession } from "./ar/ARSession";
import { ARError, TrackingState } from "./ar/ARState";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import type * as THREE from "three";
import { DebugOverlay, type HudCandidate } from "./debug/DebugOverlay";
import { rotationToEulerDeg } from "./math/Pose";
import type { PoseCandidateReport } from "./vision/types";
import type { ARObject } from "./rendering/ARObject";
import { GravityProvider, parseGravityOverride } from "./sensors/GravityProvider";
import "./style.css";

/**
 * Demo: camera → feature tracking → relative pose → landmark map → plane
 *       → tap to place a cube or a GLB model.
 *
 * Query parameters:
 *   ?debug=1          enable `[AR]` console logs
 *   ?worker=0         run the vision engine on the main thread
 *   ?hud=0            hide the HUD
 *   ?gravity=x,y,z    override the gravity direction (camera frame), for tests
 *   ?model=URL        place this GLB instead of the cube (same-origin or CORS-enabled)
 *   ?size=0.15        footprint of the model in meters (default 0.15)
 *   ?fov=66           camera field of view along the long side (degrees)
 *   ?sync=0           show the live video instead of the pose-synchronized frame
 *   ?smooth=1         enable pose smoothing (off by default while the raw pose is validated)
 *   ?dist=0.5         assumed camera→plane distance in meters (scale)
 *   ?walk=1           the placed object walks back and forth on the plane (object motion test)
 *   ?refine=1         re-enable landmark depth refinement (A/B against the fixed map, v2 §26)
 *   ?planetrack=1     experimental plane-relative pose instead of landmark PnP
 */
const params = new URLSearchParams(location.search);
const debugLog = params.get("debug") === "1";
const useWorker = params.get("worker") !== "0";
const showHud = params.get("hud") !== "0";
const gravityOverride = parseGravityOverride(params.get("gravity"));
const gravityProvider = new GravityProvider();
const modelUrl = params.get("model");
const modelSize = Number(params.get("size") ?? "0.15") || 0.15;
const fovDeg = Number(params.get("fov") ?? "") || undefined;
const syncVideo = params.get("sync") !== "0";
// Pose smoothing is off by default while the raw pose is validated (v3 §22).
const smoothingParam = params.get("smooth");
const assumedDist = Number(params.get("dist") ?? "") || undefined;
const walk = params.get("walk") === "1";
const planeTracking = params.get("planetrack") === "1";
const refineLandmarks = params.get("refine") === "1";

// Start loading the GLB early; placement waits for it.
let modelPromise: Promise<THREE.Object3D> | null = null;
if (modelUrl) {
  const loader = new GLTFLoader();
  modelPromise = loader.loadAsync(modelUrl).then((gltf) => gltf.scene);
  modelPromise.catch((e) => console.warn("[AR] GLB load failed, falling back to the cube:", e));
}

const app = document.getElementById("app") as HTMLElement;
const video = document.getElementById("video") as HTMLVideoElement;
const frameCanvas = document.getElementById("frame") as HTMLCanvasElement;
const threeCanvas = document.getElementById("three") as HTMLCanvasElement;
const overlay = document.getElementById("overlay") as HTMLCanvasElement;
const startButton = document.getElementById("start") as HTMLButtonElement;
const messageEl = document.getElementById("message") as HTMLElement;

const hud = new DebugOverlay(app);
hud.visible = showHud;
console.log(`[AR] build ${typeof __BUILD_LABEL__ === "string" ? __BUILD_LABEL__ : "dev"}`);

const session = new ARSession({
  video,
  overlayCanvas: overlay,
  frameCanvas,
  threeCanvas,
  config: {
    debug: { log: debugLog, overlay: true },
    useWorker,
    processing: {
      ...(fovDeg ? { longSideFovDeg: fovDeg } : {}),
      syncVideoToPose: syncVideo,
    },
    planeTracking: { enabled: planeTracking },
    landmarks: { enableLandmarkDepthRefinement: refineLandmarks },
    world: {
      ...(assumedDist ? { assumedPlaneDistanceMeters: assumedDist } : {}),
      ...(smoothingParam !== null ? { smoothing: smoothingParam === "1" } : {}),
    },
  },
  gravitySource: () => gravityOverride ?? gravityProvider.gravityCamera,
});

session.on("trackingStateChanged", (state) => {
  messageEl.textContent = userMessageFor(state, session.getStats().quality.lowFeature);
});

session.on("planeFound", () => {
  messageEl.textContent = userMessageFor(session.state, false);
});

// Phase 4: tap → hit test → place / move the object (spec §27, §68).
// The first tap places the cube (or the GLB when ?model= is given), later
// taps move it. Objects are fixed in world space; only the camera moves.
let placed: ARObject | undefined;
session.on("worldReady", () => {
  messageEl.textContent = "平面をタップしてオブジェクトを置いてください";
});
session.on("worldLost", () => {
  placed = undefined;
  messageEl.textContent = "トラッキングを失いました。平面を探し直しています…";
});
app.addEventListener("pointerup", async (ev) => {
  if ((ev.target as HTMLElement).tagName === "BUTTON") return;
  if (!session.isRunning) return;
  const rect = video.getBoundingClientRect();
  const hit = session.hitTest(ev.clientX - rect.left, ev.clientY - rect.top);
  if (!hit) return;
  if (placed) {
    session.moveObject(placed, hit);
  } else if (modelPromise) {
    try {
      const model = await modelPromise;
      placed = session.placeObject(model, hit, modelSize);
    } catch {
      placed = session.placeCube(hit);
    }
  } else {
    placed = session.placeCube(hit);
  }
  if (walk) startWalking(placed);
  messageEl.textContent = "";
});

// Object motion test (修正指示書 Test D / E): the object walks along world X
// at 5 cm/s and turns around every 4 s. Its position is integrated in world
// space by ARWorld.update(); the camera pose plays no part in it.
let walkTimer = 0;
function startWalking(obj: ARObject): void {
  if (walkTimer) return;
  obj.velocity.set(0.05, 0, 0);
  obj.setYaw(Math.PI / 2);
  walkTimer = window.setInterval(() => {
    obj.velocity.x = -obj.velocity.x;
    obj.setYaw(obj.velocity.x > 0 ? Math.PI / 2 : -Math.PI / 2);
  }, 4000);
}

session.on("frame", (r) => {
  if (r.quality.lowFeature && session.state !== TrackingState.TRACKING) {
    messageEl.textContent = userMessageFor(session.state, true);
  }
});

session.on("error", (err) => {
  showError(err);
});

startButton.addEventListener("click", async () => {
  startButton.disabled = true;
  startButton.textContent = "Starting…";
  try {
    // Gravity needs a user gesture on iOS; start it from the click handler.
    if (!gravityOverride) await gravityProvider.start();
    await session.start();
    startButton.style.display = "none";
  } catch (e) {
    startButton.disabled = false;
    startButton.textContent = "Retry";
    showError(e);
  }
});

function showError(e: unknown): void {
  const code = e instanceof ARError ? e.code : "ERROR";
  const msg = e instanceof Error ? e.message : String(e);
  messageEl.textContent = `${code}: ${msg}`;
}

function userMessageFor(state: TrackingState, lowFeature: boolean): string {
  switch (state) {
    case TrackingState.INITIALIZING:
      return "Initializing…";
    case TrackingState.SEARCHING_FEATURES:
      return lowFeature
        ? "周囲をゆっくり動かしてください（特徴点が足りません）"
        : "Searching features…";
    case TrackingState.TRACKING:
      return "ゆっくり横に動かして平面を探しています…";
    case TrackingState.PLANE_DETECTING:
      return "平面を検出中…（床・机をゆっくり見回してください）";
    case TrackingState.PLANE_FOUND:
      return "平面をタップして Cube を置いてください";
    case TrackingState.AR_ACTIVE:
      return "";
    case TrackingState.RELOCALIZING:
      return "位置を探しています… さっき見ていた場所にカメラを戻してください";
    case TrackingState.TRACKING_LOST:
      return "Tracking lost — ゆっくり元の位置に戻してください";
    default:
      return state;
  }
}

/** Candidate diagnostics in display units (v4 §25); `scale` NaN before the world exists. */
function hudCandidate(c: PoseCandidateReport | null, scale: number): HudCandidate | null {
  if (!c) return null;
  const rej = c.reject;
  let detail: string | null = null;
  if (rej) {
    const isLength = rej.code === "translation_jump" || (rej.code === "map_plane_disagreement" && rej.reason.includes("translation"));
    const isAngle = rej.code === "rotation_jump" || (rej.code === "map_plane_disagreement" && rej.reason.includes("rotation"));
    if (isLength) detail = Number.isFinite(scale) ? `${(rej.delta * scale * 100).toFixed(1)} > ${(rej.limit * scale * 100).toFixed(1)} cm` : `${rej.delta.toFixed(3)} > ${rej.limit.toFixed(3)} u`;
    else if (isAngle) detail = `${rej.delta.toFixed(1)} > ${rej.limit.toFixed(1)}°`;
    else if (Number.isFinite(rej.delta)) detail = `${rej.delta} < ${rej.limit}`;
  }
  return {
    inliers: c.inlierCount,
    errorPx: c.reprojectionErrorPx,
    deltaM: Number.isFinite(scale) ? c.deltaTranslation * scale : NaN,
    deltaDeg: c.deltaRotationDeg,
    trusted: c.trusted,
    rejectCode: rej?.code ?? null,
    rejectDetail: detail,
  };
}

// HUD refresh loop (independent of vision rate).
function refreshHud(): void {
  const s = session.getStats();
  hud.update({
    renderFps: s.renderFps,
    visionFps: s.visionFps,
    featureCount: s.quality.featureCount,
    trackedCount: s.quality.trackedCount,
    inlierCount: s.quality.inlierCount,
    planeConfidence: s.quality.planeConfidence,
    state: s.state,
    visionMs: s.visionMs,
    framesDropped: s.framesDropped,
    fastThreshold: s.fastThreshold,
    processingSize: `${s.processingWidth}x${s.processingHeight} f=${s.focalPx.toFixed(0)}${s.syncVideo ? " sync" : ""}`,
    backend: s.backend,
    pose: s.pose
      ? {
          ...rotationToEulerDeg(Float64Array.from(s.pose.rotation)),
          translationDirection: s.pose.translationDirection,
          model: s.pose.model,
          parallaxPx: s.pose.parallaxPx,
          confidence: s.pose.confidence,
          translationConfidence: s.pose.translationConfidence,
          correspondences: s.pose.correspondences,
        }
      : null,
    map: s.mapPose
      ? {
          landmarks: s.mapPose.landmarkCount,
          pnpInliers: s.mapPose.inlierCount,
          reprojPx: s.mapPose.meanReprojectionErrorPx,
          cameraCenter: s.mapPose.cameraCenter,
          framesSinceTracked: s.mapPose.framesSinceTracked,
          deltaTranslationM: s.worldReady ? s.mapPose.deltaTranslation * s.worldScale : NaN,
          deltaRotationDeg: s.mapPose.deltaRotationDeg,
          translationHeld: s.mapPose.translationHeld,
          translationPredicted: s.mapPose.translationPredicted,
          jumpRejected: s.mapPose.jumpRejected,
          source: s.mapPose.source,
          mapInliers: s.mapPose.mapInlierCount,
          planeInliers: s.mapPose.planeInlierCount,
          rejectReason: s.mapPose.rejectReason,
          mapCandidate: hudCandidate(s.mapPose.mapCandidate, s.worldReady ? s.worldScale : NaN),
          planeCandidate: hudCandidate(s.mapPose.planeCandidate, s.worldReady ? s.worldScale : NaN),
          gateMaxM: s.worldReady ? s.mapPose.gateMaxTranslation * s.worldScale : NaN,
          gateMaxDeg: s.mapPose.gateMaxRotationDeg,
          sourceDeltaM: s.worldReady ? s.mapPose.sourceDeltaTranslation * s.worldScale : NaN,
          sourceDeltaDeg: s.mapPose.sourceDeltaRotationDeg,
          history: s.mapPose.sourceHistory,
          relinked: s.mapPose.reassociated,
          relocalized: s.mapPose.relocalized,
        }
      : null,
    lostMs: s.lostMs,
    plane: s.plane
      ? {
          normal: s.plane.normal,
          inliers: s.plane.inlierCount,
          rms: s.plane.rmsResidual,
          horizontalness: s.plane.horizontalness,
          horizontal: s.plane.horizontal,
          stableFrames: s.plane.stableFrames,
          confidence: s.plane.confidence,
          found: s.plane.found,
          usedGravity: s.plane.usedGravity,
        }
      : null,
    gravityAvailable: s.gravityAvailable,
    planeSearch: s.planeSearch,
    world: { ready: s.worldReady, scale: s.worldScale, placed: s.placedObjects },
    planePose: s.planePose
      ? {
          tracked: s.planePose.tracked,
          inliers: s.planePose.inlierCount,
          candidates: s.planePose.candidateCount,
          ratio: s.planePose.inlierRatio,
          errorPx: s.planePose.reprojectionErrorPx,
          confidence: s.planePose.confidence,
        }
      : null,
    cameraWorld: s.cameraWorldPosition,
    objects: s.objects,
    timing: s.frameTimestampMs > 0 ? { frameMs: s.frameTimestampMs, poseMs: s.poseTimestampMs, ageMs: s.poseAgeMs, stale: s.poseStale } : null,
    reloc: s.relocalization
      ? {
          keyframes: s.relocalization.keyframes,
          attempt: s.relocalization.attempt,
          inliers: s.relocalization.inlierCount,
          errorPx: s.relocalization.meanReprojectionErrorPx,
          match: s.relocalization.matchScore,
          inlierRatio: s.relocalization.inlierRatio,
          spatialCells: s.relocalization.spatialCells,
          keyframeId: s.relocalization.keyframeId,
          successes: s.relocalization.successCount,
          reason: s.relocalization.reason,
          rejectCode: s.relocalization.rejectCode,
          jumpM: s.worldReady ? s.relocalization.jumpTranslation * s.worldScale : NaN,
          jumpDeg: s.relocalization.jumpRotationDeg,
          postM: s.worldReady ? s.relocalization.postDeltaTranslation * s.worldScale : NaN,
          postDeg: s.relocalization.postDeltaRotationDeg,
          postInconsistent: s.relocalization.postInconsistent,
        }
      : null,
    build: typeof __BUILD_LABEL__ === "string" ? __BUILD_LABEL__ : "dev",
  });
  requestAnimationFrame(refreshHud);
}
requestAnimationFrame(refreshHud);

// Test hook for the Playwright smoke test and for manual inspection.
declare global {
  interface Window {
    __ar: {
      session: ARSession;
      stats: () => ReturnType<ARSession["getStats"]>;
      start: () => Promise<void>;
    };
  }
}
window.__ar = {
  session,
  stats: () => session.getStats(),
  start: () => session.start(),
};
