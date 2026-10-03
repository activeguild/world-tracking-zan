import { ARSession } from "./ar/ARSession";
import { ARError, TrackingState } from "./ar/ARState";
import { DebugOverlay } from "./debug/DebugOverlay";
import { rotationToEulerDeg } from "./math/Pose";
import { GravityProvider, parseGravityOverride } from "./sensors/GravityProvider";
import "./style.css";

/**
 * Demo: camera → feature tracking → relative pose → landmark map → plane.
 *
 * Query parameters:
 *   ?debug=1          enable `[AR]` console logs
 *   ?worker=0         run the vision engine on the main thread
 *   ?hud=0            hide the HUD
 *   ?gravity=x,y,z    override the gravity direction (camera frame), for tests
 */
const params = new URLSearchParams(location.search);
const debugLog = params.get("debug") === "1";
const useWorker = params.get("worker") !== "0";
const showHud = params.get("hud") !== "0";
const gravityOverride = parseGravityOverride(params.get("gravity"));
const gravityProvider = new GravityProvider();

const app = document.getElementById("app") as HTMLElement;
const video = document.getElementById("video") as HTMLVideoElement;
const overlay = document.getElementById("overlay") as HTMLCanvasElement;
const startButton = document.getElementById("start") as HTMLButtonElement;
const messageEl = document.getElementById("message") as HTMLElement;

const hud = new DebugOverlay(app);
hud.visible = showHud;

const session = new ARSession({
  video,
  overlayCanvas: overlay,
  config: {
    debug: { log: debugLog, overlay: true },
    useWorker,
  },
  gravitySource: () => gravityOverride ?? gravityProvider.gravityCamera,
});

session.on("trackingStateChanged", (state) => {
  messageEl.textContent = userMessageFor(state, session.getStats().quality.lowFeature);
});

session.on("planeFound", () => {
  messageEl.textContent = userMessageFor(session.state, false);
});

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
      return "平面を検出しました";
    case TrackingState.TRACKING_LOST:
      return "Tracking lost — ゆっくり元の位置に戻してください";
    default:
      return state;
  }
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
    fastThreshold: s.fastThreshold,
    processingSize: `${s.processingWidth}x${s.processingHeight}`,
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
          translation: s.mapPose.translation,
          framesSinceTracked: s.mapPose.framesSinceTracked,
        }
      : null,
    plane: s.plane
      ? {
          normal: s.plane.normal,
          inliers: s.plane.inlierCount,
          horizontalness: s.plane.horizontalness,
          horizontal: s.plane.horizontal,
          stableFrames: s.plane.stableFrames,
          confidence: s.plane.confidence,
          found: s.plane.found,
          usedGravity: s.plane.usedGravity,
        }
      : null,
    gravityAvailable: s.gravityAvailable,
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
