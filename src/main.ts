import { ARSession } from "./ar/ARSession";
import { ARError } from "./ar/ARState";
import { GUIDANCE_TEXT_JA, getGuidance, worldPhase } from "./ar/Guidance";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import type * as THREE from "three";
import { DebugOverlay, type HudCandidate, type HudRelocDiagnostics } from "./debug/DebugOverlay";
import { rotationToEulerDeg } from "./math/Pose";
import type { RelocalizationDiagnostics } from "./vision/Relocalizer";
import type { PoseCandidateReport } from "./vision/types";
import type { ARObject } from "./rendering/ARObject";
import { GravityProvider, parseGravityOverride } from "./sensors/GravityProvider";
import "./style.css";

/**
 * Demo: camera → feature tracking → relative pose → landmark map → plane
 *       → tap to place a cube or a GLB model.
 *
 * Query parameters:
 *   ?debug=1          debug mode: HUD + debug drawing on at start, `[AR]` console logs
 *                     (default off — the ☰ button toggles the HUD at any time, v7 §15–§20)
 *   ?worker=0         run the vision engine on the main thread
 *   ?hud=0 / ?hud=1   force the initial HUD state independently of ?debug
 *   ?gravity=x,y,z    override the gravity direction (camera frame), for tests
 *   ?model=URL        place this GLB instead of the cube (same-origin or CORS-enabled)
 *   ?size=0.15        footprint of the model in meters (default 0.15)
 *   ?fov=66           camera field of view along the long side (degrees)
 *   ?sync=0           show the live video instead of the pose-synchronized frame
 *   ?grab=2d|gl|auto  frame grabber: 2D canvas readback, WebGL shader, or auto (default: measure, then pick)
 *   ?smooth=1         enable pose smoothing (off by default while the raw pose is validated)
 *   ?dist=0.5         assumed camera→plane distance in meters (scale)
 *   ?walk=1           the placed object walks back and forth on the plane (object motion test)
 *   ?refine=1         re-enable landmark depth refinement (A/B against the fixed map, v2 §26)
 *   ?planetrack=1     experimental plane-relative pose instead of landmark PnP
 */
const params = new URLSearchParams(location.search);
const debugLog = params.get("debug") === "1";
const useWorker = params.get("worker") !== "0";
// HUD off by default for the AR experience; ?debug=1 (or ?hud=1) starts with it on.
const hudParam = params.get("hud");
const showHud = hudParam !== null ? hudParam === "1" : debugLog;
const gravityOverride = parseGravityOverride(params.get("gravity"));
const gravityProvider = new GravityProvider();
const modelUrl = params.get("model");
const modelSize = Number(params.get("size") ?? "0.15") || 0.15;
const fovDeg = Number(params.get("fov") ?? "") || undefined;
const syncVideo = params.get("sync") !== "0";
// Frame grabber (v16): auto picks canvas2d or webgl from the measured grab time; ?grab=2d / ?grab=gl force one.
const grabParam = ({ "2d": "canvas2d", gl: "webgl", auto: "auto" } as const)[params.get("grab") ?? ""];
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
const debugToggle = document.getElementById("debug-toggle") as HTMLButtonElement;
// Console output only in debug mode (v9 §23–§26); real errors still surface through showError().
if (debugLog) console.log(`[AR] build ${typeof __BUILD_LABEL__ === "string" ? __BUILD_LABEL__ : "dev"}`);

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
      ...(grabParam ? { grabber: grabParam } : {}),
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

// Debug HUD on/off (v7 §15–§18): UI state only. The engine, its diagnostics
// and the stats keep running; only the HUD, the feature overlay and the
// plane grid are shown or hidden.
let hudVisible = showHud;
function applyHudVisibility(): void {
  hud.visible = hudVisible;
  overlay.style.display = hudVisible ? "" : "none";
  session.setDebugVisualization(hudVisible);
  debugToggle.classList.toggle("on", hudVisible);
  debugToggle.setAttribute("aria-pressed", String(hudVisible));
}
debugToggle.addEventListener("click", (ev) => {
  ev.stopPropagation();
  hudVisible = !hudVisible;
  applyHudVisibility();
});
applyHudVisibility();

// Guidance (v10 §13–§17): decided every frame from the state and a few
// quality facts by the pure Guidance module; the DOM is touched only when
// the text changes. "Go back to where you were" is produced only for an
// established world that has been lost for relocGuidanceDelayMs.
let lastGuidanceText: string | null = null;
function refreshGuidance(): void {
  const s = session.getStats();
  const key = getGuidance({
    state: s.state,
    worldEstablished: s.worldEstablished,
    lowFeature: s.quality.lowFeature,
    planeCandidate: s.plane !== null,
    lostMs: s.lostMs,
    relocGuidanceDelayMs: session.config.world.relocGuidanceDelayMs,
    planeRecovery: s.planeRecovery?.state,
    planeSearchStage: s.planeSearch?.stage ?? null,
    planeSearchStageMs: s.planeSearchStageMs,
    sidewaysGuidanceDelayMs: session.config.world.sidewaysGuidanceDelayMs,
  });
  let text = GUIDANCE_TEXT_JA[key];
  if (key === "TAP_TO_PLACE" && placed) text = "";
  if (text !== lastGuidanceText) {
    lastGuidanceText = text;
    messageEl.textContent = text;
  }
}
session.on("frame", refreshGuidance);
session.on("trackingStateChanged", refreshGuidance);

// Phase 4: tap → hit test → place / move the object (spec §27, §68).
// The first tap places the cube (or the GLB when ?model= is given), later
// taps move it. Objects are fixed in world space; only the camera moves.
let placed: ARObject | undefined;
session.on("worldLost", () => {
  placed = undefined;
  lastGuidanceText = null; // re-evaluate on the next frame
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
  refreshGuidance();
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

/** Short stage names for the RELOC section (v6 §6, v9 §29: unambiguous). */
const RELOC_FAIL_SHORT: Record<string, string> = {
  coarse: "ncc_match",
  landmarks: "landmarks_left",
  lk: "lk_tracking",
  pnp_inliers: "inliers",
  invalid: "pose_invalid",
  reprojection: "reprojection_error",
  ratio: "inlier_ratio",
  spatial: "spatial_distribution",
  ncc: "ncc",
  jump: "pose_jump",
};

function hudRelocDiagnostics(d: RelocalizationDiagnostics | null, age: number): HudRelocDiagnostics | null {
  if (!d) return null;
  const b = d.best;
  return {
    keyframes: d.keyframes,
    tried: d.candidatesTried,
    coarsePassed: d.coarsePassed,
    lkPassed: d.lkPassed,
    pnpTested: d.pnpTested,
    pnpPassed: d.pnpPassed,
    validated: d.validated,
    bestCoarseScore: d.bestCoarseScore,
    ranked: d.ranked.map((k) => ({ keyframeId: k.keyframeId, score: k.rankScore, selected: k.selected, suppressed: k.suppressed, unusable: k.unusable, alive: k.alive })),
    lkCandidates: d.lkCandidates,
    pnpCandidates: d.pnpCandidates,
    retrySuppressed: d.retrySuppressed,
    unusableKeyframes: d.unusableKeyframes,
    usedPreparedRanking: d.usedPreparedRanking,
    jumpLimitsActive: d.jumpLimitsActive,
    best: b
      ? {
          keyframeId: b.keyframeId,
          stage: b.stage,
          inliers: b.inlierCount,
          errorPx: b.meanReprojectionErrorPx,
          inlierRatio: b.inlierRatio,
          spatialCells: b.spatialCells,
          coarseScore: b.coarseScore,
          lk: {
            observations: b.lkObservations,
            tracked: b.lkTracked,
            fb: b.lkStatus.fbError,
            residual: b.lkStatus.highResidual,
            far: b.lkStatus.tooFar,
            diverged: b.lkStatus.diverged,
            oob: b.lkStatus.outOfBounds,
            texture: b.lkStatus.lowTexture,
            reason: b.lkFailureReason,
          },
          validation: b.validation
            ? {
                inliers: b.validation.inliers,
                requiredInliers: b.validation.requiredInliers,
                errorPx: b.validation.reprojectionErrorPx,
                maxErrorPx: b.validation.maxReprojectionErrorPx,
                relaxedErrorPx: b.validation.relaxedReprojectionErrorPx,
                level: b.validation.level,
                ncc: b.validation.nccScore,
                requiredNcc: b.validation.requiredNccScore,
                nccPassed: b.validation.nccPassed,
                translationJumpPassed: b.validation.translationJumpPassed,
                rotationJumpPassed: b.validation.rotationJumpPassed,
                translationJump: b.validation.translationJump,
                rotationJumpDeg: b.validation.rotationJumpDeg,
                maxTranslationJump: b.validation.maxTranslationJump,
                maxRotationJumpDeg: b.validation.maxRotationJumpDeg,
                ratio: b.validation.inlierRatio,
                minRatio: b.validation.minInlierRatio,
                cells: b.validation.coveredCells,
                totalCells: b.validation.totalCells,
                minCells: b.validation.minSpatialCells,
                coverage: b.validation.spatialCoverage,
                minCoverage: b.validation.minSpatialCoverage,
                inliersPassed: b.validation.inliersPassed,
                reprojectionPassed: b.validation.reprojectionPassed,
                ratioPassed: b.validation.ratioPassed,
                spatialPassed: b.validation.spatialPassed,
                coveragePassed: b.validation.coveragePassed,
                posePassed: b.validation.posePassed,
                rejectReason: b.validation.rejectReason,
              }
            : null,
        }
      : null,
    fail: b && b.stage !== "ok" ? (RELOC_FAIL_SHORT[b.stage] ?? b.stage) : d.rejectCode === "no_keyframes" ? "no keyframes" : null,
    age: Math.max(0, age),
  };
}

// HUD refresh loop (independent of vision rate). With the HUD hidden nothing
// is formatted or written to the DOM (v9 §28); the engine's diagnostics are
// still computed and available through getStats().
let lastHudMs = 0;
function refreshHud(): void {
  if (!hudVisible) {
    requestAnimationFrame(refreshHud);
    return;
  }
  const hud0 = performance.now();
  const s = session.getStats();
  hud.update({
    renderFps: s.renderFps,
    visionFps: s.visionFps,
    featureCount: s.quality.featureCount,
    trackedCount: s.quality.trackedCount,
    inlierCount: s.quality.inlierCount,
    planeConfidence: s.quality.planeConfidence,
    state: s.state,
    phase: worldPhase({
      state: s.state,
      worldEstablished: s.worldEstablished,
      planeCandidate: s.plane !== null,
      lostMs: s.lostMs,
      relocGuidanceDelayMs: session.config.world.relocGuidanceDelayMs,
      planeRecovery: s.planeRecovery?.state,
    }),
    worldEstablished: s.worldEstablished,
    visionMs: s.visionMs,
    framesDropped: s.framesDropped,
    engineTiming: s.engineTiming,
    mainTiming: s.mainTiming,
    hudMs: lastHudMs,
    fastThreshold: s.fastThreshold,
    processingSize: `${s.processingWidth}x${s.processingHeight} f=${s.focalPx.toFixed(0)}${s.syncVideo ? " sync" : ""}${s.grabber ? ` grab ${s.grabber === "webgl" ? "gl" : "2d"}` : ""}`,
    backend: s.backend,
    motion: s.motion
      ? {
          level: s.motion.level,
          medianPx: s.motion.medianDisplacementPx,
          maxPx: s.motion.maxDisplacementPx,
          before: s.motion.trackedBefore,
          after: s.motion.trackedAfter,
          fbRejects: s.motion.forwardBackwardRejects,
          tooFar: s.motion.tooFarRejects,
          prediction: s.motion.predictionMode,
          searchScale: s.motion.searchScale,
        }
      : null,
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
          reprojection: s.mapPose.reprojection,
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
          observations: s.mapPose.observations,
          requiredInliers: s.mapPose.requiredInliers,
          recoveryMode: s.mapPose.recoveryMode,
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
    planeRecovery: s.planeRecovery
      ? {
          active: s.planeRecovery.active,
          reason: s.planeRecovery.reason,
          state: s.planeRecovery.state,
          mapHealthy: s.planeRecovery.mapHealthy,
          mapInliers: s.planeRecovery.mapInliers,
          seedCandidates: s.planeRecovery.seedCandidates,
          seededPoints: s.planeRecovery.seededPoints,
          candidateFound: s.planeRecovery.candidateFound,
          candidateCommitted: s.planeRecovery.candidateCommitted,
          previousCandidateHeld: s.planeRecovery.previousCandidateHeld,
          stableFrames: s.planeRecovery.stableFrames,
          requiredStableFrames: s.planeRecovery.requiredStableFrames,
          elapsedMs: s.planeRecovery.recoveryElapsedMs,
          recoveries: s.planeRecovery.recoveries,
        }
      : null,
    triangulation: s.mapPose?.triangulation ?? null,
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
    objectVisibility: s.objectVisibility,
    timing: s.frameTimestampMs > 0 ? { frameMs: s.frameTimestampMs, poseMs: s.poseTimestampMs, ageMs: s.poseAgeMs, stale: s.poseStale } : null,
    reloc: s.relocalization
      ? {
          keyframes: s.relocalization.keyframes,
          keyframesCreated: s.relocalization.keyframesCreated,
          keyframesEvicted: s.relocalization.keyframesEvicted,
          attempt: s.relocalization.attempt,
          inliers: s.relocalization.inlierCount,
          errorPx: s.relocalization.meanReprojectionErrorPx,
          level: s.relocalization.level,
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
          diag: hudRelocDiagnostics(s.relocalization.diagnostics, s.relocalization.framesSinceAttempt),
          searchStage: s.relocalization.searchStage,
          preparing: s.relocalization.preparing,
          pendingKeyframeId: s.relocalization.pendingKeyframeId,
          candidateAgeFrames: s.relocalization.candidateAgeFrames,
          timeline: s.relocalization.timeline
            ? {
                attempts: s.relocalization.timeline.attempts,
                firstCoarseMs: s.relocalization.timeline.firstCoarseMatchMs,
                firstLkMs: s.relocalization.timeline.firstLkSuccessMs,
                firstPnpMs: s.relocalization.timeline.firstPnpSuccessMs,
                validationMs: s.relocalization.timeline.validationSuccessMs,
                confirmationMs: s.relocalization.timeline.confirmationSuccessMs,
              }
            : null,
        }
      : null,
    build: typeof __BUILD_LABEL__ === "string" ? __BUILD_LABEL__ : "dev",
  });
  // v16: the HUD's own cost (formatting + DOM) is part of the main-thread budget.
  lastHudMs = performance.now() - hud0;
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
      /** Compare the 2D and WebGL grabbers on the current frame (v16 test hook). */
      grabCompare: () => ReturnType<ARSession["compareGrabbers"]>;
    };
  }
}
window.__ar = {
  session,
  stats: () => session.getStats(),
  start: () => session.start(),
  grabCompare: () => session.compareGrabbers(),
};
