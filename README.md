# world-tracking-zan

Self-built WebAR plane detection and visual tracking engine for iOS Safari /
Android Chrome. No ARKit / ARCore / WebXR plane detection, no 8th Wall: a
single rear camera, `getUserMedia()`, and our own vision pipeline running in a
Web Worker.

The full specification lives in [CLAUDE.md](./CLAUDE.md). Development is
strictly phased; **Phase 1 (feature tracking) is implemented**, later phases
are not started.

## Phase 1 — what works today

```
Camera (getUserMedia, 1280×720 ideal)
  → processing frame 640×480 / 640×360, grayscale (main thread, pooled buffers)
  → Web Worker (Transferable ArrayBuffer)
      → 3-level image pyramid + gradients
      → pyramidal Lucas-Kanade (15×15 window) with forward-backward check
      → Homography RANSAC outlier rejection
      → FAST-9 replenishment: NMS, adaptive threshold, grid distribution, min distance
      → TrackingQuality + state machine (INITIALIZING / SEARCHING_FEATURES / TRACKING / TRACKING_LOST)
  → main thread: HUD + feature / motion-vector overlay
```

Everything is TypeScript on TypedArrays (no OpenCV.js). Each stage is a
separate module with explicit input / output / failure conditions so that a
WASM/SIMD port can replace it later without changing the pipeline.

## Running

```bash
npm install
npm run dev          # https://<your-lan-ip>:5173  (self-signed cert; accept it on the phone)
```

Open the URL on an iPhone (Safari) or Android (Chrome), tap **Start Camera**,
point at a textured desk or floor. Green dots are tracked features (brighter =
older), yellow dots are freshly detected, white lines are per-frame motion.

Query parameters:

| param       | effect                                      |
|-------------|---------------------------------------------|
| `?debug=1`  | `[AR] state=… features=… tracked=…` console log every second |
| `?worker=0` | run the vision engine on the main thread    |
| `?hud=0`    | hide the HUD                                |

## Tests

```bash
npm run typecheck     # tsc --noEmit
npm test              # Vitest unit tests (synthetic images, deterministic RNG)
npm run test:browser  # build + headless Chromium with a synthetic camera feed (.y4m)
npm run check         # typecheck + unit tests
```

The browser test drives the real page through the real worker and prints a
report (`visionFps`, `visionMs`, feature / tracked / inlier counts, state
histogram).

## Tuning

All thresholds live in `src/ar/ARConfig.ts` (`DEFAULT_CONFIG`) and can be
overridden per session:

```ts
const ar = new ARSession({
  video,
  overlayCanvas,
  config: { features: { maxFeatures: 400 }, tracker: { windowSize: 21 } },
});
await ar.start();
ar.on("trackingStateChanged", (state) => console.log(state));
ar.on("frame", (r) => console.log(r.quality.trackedCount));
```

## Layout

```
src/
  ar/        ARSession (public API), ARState (state machine, error codes), ARConfig
  camera/    CameraManager (getUserMedia), CameraFrame (resize + grayscale), CameraIntrinsics
  vision/    ImagePyramid, FeatureDetector (FAST-9), FeatureTracker (LK + FB),
             OutlierRejection (Homography RANSAC), VisionEngine, TrackingQuality, types
  math/      Matrix (3×3, linear solve), Homography (normalized DLT)
  worker/    protocol, VisionWorker (worker entry), VisionWorkerClient (+ main-thread fallback)
  debug/     DebugOverlay (HUD), FeatureRenderer, Logger
tests/
  math/ vision/ ar/ camera/   unit tests
  integration/                browser smoke test (Playwright + Chromium)
  helpers/                    synthetic image generators, y4m writer
```
