# world-tracking-zan

Self-built WebAR plane detection and visual tracking engine for iOS Safari /
Android Chrome. No ARKit / ARCore / WebXR plane detection, no 8th Wall: a
single rear camera, `getUserMedia()`, and our own vision pipeline running in a
Web Worker.

The full specification lives in [CLAUDE.md](./CLAUDE.md). Development is
strictly phased; **Phases 1–5 (feature tracking, relative camera pose,
landmark map + plane detection, world coordinate + Three.js placement,
keyframes + relocalization) are implemented**, later phases are not started.

## Phase 5 — keyframes and relocalization

```
tracked frames → keyframes (pose, image pyramid, landmark observations;
                 new one after 10° rotation / 40 px parallax / 90 frames; max 8)
camera lost in the map (PnP fails) → RELOCALIZING, world kept, objects hold 1.5 s
  → per frame, most recent keyframes first:
      coarse 1/8-res NCC shift search (±192 px)
      → pyramidal LK keyframe → current, seeded with the shift
      → PnP from the keyframe pose (LM + Huber), ≥ 20 inliers, ≤ 2 px
  → success: pose restored in the same map, observations become live tracks,
             placed objects reappear where they were
  → no success for 5 s → map reset (world lost)
```

Placing a GLB instead of the cube: `?model=URL&size=0.15` (same-origin or
CORS-enabled URL); in code `ar.placeObject(gltf.scene, hit, 0.15)`.

## Phase 4 — world coordinate, hit test, Three.js placement

```
first PLANE_FOUND
  → world frame fixed once: origin = plane center, +Y = plane normal,
    −Z ≈ camera view projected onto the plane, scale from an assumed
    camera→plane distance (0.5 m, `world.assumedPlaneDistanceMeters`)
  → every frame: map-frame camera pose → Three.js camera (CoordinateSystem.ts),
    One Euro smoothing on position / rotation, projection from intrinsics
  → tap: CSS px → processing px → camera ray → map ray → ∩ plane Y = 0 → world point
  → cube placed at the hit stays fixed in world space; only the camera moves
  → tracking lost: objects hold their pose for 1.5 s, then hide until tracking returns
  → map reset: world dropped (`worldLost`), objects removed
```

```ts
const ar = new ARSession({ video, overlayCanvas, threeCanvas });
await ar.start();
ar.on("worldReady", () => {});
const hit = ar.hitTest(x, y);          // CSS px on the video element
if (hit) ar.placeCube(hit);            // or ar.placeObject(gltf.scene, hit)
```

`threeScene` / `threeCamera` can be passed instead of `threeCanvas` to drive
an application-owned scene. All frame conversions live in
`src/math/CoordinateSystem.ts`.

## Phase 3 — landmark map and plane detection

```
two-view pose with enough parallax (≥ 20 px, confident translation)
  → initialize the map: triangulate reference↔current inliers
    (map frame = reference camera, |t| = 1 → monocular scale)
  → every frame: PnP (LM + Huber) against landmarks seen by live tracks
                 → camera pose in the map frame, consistent scale
                 triangulate new landmarks from each track's anchor observation
                 cull outliers / stale landmarks (cap 1000)
  → RANSAC plane on the landmarks → PCA refit
  → horizontality: |cos(normal, gravity)| ≥ 0.90   (DeviceMotion gravity, camera frame)
                   fallback without gravity: camera −Y as up, threshold 0.5
  → temporal stability (5 consecutive frames) → PLANE_FOUND, `planeFound` event
```

Gravity comes from `DeviceMotion` (`src/sensors/GravityProvider.ts`); iOS asks
for permission on the Start tap. `?gravity=x,y,z` overrides it for testing.
Landmarks and the plane grid are projected onto the 2D overlay
(`PlaneRenderer`); the Three.js version comes with Phase 4.

## Phase 2 — relative camera pose

On top of the tracked features the engine solves two-view geometry between a
*reference frame* and the current frame:

```
reference ↔ current correspondences (tracks that survived since the reference)
  → Homography RANSAC (pixels)      → H
  → Essential RANSAC (normalized)   → E   (normalized 8-point, Sampson gate)
  → model selection: H_inliers / (H_inliers + E_inliers) > 0.45 → planar / pure rotation
      homography → Faugeras decomposition (R, t/d, n), positive-depth test, twin disambiguation
      essential  → recoverPose (4 candidates, cheirality by triangulation)
      parallax < 2 px → rotation only (translation unobservable)
  → VisionOutput.pose: accumulated R (origin → current), unit t direction, model, confidences
```

The reference frame is renewed when too few tracks still link to it or when
the parallax grows large; rotations are composed across renewals. Translation
is scale-free (unit direction); scale is fixed by the plane in Phase 3/4.

## Phase 1 — feature tracking

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
| `?gravity=x,y,z` | override the gravity direction (camera frame) |
| `?model=URL&size=0.15` | place this GLB (footprint in meters) instead of the cube |
| `?fov=66`   | camera field of view along the long side (degrees) |
| `?dist=0.5` | assumed camera→plane distance in meters (monocular scale; ~1.3 for a floor) |
| `?sync=0`   | show the live video instead of the pose-synchronized frame |
| `?smooth=1` | enable pose smoothing (off by default while the raw pose is validated) |
| `?refine=1` | re-enable landmark depth refinement (A/B against the fixed map) |
| `?walk=1`   | the placed object walks back and forth on the plane (object-motion test) |
| `?planetrack=1` | experimental plane-relative pose instead of landmark PnP |

## Fixed map (drift fix, v2)

Objects slid while the phone moved. The map is the coordinate system the
world is anchored to, so the camera pose must not rewrite it: landmark depth
refinement (re-triangulation as the baseline grows) is now off by default
(`landmarks.enableLandmarkDepthRefinement`). PnP telemetry (`cameraCenter`,
per-frame Δtranslation / Δrotation, `translationHeld`, pose `source`) and a
sectioned HUD (TRACKING / CAMERA / WORLD / OBJECT / TIMING) make the on-device
A/B measurable: moving the camera 10 cm must change `world C` by ≈ 0.10 while
`Object 1` stays put. `src/vision/PlaneTracker.ts` is an experimental,
default-off alternative estimator (features lifted onto the fixed plane, PnP
against them = plane-induced homography with known n, d).

One canonical camera pose (v3): landmark PnP, plane PnP and propagation are
*candidates*; `validatePoseCandidate` (continuity against the previous pose
and, for the plane candidate, agreement with the map candidate) decides, with
a cooldown against map ↔ plane flapping. New plane points are lifted only from
a trusted map pose. A short tracking loss holds the last good pose (10 s debug
hold); map and world reset only after 10 s lost *and* ≥ 10 failed
relocalization attempts. The HUD shows the pose source, its history and the
rejection reason. Landmarks whose tracks died are re-linked every frame to
replenished tracks sitting on their projection (4 px), so a burst of fast
motion does not leave PnP without observations; pruning by age counts only
tracked frames.

Fast motion: 4 LK pyramid levels and constant-velocity seeding
(`tracker.pyramidLevels`, `tracker.predictMotion`) roughly double the per-frame
displacement that stays trackable.

Temporal gate (v4): three independent judgements decide about a pose
candidate — PnP quality (inliers / error, "trusted"), temporal continuity
(translation / rotation vs the previous canonical pose) and map ↔ plane
agreement. *Trusted never skips the gate*: a well-supported solve that moves
the camera implausibly far in one frame is a different pose, not fast motion,
and is rejected (`landmarks.jumpReject*`, `JUMP` on the HUD's PnP row, the
structured reason on the `MAP cand` / `PLANE cand` rows). While lost the
reference pose is a prediction, so the limits widen with the lost frames
(`jumpRejectLostGrowthPerFrame`, capped at `jumpRejectMaxLostGrowth`); after
`longLostFrames` the normal recovery needs `minRecoveryInliersLong` inliers.
The plane tracker's probation / off-plane state is updated only from an
accepted candidate, against the canonical pose.

Relocalization (v5): a keyframe match is a *candidate* that must pass a global
validation — PnP inliers, reprojection error, coarse match score, inlier ratio
and the spatial distribution of the inliers over a 3×3 grid
(`relocalization.minInlierRatio`, `minSpatialCells`) — and, unless it is
clearly high quality (`immediateInliers`, `immediateMaxErrorPx`), a
confirmation frame in which the next attempt reproduces the same pose
(`confirmationFrames`, `confirmTranslationDepthRatio`, `confirmRotationDeg`).
It is not gated against the held pose (a correct return can be far from it);
instead the map PnP of the following `postRelocMonitorFrames` is compared with
the relocalized pose (`postDelta*`, `postInconsistent`). The HUD `RELOC` rows
show the candidate's quality, its jump from the held pose, the structured
reject code and the post-relocalization agreement; `R` marks it in the source
history.

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
             OutlierRejection (Homography RANSAC), PoseEstimator (H/E model selection),
             LandmarkMap, MapTracker (init / PnP / triangulation), PlaneDetector,
             PlaneTracker (experimental plane-relative pose),
             Keyframe, Relocalizer (coarse NCC shift + LK + PnP), VisionEngine, TrackingQuality, types
  math/      Matrix (3×3, linear solve), Homography (normalized DLT), Decomposition (Jacobi eigen, SVD),
             Pose (rotations, quaternions), EssentialMatrix (8-point, RANSAC, recoverPose),
             HomographyDecomposition (Faugeras), Triangulation, Plane (RANSAC), PnP (LM + Huber)
             CoordinateSystem (map ↔ world ↔ Three.js, projection), Ray, OneEuroFilter
  ar/        WorldAnchor (world frame from the first plane, hit test)
  rendering/ ARCamera, ARWorld, ARObject (world-space motion API), ARRenderer, FramePresenter (pose-synchronized frame)
  sensors/   GravityProvider (DeviceMotion → camera frame)
  worker/    protocol, VisionWorker (worker entry), VisionWorkerClient (+ main-thread fallback)
  debug/     DebugOverlay (HUD), FeatureRenderer, PlaneRenderer, Logger
tests/
  math/ vision/ plane/ ar/ camera/   unit tests
  integration/                browser smoke test (Playwright + Chromium)
  helpers/                    synthetic image generators, y4m writer
```
