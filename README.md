# world-tracking-zan

Self-built WebAR plane detection and visual tracking engine for iOS Safari /
Android Chrome. No ARKit / ARCore / WebXR plane detection, no 8th Wall: a
single rear camera, `getUserMedia()`, and our own vision pipeline running in a
Web Worker.

The full specification lives in [CLAUDE.md](./CLAUDE.md). Development is
strictly phased; **Phases 1–5 (feature tracking, relative camera pose,
landmark map + plane detection, world coordinate + Three.js placement,
keyframes + relocalization) and Phase 7 (local bundle adjustment) are
implemented**; Phase 6 (IMU) and Phase 8 (WASM / SIMD) are not started.

## Phase 7 — local bundle adjustment

```
new keyframe → bundleAdjust(stored keyframes, landmarks seen by ≥ 1 of them)
  first keyframe fixed (map origin), other poses + landmark positions free
  Levenberg–Marquardt, Huber 2 px, gross outliers (> 10 px) left out,
  landmarks eliminated with the Schur complement (dense ≤ 8×6 pose block)
  weak priors: a landmark moving 10% of its depth / a keyframe moving 5% of
    the median depth or 3° costs like one observation off by 2 px
    (keyframes made by rotation alone leave depths nearly free otherwise)
  scale re-normalized to the median depth of the first keyframe's landmarks
  → the whole run is rejected when > 10% of the landmarks moved > 20% of
    their depth (never a partial write-back: that splits the map in two),
    or when the error did not drop by 15% (a consistent map gains nothing
    and would only be nudged along weakly constrained directions)
  → otherwise landmark positions and keyframe poses updated in place (next
    PnP, relocalization priors and plane fit see the refined map)
  → landmarks no keyframe observes are carried with their reference
    keyframe (the newest keyframe when they were triangulated): the whole
    map stays one frame, not a solved part and a stale part
  → the newest keyframe's correction (old map → new map) is applied to
    the canonical pose and, in the session, to the world anchor: placed
    objects stay put relative to the local map and do not hop on screen
  → track anchors reset (new triangulations use refined poses only)
```

Why: on Android the PnP error grew from 1 px (fresh map) to 2.5–3 px while
staying uniform over image position, parallax, track age and landmark age,
and the same landmarks at the same view went 1.5 → 2.3 px once later
landmarks joined the PnP (CLAUDE.md, recordings 14–19). Landmarks were
frozen at their creation pose, so groups created at different times
disagreed; BA makes them one map. A wrong focal length was ruled out with a
synthetic floor-plus-wall scene (`tests/vision/FocalEstimate.test.ts`).
`?ba=0` disables it for A/B; the HUD `BA` row reports each run (`REJECTED no
gain | shift`, `lm N +P prop +U none`, `corr` = the correction applied to
the pose and the world anchor) and the `BAlm` row splits the PnP error by
how the last run treated each landmark (solved / carried / untouched). A
run that moves a keyframe beyond twice its prior is rejected as a pose
jump; a keyframe that does this twice in a row is left out of later solves
(`REJECTED pose jump KF12 8.3°/0.21u`, `kf 6+1/8 (−1 out)`). The `Link`
row splits the PnP error by how each track got its landmark (triangulated
from the track itself / re-associated after a re-detection / injected by
the relocalizer): a re-linked group clearly worse than the native one is
association error, not map error.

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
| `?grab=2d` / `?grab=gl` / `?grab=auto` | frame grabber: 2D canvas readback, WebGL shader, or measure-then-pick (default) |
| `?debug=1` | debug mode: HUD + debug drawing on at start, `[AR]` console logs (the ☰ button toggles the HUD at any time; off by default) |
| `?hud=1` / `?hud=0` | force the initial HUD state independently of `?debug` |
| `?smooth=1` | enable pose smoothing (off by default while the raw pose is validated) |
| `?refine=1` | re-enable landmark depth refinement (A/B against the fixed map) |
| `?freeze=1` | no new landmarks once the world is established (A/B: map inconsistency vs camera model; tracking is lost when the camera leaves the first view) |
| `?ba=0` | disable the local bundle adjustment that runs on every new keyframe (Phase 7 A/B) |
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
displacement that stays trackable. v7 adds a motion level from the previous
frame's median track displacement (`mediumMotionPx` / `fastMotionPx`), a
displacement gate that widens only at medium / fast motion
(`mediumMotionSearchScale` / `fastMotionSearchScale`) and LK seeding with the
previous frame's RANSAC homography when it was well supported
(`homographyPrediction`, `predictionMinInliers`, `predictionMinInlierRatio`),
falling back to per-track velocity otherwise; `VisionOutput.motion` carries the
level, displacements, LK rejects and the prediction mode (HUD `Motion` / `LK`
rows). PnP, RANSAC, the gates and relocalization are untouched.

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

Relocalization diagnostics (v6): every attempt records, per keyframe tried,
the stage it dropped out at (`coarse → landmarks → lk → pnp → error → ratio →
spatial → ok`) plus stage counters and the best trial
(`RelocalizationResult.diagnostics`); the last attempt stays on the output
between attempts while lost. The HUD is laid out for a phone (one short row
per value, `RELOC` section only while lost): `KF 8 / try 2`, `NCC 2`, `LK 2`,
`PnP 2 ran / 0 ok`, `VAL 0`, `Best KF3 14i 3.18px`, `Fail ratio`, and a
`Need 14/24i obs 14 (recovery)` row naming the map-PnP recovery rule in force.
No threshold changed.

Relocalization validation breakdown (v9): every condition of a PnP candidate
is evaluated independently (`validateRelocalizationCandidate`): inliers,
reprojection error, inlier ratio, covered 3×3 cells and the bounding-box
coverage of the inliers (`minSpatialCoverage`, 0 = diagnostic only), plus a
finite-pose check; the reject reason is structured (`inliers |
reprojection_error | inlier_ratio | spatial_distribution | pose_invalid`) and
the HUD shows each value against its threshold with OK / NG. The
normal-tracking jump gate is not applied to relocalization candidates. The
debug HUD, the overlay and the `[AR]` console logs are off unless `?debug=1`;
with the HUD hidden nothing is formatted or written to the DOM per frame.

Initial scan vs relocalization (v10): `RELOCALIZING` is reachable only once a
world has been established for the current map (first `PLANE_FOUND`,
`worldEstablished`). Before that a lost map just means the user is scanning
somewhere else: no relocalization attempts, the map is dropped after
`landmarks.preWorldLostResetFrames` and re-initialized where the camera looks
now. Guidance comes from the pure `src/ar/Guidance.ts`; "return to where you
were" is produced only for an established world lost for longer than
`world.relocGuidanceDelayMs` (generic recovery wording before that). The HUD
shows the v10 phase (`INITIAL_SCAN / SURFACE_SCAN / PLANE_CANDIDATE /
WORLD_TRACKING / WORLD_LOST / RELOCALIZING`) and `World Established YES/NO`.

Plane recovery after fast motion (v11): on device the camera kept tracking
against the map after a fast move (LK 284/285, PnP 42 inliers / 0.38 px,
source MAP, lost 0 ms) while `Plane search 42pt best 32/20` never became a
plane and the world was never established. That HUD line only appears when
the detector produced *no candidate*; the two stages that swallow a
32-point height window are the re-classification around the fitted plane
and the 2D-extent test (a thin strip / compact cluster is not a plane). The
search now records its stage (`points | support | reclassify | extent |
candidate`) and the HUD `PLANE` section shows Search → Stage → Cand →
Commit → Stable plus the triangulation counters (`Tri`: why landmark-less
tracks did not become landmarks). A fast motion (v7 level `fast`) with a
healthy map (located this frame, inliers ≥ `landmarks.minPnPInliers`,
finite pose) and no world yet starts a *plane recovery*
(`src/vision/PlaneRecovery.ts`): only `PlaneDetector.resetForRecovery()`
runs (previous candidate, stability streak, miss counter), and while the
recovery is active the plane search is seeded with the landmarks seen as a
PnP inlier within `plane.recoverySeedMaxAgeFrames` — the view the camera has
now. The map, the canonical pose, the keyframes and the world are never
touched, relocalization is never entered because of a fast motion, and no
plane / PnP / LK / gate threshold changes. Phases `PLANE_RECOVERY` (still
fast) → `PLANE_WARMUP` → `PLANE_CANDIDATE` → `PLANE_FOUND`; guidance
「スマホをゆっくり動かしてください」→「平らな場所をゆっくり映してください」, never
"go back" before a world exists.

v11.1 refinements: the detector is reset exactly once per recovery (the
controller reports a start only on inactive → active; further fast motion
while active never restarts it, so the stability streak can accumulate
through a shaky scan). A second trigger uses the two-view estimate: the
reference↔current parallax crossing `pose.fullConfidenceParallaxPx` with
`confidence` / `translationConfidence` ≥ `landmarks.initMinTranslationConfidence`
and inliers ≥ `pose.minCorrespondences` (the on-device `Motion MED 6.3px /
2view par 45px conf 1.00/1.00 n116` case) — an event, not a level, so a slow
scan whose parallax has simply grown does not fire it. Candidate diagnostics
are about the current frame's search (`candidateFound = stage === "candidate"`,
`candidateCommitted` = horizontal); a candidate merely held through the
detector's grace period is `previousCandidateHeld`. Recovery state:
`inactive | starting | warmup | candidate | stable`.

Relocalization validation levels (v12): `strong | acceptable | reject`. The
strict acceptance stays at `relocalization.maxMeanErrorPx` (1.5 px, strong).
A candidate above it but within `relaxedMeanErrorPx` (3.0 px) is *acceptable*
only when every other condition passes — inliers, inlier ratio, spatial
cells, coverage, finite pose, coarse NCC ≥ `relaxedMinMatchScore` (0.5) and
the relocalization-specific jump limits (`maxTranslationJumpDepthRatio` ×
scene depth, `maxRotationJumpDeg`, generous by design and applied to every
level) — and it is then always held for a confirmation frame, never applied
at once. Reject reasons follow the order pose → inliers → ratio → spatial →
ncc → translation jump → rotation jump → reprojection error; beyond the
relaxed bound the error itself is the reason. Validated strong candidates
rank above acceptable ones. The HUD shows `Error 2.44/1.50px strict NG` /
`2.44/3.00px relaxed OK` / `Level ACCEPTABLE`.

Object visibility (v13): placed objects follow the trust in the *current*
camera pose, never the last pose. `src/ar/ObjectVisibility.ts` decides
`visible` + `reason` (`TRACKING_ACTIVE | TRACKING_LOST | RELOCALIZING |
CONFIRMING | WORLD_NOT_READY | POSE_INVALID`): hidden as soon as the state
machine declares the loss (its own `state.mapLostFrameTolerance`
hysteresis — a tolerated PnP dropout keeps what is shown), hidden while
relocalizing and while an applied relocalization is still in its
post-relocalization monitor window (`CONFIRMING`, strong or acceptable
alike), shown again only on a fresh map pose. Hiding keeps the transforms,
the world anchor, the map and the keyframes; the render loop, camera and
relocalization keep running. The 10 s `holdPoseOnLostMs` hold is gone. HUD
`OBJECT`: `Visible NO / Reason RELOCALIZING`, `Obj1 hidden`.

Relocalization recovery speed (v14): the goal is to reach the *right*
keyframe sooner, not to pass candidates more easily — PnP, the v12
validation levels, the confirmation and v13 visibility are untouched.
Every attempt ranks all keyframes by coarse similarity (zero-mean NCC
shift search on a 1/16 image, `rankSearchRadius`), only the top
`maxLkCandidatesPerFrame` (3; 4 on the first attempt) get the 1/8
refinement (`coarseRefineRadius`) → LK → PnP; a strong candidate stops the
search. Shifts are chosen by NCC × √overlap (`coarseMinOverlap`) so a large
shift with little overlap cannot win by chance, and a shift that leaves
fewer than `minInliers` observations inside the image skips LK as
`out_of_bounds`; an LK run that started inside the image and ended outside
it is reported as `diverged` (what a motion-blurred frame does).

Long loss (v16): the relocalization-specific jump limits vs the held pose
(`maxTranslationJumpDepthRatio`, `maxRotationJumpDeg`) apply only while
the held pose is fresh — up to `jumpLimitMaxLostFrames` (30) lost frames.
While lost the pose is dead-reckoned and after a turn that was lost
half-way it points the other way; the correct candidate at the start view
(263 inliers, 0.2 px in the synthetic replay, NCC 0.97 on device) was being
rejected as a `rotation_jump` on every attempt. Beyond that age the jump is
diagnostic only (HUD `Jump … no limit (long loss)`); the other validators
and the confirmation / post-monitor are unchanged. The HUD shows the engine's
per-stage time right under the FPS row (`Vis pyr … lk … 2view … map … plane …
reloc … = total`) and, below it, where the rest of the frame's wall time goes
(`Main grab … cap … q↑ … eng … q↓ … rtt …  world … show … ovl … hud …`:
frame grab, display copy, worker transport both ways — the wall-clock stamps
are comparable across threads — Three.js update, synchronized blit, debug
overlay and the HUD itself; on Android Chrome the engine ran in 22–31 ms while
vision reached only 10–14 fps — the `grab` entry was 31–41 ms: the 2D-canvas
readback of the camera frame). The frame grabber therefore has a WebGL path
(`WebGLFrameGrabber`: texture upload, luma shader packing four gray pixels per
texel, `readPixels` straight into the pooled buffer) and `processing.grabber`
defaults to `auto`: the 2D path is measured on the first frames and WebGL
takes over when its median grab exceeds `grabAutoSwitchMs`; `?grab=2d|gl`
forces one, HUD `Proc … grab 2d|gl`. Two engine-side savings for slower
devices: the relocalization preparation (keyframe ranking while tracking is
weak) is gated by its own thresholds (`prepareMinInliers`, `prepareMaxErrorPx`)
rather than the `trusted` label — Android tracks healthily at 2.6–3.3 px and
never reaches the 1.5 px label, so the ranking ran all session — and once the
world exists the two-view RANSAC is capped at `pose.maxIterationsWhileMapped`
iterations (the map PnP is the canonical pose then; the two-view result is a
rotation prior). Before a world exists a plane search that keeps
failing the 2D-extent test for `sidewaysGuidanceDelayMs` asks the user to
move the phone sideways (`MOVE_SIDEWAYS`) instead of "show a flat surface".

Keyframe coverage (v15): a new keyframe must be a new *view* with respect
to every stored keyframe (rotation above `keyframeRotationDeg` or
camera-center translation above `keyframeTranslationDepthRatio` × scene
depth), not only the last one, and when the store is full the most
redundant keyframe is evicted — never the first (origin view) or the
newest. A camera swinging between two views keeps two keyframes instead of
churning through the 8 slots. HUD `Reloc … kf 8 (N made, M out)`. The
landmarks a stored keyframe observes are exempt from the map's age prune
(they still count toward `maxLandmarks`), so a keyframe stays usable after
the view has been out of sight for more than `maxLandmarkAgeFrames`; a
keyframe with fewer live landmarks than `minInliers` is ranked but never
spends an LK slot and is evicted first (`Rank KF17 0.91×26`). Keyframe → current LK has its own forward-backward / residual
bounds (`lkForwardBackwardPx` 2.0, `lkMaxResidual`); the frame-to-frame
tracker is unchanged. A keyframe that failed is not retried on an
unchanged view for `retryCooldownFrames` (image change = zero-shift NCC
vs the view it failed on below `retryImageChangeScore`); the lost pose
never picks a keyframe. While tracking is weak (PnP below the trusted
quality or fast motion) the ranking is kept warm (`prepare`) so the first
lost frame starts from it. Diagnostics: per-keyframe LK status counts and
a failure reason (`high_fb_error | high_lk_error | too_far | out_of_bounds
| low_texture | insufficient_tracks`), the ranking, the search stage
(`prepare | coarse | lk | pnp | validation | confirming | applied`), the
pending candidate's age (`candidateMaxAgeFrames`) and an episode timeline
(ms since the loss of the first coarse match / LK / PnP / validation /
applied pose). `src/vision/RelocalizationSchedule.ts` holds the pure
scheduling functions.

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
             PlaneRecovery (plane re-seed after fast motion), PlaneTracker (experimental plane-relative pose),
             Keyframe, Relocalizer (keyframe ranking + coarse NCC shift + LK + PnP),
             RelocalizationSchedule (prepare / attempt / retry / timeline), VisionEngine, TrackingQuality, types
  math/      Matrix (3×3, linear solve), Homography (normalized DLT), Decomposition (Jacobi eigen, SVD),
             Pose (rotations, quaternions), EssentialMatrix (8-point, RANSAC, recoverPose),
             HomographyDecomposition (Faugeras), Triangulation, Plane (RANSAC), PnP (LM + Huber),
             BundleAdjustment (LM + Schur complement over keyframes and landmarks),
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
