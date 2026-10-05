import { describe, expect, it } from "vitest";
import { DEFAULT_CONFIG, resolveConfig } from "../../src/ar/ARConfig";
import { TrackingState } from "../../src/ar/ARState";
import { approximateIntrinsics } from "../../src/camera/CameraIntrinsics";
import { type Mat3, mat3Invert, mat3Multiply } from "../../src/math/Matrix";
import { LandmarkMap } from "../../src/vision/LandmarkMap";
import { createRng } from "../../src/vision/OutlierRejection";
import { PlaneDetector } from "../../src/vision/PlaneDetector";
import { PlaneRecovery, significantTwoViewMotion, type PlaneRecoveryObservation, type TwoViewMotionThresholds } from "../../src/vision/PlaneRecovery";
import type { VisionInput, VisionOutput } from "../../src/vision/types";
import { VisionEngine } from "../../src/vision/VisionEngine";
import { gauss, randomBoxPoints, randomPlanePoints } from "../helpers/scene";
import { makeTexture, warpImage } from "../helpers/synthetic";

/**
 * 修正指示書 v11 / v11.1: after a significant motion with a healthy map only
 * the plane detector is recovered (reset once, re-seeded from the landmarks
 * in view); the map, the camera pose and the world are untouched,
 * relocalization is never entered, and the plane is found again where the
 * camera looks now.
 */
const obs = (o: Partial<PlaneRecoveryObservation> = {}): PlaneRecoveryObservation => ({
  motionLevel: "fast",
  twoViewMotion: false,
  mapInitialized: true,
  mapTracked: true,
  mapLost: false,
  mapInliers: 42,
  requiredInliers: DEFAULT_CONFIG.landmarks.minPnPInliers,
  poseFinite: true,
  worldEstablished: false,
  timestamp: 1000,
  ...o,
});

const TWO_VIEW: TwoViewMotionThresholds = {
  parallaxPx: DEFAULT_CONFIG.pose.fullConfidenceParallaxPx,
  minConfidence: DEFAULT_CONFIG.landmarks.initMinTranslationConfidence,
  minInliers: DEFAULT_CONFIG.pose.minCorrespondences,
};
const twoView = (o: Partial<Parameters<typeof significantTwoViewMotion>[0] & object> = {}) => ({
  parallaxPx: 45,
  confidence: 1,
  translationConfidence: 1,
  inlierCount: 116,
  referenceFrameId: 7,
  model: "homography" as const,
  ...o,
});

describe("PlaneRecovery (controller, v11 §5–§6, v11.1 §3–§10)", () => {
  it("Test 1: fast motion + healthy map (no world) starts a recovery; states follow the motion and the candidate", () => {
    const rec = new PlaneRecovery();
    expect(rec.active).toBe(false);
    expect(rec.state(false, "normal", false, 0)).toBe("inactive");
    // The on-device log: 42 inliers / 0.38 px / source MAP / lost 0 ms, motion fast.
    expect(rec.update(obs())).toBe(true);
    expect(rec.active).toBe(true);
    expect(rec.reason).toBe("fast_motion");
    expect(rec.count).toBe(1);
    expect(rec.state(true, "fast", false, 0)).toBe("starting");
    expect(rec.state(false, "fast", false, 0)).toBe("starting");
    expect(rec.state(false, "medium", false, 0)).toBe("warmup");
    expect(rec.state(false, "normal", false, 0)).toBe("warmup");
    expect(rec.state(false, "normal", true, 0)).toBe("candidate");
    expect(rec.state(false, "normal", true, 2)).toBe("stable");
    expect(rec.elapsedMs(1500)).toBe(500);
    // Plane found → done.
    rec.finish();
    expect(rec.active).toBe(false);
    expect(rec.state(false, "normal", true, 3)).toBe("inactive");
    expect(rec.elapsedMs(2000)).toBe(0);
    // Manual request works the same way under the same conditions.
    expect(rec.update(obs({ motionLevel: "normal", manual: true }))).toBe(true);
    expect(rec.reason).toBe("manual");
    expect(rec.count).toBe(2);
  });

  it("Test 2 / 3 / 4: a confident two-view parallax crossing starts a recovery at motion MED; parallax alone or a weak estimate does not", () => {
    // Test 2: Motion MED + 2view par 45px conf 1.00/1.00 n116 (the on-device log).
    expect(significantTwoViewMotion(twoView(), null, TWO_VIEW)).toBe(true);
    const rec = new PlaneRecovery();
    expect(rec.update(obs({ motionLevel: "medium", twoViewMotion: true }))).toBe(true);
    expect(rec.reason).toBe("two_view_motion");
    // Test 3: small parallax.
    expect(significantTwoViewMotion(twoView({ parallaxPx: TWO_VIEW.parallaxPx - 1 }), null, TWO_VIEW)).toBe(false);
    // Test 4 / §8: large parallax but low confidence or few inliers never triggers.
    expect(significantTwoViewMotion(twoView({ confidence: TWO_VIEW.minConfidence - 0.01 }), null, TWO_VIEW)).toBe(false);
    expect(significantTwoViewMotion(twoView({ translationConfidence: TWO_VIEW.minConfidence - 0.01 }), null, TWO_VIEW)).toBe(false);
    expect(significantTwoViewMotion(twoView({ inlierCount: TWO_VIEW.minInliers - 1 }), null, TWO_VIEW)).toBe(false);
    expect(significantTwoViewMotion(twoView({ model: "rotation" }), null, TWO_VIEW)).toBe(false);
    expect(significantTwoViewMotion(null, null, TWO_VIEW)).toBe(false);
    // An event, not a level: once the parallax is above the threshold it does not keep triggering …
    expect(significantTwoViewMotion(twoView({ parallaxPx: 50 }), twoView({ parallaxPx: 45 }), TWO_VIEW)).toBe(false);
    // … but a renewed reference (parallax restarts) can cross it again.
    expect(significantTwoViewMotion(twoView({ parallaxPx: 30, referenceFrameId: 20 }), twoView({ parallaxPx: 45, referenceFrameId: 7 }), TWO_VIEW)).toBe(true);
    const rec2 = new PlaneRecovery();
    expect(rec2.update(obs({ motionLevel: "medium", twoViewMotion: false }))).toBe(false);
    expect(rec2.active).toBe(false);
  });

  it("no healthy map, or an established world → no recovery", () => {
    const rec = new PlaneRecovery();
    expect(rec.update(obs({ mapTracked: false }))).toBe(false); // camera not located this frame
    expect(rec.update(obs({ mapInliers: DEFAULT_CONFIG.landmarks.minPnPInliers - 1 }))).toBe(false);
    expect(rec.update(obs({ poseFinite: false }))).toBe(false);
    expect(rec.update(obs({ mapInitialized: false, mapTracked: false }))).toBe(false);
    expect(rec.update(obs({ motionLevel: "normal" }))).toBe(false); // no trigger
    expect(rec.update(obs({ motionLevel: "medium" }))).toBe(false);
    expect(rec.update(obs({ worldEstablished: true }))).toBe(false); // v11 §30: the found plane is the reference
    expect(rec.active).toBe(false);
    expect(rec.count).toBe(0);
    expect(PlaneRecovery.mapHealthy(obs())).toBe(true);
    expect(PlaneRecovery.mapHealthy(obs({ mapTracked: false }))).toBe(false);
    // An active pre-world recovery ends as soon as the world is established.
    rec.update(obs());
    expect(rec.active).toBe(true);
    rec.update(obs({ worldEstablished: true, motionLevel: "normal" }));
    expect(rec.active).toBe(false);
  });

  it("Test 5 / 6 / 7 / 8 / 9: the start is reported once; further fast / two-view motion, medium and normal frames continue the recovery without a restart", () => {
    const rec = new PlaneRecovery();
    const starts: boolean[] = [];
    // Frames 100–102 fast, 103 medium, 104 normal, 105 two-view event.
    starts.push(rec.update(obs({ timestamp: 100 })));
    starts.push(rec.update(obs({ timestamp: 101 })));
    starts.push(rec.update(obs({ timestamp: 102 })));
    starts.push(rec.update(obs({ motionLevel: "medium", timestamp: 103 })));
    starts.push(rec.update(obs({ motionLevel: "normal", timestamp: 104 })));
    starts.push(rec.update(obs({ motionLevel: "normal", twoViewMotion: true, timestamp: 105 })));
    expect(starts).toEqual([true, false, false, false, false, false]);
    expect(rec.active).toBe(true);
    expect(rec.count).toBe(1); // one start = one detector reset
    expect(rec.reason).toBe("fast_motion"); // the reason is the one that started it
    expect(rec.lastMotionTimestamp).toBe(105); // later motion is only noted
    expect(rec.elapsedMs(105)).toBe(5);
    // A real loss hands over to the map / state machine (§32).
    expect(rec.update(obs({ motionLevel: "normal", mapTracked: false, mapLost: true, timestamp: 106 }))).toBe(false);
    expect(rec.active).toBe(false);
  });
});

describe("plane seeds from the map (v11 §9–§10, Tests 5–6)", () => {
  it("Test 5: a valid, recently observed, mature landmark is a plane seed", () => {
    const map = new LandmarkMap();
    const a = map.add([1, 2, 3], 1, 0);
    a.observations = 3;
    a.lastSeenFrame = 100;
    const { ids, points } = map.collect(3, 15, 100);
    expect(ids).toEqual([a.id]);
    expect(Array.from(points)).toEqual([1, 2, 3]);
    expect(map.countRecent(100, 15)).toBe(1);
  });

  it("Test 6: stale (not in view), young and non-finite landmarks are not plane seeds", () => {
    const map = new LandmarkMap();
    const fresh = map.add([1, 2, 3], 1, 0);
    fresh.observations = 3;
    fresh.lastSeenFrame = 100;
    const stale = map.add([1, 2, 4], 2, 0); // the view before the motion
    stale.observations = 8;
    stale.lastSeenFrame = 60;
    const young = map.add([1, 2, 5], 3, 0); // triangulated, never confirmed by PnP
    young.observations = 2;
    young.lastSeenFrame = 100;
    const broken = map.add([Number.NaN, 2, 6], 4, 0);
    broken.observations = 5;
    broken.lastSeenFrame = 100;
    // Recovery seeds: in view within the window, mature, finite.
    expect(map.collect(3, 15, 100).ids).toEqual([fresh.id]);
    // Normal (no window): the stale landmark still votes, the young / broken ones never do.
    expect(map.collect(3).ids.sort()).toEqual([fresh.id, stale.id].sort());
    expect(map.countRecent(100, 15)).toBe(3);
  });
});

describe("PlaneDetector recovery and stage diagnostics (v11 §8, §22, Tests 7–8)", () => {
  const gravityFloor = new Float64Array([0, Math.SQRT1_2, Math.SQRT1_2]);
  function floorScene(rng: () => number, n = 150, clutter = 40) {
    const plane = randomPlanePoints(rng, n, [0, Math.SQRT1_2, Math.SQRT1_2], 1.2, 0.8);
    for (let i = 0; i < plane.length; i++) plane[i] += gauss(rng) * 0.004;
    const box = randomBoxPoints(rng, clutter, 0.4, 1.0, 0.6);
    const pts = new Float64Array(plane.length + box.length);
    pts.set(plane);
    pts.set(box, plane.length);
    return { pts, ids: Array.from({ length: n + clutter }, (_, i) => i + 1), count: n + clutter };
  }

  it("v11.1 Test 13–16: a held earlier candidate is not a current-frame candidate", () => {
    const rng = createRng(31);
    const det = new PlaneDetector(DEFAULT_CONFIG.plane, createRng(32));
    const { pts, ids, count } = floorScene(rng);
    for (let f = 0; f < 8; f++) det.update(pts, ids, count, gravityFloor);
    expect(det.isFound).toBe(true);
    // Test 14 / 15: this frame's search produced a horizontal candidate.
    expect(det.lastSearch.stage).toBe("candidate");
    expect(det.current!.horizontal).toBe(true);
    // Test 13 / 16: no points this frame → the search fails ("points") while
    // the detector still holds the previous plane through its grace period.
    const held = det.update(new Float64Array(0), [], 0, gravityFloor);
    expect(held).not.toBeNull();
    expect(held!.found).toBe(true);
    expect(det.lastSearch.stage).toBe("points");
    expect(det.lastSearch.stage === "candidate").toBe(false); // candidateFound of this frame
    expect(det.stableFrameCount).toBe(0);
  });

  it("Test 7 / 8: resetForRecovery forgets the stability streak; one candidate frame is not FOUND, the required streak is", () => {
    const rng = createRng(11);
    const det = new PlaneDetector(DEFAULT_CONFIG.plane, createRng(12));
    const { pts, ids, count } = floorScene(rng);
    for (let f = 0; f < 3; f++) det.update(pts, ids, count, gravityFloor);
    expect(det.stableFrameCount).toBe(2);
    expect(det.current).not.toBeNull();
    det.resetForRecovery();
    expect(det.stableFrameCount).toBe(0);
    expect(det.current).toBeNull();
    expect(det.isFound).toBe(false);
    // Test 7: a candidate in the very next frame does not establish anything.
    const first = det.update(pts, ids, count, gravityFloor)!;
    expect(first).not.toBeNull();
    expect(first.found).toBe(false);
    expect(first.stableFrames).toBe(0);
    expect(det.lastSearch.stage).toBe("candidate");
    // Test 8: the existing stability rule decides — unchanged threshold.
    let last = first;
    for (let f = 0; f < DEFAULT_CONFIG.plane.stableFramesRequired; f++) last = det.update(pts, ids, count, gravityFloor)!;
    expect(last.found).toBe(true);
    expect(last.stableFrames).toBe(DEFAULT_CONFIG.plane.stableFramesRequired);
  });

  it("stage diagnostics name where the search stopped: points / extent / support, candidate on a floor", () => {
    const det = new PlaneDetector({ ...DEFAULT_CONFIG.plane, minInliers: 10 }, createRng(21));
    // Too few landmarks.
    det.update(new Float64Array(9), [1, 2, 3], 3, gravityFloor);
    expect(det.lastSearch.stage).toBe("points");
    // A horizontal strip on a wall: enough height support, no 2D extent (the
    // on-device "best 32/20 but no plane" shape).
    const pts = new Float64Array(40 * 3);
    const rng = createRng(22);
    for (let i = 0; i < 40; i++) {
      pts[i * 3] = (rng() - 0.5) * 1.6;
      pts[i * 3 + 1] = 0.1 + gauss(rng) * 0.002;
      pts[i * 3 + 2] = 1.5 + gauss(rng) * 0.002;
    }
    const ids = Array.from({ length: 40 }, (_, i) => i + 1);
    expect(det.update(pts, ids, 40, new Float64Array([0, 1, 0]))).toBeNull();
    const s = det.lastSearch;
    expect(s.stage).toBe("extent");
    expect(s.bestInliers).toBeGreaterThanOrEqual(10);
    expect(s.inliers).toBeGreaterThanOrEqual(10);
    expect(s.extentMinor).toBeLessThan(s.extentRequired);
    // A wall under an upright phone: no dense height window at all.
    const wall = randomPlanePoints(createRng(23), 150, [0, 0, 1], 1.5, 0.8);
    const wallIds = Array.from({ length: 150 }, (_, i) => i + 1);
    const det2 = new PlaneDetector(DEFAULT_CONFIG.plane, createRng(24));
    det2.update(wall, wallIds, 150, new Float64Array([0, 1, 0]));
    expect(det2.lastSearch.stage).not.toBe("candidate");
    console.log(`[plane] wall stage ${det2.lastSearch.stage} best ${det2.lastSearch.bestInliers} inliers ${det2.lastSearch.inliers}`);
    // A floor: candidate.
    const det3 = new PlaneDetector(DEFAULT_CONFIG.plane, createRng(25));
    const floor = floorScene(createRng(26));
    expect(det3.update(floor.pts, floor.ids, floor.count, gravityFloor)).not.toBeNull();
    expect(det3.lastSearch.stage).toBe("candidate");
    expect(det3.lastSearch.extentMinor).toBeGreaterThanOrEqual(det3.lastSearch.extentRequired);
  });
});

// ---- Engine level: fast motion over a desk before the plane is found ----
const W = 640;
const H = 480;
const K = approximateIntrinsics(W, H);
const Kmat = new Float64Array([K.fx, 0, K.cx, 0, K.fy, K.cy, 0, 0, 1]);
const KInv = mat3Invert(Kmat)!;
const I: Mat3 = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
const base = makeTexture(W, H, createRng(2025), [12, 30, 70, 160]);
const DESK_N = [0, Math.SQRT1_2, Math.SQRT1_2];
const GRAVITY = [0, Math.SQRT1_2, Math.SQRT1_2];

function pixelHomography(r: Mat3, t: ArrayLike<number>, n: ArrayLike<number>, d: number): Mat3 {
  const h = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) h[i * 3 + j] = r[i * 3 + j] + (t[i] * n[j]) / d;
  return mat3Multiply(mat3Multiply(Kmat, h), KInv);
}

/** Camera translating sideways over the desk with per-frame steps `steps[f]` (map-independent units, plane at d = 1). */
function runDesk(steps: number[], seed: number): VisionOutput[] {
  const engine = new VisionEngine(W, H, resolveConfig(), createRng(seed));
  const outs: VisionOutput[] = [];
  let x = 0;
  for (let f = 0; f < steps.length; f++) {
    x += steps[f];
    const t = [-x, -0.3 * x, 0];
    const img = f === 0 ? base : warpImage(base, W, H, pixelHomography(I, t, DESK_N, 1), 128);
    const input: VisionInput = { frameId: f, timestamp: f * 33.3, width: W, height: H, gray: img, intrinsics: K, gravity: GRAVITY };
    outs.push(engine.process(input));
  }
  return outs;
}

describe("VisionEngine plane recovery after fast motion (v11 §47–§48, Tests 3–4, AC-1/2/10/12)", () => {
  it("fast motion before the plane is found: map kept, pose from the map, recovery runs, plane found at the new view", () => {
    // Slow scan until the map exists, then three fast frames, then slow again.
    const slow = 0.004;
    const probe = runDesk(Array(30).fill(slow), 1);
    const initIdx = probe.findIndex((o) => o.mapPose !== null);
    const foundIdx = probe.findIndex((o) => o.state === TrackingState.PLANE_FOUND);
    console.log(`[v11] slow-only: map init at ${initIdx}, PLANE_FOUND at ${foundIdx}`);
    expect(initIdx).toBeGreaterThan(0);
    const fastAt = initIdx + 2;
    const steps = [...Array(fastAt).fill(slow), 0.06, 0.06, 0.06, ...Array(40).fill(slow)];
    const outs = runDesk(steps, 1);
    const init2 = outs.findIndex((o) => o.mapPose !== null);
    expect(init2).toBe(initIdx);
    const fastFrames = outs.map((o, i) => (o.motion.level === "fast" ? i : -1)).filter((i) => i >= 0);
    console.log(
      `[v11] fast frames ${fastFrames.join(",")}  motion ${outs
        .slice(fastAt, fastAt + 5)
        .map((o) => `${o.motion.level}:${o.motion.medianDisplacementPx.toFixed(0)}px`)
        .join(" ")}`,
    );
    expect(fastFrames.length).toBeGreaterThan(0);
    const recoveryFrames = outs.map((o, i) => (o.planeRecovery.active ? i : -1)).filter((i) => i >= 0);
    console.log(
      `[v11] recovery frames ${recoveryFrames[0]}..${recoveryFrames[recoveryFrames.length - 1]} (${recoveryFrames.length})  ` +
        outs
          .filter((o) => o.planeRecovery.active)
          .slice(0, 12)
          .map((o) => `${o.planeRecovery.state}/${o.planeRecovery.searchStage}/${o.planeRecovery.seededPoints}pt/${o.planeRecovery.stableFrames}`)
          .join(" "),
    );
    // Test 1 at engine level: the recovery started because of the fast motion with the map tracked.
    expect(recoveryFrames.length).toBeGreaterThan(0);
    const start = outs[recoveryFrames[0]];
    expect(start.planeRecovery.reason).toBe("fast_motion");
    expect(start.planeRecovery.mapHealthy).toBe(true);
    expect(start.planeRecovery.state).toBe("starting");
    expect(start.mapPose!.framesSinceTracked).toBe(0);
    // v11.1 Test 5–7 / AC-4–AC-6: three consecutive fast frames → ONE recovery
    // (one detector reset); the stability streak then climbs monotonically.
    for (const f of recoveryFrames) expect(outs[f].planeRecovery.recoveries, `frame ${f} recoveries`).toBe(1);
    const streak = recoveryFrames.map((f) => outs[f].planeRecovery.stableFrames);
    for (let i = 1; i < streak.length; i++) {
      if (outs[recoveryFrames[i]].planeRecovery.candidateFound && outs[recoveryFrames[i - 1]].planeRecovery.candidateFound) {
        expect(streak[i], `stable streak at frame ${recoveryFrames[i]}`).toBe(streak[i - 1] + 1);
      }
    }
    // Current-frame candidate semantics (v11.1 §21–§24).
    for (const o of outs) {
      expect(o.planeRecovery.candidateFound).toBe(o.planeSearch?.stage === "candidate" && o.plane !== null);
      if (o.planeRecovery.candidateCommitted) expect(o.planeRecovery.candidateFound).toBe(true);
      if (o.planeRecovery.previousCandidateHeld) expect(o.planeRecovery.candidateFound).toBe(false);
    }
    // Test 3 / AC-1: the map was never reset — one map id from initialization to the end.
    for (let f = init2; f < outs.length; f++) expect(outs[f].mapPose!.mapFrameId, `frame ${f} map id`).toBe(outs[init2].mapPose!.mapFrameId);
    // AC-10 / §29: never RELOCALIZING, never lost.
    for (const o of outs) expect(o.state).not.toBe(TrackingState.RELOCALIZING);
    for (let f = init2; f < outs.length; f++) expect(outs[f].mapPose!.framesSinceTracked, `frame ${f} lost`).toBe(0);
    // Test 4 / §43: the camera pose comes from the map PnP; no plane pose exists to overwrite it.
    for (let f = init2 + 1; f < outs.length; f++) {
      expect(outs[f].mapPose!.source, `frame ${f} source`).toBe("map");
      expect(outs[f].planePose).toBeNull();
      expect(outs[f].planeAnchor).toBeNull();
    }
    // AC-12 / Test 8: plane detection resumes at the new view and the world is established.
    const found2 = outs.findIndex((o) => o.state === TrackingState.PLANE_FOUND);
    console.log(`[v11] with fast motion: PLANE_FOUND at ${found2}, recoveries ${outs[outs.length - 1].planeRecovery.recoveries}`);
    expect(found2).toBeGreaterThan(recoveryFrames[0]);
    const last = outs[outs.length - 1];
    expect(last.worldEstablished).toBe(true);
    expect(last.planeRecovery.active).toBe(false);
    expect(last.planeRecovery.state).toBe("inactive");
    expect(last.planeRecovery.recoveries).toBe(1);
    // The recovery was still active right before the plane was found and reported the candidate phase.
    const before = outs[found2 - 1].planeRecovery;
    expect(before.active).toBe(true);
    expect(before.candidateFound).toBe(true);
    expect(before.candidateCommitted).toBe(true);
    expect(before.requiredStableFrames).toBe(DEFAULT_CONFIG.plane.stableFramesRequired);
  });

  it("slow motion only: no recovery is ever started (no regression, AC-5/§20); the two-view trigger is an event, not a level", () => {
    const outs = runDesk(Array(45).fill(0.004), 1);
    for (const o of outs) expect(o.planeRecovery.active).toBe(false);
    expect(outs[outs.length - 1].planeRecovery.recoveries).toBe(0);
    expect(outs[outs.length - 1].state).toBe(TrackingState.PLANE_FOUND);
    // The slow scan does reach large two-view parallax with a confident
    // estimate — exactly what a level test would have fired on.
    expect(outs.some((o) => o.pose && o.pose.parallaxPx >= DEFAULT_CONFIG.pose.fullConfidenceParallaxPx && o.pose.translationConfidence >= 0.5)).toBe(true);
    // Diagnostics are produced regardless (numbers only).
    const mid = outs.find((o) => o.mapPose !== null && o.plane === null);
    if (mid) expect(["points", "support", "reclassify", "extent"]).toContain(mid.planeRecovery.searchStage);
  });
});
