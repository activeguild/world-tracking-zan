import { describe, expect, it } from "vitest";
import { DEFAULT_CONFIG } from "../../src/ar/ARConfig";
import { angleBetween, rotationAxisAngle, rotationDistance, type RigidTransform } from "../../src/math/Pose";
import { MapTracker } from "../../src/vision/MapTracker";
import { ransacHomography } from "../../src/vision/OutlierRejection";
import { PoseEstimator } from "../../src/vision/PoseEstimator";
import type { Track } from "../../src/vision/types";
import { gauss, poseFromCenter, randomBoxPoints, TEST_K } from "../helpers/scene";
import { createRng } from "../helpers/synthetic";

const deg = (rad: number) => (rad * 180) / Math.PI;

/**
 * Simulate tracks of a static 3D scene seen from a moving camera: camera
 * centers along a path; tracks carry pixel positions per frame.
 */
function makeScene(seed: number, n = 200) {
  const rng = createRng(seed);
  const points = randomBoxPoints(rng, n, 1.5, 4, 1.5);
  return { rng, points };
}

function cameraAt(f: number): RigidTransform {
  // Lateral + slight forward motion with a little yaw; frame 0 is identity.
  const R = rotationAxisAngle([0, 1, 0], 0.004 * f);
  return poseFromCenter(R, [0.03 * f, 0.005 * f, 0.01 * f]);
}

function project(points: Float64Array, pose: RigidTransform, i: number, noise: number, rng: () => number): [number, number] | null {
  const X = points[i * 3], Y = points[i * 3 + 1], Z = points[i * 3 + 2];
  const r = pose.rotation, t = pose.translation;
  const z = r[6] * X + r[7] * Y + r[8] * Z + t[2];
  if (z <= 0.1) return null;
  const u = ((r[0] * X + r[1] * Y + r[2] * Z + t[0]) / z) * TEST_K.fx + TEST_K.cx + gauss(rng) * noise;
  const v = ((r[3] * X + r[4] * Y + r[5] * Z + t[1]) / z) * TEST_K.fy + TEST_K.cy + gauss(rng) * noise;
  if (u < 0 || u >= TEST_K.width || v < 0 || v >= TEST_K.height) return null;
  return [u, v];
}

function makeTracks(points: Float64Array, pose: RigidTransform, refPose: RigidTransform, refFrame: number, rng: () => number): Track[] {
  const n = points.length / 3;
  const tracks: Track[] = [];
  for (let i = 0; i < n; i++) {
    const cur = project(points, pose, i, 0.3, rng);
    const ref = project(points, refPose, i, 0.3, rng);
    if (!cur || !ref) continue;
    tracks.push({
      id: i + 1, x: cur[0], y: cur[1], prevX: cur[0], prevY: cur[1], age: 5, score: 10, inlier: true, outlierStreak: 0,
      refX: ref[0], refY: ref[1], refFrame,
      landmarkId: -1, anchorFrame: -1, anchorX: cur[0], anchorY: cur[1], anchorPose: null,
      planePoint: null, planeStreak: 0, planeOutliers: 0, offPlane: false,
    });
  }
  return tracks;
}

describe("MapTracker", () => {
  it("initializes from a two-view pose, then tracks the camera with consistent scale", () => {
    const { rng, points } = makeScene(1);
    const cfg = DEFAULT_CONFIG.landmarks;
    const tracker = new MapTracker(cfg);
    const estimator = new PoseEstimator(DEFAULT_CONFIG.pose, DEFAULT_CONFIG.ransac, createRng(2));
    const refPose = cameraAt(0);

    // Frame 8: enough baseline to initialize.
    const f0 = 8;
    let tracks = makeTracks(points, cameraAt(f0), refPose, 0, rng);
    const n = tracks.length;
    const x1 = Float64Array.from(tracks, (t) => t.refX), y1 = Float64Array.from(tracks, (t) => t.refY);
    const x2 = Float64Array.from(tracks, (t) => t.x), y2 = Float64Array.from(tracks, (t) => t.y);
    const rel = estimator.estimate(x1, y1, x2, y2, n, TEST_K);
    expect(rel.model).toBe("essential");
    expect(tracker.tryInitialize(tracks, rel, 0, f0, TEST_K)).toBe(true);
    expect(tracker.initialized).toBe(true);
    expect(tracker.map.size).toBeGreaterThan(cfg.initMinLandmarks);

    // The map scale is |t| = 1 at frame f0; the true baseline is |C(f0)|.
    const trueBaseline = Math.hypot(0.03 * f0, 0.005 * f0, 0.01 * f0);
    const scale = 1 / trueBaseline;

    // Subsequent frames: keep the same track ids (landmark links), update positions.
    const byId = new Map(tracks.map((t) => [t.id, t]));
    for (let f = f0 + 1; f <= f0 + 20; f++) {
      const pose = cameraAt(f);
      const next: Track[] = [];
      for (let i = 0; i < points.length / 3; i++) {
        const t = byId.get(i + 1);
        if (!t) continue;
        const p = project(points, pose, i, 0.3, rng);
        if (!p) continue;
        t.prevX = t.x; t.prevY = t.y; t.x = p[0]; t.y = p[1]; t.age++;
        next.push(t);
      }
      tracks = next;
      const res = tracker.update(tracks, f, TEST_K, null);
      expect(res.tracked, `frame ${f}`).toBe(true);
      expect(res.inlierCount).toBeGreaterThan(cfg.minPnPInliers);
      // 0.3 px Gaussian noise on every observation (reference and current) →
      // triangulated landmarks carry ~1 px of reprojection error.
      expect(res.meanReprojectionErrorPx).toBeLessThan(1.5);

      // Pose check: rotation matches, translation matches up to the map scale.
      expect(deg(rotationDistance(tracker.pose.rotation, pose.rotation)), `frame ${f} R`).toBeLessThan(0.3);
      const tt = tracker.pose.translation;
      const truth = pose.translation.map((v) => v * scale);
      expect(deg(angleBetween(tt, truth)), `frame ${f} t dir`).toBeLessThan(2);
      const mag = Math.hypot(tt[0], tt[1], tt[2]) / Math.hypot(truth[0], truth[1], truth[2]);
      expect(mag, `frame ${f} scale`).toBeCloseTo(1, 1);
    }
  });

  it("jump gate (v2 §8): rejects an implausible pose from a weak solve, accepts the same motion when well supported", () => {
    const cfg = DEFAULT_CONFIG.landmarks;
    const f0 = 8;
    // The camera suddenly appears 0.4 units (≈ 90 px of image shift) to the side.
    const jumped = (f: number): RigidTransform => poseFromCenter(rotationAxisAngle([0, 1, 0], 0.004 * f), [0.03 * f + 0.4, 0.005 * f, 0.01 * f]);

    const setup = (seed: number) => {
      const { rng, points } = makeScene(seed);
      const tracker = new MapTracker(cfg);
      const estimator = new PoseEstimator(DEFAULT_CONFIG.pose, DEFAULT_CONFIG.ransac, createRng(seed + 1));
      const tracks = makeTracks(points, cameraAt(f0), cameraAt(0), 0, rng);
      const rel = estimator.estimate(
        Float64Array.from(tracks, (t) => t.refX), Float64Array.from(tracks, (t) => t.refY),
        Float64Array.from(tracks, (t) => t.x), Float64Array.from(tracks, (t) => t.y),
        tracks.length, TEST_K,
      );
      expect(tracker.tryInitialize(tracks, rel, 0, f0, TEST_K)).toBe(true);
      const byId = new Map(tracks.map((t) => [t.id, t]));
      const step = (ids: Iterable<number>, pose: RigidTransform): Track[] => {
        const next: Track[] = [];
        for (const id of ids) {
          const t = byId.get(id);
          if (!t) continue;
          const p = project(points, pose, id - 1, 0.3, rng);
          if (!p) continue;
          t.prevX = t.x; t.prevY = t.y; t.x = p[0]; t.y = p[1];
          next.push(t);
        }
        return next;
      };
      // One normal frame so the tracker has a previous delta.
      const normal = step(byId.keys(), cameraAt(f0 + 1));
      const res = tracker.update(normal, f0 + 1, TEST_K, null);
      expect(res.tracked).toBe(true);
      expect(res.jumpRejected).toBe(false);
      return { tracker, byId, step, landmarkIds: normal.filter((t) => t.landmarkId >= 0).map((t) => t.id) };
    };

    // Weak solve: only 15 landmark tracks see the jumped camera → PnP finds
    // the jump with few inliers → rejected, pose held.
    const a = setup(21);
    expect(a.landmarkIds.length).toBeGreaterThan(60);
    const centerBefore = Array.from(a.tracker.cameraCenter());
    const weak = a.step(a.landmarkIds.slice(0, 15), jumped(f0 + 2));
    const resWeak = a.tracker.update(weak, f0 + 2, TEST_K, null);
    expect(resWeak.jumpRejected).toBe(true);
    expect(resWeak.tracked).toBe(false);
    // The rejected 0.4-unit jump (≈ 1.5 map units) is not applied; the camera
    // center only advances by the last tracked velocity (constant-velocity
    // prediction while lost).
    expect(resWeak.translationPredicted).toBe(true);
    const after = Array.from(a.tracker.cameraCenter());
    expect(Math.hypot(after[0] - centerBefore[0], after[1] - centerBefore[1], after[2] - centerBefore[2])).toBeLessThan(0.3);

    // Trusted solve: every landmark track sees the jumped camera (many
    // inliers, small error) → accepted as genuine fast motion.
    const b = setup(31);
    const all = b.step(b.landmarkIds, jumped(f0 + 2));
    const resAll = b.tracker.update(all, f0 + 2, TEST_K, null);
    expect(resAll.tracked).toBe(true);
    expect(resAll.jumpRejected).toBe(false);
    expect(resAll.inlierCount).toBeGreaterThanOrEqual(cfg.jumpRejectTrustedInliers);
    expect(deg(rotationDistance(b.tracker.pose.rotation, jumped(f0 + 2).rotation))).toBeLessThan(0.5);
  });

  it("plane candidate (v3 §3–§7): validated like the map candidate, compared with it, adopted only after the cooldown", () => {
    const cfg = DEFAULT_CONFIG.landmarks;
    const f0 = 8;
    const { rng, points } = makeScene(41);
    const tracker = new MapTracker(cfg);
    const estimator = new PoseEstimator(DEFAULT_CONFIG.pose, DEFAULT_CONFIG.ransac, createRng(42));
    const tracks = makeTracks(points, cameraAt(f0), cameraAt(0), 0, rng);
    const rel = estimator.estimate(
      Float64Array.from(tracks, (t) => t.refX), Float64Array.from(tracks, (t) => t.refY),
      Float64Array.from(tracks, (t) => t.x), Float64Array.from(tracks, (t) => t.y),
      tracks.length, TEST_K,
    );
    expect(tracker.tryInitialize(tracks, rel, 0, f0, TEST_K)).toBe(true);
    const byId = new Map(tracks.map((t) => [t.id, t]));
    const step = (f: number): Track[] => {
      const next: Track[] = [];
      for (const t of byId.values()) {
        const p = project(points, cameraAt(f), t.id - 1, 0.3, rng);
        if (!p) continue;
        t.prevX = t.x; t.prevY = t.y; t.x = p[0]; t.y = p[1];
        next.push(t);
      }
      return next;
    };
    // The map scale is |t| = 1 at f0 (true baseline |C(f0)|): express the
    // "plane" candidate in map units so it agrees with the map candidate.
    const scale = 1 / Math.hypot(0.03 * f0, 0.005 * f0, 0.01 * f0);
    const inMapUnits = (p: RigidTransform): RigidTransform => ({
      rotation: p.rotation,
      translation: Float64Array.from(p.translation, (v) => v * scale),
    });

    // 1) A plane candidate that agrees with the map: not adopted before the
    //    cooldown (hysteresis), adopted afterwards.
    let adoptedAt = -1;
    for (let f = f0 + 1; f <= f0 + 25; f++) {
      const live = step(f);
      const res = tracker.update(live, f, TEST_K, null, { pose: inMapUnits(cameraAt(f)), inlierCount: 60, meanErrorPx: 0.8 });
      expect(res.tracked, `frame ${f}`).toBe(true);
      const sel = tracker.selection;
      if (sel.source === "plane" && adoptedAt < 0) adoptedAt = f;
      if (adoptedAt < 0) {
        expect(sel.source).toBe("map");
        expect(sel.planeReject).toMatch(/cooldown/);
      }
      // Map candidate vs the synthetic truth: ~0.1 map units (noise + init scale).
      expect(sel.sourceDeltaTranslation).toBeLessThan(0.3);
    }
    expect(adoptedAt).toBeGreaterThanOrEqual(f0 + 1 + cfg.sourceSwitchCooldownFrames - 1);
    expect(adoptedAt).toBeGreaterThan(0);

    // 2) A plane candidate far from the map candidate is rejected (plane vs
    //    map disagreement) and the map keeps the pose; no jump in the camera.
    const centerBefore = Array.from(tracker.cameraCenter());
    const f = f0 + 26;
    const live = step(f);
    const wrong = inMapUnits(poseFromCenter(rotationAxisAngle([0, 1, 0], 0.004 * f), [0.03 * f + 0.5, 0.005 * f, 0.01 * f]));
    const res = tracker.update(live, f, TEST_K, null, { pose: wrong, inlierCount: 60, meanErrorPx: 0.8 });
    expect(res.tracked).toBe(true);
    expect(tracker.selection.source).toBe("map");
    expect(tracker.selection.planeReject).toMatch(/plane/);
    const after = Array.from(tracker.cameraCenter());
    expect(Math.hypot(after[0] - centerBefore[0], after[1] - centerBefore[1], after[2] - centerBefore[2])).toBeLessThan(0.3);
  });

  it("re-associates landmarks to replenished tracks and does not prune the map while lost (v3 §15)", () => {
    const cfg = DEFAULT_CONFIG.landmarks;
    const f0 = 8;
    const { rng, points } = makeScene(51);
    const tracker = new MapTracker(cfg);
    const estimator = new PoseEstimator(DEFAULT_CONFIG.pose, DEFAULT_CONFIG.ransac, createRng(52));
    let tracks = makeTracks(points, cameraAt(f0), cameraAt(0), 0, rng);
    const rel = estimator.estimate(
      Float64Array.from(tracks, (t) => t.refX), Float64Array.from(tracks, (t) => t.refY),
      Float64Array.from(tracks, (t) => t.x), Float64Array.from(tracks, (t) => t.y),
      tracks.length, TEST_K,
    );
    expect(tracker.tryInitialize(tracks, rel, 0, f0, TEST_K)).toBe(true);
    const byId = new Map(tracks.map((t) => [t.id, t]));
    const pointOf = new Map(tracks.map((t) => [t.id, t.id - 1]));
    const step = (f: number): Track[] => {
      const next: Track[] = [];
      for (const t of byId.values()) {
        const p = project(points, cameraAt(f), pointOf.get(t.id)!, 0.3, rng);
        if (!p) continue;
        t.prevX = t.x; t.prevY = t.y; t.x = p[0]; t.y = p[1];
        next.push(t);
      }
      return next;
    };
    // A few normal frames so landmarks mature (≥ 3 observations).
    let f = f0;
    for (let i = 0; i < 4; i++) {
      f++;
      tracks = step(f);
      expect(tracker.update(tracks, f, TEST_K, null).tracked).toBe(true);
    }
    const landmarksBefore = tracker.map.size;

    // Fast motion kills every track: FAST re-detects the same corners as
    // brand-new tracks (new ids, no landmark). Without re-association PnP
    // would have no observations at all.
    f++;
    let nextId = 10000;
    const fresh: Track[] = step(f).map((t) => {
      const copy: Track = { ...t, id: nextId++, landmarkId: -1, age: 0, anchorFrame: -1, anchorPose: null };
      pointOf.set(copy.id, pointOf.get(t.id)!);
      return copy;
    });
    for (const t of tracks) byId.delete(t.id);
    fresh.forEach((t) => byId.set(t.id, t));
    const lost = tracker.update(fresh, f, TEST_K, null);
    expect(lost.tracked).toBe(false); // nothing linked in this frame…
    expect(lost.reassociated).toBeGreaterThan(cfg.minPnPInliers); // …but the landmarks were re-linked for the next one
    expect(tracker.map.size).toBe(landmarksBefore); // and nothing was pruned while lost

    f++;
    const recovered = tracker.update(step(f), f, TEST_K, null);
    expect(recovered.tracked).toBe(true);
    expect(recovered.inlierCount).toBeGreaterThanOrEqual(cfg.minPnPInliers);
    expect(deg(rotationDistance(tracker.pose.rotation, cameraAt(f).rotation))).toBeLessThan(0.5);

    // Long loss: no observations for many frames keeps every landmark.
    const emptyFrames: Track[] = [];
    for (let i = 0; i < 200; i++) {
      f++;
      tracker.update(emptyFrames, f, TEST_K, null);
    }
    expect(tracker.map.size).toBe(landmarksBefore);
  });

  it("guided recovery: a few links pin a seed pose, the rest are re-associated and PnP recovers in the same frame", () => {
    const cfg = DEFAULT_CONFIG.landmarks;
    const f0 = 8;
    const { rng, points } = makeScene(41);
    const tracker = new MapTracker(cfg);
    const estimator = new PoseEstimator(DEFAULT_CONFIG.pose, DEFAULT_CONFIG.ransac, createRng(42));
    let tracks = makeTracks(points, cameraAt(f0), cameraAt(0), 0, rng);
    const rel = estimator.estimate(
      Float64Array.from(tracks, (t) => t.refX), Float64Array.from(tracks, (t) => t.refY),
      Float64Array.from(tracks, (t) => t.x), Float64Array.from(tracks, (t) => t.y),
      tracks.length, TEST_K,
    );
    expect(tracker.tryInitialize(tracks, rel, 0, f0, TEST_K)).toBe(true);
    const byId = new Map(tracks.map((t) => [t.id, t]));
    const pointOf = new Map(tracks.map((t) => [t.id, t.id - 1]));
    const step = (pose: RigidTransform): Track[] => {
      const next: Track[] = [];
      for (const t of byId.values()) {
        const p = project(points, pose, pointOf.get(t.id)!, 0.3, rng);
        if (!p) continue;
        t.prevX = t.x; t.prevY = t.y; t.x = p[0]; t.y = p[1];
        next.push(t);
      }
      return next;
    };
    let f = f0;
    for (let i = 0; i < 4; i++) {
      f++;
      tracks = step(cameraAt(f));
      expect(tracker.update(tracks, f, TEST_K, null).tracked).toBe(true);
    }
    // The camera stops (so the velocity prediction is wrong) while every
    // track is replaced, and the view sits ~0.012 units off the predicted
    // pose: most landmarks project 2–5 px away from the re-detected corners,
    // only the nearest few fall inside the 2.5 px re-association radius.
    f++;
    const stopped = poseFromCenter(rotationAxisAngle([0, 1, 0], 0.004 * (f - 1)), [0.03 * (f - 1) - 0.012, 0.005 * (f - 1), 0.01 * (f - 1)]);
    let nextId = 20000;
    const fresh: Track[] = step(stopped).map((t) => {
      const copy: Track = { ...t, id: nextId++, landmarkId: -1, age: 0, anchorFrame: -1, anchorPose: null };
      pointOf.set(copy.id, pointOf.get(t.id)!);
      return copy;
    });
    for (const t of tracks) byId.delete(t.id);
    fresh.forEach((t) => byId.set(t.id, t));
    // Frame-to-frame image motion of this frame (what the engine's outlier
    // rejection computes every frame from the surviving tracks).
    const motion = (ts: Track[]) =>
      ransacHomography(
        Float64Array.from(ts, (t) => t.prevX), Float64Array.from(ts, (t) => t.prevY),
        Float64Array.from(ts, (t) => t.x), Float64Array.from(ts, (t) => t.y),
        ts.length, DEFAULT_CONFIG.ransac, createRng(7),
      ).homography;
    const r1 = tracker.update(fresh, f, TEST_K, null, null, motion(fresh)); // links made, nothing to solve yet
    expect(r1.tracked).toBe(false);
    expect(r1.reassociated).toBeGreaterThanOrEqual(cfg.minRecoveryInliers);
    f++;
    const again = step(stopped);
    const r2 = tracker.update(again, f, TEST_K, null, null, motion(again));
    console.log(`[guided] links ${r1.reassociated} → recovered ${r2.tracked} inliers ${r2.inlierCount} reject ${tracker.selection.mapReject}`);
    expect(r2.tracked).toBe(true);
    expect(r2.inlierCount).toBeGreaterThanOrEqual(cfg.minRecoveryInliers);
    expect(deg(rotationDistance(tracker.pose.rotation, stopped.rotation))).toBeLessThan(1.0);
  });

  it("triangulates new landmarks from anchors once parallax is sufficient", () => {
    const { rng, points } = makeScene(3, 120);
    const cfg = DEFAULT_CONFIG.landmarks;
    const tracker = new MapTracker(cfg);
    const estimator = new PoseEstimator(DEFAULT_CONFIG.pose, DEFAULT_CONFIG.ransac, createRng(4));
    const refPose = cameraAt(0);
    const f0 = 12;
    // Only the first 70 points exist at initialization; the rest appear later.
    const initPts = points.subarray(0, 70 * 3);
    let tracks = makeTracks(initPts, cameraAt(f0), refPose, 0, rng);
    const rel = estimator.estimate(
      Float64Array.from(tracks, (t) => t.refX), Float64Array.from(tracks, (t) => t.refY),
      Float64Array.from(tracks, (t) => t.x), Float64Array.from(tracks, (t) => t.y),
      tracks.length, TEST_K,
    );
    expect(tracker.tryInitialize(tracks, rel, 0, f0, TEST_K)).toBe(true);
    const initial = tracker.map.size;

    // New tracks (ids 71..120) appear at frame f0+1 without landmarks.
    const byId = new Map(tracks.map((t) => [t.id, t]));
    for (let i = 70; i < 120; i++) {
      const p = project(points, cameraAt(f0 + 1), i, 0.3, rng);
      if (!p) continue;
      byId.set(i + 1, {
        id: i + 1, x: p[0], y: p[1], prevX: p[0], prevY: p[1], age: 0, score: 10, inlier: true, outlierStreak: 0,
        refX: p[0], refY: p[1], refFrame: -1,
        landmarkId: -1, anchorFrame: -1, anchorX: p[0], anchorY: p[1], anchorPose: null,
        planePoint: null, planeStreak: 0, planeOutliers: 0, offPlane: false,
      });
    }
    let created = 0;
    // landmark id → point index, recorded while the observing track is alive
    // (tracks that leave the frame unlink from their landmark).
    const lmPoint = new Map<number, number>();
    for (let f = f0 + 1; f <= f0 + 25; f++) {
      const pose = cameraAt(f);
      const next: Track[] = [];
      for (const [id, t] of byId) {
        const p = project(points, pose, id - 1, 0.3, rng);
        if (!p) continue;
        if (f > f0 + 1 || id <= 70) { t.prevX = t.x; t.prevY = t.y; t.x = p[0]; t.y = p[1]; }
        next.push(t);
      }
      const res = tracker.update(next, f, TEST_K, null);
      expect(res.tracked).toBe(true);
      created += res.newLandmarks;
      for (const lm of tracker.map.values()) {
        if (lm.trackId > 70) lmPoint.set(lm.id, lm.trackId - 1);
      }
    }
    expect(created).toBeGreaterThan(20);
    expect(tracker.map.size).toBeGreaterThan(initial + 20);
    // New landmarks are accurate in the map frame (scale = 1 / true baseline at f0).
    const scale = 1 / Math.hypot(0.03 * f0, 0.005 * f0, 0.01 * f0);
    const relErrors: number[] = [];
    for (const lm of tracker.map.values()) {
      const i = lmPoint.get(lm.id);
      if (i === undefined) continue;
      const err = Math.hypot(
        lm.position[0] - points[i * 3] * scale,
        lm.position[1] - points[i * 3 + 1] * scale,
        lm.position[2] - points[i * 3 + 2] * scale,
      );
      relErrors.push(err / (Math.hypot(points[i * 3], points[i * 3 + 1], points[i * 3 + 2]) * scale));
    }
    expect(relErrors.length).toBeGreaterThan(15);
    relErrors.sort((a, b) => a - b);
    // Depth from a ~1–3° parallax with 0.3 px noise: a few % relative error.
    expect(relErrors[relErrors.length >> 1]).toBeLessThan(0.05);
    expect(relErrors[relErrors.length - 1]).toBeLessThan(0.15);
  });

  it("does not initialize without parallax", () => {
    const { rng, points } = makeScene(5);
    const tracker = new MapTracker(DEFAULT_CONFIG.landmarks);
    const estimator = new PoseEstimator(DEFAULT_CONFIG.pose, DEFAULT_CONFIG.ransac, createRng(6));
    const tracks = makeTracks(points, cameraAt(0), cameraAt(0), 0, rng);
    const rel = estimator.estimate(
      Float64Array.from(tracks, (t) => t.refX), Float64Array.from(tracks, (t) => t.refY),
      Float64Array.from(tracks, (t) => t.x), Float64Array.from(tracks, (t) => t.y),
      tracks.length, TEST_K,
    );
    expect(tracker.tryInitialize(tracks, rel, 0, 1, TEST_K)).toBe(false);
    expect(tracker.initialized).toBe(false);
  });
});
