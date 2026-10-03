import { describe, expect, it } from "vitest";
import { DEFAULT_CONFIG } from "../../src/ar/ARConfig";
import { angleBetween, rotationAxisAngle, rotationDistance, type RigidTransform } from "../../src/math/Pose";
import { MapTracker } from "../../src/vision/MapTracker";
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
      id: i + 1, x: cur[0], y: cur[1], prevX: cur[0], prevY: cur[1], age: 5, score: 10, inlier: true,
      refX: ref[0], refY: ref[1], refFrame,
      landmarkId: -1, anchorFrame: -1, anchorX: cur[0], anchorY: cur[1], anchorPose: null,
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

  it("triangulates new landmarks from anchors once parallax is sufficient", () => {
    const { rng, points } = makeScene(3, 120);
    const cfg = DEFAULT_CONFIG.landmarks;
    const tracker = new MapTracker(cfg);
    const estimator = new PoseEstimator(DEFAULT_CONFIG.pose, DEFAULT_CONFIG.ransac, createRng(4));
    const refPose = cameraAt(0);
    const f0 = 8;
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
        id: i + 1, x: p[0], y: p[1], prevX: p[0], prevY: p[1], age: 0, score: 10, inlier: true,
        refX: p[0], refY: p[1], refFrame: -1,
        landmarkId: -1, anchorFrame: -1, anchorX: p[0], anchorY: p[1], anchorPose: null,
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
