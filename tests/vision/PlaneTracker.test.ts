import { describe, expect, it } from "vitest";
import { resolveConfig } from "../../src/ar/ARConfig";
import { mulMatVec3 } from "../../src/math/Decomposition";
import { type Mat3, mat3Identity } from "../../src/math/Matrix";
import { rotationDistance, rotationY, type RigidTransform } from "../../src/math/Pose";
import { createRng } from "../../src/vision/OutlierRejection";
import { PlaneTracker } from "../../src/vision/PlaneTracker";
import type { Track } from "../../src/vision/types";
import { gauss, poseFromCenter, randomBoxPoints, TEST_K } from "../helpers/scene";

/**
 * Plane-anchored pose (修正指示書 §5–§9): with the plane fixed, the camera pose
 * is recovered from features lifted onto the plane, without any triangulated
 * depth, and features that are not on the plane are rejected.
 */
const K = TEST_K;
// Desk tilted 45°, crossing the optical axis 2 units ahead: 0.7071 y + 0.7071 z = √2.
const PLANE_UP = [0, Math.SQRT1_2, Math.SQRT1_2];
const PLANE_DIST = Math.SQRT2;
// Engine convention: n·X + d = 0 with n pointing toward the camera (d > 0).
const PLANE = { normal: [-PLANE_UP[0], -PLANE_UP[1], -PLANE_UP[2]], d: PLANE_DIST, center: [0, 0, 2] };

/** Points on the desk inside the camera-0 frustum: (b, 0, 2) + a · (0, √½, −√½). */
function deskPoints(rng: () => number, count: number): Float64Array {
  const pts = new Float64Array(count * 3);
  for (let i = 0; i < count; i++) {
    const a = (rng() * 2 - 1) * 0.5;
    const b = (rng() * 2 - 1) * 0.8;
    pts[i * 3] = b;
    pts[i * 3 + 1] = Math.SQRT1_2 * a;
    pts[i * 3 + 2] = 2 - Math.SQRT1_2 * a;
  }
  return pts;
}

function makeTrack(id: number, x: number, y: number): Track {
  return {
    id, x, y, prevX: x, prevY: y, age: 1, score: 0, inlier: true, outlierStreak: 0,
    refX: x, refY: y, refFrame: -1, landmarkId: -1, anchorFrame: -1, anchorX: x, anchorY: y, anchorPose: null,
    planePoint: null, planeStreak: 0, planeOutliers: 0, offPlane: false,
  };
}

function project(pose: RigidTransform, p: ArrayLike<number>): [number, number] | null {
  const c = mulMatVec3(pose.rotation, p);
  c[0] += pose.translation[0];
  c[1] += pose.translation[1];
  c[2] += pose.translation[2];
  if (c[2] <= 0.05) return null;
  const u = (c[0] / c[2]) * K.fx + K.cx;
  const v = (c[1] / c[2]) * K.fy + K.cy;
  if (u < 0 || u >= K.width || v < 0 || v >= K.height) return null;
  return [u, v];
}

function truePose(f: number): RigidTransform {
  const r: Mat3 = rotationY(0.003 * f);
  return poseFromCenter(r, [0.01 * f, 0.004 * f, 0.006 * f]);
}

describe("PlaneTracker", () => {
  it("recovers the camera pose from plane-lifted features and rejects off-plane ones", () => {
    const rng = createRng(31);
    const cfg = resolveConfig().planeTracking;
    const tracker = new PlaneTracker(cfg);
    const onPlane = deskPoints(rng, 120);
    const offPlane = randomBoxPoints(rng, 50, 1.0, 3.0, 0.8);
    // Sanity: the generated desk points satisfy the plane equation.
    for (let i = 0; i < 120; i++) {
      expect(PLANE.normal[1] * onPlane[i * 3 + 1] + PLANE.normal[2] * onPlane[i * 3 + 2] + PLANE.d).toBeCloseTo(0, 12);
    }

    // Frame 0: camera at the map origin; every visible point becomes a track.
    const identity: RigidTransform = { rotation: mat3Identity(), translation: new Float64Array(3) };
    const tracks: Track[] = [];
    const truth = new Map<number, { p: number[]; onPlane: boolean }>();
    let id = 1;
    const addAll = (pts: Float64Array, isPlane: boolean) => {
      for (let i = 0; i < pts.length / 3; i++) {
        const p = [pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2]];
        const uv = project(identity, p);
        if (!uv) continue;
        tracks.push(makeTrack(id, uv[0], uv[1]));
        truth.set(id, { p, onPlane: isPlane });
        id++;
      }
    };
    addAll(onPlane, true);
    addAll(offPlane, false);
    const planeTrackCount = [...truth.values()].filter((t) => t.onPlane).length;
    expect(planeTrackCount).toBeGreaterThan(60);

    tracker.setAnchor(PLANE, identity, 0);
    const lifted = tracker.lift(tracks, identity, K, () => false);
    expect(lifted).toBe(tracks.length);
    // Lifted plane points coincide with the true 3D points for on-plane features.
    for (const t of tracks) {
      const tr = truth.get(t.id)!;
      if (!tr.onPlane) continue;
      expect(Math.hypot(t.planePoint![0] - tr.p[0], t.planePoint![1] - tr.p[1], t.planePoint![2] - tr.p[2])).toBeLessThan(1e-9);
    }

    let prior = identity;
    let live = tracks;
    let lastInliers = 0;
    for (let f = 1; f <= 40; f++) {
      const pose = truePose(f);
      const next: Track[] = [];
      for (const t of live) {
        const uv = project(pose, truth.get(t.id)!.p);
        if (!uv) continue;
        t.x = uv[0] + gauss(rng) * 0.3;
        t.y = uv[1] + gauss(rng) * 0.3;
        next.push(t);
      }
      live = next;
      const res = tracker.update(live, prior, K);
      expect(res.tracked, `frame ${f}`).toBe(true);
      prior = res.pose!;
      lastInliers = res.inlierCount;

      const rotErrDeg = (rotationDistance(res.pose!.rotation, pose.rotation) * 180) / Math.PI;
      const tErr = Math.hypot(
        res.pose!.translation[0] - pose.translation[0],
        res.pose!.translation[1] - pose.translation[1],
        res.pose!.translation[2] - pose.translation[2],
      );
      // During probation the off-plane features still take part (Huber-
      // weighted); once rejected the estimate is tight.
      // Off-plane features near the desk surface take a while to show
      // parallax; until they are rejected they enter Huber-weighted and bias
      // the estimate slightly. Early on: loose; at the end: tight.
      const settled = f > cfg.probationFrames + cfg.maxOutlierStreak + 2;
      expect(rotErrDeg, `frame ${f} rotation`).toBeLessThan(settled ? 0.25 : 0.4);
      expect(tErr, `frame ${f} translation`).toBeLessThan(settled ? 0.006 : 0.01);
      if (f === 40) {
        expect(rotErrDeg, "final rotation").toBeLessThan(0.1);
        expect(tErr, "final translation").toBeLessThan(0.003);
      }
      if (f % 10 === 0) console.log(`[plane-tracker] f=${f} rot ${rotErrDeg.toFixed(3)}° t ${tErr.toFixed(4)} inliers ${res.inlierCount}/${res.candidateCount}`);
    }

    // Off-plane features with visible parallax have been identified; plane
    // features are all inliers. (Box points within a few cm of the desk are
    // indistinguishable from desk points at this baseline and may remain.)
    const offLive = live.filter((t) => !truth.get(t.id)!.onPlane);
    const rejected = offLive.filter((t) => t.offPlane).length;
    console.log(`[plane-tracker] off-plane rejected ${rejected}/${offLive.length}`);
    expect(rejected / offLive.length).toBeGreaterThan(0.6);
    const planeLive = live.filter((t) => truth.get(t.id)!.onPlane);
    expect(planeLive.every((t) => !t.offPlane)).toBe(true);
    expect(lastInliers).toBeGreaterThanOrEqual(planeLive.length * 0.95);
    const r = tracker.result;
    expect(r.inlierRatio).toBeGreaterThan(0.85);
    expect(r.meanErrorPx).toBeLessThan(0.8);
    expect(r.confidence).toBeGreaterThan(0.7);
  });

  it("reports not tracked when too few plane features remain (fallback to the map)", () => {
    const cfg = resolveConfig().planeTracking;
    const tracker = new PlaneTracker(cfg);
    const identity: RigidTransform = { rotation: mat3Identity(), translation: new Float64Array(3) };
    const tracks = [makeTrack(1, 300, 200), makeTrack(2, 340, 260), makeTrack(3, 280, 300), makeTrack(4, 400, 220)];
    tracker.setAnchor(PLANE, identity, 0);
    tracker.lift(tracks, identity, K, () => true);
    const res = tracker.update(tracks, identity, K);
    expect(res.tracked).toBe(false);
    expect(res.candidateCount).toBe(4);
  });

  it("does not lift grazing rays or points far beyond the anchor distance", () => {
    const cfg = resolveConfig().planeTracking;
    const tracker = new PlaneTracker(cfg);
    const identity: RigidTransform = { rotation: mat3Identity(), translation: new Float64Array(3) };
    // A plane nearly parallel to the viewing direction: a floor seen edge-on.
    const floor = { normal: [0, -1, 0.02], d: 0.3, center: [0, 0.3, 0] };
    tracker.setAnchor(floor, identity, 0);
    const near = makeTrack(1, 320, 470); // looks steeply down → ok
    const far = makeTrack(2, 320, 250); // near the horizon → grazing / far
    tracker.lift([near, far], identity, K, () => false);
    expect(near.planePoint).not.toBeNull();
    expect(far.planePoint).toBeNull();
  });
});
