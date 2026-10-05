import { describe, expect, it } from "vitest";
import {
  decomposeEssential,
  estimateEssential8pt,
  ransacEssential,
  recoverPose,
  sampsonErrorSq,
} from "../../src/math/EssentialMatrix";
import { angleBetween, rotationAxisAngle, rotationDistance } from "../../src/math/Pose";
import { triangulatePoint } from "../../src/math/Triangulation";
import { corrupt, poseFromCenter, projectTwoViews, randomBoxPoints, TEST_K } from "../helpers/scene";
import { createRng } from "../helpers/synthetic";

const deg = (rad: number) => (rad * 180) / Math.PI;

describe("Essential matrix", () => {
  const rng = createRng(11);
  const R = rotationAxisAngle([0.1, 1, 0.05], 0.12);
  const pose = poseFromCenter(R, [0.3, 0.05, 0.1]);
  const points = randomBoxPoints(rng, 150);
  const c = projectTwoViews(points, pose, null); // normalized coordinates

  it("8-point on exact data satisfies the epipolar constraint", () => {
    const e = estimateEssential8pt(c.x1, c.y1, c.x2, c.y2, null, c.n)!;
    expect(e).not.toBeNull();
    let maxErr = 0;
    for (let i = 0; i < c.n; i++) maxErr = Math.max(maxErr, Math.sqrt(sampsonErrorSq(e, c.x1[i], c.y1[i], c.x2[i], c.y2[i])));
    expect(maxErr).toBeLessThan(1e-6);
  });

  it("decomposes into a candidate matching the true rotation and translation direction", () => {
    const e = estimateEssential8pt(c.x1, c.y1, c.x2, c.y2, null, c.n)!;
    const { rotations, translation } = decomposeEssential(e);
    const rotErr = Math.min(rotationDistance(rotations[0], R), rotationDistance(rotations[1], R));
    expect(deg(rotErr)).toBeLessThan(0.01);
    expect(deg(angleBetween(translation, pose.translation, true))).toBeLessThan(0.01);
  });

  it("recoverPose selects the physically valid candidate via cheirality", () => {
    const e = estimateEssential8pt(c.x1, c.y1, c.x2, c.y2, null, c.n)!;
    const rec = recoverPose(e, c.x1, c.y1, c.x2, c.y2, null, c.n, 0.01)!;
    expect(rec).not.toBeNull();
    expect(rec.good).toBeGreaterThanOrEqual(c.n - 1);
    expect(rec.secondBest).toBeLessThan(c.n * 0.2);
    expect(deg(rotationDistance(rec.pose.rotation, R))).toBeLessThan(0.01);
    expect(deg(angleBetween(rec.pose.translation, pose.translation))).toBeLessThan(0.01);
  });

  it("RANSAC rejects outliers and recovers the motion with pixel noise", () => {
    const rng2 = createRng(5);
    const pts = randomBoxPoints(rng2, 200);
    const cp = projectTwoViews(pts, pose, TEST_K, 0.5, rng2);
    const truth = corrupt(cp, 0.25, TEST_K, rng2);
    // to normalized
    const n = cp.n;
    const nx1 = new Float64Array(n), ny1 = new Float64Array(n), nx2 = new Float64Array(n), ny2 = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      nx1[i] = (cp.x1[i] - TEST_K.cx) / TEST_K.fx;
      ny1[i] = (cp.y1[i] - TEST_K.cy) / TEST_K.fy;
      nx2[i] = (cp.x2[i] - TEST_K.cx) / TEST_K.fx;
      ny2[i] = (cp.y2[i] - TEST_K.cy) / TEST_K.fy;
    }
    const res = ransacEssential(nx1, ny1, nx2, ny2, n, { threshold: 2 / 640, confidence: 0.99, maxIterations: 500, minCorrespondences: 20 }, createRng(9));
    expect(res.essential).not.toBeNull();
    let tp = 0, fp = 0, total = 0;
    for (let i = 0; i < n; i++) {
      if (truth[i]) { total++; if (res.inliers[i]) tp++; } else if (res.inliers[i]) fp++;
    }
    // 0.5 px Gaussian noise on both views vs a 2 px Sampson gate: a few true
    // inliers legitimately fall outside the gate.
    expect(tp / total).toBeGreaterThan(0.85);
    expect(fp).toBeLessThanOrEqual(3);
    const rec = recoverPose(res.essential!, nx1, ny1, nx2, ny2, res.inliers, n, 4 / 640)!;
    expect(deg(rotationDistance(rec.pose.rotation, R))).toBeLessThan(0.5);
    expect(deg(angleBetween(rec.pose.translation, pose.translation))).toBeLessThan(3);
  });

  it("triangulation recovers the points up to the translation scale", () => {
    for (let i = 0; i < c.n; i += 10) {
      const tri = triangulatePoint(pose, c.x1[i], c.y1[i], c.x2[i], c.y2[i]);
      expect(tri.depth1).toBeGreaterThan(0);
      expect(tri.depth2).toBeGreaterThan(0);
      expect(tri.point[0]).toBeCloseTo(c.points[i * 3], 6);
      expect(tri.point[1]).toBeCloseTo(c.points[i * 3 + 1], 6);
      expect(tri.point[2]).toBeCloseTo(c.points[i * 3 + 2], 6);
      expect(tri.error).toBeLessThan(1e-8);
      expect(tri.parallax).toBeGreaterThan(0.01);
    }
  });
});
