import { describe, expect, it } from "vitest";
import { fitHorizontalPlane, fitPlaneLeastSquares, horizontalness, planeDistance, planeFromPoints, ransacPlane } from "../../src/math/Plane";
import { angleBetween } from "../../src/math/Pose";
import { randomBoxPoints, randomPlanePoints, gauss } from "../helpers/scene";
import { createRng } from "../helpers/synthetic";

const deg = (rad: number) => (rad * 180) / Math.PI;

describe("Plane", () => {
  it("planeFromPoints / planeDistance", () => {
    const p = planeFromPoints(0, 0, 1, 1, 0, 1, 0, 1, 1)!; // z = 1
    expect(Math.abs(p.normal[2])).toBeCloseTo(1, 10);
    expect(Math.abs(planeDistance(p, 5, -3, 1))).toBeLessThan(1e-12);
    expect(Math.abs(planeDistance(p, 0, 0, 3))).toBeCloseTo(2, 10);
    expect(planeFromPoints(0, 0, 0, 1, 1, 1, 2, 2, 2)).toBeNull();
  });

  it("least-squares fit recovers a noisy plane", () => {
    const rng = createRng(1);
    const normal = [0.2, 0.9, 0.3];
    const pts = randomPlanePoints(rng, 200, normal, 1.5, 1);
    for (let i = 0; i < pts.length; i++) pts[i] += gauss(rng) * 0.005;
    const p = fitPlaneLeastSquares(pts, null, 200)!;
    expect(p).not.toBeNull();
    expect(deg(angleBetween(p.normal, normal, true))).toBeLessThan(0.5);
  });

  it("RANSAC separates plane inliers from 3D clutter", () => {
    const rng = createRng(2);
    const normal = [0, 0.8, 0.6];
    const plane = randomPlanePoints(rng, 150, normal, 1.2, 1);
    for (let i = 0; i < plane.length; i++) plane[i] += gauss(rng) * 0.004;
    const clutter = randomBoxPoints(rng, 80, 0.5, 3, 1.5);
    const all = new Float64Array(plane.length + clutter.length);
    all.set(plane);
    all.set(clutter, plane.length);
    const n = 230;
    const res = ransacPlane(all, n, { threshold: 0.02, confidence: 0.99, maxIterations: 300, minInliers: 30 }, createRng(9));
    expect(res.plane).not.toBeNull();
    let tp = 0, fp = 0;
    for (let i = 0; i < n; i++) {
      if (i < 150) { if (res.inliers[i]) tp++; } else if (res.inliers[i]) fp++;
    }
    expect(tp / 150).toBeGreaterThan(0.95);
    expect(fp).toBeLessThanOrEqual(4);
    expect(deg(angleBetween(res.plane!.normal, normal, true))).toBeLessThan(1);
    expect(res.rmsResidual).toBeLessThan(0.01);
  });

  it("gravity-constrained fit recovers a floor even with a 5° gravity error, rejects a wall", () => {
    const rng = createRng(31);
    const normal = [0, -Math.SQRT1_2, -Math.SQRT1_2];
    const floor = randomPlanePoints(rng, 150, [0, Math.SQRT1_2, Math.SQRT1_2], 1.2, 0.8);
    for (let i = 0; i < floor.length; i++) floor[i] += gauss(rng) * 0.01;
    const clutter = randomBoxPoints(rng, 40, 0.4, 1.0, 0.6);
    const pts = new Float64Array(floor.length + clutter.length);
    pts.set(floor);
    pts.set(clutter, floor.length);
    // Gravity reading tilted by 5° about X.
    const a = (5 * Math.PI) / 180;
    const up = [0, Math.cos(a) * normal[1] - Math.sin(a) * normal[2], Math.sin(a) * normal[1] + Math.cos(a) * normal[2]];
    const res = fitHorizontalPlane(pts, 190, up, 0.06, 20);
    expect(res.plane).not.toBeNull();
    expect(res.inlierCount).toBeGreaterThan(120);
    expect(deg(angleBetween(res.plane!.normal, normal, true))).toBeLessThan(1.5);

    // A wall (heights spread over 1.6 units) has no dense height cluster.
    const wall = randomPlanePoints(rng, 150, [0, 0, 1], 1.5, 0.8);
    const w = fitHorizontalPlane(wall, 150, [0, 1, 0], 0.06, 20);
    expect(w.plane === null || w.inlierCount < 40).toBe(true);
  });

  it("horizontalness", () => {
    expect(horizontalness([0, 1, 0], [0, -9.8, 0])).toBeCloseTo(1, 10);
    expect(horizontalness([0, 0, 1], [0, 1, 0])).toBeCloseTo(0, 10);
    expect(horizontalness([0, Math.SQRT1_2, Math.SQRT1_2], [0, 1, 0])).toBeCloseTo(Math.SQRT1_2, 10);
  });
});
