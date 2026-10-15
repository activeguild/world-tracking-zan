import { describe, expect, it } from "vitest";
import { rotationAxisAngle, type RigidTransform } from "../../src/math/Pose";
import { triangulateMultiView, triangulatePoint, type TriangulationView } from "../../src/math/Triangulation";
import { createRng } from "../../src/vision/OutlierRejection";
import { gauss } from "../helpers/scene";

/** v25: the linear multi-view triangulation used to rebuild landmarks after an init stage. */
function viewOf(pose: RigidTransform, X: number[], noise: number, rng: () => number): TriangulationView {
  const r = pose.rotation, t = pose.translation;
  const z = r[6] * X[0] + r[7] * X[1] + r[8] * X[2] + t[2];
  return {
    pose,
    x: (r[0] * X[0] + r[1] * X[1] + r[2] * X[2] + t[0]) / z + noise * gauss(rng),
    y: (r[3] * X[0] + r[4] * X[1] + r[5] * X[2] + t[1]) / z + noise * gauss(rng),
  };
}

function poseAt(center: number[], yaw: number): RigidTransform {
  const r = rotationAxisAngle([0, 1, 0], yaw);
  // t = −R C
  const t = new Float64Array([-(r[0] * center[0] + r[1] * center[1] + r[2] * center[2]), -(r[3] * center[0] + r[4] * center[1] + r[5] * center[2]), -(r[6] * center[0] + r[7] * center[1] + r[8] * center[2])]);
  return { rotation: r, translation: t };
}

describe("linear multi-view triangulation (v25)", () => {
  it("recovers the point exactly from noise-free views and matches the two-view solve on two views", () => {
    const rng = createRng(3);
    const X = [0.4, -0.2, 3.1];
    const poses = [poseAt([0, 0, 0], 0), poseAt([0.3, 0.05, 0.1], -0.05), poseAt([-0.25, 0.1, 0.0], 0.04), poseAt([0.5, -0.1, 0.2], -0.1)];
    const views = poses.map((p) => viewOf(p, X, 0, rng));
    const res = triangulateMultiView(views)!;
    expect(res).not.toBeNull();
    for (let i = 0; i < 3; i++) expect(res.point[i]).toBeCloseTo(X[i], 9);
    expect(res.minDepth).toBeGreaterThan(0);
    expect(res.maxError).toBeLessThan(1e-9);
    expect(res.parallax).toBeGreaterThan(0.05);
    // Two views: the same point as the two-view DLT (camera 1 at the identity).
    const two = triangulateMultiView(views.slice(0, 2))!;
    const tv = triangulatePoint(poses[1], views[0].x, views[0].y, views[1].x, views[1].y);
    for (let i = 0; i < 3; i++) expect(two.point[i]).toBeCloseTo(tv.point[i], 8);
  });

  it("averages the noise over many views: four noisy views beat the best pair, a point behind a camera is flagged", () => {
    const rng = createRng(7);
    const X = [0.4, -0.2, 3.1];
    const poses = [poseAt([0, 0, 0], 0), poseAt([0.3, 0.05, 0.1], -0.05), poseAt([-0.25, 0.1, 0.0], 0.04), poseAt([0.5, -0.1, 0.2], -0.1)];
    let errAll = 0, errPair = 0;
    const trials = 200;
    for (let k = 0; k < trials; k++) {
      const views = poses.map((p) => viewOf(p, X, 0.5 / 640, rng));
      const all = triangulateMultiView(views)!;
      const pair = triangulateMultiView([views[0], views[3]])!;
      errAll += Math.hypot(all.point[0] - X[0], all.point[1] - X[1], all.point[2] - X[2]);
      errPair += Math.hypot(pair.point[0] - X[0], pair.point[1] - X[1], pair.point[2] - X[2]);
    }
    console.log(`[mvtri] mean 3D error: 4 views ${(errAll / trials).toFixed(5)} vs widest pair ${(errPair / trials).toFixed(5)}`);
    expect(errAll).toBeLessThan(errPair);
    // Behind: a camera looking away.
    const behind = poseAt([0, 0, 6], 0);
    const v = [viewOf(poses[0], X, 0, rng), viewOf(poses[1], X, 0, rng), { pose: behind, x: 0.1, y: 0.1 }];
    const res = triangulateMultiView(v)!;
    expect(res.minDepth).toBeLessThanOrEqual(0);
    expect(res.maxError).toBe(Number.POSITIVE_INFINITY);
    expect(triangulateMultiView([v[0]])).toBeNull();
  });
});
