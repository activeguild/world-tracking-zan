import { describe, expect, it } from "vitest";
import { mat3Multiply } from "../../src/math/Matrix";
import { refinePosePnP } from "../../src/math/PnP";
import { rotationAxisAngle, rotationDistance, type RigidTransform } from "../../src/math/Pose";
import { gauss, poseFromCenter, randomBoxPoints, randomPlanePoints } from "../helpers/scene";
import { createRng } from "../helpers/synthetic";

const deg = (rad: number) => (rad * 180) / Math.PI;
const F = 640;

function observe(points: Float64Array, pose: RigidTransform, noisePx: number, rng: () => number) {
  const n = points.length / 3;
  const ox = new Float64Array(n);
  const oy = new Float64Array(n);
  const r = pose.rotation;
  const t = pose.translation;
  for (let i = 0; i < n; i++) {
    const X = points[i * 3], Y = points[i * 3 + 1], Z = points[i * 3 + 2];
    const x = r[0] * X + r[1] * Y + r[2] * Z + t[0];
    const y = r[3] * X + r[4] * Y + r[5] * Z + t[1];
    const z = r[6] * X + r[7] * Y + r[8] * Z + t[2];
    ox[i] = x / z + (gauss(rng) * noisePx) / F;
    oy[i] = y / z + (gauss(rng) * noisePx) / F;
  }
  return { ox, oy, n };
}

const opts = { huber: 3 / F, inlierThreshold: 4 / F, maxIterations: 15, epsilon: 1e-8 };

describe("refinePosePnP", () => {
  it("converges from a perturbed prior on a general scene", () => {
    const rng = createRng(1);
    const pts = randomBoxPoints(rng, 120, 1, 4);
    const truth = poseFromCenter(rotationAxisAngle([0.2, 1, 0.1], 0.3), [0.4, -0.1, 0.2]);
    const { ox, oy, n } = observe(pts, truth, 0.3, rng);
    const prior: RigidTransform = {
      rotation: mat3Multiply(rotationAxisAngle([0, 1, 0], 0.05), truth.rotation),
      translation: new Float64Array([truth.translation[0] + 0.05, truth.translation[1] - 0.03, truth.translation[2] + 0.04]),
    };
    const res = refinePosePnP(prior, pts, ox, oy, n, opts);
    expect(deg(rotationDistance(res.pose.rotation, truth.rotation))).toBeLessThan(0.1);
    for (let i = 0; i < 3; i++) expect(res.pose.translation[i]).toBeCloseTo(truth.translation[i], 2);
    expect(res.inlierCount).toBeGreaterThan(n * 0.95);
    expect(res.meanError * F).toBeLessThan(0.6);
  });

  it("handles planar point sets", () => {
    const rng = createRng(2);
    const pts = randomPlanePoints(rng, 100, [0, -0.6, 0.8], 1.5, 1);
    const truth = poseFromCenter(rotationAxisAngle([1, 0.2, 0], -0.2), [0.2, 0.1, 0.1]);
    const { ox, oy, n } = observe(pts, truth, 0.3, rng);
    const prior: RigidTransform = {
      rotation: mat3Multiply(rotationAxisAngle([1, 0, 0], -0.04), truth.rotation),
      translation: new Float64Array([truth.translation[0] - 0.04, truth.translation[1], truth.translation[2] + 0.05]),
    };
    const res = refinePosePnP(prior, pts, ox, oy, n, opts);
    expect(deg(rotationDistance(res.pose.rotation, truth.rotation))).toBeLessThan(0.15);
    for (let i = 0; i < 3; i++) expect(res.pose.translation[i]).toBeCloseTo(truth.translation[i], 2);
  });

  it("is robust to outliers (Huber) and flags them", () => {
    const rng = createRng(3);
    const pts = randomBoxPoints(rng, 150, 1, 4);
    const truth = poseFromCenter(rotationAxisAngle([0, 1, 0], 0.1), [0.1, 0, 0.1]);
    const { ox, oy, n } = observe(pts, truth, 0.3, rng);
    const bad = new Set<number>();
    for (let i = 0; i < n; i++) {
      if (rng() < 0.2) {
        bad.add(i);
        ox[i] += (rng() - 0.5) * 0.3;
        oy[i] += (rng() - 0.5) * 0.3;
      }
    }
    const prior: RigidTransform = {
      rotation: mat3Multiply(rotationAxisAngle([0, 0, 1], 0.03), truth.rotation),
      translation: new Float64Array([truth.translation[0] + 0.03, truth.translation[1] + 0.02, truth.translation[2]]),
    };
    const res = refinePosePnP(prior, pts, ox, oy, n, opts);
    expect(deg(rotationDistance(res.pose.rotation, truth.rotation))).toBeLessThan(0.2);
    let flagged = 0;
    for (const i of bad) if (!res.inliers[i]) flagged++;
    expect(flagged / bad.size).toBeGreaterThan(0.9);
    expect(res.inlierCount).toBeGreaterThanOrEqual(n - bad.size - 3);
  });
});
