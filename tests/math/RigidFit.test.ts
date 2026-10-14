import { describe, expect, it } from "vitest";
import { applyTransform, rotationAxisAngle, rotationDistance, type RigidTransform } from "../../src/math/Pose";
import { rigidFit } from "../../src/math/RigidFit";
import { createRng } from "../../src/vision/OutlierRejection";

const deg = (rad: number) => (rad * 180) / Math.PI;

function cloud(rng: () => number, n: number): Float64Array {
  const p = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) {
    p[i * 3] = (rng() - 0.5) * 2;
    p[i * 3 + 1] = 0.7 + (rng() - 0.5) * 0.1;
    p[i * 3 + 2] = 1 + rng() * 1.5;
  }
  return p;
}

describe("rigidFit (改善指示書 Phase 2)", () => {
  it("recovers a rigid motion exactly, and within the noise with 1% jitter", () => {
    const rng = createRng(31);
    const before = cloud(rng, 200);
    const T: RigidTransform = { rotation: rotationAxisAngle([0.2, 1, -0.3], 0.03), translation: new Float64Array([0.04, -0.02, 0.01]) };
    const after = new Float64Array(before.length);
    for (let i = 0; i < 200; i++) after.set(applyTransform(T, before.subarray(i * 3, i * 3 + 3)), i * 3);
    const exact = rigidFit(before, after, 200)!;
    expect(deg(rotationDistance(exact.transform.rotation, T.rotation))).toBeLessThan(1e-6);
    for (let a = 0; a < 3; a++) expect(exact.transform.translation[a]).toBeCloseTo(T.translation[a], 9);
    expect(exact.inliers).toBe(200);
    const noisy = Float64Array.from(after);
    for (let i = 0; i < noisy.length; i++) noisy[i] += (rng() - 0.5) * 0.02;
    const fit = rigidFit(before, noisy, 200)!;
    expect(deg(rotationDistance(fit.transform.rotation, T.rotation))).toBeLessThan(0.2);
    for (let a = 0; a < 3; a++) expect(Math.abs(fit.transform.translation[a] - T.translation[a])).toBeLessThan(0.003);
  });

  it("ignores points that moved on their own (20% refined far from the rigid motion) and reports identity for an unmoved cloud", () => {
    const rng = createRng(32);
    const before = cloud(rng, 300);
    const T: RigidTransform = { rotation: rotationAxisAngle([0, 1, 0], 0.02), translation: new Float64Array([0.03, 0, -0.02]) };
    const after = new Float64Array(before.length);
    for (let i = 0; i < 300; i++) {
      const p = applyTransform(T, before.subarray(i * 3, i * 3 + 3));
      if (i % 5 === 0) for (let a = 0; a < 3; a++) p[a] += (rng() - 0.5) * 0.4; // a landmark the BA moved along its ray
      after.set(p, i * 3);
    }
    const fit = rigidFit(before, after, 300)!;
    expect(deg(rotationDistance(fit.transform.rotation, T.rotation))).toBeLessThan(0.05);
    for (let a = 0; a < 3; a++) expect(Math.abs(fit.transform.translation[a] - T.translation[a])).toBeLessThan(0.002);
    expect(fit.inliers).toBeGreaterThan(220);
    const still = rigidFit(before, before, 300)!;
    expect(deg(rotationDistance(still.transform.rotation, new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1])))).toBeLessThan(1e-9);
    for (let a = 0; a < 3; a++) expect(Math.abs(still.transform.translation[a])).toBeLessThan(1e-12);
    expect(rigidFit(before, after, 2)).toBeNull();
  });
});
