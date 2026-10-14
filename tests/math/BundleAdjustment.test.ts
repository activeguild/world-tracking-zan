import { describe, expect, it } from "vitest";
import { type BAProblem, bundleAdjust, bundleError } from "../../src/math/BundleAdjustment";
import { rotationAxisAngle, rotationDistance, type RigidTransform } from "../../src/math/Pose";
import { createRng } from "../../src/vision/OutlierRejection";

/**
 * Phase 7: local bundle adjustment on a synthetic problem. Ground-truth
 * cameras and points, exact observations; the starting state is perturbed
 * (poses and points). BA must bring the reprojection error back to the
 * observation noise and the free poses back to the truth.
 */
const F = 640; // px per normalized unit, for readable thresholds

function cameraAt(C: number[], yawDeg: number, pitchDeg: number): RigidTransform {
  const ry = rotationAxisAngle([0, 1, 0], (yawDeg * Math.PI) / 180);
  const rx = rotationAxisAngle([1, 0, 0], (pitchDeg * Math.PI) / 180);
  // World-from-camera = Ry Rx; camera matrix is the transpose.
  const wc = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) wc[i * 3 + j] = ry[i * 3] * rx[j] + ry[i * 3 + 1] * rx[3 + j] + ry[i * 3 + 2] * rx[6 + j];
  const R = new Float64Array([wc[0], wc[3], wc[6], wc[1], wc[4], wc[7], wc[2], wc[5], wc[8]]);
  return { rotation: R, translation: new Float64Array([-(R[0] * C[0] + R[1] * C[1] + R[2] * C[2]), -(R[3] * C[0] + R[4] * C[1] + R[5] * C[2]), -(R[6] * C[0] + R[7] * C[1] + R[8] * C[2])]) };
}

function center(p: RigidTransform): number[] {
  const r = p.rotation, t = p.translation;
  return [-(r[0] * t[0] + r[3] * t[1] + r[6] * t[2]), -(r[1] * t[0] + r[4] * t[1] + r[7] * t[2]), -(r[2] * t[0] + r[5] * t[1] + r[8] * t[2])];
}

function build(seed: number, noisePx: number) {
  const rng = createRng(seed);
  // Points: a floor patch (y = 1, 45° below the first camera) plus a back wall.
  const nL = 300;
  const truth = new Float64Array(nL * 3);
  for (let i = 0; i < nL; i++) {
    if (i < 200) {
      truth[i * 3] = (rng() - 0.5) * 2.0;
      truth[i * 3 + 1] = 0.7 + (rng() - 0.5) * 0.05;
      truth[i * 3 + 2] = 1.0 + rng() * 1.5;
    } else {
      truth[i * 3] = (rng() - 0.5) * 2.0;
      truth[i * 3 + 1] = (rng() - 0.5) * 1.0;
      truth[i * 3 + 2] = 2.6 + (rng() - 0.5) * 0.1;
    }
  }
  const truePoses = [
    cameraAt([0, 0, 0], 0, 0),
    cameraAt([0.15, 0.02, 0.05], 4, 1),
    cameraAt([0.3, 0.03, 0.1], 8, -2),
    cameraAt([0.1, -0.05, 0.2], -5, 6),
    cameraAt([-0.2, 0.04, 0.15], -9, -4),
  ];
  const observations: BAProblem["observations"] = [];
  for (let k = 0; k < truePoses.length; k++) {
    const p = truePoses[k];
    const r = p.rotation, t = p.translation;
    for (let i = 0; i < nL; i++) {
      const X = truth[i * 3], Y = truth[i * 3 + 1], Z = truth[i * 3 + 2];
      const z = r[6] * X + r[7] * Y + r[8] * Z + t[2];
      const u = (r[0] * X + r[1] * Y + r[2] * Z + t[0]) / z;
      const v = (r[3] * X + r[4] * Y + r[5] * Z + t[1]) / z;
      if (Math.abs(u) > 0.6 || Math.abs(v) > 0.45) continue; // outside a ~66° view
      observations.push({ keyframe: k, landmark: i, x: u + ((rng() - 0.5) * noisePx) / F, y: v + ((rng() - 0.5) * noisePx) / F });
    }
  }
  // Perturbed start: poses (except the first) off by ~1° / 2 cm, points by ~2% of depth.
  const keyframes = truePoses.map((p, k) => {
    if (k === 0) return { pose: { rotation: Float64Array.from(p.rotation), translation: Float64Array.from(p.translation) }, fixed: true };
    const dR = rotationAxisAngle([rng() - 0.5, rng() - 0.5, rng() - 0.5], (1.0 * Math.PI) / 180);
    const rot = new Float64Array(9);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) rot[i * 3 + j] = dR[i * 3] * p.rotation[j] + dR[i * 3 + 1] * p.rotation[3 + j] + dR[i * 3 + 2] * p.rotation[6 + j];
    return {
      pose: { rotation: rot, translation: new Float64Array([p.translation[0] + (rng() - 0.5) * 0.04, p.translation[1] + (rng() - 0.5) * 0.04, p.translation[2] + (rng() - 0.5) * 0.04]) },
      fixed: false,
    };
  });
  const landmarks = Float64Array.from(truth);
  for (let i = 0; i < nL * 3; i++) landmarks[i] += (rng() - 0.5) * 0.04;
  const problem: BAProblem = { keyframes, landmarks, landmarkCount: nL, observations };
  return { problem, truth, truePoses };
}

describe("bundle adjustment (Phase 7)", () => {
  it("brings a perturbed map and poses back to the observations (exact observations)", () => {
    const { problem, truePoses } = build(1, 0);
    const before = bundleError(problem) * F;
    const res = bundleAdjust(problem, { huber: 2 / F, maxIterations: 30, epsilon: 1e-9 });
    const after = bundleError(problem) * F;
    console.log(`[ba] exact: ${before.toFixed(2)} → ${after.toFixed(4)} px in ${res.iterations} it, pose shift max ${(res.maxPoseShift * 100).toFixed(2)} cm / ${res.maxPoseRotationDeg.toFixed(2)}°, lm shift max ${(res.maxLandmarkShift * 100).toFixed(2)} cm`);
    expect(before).toBeGreaterThan(3);
    expect(after).toBeLessThan(0.05);
    expect(res.errorAfter * F).toBeCloseTo(after, 6);
    expect(res.freeKeyframes).toBe(4);
    // Free poses back to the truth (the scale gauge keeps the start's median depth, which is the truth's up to the symmetric noise).
    for (let k = 1; k < truePoses.length; k++) {
      const c = center(problem.keyframes[k].pose), ct = center(truePoses[k]);
      expect(Math.hypot(c[0] - ct[0], c[1] - ct[1], c[2] - ct[2])).toBeLessThan(0.01);
      expect((rotationDistance(problem.keyframes[k].pose.rotation, truePoses[k].rotation) * 180) / Math.PI).toBeLessThan(0.2);
    }
    // The fixed keyframe did not move.
    for (const v of problem.keyframes[0].pose.translation) expect(Math.abs(v)).toBe(0);
  });

  it("with 1 px observation noise the error settles near the noise floor and the Huber kernel tolerates outliers", () => {
    const { problem } = build(2, 1.0);
    // Corrupt 5% of the observations grossly (wrong landmark association).
    const rng = createRng(77);
    for (const o of problem.observations) if (rng() < 0.05) {
      o.x += (rng() - 0.5) * 0.1;
      o.y += (rng() - 0.5) * 0.1;
    }
    const before = bundleError(problem) * F;
    const res = bundleAdjust(problem, { huber: 2 / F, maxIterations: 30, epsilon: 1e-9 });
    // Error of the clean observations only.
    let sum = 0, n = 0;
    for (const o of problem.observations) {
      const p = problem.keyframes[o.keyframe].pose;
      const r = p.rotation, t = p.translation;
      const X = problem.landmarks[o.landmark * 3], Y = problem.landmarks[o.landmark * 3 + 1], Z = problem.landmarks[o.landmark * 3 + 2];
      const z = r[6] * X + r[7] * Y + r[8] * Z + t[2];
      const e = Math.hypot((r[0] * X + r[1] * Y + r[2] * Z + t[0]) / z - o.x, (r[3] * X + r[4] * Y + r[5] * Z + t[1]) / z - o.y) * F;
      if (e < 6) {
        sum += e;
        n++;
      }
    }
    const cleanAfter = sum / n;
    console.log(`[ba] noisy: ${before.toFixed(2)} → clean ${cleanAfter.toFixed(2)} px (${n}/${problem.observations.length} within 6 px) in ${res.iterations} it`);
    expect(before).toBeGreaterThan(3);
    expect(cleanAfter).toBeLessThan(0.6); // uniform ±0.5 px noise → mean |e| ≈ 0.3–0.4 px
    expect(n).toBeGreaterThan(problem.observations.length * 0.9);
  });

  it("does nothing harmful on an already consistent problem", () => {
    const { problem } = build(3, 0);
    // Replace the perturbed start with the truth: re-build with zero perturbation by solving once, then run again.
    bundleAdjust(problem, { huber: 2 / F, maxIterations: 30, epsilon: 1e-9 });
    const res = bundleAdjust(problem, { huber: 2 / F, maxIterations: 10, epsilon: 1e-9 });
    expect(res.errorAfter * F).toBeLessThan(0.05);
    expect(res.maxLandmarkShift).toBeLessThan(1e-3);
    expect(res.maxPoseShift).toBeLessThan(1e-3);
  });
});
