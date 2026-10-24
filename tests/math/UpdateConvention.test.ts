import { describe, expect, it } from "vitest";
import { type BAProblem, bundleAdjust, bundleError } from "../../src/math/BundleAdjustment";
import { mat3Multiply, type Mat3 } from "../../src/math/Matrix";
import { refinePosePnP } from "../../src/math/PnP";
import { rotationAxisAngle, rotationDistance, type RigidTransform } from "../../src/math/Pose";
import { projectNormalized, reprojectionJacobian } from "../../src/math/Reprojection";
import { createRng } from "../../src/vision/OutlierRejection";
import { poseFromCenter } from "../helpers/scene";

/**
 * 修正指示書「スマホ傾斜時にワールド固定オブジェクトの回転が徐々にずれる」§1–§2, §6-A.
 *
 * The one convention every solver in this engine uses (documented here as
 * the test's premise and asserted below):
 *
 *   pose        world-to-camera:  X_cam = R · X_map + t          (RigidTransform)
 *   rotation    left perturbation: R' = Exp(δω) · R, δω in radians, camera axes
 *   translation independent:       t' = t + δt   (not rotated with R)
 *   landmark                       X' = X + δX
 *   residual    r = π(R X + t) − observation,  π(p) = (p.x / p.z, p.y / p.z)
 *
 * so  δX_cam ≈ −[R X]× δω + δt  and the Jacobian is `reprojectionJacobian`
 * (shared by PnP and BA). The tests compare it against central differences
 * of the *actual* projection under the *actual* update, in explicit cases
 * (origin / away from the origin, rotation only / translation only / both,
 * roll / pitch / yaw separately, several points), check that a finite step
 * is predicted to second order (so the cross terms are right too), that the
 * rotation update never moves t, and that BA keeps every rotation
 * orthonormal with det +1.
 */
const F = 640;
const deg = (rad: number) => (rad * 180) / Math.PI;
const I3: Mat3 = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);

/** The update exactly as PnP.ts / BundleAdjustment.ts apply it. */
function applyUpdate(pose: RigidTransform, dw: ArrayLike<number>, dt: ArrayLike<number>): RigidTransform {
  const a = Math.hypot(dw[0], dw[1], dw[2]);
  const dR = a > 1e-12 ? rotationAxisAngle([dw[0] / a, dw[1] / a, dw[2] / a], a) : I3;
  return {
    rotation: mat3Multiply(dR, pose.rotation),
    translation: new Float64Array([pose.translation[0] + dt[0], pose.translation[1] + dt[1], pose.translation[2] + dt[2]]),
  };
}

function residual(pose: RigidTransform, X: number[], obs: [number, number]): [number, number] {
  const out = new Float64Array(2);
  expect(projectNormalized(pose.rotation, pose.translation, X[0], X[1], X[2], out)).toBe(true);
  return [out[0] - obs[0], out[1] - obs[1]];
}

/** Central-difference column k (0–2 δω, 3–5 δt, 6–8 δX) of the residual. */
function numericColumn(pose: RigidTransform, X: number[], k: number, h: number): [number, number] {
  const e = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  const at = (sign: number) => {
    e.fill(0);
    e[k] = sign * h;
    const p = applyUpdate(pose, e.slice(0, 3), e.slice(3, 6));
    return residual(p, [X[0] + e[6], X[1] + e[7], X[2] + e[8]], [0, 0]);
  };
  const plus = at(1), minus = at(-1);
  return [(plus[0] - minus[0]) / (2 * h), (plus[1] - minus[1]) / (2 * h)];
}

/** A map point seen at camera coordinates (xc, yc, depth) by `pose`. */
function pointInFront(pose: RigidTransform, xc: number, yc: number, depth: number): number[] {
  const r = pose.rotation, t = pose.translation;
  const dx = xc - t[0], dy = yc - t[1], dz = depth - t[2];
  return [r[0] * dx + r[3] * dy + r[6] * dz, r[1] * dx + r[4] * dy + r[7] * dz, r[2] * dx + r[5] * dy + r[8] * dz];
}

const cases: { name: string; pose: RigidTransform }[] = [
  { name: "camera at the origin, identity", pose: poseFromCenter(I3, [0, 0, 0]) },
  { name: "camera at the origin, yaw 30°", pose: poseFromCenter(rotationAxisAngle([0, 1, 0], 0.52), [0, 0, 0]) },
  { name: "camera 3.7 u from the origin, identity", pose: poseFromCenter(I3, [2, -1, 3]) },
  { name: "camera 3.7 u from the origin, yaw 30° pitch −10°", pose: poseFromCenter(mat3Multiply(rotationAxisAngle([0, 1, 0], 0.52), rotationAxisAngle([1, 0, 0], -0.17)), [2, -1, 3]) },
  { name: "camera 11 u from the origin, roll 20° pitch 15°", pose: poseFromCenter(mat3Multiply(rotationAxisAngle([0, 0, 1], 0.35), rotationAxisAngle([1, 0, 0], 0.26)), [-6, 4, 8]) },
];
const points = [
  [0, 0, 2.0],
  [0.6, -0.3, 1.5],
  [-0.9, 0.5, 3.0],
  [1.2, 0.8, 6.0],
  [-0.2, -0.7, 0.8],
];

describe("update convention and Jacobian (傾斜時の回転ずれ §1–§2)", () => {
  it("§2: the analytic Jacobian matches central differences column by column — pitch (δω_x), yaw (δω_y), roll (δω_z), δt, δX — at the origin and far from it, for several points", () => {
    const ju = new Float64Array(9), jv = new Float64Array(9);
    const h = 1e-6;
    let worst = 0;
    for (const c of cases) {
      for (const [xc, yc, depth] of points) {
        const X = pointInFront(c.pose, xc, yc, depth);
        expect(reprojectionJacobian(c.pose.rotation, c.pose.translation, X[0], X[1], X[2], ju, jv)).toBe(true);
        for (let k = 0; k < 9; k++) {
          const [nu, nv] = numericColumn(c.pose, X, k, h);
          const scale = Math.max(1, Math.abs(nu), Math.abs(nv));
          const eu = Math.abs(ju[k] - nu) / scale, ev = Math.abs(jv[k] - nv) / scale;
          worst = Math.max(worst, eu, ev);
          expect(eu, `${c.name}: column ${k} of u`).toBeLessThan(1e-6);
          expect(ev, `${c.name}: column ${k} of v`).toBeLessThan(1e-6);
        }
      }
    }
    console.log(`[convention] ${cases.length} poses × ${points.length} points × 9 columns: worst relative mismatch ${worst.toExponential(2)}`);
  });

  it("§2: a finite step (rotation only / translation only / both) is predicted by J to second order — the cross terms of the update are right, including far from the origin", () => {
    const ju = new Float64Array(9), jv = new Float64Array(9);
    const steps: { name: string; dw: number[]; dt: number[] }[] = [
      { name: "pitch only", dw: [1, 0, 0], dt: [0, 0, 0] },
      { name: "yaw only", dw: [0, 1, 0], dt: [0, 0, 0] },
      { name: "roll only", dw: [0, 0, 1], dt: [0, 0, 0] },
      { name: "rotation only (all axes)", dw: [0.5, -0.7, 0.3], dt: [0, 0, 0] },
      { name: "translation only", dw: [0, 0, 0], dt: [0.6, -0.3, 0.7] },
      { name: "rotation and translation", dw: [0.4, 0.6, -0.5], dt: [-0.5, 0.4, 0.3] },
    ];
    for (const c of cases) {
      for (const s of steps) {
        // Two step sizes: the remainder must shrink by ~4× when the step halves (second order).
        const remainders: number[] = [];
        let bound = 0;
        for (const eps of [2e-3, 1e-3]) {
          let maxRem = 0;
          for (const [xc, yc, depth] of points) {
            const X = pointInFront(c.pose, xc, yc, depth);
            reprojectionJacobian(c.pose.rotation, c.pose.translation, X[0], X[1], X[2], ju, jv);
            // The second-order term of a rotation step grows with |R X|² (the lever arm the
            // rotation acts on, ≈ |t| far from the origin): the absolute bound scales with it.
            const r = c.pose.rotation;
            const lever = Math.max(1, Math.hypot(r[0] * X[0] + r[1] * X[1] + r[2] * X[2], r[3] * X[0] + r[4] * X[1] + r[5] * X[2], r[6] * X[0] + r[7] * X[1] + r[8] * X[2]));
            if (eps === 1e-3) bound = Math.max(bound, 20 * eps * eps * lever * lever);
            const dw = s.dw.map((v) => v * eps), dt = s.dt.map((v) => v * eps);
            const r0 = residual(c.pose, X, [0, 0]);
            const r1 = residual(applyUpdate(c.pose, dw, dt), X, [0, 0]);
            const d = [...dw, ...dt, 0, 0, 0];
            let pu = 0, pv = 0;
            for (let k = 0; k < 9; k++) {
              pu += ju[k] * d[k];
              pv += jv[k] * d[k];
            }
            maxRem = Math.max(maxRem, Math.hypot(r1[0] - r0[0] - pu, r1[1] - r0[1] - pv));
          }
          remainders.push(maxRem);
        }
        // Remainder is O(eps²) with a coefficient of the order of |R X|² / z (the curvature of π
        // along the step), and it shrinks by 4× when the step halves (3–5.5 allowed; a
        // first-order error in J would give a ratio of 2 and a remainder ∝ eps).
        expect(remainders[1], `${c.name}: ${s.name}`).toBeLessThan(bound);
        if (remainders[1] > 1e-12) {
          const ratio = remainders[0] / remainders[1];
          expect(ratio, `${c.name}: ${s.name} (second order)`).toBeGreaterThan(3);
          expect(ratio, `${c.name}: ${s.name} (second order)`).toBeLessThan(5.5);
        }
      }
    }
  });

  it("§1: the rotation update leaves t bit-identical and the translation update leaves R bit-identical; the rotation stays orthonormal with det +1", () => {
    const pose = cases[3].pose;
    const rotOnly = applyUpdate(pose, [0.01, -0.02, 0.015], [0, 0, 0]);
    expect(Array.from(rotOnly.translation)).toEqual(Array.from(pose.translation));
    expect(rotationDistance(rotOnly.rotation, pose.rotation)).toBeGreaterThan(0.02);
    const transOnly = applyUpdate(pose, [0, 0, 0], [0.1, 0.2, -0.3]);
    expect(Array.from(transOnly.rotation)).toEqual(Array.from(pose.rotation));
    expectRotation(rotOnly.rotation, 1e-14);
  });

  it("§1: after a bundle adjustment every keyframe rotation is orthonormal with det +1 (exact and noisy observations)", () => {
    for (const noise of [0, 1.0]) {
      const { problem } = buildProblem(21 + noise, noise);
      const res = bundleAdjust(problem, { huber: 2 / F, maxIterations: 30, epsilon: 1e-9 });
      expect(res.iterations).toBeGreaterThan(0);
      for (const kf of problem.keyframes) expectRotation(kf.pose.rotation, 1e-12);
    }
  });

  it("§6-A: a rotation-only perturbation of the free keyframes is removed by BA without the translations wandering off (exact observations)", () => {
    const { problem, truePoses } = buildProblem(31, 0, { rotationOnlyDeg: 1.5 });
    const before = bundleError(problem) * F;
    const res = bundleAdjust(problem, { huber: 2 / F, maxIterations: 30, epsilon: 1e-9 });
    const after = bundleError(problem) * F;
    let maxRot = 0, maxCentre = 0;
    for (let k = 1; k < truePoses.length; k++) {
      maxRot = Math.max(maxRot, deg(rotationDistance(problem.keyframes[k].pose.rotation, truePoses[k].rotation)));
      const c = centre(problem.keyframes[k].pose), ct = centre(truePoses[k]);
      maxCentre = Math.max(maxCentre, Math.hypot(c[0] - ct[0], c[1] - ct[1], c[2] - ct[2]));
    }
    console.log(`[convention] rotation-only perturbation 1.5°: ${before.toFixed(2)} → ${after.toFixed(4)} px in ${res.iterations} it; residual rotation ${maxRot.toFixed(4)}°, camera centres within ${(maxCentre * 100).toFixed(3)} cm of the truth`);
    expect(before).toBeGreaterThan(3);
    expect(after).toBeLessThan(0.05);
    expect(maxRot).toBeLessThan(0.05);
    // Landmarks were exact and the gauge keeps the first keyframe's median depth: the centres do not drift.
    expect(maxCentre).toBeLessThan(0.005);
  });

  it("§1: PnP with a pure-rotation prior error recovers the rotation with the translation staying at the truth", () => {
    const rng = createRng(41);
    const truth = poseFromCenter(mat3Multiply(rotationAxisAngle([0, 1, 0], 0.4), rotationAxisAngle([1, 0, 0], -0.2)), [5, -2, 4]);
    const n = 150;
    const pts = new Float64Array(n * 3), ox = new Float64Array(n), oy = new Float64Array(n);
    const out = new Float64Array(2);
    for (let i = 0; i < n; i++) {
      const depth = 1 + 3 * rng();
      const X = pointInFront(truth, (rng() - 0.5) * 1.2 * depth, (rng() - 0.5) * 0.9 * depth, depth);
      pts.set(X, i * 3);
      projectNormalized(truth.rotation, truth.translation, X[0], X[1], X[2], out);
      ox[i] = out[0];
      oy[i] = out[1];
    }
    // Prior: 3° off in rotation, translation exact.
    const prior: RigidTransform = { rotation: mat3Multiply(rotationAxisAngle([0.3, 1, -0.2], 0.052), truth.rotation), translation: Float64Array.from(truth.translation) };
    const res = refinePosePnP(prior, pts, ox, oy, n, { huber: 3 / F, inlierThreshold: 4 / F, maxIterations: 10, epsilon: 1e-10 });
    const dt = Math.hypot(res.pose.translation[0] - truth.translation[0], res.pose.translation[1] - truth.translation[1], res.pose.translation[2] - truth.translation[2]);
    console.log(`[convention] PnP rotation-only prior error: rotation ${deg(rotationDistance(res.pose.rotation, truth.rotation)).toFixed(5)}°, translation moved ${dt.toExponential(2)} u, ${res.iterations} it`);
    expect(deg(rotationDistance(res.pose.rotation, truth.rotation))).toBeLessThan(1e-3);
    expect(dt).toBeLessThan(1e-4);
    expectRotation(res.pose.rotation, 1e-12);
  });
});

function expectRotation(r: Mat3, tol: number): void {
  // R Rᵀ = I
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      const s = r[i * 3] * r[j * 3] + r[i * 3 + 1] * r[j * 3 + 1] + r[i * 3 + 2] * r[j * 3 + 2];
      expect(Math.abs(s - (i === j ? 1 : 0))).toBeLessThan(tol);
    }
  }
  const det = r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6]) + r[2] * (r[3] * r[7] - r[4] * r[6]);
  expect(Math.abs(det - 1)).toBeLessThan(tol);
}

function centre(p: RigidTransform): number[] {
  const r = p.rotation, t = p.translation;
  return [-(r[0] * t[0] + r[3] * t[1] + r[6] * t[2]), -(r[1] * t[0] + r[4] * t[1] + r[7] * t[2]), -(r[2] * t[0] + r[5] * t[1] + r[8] * t[2])];
}

function cameraAt(C: number[], yawDeg: number, pitchDeg: number): RigidTransform {
  return poseFromCenter(mat3Multiply(rotationAxisAngle([0, 1, 0], (yawDeg * Math.PI) / 180), rotationAxisAngle([1, 0, 0], (pitchDeg * Math.PI) / 180)), C);
}

/** Floor + wall, 5 cameras (the Phase 7 BA test's scene), with the chosen perturbation of the free keyframes. */
function buildProblem(seed: number, noisePx: number, opts: { rotationOnlyDeg?: number } = {}) {
  const rng = createRng(seed);
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
  const truePoses = [cameraAt([0, 0, 0], 0, 0), cameraAt([0.15, 0.02, 0.05], 4, 1), cameraAt([0.3, 0.03, 0.1], 8, -2), cameraAt([0.1, -0.05, 0.2], -5, 6), cameraAt([-0.2, 0.04, 0.15], -9, -4)];
  const observations: BAProblem["observations"] = [];
  const out = new Float64Array(2);
  for (let k = 0; k < truePoses.length; k++) {
    const p = truePoses[k];
    for (let i = 0; i < nL; i++) {
      if (!projectNormalized(p.rotation, p.translation, truth[i * 3], truth[i * 3 + 1], truth[i * 3 + 2], out)) continue;
      if (Math.abs(out[0]) > 0.6 || Math.abs(out[1]) > 0.45) continue;
      observations.push({ keyframe: k, landmark: i, x: out[0] + ((rng() - 0.5) * noisePx) / F, y: out[1] + ((rng() - 0.5) * noisePx) / F });
    }
  }
  const keyframes = truePoses.map((p, k) => {
    if (k === 0) return { pose: { rotation: Float64Array.from(p.rotation), translation: Float64Array.from(p.translation) }, fixed: true };
    const angle = ((opts.rotationOnlyDeg ?? 1.0) * Math.PI) / 180;
    const rot = mat3Multiply(rotationAxisAngle([rng() - 0.5, rng() - 0.5, rng() - 0.5], angle), p.rotation);
    const translation = opts.rotationOnlyDeg !== undefined
      ? Float64Array.from(p.translation)
      : new Float64Array([p.translation[0] + (rng() - 0.5) * 0.04, p.translation[1] + (rng() - 0.5) * 0.04, p.translation[2] + (rng() - 0.5) * 0.04]);
    return { pose: { rotation: rot, translation }, fixed: false };
  });
  const landmarks = Float64Array.from(truth);
  if (opts.rotationOnlyDeg === undefined) for (let i = 0; i < nL * 3; i++) landmarks[i] += (rng() - 0.5) * 0.04;
  const problem: BAProblem = { keyframes, landmarks, landmarkCount: nL, observations };
  return { problem, truth, truePoses };
}
