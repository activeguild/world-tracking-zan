import { describe, expect, it } from "vitest";
import { mat3Multiply } from "../../src/math/Matrix";
import { refinePosePnP } from "../../src/math/PnP";
import { rotationAxisAngle, rotationDistance, type RigidTransform } from "../../src/math/Pose";
import { projectNormalized, reprojectionJacobian } from "../../src/math/Reprojection";
import { gauss, poseFromCenter } from "../helpers/scene";
import { createRng } from "../helpers/synthetic";

/**
 * 改善指示書 §4.1–4.2: the analytic Jacobian of the reprojection must be the
 * derivative of the update that is actually applied,
 *
 *   R' = exp(δω) R,   t' = t + δt,   X' = X + δX,   X_cam = R X + t.
 *
 * Under that update ∂X_cam/∂δω = −[R X]× (the rotation acts on R X, not on
 * t). The previous inline Jacobians used −[X_cam]×, which is the derivative
 * of the SE(3) left perturbation (t' = exp(δω) t + δt) that nobody applied:
 * the two differ by δω × t, i.e. they agree only while the camera sits near
 * the map origin.
 */
const F = 640;
const deg = (rad: number) => (rad * 180) / Math.PI;

function applyUpdate(pose: RigidTransform, dw: ArrayLike<number>, dt: ArrayLike<number>): RigidTransform {
  const a = Math.hypot(dw[0], dw[1], dw[2]);
  const dR = a > 0 ? rotationAxisAngle([dw[0] / a, dw[1] / a, dw[2] / a], a) : new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  return {
    rotation: mat3Multiply(dR, pose.rotation),
    translation: new Float64Array([pose.translation[0] + dt[0], pose.translation[1] + dt[1], pose.translation[2] + dt[2]]),
  };
}

/** Numerical Jacobian of (u, v) w.r.t. [δω, δt, δX] by central differences. */
function numericJacobian(pose: RigidTransform, X: number[], h: number): { ju: number[]; jv: number[] } {
  const ju: number[] = [], jv: number[] = [];
  const out = new Float64Array(2);
  const at = (dw: number[], dt: number[], dX: number[]): [number, number] => {
    const p = applyUpdate(pose, dw, dt);
    expect(projectNormalized(p.rotation, p.translation, X[0] + dX[0], X[1] + dX[1], X[2] + dX[2], out)).toBe(true);
    return [out[0], out[1]];
  };
  for (let k = 0; k < 9; k++) {
    const e = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    e[k] = h;
    const plus = at(e.slice(0, 3), e.slice(3, 6), e.slice(6, 9));
    e[k] = -h;
    const minus = at(e.slice(0, 3), e.slice(3, 6), e.slice(6, 9));
    ju.push((plus[0] - minus[0]) / (2 * h));
    jv.push((plus[1] - minus[1]) / (2 * h));
  }
  return { ju, jv };
}

/** The pre-fix rotation columns (−[X_cam]×), kept here only to show what they get wrong. */
function oldRotationColumns(pose: RigidTransform, X: number[]): { ju: number[]; jv: number[] } {
  const r = pose.rotation, t = pose.translation;
  const xc = r[0] * X[0] + r[1] * X[1] + r[2] * X[2] + t[0];
  const yc = r[3] * X[0] + r[4] * X[1] + r[5] * X[2] + t[1];
  const zc = r[6] * X[0] + r[7] * X[1] + r[8] * X[2] + t[2];
  const iz = 1 / zc, u = xc * iz, v = yc * iz;
  return { ju: [-u * iz * yc, iz * zc + u * iz * xc, -iz * yc], jv: [-iz * zc - v * iz * yc, v * iz * xc, iz * xc] };
}

describe("reprojection Jacobian (改善指示書 §4.1–4.2)", () => {
  it("matches central finite differences for rotation, translation and landmark, near and far from the map origin", () => {
    const rng = createRng(11);
    const ju = new Float64Array(9), jv = new Float64Array(9);
    const h = 1e-6;
    // Camera centers from the origin out to 12 map units (on device the camera
    // walks several units from the initialization view); points in front at
    // depths 0.3–8, including oblique and off-center ones.
    const centers = [
      [0, 0, 0],
      [0.3, -0.2, 0.1],
      [3, 1, -2],
      [-8, 2, 5],
      [12, -4, 3],
    ];
    let worstRel = 0;
    for (const C of centers) {
      for (let trial = 0; trial < 6; trial++) {
        const axis = [rng() - 0.5, rng() - 0.5, rng() - 0.5];
        const pose = poseFromCenter(rotationAxisAngle(axis, (rng() - 0.5) * 2.5), C);
        // A point in front of the camera: camera-frame coordinates picked, mapped back to the map.
        const depth = 0.3 + rng() * 7.7;
        const xc = (rng() - 0.5) * 1.6 * depth, yc = (rng() - 0.5) * 1.2 * depth;
        const r = pose.rotation, t = pose.translation;
        const dx = xc - t[0], dy = yc - t[1], dz = depth - t[2];
        const X = [r[0] * dx + r[3] * dy + r[6] * dz, r[1] * dx + r[4] * dy + r[7] * dz, r[2] * dx + r[5] * dy + r[8] * dz];
        expect(reprojectionJacobian(r, t, X[0], X[1], X[2], ju, jv)).toBe(true);
        const num = numericJacobian(pose, X, h);
        // Relative tolerance against the column scale: central differences
        // are O(h²) exact, so the mismatch is dominated by floating point
        // round-off ≈ |J| · 1e-16 / h plus curvature · h².
        for (let k = 0; k < 9; k++) {
          const scale = Math.max(1, Math.abs(num.ju[k]), Math.abs(num.jv[k]));
          const eu = Math.abs(ju[k] - num.ju[k]) / scale;
          const ev = Math.abs(jv[k] - num.jv[k]) / scale;
          worstRel = Math.max(worstRel, eu, ev);
          expect(eu).toBeLessThan(1e-6);
          expect(ev).toBeLessThan(1e-6);
        }
      }
    }
    console.log(`[jacobian] worst relative mismatch vs central differences: ${worstRel.toExponential(2)}`);
  });

  it("the previous −[X_cam]× rotation columns are only right at the map origin: the error grows with |t|", () => {
    const rng = createRng(12);
    const X = [0.4, -0.2, 3.0];
    const rows: string[] = [];
    let prevErr = -1;
    for (const d of [0, 0.5, 2, 6, 12]) {
      const pose = poseFromCenter(rotationAxisAngle([0.2, 1, 0.1], 0.4), [d * 0.6, -d * 0.3, d * 0.2]);
      // Keep the point in front: shift it with the camera.
      const C = [d * 0.6, -d * 0.3, d * 0.2];
      const Xd = [X[0] + C[0], X[1] + C[1], X[2] + C[2]];
      const num = numericJacobian(pose, Xd, 1e-6);
      const old = oldRotationColumns(pose, Xd);
      let err = 0, scale = 0;
      for (let k = 0; k < 3; k++) {
        err = Math.max(err, Math.abs(old.ju[k] - num.ju[k]), Math.abs(old.jv[k] - num.jv[k]));
        scale = Math.max(scale, Math.abs(num.ju[k]), Math.abs(num.jv[k]));
      }
      const rel = err / Math.max(scale, 1e-9);
      rows.push(`|C| ${Math.hypot(C[0], C[1], C[2]).toFixed(1)} → relative error ${rel.toFixed(3)}`);
      if (d === 0) expect(rel).toBeLessThan(1e-6);
      else {
        expect(rel).toBeGreaterThan(prevErr);
        if (d >= 6) expect(rel).toBeGreaterThan(0.5);
      }
      prevErr = rel;
      void rng;
    }
    console.log(`[jacobian] old rotation columns: ${rows.join(", ")}`);
  });

  it("PnP converges within the engine's 10 iterations when the camera is far from the map origin", () => {
    // On device the camera center is 5–15 map units from the initialization
    // view; with the inconsistent Jacobian the LM step was not a descent
    // direction there and PnP hit the iteration cap with a biased pose.
    const rng = createRng(13);
    const opts = { huber: 3 / F, inlierThreshold: 4 / F, maxIterations: 10, epsilon: 1e-8 };
    let maxRot = 0, maxTrans = 0, maxErr = 0, notConverged = 0;
    for (let trial = 0; trial < 12; trial++) {
      const C = [8 + 4 * rng(), -3 + 6 * rng(), 2 + 4 * rng()];
      const truth = poseFromCenter(rotationAxisAngle([rng() - 0.5, 1, rng() - 0.5], (rng() - 0.5) * 2), C);
      // Points 1–4 units in front of this camera, mapped to the map frame.
      const n = 120;
      const pts = new Float64Array(n * 3);
      const r = truth.rotation, t = truth.translation;
      for (let i = 0; i < n; i++) {
        const depth = 1 + 3 * rng();
        const xc = (rng() - 0.5) * 1.2 * depth, yc = (rng() - 0.5) * 0.9 * depth;
        const dx = xc - t[0], dy = yc - t[1], dz = depth - t[2];
        pts[i * 3] = r[0] * dx + r[3] * dy + r[6] * dz;
        pts[i * 3 + 1] = r[1] * dx + r[4] * dy + r[7] * dz;
        pts[i * 3 + 2] = r[2] * dx + r[5] * dy + r[8] * dz;
      }
      const ox = new Float64Array(n), oy = new Float64Array(n);
      const out = new Float64Array(2);
      for (let i = 0; i < n; i++) {
        projectNormalized(r, t, pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2], out);
        ox[i] = out[0] + (gauss(rng) * 0.3) / F;
        oy[i] = out[1] + (gauss(rng) * 0.3) / F;
      }
      // Prior: 2° off and 3% of the depth off, like one fast frame.
      const prior: RigidTransform = {
        rotation: mat3Multiply(rotationAxisAngle([rng() - 0.5, rng() - 0.5, rng() - 0.5], 0.035), truth.rotation),
        translation: new Float64Array([t[0] + 0.06, t[1] - 0.04, t[2] + 0.05]),
      };
      const res = refinePosePnP(prior, pts, ox, oy, n, opts);
      maxRot = Math.max(maxRot, deg(rotationDistance(res.pose.rotation, truth.rotation)));
      maxTrans = Math.max(maxTrans, Math.hypot(res.pose.translation[0] - t[0], res.pose.translation[1] - t[1], res.pose.translation[2] - t[2]));
      maxErr = Math.max(maxErr, res.meanError * F);
      if (!res.converged) notConverged++;
    }
    console.log(`[jacobian] PnP far from origin: worst rotation ${maxRot.toFixed(3)}°, translation ${maxTrans.toFixed(4)} u, mean error ${maxErr.toFixed(2)} px, not converged ${notConverged}/12`);
    expect(maxRot).toBeLessThan(0.1);
    expect(maxTrans).toBeLessThan(0.01);
    expect(maxErr).toBeLessThan(0.5);
    expect(notConverged).toBe(0);
  });
});
