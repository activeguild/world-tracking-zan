import { describe, expect, it } from "vitest";
import { det3, svd3, symmetricEigen, transpose3 } from "../../src/math/Decomposition";
import { mat3Multiply } from "../../src/math/Matrix";
import {
  composeTransforms,
  invertTransform,
  projectToRotation,
  rotationAngle,
  rotationAxisAngle,
  rotationDistance,
  rotationToEulerDeg,
  rotationToQuaternion,
  rotationX,
  rotationY,
  rotationZ,
} from "../../src/math/Pose";
import { createRng } from "../helpers/synthetic";

describe("symmetricEigen", () => {
  it("diagonalizes a symmetric matrix", () => {
    const a = new Float64Array([4, 1, 2, 1, 3, 0, 2, 0, 5]);
    const { values, vectors } = symmetricEigen(a, 3);
    expect(values[0]).toBeLessThanOrEqual(values[1]);
    expect(values[1]).toBeLessThanOrEqual(values[2]);
    // A v = λ v for each column
    for (let c = 0; c < 3; c++) {
      for (let r = 0; r < 3; r++) {
        const av = a[r * 3] * vectors[c] + a[r * 3 + 1] * vectors[3 + c] + a[r * 3 + 2] * vectors[6 + c];
        expect(av).toBeCloseTo(values[c] * vectors[r * 3 + c], 8);
      }
    }
    // trace preserved
    expect(values[0] + values[1] + values[2]).toBeCloseTo(12, 8);
  });

  it("finds the null vector of a rank-deficient 9x9 Gram matrix", () => {
    const rng = createRng(1);
    const truth = new Float64Array(9);
    for (let i = 0; i < 9; i++) truth[i] = rng() - 0.5;
    const ata = new Float64Array(81);
    // Build rows orthogonal to `truth`.
    for (let k = 0; k < 20; k++) {
      const row = new Float64Array(9);
      for (let i = 0; i < 9; i++) row[i] = rng() - 0.5;
      const d = row.reduce((s, v, i) => s + v * truth[i], 0) / truth.reduce((s, v) => s + v * v, 0);
      for (let i = 0; i < 9; i++) row[i] -= d * truth[i];
      for (let i = 0; i < 9; i++) for (let j = 0; j < 9; j++) ata[i * 9 + j] += row[i] * row[j];
    }
    const { values, vectors } = symmetricEigen(ata, 9);
    expect(Math.abs(values[0])).toBeLessThan(1e-9);
    const nv = Array.from({ length: 9 }, (_, r) => vectors[r * 9]);
    const dot = nv.reduce((s, v, i) => s + v * truth[i], 0);
    const norm = Math.sqrt(truth.reduce((s, v) => s + v * v, 0));
    expect(Math.abs(dot) / norm).toBeCloseTo(1, 6);
  });
});

describe("svd3", () => {
  it("reconstructs the matrix and yields orthonormal factors", () => {
    const m = new Float64Array([0.9, -0.3, 12, 0.2, 1.1, -5, 0.001, 0.002, 1]);
    const { u, s, v } = svd3(m);
    expect(s[0]).toBeGreaterThanOrEqual(s[1]);
    expect(s[1]).toBeGreaterThanOrEqual(s[2]);
    const d = new Float64Array([s[0], 0, 0, 0, s[1], 0, 0, 0, s[2]]);
    const rec = mat3Multiply(mat3Multiply(u, d), transpose3(v));
    for (let i = 0; i < 9; i++) expect(rec[i]).toBeCloseTo(m[i], 7);
    const utu = mat3Multiply(transpose3(u), u);
    const vtv = mat3Multiply(transpose3(v), v);
    for (let i = 0; i < 9; i++) {
      expect(utu[i]).toBeCloseTo(i % 4 === 0 ? 1 : 0, 7);
      expect(vtv[i]).toBeCloseTo(i % 4 === 0 ? 1 : 0, 7);
    }
  });

  it("handles rank-2 matrices (essential-like)", () => {
    const r = rotationAxisAngle([0.3, 1, 0.2], 0.4);
    const t = [0.2, -0.5, 1];
    const tx = new Float64Array([0, -t[2], t[1], t[2], 0, -t[0], -t[1], t[0], 0]);
    const e = mat3Multiply(tx, r);
    const { u, s } = svd3(e);
    expect(s[2]).toBeLessThan(1e-7);
    expect(s[0]).toBeCloseTo(s[1], 7);
    const utu = mat3Multiply(transpose3(u), u);
    for (let i = 0; i < 9; i++) expect(utu[i]).toBeCloseTo(i % 4 === 0 ? 1 : 0, 7);
    expect(Math.abs(det3(u))).toBeCloseTo(1, 7);
  });
});

describe("Pose utilities", () => {
  it("axis-angle rotations are proper and have the right angle", () => {
    const r = rotationAxisAngle([0, 1, 0], 0.7);
    expect(det3(r)).toBeCloseTo(1, 10);
    expect(rotationAngle(r)).toBeCloseTo(0.7, 10);
    const ry = rotationY(0.7);
    for (let i = 0; i < 9; i++) expect(r[i]).toBeCloseTo(ry[i], 10);
  });

  it("projectToRotation recovers a rotation from a perturbed matrix", () => {
    const r = mat3Multiply(rotationX(0.2), rotationZ(-0.5));
    const noisy = Float64Array.from(r, (v, i) => v * 1.3 + (i % 2 ? 0.01 : -0.01));
    const p = projectToRotation(noisy);
    expect(det3(p)).toBeCloseTo(1, 8);
    expect(rotationDistance(p, r)).toBeLessThan(0.02);
  });

  it("quaternion round trip and euler angles", () => {
    const yaw = 0.3;
    const pitch = -0.2;
    const roll = 0.1;
    const r = mat3Multiply(mat3Multiply(rotationY(yaw), rotationX(pitch)), rotationZ(roll));
    const e = rotationToEulerDeg(r);
    expect(e.yaw).toBeCloseTo((yaw * 180) / Math.PI, 6);
    expect(e.pitch).toBeCloseTo((pitch * 180) / Math.PI, 6);
    expect(e.roll).toBeCloseTo((roll * 180) / Math.PI, 6);
    const q = rotationToQuaternion(r);
    const n = Math.hypot(q[0], q[1], q[2], q[3]);
    expect(n).toBeCloseTo(1, 10);
    // Rotation angle from quaternion w
    expect(2 * Math.acos(Math.abs(q[3]))).toBeCloseTo(rotationAngle(r), 8);
  });

  it("compose / invert are consistent", () => {
    const a = { rotation: rotationY(0.4), translation: new Float64Array([1, 0, 0.5]) };
    const b = { rotation: rotationX(-0.3), translation: new Float64Array([0, 2, -1]) };
    const ab = composeTransforms(b, a);
    const inv = invertTransform(ab);
    const id = composeTransforms(inv, ab);
    for (let i = 0; i < 9; i++) expect(id.rotation[i]).toBeCloseTo(i % 4 === 0 ? 1 : 0, 9);
    for (let i = 0; i < 3; i++) expect(id.translation[i]).toBeCloseTo(0, 9);
  });
});
