import { describe, expect, it } from "vitest";
import { estimateHomography, homographyTransferErrorSq, isDegenerateQuad } from "../../src/math/Homography";
import { mat3Invert, mat3Multiply, mat3TransformPoint, solveLinearSystem } from "../../src/math/Matrix";
import { createRng } from "../helpers/synthetic";

function applyH(h: Float64Array, xs: Float64Array, ys: Float64Array, n: number) {
  const ox = new Float64Array(n);
  const oy = new Float64Array(n);
  const pt = new Float64Array(2);
  for (let i = 0; i < n; i++) {
    mat3TransformPoint(h, xs[i], ys[i], pt);
    ox[i] = pt[0];
    oy[i] = pt[1];
  }
  return { ox, oy };
}

describe("Matrix", () => {
  it("solves a linear system", () => {
    const a = new Float64Array([2, 1, -1, -3, -1, 2, -2, 1, 2]);
    const b = new Float64Array([8, -11, -3]);
    expect(solveLinearSystem(a, b, 3)).toBe(true);
    expect(b[0]).toBeCloseTo(2, 9);
    expect(b[1]).toBeCloseTo(3, 9);
    expect(b[2]).toBeCloseTo(-1, 9);
  });

  it("inverts a 3x3 matrix", () => {
    const m = new Float64Array([1.2, 0.1, 30, -0.2, 0.9, -12, 0.0001, 0.0002, 1]);
    const inv = mat3Invert(m)!;
    const id = mat3Multiply(m, inv);
    for (let i = 0; i < 9; i++) expect(id[i]).toBeCloseTo(i % 4 === 0 ? 1 : 0, 9);
  });
});

describe("estimateHomography", () => {
  const H = new Float64Array([1.05, -0.08, 12.3, 0.06, 0.97, -7.1, 0.00012, -0.00008, 1]);

  it("recovers an exact homography from 4 points", () => {
    const xs = new Float64Array([50, 600, 580, 40]);
    const ys = new Float64Array([40, 60, 440, 420]);
    const { ox, oy } = applyH(H, xs, ys, 4);
    const est = estimateHomography(xs, ys, ox, oy, null, 4)!;
    expect(est).not.toBeNull();
    for (let i = 0; i < 9; i++) expect(est[i]).toBeCloseTo(H[i], 7);
  });

  it("recovers a homography from many noisy points (least squares)", () => {
    const rng = createRng(7);
    const n = 120;
    const xs = new Float64Array(n);
    const ys = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      xs[i] = rng() * 640;
      ys[i] = rng() * 480;
    }
    const { ox, oy } = applyH(H, xs, ys, n);
    for (let i = 0; i < n; i++) {
      ox[i] += (rng() - 0.5) * 0.4;
      oy[i] += (rng() - 0.5) * 0.4;
    }
    const est = estimateHomography(xs, ys, ox, oy, null, n)!;
    expect(est).not.toBeNull();
    let maxErr = 0;
    for (let i = 0; i < n; i++) {
      maxErr = Math.max(maxErr, Math.sqrt(homographyTransferErrorSq(est, xs[i], ys[i], ox[i], oy[i])));
    }
    expect(maxErr).toBeLessThan(0.6);
  });

  it("works with an index subset", () => {
    const xs = new Float64Array([0, 0, 50, 600, 580, 40, 0]);
    const ys = new Float64Array([0, 0, 40, 60, 440, 420, 0]);
    const { ox, oy } = applyH(H, xs, ys, 7);
    const idx = new Int32Array([2, 3, 4, 5]);
    const est = estimateHomography(xs, ys, ox, oy, idx, 4)!;
    for (let i = 0; i < 9; i++) expect(est[i]).toBeCloseTo(H[i], 7);
  });

  it("rejects collinear configurations", () => {
    const xs = new Float64Array([0, 10, 20, 30]);
    const ys = new Float64Array([0, 10, 20, 30]);
    const idx = new Int32Array([0, 1, 2, 3]);
    expect(isDegenerateQuad(xs, ys, idx)).toBe(true);
    const good = new Float64Array([0, 100, 100, 0]);
    const goodY = new Float64Array([0, 0, 100, 100]);
    expect(isDegenerateQuad(good, goodY, idx)).toBe(false);
  });
});
