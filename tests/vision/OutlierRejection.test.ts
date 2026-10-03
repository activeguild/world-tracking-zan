import { describe, expect, it } from "vitest";
import { DEFAULT_CONFIG } from "../../src/ar/ARConfig";
import { homographyTransferErrorSq } from "../../src/math/Homography";
import { mat3TransformPoint } from "../../src/math/Matrix";
import { createRng, ransacHomography } from "../../src/vision/OutlierRejection";

function makeCorrespondences(seed: number, n: number, outlierRatio: number, noise: number) {
  const rng = createRng(seed);
  const H = new Float64Array([1.02, -0.03, 4.5, 0.025, 0.99, -2.0, 0.00004, -0.00002, 1]);
  const x1 = new Float32Array(n);
  const y1 = new Float32Array(n);
  const x2 = new Float32Array(n);
  const y2 = new Float32Array(n);
  const truth = new Uint8Array(n);
  const pt = new Float64Array(2);
  for (let i = 0; i < n; i++) {
    x1[i] = 20 + rng() * 600;
    y1[i] = 20 + rng() * 440;
    mat3TransformPoint(H, x1[i], y1[i], pt);
    if (rng() < outlierRatio) {
      // Gross error: random displacement of 10–60 px.
      const ang = rng() * Math.PI * 2;
      const mag = 10 + rng() * 50;
      x2[i] = pt[0] + Math.cos(ang) * mag;
      y2[i] = pt[1] + Math.sin(ang) * mag;
      truth[i] = 0;
    } else {
      x2[i] = pt[0] + (rng() - 0.5) * 2 * noise;
      y2[i] = pt[1] + (rng() - 0.5) * 2 * noise;
      truth[i] = 1;
    }
  }
  return { H, x1, y1, x2, y2, truth, rng };
}

describe("ransacHomography", () => {
  it("separates inliers from 30% gross outliers", () => {
    const n = 200;
    const d = makeCorrespondences(1, n, 0.3, 0.3);
    const r = ransacHomography(d.x1, d.y1, d.x2, d.y2, n, DEFAULT_CONFIG.ransac, createRng(123));
    expect(r.homography).not.toBeNull();

    let truePos = 0;
    let falsePos = 0;
    let trueTotal = 0;
    for (let i = 0; i < n; i++) {
      if (d.truth[i]) {
        trueTotal++;
        if (r.inliers[i]) truePos++;
      } else if (r.inliers[i]) {
        falsePos++;
      }
    }
    expect(truePos / trueTotal).toBeGreaterThan(0.97);
    expect(falsePos).toBeLessThanOrEqual(2);
    expect(r.inlierCount).toBe(truePos + falsePos);
    expect(r.meanError).toBeLessThan(0.5);

    // Recovered homography reproduces the ground-truth mapping.
    for (let i = 0; i < n; i += 7) {
      if (!d.truth[i]) continue;
      const e = Math.sqrt(homographyTransferErrorSq(r.homography!, d.x1[i], d.y1[i], d.x2[i], d.y2[i]));
      expect(e).toBeLessThan(1.0);
    }
  });

  it("keeps everything when all correspondences are consistent", () => {
    const n = 80;
    const d = makeCorrespondences(2, n, 0, 0.2);
    const r = ransacHomography(d.x1, d.y1, d.x2, d.y2, n, DEFAULT_CONFIG.ransac, createRng(5));
    expect(r.inlierCount).toBeGreaterThanOrEqual(n - 1);
    // Adaptive termination should stop early.
    expect(r.iterations).toBeLessThan(DEFAULT_CONFIG.ransac.maxIterations);
  });

  it("handles the static-camera case (identity motion)", () => {
    const n = 60;
    const rng = createRng(9);
    const x = new Float32Array(n);
    const y = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      x[i] = rng() * 640;
      y[i] = rng() * 480;
    }
    const r = ransacHomography(x, y, x, y, n, DEFAULT_CONFIG.ransac, createRng(1));
    expect(r.inlierCount).toBe(n);
    expect(r.homography).not.toBeNull();
    const h = r.homography!;
    expect(h[0]).toBeCloseTo(1, 4);
    expect(h[4]).toBeCloseTo(1, 4);
    expect(h[2]).toBeCloseTo(0, 3);
    expect(h[5]).toBeCloseTo(0, 3);
  });

  it("does not reject anything when there are too few correspondences", () => {
    const n = 6;
    const d = makeCorrespondences(3, n, 0.5, 0.2);
    const r = ransacHomography(d.x1, d.y1, d.x2, d.y2, n, DEFAULT_CONFIG.ransac, createRng(1));
    expect(r.homography).toBeNull();
    expect(r.inlierCount).toBe(n);
    for (let i = 0; i < n; i++) expect(r.inliers[i]).toBe(1);
  });

  it("is deterministic for a seeded RNG", () => {
    const n = 150;
    const d = makeCorrespondences(4, n, 0.25, 0.3);
    const a = ransacHomography(d.x1, d.y1, d.x2, d.y2, n, DEFAULT_CONFIG.ransac, createRng(77));
    const b = ransacHomography(d.x1, d.y1, d.x2, d.y2, n, DEFAULT_CONFIG.ransac, createRng(77));
    expect(Array.from(a.inliers)).toEqual(Array.from(b.inliers));
  });
});
