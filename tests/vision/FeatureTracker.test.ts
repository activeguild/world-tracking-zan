import { describe, expect, it } from "vitest";
import { DEFAULT_CONFIG } from "../../src/ar/ARConfig";
import { FeatureDetector } from "../../src/vision/FeatureDetector";
import { FeatureTracker, TrackStatus } from "../../src/vision/FeatureTracker";
import { ImagePyramid } from "../../src/vision/ImagePyramid";
import { createRng, makeTexture, median, similarityH, translateImage, warpImage } from "../helpers/synthetic";

const W = 640;
const H = 480;

function detectPoints(img: Uint8Array, wanted = 300): Float32Array {
  const det = new FeatureDetector(W, H, { ...DEFAULT_CONFIG.features, borderMargin: 40 });
  const corners = det.detect(img, { wanted });
  const pts = new Float32Array(corners.length * 2);
  corners.forEach((c, i) => {
    pts[i * 2] = c.x;
    pts[i * 2 + 1] = c.y;
  });
  return pts;
}

function buildPyr(img: Uint8Array): ImagePyramid {
  const p = new ImagePyramid(W, H, DEFAULT_CONFIG.tracker.pyramidLevels);
  p.build(img);
  return p;
}

describe("FeatureTracker (pyramidal LK + forward-backward)", () => {
  const tex = makeTexture(W, H, createRng(42));

  it("tracks a sub-pixel translation accurately", () => {
    const dx = 3.3;
    const dy = -2.7;
    const moved = translateImage(tex, W, H, dx, dy);
    const pts = detectPoints(tex);
    const n = pts.length / 2;
    expect(n).toBeGreaterThan(100);

    const tracker = new FeatureTracker(DEFAULT_CONFIG.tracker);
    const res = tracker.track(buildPyr(tex), buildPyr(moved), pts, n);

    const errs: number[] = [];
    for (let i = 0; i < n; i++) {
      if (res.status[i] !== TrackStatus.OK) continue;
      errs.push(Math.hypot(res.positions[i * 2] - (pts[i * 2] + dx), res.positions[i * 2 + 1] - (pts[i * 2 + 1] + dy)));
    }
    expect(res.okCount / n).toBeGreaterThan(0.9);
    expect(median(errs)).toBeLessThan(0.25);
    expect(errs.filter((e) => e > 1).length / errs.length).toBeLessThan(0.02);
  });

  it("handles large motion through the pyramid", () => {
    const dx = 22;
    const dy = 15;
    const moved = translateImage(tex, W, H, dx, dy);
    const pts = detectPoints(tex);
    const n = pts.length / 2;
    const tracker = new FeatureTracker(DEFAULT_CONFIG.tracker);
    const res = tracker.track(buildPyr(tex), buildPyr(moved), pts, n);

    const errs: number[] = [];
    for (let i = 0; i < n; i++) {
      if (res.status[i] !== TrackStatus.OK) continue;
      errs.push(Math.hypot(res.positions[i * 2] - (pts[i * 2] + dx), res.positions[i * 2 + 1] - (pts[i * 2 + 1] + dy)));
    }
    expect(res.okCount / n).toBeGreaterThan(0.7);
    expect(median(errs)).toBeLessThan(0.5);
  });

  it("tracks under a small rotation + scale", () => {
    const Hm = similarityH(W / 2, H / 2, 0.02, 1.015, 1.5, -1.0);
    const moved = warpImage(tex, W, H, Hm);
    const pts = detectPoints(tex);
    const n = pts.length / 2;
    const tracker = new FeatureTracker(DEFAULT_CONFIG.tracker);
    const res = tracker.track(buildPyr(tex), buildPyr(moved), pts, n);

    const errs: number[] = [];
    const pt = new Float64Array(2);
    for (let i = 0; i < n; i++) {
      if (res.status[i] !== TrackStatus.OK) continue;
      const x = pts[i * 2];
      const y = pts[i * 2 + 1];
      const w = Hm[6] * x + Hm[7] * y + Hm[8];
      pt[0] = (Hm[0] * x + Hm[1] * y + Hm[2]) / w;
      pt[1] = (Hm[3] * x + Hm[4] * y + Hm[5]) / w;
      errs.push(Math.hypot(res.positions[i * 2] - pt[0], res.positions[i * 2 + 1] - pt[1]));
    }
    expect(res.okCount / n).toBeGreaterThan(0.8);
    expect(median(errs)).toBeLessThan(0.5);
  });

  it("returns identical positions for a static scene", () => {
    const pts = detectPoints(tex);
    const n = pts.length / 2;
    const tracker = new FeatureTracker(DEFAULT_CONFIG.tracker);
    const res = tracker.track(buildPyr(tex), buildPyr(tex), pts, n);
    expect(res.okCount / n).toBeGreaterThan(0.95);
    for (let i = 0; i < n; i++) {
      if (res.status[i] !== TrackStatus.OK) continue;
      expect(Math.abs(res.positions[i * 2] - pts[i * 2])).toBeLessThan(0.05);
      expect(Math.abs(res.positions[i * 2 + 1] - pts[i * 2 + 1])).toBeLessThan(0.05);
    }
  });

  it("rejects occluded points via forward-backward check", () => {
    // Move the whole image, then paste a different texture over a region.
    const moved = translateImage(tex, W, H, 4, 2);
    const other = makeTexture(W, H, createRng(99));
    const rx0 = 200;
    const ry0 = 120;
    const rx1 = 440;
    const ry1 = 360;
    for (let y = ry0; y < ry1; y++) {
      for (let x = rx0; x < rx1; x++) moved[y * W + x] = other[y * W + x];
    }
    const pts = detectPoints(tex);
    const n = pts.length / 2;
    const tracker = new FeatureTracker(DEFAULT_CONFIG.tracker);
    const res = tracker.track(buildPyr(tex), buildPyr(moved), pts, n);

    let inOk = 0;
    let inTotal = 0;
    let outOk = 0;
    let outTotal = 0;
    for (let i = 0; i < n; i++) {
      const x = pts[i * 2];
      const y = pts[i * 2 + 1];
      const inside = x > rx0 + 12 && x < rx1 - 12 && y > ry0 + 12 && y < ry1 - 12;
      const ok = res.status[i] === TrackStatus.OK;
      if (inside) {
        inTotal++;
        if (ok) inOk++;
      } else {
        outTotal++;
        if (ok) outOk++;
      }
    }
    expect(inTotal).toBeGreaterThan(20);
    expect(outTotal).toBeGreaterThan(50);
    expect(inOk / inTotal).toBeLessThan(0.25);
    expect(outOk / outTotal).toBeGreaterThan(0.8);
  });

  it("rejects texture-less points", () => {
    const flat = new Uint8Array(W * H).fill(100);
    const pts = new Float32Array([100, 100, 300, 200, 500, 400]);
    const tracker = new FeatureTracker(DEFAULT_CONFIG.tracker);
    const res = tracker.track(buildPyr(flat), buildPyr(flat), pts, 3);
    expect(res.okCount).toBe(0);
    for (let i = 0; i < 3; i++) expect(res.status[i]).toBe(TrackStatus.LOW_TEXTURE);
  });

  it("reports out-of-bounds for points near the border", () => {
    const pts = new Float32Array([2, 2, W - 2, H - 2]);
    const tracker = new FeatureTracker(DEFAULT_CONFIG.tracker);
    const res = tracker.track(buildPyr(tex), buildPyr(tex), pts, 2);
    expect(res.status[0]).toBe(TrackStatus.OUT_OF_BOUNDS);
    expect(res.status[1]).toBe(TrackStatus.OUT_OF_BOUNDS);
  });

  it("tracks 300 points at 640x480 within budget", () => {
    const moved = translateImage(tex, W, H, 2, 1);
    const pts = detectPoints(tex, 300);
    const n = pts.length / 2;
    const tracker = new FeatureTracker(DEFAULT_CONFIG.tracker);
    const a = buildPyr(tex);
    const b = buildPyr(moved);
    tracker.track(a, b, pts, n); // warm-up
    const t0 = performance.now();
    const N = 5;
    for (let i = 0; i < N; i++) tracker.track(a, b, pts, n);
    const ms = (performance.now() - t0) / N;
    // Includes the backward pass. Generous for CI machines.
    expect(ms).toBeLessThan(150);
  });
});
