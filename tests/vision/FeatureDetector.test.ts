import { describe, expect, it } from "vitest";
import { DEFAULT_CONFIG } from "../../src/ar/ARConfig";
import { FeatureDetector } from "../../src/vision/FeatureDetector";
import { createRng, makeSquares, makeTexture } from "../helpers/synthetic";

const W = 320;
const H = 240;

describe("FeatureDetector (FAST-9)", () => {
  it("finds the corners of black squares on a white background", () => {
    const squares = [
      { x: 40, y: 30, size: 40 },
      { x: 150, y: 60, size: 60 },
      { x: 220, y: 150, size: 50 },
      { x: 60, y: 140, size: 45 },
    ];
    const { image, corners } = makeSquares(W, H, squares);
    const det = new FeatureDetector(W, H, { ...DEFAULT_CONFIG.features, minDistance: 5, borderMargin: 4 });
    const found = det.detect(image, { wanted: 100 });

    // Every true corner should have a detection within 2.5 px.
    let hit = 0;
    for (const c of corners) {
      const ok = found.some((f) => Math.hypot(f.x - c.x, f.y - c.y) <= 2.5);
      if (ok) hit++;
    }
    expect(hit).toBeGreaterThanOrEqual(corners.length - 1);
    // And nothing far from any corner (edges of squares are not corners).
    for (const f of found) {
      const d = Math.min(...corners.map((c) => Math.hypot(f.x - c.x, f.y - c.y)));
      expect(d).toBeLessThanOrEqual(3);
    }
  });

  it("returns at most `wanted` corners sorted by score and respects minDistance", () => {
    const tex = makeTexture(W, H, createRng(3));
    const cfg = { ...DEFAULT_CONFIG.features, minDistance: 12, maxPerCell: 50 };
    const det = new FeatureDetector(W, H, cfg);
    const found = det.detect(tex, { wanted: 60 });
    expect(found.length).toBeGreaterThan(20);
    expect(found.length).toBeLessThanOrEqual(60);
    for (let i = 1; i < found.length; i++) expect(found[i - 1].score).toBeGreaterThanOrEqual(found[i].score);
    for (let i = 0; i < found.length; i++) {
      for (let j = i + 1; j < found.length; j++) {
        const d = Math.hypot(found[i].x - found[j].x, found[i].y - found[j].y);
        expect(d).toBeGreaterThanOrEqual(cfg.minDistance);
      }
    }
  });

  it("distributes corners across the grid (per-cell quota)", () => {
    const tex = makeTexture(W, H, createRng(11));
    const cfg = { ...DEFAULT_CONFIG.features, minDistance: 4, gridCols: 4, gridRows: 3, maxPerCell: 5 };
    const det = new FeatureDetector(W, H, cfg);
    const found = det.detect(tex, { wanted: 200 });
    const counts = new Map<number, number>();
    for (const f of found) {
      const cx = Math.min(3, Math.floor(f.x / (W / 4)));
      const cy = Math.min(2, Math.floor(f.y / (H / 3)));
      const k = cy * 4 + cx;
      counts.set(k, (counts.get(k) ?? 0) + 1);
    }
    for (const v of counts.values()) expect(v).toBeLessThanOrEqual(5);
    expect(counts.size).toBeGreaterThanOrEqual(8);
  });

  it("honours an occupancy mask", () => {
    const tex = makeTexture(W, H, createRng(5));
    const det = new FeatureDetector(W, H, { ...DEFAULT_CONFIG.features, minDistance: 4 });
    const mask = new Uint8Array(W * H);
    // Block the left half.
    for (let y = 0; y < H; y++) mask.fill(1, y * W, y * W + W / 2);
    const found = det.detect(tex, { wanted: 300, mask });
    expect(found.length).toBeGreaterThan(10);
    for (const f of found) expect(f.x).toBeGreaterThanOrEqual(W / 2);
  });

  it("adapts its threshold downwards on low-texture images", () => {
    const flat = new Uint8Array(W * H).fill(128);
    const det = new FeatureDetector(W, H, DEFAULT_CONFIG.features);
    const start = det.threshold;
    det.detect(flat, { wanted: 100 });
    det.detect(flat, { wanted: 100 });
    expect(det.threshold).toBeLessThan(start);
    expect(det.threshold).toBeGreaterThanOrEqual(DEFAULT_CONFIG.features.fastThresholdMin);
  });

  it("is fast enough at 640x480", () => {
    const tex = makeTexture(640, 480, createRng(9));
    const det = new FeatureDetector(640, 480, DEFAULT_CONFIG.features);
    det.detect(tex, { wanted: 300 }); // warm-up
    const t0 = performance.now();
    const N = 10;
    for (let i = 0; i < N; i++) det.detect(tex, { wanted: 300 });
    const ms = (performance.now() - t0) / N;
    // Generous bound for CI; typical is a few ms.
    expect(ms).toBeLessThan(60);
  });
});
