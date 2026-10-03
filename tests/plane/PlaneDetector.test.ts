import { describe, expect, it } from "vitest";
import { DEFAULT_CONFIG } from "../../src/ar/ARConfig";
import { angleBetween } from "../../src/math/Pose";
import { PlaneDetector } from "../../src/vision/PlaneDetector";
import { gauss, randomBoxPoints, randomPlanePoints } from "../helpers/scene";
import { createRng } from "../helpers/synthetic";

const deg = (rad: number) => (rad * 180) / Math.PI;

/** Floor seen from a phone pitched 45° down: normal (toward camera) ≈ (0, −0.707, −0.707) in the map frame. */
function floorScene(rng: () => number, n = 150, clutter = 40) {
  const normal = [0, -Math.SQRT1_2, -Math.SQRT1_2];
  const plane = randomPlanePoints(rng, n, [0, Math.SQRT1_2, Math.SQRT1_2], 1.2, 0.8);
  for (let i = 0; i < plane.length; i++) plane[i] += gauss(rng) * 0.004;
  const box = randomBoxPoints(rng, clutter, 0.4, 1.0, 0.6);
  const pts = new Float64Array(plane.length + box.length);
  pts.set(plane);
  pts.set(box, plane.length);
  const ids = Array.from({ length: n + clutter }, (_, i) => i + 1);
  return { pts, ids, count: n + clutter, normal };
}

describe("PlaneDetector", () => {
  const gravityFloor = new Float64Array([0, Math.SQRT1_2, Math.SQRT1_2]); // down, 45° pitch

  it("finds a horizontal floor plane and becomes stable after N frames", () => {
    const rng = createRng(5);
    const det = new PlaneDetector(DEFAULT_CONFIG.plane, createRng(1));
    const { pts, ids, count, normal } = floorScene(rng);
    let last = null;
    for (let f = 0; f < DEFAULT_CONFIG.plane.stableFramesRequired + 2; f++) {
      // jitter the points slightly each frame
      const jit = Float64Array.from(pts, (v) => v + gauss(rng) * 0.001);
      last = det.update(jit, ids, count, gravityFloor);
      expect(last).not.toBeNull();
      expect(deg(angleBetween(last!.normal, normal, true))).toBeLessThan(2);
      expect(last!.horizontal).toBe(true);
      expect(last!.inlierCount).toBeGreaterThan(130);
      if (f < DEFAULT_CONFIG.plane.stableFramesRequired) expect(last!.found).toBe(false);
    }
    expect(last!.found).toBe(true);
    expect(last!.stableFrames).toBeGreaterThanOrEqual(DEFAULT_CONFIG.plane.stableFramesRequired);
    expect(last!.confidence).toBeGreaterThan(0.5);
    expect(last!.usedGravity).toBe(true);
    // Normal points toward the camera (origin): n·(0 − c) > 0.
    const c = last!.center;
    expect(-(last!.normal[0] * c[0] + last!.normal[1] * c[1] + last!.normal[2] * c[2])).toBeGreaterThan(0);
  });

  it("rejects a wall (vertical plane) as non-horizontal", () => {
    const rng = createRng(6);
    const det = new PlaneDetector(DEFAULT_CONFIG.plane, createRng(2));
    // Fronto-parallel wall at z = 1.5, phone upright: gravity along +Y (down).
    const wall = randomPlanePoints(rng, 150, [0, 0, 1], 1.5, 0.8);
    const ids = Array.from({ length: 150 }, (_, i) => i + 1);
    let last = null;
    for (let f = 0; f < 10; f++) last = det.update(wall, ids, 150, new Float64Array([0, 1, 0]));
    expect(last).not.toBeNull();
    expect(last!.horizontalness).toBeLessThan(0.1);
    expect(last!.horizontal).toBe(false);
    expect(last!.found).toBe(false);
    expect(det.isFound).toBe(false);
  });

  it("without gravity falls back to the camera −Y axis as up", () => {
    const rng = createRng(7);
    const det = new PlaneDetector(DEFAULT_CONFIG.plane, createRng(3));
    const { pts, ids, count } = floorScene(rng);
    let last = null;
    for (let f = 0; f < 8; f++) last = det.update(pts, ids, count, null);
    expect(last!.usedGravity).toBe(false);
    // |n·(0,-1,0)| = 0.707 ≥ fallback threshold 0.5 → horizontal
    expect(last!.horizontalness).toBeCloseTo(Math.SQRT1_2, 1);
    expect(last!.horizontal).toBe(true);
    expect(last!.found).toBe(true);
  });

  it("low landmark count → no plane; found plane survives a short dropout", () => {
    const rng = createRng(8);
    const det = new PlaneDetector(DEFAULT_CONFIG.plane, createRng(4));
    expect(det.update(new Float64Array(0), [], 0, gravityFloor)).toBeNull();
    const { pts, ids, count } = floorScene(rng);
    for (let f = 0; f < 8; f++) det.update(pts, ids, count, gravityFloor);
    expect(det.isFound).toBe(true);
    for (let f = 0; f < DEFAULT_CONFIG.plane.lostFrames; f++) {
      const r = det.update(new Float64Array(0), [], 0, gravityFloor);
      expect(r).not.toBeNull();
      expect(r!.found).toBe(true);
    }
    expect(det.update(new Float64Array(0), [], 0, gravityFloor)).toBeNull();
    expect(det.isFound).toBe(false);
  });
});
