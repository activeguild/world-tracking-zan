import { describe, expect, it } from "vitest";
import { localFloorHit, type LocalFloorConfig } from "../../src/ar/LocalFloor";
import { LANDMARK_STRIDE } from "../../src/vision/types";

/**
 * Recordings 45 / 46: the landmarks around the placed cube were 7–16 % of
 * the depth below the world plane. The hit test asks the landmarks near the
 * tap how high the floor is and lands there.
 *
 * Camera at the origin looking +Z (Y down); world plane y = 1 with the up
 * vector towards the camera (0, −1, 0). Height h above the plane = map y of 1 − h.
 */
const cfg: LocalFloorConfig = { radiusDepthRatio: 0.1, heightWindowDepthRatio: 0.25, minLandmarks: 6, maxSpreadDepthRatio: 0.05 };
const C = [0, 0, 0];
const forward = [0, 0, 1];
const up = [0, -1, 0];
const origin = [0, 1, 2];
// Ray through the floor point (0.2, 1, 2).
const dir = [0.2, 1, 2];

function grid(height: (x: number, z: number, i: number) => number, step = 0.05, keep: (x: number, z: number) => boolean = () => true): { packed: Float32Array; count: number } {
  const pts: number[] = [];
  let i = 0;
  for (let x = -1; x <= 1.0001; x += step) {
    for (let z = 1; z <= 3.0001; z += step) {
      if (keep(x, z)) pts.push(x, 1 - height(x, z, i), z, 1, 50);
      i++;
    }
  }
  return { packed: Float32Array.from(pts), count: pts.length / LANDMARK_STRIDE };
}

describe("localFloorHit", () => {
  it("lands on the world plane when the floor is on it, and without landmarks", () => {
    const { packed, count } = grid(() => 0);
    const hit = localFloorHit(C, dir, forward, up, origin, packed, count, cfg)!;
    expect(hit.surface).toBe("local");
    expect(Math.abs(hit.height)).toBeLessThan(1e-6);
    expect(hit.point[0]).toBeCloseTo(0.2, 6);
    expect(hit.point[1]).toBeCloseTo(1, 6);
    expect(hit.point[2]).toBeCloseTo(2, 6);
    expect(hit.depth).toBeCloseTo(2, 6);
    expect(hit.neighbors).toBeGreaterThan(20);

    const none = localFloorHit(C, dir, forward, up, origin, null, 0, cfg)!;
    expect(none.surface).toBe("plane");
    expect(none.fallback).toBe("few_landmarks");
    expect(none.height).toBe(0);
    expect(Array.from(none.point)).toEqual(Array.from(hit.point).map((v) => expect.closeTo(v, 6)));
    expect(localFloorHit(C, dir, forward, up, origin, packed, count, null)!.surface).toBe("plane");
    // A ray that never reaches the plane.
    expect(localFloorHit(C, [0, -1, 1], forward, up, origin, packed, count, cfg)).toBeNull();
  });

  it("follows a floor below / above the plane along the ray (recording 45: −16 % of the depth)", () => {
    for (const h of [-0.32, -0.15, 0.1]) {
      const { packed, count } = grid(() => h);
      const hit = localFloorHit(C, dir, forward, up, origin, packed, count, cfg)!;
      expect(hit.surface).toBe("local");
      expect(hit.height).toBeCloseTo(h, 5);
      // On the ray, at the floor's height: y = 1 − h, the rest scales with it.
      const s = (1 - h) / 1;
      expect(hit.point[1]).toBeCloseTo(1 - h, 5);
      expect(hit.point[0]).toBeCloseTo(0.2 * s, 5);
      expect(hit.point[2]).toBeCloseTo(2 * s, 5);
      expect(hit.t).toBeCloseTo(s, 5);
    }
  });

  it("clutter on the floor does not lift the hit; a neighbourhood without a common height falls back", () => {
    // A third of the points are things standing 0.15 above the floor.
    const cluttered = grid((_x, _z, i) => (i % 3 === 0 ? -0.2 + 0.15 : -0.2));
    const hit = localFloorHit(C, dir, forward, up, origin, cluttered.packed, cluttered.count, cfg)!;
    expect(hit.surface).toBe("local");
    expect(hit.height).toBeCloseTo(-0.2, 5);

    // Heights all over ± 0.3 (15 % of the depth): not a floor.
    let seed = 3;
    const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff - 0.5) * 2;
    const rough = grid(() => rnd() * 0.3);
    const fb = localFloorHit(C, dir, forward, up, origin, rough.packed, rough.count, cfg)!;
    expect(fb.surface).toBe("plane");
    expect(fb.fallback).toBe("spread");
    expect(fb.height).toBe(0);
    expect(fb.point[1]).toBeCloseTo(1, 9);
  });

  it("widens the neighbourhood once, then gives up", () => {
    // Nothing within 0.2 (= 0.1 × depth) of the hit, a ring of landmarks between 0.25 and 0.38.
    const ring = grid(() => -0.1, 0.05, (x, z) => {
      const d = Math.hypot(x - 0.22, z - 2.2);
      return d > 0.25 && d < 0.38;
    });
    const hit = localFloorHit(C, dir, forward, up, origin, ring.packed, ring.count, cfg)!;
    expect(hit.surface).toBe("local");
    expect(hit.height).toBeCloseTo(-0.1, 5);
    // Farther than twice the radius: world plane.
    const far = grid(() => -0.1, 0.05, (x, z) => Math.hypot(x - 0.2, z - 2) > 0.6);
    const fb = localFloorHit(C, dir, forward, up, origin, far.packed, far.count, cfg)!;
    expect(fb.surface).toBe("plane");
    expect(fb.fallback).toBe("few_landmarks");
    // Fewer than minLandmarks.
    const few = grid(() => -0.1, 0.05, (x, z) => Math.hypot(x - 0.2, z - 2) < 0.06);
    expect(few.count).toBeLessThan(6);
    expect(localFloorHit(C, dir, forward, up, origin, few.packed, few.count, cfg)!.surface).toBe("plane");
  });
});
