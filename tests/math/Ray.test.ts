import { describe, expect, it } from "vitest";
import { OneEuroFilter, QuaternionSmoother } from "../../src/math/OneEuroFilter";
import { rotationToQuaternion, rotationY } from "../../src/math/Pose";
import { intersectRayPlane, pixelRay, transformRay } from "../../src/math/Ray";

describe("Ray / plane", () => {
  it("intersects a ray with a plane", () => {
    const ray = { origin: new Float64Array([0, 1, 0]), direction: new Float64Array([0, -Math.SQRT1_2, Math.SQRT1_2]) };
    const hit = intersectRayPlane(ray, [0, 1, 0], 0)!; // plane y = 0
    expect(hit).not.toBeNull();
    expect(hit.point[1]).toBeCloseTo(0, 10);
    expect(hit.point[2]).toBeCloseTo(1, 10);
    expect(hit.t).toBeCloseTo(Math.SQRT2, 10);
  });

  it("returns null for parallel rays and hits behind the origin", () => {
    const parallel = { origin: new Float64Array([0, 1, 0]), direction: new Float64Array([1, 0, 0]) };
    expect(intersectRayPlane(parallel, [0, 1, 0], 0)).toBeNull();
    const away = { origin: new Float64Array([0, 1, 0]), direction: new Float64Array([0, 1, 0]) };
    expect(intersectRayPlane(away, [0, 1, 0], 0)).toBeNull();
  });

  it("pixelRay goes through the pixel; transformRay rotates and translates", () => {
    const r = pixelRay(320 + 64, 240, 640, 640, 320, 240);
    expect(r.direction[0] / r.direction[2]).toBeCloseTo(0.1, 10);
    expect(Math.hypot(...r.direction)).toBeCloseTo(1, 10);
    const R = rotationY(Math.PI / 2);
    const tr = transformRay(r, R, [1, 2, 3]);
    expect(tr.origin[0]).toBeCloseTo(1, 10);
    // z axis rotates to +x under Ry(90°)
    expect(tr.direction[0]).toBeCloseTo(r.direction[2], 10);
  });
});

describe("One Euro filter", () => {
  it("passes a constant unchanged and converges after a step", () => {
    const f = new OneEuroFilter({ minCutoff: 1, beta: 0.1, dCutoff: 1 });
    for (let i = 0; i < 10; i++) expect(f.filter(5, i / 30)).toBeCloseTo(5, 10);
    let v = 0;
    for (let i = 10; i < 70; i++) v = f.filter(10, i / 30);
    expect(v).toBeGreaterThan(9.5);
  });

  it("follows fast motion closely (low latency) but smooths jitter", () => {
    const fast = new OneEuroFilter({ minCutoff: 1, beta: 0.5, dCutoff: 1 });
    let maxLag = 0;
    for (let i = 0; i < 60; i++) {
      const t = i / 30;
      const truth = t * 2; // 2 units / s
      const out = fast.filter(truth, t);
      if (i > 5) maxLag = Math.max(maxLag, Math.abs(truth - out));
    }
    expect(maxLag).toBeLessThan(0.3);

    const slow = new OneEuroFilter({ minCutoff: 1, beta: 0.1, dCutoff: 1 });
    let jitterOut = 0;
    for (let i = 0; i < 120; i++) {
      const t = i / 30;
      const noisy = 1 + (i % 2 === 0 ? 0.05 : -0.05);
      const out = slow.filter(noisy, t);
      if (i > 30) jitterOut = Math.max(jitterOut, Math.abs(out - 1));
    }
    expect(jitterOut).toBeLessThan(0.03);
  });

  it("quaternion smoother converges to a target rotation", () => {
    const s = new QuaternionSmoother({ minCutoff: 2, beta: 0.5, dCutoff: 1 });
    const a = rotationToQuaternion(rotationY(0));
    const b = rotationToQuaternion(rotationY(0.3));
    s.filter(a, 0);
    let q = a;
    for (let i = 1; i <= 60; i++) q = s.filter(b, i / 30);
    const dot = Math.abs(q[0] * b[0] + q[1] * b[1] + q[2] * b[2] + q[3] * b[3]);
    expect(2 * Math.acos(Math.min(1, dot))).toBeLessThan(0.01);
  });
});
