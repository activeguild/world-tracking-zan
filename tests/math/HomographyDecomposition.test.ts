import { describe, expect, it } from "vitest";
import { estimateHomography } from "../../src/math/Homography";
import { decomposeHomography, homographySolutionSupport } from "../../src/math/HomographyDecomposition";
import { angleBetween, rotationAxisAngle, rotationDistance } from "../../src/math/Pose";
import { poseFromCenter, projectTwoViews, randomPlanePoints } from "../helpers/scene";
import { createRng } from "../helpers/synthetic";

const deg = (rad: number) => (rad * 180) / Math.PI;

describe("Homography decomposition", () => {
  it("recovers R, t/d and n for a planar scene", () => {
    const rng = createRng(3);
    // Desk-like plane: tilted normal pointing back toward the camera, 1 m away.
    const normal = [0, -0.5, 0.866];
    const d = 1.0;
    const R = rotationAxisAngle([0.2, 1, 0.1], 0.1);
    const center = [0.2, -0.05, 0.08];
    const pose = poseFromCenter(R, center);
    const pts = randomPlanePoints(rng, 120, normal, d, 0.8);
    const c = projectTwoViews(pts, pose, null);
    expect(c.n).toBeGreaterThan(80);

    const h = estimateHomography(c.x1, c.y1, c.x2, c.y2, null, c.n)!;
    expect(h).not.toBeNull();
    const sols = decomposeHomography(h);
    expect(sols.length).toBe(4);

    const nn = normal.map((v) => v / Math.hypot(...normal));
    const tOverD = pose.translation.map((v) => v / d);
    const match = sols.filter(
      (s) =>
        deg(rotationDistance(s.rotation, R)) < 0.05 &&
        deg(angleBetween(s.normal, nn)) < 0.5 &&
        deg(angleBetween(s.translation, tOverD)) < 0.5 &&
        Math.abs(Math.hypot(...s.translation) - Math.hypot(...tOverD)) < 0.01,
    );
    expect(match.length).toBe(1);

    // Positive-depth support picks the right solution (or its t/n-swapped twin).
    const supports = sols.map((s) => homographySolutionSupport(s, c.x1, c.y1, null, c.n));
    const bestIdx = supports.indexOf(Math.max(...supports));
    expect(supports[bestIdx]).toBe(c.n);
    expect(supports[sols.indexOf(match[0])]).toBe(c.n);
  });

  it("detects pure rotation", () => {
    const rng = createRng(4);
    const R = rotationAxisAngle([0, 1, 0], 0.15);
    const pose = { rotation: R, translation: new Float64Array(3) };
    const pts = randomPlanePoints(rng, 100, [0, 0, 1], 2, 1.2);
    const c = projectTwoViews(pts, pose, null);
    const h = estimateHomography(c.x1, c.y1, c.x2, c.y2, null, c.n)!;
    const sols = decomposeHomography(h);
    expect(sols.length).toBe(1);
    expect(sols[0].pureRotation).toBe(true);
    expect(deg(rotationDistance(sols[0].rotation, R))).toBeLessThan(0.01);
  });
});
