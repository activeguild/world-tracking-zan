import { describe, expect, it } from "vitest";
import { DEFAULT_CONFIG } from "../../src/ar/ARConfig";
import { angleBetween, rotationAxisAngle, rotationDistance } from "../../src/math/Pose";
import { PoseEstimator } from "../../src/vision/PoseEstimator";
import { corrupt, poseFromCenter, projectTwoViews, randomBoxPoints, randomPlanePoints, TEST_K } from "../helpers/scene";
import { createRng } from "../helpers/synthetic";

const deg = (rad: number) => (rad * 180) / Math.PI;

function estimator(seed = 1) {
  return new PoseEstimator(DEFAULT_CONFIG.pose, DEFAULT_CONFIG.ransac, createRng(seed));
}

describe("PoseEstimator (two-view relative pose, pixel input)", () => {
  it("non-planar scene → essential model with correct R and t direction", () => {
    const rng = createRng(21);
    const R = rotationAxisAngle([0.05, 1, 0.1], 0.08);
    const pose = poseFromCenter(R, [0.25, 0.02, 0.05]);
    const pts = randomBoxPoints(rng, 250, 1, 4);
    const c = projectTwoViews(pts, pose, TEST_K, 0.4, rng);
    corrupt(c, 0.15, TEST_K, rng);

    const rel = estimator().estimate(c.x1, c.y1, c.x2, c.y2, c.n, TEST_K);
    expect(rel.model).toBe("essential");
    expect(deg(rotationDistance(rel.rotation, R))).toBeLessThan(0.6);
    expect(deg(angleBetween(rel.translationDirection, pose.translation))).toBeLessThan(4);
    expect(rel.translationConfidence).toBeGreaterThan(0.3);
    expect(rel.confidence).toBeGreaterThan(0.5);
    expect(rel.parallaxPx).toBeGreaterThan(DEFAULT_CONFIG.pose.minParallaxPx);
  });

  it("planar scene → homography model with correct R, t direction and plane normal", () => {
    const rng = createRng(22);
    const normal = [0, -0.6, 0.8];
    const R = rotationAxisAngle([1, 0.2, 0], -0.06);
    const pose = poseFromCenter(R, [0.15, 0.0, 0.1]);
    const pts = randomPlanePoints(rng, 250, normal, 1.0, 0.9);
    const c = projectTwoViews(pts, pose, TEST_K, 0.4, rng);
    corrupt(c, 0.1, TEST_K, rng);

    const rel = estimator().estimate(c.x1, c.y1, c.x2, c.y2, c.n, TEST_K);
    expect(rel.model).toBe("homography");
    expect(deg(rotationDistance(rel.rotation, R))).toBeLessThan(1.0);
    expect(deg(angleBetween(rel.translationDirection, pose.translation))).toBeLessThan(6);
    expect(rel.planeNormal).not.toBeNull();
    const nn = normal.map((v) => v / Math.hypot(...normal));
    expect(deg(angleBetween(rel.planeNormal!, nn))).toBeLessThan(6);
  });

  it("uses the previous normal to disambiguate the homography twin solution", () => {
    const rng = createRng(23);
    const normal = [0.3, -0.3, 0.9];
    const R = rotationAxisAngle([0, 1, 0], 0.03);
    const pose = poseFromCenter(R, [0.12, 0.0, 0.0]);
    const pts = randomPlanePoints(rng, 200, normal, 1.0, 0.9);
    const c = projectTwoViews(pts, pose, TEST_K, 0.3, rng);
    const nn = Float64Array.from(normal.map((v) => v / Math.hypot(...normal)));
    const rel = estimator().estimate(c.x1, c.y1, c.x2, c.y2, c.n, TEST_K, nn);
    expect(rel.model).toBe("homography");
    expect(deg(angleBetween(rel.planeNormal!, nn))).toBeLessThan(5);
    expect(deg(angleBetween(rel.translationDirection, pose.translation))).toBeLessThan(6);
  });

  it("pure rotation → rotation-only model with accurate R", () => {
    const rng = createRng(24);
    const R = rotationAxisAngle([0.1, 1, 0], 0.1);
    const pose = { rotation: R, translation: new Float64Array(3) };
    const pts = randomBoxPoints(rng, 250, 1, 4);
    const c = projectTwoViews(pts, pose, TEST_K, 0.4, rng);

    const rel = estimator().estimate(c.x1, c.y1, c.x2, c.y2, c.n, TEST_K);
    expect(rel.model).toBe("rotation");
    expect(deg(rotationDistance(rel.rotation, R))).toBeLessThan(0.5);
    expect(rel.translationConfidence).toBe(0);
    expect(Math.hypot(...rel.translationDirection)).toBe(0);
  });

  it("tiny parallax → rotation-only model", () => {
    const rng = createRng(25);
    const R = rotationAxisAngle([0, 1, 0], 0.002);
    const pose = poseFromCenter(R, [0.001, 0, 0]);
    const pts = randomBoxPoints(rng, 200, 1, 4);
    const c = projectTwoViews(pts, pose, TEST_K, 0.2, rng);
    const rel = estimator().estimate(c.x1, c.y1, c.x2, c.y2, c.n, TEST_K);
    expect(rel.parallaxPx).toBeLessThan(DEFAULT_CONFIG.pose.minParallaxPx);
    expect(rel.model).toBe("rotation");
    expect(deg(rotationDistance(rel.rotation, R))).toBeLessThan(0.3);
  });

  it("forward motion (dolly in) → essential model, t ≈ +Z", () => {
    const rng = createRng(26);
    const R = rotationAxisAngle([0, 1, 0], 0.0);
    const pose = poseFromCenter(R, [0, 0, 0.3]); // camera moves forward
    const pts = randomBoxPoints(rng, 300, 1.5, 5, 2);
    const c = projectTwoViews(pts, pose, TEST_K, 0.4, rng);
    const rel = estimator().estimate(c.x1, c.y1, c.x2, c.y2, c.n, TEST_K);
    expect(rel.model).toBe("essential");
    expect(deg(rotationDistance(rel.rotation, R))).toBeLessThan(0.6);
    // t = −R C = (0, 0, −0.3): camera moved forward → scene moves toward −Z in cam2 frame.
    expect(deg(angleBetween(rel.translationDirection, [0, 0, -1]))).toBeLessThan(6);
  });

  it("too few correspondences → none", () => {
    const x = new Float32Array(10);
    const rel = estimator().estimate(x, x, x, x, 10, TEST_K);
    expect(rel.model).toBe("none");
    expect(rel.confidence).toBe(0);
  });
});
