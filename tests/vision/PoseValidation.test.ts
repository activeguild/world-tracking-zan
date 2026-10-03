import { describe, expect, it } from "vitest";
import { mat3Identity } from "../../src/math/Matrix";
import { rotationAxisAngle, type RigidTransform } from "../../src/math/Pose";
import { cameraCenterOf, poseDelta, validatePoseCandidate } from "../../src/vision/PoseValidation";
import { poseFromCenter } from "../helpers/scene";

/** v3 §2–§6: one validation for every pose candidate. */
describe("validatePoseCandidate", () => {
  const ref: RigidTransform = poseFromCenter(mat3Identity(), [0, 0, 0]);
  const limits = { maxTranslation: 0.1, maxRotationDeg: 5 };

  it("accepts a small move and reports the deltas", () => {
    const cand = poseFromCenter(rotationAxisAngle([0, 1, 0], 0.02), [0.05, 0, 0]);
    const v = validatePoseCandidate(cand, ref, limits, "map");
    expect(v.accepted).toBe(true);
    expect(v.reason).toBeNull();
    expect(v.translationDelta).toBeCloseTo(0.05, 9);
    expect(v.rotationDeltaDeg).toBeCloseTo((0.02 * 180) / Math.PI, 6);
  });

  it("rejects a translation jump and names the candidate", () => {
    const v = validatePoseCandidate(poseFromCenter(mat3Identity(), [0.3, 0, 0]), ref, limits, "plane");
    expect(v.accepted).toBe(false);
    expect(v.reason).toMatch(/^plane translation jump/);
    expect(v.translationDelta).toBeCloseTo(0.3, 9);
  });

  it("rejects a rotation jump", () => {
    const v = validatePoseCandidate(poseFromCenter(rotationAxisAngle([0, 0, 1], 0.2), [0, 0, 0]), ref, limits, "map");
    expect(v.accepted).toBe(false);
    expect(v.reason).toMatch(/^map rotation jump/);
    expect(v.rotationDeltaDeg).toBeCloseTo((0.2 * 180) / Math.PI, 6);
  });

  it("camera center and pose delta are symmetric", () => {
    const a = poseFromCenter(rotationAxisAngle([1, 0, 0], 0.1), [0.2, -0.1, 0.4]);
    const b = poseFromCenter(rotationAxisAngle([0, 1, 0], -0.05), [-0.1, 0.3, 0.2]);
    expect(Array.from(cameraCenterOf(a)).map((v) => +v.toFixed(12))).toEqual([0.2, -0.1, 0.4]);
    const ab = poseDelta(a, b);
    const ba = poseDelta(b, a);
    expect(ab.translation).toBeCloseTo(ba.translation, 12);
    expect(ab.rotationDeg).toBeCloseTo(ba.rotationDeg, 9);
    expect(ab.translation).toBeCloseTo(Math.hypot(0.3, 0.4, 0.2), 12);
  });
});
