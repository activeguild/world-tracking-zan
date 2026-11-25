import { describe, expect, it } from "vitest";
import { mat3Identity } from "../../src/math/Matrix";
import { rotationAxisAngle, type RigidTransform } from "../../src/math/Pose";
import { cameraCenterOf, isJumpRejection, poseDelta, validatePoseCandidate } from "../../src/vision/PoseValidation";

/** Edge cases of the pose continuity check (found by mutation testing). */
const pose = (rotation: Float64Array, t: number[]): RigidTransform => ({ rotation, translation: new Float64Array(t) });
const identity = pose(mat3Identity(), [0, 0, 0]);

describe("cameraCenterOf", () => {
  it("is the point the pose maps to the camera origin: R·C + t = 0, for a general rotation", () => {
    const p = pose(rotationAxisAngle([1, 2, 3], 0.7), [0.3, -1.2, 2.5]);
    const c = cameraCenterOf(p);
    const r = p.rotation;
    for (let i = 0; i < 3; i++) {
      expect(r[i * 3] * c[0] + r[i * 3 + 1] * c[1] + r[i * 3 + 2] * c[2] + p.translation[i]).toBeCloseTo(0, 12);
    }
    expect(Math.hypot(c[0], c[1], c[2])).toBeCloseTo(Math.hypot(0.3, 1.2, 2.5), 12);
  });
});

describe("validatePoseCandidate: limits are inclusive", () => {
  it("a move of exactly the translation limit is accepted", () => {
    const moved = pose(mat3Identity(), [1, 0, 0]);
    expect(poseDelta(moved, identity).translation).toBe(1);
    expect(validatePoseCandidate(moved, identity, { maxTranslation: 1, maxRotationDeg: 20 }).accepted).toBe(true);
    expect(validatePoseCandidate(moved, identity, { maxTranslation: 0.999, maxRotationDeg: 20 }).rejection?.code).toBe("translation_jump");
  });

  it("a rotation of exactly the rotation limit is accepted", () => {
    const turned = pose(rotationAxisAngle([0, 1, 0], 0.2), [0, 0, 0]);
    const deg = poseDelta(turned, identity).rotationDeg;
    expect(deg).toBeCloseTo((0.2 * 180) / Math.PI, 9);
    expect(validatePoseCandidate(turned, identity, { maxTranslation: 1, maxRotationDeg: deg }).accepted).toBe(true);
    expect(validatePoseCandidate(turned, identity, { maxTranslation: 1, maxRotationDeg: deg - 1e-6 }).rejection?.code).toBe("rotation_jump");
  });
});

describe("validatePoseCandidate: rejection wording and codes", () => {
  const limits = { maxTranslation: 0.1, maxRotationDeg: 5 };
  const turned = pose(rotationAxisAngle([0, 1, 0], 0.5), [0, 0, 0]);

  it("the default label is `pose`", () => {
    const r = validatePoseCandidate(pose(mat3Identity(), [1, 0, 0]), identity, limits);
    expect(r.reason).toMatch(/^pose translation jump /);
  });

  it("a non-finite candidate says so, with its label", () => {
    const r = validatePoseCandidate(pose(mat3Identity(), [NaN, 0, 0]), identity, limits, "plane");
    expect(r.rejection?.code).toBe("invalid_pose");
    expect(r.reason).toBe("plane invalid pose");
  });

  it("a rotation difference between two candidates is a disagreement, not a jump", () => {
    const r = validatePoseCandidate(turned, identity, limits, "plane vs map", true);
    expect(r.rejection?.code).toBe("map_plane_disagreement");
    expect(isJumpRejection(r.rejection)).toBe(false);
  });

  it("isJumpRejection: translation and rotation jumps only", () => {
    const base = { reason: "", delta: 0, limit: 0 };
    expect(isJumpRejection({ ...base, code: "rotation_jump" })).toBe(true);
    expect(isJumpRejection({ ...base, code: "translation_jump" })).toBe(true);
    expect(isJumpRejection({ ...base, code: "insufficient_inliers" })).toBe(false);
    expect(isJumpRejection(null)).toBe(false);
  });
});
