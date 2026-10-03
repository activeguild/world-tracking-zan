import { describe, expect, it } from "vitest";
import { cameraPoseToThree, threePoseToCamera, worldFromPlane } from "../../src/math/CoordinateSystem";
import { mat3Identity } from "../../src/math/Matrix";
import { rotationAxisAngle, rotationDistance, type RigidTransform } from "../../src/math/Pose";
import { createRng } from "../../src/vision/OutlierRejection";
import { poseFromCenter } from "../helpers/scene";

/**
 * 修正指示書 v2 §19–§20: numeric round-trip tests of the map ↔ Three.js
 * camera conversion. Errors must be at floating-point level; otherwise the
 * coordinate conversion, not the tracker, is what moves the objects.
 */
const identity: RigidTransform = { rotation: mat3Identity(), translation: new Float64Array(3) };
// Floor 1 map unit below a forward-looking camera: normal toward the camera
// (−Y in CV), so world +Y = CV −Y and world +X = CV +X.
const floorNormal = [0, -1, 0];

describe("CoordinateSystem round trip (v2 §19)", () => {
  it("Test 1: identity map camera → Three camera at the world origin when the plane center is the camera center", () => {
    const w = worldFromPlane(floorNormal, [0, 0, 0], identity, 1);
    const t = cameraPoseToThree(w, identity);
    for (let i = 0; i < 3; i++) expect(Math.abs(t.position[i])).toBeLessThan(1e-12);
    // Looks along world −Z: forward = R·(0,0,−1) with the identity quaternion → (0,0,−1).
    expect(t.quaternion[3]).toBeCloseTo(1, 12);
  });

  it("Test 2: moving the map camera along +X moves the Three camera along world +X by the same (scaled) amount", () => {
    const w = worldFromPlane(floorNormal, [0, 1, 0], identity, 0.5);
    const a = cameraPoseToThree(w, identity);
    const b = cameraPoseToThree(w, poseFromCenter(mat3Identity(), [0.2, 0, 0]));
    expect(b.position[0] - a.position[0]).toBeCloseTo(0.2 * 0.5, 12);
    expect(b.position[1] - a.position[1]).toBeCloseTo(0, 12);
    expect(b.position[2] - a.position[2]).toBeCloseTo(0, 12);
    // Camera height above the floor: 1 map unit × 0.5 m/unit.
    expect(a.position[1]).toBeCloseTo(0.5, 12);
    // Forward (CV +Z) is world −Z: moving forward decreases world Z.
    const c = cameraPoseToThree(w, poseFromCenter(mat3Identity(), [0, 0, 0.3]));
    expect(c.position[2] - a.position[2]).toBeCloseTo(-0.3 * 0.5, 12);
  });

  it("Test 3: pose → Three → pose round trip error < 1e-6 for random poses and planes", () => {
    const rng = createRng(99);
    let maxPos = 0;
    let maxRot = 0;
    for (let i = 0; i < 200; i++) {
      const axis = [rng() - 0.5, rng() - 0.5, rng() - 0.5];
      const n = [rng() - 0.5, -(0.5 + rng()), rng() - 0.5];
      const w = worldFromPlane(n, [rng(), 1 + rng(), rng() * 2], identity, 0.2 + rng());
      const pose = poseFromCenter(rotationAxisAngle(axis, (rng() - 0.5) * 2), [rng() - 0.5, rng() - 0.5, rng() - 0.5]);
      const t = cameraPoseToThree(w, pose);
      const back = threePoseToCamera(w, t.position, t.quaternion);
      maxRot = Math.max(maxRot, rotationDistance(back.rotation, pose.rotation));
      maxPos = Math.max(
        maxPos,
        Math.hypot(back.translation[0] - pose.translation[0], back.translation[1] - pose.translation[1], back.translation[2] - pose.translation[2]),
      );
    }
    console.log(`[coord] round-trip max position error ${maxPos.toExponential(2)} map units, max rotation error ${maxRot.toExponential(2)} rad`);
    expect(maxPos).toBeLessThan(1e-6);
    expect(maxRot).toBeLessThan(1e-6);
  });
});
