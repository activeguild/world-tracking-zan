import { describe, expect, it } from "vitest";
import { cameraPoseToThree, mapDirToWorld, mapToWorld, threePoseToCamera, worldFromPlane, worldToMap } from "../../src/math/CoordinateSystem";
import { mat3Identity } from "../../src/math/Matrix";
import { cameraCenterOf, rotationAxisAngle, rotationDistance, type RigidTransform } from "../../src/math/Pose";
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

  it("Test 4 (指示書 §4.3): one convention — X_cam = R X_map + t; the Three camera sits at the CV camera centre and looks along CV +Z; map ↔ world round trips are exact", () => {
    const rng = createRng(7);
    for (let i = 0; i < 100; i++) {
      const n = [rng() - 0.5, -(0.5 + rng()), rng() - 0.5];
      const w = worldFromPlane(n, [rng(), 1 + rng(), rng() * 2], identity, 0.2 + rng());
      const axis = [rng() - 0.5, rng() - 0.5, rng() - 0.5];
      const pose = poseFromCenter(rotationAxisAngle(axis, (rng() - 0.5) * 2), [rng() - 0.5, rng() - 0.5, rng() - 0.5]);
      const t = cameraPoseToThree(w, pose);
      // Position: the world point of the camera centre C = −Rᵀ t.
      const c = mapToWorld(w, cameraCenterOf(pose));
      for (let a = 0; a < 3; a++) expect(t.position[a]).toBeCloseTo(c[a], 10);
      // Orientation: the Three camera looks along its −Z, which is CV +Z (third row of R) in world.
      const [x, y, z, q] = t.quaternion;
      const lookThree = [-(2 * (x * z + y * q)), -(2 * (y * z - x * q)), -(1 - 2 * (x * x + y * y))];
      const r = pose.rotation;
      const lookCv = mapDirToWorld(w, [r[6], r[7], r[8]]);
      for (let a = 0; a < 3; a++) expect(lookThree[a]).toBeCloseTo(lookCv[a], 10);
      // Points: map → world → map and world → map → world are exact.
      const p = [rng() * 2 - 1, rng() * 2 - 1, rng() * 2 - 1];
      const back = worldToMap(w, mapToWorld(w, p));
      for (let a = 0; a < 3; a++) expect(back[a]).toBeCloseTo(p[a], 12);
      const pw = mapToWorld(w, worldToMap(w, p));
      for (let a = 0; a < 3; a++) expect(pw[a]).toBeCloseTo(p[a], 12);
    }
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
