import { describe, expect, it } from "vitest";
import {
  cameraPoseToThree,
  mapDirToWorld,
  mapToWorld,
  projectionMatrixFromIntrinsics,
  threePoseToCamera,
  viewportIntrinsics,
  worldFromPlane,
  worldToMap,
} from "../../src/math/CoordinateSystem";
import { det3 } from "../../src/math/Decomposition";
import { mat3Multiply } from "../../src/math/Matrix";
import { rotationAxisAngle, rotationDistance, type RigidTransform } from "../../src/math/Pose";
import { poseFromCenter, TEST_K } from "../helpers/scene";

const deg = (rad: number) => (rad * 180) / Math.PI;

/** Desk-like plane in the map frame: normal toward the camera, 1.2 units away along the view. */
const normal = [0, -Math.SQRT1_2, -Math.SQRT1_2];
const center = [0.1, 0.85, 0.85];
const cam: RigidTransform = poseFromCenter(rotationAxisAngle([0, 1, 0], 0.05), [0.05, -0.02, 0.0]);

describe("worldFromPlane", () => {
  const w = worldFromPlane(normal, center, cam, 0.5);

  it("builds a right-handed orthonormal frame with +Y = plane normal", () => {
    const r = w.rotation;
    const rrt = mat3Multiply(r, new Float64Array([r[0], r[3], r[6], r[1], r[4], r[7], r[2], r[5], r[8]]));
    for (let i = 0; i < 9; i++) expect(rrt[i]).toBeCloseTo(i % 4 === 0 ? 1 : 0, 10);
    expect(det3(r)).toBeCloseTo(1, 10);
    expect(w.plane.up[0]).toBeCloseTo(normal[0], 10);
    expect(w.plane.up[1]).toBeCloseTo(normal[1], 10);
    expect(w.plane.up[2]).toBeCloseTo(normal[2], 10);
    // forward is in the plane and points away from the camera along its view
    const fwd = w.plane.forward;
    expect(fwd[0] * normal[0] + fwd[1] * normal[1] + fwd[2] * normal[2]).toBeCloseTo(0, 10);
    expect(fwd[2]).toBeGreaterThan(0.5);
  });

  it("puts plane points at Y = 0 and the origin at the plane center", () => {
    expect(Array.from(mapToWorld(w, center))).toEqual([0, 0, 0]);
    // A point on the plane away from the center
    const p = [center[0] + w.plane.right[0] * 0.3, center[1] + w.plane.right[1] * 0.3, center[2] + w.plane.right[2] * 0.3];
    const pw = mapToWorld(w, p);
    expect(pw[1]).toBeCloseTo(0, 10);
    expect(pw[0]).toBeCloseTo(0.3 * 0.5, 10); // scaled by 0.5 m/unit
  });

  it("map ↔ world round trips", () => {
    const p = [0.3, -0.2, 1.7];
    const back = worldToMap(w, mapToWorld(w, p));
    for (let i = 0; i < 3; i++) expect(back[i]).toBeCloseTo(p[i], 10);
    const d = mapDirToWorld(w, normal);
    expect(d[1]).toBeCloseTo(1, 10);
  });
});

describe("camera pose conversion", () => {
  const w = worldFromPlane(normal, center, cam, 0.5);

  it("Three.js camera sits above the plane and looks toward it", () => {
    const t = cameraPoseToThree(w, cam);
    expect(t.position[1]).toBeGreaterThan(0); // above the plane
    // Three.js camera looks along local −Z; rotate (0,0,−1) by the quaternion.
    const [x, y, z, q] = t.quaternion;
    const fz = -(2 * (x * z + y * q));
    const fy = -(2 * (y * z - x * q));
    const fx = -(1 - 2 * (x * x + y * y));
    // forward in world = R · (0,0,−1): columns... compute via matrix
    const fwd = [-(2 * (x * z + y * q)), -(2 * (y * z - x * q)), -(1 - 2 * (x * x + y * y))];
    void fx; void fy; void fz;
    // Looking down toward the plane: negative Y component
    expect(fwd[1]).toBeLessThan(-0.5);
    // Roughly along world −Z (the forward direction used to build the frame)
    expect(fwd[2]).toBeLessThan(-0.5);
  });

  it("round-trips through threePoseToCamera", () => {
    const t = cameraPoseToThree(w, cam);
    const back = threePoseToCamera(w, t.position, t.quaternion);
    expect(deg(rotationDistance(back.rotation, cam.rotation))).toBeLessThan(1e-6);
    for (let i = 0; i < 3; i++) expect(back.translation[i]).toBeCloseTo(cam.translation[i], 8);
  });
});

describe("projectionMatrixFromIntrinsics", () => {
  it("projects a point to the same pixel as the pinhole model", () => {
    const W = 640, H = 480;
    const P = projectionMatrixFromIntrinsics(TEST_K.fx, TEST_K.fy, TEST_K.cx + 7, TEST_K.cy - 5, W, H, 0.01, 50);
    // CV camera point
    const X = 0.3, Y = -0.1, Z = 2.0;
    const u = (TEST_K.fx * X) / Z + TEST_K.cx + 7;
    const v = (TEST_K.fy * Y) / Z + TEST_K.cy - 5;
    // Three.js camera coordinates: (X, −Y, −Z)
    const px = X, py = -Y, pz = -Z;
    // clip = P · [px py pz 1] (column-major)
    const cx = P[0] * px + P[4] * py + P[8] * pz + P[12];
    const cy = P[1] * px + P[5] * py + P[9] * pz + P[13];
    const cw = P[3] * px + P[7] * py + P[11] * pz + P[15];
    const ndcX = cx / cw;
    const ndcY = cy / cw;
    const su = ((ndcX + 1) / 2) * W;
    const sv = ((1 - ndcY) / 2) * H;
    expect(su).toBeCloseTo(u, 6);
    expect(sv).toBeCloseTo(v, 6);
  });

  it("viewportIntrinsics follows object-fit: cover", () => {
    // 640×480 image shown in a 390×844 portrait viewport → scale by height
    const v = viewportIntrinsics(TEST_K, 390, 844);
    expect(v.scale).toBeCloseTo(844 / 480, 10);
    expect(v.offsetY).toBeCloseTo(0, 10);
    expect(v.offsetX).toBeLessThan(0);
    expect(v.cx).toBeCloseTo(390 / 2, 6);
  });
});
