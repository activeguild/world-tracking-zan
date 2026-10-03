import { describe, expect, it } from "vitest";
import { WorldAnchor } from "../../src/ar/WorldAnchor";
import { worldToMap } from "../../src/math/CoordinateSystem";
import { rotationAxisAngle, type RigidTransform } from "../../src/math/Pose";
import type { MapPoseOutput, PlaneOutput } from "../../src/vision/types";
import { poseFromCenter, TEST_K } from "../helpers/scene";

function mapPose(p: RigidTransform, mapFrameId = 0): MapPoseOutput {
  return {
    rotation: Array.from(p.rotation),
    translation: Array.from(p.translation),
    inlierCount: 100,
    meanReprojectionErrorPx: 0.2,
    landmarkCount: 300,
    mapFrameId,
    framesSinceTracked: 0,
    cameraCenter: [0, 0, 0],
    deltaTranslation: 0,
    deltaRotationDeg: 0,
    translationHeld: false,
    translationPredicted: false,
    jumpRejected: false,
    reassociated: 0,
    mapInlierCount: 100,
    planeInlierCount: 0,
    rejectReason: null,
    sourceDeltaTranslation: 0,
    sourceDeltaRotationDeg: 0,
    sourceHistory: "M",
    source: "map",
  };
}

function plane(normal: number[], center: number[]): PlaneOutput {
  const d = -(normal[0] * center[0] + normal[1] * center[1] + normal[2] * center[2]);
  return {
    normal, d, center, inlierCount: 200, rmsResidual: 0.001, areaEstimate: 1, horizontalness: 1,
    horizontal: true, confidence: 1, stableFrames: 10, found: true, usedGravity: true,
  };
}

function project(p: RigidTransform, X: number[]): [number, number] {
  const r = p.rotation, t = p.translation;
  const x = r[0] * X[0] + r[1] * X[1] + r[2] * X[2] + t[0];
  const y = r[3] * X[0] + r[4] * X[1] + r[5] * X[2] + t[1];
  const z = r[6] * X[0] + r[7] * X[1] + r[8] * X[2] + t[2];
  return [(x / z) * TEST_K.fx + TEST_K.cx, (y / z) * TEST_K.fy + TEST_K.cy];
}

describe("WorldAnchor", () => {
  // Desk plane 1.2 map units in front of the init camera, tilted 45°.
  const n = [0, -Math.SQRT1_2, -Math.SQRT1_2];
  const c = [0, 0.85, 0.85];
  const pl = plane(n, c);
  const cam0: RigidTransform = { rotation: new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]), translation: new Float64Array(3) };

  it("creates the world with the assumed camera distance as scale", () => {
    const anchor = new WorldAnchor({ assumedPlaneDistanceMeters: 0.5 });
    expect(anchor.isReady).toBe(false);
    expect(anchor.create(pl, mapPose(cam0))).toBe(true);
    expect(anchor.isReady).toBe(true);
    // Camera (at the map origin) distance to the plane in map units is |d| → scale 0.5/|d|
    expect(anchor.frame!.scale).toBeCloseTo(0.5 / Math.abs(pl.d), 6);
    const pose = anchor.cameraPose(mapPose(cam0))!;
    expect(pose.position[1]).toBeCloseTo(0.5, 6); // 0.5 m above the plane
    // Second create for the same map is a no-op.
    expect(anchor.create(pl, mapPose(cam0))).toBe(false);
  });

  it("hit test returns the plane point under the pixel, from any camera", () => {
    const anchor = new WorldAnchor({ assumedPlaneDistanceMeters: 0.5 });
    anchor.create(pl, mapPose(cam0));
    // A known plane point (map frame): center + 0.3·right + 0.2·forward
    const f = anchor.frame!;
    const P = [
      c[0] + f.plane.right[0] * 0.3 + f.plane.forward[0] * 0.2,
      c[1] + f.plane.right[1] * 0.3 + f.plane.forward[1] * 0.2,
      c[2] + f.plane.right[2] * 0.3 + f.plane.forward[2] * 0.2,
    ];
    const expected = anchor.toWorld(P)!;
    expect(expected[1]).toBeCloseTo(0, 9);
    for (const cam of [cam0, poseFromCenter(rotationAxisAngle([0, 1, 0], 0.1), [0.2, -0.1, 0.1])]) {
      const [u, v] = project(cam, P);
      const hit = anchor.hitTest(u, v, TEST_K, mapPose(cam))!;
      expect(hit).not.toBeNull();
      for (let i = 0; i < 3; i++) expect(hit.position[i]).toBeCloseTo(expected[i], 6);
      expect(hit.normal[1]).toBeCloseTo(1, 9);
      expect(hit.distance).toBeGreaterThan(0);
    }
  });

  it("hit test round trip: world point → projected pixel → hit test → same world point (v2 §21)", () => {
    const anchor = new WorldAnchor({ assumedPlaneDistanceMeters: 0.5 });
    anchor.create(pl, mapPose(cam0));
    const cams = [cam0, poseFromCenter(rotationAxisAngle([0, 1, 0], 0.1), [0.2, -0.1, 0.1]), poseFromCenter(rotationAxisAngle([1, 0, 0], -0.15), [-0.1, 0.05, -0.2])];
    for (const [u0, v0] of [[320, 300], [120, 400], [540, 260]]) {
      const first = anchor.hitTest(u0, v0, TEST_K, mapPose(cam0))!;
      expect(first).not.toBeNull();
      for (const cam of cams) {
        // World → map → camera projection → hit test from that camera.
        const pm = Array.from(worldToMap(anchor.frame!, first.position));
        const [u, v] = project(cam, pm);
        const again = anchor.hitTest(u, v, TEST_K, mapPose(cam))!;
        expect(again).not.toBeNull();
        const err = Math.hypot(again.position[0] - first.position[0], again.position[1] - first.position[1], again.position[2] - first.position[2]);
        expect(err).toBeLessThan(1e-6);
      }
    }
  });

  it("misses when the ray does not reach the plane", () => {
    const anchor = new WorldAnchor({ assumedPlaneDistanceMeters: 0.5 });
    anchor.create(pl, mapPose(cam0));
    // Looking up (pixel far above the principal point: ray tilts toward −Y, away from the plane below).
    expect(anchor.hitTest(320, -5000, TEST_K, mapPose(cam0))).toBeNull();
  });

  it("drops the world when the map is reset", () => {
    const anchor = new WorldAnchor({ assumedPlaneDistanceMeters: 0.5 });
    anchor.create(pl, mapPose(cam0, 3));
    expect(anchor.checkMap(mapPose(cam0, 3))).toBe(false);
    expect(anchor.checkMap(mapPose(cam0, 9))).toBe(true);
    expect(anchor.isReady).toBe(false);
    expect(anchor.hitTest(320, 240, TEST_K, mapPose(cam0, 9))).toBeNull();
  });
});
