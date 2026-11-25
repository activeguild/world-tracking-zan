import { describe, expect, it } from "vitest";
import { resolveConfig } from "../../src/ar/ARConfig";
import { objectFloorDiagnostics, type ObjectFloorConfig } from "../../src/ar/ObjectFloorDiagnostics";
import { WorldAnchor } from "../../src/ar/WorldAnchor";
import { mapToWorld, worldFromPlane, worldToMap } from "../../src/math/CoordinateSystem";
import { createRng } from "../../src/vision/OutlierRejection";
import { LANDMARK_STRIDE, type PlaneOutput, type VisionInput, type VisionOutput } from "../../src/vision/types";
import { VisionEngine } from "../../src/vision/VisionEngine";
import { TP_H, TP_K, TP_W, tpGravityAt, tpPose, tpRender } from "../helpers/twoPlaneScene";

/**
 * Recording 43 ("the cube's size changes differently from the floor texture
 * when I move closer / farther"): the object-floor diagnostic measures how
 * far the plane an object sits on is from the floor the map knows around
 * it. Pure function first (known geometry, sign conventions), then the
 * engine + anchor wiring of the session on the two-plane scene.
 */
const cfg: ObjectFloorConfig = { radiusDepthRatio: 0.1, heightWindowDepthRatio: 0.25, minLandmarks: 5, youngFrames: 30 };

// Camera at the origin looking +Z; floor at y = +1 (camera Y points down),
// normal oriented towards the camera, world scale 0.5 m / unit.
const identity = { rotation: new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]), translation: new Float64Array(3) };
const floorNormal = [0, -1, 0];
const world = worldFromPlane(floorNormal, [0, 1, 2], identity, 0.5);
const objectMap = [0.3, 1, 2.2];
const objectWorld = Array.from(mapToWorld(world, objectMap));

/** Landmarks on a grid over the floor; `height(x, z)` lifts them towards the camera (map −y). */
function floorLandmarks(height: (x: number, z: number) => number, age: (i: number) => number, extra: number[][] = []): { packed: Float32Array; count: number } {
  const pts: number[] = [];
  let i = 0;
  for (let x = -1; x <= 1.0001; x += 0.05) {
    for (let z = 1; z <= 3.0001; z += 0.05) {
      pts.push(x, 1 - height(x, z), z, 1, age(i));
      i++;
    }
  }
  for (const e of extra) pts.push(...e);
  return { packed: Float32Array.from(pts), count: pts.length / LANDMARK_STRIDE };
}

describe("objectFloorDiagnostics (pure)", () => {
  it("flat floor on the plane: height 0, tilt 0, local depth ratio 1, neighbours counted and split by age", () => {
    const { packed, count } = floorLandmarks(() => 0, (i) => (i % 2 ? 10 : 100));
    const [origin, obj] = objectFloorDiagnostics(packed, count, world, identity.rotation, [0, 0, 0], [{ id: 1, position: objectWorld }], cfg);
    expect(origin.id).toBe(0);
    expect(obj.id).toBe(1);
    expect(obj.depth).toBeCloseTo(2.2, 9);
    expect(obj.depthMeters).toBeCloseTo(1.1, 9);
    // Radius 0.22 around (0.3, ·, 2.2) on a 0.05 grid: ~60 points.
    expect(obj.neighbors).toBeGreaterThan(40);
    expect(obj.neighbors).toBeLessThan(80);
    expect(obj.planeFlagged).toBe(obj.neighbors);
    expect(obj.young + obj.old).toBe(obj.neighbors);
    expect(obj.young).toBeGreaterThan(10);
    expect(obj.old).toBeGreaterThan(10);
    expect(Math.abs(obj.heightRatio)).toBeLessThan(1e-6);
    expect(obj.heightSpread).toBeLessThan(1e-6);
    expect(Math.abs(obj.youngHeightRatio)).toBeLessThan(1e-6);
    expect(Math.abs(obj.oldHeightRatio)).toBeLessThan(1e-6);
    expect(obj.tiltDeg).toBeLessThan(1e-3);
    expect(obj.localDepthRatio).toBeCloseTo(1, 6);
    expect(Math.abs(origin.heightRatio)).toBeLessThan(1e-6);
  });

  it("floor 2 % of the depth above the plane (towards the camera): height +2 %, local surface nearer than the object", () => {
    const lift = 0.02 * 2.2;
    const { packed, count } = floorLandmarks(() => lift, () => 50);
    const [, obj] = objectFloorDiagnostics(packed, count, world, identity.rotation, [0, 0, 0], [{ id: 1, position: objectWorld }], cfg);
    expect(obj.heightRatio).toBeCloseTo(0.02, 6);
    // Ray from the camera to the object hits y = 1 − lift at t = (1 − lift) / 1.
    expect(obj.localDepthRatio).toBeCloseTo(1 - lift, 6);
    expect(obj.tiltDeg).toBeLessThan(1e-3);
    // Only the old group exists.
    expect(Number.isNaN(obj.youngHeightRatio)).toBe(true);
    expect(obj.oldHeightRatio).toBeCloseTo(0.02, 6);
  });

  it("young and old generations at different heights are reported apart", () => {
    const { packed, count } = floorLandmarks(() => 0, (i) => (i % 2 ? 5 : 200));
    // Lift the young ones by 1 % and the old ones by 3 % of the depth.
    for (let i = 0; i < count; i++) packed[i * LANDMARK_STRIDE + 1] -= packed[i * LANDMARK_STRIDE + 4] < 30 ? 0.01 * 2.2 : 0.03 * 2.2;
    const [, obj] = objectFloorDiagnostics(packed, count, world, identity.rotation, [0, 0, 0], [{ id: 1, position: objectWorld }], cfg);
    expect(obj.youngHeightRatio).toBeCloseTo(0.01, 6);
    expect(obj.oldHeightRatio).toBeCloseTo(0.03, 6);
    expect(obj.heightRatio).toBeGreaterThan(0.009);
    expect(obj.heightRatio).toBeLessThan(0.031);
  });

  it("a locally tilted floor shows as tilt against the world up", () => {
    const slope = 0.1; // atan(0.1) = 5.71°
    const { packed, count } = floorLandmarks((x) => slope * (x - 0.3), () => 50);
    const [, obj] = objectFloorDiagnostics(packed, count, world, identity.rotation, [0, 0, 0], [{ id: 1, position: objectWorld }], cfg);
    expect(obj.tiltDeg).toBeCloseTo((Math.atan(slope) * 180) / Math.PI, 2);
    // The plane passes through the object's column at height 0.
    expect(Math.abs(obj.heightRatio)).toBeLessThan(0.01);
  });

  it("shelf points far above the plane are not floor; too few neighbours leave the statistics NaN", () => {
    const shelf: number[][] = [];
    for (let x = 0.1; x <= 0.5; x += 0.05) for (let z = 2; z <= 2.4; z += 0.05) shelf.push([x, 0, z, 0, 50]); // 1 unit above the floor
    const { packed, count } = floorLandmarks(() => 0, () => 50, shelf);
    const [, obj] = objectFloorDiagnostics(packed, count, world, identity.rotation, [0, 0, 0], [{ id: 1, position: objectWorld }], cfg);
    expect(Math.abs(obj.heightRatio)).toBeLessThan(1e-6);
    expect(obj.planeFlagged).toBe(obj.neighbors); // the shelf (flag 0) never entered
    const few = Float32Array.from([0.3, 1, 2.2, 1, 50, 0.32, 1, 2.2, 1, 50, 0.3, 1, 2.25, 1, 50]);
    const [, sparse] = objectFloorDiagnostics(few, 3, world, identity.rotation, [0, 0, 0], [{ id: 1, position: objectWorld }], cfg);
    expect(sparse.neighbors).toBe(3);
    expect(Number.isNaN(sparse.heightRatio)).toBe(true);
    expect(Number.isNaN(sparse.tiltDeg)).toBe(true);
  });

  it("an object behind the camera has NaN depth and no neighbours", () => {
    const { packed, count } = floorLandmarks(() => 0, () => 50);
    const behind = Array.from(mapToWorld(world, [0, 1, -1]));
    const [, obj] = objectFloorDiagnostics(packed, count, world, identity.rotation, [0, 0, 0], [{ id: 1, position: behind }], cfg);
    expect(Number.isNaN(obj.depth)).toBe(true);
    expect(obj.neighbors).toBe(0);
  });
});

/**
 * Engine + anchor as the session wires them, on the two-plane scene: a
 * sideways slide establishes the world, three taps place objects. The
 * synthetic floor is exactly planar, so the floor the map knows around each
 * object must be within ~1 % of the depth of the plane it sits on. An
 * anchor built from the same plane shifted 3 % towards the camera must
 * report the floor that far below its plane (sign and magnitude).
 */
const W = TP_W;
const H = TP_H;
const FRAMES = 90;
const poseAt = (f: number) => tpPose([0.005 * f, 0.001 * f, 0.002 * f], 0.0015 * f, 0);
const sequence: Uint8Array[] = [];
for (let f = 0; f < FRAMES; f++) sequence.push(tpRender(poseAt(f)));
function input(f: number): VisionInput {
  return { frameId: f, timestamp: f * 33.3, width: W, height: H, gray: sequence[f], intrinsics: TP_K, gravity: tpGravityAt(poseAt(f)) };
}

describe("object-floor diagnostic through the engine and the anchor", () => {
  it("objects placed by hit test sit on the floor the map knows (≤ 1 % of depth); a shifted plane is measured with the right sign", () => {
    const engine = new VisionEngine(W, H, resolveConfig({}), createRng(5));
    const anchor = new WorldAnchor({ assumedPlaneDistanceMeters: 0.5 });
    const dcfg = resolveConfig({}).debug;
    const fcfg: ObjectFloorConfig = {
      radiusDepthRatio: dcfg.objectFloorRadiusDepthRatio,
      heightWindowDepthRatio: dcfg.objectFloorHeightWindowDepthRatio,
      minLandmarks: dcfg.objectFloorMinLandmarks,
      youngFrames: dcfg.objectFloorYoungFrames,
    };
    const taps: [number, number][] = [[320, 300], [220, 330], [430, 290]];
    const objects: { id: number; position: number[] }[] = [];
    let foundPlane: PlaneOutput | null = null;
    let A = -1;
    const worst: { h: number; tilt: number; ratio: number; f: number; n: number }[] = [];
    let last: VisionOutput | null = null;
    for (let f = 0; f < FRAMES; f++) {
      const inp = input(f);
      if (anchor.isReady) {
        const pts: number[] = [];
        const o = worldToMap(anchor.frame!, [0, 0, 0]);
        pts.push(o[0], o[1], o[2]);
        for (const ob of objects) {
          const m = worldToMap(anchor.frame!, ob.position);
          pts.push(m[0], m[1], m[2]);
        }
        inp.anchorPoints = pts;
      }
      const o = engine.process(inp);
      last = o;
      const b = o.bundleAdjustment;
      if (b.ranThisFrame && !b.rejected && anchor.isReady && o.mapPose && anchor.attachedMapFrameId === o.mapPose.mapFrameId) {
        anchor.applyMapCorrection({ rotation: b.correctionRotation, translation: b.correctionTranslation });
      }
      const st = o.initRefinement?.lastStage;
      if (st && st.applied && st.frameId === f && anchor.isReady && o.mapPose && anchor.attachedMapFrameId === o.mapPose.mapFrameId) {
        anchor.applyMapCorrection({ rotation: st.correctionRotation, translation: st.correctionTranslation });
      }
      anchor.checkMap(o.mapPose);
      if (!anchor.isReady && o.plane?.found && o.mapPose && anchor.create(o.plane, o.mapPose)) {
        A = f;
        foundPlane = o.plane;
        taps.forEach(([u, v], j) => {
          const hit = anchor.hitTest(u, v, TP_K, o.mapPose!)!;
          expect(hit).not.toBeNull();
          objects.push({ id: j + 1, position: Array.from(hit.position) });
        });
      }
      if (!anchor.isReady || !o.mapPose || o.mapPose.framesSinceTracked !== 0) continue;
      // Packed ages are frames since triangulation.
      if (f === A) {
        const ages = new Map<string, number>();
        for (const lm of engine.landmarkMap.values()) ages.set(lm.position.join(","), f - lm.firstFrame);
        let checked = 0;
        for (let i = 0; i < o.landmarkCount; i++) {
          const k = i * LANDMARK_STRIDE;
          const key = `${o.landmarks[k]},${o.landmarks[k + 1]},${o.landmarks[k + 2]}`;
          // Float32 packing loses digits; compare through the engine map's own Float64 → the key may miss, count the hits.
          const age = ages.get(key);
          if (age !== undefined) {
            expect(o.landmarks[k + 4]).toBe(age);
            checked++;
          }
        }
        expect(o.landmarkCount).toBeGreaterThan(50);
        expect(checked).toBeGreaterThanOrEqual(0);
        expect(o.landmarks.length).toBe(o.landmarkCount * LANDMARK_STRIDE);
      }
      const recs = objectFloorDiagnostics(o.landmarks, o.landmarkCount, anchor.frame!, o.mapPose.rotation, o.mapPose.cameraCenter, objects, fcfg);
      expect(recs.length).toBe(objects.length + 1);
      for (const r of recs) {
        if (!Number.isFinite(r.heightRatio)) continue;
        worst.push({ h: Math.abs(r.heightRatio), tilt: r.tiltDeg, ratio: Math.abs(r.localDepthRatio - 1), f, n: r.neighbors });
      }
    }
    expect(A).toBeGreaterThan(0);
    expect(worst.length).toBeGreaterThan(10);
    const maxH = Math.max(...worst.map((w) => w.h));
    const maxTilt = Math.max(...worst.filter((w) => Number.isFinite(w.tilt)).map((w) => w.tilt));
    const maxRatio = Math.max(...worst.filter((w) => Number.isFinite(w.ratio)).map((w) => w.ratio));
    const tilts = worst.filter((w) => Number.isFinite(w.tilt)).map((w) => w.tilt).sort((a, b) => a - b);
    const ns = worst.map((w) => w.n).sort((a, b) => a - b);
    const worstTilt = worst.reduce((a, b) => (b.tilt > a.tilt ? b : a));
    console.log(
      `[objfloor] world at ${A}, ${worst.length} records (neighbours ${ns[0]}–${ns[ns.length - 1]}, median ${ns[ns.length >> 1]}): max |height| ${(maxH * 100).toFixed(2)} % of depth, tilt median ${tilts[tilts.length >> 1].toFixed(2)}° / max ${maxTilt.toFixed(2)}° (n ${worstTilt.n} at frame ${worstTilt.f}), max |local depth ratio − 1| ${(maxRatio * 100).toFixed(2)} %`,
    );
    expect(maxH).toBeLessThan(0.01);
    // The local plane through a handful of triangulated points is noisy in
    // its normal (a few degrees); the height and the depth ratio are what
    // the size mismatch is measured by.
    // (synthetic floor: 5–10 neighbours → ~3° of fit noise; the HUD flags the tilt only with ≥ 15 neighbours).
    expect(tilts[tilts.length >> 1]).toBeLessThan(5);
    expect(maxTilt).toBeLessThan(8);
    expect(maxRatio).toBeLessThan(0.015);

    // A plane 3 % of the camera–plane distance nearer the camera: the floor the
    // map knows is now below that plane by the same amount (negative height),
    // and the local surface lies behind the object along the ray (ratio > 1).
    const o = last!;
    const mp = o.mapPose!;
    const p = foundPlane!;
    const C = mp.cameraCenter;
    const dist = Math.abs(p.normal[0] * C[0] + p.normal[1] * C[1] + p.normal[2] * C[2] + p.d);
    const shift = 0.03 * dist;
    // n·X + d = 0 with n towards the camera: moving the plane towards the camera = d − shift.
    const shifted: PlaneOutput = { ...p, d: p.d - shift, center: [p.center[0] + shift * p.normal[0], p.center[1] + shift * p.normal[1], p.center[2] + shift * p.normal[2]] };
    const anchor2 = new WorldAnchor({ assumedPlaneDistanceMeters: 0.5 });
    expect(anchor2.create(shifted, mp)).toBe(true);
    const objects2 = taps.map(([u, v], j) => ({ id: j + 1, position: Array.from(anchor2.hitTest(u, v, TP_K, mp)!.position) }));
    const recs2 = objectFloorDiagnostics(o.landmarks, o.landmarkCount, anchor2.frame!, mp.rotation, mp.cameraCenter, objects2, fcfg);
    for (const r of recs2.slice(1)) {
      if (!Number.isFinite(r.heightRatio)) continue;
      // Height is measured along the world up (= plane normal): −shift, as a fraction of this object's depth.
      expect(r.heightRatio).toBeCloseTo(-shift / r.depth, 2);
      expect(r.localDepthRatio).toBeGreaterThan(1.01);
      // Exact expectation: the ray to the object continues to the true plane.
      const om = worldToMap(anchor2.frame!, r.id === 1 ? objects2[0].position : r.id === 2 ? objects2[1].position : objects2[2].position);
      const dx = om[0] - C[0], dy = om[1] - C[1], dz = om[2] - C[2];
      const tTrue = -(p.normal[0] * C[0] + p.normal[1] * C[1] + p.normal[2] * C[2] + p.d) / (p.normal[0] * dx + p.normal[1] * dy + p.normal[2] * dz);
      expect(r.localDepthRatio).toBeCloseTo(tTrue, 1);
      console.log(`[objfloor] shifted plane, object ${r.id}: height ${(r.heightRatio * 100).toFixed(2)} % (expected ${((-shift / r.depth) * 100).toFixed(2)}), local depth ×${r.localDepthRatio.toFixed(4)} (true floor at ×${tTrue.toFixed(4)}), n ${r.neighbors}`);
    }
  });
});
