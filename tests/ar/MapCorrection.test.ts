import { describe, expect, it } from "vitest";
import { resolveConfig } from "../../src/ar/ARConfig";
import { TrackingState } from "../../src/ar/ARState";
import { WorldAnchor } from "../../src/ar/WorldAnchor";
import { worldToMap } from "../../src/math/CoordinateSystem";
import { applyTransform, composeTransforms, invertTransform, rotationAxisAngle, type RigidTransform } from "../../src/math/Pose";
import { createRng } from "../../src/vision/OutlierRejection";
import { emptyRelinkStats, emptyReprojectionBreakdown, type MapPoseOutput, type PlaneOutput, type VisionInput, type VisionOutput } from "../../src/vision/types";
import { VisionEngine } from "../../src/vision/VisionEngine";
import { poseFromCenter } from "../helpers/scene";
import { TP_H, TP_K, TP_W, tpFloorPoint, tpGravity, tpPose, tpProject, tpRender } from "../helpers/twoPlaneScene";

/**
 * 改善指示書 Phase 2: map corrections and the world anchor. A bundle
 * adjustment moves the map under the camera by a rigid C (old → new); the
 * engine re-expresses the canonical pose in the new map and the session
 * applies the same C to the world anchor. Over a session this happens at
 * every accepted run, so (1) corrections must compose exactly, (2) applying
 * each one once must keep placed objects on their true pixels across many
 * runs, and (3) applying one twice or skipping one must be visible — the
 * failure modes the tests are there to catch.
 */
function mapPoseOf(p: RigidTransform, mapFrameId = 0): MapPoseOutput {
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
    relink: emptyRelinkStats(),
    reanchor: null,
    mapInlierCount: 100,
    planeInlierCount: 0,
    rejectReason: null,
    mapCandidate: null,
    planeCandidate: null,
    gateMaxTranslation: 0,
    gateMaxRotationDeg: 0,
    observations: 100,
    requiredInliers: 12,
    recoveryMode: "tracking",
    sourceDeltaTranslation: 0,
    sourceDeltaRotationDeg: 0,
    sourceHistory: "M",
    relocalized: false,
    triangulation: { candidates: 0, parallaxRejected: 0, cheiralityRejected: 0, angleRejected: 0, errorRejected: 0, depthRejected: 0, added: 0 },
    reprojection: emptyReprojectionBreakdown(),
    source: "map",
  };
}

// Desk plane 1.2 map units in front of the init camera, tilted 45°.
const planeNormal = [0, -Math.SQRT1_2, -Math.SQRT1_2];
const planeCenter = [0, 0.85, 0.85];
const plane: PlaneOutput = {
  normal: planeNormal,
  d: -(planeNormal[0] * planeCenter[0] + planeNormal[1] * planeCenter[1] + planeNormal[2] * planeCenter[2]),
  center: planeCenter,
  inlierCount: 200,
  rmsResidual: 0.001,
  areaEstimate: 1,
  horizontalness: 1,
  horizontal: true,
  confidence: 1,
  stableFrames: 10,
  found: true,
  usedGravity: true,
};

function randomCorrection(rng: () => number, maxDeg: number, maxShift: number): RigidTransform {
  const axis = [rng() - 0.5, rng() - 0.5, rng() - 0.5];
  return {
    rotation: rotationAxisAngle(axis, ((rng() * maxDeg) * Math.PI) / 180),
    translation: new Float64Array([(rng() - 0.5) * 2 * maxShift, (rng() - 0.5) * 2 * maxShift, (rng() - 0.5) * 2 * maxShift]),
  };
}

function frameDistance(a: WorldAnchor, b: WorldAnchor): number {
  const fa = a.frame!, fb = b.frame!;
  let d = 0;
  for (let i = 0; i < 9; i++) d = Math.max(d, Math.abs(fa.rotation[i] - fb.rotation[i]));
  for (let i = 0; i < 16; i++) d = Math.max(d, Math.abs(fa.plane.matrix[i] - fb.plane.matrix[i]));
  for (let i = 0; i < 3; i++) d = Math.max(d, Math.abs(fa.plane.origin[i] - fb.plane.origin[i]), Math.abs(fa.plane.up[i] - fb.plane.up[i]));
  return Math.max(d, Math.abs(fa.scale - fb.scale));
}

describe("map corrections on the world anchor (改善指示書 Phase 2)", () => {
  const cam0 = poseFromCenter(rotationAxisAngle([1, 0, 0], 0.0), [0, 0, 0]);

  it("ten corrections applied one after another equal their composition applied once", () => {
    const rng = createRng(21);
    const steps: RigidTransform[] = [];
    for (let i = 0; i < 10; i++) steps.push(randomCorrection(rng, 2, 0.05));
    const sequential = new WorldAnchor({ assumedPlaneDistanceMeters: 0.5 });
    const composed = new WorldAnchor({ assumedPlaneDistanceMeters: 0.5 });
    sequential.create(plane, mapPoseOf(cam0));
    composed.create(plane, mapPoseOf(cam0));
    // Composition: C = C10 ∘ … ∘ C1 (apply C1 first).
    let total: RigidTransform = { rotation: new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]), translation: new Float64Array(3) };
    for (const c of steps) {
      sequential.applyMapCorrection(c);
      total = composeTransforms(c, total);
    }
    composed.applyMapCorrection(total);
    expect(frameDistance(sequential, composed)).toBeLessThan(1e-9);
    // A map point carried through every step keeps its world coordinates.
    const p = [0.2, 0.6, 1.1];
    const fresh = new WorldAnchor({ assumedPlaneDistanceMeters: 0.5 });
    fresh.create(plane, mapPoseOf(cam0));
    const w0 = Array.from(fresh.toWorld(p)!);
    const moved = applyTransform(total, p);
    const w1 = sequential.toWorld(moved)!;
    for (let a = 0; a < 3; a++) expect(w1[a]).toBeCloseTo(w0[a], 9);
  });

  it("stays orthonormal, keeps its scale and keeps world coordinates exact through 500 corrections", () => {
    const rng = createRng(22);
    const anchor = new WorldAnchor({ assumedPlaneDistanceMeters: 0.5 });
    anchor.create(plane, mapPoseOf(cam0));
    const scale0 = anchor.frame!.scale;
    const p = [0.2, 0.6, 1.1];
    const w0 = Array.from(anchor.toWorld(p)!);
    let moved: Float64Array = Float64Array.from(p);
    let cam = cam0;
    const camWorld0 = anchor.cameraPose(mapPoseOf(cam0))!;
    for (let i = 0; i < 500; i++) {
      const c = randomCorrection(rng, 3, 0.05);
      anchor.applyMapCorrection(c);
      moved = applyTransform(c, moved);
      cam = composeTransforms(cam, invertTransform(c));
    }
    expect(anchor.frame!.scale).toBeCloseTo(scale0, 12);
    const r = anchor.frame!.rotation;
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
      const dot = r[i * 3] * r[j * 3] + r[i * 3 + 1] * r[j * 3 + 1] + r[i * 3 + 2] * r[j * 3 + 2];
      expect(dot).toBeCloseTo(i === j ? 1 : 0, 9);
    }
    const w1 = anchor.toWorld(moved)!;
    for (let a = 0; a < 3; a++) expect(w1[a]).toBeCloseTo(w0[a], 8);
    // The camera re-expressed through every correction has the same Three.js pose.
    const camWorld1 = anchor.cameraPose(mapPoseOf(cam))!;
    for (let a = 0; a < 3; a++) expect(camWorld1.position[a]).toBeCloseTo(camWorld0.position[a], 8);
    const q = camWorld1.quaternion, q0 = camWorld0.quaternion;
    expect(Math.abs(q[0] * q0[0] + q[1] * q0[1] + q[2] * q0[2] + q[3] * q0[3])).toBeCloseTo(1, 9);
    // The world → map matrix still inverts the map → world transform.
    const back = worldToMap(anchor.frame!, w1);
    for (let a = 0; a < 3; a++) expect(back[a]).toBeCloseTo(moved[a], 8);
  });
});

/**
 * Engine + anchor over a whole sequence, the way the session wires them:
 * every accepted BA run's correction goes to the anchor once. The map is
 * pushed off twice by a rigid drift (1° / 1 % depth, as in the v18 test) so
 * the following runs produce corrections large enough to see; a point placed
 * before the first drift must stay on its true floor pixel throughout, with
 * no hop at the BA frames. An anchor that applies a correction twice, or
 * never, must be visibly off after the first such run.
 */
const W = TP_W;
const H = TP_H;
const FRAMES = 170;
function poseAt(f: number) {
  const s = Math.min(f, 90);
  const pitch = f > 90 ? ((12 * Math.PI) / 180) * Math.sin(((f - 90) / 60) * 2 * Math.PI) : 0;
  return tpPose([0.005 * s, 0.001 * s, 0.002 * s], 0.0015 * s, pitch);
}
const sequence: Uint8Array[] = [];
for (let f = 0; f < FRAMES; f++) sequence.push(tpRender(poseAt(f)));
function input(f: number): VisionInput {
  return { frameId: f, timestamp: f * 33.3, width: W, height: H, gray: sequence[f], intrinsics: TP_K, gravity: tpGravity };
}

describe("accumulated BA corrections through a session (改善指示書 Phase 2)", () => {
  type Mode = "local" | "keyframe" | "none";
  /**
   * Runs the sequence with the engine in one anchor-correction mode and
   * wires the anchor the way the session does (accepted correction applied
   * once; the anchor points = world origin + placed points go back to the
   * engine every frame). A second anchor applies every correction twice.
   */
  function runSession(mode: Mode) {
    const engine = new VisionEngine(W, H, resolveConfig({ bundleAdjustment: { anchorCorrection: mode } }), createRng(5));
    const once = new WorldAnchor({ assumedPlaneDistanceMeters: 0.5 });
    const twice = new WorldAnchor({ assumedPlaneDistanceMeters: 0.5 });
    const anchors = [once, twice];
    let A = -1;
    const taps: [number, number][] = [[320, 300], [220, 330], [430, 290]];
    let placed: { position: Float64Array }[] = [];
    let truth: number[][] = [];
    const errs: number[][] = [[], []];
    const frameOf: number[] = [];
    const baFrames: { f: number; corr: number; kf: number; local: number; n: number }[] = [];
    const driftAt = new Set([70, 120]);
    let lost = 0;
    const project = (anchor: WorldAnchor, hit: { position: Float64Array }, mp: MapPoseOutput): [number, number] => {
      const pm = worldToMap(anchor.frame!, hit.position);
      const r = mp.rotation, t = mp.translation;
      const x = r[0] * pm[0] + r[1] * pm[1] + r[2] * pm[2] + t[0];
      const y = r[3] * pm[0] + r[4] * pm[1] + r[5] * pm[2] + t[1];
      const z = r[6] * pm[0] + r[7] * pm[1] + r[8] * pm[2] + t[2];
      return [(x / z) * TP_K.fx + TP_K.cx, (y / z) * TP_K.fy + TP_K.cy];
    };
    for (let f = 0; f < FRAMES; f++) {
      const inp: VisionInput = input(f);
      if (A >= 0) {
        // Session: anchor points = world origin and the placed objects, in the current map frame.
        const pts: number[] = [];
        const o = worldToMap(once.frame!, [0, 0, 0]);
        pts.push(o[0], o[1], o[2]);
        for (const p of placed) {
          const m = worldToMap(once.frame!, p.position);
          pts.push(m[0], m[1], m[2]);
        }
        inp.anchorPoints = pts;
      }
      const o: VisionOutput = engine.process(inp);
      if (A < 0 && o.state === TrackingState.PLANE_FOUND) {
        A = f;
        for (const a of anchors) expect(a.create(o.plane!, o.mapPose!)).toBe(true);
        placed = taps.map(([u, v]) => once.hitTest(u, v, TP_K, o.mapPose!)!);
        truth = taps.map(([u, v]) => tpFloorPoint(poseAt(A), u, v)!);
        continue;
      }
      if (A < 0) continue;
      const b = o.bundleAdjustment;
      if (b.ranThisFrame && !b.rejected && o.mapPose && once.attachedMapFrameId === o.mapPose.mapFrameId) {
        const c = { rotation: b.correctionRotation, translation: b.correctionTranslation };
        once.applyMapCorrection(c);
        twice.applyMapCorrection(c);
        twice.applyMapCorrection(c);
        baFrames.push({ f, corr: Math.hypot(c.translation[0], c.translation[1], c.translation[2]), kf: b.keyframeCorrectionShift, local: b.localCorrectionShift, n: b.localFitLandmarks });
      }
      const mp = o.mapPose;
      if (mp && mp.framesSinceTracked > 0) lost++;
      if (mp && mp.framesSinceTracked === 0) {
        frameOf.push(f);
        anchors.forEach((anchor, i) => {
          let worst = 0;
          for (let j = 0; j < taps.length; j++) {
            const expected = tpProject(poseAt(f), truth[j]);
            if (!expected) continue;
            const p = project(anchor, placed[j], mp);
            worst = Math.max(worst, Math.hypot(p[0] - expected[0], p[1] - expected[1]));
          }
          errs[i].push(worst);
        });
      }
      // Push the whole state off rigidly (map, free keyframes, both anchors):
      // self-consistent, but no longer agreeing with the fixed first keyframe —
      // the BA pulls it back and the anchor must come along.
      if (driftAt.has(f)) {
        const dT: RigidTransform = { rotation: rotationAxisAngle([0.1, 1, 0.2], 0.0175), translation: new Float64Array([0.03, -0.01, 0.02]) };
        for (const lm of engine.landmarkMap.values()) lm.position.set(applyTransform(dT, lm.position));
        for (let i = 1; i < engine.keyframes.length; i++) {
          const kf = engine.keyframes[i];
          const p = composeTransforms(kf.pose, invertTransform(dT));
          kf.pose.rotation.set(p.rotation);
          kf.pose.translation.set(p.translation);
        }
        for (const a of anchors) a.applyMapCorrection(dT);
      }
    }
    expect(A).toBeGreaterThan(0);
    const stats = (e: number[]) => {
      const s = [...e].sort((a, b) => a - b);
      return { median: s[s.length >> 1], max: s[s.length - 1] };
    };
    const hops = baFrames.map((b) => {
      const i = frameOf.indexOf(b.f);
      return i > 0 ? Math.abs(errs[0][i] - errs[0][i - 1]) : 0;
    });
    const once$ = stats(errs[0]), twice$ = stats(errs[1]);
    const afterDrift = baFrames.filter((b) => b.f > 70);
    console.log(
      `[correction ${mode}] BA ${baFrames.map((b) => `${b.f}(${b.corr.toFixed(3)}u kf ${b.kf.toFixed(3)} local ${b.local.toFixed(3)} n${b.n})`).join(" ")}; lost ${lost}; ` +
        `error once median ${once$.median.toFixed(2)} / max ${once$.max.toFixed(2)} px, twice ${twice$.median.toFixed(2)} / ${twice$.max.toFixed(2)}; hops ${hops.map((h) => h.toFixed(2)).join(" ")} px; ` +
        `timeline ${frameOf.map((f, i) => (i % 8 === 0 ? `${f}:${errs[0][i].toFixed(1)}` : "")).filter(Boolean).join(" ")}`,
    );
    return { once: once$, twice: twice$, hops, baFrames, afterDrift, lost };
  }

  it("local fit: a placed point stays on its floor pixel through drifts and refinements with sub-pixel hops; applied twice it does not", () => {
    const r = runSession("local");
    expect(r.lost).toBe(0);
    expect(r.afterDrift.length).toBeGreaterThanOrEqual(2);
    expect(Math.max(...r.afterDrift.map((b) => b.corr))).toBeGreaterThan(0.005);
    // Every accepted run after the world existed fitted locally (anchor points were known).
    for (const b of r.baFrames) expect(b.n).toBeGreaterThanOrEqual(10);
    // Measured: local (keyframe rotation + translation from the landmarks
    // next to the objects) median 0.87 / max 1.45 px; a full local rigid fit
    // gave 1.38 / 1.77, the keyframe mode 2.38 / 2.60, none 13.8 / 29.5 once
    // the drifted map is pulled back. The hops at the BA frames stay well
    // under a pixel.
    expect(r.once.median).toBeLessThan(1.2);
    expect(r.once.max).toBeLessThan(2.0);
    for (const h of r.hops) expect(h).toBeLessThan(1.0);
    expect(r.twice.max).toBeGreaterThan(5);
    expect(r.twice.max).toBeGreaterThan(r.once.max * 3);
  });

  it("keyframe correction: no hop by construction but the object keeps each keyframe's pose error; none: off once the map is pulled back", () => {
    const kf = runSession("keyframe");
    const none = runSession("none");
    for (const h of kf.hops) expect(h).toBeLessThan(0.3);
    expect(kf.once.max).toBeLessThan(4.0);
    // The keyframe mode carries each run's pose error into the object (2.4 px here vs 0.9 px local).
    expect(kf.once.median).toBeGreaterThan(2.0);
    expect(none.once.max).toBeGreaterThan(5);
  });
});
