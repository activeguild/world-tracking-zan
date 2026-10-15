import { describe, expect, it } from "vitest";
import { resolveConfig } from "../../src/ar/ARConfig";
import { TrackingState } from "../../src/ar/ARState";
import { WorldAnchor } from "../../src/ar/WorldAnchor";
import { buildWorldDiagnostics, type WorldTrackingDiagnostics, type WorldTransformChangeReason } from "../../src/ar/WorldDiagnostics";
import { cameraPoseToThree, worldToMap, type WorldFrame } from "../../src/math/CoordinateSystem";
import { mat3Multiply } from "../../src/math/Matrix";
import { cameraCenterOf, rotationDistance, type RigidTransform } from "../../src/math/Pose";
import { transpose3 } from "../../src/math/Decomposition";
import { fitPlaneLeastSquares } from "../../src/math/Plane";
import { intersectRayPlane, pixelRay, transformRay } from "../../src/math/Ray";
import { createRng } from "../../src/vision/OutlierRejection";
import type { MapPoseOutput, VisionInput, VisionOutput } from "../../src/vision/types";
import { VisionEngine } from "../../src/vision/VisionEngine";
import { TP_H, TP_K, TP_W, tpFloorPoint, tpGravityAt, tpPose, tpProject, tpRender, type TPPose } from "../helpers/twoPlaneScene";

/**
 * 修正指示書「World Tracking の空間固定性・姿勢安定性」§4, §9.2: the world
 * anchor must not change except for a named reason, placed objects must keep
 * their world transform, and the camera pose estimated through the whole
 * scenario — world init → plane → placement → yaw → pitch → return →
 * translation → (BA on keyframes) → loss → relocalization — must keep the
 * placed points on their true floor pixels.
 *
 * The engine and the anchor are wired exactly as ARSession wires them (the
 * accepted BA correction goes to the anchor once, anchor points go back to
 * the engine every frame, objects are world positions nobody rewrites).
 * ARSession itself needs a DOM; this is the same logic without it.
 */
const W = TP_W;
const H = TP_H;

// Scenario phases (frame ranges) — §9.2.
const SLIDE_END = 90; // world init + plane: sideways slide with a slow yaw
const YAW_END = SLIDE_END + 40; // yaw-only ±15° about the fixed centre
const PITCH_END = YAW_END + 40; // pitch-only ±12°
const RETURN_END = PITCH_END + 10; // back at the slide's last pose
const TRANSLATE_END = RETURN_END + 40; // translation only, 0.16 u sideways
const BLANK_END = TRANSLATE_END + 8; // occlusion: blank frames → lost
const FRAMES = BLANK_END + 45; // back at the last translated pose → relocalize

const base = tpPose([0.005 * SLIDE_END, 0.001 * SLIDE_END, 0.002 * SLIDE_END], 0.0015 * SLIDE_END, 0);
const yaw0 = 0.0015 * SLIDE_END;
const C0 = [0.005 * SLIDE_END, 0.001 * SLIDE_END, 0.002 * SLIDE_END];
const lastTranslated = (() => {
  const s = TRANSLATE_END - 1 - RETURN_END;
  return tpPose([C0[0] + 0.004 * s, C0[1], C0[2] + 0.001 * s], yaw0, 0);
})();

type Phase = "slide" | "yaw" | "pitch" | "return" | "translate" | "blank" | "reloc";
function phaseOf(f: number): Phase {
  if (f < SLIDE_END) return "slide";
  if (f < YAW_END) return "yaw";
  if (f < PITCH_END) return "pitch";
  if (f < RETURN_END) return "return";
  if (f < TRANSLATE_END) return "translate";
  if (f < BLANK_END) return "blank";
  return "reloc";
}

function poseAt(f: number): TPPose {
  switch (phaseOf(f)) {
    case "slide":
      return tpPose([0.005 * f, 0.001 * f, 0.002 * f], 0.0015 * f, 0);
    case "yaw":
      return tpPose(C0, yaw0 + ((15 * Math.PI) / 180) * Math.sin(((f - SLIDE_END) / 40) * 2 * Math.PI), 0);
    case "pitch":
      return tpPose(C0, yaw0, ((12 * Math.PI) / 180) * Math.sin(((f - YAW_END) / 40) * 2 * Math.PI));
    case "return":
      return base;
    case "translate": {
      const s = f - RETURN_END;
      return tpPose([C0[0] + 0.004 * s, C0[1], C0[2] + 0.001 * s], yaw0, 0);
    }
    case "blank":
    case "reloc":
      return lastTranslated;
  }
}

const blank = new Uint8Array(W * H).fill(96);
const sequence: Uint8Array[] = [];
for (let f = 0; f < FRAMES; f++) sequence.push(phaseOf(f) === "blank" ? blank : tpRender(poseAt(f)));

function input(f: number): VisionInput {
  // Gravity as the device reports it: in the camera frame of this frame's pose.
  return { frameId: f, timestamp: f * 33.3, width: W, height: H, gray: sequence[f], intrinsics: TP_K, gravity: tpGravityAt(poseAt(f)) };
}

function rigid(mp: MapPoseOutput): RigidTransform {
  return { rotation: Float64Array.from(mp.rotation), translation: Float64Array.from(mp.translation) };
}

/** Project a world point with the anchor and a map pose (processing pixels). */
function projectWorld(anchor: WorldAnchor, position: ArrayLike<number>, mp: MapPoseOutput): [number, number] {
  const pm = worldToMap(anchor.frame!, position);
  const r = mp.rotation, t = mp.translation;
  const x = r[0] * pm[0] + r[1] * pm[1] + r[2] * pm[2] + t[0];
  const y = r[3] * pm[0] + r[4] * pm[1] + r[5] * pm[2] + t[1];
  const z = r[6] * pm[0] + r[7] * pm[1] + r[8] * pm[2] + t[2];
  return [(x / z) * TP_K.fx + TP_K.cx, (y / z) * TP_K.fy + TP_K.cy];
}

/** Rotation of camera g → camera f (frame independent): R_f R_gᵀ. */
function relativeRotation(rf: ArrayLike<number>, rg: ArrayLike<number>): Float64Array {
  return mat3Multiply(Float64Array.from(rf), transpose3(Float64Array.from(rg)));
}

interface FrameRecord {
  f: number;
  phase: Phase;
  state: TrackingState;
  tracked: boolean;
  mapPose: MapPoseOutput | null;
  diag: WorldTrackingDiagnostics;
  /** Worst placed-point error against its true floor pixel (px), NaN when not tracked. */
  placedErrPx: number;
  /** The same for a point placed on the local landmark surface instead of the global plane (oracle). */
  localErrPx: number;
  /** Decomposition (filled after the run): pose/map error alone, and the placement (plane height) error alone. */
  poseErrPx: number;
  placeErrPx: number;
  threePosition: Float64Array | null;
}

/** Ray through a processing pixel in map coordinates for a map pose. */
function mapRay(u: number, v: number, mp: MapPoseOutput) {
  const cam = rigid(mp);
  const rt = transpose3(cam.rotation);
  const c = cameraCenterOf(cam);
  return transformRay(pixelRay(u, v, TP_K.fx, TP_K.fy, TP_K.cx, TP_K.cy), rt, c);
}

function runScenario(config: Parameters<typeof resolveConfig>[0] = {}) {
  const engine = new VisionEngine(W, H, resolveConfig(config), createRng(5));
  const anchor = new WorldAnchor({ assumedPlaneDistanceMeters: 0.5 });
  const taps: [number, number][] = [[320, 300], [220, 330], [430, 290]];
  let A = -1;
  let mapFrameId = -1;
  const placed: Float64Array[] = [];
  const placedCopies: number[][] = [];
  /** Oracle placement: the same tap ray intersected with the local landmark surface (world coordinates). */
  const placedLocal: Float64Array[] = [];
  /** Placement height diagnostics at frame A (map units, relative to the camera–plane distance). */
  const placement: { rayDepth: number; localOffset: number; planeInliers: number; planeRms: number; localLandmarks: number }[] = [];
  let truth: number[][] = [];
  const records: FrameRecord[] = [];
  let prevWorld: WorldFrame | null = null;
  let prevNormal: number[] | null = null;
  for (let f = 0; f < FRAMES; f++) {
    const inp = input(f);
    if (A >= 0) {
      // Session: anchor points = world origin + placed objects, in the current map frame.
      const pts: number[] = [];
      const o = worldToMap(anchor.frame!, [0, 0, 0]);
      pts.push(o[0], o[1], o[2]);
      for (const p of placed) {
        const m = worldToMap(anchor.frame!, p);
        pts.push(m[0], m[1], m[2]);
      }
      inp.anchorPoints = pts;
    }
    const o: VisionOutput = engine.process(inp);
    let reason: WorldTransformChangeReason = "none";
    // ARSession.handleResult: the accepted BA correction moves the anchor once.
    const b = o.bundleAdjustment;
    if (b.ranThisFrame && !b.rejected && anchor.isReady && o.mapPose && anchor.attachedMapFrameId === o.mapPose.mapFrameId) {
      if (anchor.applyMapCorrection({ rotation: b.correctionRotation, translation: b.correctionTranslation })) reason = "ba_correction";
    }
    // ARSession.updateWorld: drop with the map, create on the first found plane.
    if (anchor.checkMap(o.mapPose)) reason = "map_reset";
    if (!anchor.isReady && o.plane?.found && o.mapPose && anchor.create(o.plane, o.mapPose)) {
      reason = "created";
      A = f;
      mapFrameId = o.mapPose.mapFrameId;
      // Tap → hit test → place (the object keeps that world position forever).
      for (const [u, v] of taps) {
        const hit = anchor.hitTest(u, v, TP_K, o.mapPose)!;
        expect(hit).not.toBeNull();
        placed.push(hit.position);
        placedCopies.push(Array.from(hit.position));
        // Oracle: the landmarks within 80 px of the tap define the surface the
        // map actually knows there; intersect the same ray with their plane.
        const ray = mapRay(u, v, o.mapPose);
        const near: number[] = [];
        const r = o.mapPose.rotation, t = o.mapPose.translation;
        for (const lm of engine.landmarkMap.values()) {
          const X = lm.position;
          const z = r[6] * X[0] + r[7] * X[1] + r[8] * X[2] + t[2];
          if (z <= 1e-6) continue;
          const px = ((r[0] * X[0] + r[1] * X[1] + r[2] * X[2] + t[0]) / z) * TP_K.fx + TP_K.cx;
          const py = ((r[3] * X[0] + r[4] * X[1] + r[5] * X[2] + t[1]) / z) * TP_K.fy + TP_K.cy;
          if (Math.hypot(px - u, py - v) < 80) near.push(X[0], X[1], X[2]);
        }
        const localPlane = fitPlaneLeastSquares(Float64Array.from(near), null, near.length / 3);
        const local = localPlane ? intersectRayPlane(ray, localPlane.normal, localPlane.d) : null;
        expect(local, `local surface at tap (${u}, ${v}): ${near.length / 3} landmarks within 80 px`).not.toBeNull();
        placedLocal.push(anchor.toWorld(local!.point)!);
        const globalHit = intersectRayPlane(ray, anchor.frame!.plane.up, -(anchor.frame!.plane.up[0] * anchor.frame!.plane.origin[0] + anchor.frame!.plane.up[1] * anchor.frame!.plane.origin[1] + anchor.frame!.plane.up[2] * anchor.frame!.plane.origin[2]))!;
        placement.push({ rayDepth: globalHit.t, localOffset: (local!.t - globalHit.t) / globalHit.t, planeInliers: o.plane.inlierCount, planeRms: o.plane.rmsResidual, localLandmarks: near.length / 3 });
      }
      truth = taps.map(([u, v]) => tpFloorPoint(poseAt(A), u, v)!);
    }
    const diag = buildWorldDiagnostics({
      frameId: f,
      quality: o.quality,
      mapPose: o.mapPose,
      plane: o.plane,
      previousPlaneNormal: prevNormal,
      planeRecovery: o.planeRecovery,
      bundleAdjustment: o.bundleAdjustment,
      previousWorld: prevWorld,
      world: anchor.frame,
      changeReason: reason,
    });
    prevWorld = anchor.frame;
    prevNormal = o.plane ? o.plane.normal : null;
    const mp = o.mapPose;
    const tracked = !!mp && mp.framesSinceTracked === 0 && anchor.isReady && o.state !== TrackingState.RELOCALIZING && o.state !== TrackingState.TRACKING_LOST;
    let placedErrPx = NaN, localErrPx = NaN;
    let threePosition: Float64Array | null = null;
    if (tracked && mp) {
      placedErrPx = 0;
      localErrPx = 0;
      for (let j = 0; j < taps.length; j++) {
        const expected = tpProject(poseAt(f), truth[j]);
        if (!expected) continue;
        const p = projectWorld(anchor, placed[j], mp);
        placedErrPx = Math.max(placedErrPx, Math.hypot(p[0] - expected[0], p[1] - expected[1]));
        const q = projectWorld(anchor, placedLocal[j], mp);
        localErrPx = Math.max(localErrPx, Math.hypot(q[0] - expected[0], q[1] - expected[1]));
      }
      threePosition = cameraPoseToThree(anchor.frame!, rigid(mp)).position;
    }
    records.push({ f, phase: phaseOf(f), state: o.state, tracked, mapPose: mp, diag, placedErrPx, localErrPx, poseErrPx: NaN, placeErrPx: NaN, threePosition });
  }
  const last = engine.process(input(FRAMES - 1)); // one more look at the final state

  // Decomposition: the map frame is, by construction, the CV frame of the
  // two-view reference camera (mapFrameId) with |t| = 1 for the baseline to
  // the initialization frame. That similarity to camera-0 coordinates is
  // known exactly from the truth, so the true floor points can be written
  // in map coordinates ("ideal" placement) and each frame's error split
  // into what the estimated pose contributes (ideal point seen through the
  // estimated pose) and what the placement contributes (the rest).
  const initFrame = records.find((r) => r.mapPose !== null)!.f;
  const refPose = poseAt(mapFrameId), initPose = poseAt(initFrame);
  const baseline = Math.hypot(initPose.C[0] - refPose.C[0], initPose.C[1] - refPose.C[1], initPose.C[2] - refPose.C[2]);
  const initCentre = cameraCenterOf(rigid(records[initFrame].mapPose!));
  // Map units per camera-0 unit: estimated init baseline over the true one.
  const scale = Math.hypot(initCentre[0], initCentre[1], initCentre[2]) / baseline;
  const toMap = (p: number[]) => {
    const d = [p[0] - refPose.C[0], p[1] - refPose.C[1], p[2] - refPose.C[2]];
    const r = refPose.R;
    return [(r[0] * d[0] + r[1] * d[1] + r[2] * d[2]) * scale, (r[3] * d[0] + r[4] * d[1] + r[5] * d[2]) * scale, (r[6] * d[0] + r[7] * d[1] + r[8] * d[2]) * scale];
  };
  const ideal = truth.map(toMap);
  // Scale check: |t| = 1 at the init frame by construction; the local scale
  // factor along the trajectory (estimated / true camera-centre moves since
  // the init frame, relative to the init scale) says whether the map scale
  // is conserved while the map grows.
  const initRec = records[initFrame];
  const cInit = cameraCenterOf(rigid(initRec.mapPose!));
  const scaleAt = (f: number): number => {
    const r = records[f];
    if (!r.tracked) return NaN;
    const c = cameraCenterOf(rigid(r.mapPose!));
    const tc = poseAt(r.f).C;
    const est = Math.hypot(c[0] - cInit[0], c[1] - cInit[1], c[2] - cInit[2]);
    const tru = Math.hypot(tc[0] - initPose.C[0], tc[1] - initPose.C[1], tc[2] - initPose.C[2]) * scale;
    return est / tru;
  };
  const scaleCheck = [20, 30, 40, 50, 60, 70, 80, 90, 120, 180, 209, 219].map((f) => `${f}:${scaleAt(f).toFixed(4)}`);
  console.log(`[anchor] |C_est(init)| = ${Math.hypot(cInit[0], cInit[1], cInit[2]).toFixed(4)} (1 from the essential matrix, median depth kept by the refinement); local scale / init scale: ${scaleCheck.join(" ")}`);
  // How far the placed points sit from the true floor, along the tap ray (map units, % of depth) — the placement error itself.
  const idealOffset = placed.map((p, j) => {
    const pm = worldToMap(anchor.frame!, p);
    const o = ideal[j];
    return Math.hypot(pm[0] - o[0], pm[1] - o[1], pm[2] - o[2]) / placement[j].rayDepth;
  });
  const projectMap = (X: number[], mp: MapPoseOutput): [number, number] => {
    const r = mp.rotation, t = mp.translation;
    const x = r[0] * X[0] + r[1] * X[1] + r[2] * X[2] + t[0];
    const y = r[3] * X[0] + r[4] * X[1] + r[5] * X[2] + t[1];
    const z = r[6] * X[0] + r[7] * X[1] + r[8] * X[2] + t[2];
    return [(x / z) * TP_K.fx + TP_K.cx, (y / z) * TP_K.fy + TP_K.cy];
  };
  // World frames per frame are not stored; the placed point in *current* map
  // coordinates is what the session projects — recompute it with the final
  // anchor only for frames after the last BA correction, and with the frame's
  // own anchor otherwise. Simpler and exact: compare in the image through the
  // same pose, placed (as recorded in placedErrPx) vs ideal.
  for (const r of records) {
    if (!r.tracked) continue;
    let poseErr = 0;
    for (let j = 0; j < taps.length; j++) {
      const expected = tpProject(poseAt(r.f), truth[j]);
      if (!expected) continue;
      const p = projectMap(ideal[j], r.mapPose!);
      poseErr = Math.max(poseErr, Math.hypot(p[0] - expected[0], p[1] - expected[1]));
    }
    r.poseErrPx = poseErr;
  }
  return { engine, anchor, A, mapFrameId, initFrame, placed, placedCopies, placement, idealOffset, records, last, mapScale: scale, scaleAt };
}

const scenario = runScenario();
const scenarioNoInitBA = runScenario({ landmarks: { initRefineWithBA: false } });

describe("anchor invariance through the §9.2 scenario", () => {
  const { A, mapFrameId, records, placed, placedCopies, last } = scenario;

  it("A/B: the two-view init refinement removes the map-scale drift along the slide and the placed-point drift with it", () => {
    const frames = [20, 30, 40, 50, 60, 70, 80, 90, 120, 180, 209, 219];
    console.log(`[anchor] scale, init refined:   ${frames.map((f) => `${f}:${scenario.scaleAt(f).toFixed(4)}`).join(" ")}`);
    console.log(`[anchor] scale, init raw:       ${frames.map((f) => `${f}:${scenarioNoInitBA.scaleAt(f).toFixed(4)}`).join(" ")}`);
    const errs = (s: typeof scenario) => s.records.filter((r) => r.tracked && r.f > s.A && r.phase !== "reloc").map((r) => r.placedErrPx).sort((a, b) => a - b);
    const a = errs(scenario), b = errs(scenarioNoInitBA);
    console.log(`[anchor] placed-point error: refined median ${a[a.length >> 1].toFixed(2)} / max ${a[a.length - 1].toFixed(2)} px, raw median ${b[b.length >> 1].toFixed(2)} / max ${b[b.length - 1].toFixed(2)} px`);
    // Raw: the scale read by PnP grows ~4 % along the slide and the placed points drift up to 3.6 px.
    expect(Math.abs(scenarioNoInitBA.scaleAt(90) - 1)).toBeGreaterThan(0.03);
    expect(b[b.length - 1]).toBeGreaterThan(2.5);
    // Refined: scale within 1 % throughout, placed points within the LK noise.
    for (const f of frames) if (Number.isFinite(scenario.scaleAt(f))) expect(Math.abs(scenario.scaleAt(f) - 1)).toBeLessThan(0.01);
    expect(a[a.length - 1]).toBeLessThan(b[b.length - 1] / 2);
  });

  it("establishes the world during the initial slide and never resets the map afterwards", () => {
    expect(A).toBeGreaterThan(0);
    expect(A).toBeLessThan(SLIDE_END);
    for (const r of records) if (r.f > A && r.mapPose) expect(r.mapPose.mapFrameId).toBe(mapFrameId);
    expect(last.mapPose?.mapFrameId).toBe(mapFrameId);
  });

  it("the world transform changes only for a named reason (created once, accepted BA corrections), never silently", () => {
    const changes = records.filter((r) => r.diag.worldTransformChanged);
    const reasons = changes.map((r) => `${r.f}:${r.diag.worldTransformChangeReason}`);
    console.log(`[anchor] world transform changes: ${reasons.join(" ")}`);
    const unexplained = changes.filter((r) => r.diag.worldTransformChangeReason === "none");
    expect(unexplained).toEqual([]);
    expect(changes.filter((r) => r.diag.worldTransformChangeReason === "created").map((r) => r.f)).toEqual([A]);
    expect(changes.some((r) => r.diag.worldTransformChangeReason === "map_reset")).toBe(false);
    // Every other change is an accepted BA run of this very frame, and it is small
    // (a consistent synthetic map needs only refinements, not repairs).
    for (const r of changes) {
      if (r.diag.worldTransformChangeReason !== "ba_correction") continue;
      expect(r.diag.baExecuted && r.diag.baAccepted).toBe(true);
      expect(r.diag.anchorPositionDelta).toBeLessThan(0.05); // m
      expect(r.diag.anchorRotationDeltaDeg).toBeLessThan(2);
    }
    // Frames where BA ran but was rejected did not touch the world.
    for (const r of records) if (r.diag.baExecuted && !r.diag.baAccepted) expect(r.diag.worldTransformChanged).toBe(false);
  });

  it("placed objects keep their world transform; nothing in the session path rewrites them", () => {
    expect(placed.length).toBe(3);
    for (let j = 0; j < placed.length; j++) expect(Array.from(placed[j])).toEqual(placedCopies[j]);
  });

  it("the placed points stay on their true floor pixels through yaw, pitch, return, translation and BA, with sub-pixel hops at BA frames", () => {
    const tracked = records.filter((r) => r.tracked && r.f > A && r.phase !== "reloc");
    const errs = tracked.map((r) => r.placedErrPx).sort((a, b) => a - b);
    const median = errs[errs.length >> 1], max = errs[errs.length - 1];
    const byPhase = (p: Phase) => {
      const e = tracked.filter((r) => r.phase === p).map((r) => r.placedErrPx);
      return e.length ? Math.max(...e) : NaN;
    };
    const hops: number[] = [];
    for (let i = 1; i < tracked.length; i++) {
      if (tracked[i].diag.baExecuted && tracked[i].diag.baAccepted && tracked[i].f === tracked[i - 1].f + 1) hops.push(Math.abs(tracked[i].placedErrPx - tracked[i - 1].placedErrPx));
    }
    console.log(
      `[anchor] placed-point error median ${median.toFixed(2)} / max ${max.toFixed(2)} px (yaw ${byPhase("yaw").toFixed(2)} pitch ${byPhase("pitch").toFixed(2)} return ${byPhase("return").toFixed(2)} translate ${byPhase("translate").toFixed(2)}); BA hops ${hops.map((h) => h.toFixed(2)).join(" ")} px`,
    );
    console.log(
      `[anchor] per-frame total/local-surface/pose-only: ${tracked.map((r) => `${r.f}${r.diag.baExecuted ? (r.diag.baAccepted ? "B" : "b") : ""}:${r.placedErrPx.toFixed(1)}/${r.localErrPx.toFixed(1)}/${r.poseErrPx.toFixed(1)}`).join(" ")}`,
    );
    const { placement, idealOffset, initFrame, mapFrameId, mapScale } = scenario;
    console.log(
      `[anchor] map: ref frame ${mapFrameId}, init frame ${initFrame}, ${mapScale.toFixed(2)} u per camera-0 unit; placement at A: ${placement.map((p, j) => `depth ${p.rayDepth.toFixed(2)}u local-surface offset ${(p.localOffset * 100).toFixed(2)}% (${p.localLandmarks} lm) true-floor offset ${(idealOffset[j] * 100).toFixed(2)}%`).join("; ")}; plane inliers ${placement[0].planeInliers} rms ${placement[0].planeRms.toFixed(4)}u`,
    );
    expect(tracked.length).toBeGreaterThan(150);
    expect(median).toBeLessThan(1.0);
    expect(max).toBeLessThan(2.5);
    for (const h of hops) expect(h).toBeLessThan(1.0);
  });

  it("the Three.js camera pose is finite and continuous on every tracked frame", () => {
    const tracked = records.filter((r) => r.tracked);
    let prev: FrameRecord | null = null;
    for (const r of tracked) {
      for (const v of r.threePosition!) expect(Number.isFinite(v)).toBe(true);
      if (prev && r.f === prev.f + 1) {
        expect(r.diag.cameraRotationDeltaDeg).toBeLessThan(10);
        const d = Math.hypot(r.threePosition![0] - prev.threePosition![0], r.threePosition![1] - prev.threePosition![1], r.threePosition![2] - prev.threePosition![2]);
        expect(d).toBeLessThan(0.05); // 5 cm per frame at most (the scenario moves ≤ 0.4 cm/frame in world meters)
      }
      prev = r;
    }
  });

  it("§4.1 rotation only: the estimated relative rotation follows the truth and the camera centre stays put", () => {
    const ref = records[SLIDE_END - 1];
    expect(ref.tracked).toBe(true);
    let maxRotErr = 0, maxCentreMove = 0, n = 0;
    for (const r of records) {
      if (r.phase !== "yaw" && r.phase !== "pitch") continue;
      if (!r.tracked) continue;
      n++;
      const est = relativeRotation(r.mapPose!.rotation, ref.mapPose!.rotation);
      const tru = relativeRotation(poseAt(r.f).R, poseAt(ref.f).R);
      maxRotErr = Math.max(maxRotErr, (rotationDistance(est, tru) * 180) / Math.PI);
      const c = r.threePosition!, c0 = ref.threePosition!;
      maxCentreMove = Math.max(maxCentreMove, Math.hypot(c[0] - c0[0], c[1] - c0[1], c[2] - c0[2]));
    }
    console.log(`[anchor] rotation-only: ${n} frames, relative rotation error max ${maxRotErr.toFixed(3)}°, camera centre moved max ${(maxCentreMove * 100).toFixed(2)} cm`);
    expect(n).toBeGreaterThan(60);
    expect(maxRotErr).toBeLessThan(1.0);
    expect(maxCentreMove).toBeLessThan(0.02);
  });

  it("§4.2 translation only: the rotation stays, the camera centre moves along the true direction", () => {
    const ref = records[RETURN_END];
    expect(ref.tracked).toBe(true);
    // map → camera-0 rotation from the reference frame: R_true(g)ᵀ R_est(g).
    const toCam0 = mat3Multiply(transpose3(Float64Array.from(poseAt(ref.f).R)), Float64Array.from(ref.mapPose!.rotation));
    const cRef = cameraCenterOf(rigid(ref.mapPose!));
    let maxRotErr = 0, maxDirErr = 0, n = 0;
    const ratios: number[] = [];
    for (const r of records) {
      if (r.phase !== "translate" || !r.tracked || r.f < RETURN_END + 10) continue;
      n++;
      const est = relativeRotation(r.mapPose!.rotation, ref.mapPose!.rotation);
      const tru = relativeRotation(poseAt(r.f).R, poseAt(ref.f).R);
      maxRotErr = Math.max(maxRotErr, (rotationDistance(est, tru) * 180) / Math.PI);
      const c = cameraCenterOf(rigid(r.mapPose!));
      const dm = [c[0] - cRef[0], c[1] - cRef[1], c[2] - cRef[2]];
      const d0 = [
        toCam0[0] * dm[0] + toCam0[1] * dm[1] + toCam0[2] * dm[2],
        toCam0[3] * dm[0] + toCam0[4] * dm[1] + toCam0[5] * dm[2],
        toCam0[6] * dm[0] + toCam0[7] * dm[1] + toCam0[8] * dm[2],
      ];
      const t = poseAt(r.f).C, t0 = poseAt(ref.f).C;
      const dt = [t[0] - t0[0], t[1] - t0[1], t[2] - t0[2]];
      const ne = Math.hypot(d0[0], d0[1], d0[2]), nt = Math.hypot(dt[0], dt[1], dt[2]);
      const cos = (d0[0] * dt[0] + d0[1] * dt[1] + d0[2] * dt[2]) / (ne * nt);
      maxDirErr = Math.max(maxDirErr, (Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI);
      ratios.push(ne / nt);
    }
    const rMin = Math.min(...ratios), rMax = Math.max(...ratios);
    console.log(`[anchor] translation-only: ${n} frames, rotation error max ${maxRotErr.toFixed(3)}°, direction error max ${maxDirErr.toFixed(2)}°, |est|/|true| ${rMin.toFixed(3)}–${rMax.toFixed(3)}`);
    expect(n).toBeGreaterThan(20);
    expect(maxRotErr).toBeLessThan(1.0);
    expect(maxDirErr).toBeLessThan(10);
    // The map scale is arbitrary but constant: the ratio must not drift along the move.
    expect(rMax / rMin).toBeLessThan(1.2);
  });

  it("loss and relocalization: the camera comes back into the same world and the placed points are where they were", () => {
    const blankStates = records.filter((r) => r.phase === "blank").map((r) => r.state);
    expect(blankStates.some((s) => s === TrackingState.TRACKING_LOST || s === TrackingState.RELOCALIZING)).toBe(true);
    const reloc = records.filter((r) => r.phase === "reloc");
    const back = reloc.filter((r) => r.tracked);
    const applied = reloc.filter((r) => r.mapPose?.relocalized).map((r) => r.f);
    console.log(`[anchor] lost states ${[...new Set(blankStates)].join(",")}; relocalized frames ${applied.join(" ")}; tracked again ${back.length}/${reloc.length}; reloc success count ${last.relocalization.successCount}`);
    expect(back.length).toBeGreaterThan(10);
    // Same map, same world: the points placed before the loss project back onto their floor pixels.
    const errs = back.slice(3).map((r) => r.placedErrPx);
    expect(Math.max(...errs)).toBeLessThan(3.0);
    for (const r of reloc) expect(r.diag.worldTransformChangeReason).not.toBe("map_reset");
  });
});
