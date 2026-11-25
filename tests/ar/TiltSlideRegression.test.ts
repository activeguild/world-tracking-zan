import { describe, expect, it } from "vitest";
import { resolveConfig } from "../../src/ar/ARConfig";
import { TrackingState } from "../../src/ar/ARState";
import { WorldAnchor } from "../../src/ar/WorldAnchor";
import { buildWorldDiagnostics, type WorldTrackingDiagnostics, type WorldTransformChangeReason } from "../../src/ar/WorldDiagnostics";
import { cameraPoseToThree, worldFromPlane, worldToMap, type WorldFrame } from "../../src/math/CoordinateSystem";
import { transpose3 } from "../../src/math/Decomposition";
import { mat3Multiply, type Mat3 } from "../../src/math/Matrix";
import { rotationDistance, type RigidTransform } from "../../src/math/Pose";
import { createRng } from "../../src/vision/OutlierRejection";
import type { MapPoseOutput, VisionInput, VisionOutput } from "../../src/vision/types";
import { VisionEngine } from "../../src/vision/VisionEngine";
import { TP_H, TP_K, TP_W, tpFloorD, tpFloorNormal, tpFloorPoint, tpGravityAt, tpPose, tpProject, tpRender, type TPPose } from "../helpers/twoPlaneScene";

/**
 * 修正指示書「スマホ傾斜時にワールド固定オブジェクトの回転が徐々にずれる」§6-B.
 *
 * A static world (textured floor + back wall), a known camera trajectory:
 * world init → place three objects → then, three times over, pitch the camera
 * down 15°, slide left/right while tilted, roll ±10°, pitch back, hold the
 * start pose. The second cycle has 5 blank frames (occlusion) in the slide.
 *
 * Measured per frame against the truth: the orientation of the rendered
 * camera in the world (= the orientation the fixed objects appear to have,
 * since nothing rewrites them), split into yaw about the world up axis and
 * tilt of the up axis; the placed points on their true floor pixels; the
 * world position when the camera is back at the start pose. Expected: no
 * growth cycle over cycle, the return pose within tolerance, the anchor
 * never recreated — not even through the loss.
 *
 * Wired as ARSession wires it (accepted BA / init-stage corrections to the
 * anchor once, anchor points back to the engine, objects never rewritten).
 */
const W = TP_W;
const H = TP_H;

const INIT_END = 80;
const PITCH_N = 25, SLIDE_N = 40, ROLL_N = 16, RETURN_N = 16, HOLD_N = 8;
const CYCLE = PITCH_N + SLIDE_N + ROLL_N + RETURN_N + HOLD_N;
const CYCLES = 3;
const FRAMES = INIT_END + CYCLES * CYCLE;
const PITCH = (15 * Math.PI) / 180;
const ROLL = (10 * Math.PI) / 180;
const SLIDE = 0.15;
const BLANK_CYCLE = 1;
const BLANK_FROM = PITCH_N + 8, BLANK_TO = PITCH_N + 13; // 5 frames of the slide

const yaw0 = 0.0015 * (INIT_END - 1);
const C0 = [0.005 * (INIT_END - 1), 0.001 * (INIT_END - 1), 0.002 * (INIT_END - 1)];
const base = tpPose(C0, yaw0, 0);

type Phase = "init" | "pitch" | "slide" | "blank" | "roll" | "return" | "hold";
function cycleOf(f: number): number {
  return f < INIT_END ? -1 : Math.floor((f - INIT_END) / CYCLE);
}
function phaseOf(f: number): Phase {
  if (f < INIT_END) return "init";
  const s = (f - INIT_END) % CYCLE;
  if (s < PITCH_N) return "pitch";
  if (s < PITCH_N + SLIDE_N) return cycleOf(f) === BLANK_CYCLE && s >= BLANK_FROM && s < BLANK_TO ? "blank" : "slide";
  if (s < PITCH_N + SLIDE_N + ROLL_N) return "roll";
  if (s < PITCH_N + SLIDE_N + ROLL_N + RETURN_N) return "return";
  return "hold";
}
function poseAt(f: number): TPPose {
  if (f < INIT_END) return tpPose([0.005 * f, 0.001 * f, 0.002 * f], 0.0015 * f, 0);
  const s = (f - INIT_END) % CYCLE;
  if (s < PITCH_N) return tpPose(C0, yaw0, (PITCH * (1 - Math.cos((Math.PI * (s + 1)) / PITCH_N))) / 2);
  if (s < PITCH_N + SLIDE_N) {
    const u = (s - PITCH_N + 1) / SLIDE_N;
    return tpPose([C0[0] + SLIDE * Math.sin(2 * Math.PI * u), C0[1], C0[2] + 0.03 * Math.sin(2 * Math.PI * u)], yaw0, PITCH);
  }
  if (s < PITCH_N + SLIDE_N + ROLL_N) {
    const u = (s - PITCH_N - SLIDE_N + 1) / ROLL_N;
    return tpPose(C0, yaw0, PITCH, ROLL * Math.sin(Math.PI * u));
  }
  if (s < PITCH_N + SLIDE_N + ROLL_N + RETURN_N) {
    const u = (s - PITCH_N - SLIDE_N - ROLL_N + 1) / RETURN_N;
    return tpPose(C0, yaw0, (PITCH * (1 + Math.cos(Math.PI * u))) / 2);
  }
  return base;
}

const blank = new Uint8Array(W * H).fill(96);
const sequence: Uint8Array[] = [];
for (let f = 0; f < FRAMES; f++) sequence.push(phaseOf(f) === "blank" ? blank : tpRender(poseAt(f)));

function input(f: number): VisionInput {
  return { frameId: f, timestamp: f * 33.3, width: W, height: H, gray: sequence[f], intrinsics: TP_K, gravity: tpGravityAt(poseAt(f)) };
}
function rigid(mp: MapPoseOutput): RigidTransform {
  return { rotation: Float64Array.from(mp.rotation), translation: Float64Array.from(mp.translation) };
}
function trueRigid(p: TPPose): RigidTransform {
  const r = p.R, c = p.C;
  return { rotation: Float64Array.from(r), translation: new Float64Array([-(r[0] * c[0] + r[1] * c[1] + r[2] * c[2]), -(r[3] * c[0] + r[4] * c[1] + r[5] * c[2]), -(r[6] * c[0] + r[7] * c[1] + r[8] * c[2])]) };
}
function projectWorld(anchor: WorldAnchor, position: ArrayLike<number>, mp: MapPoseOutput): [number, number] {
  const pm = worldToMap(anchor.frame!, position);
  const r = mp.rotation, t = mp.translation;
  const x = r[0] * pm[0] + r[1] * pm[1] + r[2] * pm[2] + t[0];
  const y = r[3] * pm[0] + r[4] * pm[1] + r[5] * pm[2] + t[1];
  const z = r[6] * pm[0] + r[7] * pm[1] + r[8] * pm[2] + t[2];
  return [(x / z) * TP_K.fx + TP_K.cx, (y / z) * TP_K.fy + TP_K.cy];
}
/** world ← CV camera rotation: world axes are the rows of R_wm (map coords), camera axes the rows of R_cam. */
function worldFromCamera(rWm: Mat3, rCam: ArrayLike<number>): Mat3 {
  return mat3Multiply(rWm, transpose3(Float64Array.from(rCam)));
}

interface Rec {
  f: number;
  phase: Phase;
  cycle: number;
  state: TrackingState;
  tracked: boolean;
  diag: WorldTrackingDiagnostics;
  mapFrameId: number;
  /** Orientation error of the rendered camera in the world vs the truth (deg) and its yaw / tilt parts. */
  orientErrDeg: number;
  yawErrDeg: number;
  tiltErrDeg: number;
  /** Worst placed-point error against its true floor pixel (px). */
  placedErrPx: number;
  /** Three.js camera position (world meters). */
  threePosition: Float64Array | null;
}

function run() {
  const engine = new VisionEngine(W, H, resolveConfig(), createRng(5));
  const anchor = new WorldAnchor({ assumedPlaneDistanceMeters: 0.5 });
  const taps: [number, number][] = [[320, 300], [220, 330], [430, 290]];
  let A = -1;
  let mapFrameId = -1;
  const placed: Float64Array[] = [];
  const placedCopies: number[][] = [];
  let truth: number[][] = [];
  // True world frame: the true floor, "forward" from the camera at the placement frame.
  let rWt: Mat3 | null = null;
  let K: Mat3 | null = null; // estimated world ← true world (fixed at A)
  const records: Rec[] = [];
  let prevWorld: WorldFrame | null = null;
  let prevNormal: number[] | null = null;
  for (let f = 0; f < FRAMES; f++) {
    const inp = input(f);
    if (A >= 0) {
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
    const b = o.bundleAdjustment;
    if (b.ranThisFrame && !b.rejected && anchor.isReady && o.mapPose && anchor.attachedMapFrameId === o.mapPose.mapFrameId) {
      if (anchor.applyMapCorrection({ rotation: b.correctionRotation, translation: b.correctionTranslation })) reason = "ba_correction";
    }
    const st = o.initRefinement?.lastStage;
    if (st && st.applied && st.frameId === f && anchor.isReady && o.mapPose && anchor.attachedMapFrameId === o.mapPose.mapFrameId) {
      if (anchor.applyMapCorrection({ rotation: st.correctionRotation, translation: st.correctionTranslation })) reason = "init_stage";
    }
    if (anchor.checkMap(o.mapPose)) reason = "map_reset";
    if (!anchor.isReady && o.plane?.found && o.mapPose && anchor.create(o.plane, o.mapPose)) {
      reason = "created";
      A = f;
      mapFrameId = o.mapPose.mapFrameId;
      for (const [u, v] of taps) {
        const hit = anchor.hitTest(u, v, TP_K, o.mapPose)!;
        expect(hit).not.toBeNull();
        placed.push(hit.position);
        placedCopies.push(Array.from(hit.position));
      }
      truth = taps.map(([u, v]) => tpFloorPoint(poseAt(A), u, v)!);
      const floorCenter = [tpFloorNormal[0] * tpFloorD, tpFloorNormal[1] * tpFloorD, tpFloorNormal[2] * tpFloorD];
      rWt = worldFromPlane(tpFloorNormal, floorCenter, trueRigid(poseAt(A)), 1).rotation;
      const E = worldFromCamera(anchor.frame!.rotation, o.mapPose.rotation);
      const T = worldFromCamera(rWt, poseAt(A).R);
      K = mat3Multiply(E, transpose3(T));
    }
    const diag = buildWorldDiagnostics({
      frameId: f,
      quality: o.quality,
      mapPose: o.mapPose,
      plane: o.plane,
      previousPlaneNormal: prevNormal,
      planeRecovery: o.planeRecovery,
      bundleAdjustment: o.bundleAdjustment,
      initStage: o.initRefinement?.lastStage ?? null,
      previousWorld: prevWorld,
      world: anchor.frame,
      changeReason: reason,
    });
    prevWorld = anchor.frame;
    prevNormal = o.plane ? o.plane.normal : null;
    const mp = o.mapPose;
    const tracked = !!mp && mp.framesSinceTracked === 0 && anchor.isReady && o.state !== TrackingState.RELOCALIZING && o.state !== TrackingState.TRACKING_LOST;
    let orientErrDeg = NaN, yawErrDeg = NaN, tiltErrDeg = NaN, placedErrPx = NaN;
    let threePosition: Float64Array | null = null;
    if (tracked && mp && K && rWt) {
      const E = worldFromCamera(anchor.frame!.rotation, mp.rotation);
      const T = mat3Multiply(K, worldFromCamera(rWt, poseAt(f).R));
      orientErrDeg = (rotationDistance(E, T) * 180) / Math.PI;
      // D = E Tᵀ: yaw = rotation of the world X axis about the up axis, tilt = angle of the up axis.
      const D = mat3Multiply(E, transpose3(T));
      yawErrDeg = (Math.atan2(-D[6], D[0]) * 180) / Math.PI;
      tiltErrDeg = (Math.acos(Math.max(-1, Math.min(1, D[4]))) * 180) / Math.PI;
      placedErrPx = 0;
      for (let j = 0; j < taps.length; j++) {
        const expected = tpProject(poseAt(f), truth[j]);
        if (!expected) continue;
        const p = projectWorld(anchor, placed[j], mp);
        placedErrPx = Math.max(placedErrPx, Math.hypot(p[0] - expected[0], p[1] - expected[1]));
      }
      threePosition = cameraPoseToThree(anchor.frame!, rigid(mp)).position;
    }
    records.push({ f, phase: phaseOf(f), cycle: cycleOf(f), state: o.state, tracked, diag, mapFrameId: mp?.mapFrameId ?? -1, orientErrDeg, yawErrDeg, tiltErrDeg, placedErrPx, threePosition });
  }
  return { engine, anchor, A, mapFrameId, placed, placedCopies, records };
}

const S = run();

const fmt = (v: number, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : "-");
const maxOf = (xs: number[]) => (xs.length ? Math.max(...xs) : NaN);
const medianOf = (xs: number[]) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length ? s[s.length >> 1] : NaN;
};

describe("tilt + slide + roll regression (傾斜時の回転ずれ §6-B)", () => {
  const { A, mapFrameId, records, placed, placedCopies, anchor } = S;
  const after = records.filter((r) => r.tracked && r.f > A);

  it("the world is created once during the init slide; the map is never reset and the anchor never recreated, not even through the occlusion", () => {
    expect(A).toBeGreaterThan(0);
    expect(A).toBeLessThan(INIT_END);
    const changes = records.filter((r) => r.diag.worldTransformChanged);
    const reasons = changes.map((r) => `${r.f}:${r.diag.worldTransformChangeReason}`);
    console.log(`[tilt] world created at ${A}; world transform changes: ${reasons.join(" ")}`);
    expect(changes.filter((r) => r.diag.worldTransformChangeReason === "created").map((r) => r.f)).toEqual([A]);
    expect(changes.filter((r) => r.diag.worldTransformChangeReason === "none")).toEqual([]);
    expect(changes.some((r) => r.diag.worldTransformChangeReason === "map_reset")).toBe(false);
    for (const r of records) if (r.f > A && r.mapFrameId >= 0) expect(r.mapFrameId).toBe(mapFrameId);
    expect(anchor.attachedMapFrameId).toBe(mapFrameId);
    // Every accepted correction is a refinement, not a repair.
    for (const r of changes) {
      if (r.diag.worldTransformChangeReason === "created") continue;
      expect(r.diag.anchorPositionDelta).toBeLessThan(0.05);
      expect(r.diag.anchorRotationDeltaDeg).toBeLessThan(2);
    }
    // The occlusion was a real loss (lost / relocalizing states appeared) and tracking resumed in the same map.
    const blankStates = records.filter((r) => r.phase === "blank").map((r) => r.state);
    const lostSeen = records.some((r) => r.cycle === BLANK_CYCLE && (r.state === TrackingState.TRACKING_LOST || r.state === TrackingState.RELOCALIZING));
    console.log(`[tilt] blank states ${[...new Set(blankStates)].join(",")}; loss seen in cycle ${BLANK_CYCLE}: ${lostSeen}; tracked frames after A: ${after.length}/${records.length - A - 1}`);
    expect(lostSeen).toBe(true);
    expect(after.filter((r) => r.cycle === BLANK_CYCLE && r.phase === "hold").length).toBeGreaterThan(0);
    // Objects are never rewritten.
    for (let j = 0; j < placed.length; j++) expect(Array.from(placed[j])).toEqual(placedCopies[j]);
  });

  it("the objects' apparent orientation in the world stays within tolerance through pitch, tilted slide and roll — yaw and tilt alike", () => {
    const byPhase = (p: Phase, pick: (r: Rec) => number) => fmt(maxOf(after.filter((r) => r.phase === p).map(pick)));
    console.log(
      `[tilt] orientation error (deg): max ${fmt(maxOf(after.map((r) => r.orientErrDeg)), 3)} median ${fmt(medianOf(after.map((r) => r.orientErrDeg)), 3)}; by phase pitch ${byPhase("pitch", (r) => r.orientErrDeg)} slide ${byPhase("slide", (r) => r.orientErrDeg)} roll ${byPhase("roll", (r) => r.orientErrDeg)} return ${byPhase("return", (r) => r.orientErrDeg)} hold ${byPhase("hold", (r) => r.orientErrDeg)}; |yaw| max ${fmt(maxOf(after.map((r) => Math.abs(r.yawErrDeg))), 3)} tilt max ${fmt(maxOf(after.map((r) => r.tiltErrDeg)), 3)}`,
    );
    console.log(`[tilt] per frame: ${after.map((r) => `${r.f}${r.diag.baExecuted ? (r.diag.baAccepted ? "B" : "b") : ""}${r.diag.initStageExecuted ? "S" : ""}:${fmt(r.orientErrDeg)}/${fmt(r.yawErrDeg)}/${fmt(r.placedErrPx, 1)}`).join(" ")}`);
    // Measured on the first run: max 0.100°, median 0.078°, |yaw| 0.065°, tilt 0.095°;
    // the limits are ~3× that so that a real drift (degrees, as on device) fails loudly.
    expect(after.length).toBeGreaterThan(250);
    expect(maxOf(after.map((r) => r.orientErrDeg))).toBeLessThan(0.3);
    expect(medianOf(after.map((r) => r.orientErrDeg))).toBeLessThan(0.15);
    expect(maxOf(after.map((r) => Math.abs(r.yawErrDeg)))).toBeLessThan(0.3);
    expect(maxOf(after.map((r) => r.tiltErrDeg))).toBeLessThan(0.3);
  });

  it("the placed points stay on their true floor pixels (also after the loss)", () => {
    const errs = after.map((r) => r.placedErrPx);
    const byCycle = [0, 1, 2].map((c) => fmt(maxOf(after.filter((r) => r.cycle === c).map((r) => r.placedErrPx))));
    console.log(`[tilt] placed-point error median ${fmt(medianOf(errs))} / max ${fmt(maxOf(errs))} px; max per cycle ${byCycle.join(" ")}`);
    // Measured: median 0.50 / max 1.05 px (the LK noise floor of the synthetic scene).
    expect(medianOf(errs)).toBeLessThan(1.0);
    expect(maxOf(errs)).toBeLessThan(2.0);
  });

  it("back at the start pose after every cycle: orientation and world position return within tolerance and do not accumulate", () => {
    const ref = records[INIT_END - 1]; // the same true pose as every hold frame
    expect(ref.tracked).toBe(true);
    const holds = [0, 1, 2].map((c) => after.filter((r) => r.cycle === c && r.phase === "hold"));
    const orient = holds.map((h) => maxOf(h.map((r) => r.orientErrDeg)));
    const yaw = holds.map((h) => maxOf(h.map((r) => Math.abs(r.yawErrDeg))));
    const pos = holds.map((h) => maxOf(h.map((r) => Math.hypot(r.threePosition![0] - ref.threePosition![0], r.threePosition![1] - ref.threePosition![1], r.threePosition![2] - ref.threePosition![2]))));
    console.log(`[tilt] return (hold) per cycle: orientation ${orient.map((v) => fmt(v, 3)).join(" / ")}°, |yaw| ${yaw.map((v) => fmt(v, 3)).join(" / ")}°, world position vs frame ${ref.f} ${pos.map((v) => fmt(v * 100)).join(" / ")} cm; ref frame orientation error ${fmt(ref.orientErrDeg, 3)}°`);
    // Measured: 0.080 / 0.088 / 0.090° and 0.09 / 0.11 / 0.10 cm.
    for (let c = 0; c < CYCLES; c++) {
      expect(holds[c].length).toBeGreaterThan(0);
      expect(orient[c]).toBeLessThan(0.25);
      expect(pos[c]).toBeLessThan(0.005);
    }
    // No accumulation: the third return is not worse than the first by more than the frame noise.
    expect(orient[2]).toBeLessThan(orient[0] + 0.1);
    expect(pos[2]).toBeLessThan(pos[0] + 0.002);
  });

  it("the orientation error does not grow cycle over cycle", () => {
    const perCycle = [0, 1, 2].map((c) => maxOf(after.filter((r) => r.cycle === c && r.phase !== "hold").map((r) => r.orientErrDeg)));
    console.log(`[tilt] max orientation error per cycle: ${perCycle.map((v) => fmt(v, 3)).join(" / ")}°`);
    // Measured: 0.082 / 0.100 / 0.100°.
    expect(perCycle[2]).toBeLessThan(perCycle[0] + 0.1);
    expect(perCycle[1]).toBeLessThan(perCycle[0] + 0.1);
  });
});
