import { describe, expect, it } from "vitest";
import { resolveConfig } from "../../src/ar/ARConfig";
import { approximateIntrinsics } from "../../src/camera/CameraIntrinsics";
import type { Mat3 } from "../../src/math/Matrix";
import { rotationAxisAngle, rotationDistance } from "../../src/math/Pose";
import { createRng } from "../../src/vision/OutlierRejection";
import type { VisionInput, VisionOutput } from "../../src/vision/types";
import { VisionEngine } from "../../src/vision/VisionEngine";
import { makeTexture, sampleBilinear } from "../helpers/synthetic";

/**
 * On-device report (2026-10-08, recording 7): after turning the phone a full
 * circle the HUD kept asking to "return the camera to the previous place",
 * although the camera was back at the original view (keyframe NCC 0.97, LK
 * 130/183 observations) — every PnP candidate failed validation.
 *
 * Scenario: the map tracks the first part of the turn (new landmarks from
 * the lever arm of a hand-held phone), tracking is lost half-way (occlusion /
 * blur), the user completes the turn and ends at the start view. The held
 * pose then points the other way: the relocalization-specific jump limits
 * (v12 §9: 90° / 1.0 × depth vs the held pose) must not reject the correct
 * candidate after such a loss.
 *
 * The scene is a textured floor rendered for an arbitrary camera pose
 * (exact plane-induced projection), so a full turn with a lever arm can be
 * synthesized.
 */
const W = 640;
const H = 480;
const K = approximateIntrinsics(W, H);

const n = [0, Math.SQRT1_2, Math.SQRT1_2]; // floor normal, camera-0 frame
const dPlane = 1; // nᵀX = 1 (1 m along the normal)
const gravity = [0, Math.SQRT1_2, Math.SQRT1_2];
// Floor basis (u, v) and the point under the optical axis.
const planeCenter = [n[0] * dPlane, n[1] * dPlane, n[2] * dPlane];
const uAxis = [1, 0, 0];
const vAxis = [0, -Math.SQRT1_2, Math.SQRT1_2]; // n × u
const PPM = 640; // texture pixels per metre on the floor
const TEX = 3200; // 5 m × 5 m of floor
const floor = makeTexture(TEX, TEX, createRng(9090), [20, 48, 120, 280]);

interface Pose {
  /** World → camera rotation (X_cam = R (X − C)). */
  R: Mat3;
  /** Camera centre in the camera-0 frame. */
  C: number[];
}

function render(p: Pose): Uint8Array {
  const out = new Uint8Array(W * H);
  const { R, C } = p;
  const nC = n[0] * C[0] + n[1] * C[1] + n[2] * C[2];
  for (let y = 0; y < H; y++) {
    const dy = (y - K.cy) / K.fy;
    for (let x = 0; x < W; x++) {
      const dx = (x - K.cx) / K.fx;
      // World direction = Rᵀ · (dx, dy, 1).
      const wx = R[0] * dx + R[3] * dy + R[6];
      const wy = R[1] * dx + R[4] * dy + R[7];
      const wz = R[2] * dx + R[5] * dy + R[8];
      const denom = n[0] * wx + n[1] * wy + n[2] * wz;
      if (denom <= 1e-6) {
        out[y * W + x] = 128;
        continue;
      }
      const s = (dPlane - nC) / denom;
      if (s <= 0) {
        out[y * W + x] = 128;
        continue;
      }
      const px = C[0] + s * wx - planeCenter[0];
      const py = C[1] + s * wy - planeCenter[1];
      const pz = C[2] + s * wz - planeCenter[2];
      const a = px * uAxis[0] + py * uAxis[1] + pz * uAxis[2];
      const b = px * vAxis[0] + py * vAxis[1] + pz * vAxis[2];
      out[y * W + x] = Math.round(sampleBilinear(floor, TEX, TEX, a * PPM + TEX / 2, b * PPM + TEX / 2, 128));
    }
  }
  return out;
}

function input(frameId: number, gray: Uint8Array): VisionInput {
  return { frameId, timestamp: frameId * 33.3, width: W, height: H, gray, intrinsics: K, gravity };
}

const I: Mat3 = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
/** Approach frames as in the other relocalization tests: a slow slide over the floor. */
function approachPose(f: number): Pose {
  return { R: I, C: [0.004 * f, 0.001 * f, 0] };
}
/**
 * Turn about the floor normal through the body axis, the phone held 15 cm in
 * front of it (hand-held lever arm): C(θ) = body + Rot_n(θ) (C₀ − body),
 * R(θ) = Rot_n(θ)ᵀ.
 */
function turnPose(start: Pose, thetaDeg: number): Pose {
  const forward = [0, -Math.SQRT1_2, Math.SQRT1_2]; // camera z projected on the floor
  const body = [start.C[0] - 0.15 * forward[0], start.C[1] - 0.15 * forward[1], start.C[2] - 0.15 * forward[2]];
  const rot = rotationAxisAngle(n, (thetaDeg * Math.PI) / 180);
  const arm = [start.C[0] - body[0], start.C[1] - body[1], start.C[2] - body[2]];
  const C = [
    body[0] + rot[0] * arm[0] + rot[1] * arm[1] + rot[2] * arm[2],
    body[1] + rot[3] * arm[0] + rot[4] * arm[1] + rot[5] * arm[2],
    body[2] + rot[6] * arm[0] + rot[7] * arm[1] + rot[8] * arm[2],
  ];
  // R = Rotᵀ (transpose of the row-major 3×3).
  const R: Mat3 = new Float64Array([rot[0], rot[3], rot[6], rot[1], rot[4], rot[7], rot[2], rot[5], rot[8]]);
  return { R, C };
}

function cameraCenterOf(mp: { rotation: ArrayLike<number>; translation: ArrayLike<number> }): number[] {
  const r = mp.rotation, t = mp.translation;
  return [-(r[0] * t[0] + r[3] * t[1] + r[6] * t[2]), -(r[1] * t[0] + r[4] * t[1] + r[7] * t[2]), -(r[2] * t[0] + r[5] * t[1] + r[8] * t[2])];
}

interface TurnRun {
  lostAtDeg: number;
  /** First angle after the loss at which the map pose was fresh again (-1 = never during the turn). */
  recoveredAtDeg: number;
  /** Held pose vs the start pose on the frame before the recovery (or at 357°). */
  heldRotationDeg: number;
  heldTranslation: number;
  /** Verdicts of the attempts while standing at the start view. */
  trials: string[];
  /** Frame (0-based, in the standing phase) of the first fresh map pose, -1 = never. */
  recoveredStandingAt: number;
  finalRotationDeg: number;
  sameMap: boolean;
}

function runTurn(config = resolveConfig()): TurnRun {
  const engine = new VisionEngine(W, H, config, createRng(31));
  const blank = new Uint8Array(W * H).fill(96);
  let frameId = 0;
  let atStart: VisionOutput | null = null;
  for (let f = 0; f < 45; f++, frameId++) atStart = engine.process(input(frameId, render(approachPose(f))));
  expect(atStart!.worldEstablished).toBe(true);
  const mapFrameId = atStart!.mapPose!.mapFrameId;
  const start = approachPose(44);
  const startRotation: Mat3 = Float64Array.from(atStart!.mapPose!.rotation);
  const startCenter = cameraCenterOf(atStart!.mapPose!);
  const heldVsStart = (mp: NonNullable<VisionOutput["mapPose"]>) => {
    const hc = cameraCenterOf(mp);
    return {
      rot: (rotationDistance(Float64Array.from(mp.rotation), startRotation) * 180) / Math.PI,
      tr: Math.hypot(hc[0] - startCenter[0], hc[1] - startCenter[1], hc[2] - startCenter[2]),
    };
  };

  // Turn to 357° at 3°/frame; frames between 135° and 225° are occluded.
  const stepDeg = 3;
  let lostAtDeg = -1;
  let recoveredAtDeg = -1;
  let held = { rot: 0, tr: 0 };
  let prev: VisionOutput | null = null;
  for (let k = 1; k < 360 / stepDeg; k++, frameId++) {
    const theta = k * stepDeg;
    const occluded = theta > 135 && theta < 225;
    const o = engine.process(input(frameId, occluded ? blank : render(turnPose(start, theta))));
    const fresh = o.mapPose !== null && o.mapPose.framesSinceTracked === 0;
    if (lostAtDeg < 0 && !fresh && o.mapPose) lostAtDeg = theta;
    if (lostAtDeg >= 0 && recoveredAtDeg < 0 && fresh) {
      recoveredAtDeg = theta;
      held = heldVsStart(prev!.mapPose!);
    }
    prev = o;
  }
  if (recoveredAtDeg < 0) held = heldVsStart(prev!.mapPose!);

  // Standing still at the start view.
  let recoveredStandingAt = -1;
  const trials: string[] = [];
  let last: VisionOutput = prev!;
  for (let f = 0; f < 30; f++, frameId++) {
    const o = engine.process(input(frameId, render(turnPose(start, 360))));
    last = o;
    const r = o.relocalization;
    if (r.attempt !== "none" && r.diagnostics?.best) {
      const b = r.diagnostics.best;
      trials.push(
        `${r.attempt}:KF${b.keyframeId}:${b.stage}${b.validation ? ` ${b.validation.rejectReason ?? "ok"} ${b.validation.inliers}i ${b.validation.reprojectionErrorPx.toFixed(1)}px rot ${b.validation.rotationJumpDeg.toFixed(0)}°/${b.validation.maxRotationJumpDeg} tr ${b.validation.translationJump.toFixed(2)}/${b.validation.maxTranslationJump.toFixed(2)}` : ""}`,
      );
    }
    if (recoveredStandingAt < 0 && o.mapPose && o.mapPose.framesSinceTracked === 0) recoveredStandingAt = f;
  }
  return {
    lostAtDeg,
    recoveredAtDeg,
    heldRotationDeg: held.rot,
    heldTranslation: held.tr,
    trials,
    recoveredStandingAt,
    finalRotationDeg: heldVsStart(last.mapPose!).rot,
    sameMap: last.mapPose!.mapFrameId === mapFrameId,
  };
}

describe("Relocalization after a full turn (held pose far from the truth)", () => {
  it("relocalizes at the start view after a turn that was lost half-way (v16)", () => {
    const run = runTurn();
    console.log(
      `[full turn] lost at ${run.lostAtDeg}°, held pose before recovery ${run.heldRotationDeg.toFixed(0)}° / ${run.heldTranslation.toFixed(2)} u from the start, recovered at ${run.recoveredAtDeg}° (standing: ${run.recoveredStandingAt}); trials: ${run.trials.slice(0, 3).join(" | ") || "none"}`,
    );
    // The map tracked the first part of the turn and was lost at the occlusion.
    expect(run.lostAtDeg).toBeGreaterThanOrEqual(120);
    expect(run.lostAtDeg).toBeLessThanOrEqual(150);
    // The held pose points the other way when the camera is back.
    expect(run.heldRotationDeg).toBeGreaterThan(90);
    // Recovery only where a keyframe exists: near the start view.
    expect(run.recoveredAtDeg === -1 || run.recoveredAtDeg >= 330).toBe(true);
    expect(run.recoveredAtDeg >= 330 || run.recoveredStandingAt >= 0, "relocalized back into the same map at the start view").toBe(true);
    expect(run.sameMap).toBe(true);
    expect(run.finalRotationDeg).toBeLessThan(5);
  });

  it("with the jump limits always on (v12 behaviour) the correct candidate is rejected as a rotation jump forever", () => {
    const run = runTurn(resolveConfig({ relocalization: { jumpLimitMaxLostFrames: 1e9 } }));
    console.log(`[full turn, limits always on] recovered at ${run.recoveredAtDeg}° (standing: ${run.recoveredStandingAt}); trials: ${run.trials.slice(0, 2).join(" | ")}`);
    expect(run.heldRotationDeg).toBeGreaterThan(90);
    expect(run.recoveredAtDeg).toBe(-1);
    expect(run.recoveredStandingAt).toBe(-1);
    expect(run.trials.length).toBeGreaterThan(0);
    expect(run.trials.every((t) => t.includes("rotation_jump") || t.includes(":coarse") || t.includes(":lk"))).toBe(true);
    expect(run.trials.some((t) => t.includes("rotation_jump"))).toBe(true);
  });
});
