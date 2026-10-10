import { describe, expect, it } from "vitest";
import { resolveConfig } from "../../src/ar/ARConfig";
import { TrackingState } from "../../src/ar/ARState";
import { WorldAnchor } from "../../src/ar/WorldAnchor";
import { worldToMap } from "../../src/math/CoordinateSystem";
import { applyTransform, composeTransforms, invertTransform, rotationAxisAngle, rotationDistance, type RigidTransform } from "../../src/math/Pose";
import { createRng } from "../../src/vision/OutlierRejection";
import type { VisionInput, VisionOutput } from "../../src/vision/types";
import { VisionEngine } from "../../src/vision/VisionEngine";
import { TP_H, TP_K, TP_W, tpFloorPoint, tpGravity, tpPose, tpProject, tpRender } from "../helpers/twoPlaneScene";

/**
 * Phase 7 in the engine: the local bundle adjustment runs when a keyframe is
 * created, refines the keyframe poses and the landmarks they share, and must
 * (1) not disturb tracking, keyframes or the map identity, (2) repair a map
 * whose landmarks were deliberately corrupted, and (3) keep a placed world
 * point on its true floor pixel across the camera motion.
 */
const W = TP_W;
const H = TP_H;
const FRAMES = 150;
function poseAt(f: number) {
  // Sideways slide with a slow yaw, then a gentle pitch swing (views the map was not built from).
  const s = Math.min(f, 90);
  const pitch = f > 90 ? ((12 * Math.PI) / 180) * Math.sin(((f - 90) / 60) * 2 * Math.PI) : 0;
  return tpPose([0.005 * s, 0.001 * s, 0.002 * s], 0.0015 * s, pitch);
}
const sequence: Uint8Array[] = [];
for (let f = 0; f < FRAMES; f++) sequence.push(tpRender(poseAt(f)));

function input(f: number): VisionInput {
  return { frameId: f, timestamp: f * 33.3, width: W, height: H, gray: sequence[f], intrinsics: TP_K, gravity: tpGravity };
}

function meanErr(outs: VisionOutput[], from: number, to: number): number {
  let s = 0, n = 0;
  for (let f = from; f < to; f++) {
    const mp = outs[f]?.mapPose;
    if (mp && mp.framesSinceTracked === 0) {
      s += mp.meanReprojectionErrorPx;
      n++;
    }
  }
  return n ? s / n : NaN;
}

describe("local bundle adjustment in the engine (Phase 7)", () => {
  it("runs on keyframe creation, converges, and leaves tracking / keyframes / map identity intact", () => {
    const engine = new VisionEngine(W, H, resolveConfig(), createRng(5));
    const outs: VisionOutput[] = [];
    const runFrames: number[] = [];
    let lost = 0;
    for (let f = 0; f < FRAMES; f++) {
      const o = engine.process(input(f));
      outs.push(o);
      const ba = o.bundleAdjustment;
      if (ba.ranThisFrame) {
        runFrames.push(f);
        expect(ba.lastFrameId).toBe(f);
        // The solver minimizes the Huber cost, not the mean |error|; on an already
        // consistent synthetic map the mean may move by a few hundredths of a px.
        expect(ba.errorAfterPx).toBeLessThanOrEqual(ba.errorBeforePx + 0.05);
        expect(ba.errorAfterPx).toBeLessThan(1.0);
        expect(ba.freeKeyframes).toBe(ba.keyframes - 1);
        expect(ba.landmarks).toBeGreaterThan(20);
        expect(ba.observations).toBeGreaterThan(40);
        expect(engine.timing.ba).toBeGreaterThan(0);
        // The frame of the run and the next one are tracked from the map (no jump rejection, no loss).
        expect(o.mapPose?.framesSinceTracked).toBe(0);
      } else {
        expect(engine.timing.ba).toBe(0);
      }
      if (o.mapPose?.framesSinceTracked !== 0 && o.mapPose) lost++;
    }
    const last = outs[FRAMES - 1].bundleAdjustment;
    console.log(
      `[ba-engine] runs ${last.runs} at frames ${runFrames.join(",")}; last: kf ${last.keyframes} lm ${last.landmarks} obs ${last.observations} (+${last.outliers} out) ${last.errorBeforePx.toFixed(2)} → ${last.errorAfterPx.toFixed(2)} px in ${last.iterations} it, ${last.ms.toFixed(1)} ms, shift lm ${last.maxLandmarkShift.toFixed(4)} kf ${last.maxKeyframeShift.toFixed(4)} u / ${last.maxKeyframeRotationDeg.toFixed(2)}°; lost frames ${lost}`,
    );
    expect(last.runs).toBeGreaterThanOrEqual(2);
    expect(runFrames.length).toBe(last.runs);
    // Map identity is unchanged (no reset) and the keyframe count matches the relocalizer.
    const mapIds = new Set(outs.filter((o) => o.mapPose).map((o) => o.mapPose!.mapFrameId));
    expect(mapIds.size).toBe(1);
    expect(last.keyframes).toBe(engine.keyframeCount);
    expect(outs.some((o) => o.state === TrackingState.PLANE_FOUND)).toBe(true);
    expect(lost).toBeLessThanOrEqual(2);
    for (const f of runFrames) expect(outs[f + 1]?.mapPose?.framesSinceTracked ?? 0).toBe(0);
  });

  it("repairs a deliberately corrupted map: the PnP error rises with the corruption and comes back after one run", () => {
    const engine = new VisionEngine(W, H, resolveConfig({ bundleAdjustment: { enabled: false } }), createRng(5));
    const outs: VisionOutput[] = [];
    let f = 0;
    for (; f < 80; f++) outs.push(engine.process(input(f)));
    expect(outs.some((o) => o.state === TrackingState.PLANE_FOUND)).toBe(true);
    expect(engine.keyframeCount).toBeGreaterThanOrEqual(2);
    const baseline = meanErr(outs, 60, 80);
    // Corrupt every landmark by a random ±1.5% of its distance (≈ 2–4 px at f ≈ 640).
    const rng = createRng(99);
    let n = 0;
    for (const lm of engine.landmarkMap.values()) {
      const d = Math.hypot(lm.position[0], lm.position[1], lm.position[2]);
      for (let a = 0; a < 3; a++) lm.position[a] += (rng() - 0.5) * 0.03 * d;
      n++;
    }
    expect(n).toBeGreaterThan(100);
    for (; f < 95; f++) outs.push(engine.process(input(f)));
    const corrupted = meanErr(outs, 82, 95);
    // Repair.
    expect(engine.runBundleAdjustment(TP_K)).toBe(true);
    const ba = engine.bundleAdjustment;
    for (; f < 120; f++) outs.push(engine.process(input(f)));
    const repaired = meanErr(outs, 97, 120);
    console.log(
      `[ba-engine] PnP error baseline ${baseline.toFixed(2)} → corrupted ${corrupted.toFixed(2)} → repaired ${repaired.toFixed(2)} px; BA ${ba.errorBeforePx.toFixed(2)} → ${ba.errorAfterPx.toFixed(2)} px over ${ba.landmarks} lm / ${ba.observations} obs (+${ba.outliers} out) in ${ba.iterations} it`,
    );
    expect(corrupted).toBeGreaterThan(baseline + 1.0);
    expect(ba.errorAfterPx).toBeLessThan(ba.errorBeforePx / 2);
    expect(repaired).toBeLessThan(baseline + 0.4);
    expect(repaired).toBeLessThan(corrupted / 2);
    // Tracking continued through the corruption and the repair (no reset).
    expect(new Set(outs.filter((o) => o.mapPose).map((o) => o.mapPose!.mapFrameId)).size).toBe(1);
  });

  it("rotation-dominant keyframes (recording 20): the priors keep the solve from sliding landmarks along their rays", () => {
    // A hand-held yaw sweep with a 5 cm lever arm: keyframes every ~10° of
    // rotation with almost no baseline between them, like the Android session
    // where 25–60% of the landmarks moved > 20% of their depth per run.
    const frames = 160;
    const seq: Uint8Array[] = [];
    const forward = [0, -Math.SQRT1_2, Math.SQRT1_2];
    for (let f = 0; f < frames; f++) {
      const yaw = ((35 * Math.PI) / 180) * Math.sin((f / frames) * 2 * Math.PI);
      const arm = 0.05;
      const C = [arm * Math.sin(yaw) * forward[2], 0, arm * (1 - Math.cos(yaw)) * 0.5 + 0.001 * f];
      seq.push(tpRender(tpPose(C, yaw, 0)));
    }
    const run = (priors: boolean) => {
      const engine = new VisionEngine(
        W,
        H,
        resolveConfig(
          priors
            ? {}
            : { bundleAdjustment: { landmarkPriorDepthRatio: 0, posePriorTranslationDepthRatio: 0, posePriorRotationDeg: 0, maxShiftedFraction: 1 } },
        ),
        createRng(5),
      );
      let runs = 0, shifted = 0, landmarks = 0, rejected = 0, noGain = 0, maxKfShift = 0;
      let errSum = 0, errN = 0;
      for (let f = 0; f < frames; f++) {
        const o = engine.process({ frameId: f, timestamp: f * 33.3, width: W, height: H, gray: seq[f], intrinsics: TP_K, gravity: tpGravity });
        const ba = o.bundleAdjustment;
        if (ba.ranThisFrame) {
          runs++;
          shifted += ba.shifted;
          landmarks += ba.landmarks;
          // Only the shift rejection is the subject here; on a consistent
          // synthetic map a run may legitimately be dropped for no gain (v18).
          if (ba.rejected && ba.rejectReason === "shift") rejected++;
          if (ba.rejected && ba.rejectReason === "no_gain") noGain++;
          maxKfShift = Math.max(maxKfShift, ba.maxKeyframeShift);
        }
        if (f > frames / 2 && o.mapPose?.framesSinceTracked === 0) {
          errSum += o.mapPose.meanReprojectionErrorPx;
          errN++;
        }
      }
      return { runs, shiftedFraction: landmarks ? shifted / landmarks : 0, rejected, noGain, maxKfShift, pnpErr: errSum / Math.max(1, errN) };
    };
    const withPriors = run(true);
    const without = run(false);
    console.log(
      `[ba-engine] rotation sweep: with priors runs ${withPriors.runs} shifted ${(withPriors.shiftedFraction * 100).toFixed(1)}% rejected ${withPriors.rejected} (no gain ${withPriors.noGain}) kf shift max ${withPriors.maxKfShift.toFixed(3)} u PnP ${withPriors.pnpErr.toFixed(2)} px | without priors shifted ${(without.shiftedFraction * 100).toFixed(1)}% kf shift max ${without.maxKfShift.toFixed(3)} u PnP ${without.pnpErr.toFixed(2)} px`,
    );
    expect(withPriors.runs).toBeGreaterThanOrEqual(2);
    expect(withPriors.shiftedFraction).toBeLessThan(0.05);
    expect(withPriors.rejected).toBe(0);
    expect(withPriors.pnpErr).toBeLessThan(1.0);
    // The unregularized solve lets more landmarks slide on this motion.
    expect(without.shiftedFraction).toBeGreaterThanOrEqual(withPriors.shiftedFraction);
  });

  it("v18: every landmark is solved or carried with its reference keyframe; a run that gains nothing is dropped", () => {
    const engine = new VisionEngine(W, H, resolveConfig(), createRng(5));
    let accepted = 0, noGain = 0, shiftRejected = 0;
    let lastAccepted: VisionOutput["bundleAdjustment"] | null = null;
    const outs: VisionOutput[] = [];
    for (let f = 0; f < FRAMES; f++) {
      const o = engine.process(input(f));
      outs.push(o);
      const ba = o.bundleAdjustment;
      if (!ba.ranThisFrame) continue;
      if (!ba.rejected) {
        accepted++;
        lastAccepted = ba;
        // Accepted runs cut the keyframe-observation error by the configured fraction.
        expect(ba.errorAfterPx).toBeLessThanOrEqual(ba.errorBeforePx * 0.85 + 1e-9);
        // The three groups partition the map.
        expect(ba.landmarks + ba.propagated + ba.untouched).toBe(engine.landmarkCount);
        // Only landmarks whose reference keyframe is the fixed first one (no
        // correction) may stay untouched: nothing is left behind otherwise.
        for (const lm of engine.landmarkMap.values()) {
          if (lm.baMode === "none") expect(lm.refKeyframeId).toBe(engine.keyframes[0].id);
        }
      } else if (ba.rejectReason === "no_gain") {
        noGain++;
        expect(ba.errorAfterPx).toBeGreaterThan(ba.errorBeforePx * 0.85 - 1e-9);
      } else {
        shiftRejected++;
      }
    }
    const rb = outs[FRAMES - 1].mapPose!.reprojection;
    console.log(
      `[ba-engine] v18 runs accepted ${accepted} no-gain ${noGain} shift ${shiftRejected}; last accepted lm ${lastAccepted?.landmarks} prop ${lastAccepted?.propagated} none ${lastAccepted?.untouched}; final PnP groups adj ${rb.adjustedErrorPx.toFixed(2)} (${rb.adjustedCount}) prop ${rb.propagatedErrorPx.toFixed(2)} (${rb.propagatedCount}) none ${rb.untouchedErrorPx.toFixed(2)} (${rb.untouchedCount})`,
    );
    expect(shiftRejected).toBe(0);
    expect(accepted + noGain).toBeGreaterThanOrEqual(2);
    // Every landmark has a reference keyframe once the map exists.
    for (const lm of engine.landmarkMap.values()) expect(lm.refKeyframeId).toBeGreaterThanOrEqual(1);
    // The PnP breakdown accounts every inlier to one of the three groups.
    const mp = outs[FRAMES - 1].mapPose!;
    expect(rb.adjustedCount + rb.propagatedCount + rb.untouchedCount).toBe(mp.inlierCount);
    // After an accepted run the solved group and the rest agree (no split map).
    if (lastAccepted && rb.adjustedCount > 10 && rb.propagatedCount + rb.untouchedCount > 10) {
      const rest = (rb.propagatedErrorPx * rb.propagatedCount + rb.untouchedErrorPx * rb.untouchedCount) / (rb.propagatedCount + rb.untouchedCount);
      expect(Math.abs(rest - rb.adjustedErrorPx)).toBeLessThan(0.5);
    }
  });

  it("v18: landmarks no keyframe observes move with their reference keyframe when the BA corrects its pose", () => {
    const engine = new VisionEngine(W, H, resolveConfig({ bundleAdjustment: { enabled: false } }), createRng(5));
    const outs: VisionOutput[] = [];
    let f = 0;
    for (; f < 80; f++) outs.push(engine.process(input(f)));
    expect(engine.keyframes.length).toBeGreaterThanOrEqual(3);
    const baseline = meanErr(outs, 60, 80);
    const kfs = engine.keyframes;
    // The free keyframe with the most landmarks that reference it and that no
    // other keyframe observes.
    const exclusiveOf = (L: (typeof kfs)[number]) => {
      const seenElsewhere = new Set<number>();
      for (const kf of kfs) if (kf !== L) for (const o of kf.observations) seenElsewhere.add(o.landmarkId);
      return [...engine.landmarkMap.values()].filter((lm) => lm.refKeyframeId === L.id && !seenElsewhere.has(lm.id));
    };
    let L = kfs[1];
    let refL = exclusiveOf(L);
    for (let i = 2; i < kfs.length; i++) {
      const cand = exclusiveOf(kfs[i]);
      if (cand.length > refL.length) {
        L = kfs[i];
        refL = cand;
      }
    }
    expect(refL.length).toBeGreaterThan(8);
    // Half of them lose their keyframe observation: no keyframe observes them
    // any more, so the solve cannot touch them — propagation must.
    const orphan = new Set<number>();
    refL.forEach((lm, i) => {
      if (i % 2 === 0) orphan.add(lm.id);
    });
    L.observations = L.observations.filter((o) => !orphan.has(o.landmarkId));
    // Perturb L and, consistently, every landmark that references it: the
    // sub-map built from L is off by ΔT (0.5°, 0.5% of the depth ≈ 5 px, under
    // the 10 px gross-outlier gate so L's other observations still pull it back).
    const depth = Math.hypot(L.pose.translation[0], L.pose.translation[1], L.pose.translation[2]) + 1;
    const dT: RigidTransform = { rotation: rotationAxisAngle([0.2, 1, 0.1], 0.0087), translation: new Float64Array([0.005 * depth, -0.002 * depth, 0.003 * depth]) };
    const tOld: RigidTransform = { rotation: Float64Array.from(L.pose.rotation), translation: Float64Array.from(L.pose.translation) };
    const tPert = composeTransforms(dT, tOld);
    // X' = T_old⁻¹ ΔT⁻¹ T_old X keeps X' at the same image position in the perturbed L.
    const Cpert = composeTransforms(invertTransform(tOld), composeTransforms(invertTransform(dT), tOld));
    const original = new Map<number, Float64Array>();
    for (const lm of engine.landmarkMap.values()) {
      if (lm.refKeyframeId !== L.id) continue;
      original.set(lm.id, Float64Array.from(lm.position));
      lm.position.set(applyTransform(Cpert, lm.position));
    }
    L.pose.rotation.set(tPert.rotation);
    L.pose.translation.set(tPert.translation);
    expect(engine.runBundleAdjustment(TP_K)).toBe(true);
    const ba = engine.bundleAdjustment;
    expect(ba.rejected).toBe(false);
    expect(ba.propagated).toBeGreaterThanOrEqual(orphan.size - 2);
    // L moved back toward its true pose (the other keyframes' observations pull it).
    const rotBack = (rotationDistance(L.pose.rotation, tOld.rotation) * 180) / Math.PI;
    const rotPert = (rotationDistance(tPert.rotation, tOld.rotation) * 180) / Math.PI;
    // Orphans were carried with L: they stay consistent with L's new pose at
    // their original image positions, and they came back toward the truth.
    let maxOrphanErrPx = 0, orphanBack = 0, orphanBefore = 0, n = 0;
    const r = L.pose.rotation, t = L.pose.translation;
    for (const id of orphan) {
      const lm = engine.landmarkMap.get(id);
      if (!lm) continue;
      expect(lm.baMode).toBe("propagated");
      const p0 = original.get(id)!;
      // The original observation of this landmark in L: its projection through the unperturbed pose.
      const zo = tOld.rotation[6] * p0[0] + tOld.rotation[7] * p0[1] + tOld.rotation[8] * p0[2] + tOld.translation[2];
      const uo = (tOld.rotation[0] * p0[0] + tOld.rotation[1] * p0[1] + tOld.rotation[2] * p0[2] + tOld.translation[0]) / zo;
      const vo = (tOld.rotation[3] * p0[0] + tOld.rotation[4] * p0[1] + tOld.rotation[5] * p0[2] + tOld.translation[1]) / zo;
      const p = lm.position;
      const z = r[6] * p[0] + r[7] * p[1] + r[8] * p[2] + t[2];
      const u = (r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + t[0]) / z;
      const v = (r[3] * p[0] + r[4] * p[1] + r[5] * p[2] + t[1]) / z;
      maxOrphanErrPx = Math.max(maxOrphanErrPx, Math.hypot(u - uo, v - vo) * TP_K.fx);
      orphanBack += Math.hypot(p[0] - p0[0], p[1] - p0[1], p[2] - p0[2]);
      const pp = applyTransform(Cpert, p0);
      orphanBefore += Math.hypot(pp[0] - p0[0], pp[1] - p0[1], pp[2] - p0[2]);
      n++;
    }
    orphanBack /= n;
    orphanBefore /= n;
    for (; f < 110; f++) outs.push(engine.process(input(f)));
    const after = meanErr(outs, 85, 110);
    console.log(
      `[ba-engine] v18 propagation: L rotation off ${rotPert.toFixed(2)}° → ${rotBack.toFixed(2)}° after BA; ${n} orphans carried, image error vs L ${maxOrphanErrPx.toFixed(2)} px max, distance to truth ${orphanBefore.toFixed(4)} → ${orphanBack.toFixed(4)} u; PnP ${baseline.toFixed(2)} → ${after.toFixed(2)} px; propagated ${ba.propagated} untouched ${ba.untouched}`,
    );
    expect(rotBack).toBeLessThan(rotPert * 0.5);
    expect(maxOrphanErrPx).toBeLessThan(0.05);
    expect(orphanBack).toBeLessThan(orphanBefore * 0.6);
    expect(after).toBeLessThan(baseline + 0.4);
  });

  it("v19: a solve that moves a keyframe beyond its prior is rejected as pose_jump, and a repeat offender is struck out", () => {
    // Tight guard so that a 0.8° correction counts as a jump (prior σ 3° × 0.1 = 0.3°).
    const engine = new VisionEngine(W, H, resolveConfig({ bundleAdjustment: { enabled: false, maxPoseJumpPriorRatio: 0.1, maxPoseJumpStrikes: 2 } }), createRng(5));
    const outs: VisionOutput[] = [];
    let f = 0;
    for (; f < 80; f++) outs.push(engine.process(input(f)));
    const kfs = engine.keyframes;
    expect(kfs.length).toBeGreaterThanOrEqual(3);
    const L = kfs[kfs.length - 1];
    // L and the landmarks that reference it are off by 0.8° / 0.4% depth, as in the
    // propagation test: the other keyframes' observations pull L back by more than 0.3°.
    const depth = Math.hypot(L.pose.translation[0], L.pose.translation[1], L.pose.translation[2]) + 1;
    const dT: RigidTransform = { rotation: rotationAxisAngle([0.2, 1, 0.1], 0.014), translation: new Float64Array([0.004 * depth, -0.002 * depth, 0.002 * depth]) };
    const tOld: RigidTransform = { rotation: Float64Array.from(L.pose.rotation), translation: Float64Array.from(L.pose.translation) };
    const tPert = composeTransforms(dT, tOld);
    const Cpert = composeTransforms(invertTransform(tOld), composeTransforms(invertTransform(dT), tOld));
    for (const lm of engine.landmarkMap.values()) if (lm.refKeyframeId === L.id) lm.position.set(applyTransform(Cpert, lm.position));
    L.pose.rotation.set(tPert.rotation);
    L.pose.translation.set(tPert.translation);
    const before = new Map<number, Float64Array>();
    for (const lm of engine.landmarkMap.values()) before.set(lm.id, Float64Array.from(lm.position));
    // Run 1: rejected, blamed on L, nothing written back.
    expect(engine.runBundleAdjustment(TP_K)).toBe(true);
    let ba = engine.bundleAdjustment;
    expect(ba.rejected).toBe(true);
    expect(ba.rejectReason).toBe("pose_jump");
    expect(ba.jumpKeyframeId).toBe(L.id);
    expect(ba.jumpRotationDeg).toBeGreaterThan(0.3);
    expect(ba.excludedKeyframes).toBe(0);
    expect((rotationDistance(L.pose.rotation, tPert.rotation) * 180) / Math.PI).toBeLessThan(1e-4);
    for (const lm of engine.landmarkMap.values()) {
      const p0 = before.get(lm.id)!;
      expect(Math.hypot(lm.position[0] - p0[0], lm.position[1] - p0[1], lm.position[2] - p0[2])).toBeLessThan(1e-12);
    }
    expect(engine.bundleAdjustmentExcludedKeyframes.size).toBe(0);
    // Run 2: second strike → L is struck out of later solves.
    expect(engine.runBundleAdjustment(TP_K)).toBe(true);
    ba = engine.bundleAdjustment;
    expect(ba.rejectReason).toBe("pose_jump");
    expect(ba.jumpKeyframeId).toBe(L.id);
    expect(engine.bundleAdjustmentExcludedKeyframes.has(L.id)).toBe(true);
    // Run 3: solved without L — accepted or dropped for no gain, but no pose jump; L's pose untouched.
    expect(engine.runBundleAdjustment(TP_K)).toBe(true);
    ba = engine.bundleAdjustment;
    expect(ba.rejectReason).not.toBe("pose_jump");
    expect(ba.keyframes).toBe(kfs.length - 1);
    expect(ba.excludedKeyframes).toBe(1);
    expect((rotationDistance(L.pose.rotation, tPert.rotation) * 180) / Math.PI).toBeLessThan(1e-4);
    console.log(`[ba-engine] v19 pose jump: KF${ba.jumpKeyframeId === -1 ? L.id : ba.jumpKeyframeId} struck out after 2 strikes; run 3 ${ba.rejected ? `rejected (${ba.rejectReason})` : "accepted"} over ${ba.keyframes} keyframes`);
    // With the default ratio (2 × prior) the same perturbation is not a jump.
    const relaxed = new VisionEngine(W, H, resolveConfig({ bundleAdjustment: { enabled: false } }), createRng(5));
    for (let g = 0; g < 80; g++) relaxed.process(input(g));
    const L2 = relaxed.keyframes[relaxed.keyframes.length - 1];
    const t2 = composeTransforms(dT, { rotation: Float64Array.from(L2.pose.rotation), translation: Float64Array.from(L2.pose.translation) });
    L2.pose.rotation.set(t2.rotation);
    L2.pose.translation.set(t2.translation);
    expect(relaxed.runBundleAdjustment(TP_K)).toBe(true);
    expect(relaxed.bundleAdjustment.rejectReason).not.toBe("pose_jump");
  });

  it("v18: the world anchor follows the BA correction, so a placed point does not hop on screen when the map moves", () => {
    const engine = new VisionEngine(W, H, resolveConfig({ bundleAdjustment: { enabled: false } }), createRng(5));
    const outs: VisionOutput[] = [];
    let f = 0;
    for (; f < 80; f++) outs.push(engine.process(input(f)));
    const foundIdx = outs.findIndex((o) => o.state === TrackingState.PLANE_FOUND);
    expect(foundIdx).toBeGreaterThan(0);
    const A = foundIdx + 2;
    const makeAnchor = () => {
      const a = new WorldAnchor({ assumedPlaneDistanceMeters: 0.5 });
      expect(a.create(outs[A].plane!, outs[A].mapPose!)).toBe(true);
      return a;
    };
    const follow = makeAnchor();
    const stay = makeAnchor();
    const placed = follow.hitTest(320, 300, TP_K, outs[A].mapPose!)!;
    const placedStay = stay.hitTest(320, 300, TP_K, outs[A].mapPose!)!;
    // The whole state built so far — landmarks, free keyframes, both anchors —
    // drifts rigidly by ΔT (1°, 1% depth): a map that is self-consistent but
    // no longer agrees with the fixed first keyframe. BA pulls it back.
    const dT: RigidTransform = { rotation: rotationAxisAngle([0.1, 1, 0.2], 0.0175), translation: new Float64Array([0.03, -0.01, 0.02]) };
    for (const lm of engine.landmarkMap.values()) lm.position.set(applyTransform(dT, lm.position));
    for (let i = 1; i < engine.keyframes.length; i++) {
      const kf = engine.keyframes[i];
      const p = composeTransforms(kf.pose, invertTransform(dT));
      kf.pose.rotation.set(p.rotation);
      kf.pose.translation.set(p.translation);
    }
    follow.applyMapCorrection(dT);
    stay.applyMapCorrection(dT);
    const project = (anchor: WorldAnchor, hit: { position: Float64Array }, mp: NonNullable<VisionOutput["mapPose"]>): [number, number] => {
      const pm = worldToMap(anchor.frame!, hit.position);
      const r = mp.rotation, t = mp.translation;
      const x = r[0] * pm[0] + r[1] * pm[1] + r[2] * pm[2] + t[0];
      const y = r[3] * pm[0] + r[4] * pm[1] + r[5] * pm[2] + t[1];
      const z = r[6] * pm[0] + r[7] * pm[1] + r[8] * pm[2] + t[2];
      return [(x / z) * TP_K.fx + TP_K.cx, (y / z) * TP_K.fy + TP_K.cy];
    };
    // Track into the drifted map (the PnP follows the moved landmarks), then
    // run BA in the frame of a new keyframe so the correction is this frame's.
    for (; f < 92; f++) outs.push(engine.process(input(f)));
    expect(outs[f - 1].mapPose!.framesSinceTracked).toBe(0);
    const F = f - 1;
    const before = project(follow, placed, outs[F].mapPose!);
    const beforeStay = project(stay, placedStay, outs[F].mapPose!);
    // Force a keyframe at F so the BA correction belongs to this frame's pose.
    engine.createKeyframeForTests(outs[F].timestamp);
    expect(engine.keyframes[engine.keyframes.length - 1].frameId).toBe(F);
    expect(engine.runBundleAdjustment(TP_K)).toBe(true);
    const ba = engine.bundleAdjustment;
    expect(ba.rejected).toBe(false);
    const corr = Math.hypot(ba.correctionTranslation[0], ba.correctionTranslation[1], ba.correctionTranslation[2]);
    expect(corr).toBeGreaterThan(0.005);
    follow.applyMapCorrection({ rotation: ba.correctionRotation, translation: ba.correctionTranslation });
    // The rebased pose of frame F with the corrected anchor projects the
    // placed point where it was; the uncorrected anchor shows the hop.
    const cp = engine.currentMapPose;
    const rebased = { ...outs[F].mapPose!, rotation: Array.from(cp.rotation), translation: Array.from(cp.translation) };
    const afterFollow = project(follow, placed, rebased);
    const afterStay = project(stay, placedStay, rebased);
    const hopFollow = Math.hypot(afterFollow[0] - before[0], afterFollow[1] - before[1]);
    const hopStay = Math.hypot(afterStay[0] - beforeStay[0], afterStay[1] - beforeStay[1]);
    // And the next tracked frame continues from there.
    const o = engine.process(input(f));
    expect(o.mapPose!.framesSinceTracked).toBe(0);
    const next = project(follow, placed, o.mapPose!);
    const truthF = tpProject(poseAt(F), tpFloorPoint(poseAt(F), 320, 300)!)!;
    void truthF;
    const expectedMove = ((): number => {
      const X = tpFloorPoint(poseAt(A), 320, 300)!;
      const a = tpProject(poseAt(F), X)!, b = tpProject(poseAt(f), X)!;
      return Math.hypot(b[0] - a[0], b[1] - a[1]);
    })();
    const moveFollow = Math.hypot(next[0] - before[0], next[1] - before[1]);
    console.log(
      `[ba-engine] v18 anchor follow: correction ${corr.toFixed(4)} u, hop at the BA frame ${hopFollow.toFixed(2)} px (anchor follows) vs ${hopStay.toFixed(2)} px (anchor stays); next frame moved ${moveFollow.toFixed(2)} px, true floor motion ${expectedMove.toFixed(2)} px`,
    );
    expect(hopFollow).toBeLessThan(0.05);
    expect(hopStay).toBeGreaterThan(1.0);
    expect(Math.abs(moveFollow - expectedMove)).toBeLessThan(1.5);
  });

  it("a placed world point stays on its true floor pixel through the motion, with BA at least as well as without", () => {
    const drift = (enabled: boolean): { median: number; max: number } => {
      const engine = new VisionEngine(W, H, resolveConfig({ bundleAdjustment: { enabled } }), createRng(5));
      const anchor = new WorldAnchor({ assumedPlaneDistanceMeters: 0.5 });
      const outs: VisionOutput[] = [];
      for (let f = 0; f < FRAMES; f++) outs.push(engine.process(input(f)));
      const foundIdx = outs.findIndex((o) => o.state === TrackingState.PLANE_FOUND);
      expect(foundIdx).toBeGreaterThan(0);
      const A = foundIdx + 2;
      expect(anchor.create(outs[A].plane!, outs[A].mapPose!)).toBe(true);
      const taps: [number, number][] = [[320, 300], [220, 330], [430, 290]];
      const placed = taps.map(([u, v]) => anchor.hitTest(u, v, TP_K, outs[A].mapPose!)!);
      const truth = taps.map(([u, v]) => tpFloorPoint(poseAt(A), u, v)!);
      const errs: number[] = [];
      for (let B = A + 1; B < FRAMES; B++) {
        const mp = outs[B].mapPose!;
        if (mp.framesSinceTracked !== 0) continue;
        for (let j = 0; j < taps.length; j++) {
          const expected = tpProject(poseAt(B), truth[j]);
          if (!expected) continue;
          const pm = worldToMap(anchor.frame!, placed[j].position);
          const r = mp.rotation, t = mp.translation;
          const x = r[0] * pm[0] + r[1] * pm[1] + r[2] * pm[2] + t[0];
          const y = r[3] * pm[0] + r[4] * pm[1] + r[5] * pm[2] + t[1];
          const z = r[6] * pm[0] + r[7] * pm[1] + r[8] * pm[2] + t[2];
          errs.push(Math.hypot((x / z) * TP_K.fx + TP_K.cx - expected[0], (y / z) * TP_K.fy + TP_K.cy - expected[1]));
        }
      }
      errs.sort((a, b) => a - b);
      return { median: errs[errs.length >> 1], max: errs[errs.length - 1] };
    };
    const withBa = drift(true);
    const without = drift(false);
    console.log(`[ba-engine] placed-point drift: with BA median ${withBa.median.toFixed(2)} / max ${withBa.max.toFixed(2)} px, without ${without.median.toFixed(2)} / ${without.max.toFixed(2)} px`);
    expect(withBa.median).toBeLessThan(1.0);
    expect(withBa.max).toBeLessThan(4.0);
    // On a synthetic map that is consistent to begin with, BA cannot improve the
    // placement; it must not make it materially worse either.
    expect(withBa.median).toBeLessThanOrEqual(without.median + 0.3);
    expect(withBa.max).toBeLessThanOrEqual(without.max + 1.0);
  });
});
