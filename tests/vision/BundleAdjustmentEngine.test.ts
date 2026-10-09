import { describe, expect, it } from "vitest";
import { resolveConfig } from "../../src/ar/ARConfig";
import { TrackingState } from "../../src/ar/ARState";
import { WorldAnchor } from "../../src/ar/WorldAnchor";
import { worldToMap } from "../../src/math/CoordinateSystem";
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
