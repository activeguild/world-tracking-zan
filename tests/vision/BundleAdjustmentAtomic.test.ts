import { describe, expect, it } from "vitest";
import { resolveConfig } from "../../src/ar/ARConfig";
import { createRng } from "../../src/vision/OutlierRejection";
import type { VisionInput } from "../../src/vision/types";
import { VisionEngine } from "../../src/vision/VisionEngine";
import { TP_H, TP_K, TP_W, tpGravityAt, tpPose, tpRender } from "../helpers/twoPlaneScene";

/**
 * 修正指示書 §5.2 / §7: the bundle adjustment commits as a whole or not at
 * all. A rejected run leaves every landmark and keyframe pose bit-identical;
 * a non-finite value that reaches the map is kept out of the solve and can
 * never be written back into a good landmark or pose.
 */
const W = TP_W;
const H = TP_H;
const FRAMES = 120;
function poseAt(f: number) {
  const s = Math.min(f, 90);
  const pitch = f > 90 ? ((12 * Math.PI) / 180) * Math.sin(((f - 90) / 60) * 2 * Math.PI) : 0;
  return tpPose([0.005 * s, 0.001 * s, 0.002 * s], 0.0015 * s, pitch);
}
const sequence: Uint8Array[] = [];
for (let f = 0; f < FRAMES; f++) sequence.push(tpRender(poseAt(f)));
function input(f: number): VisionInput {
  return { frameId: f, timestamp: f * 33.3, width: W, height: H, gray: sequence[f], intrinsics: TP_K, gravity: tpGravityAt(poseAt(f)) };
}

function snapshot(engine: VisionEngine) {
  const landmarks = new Map<number, number[]>();
  for (const lm of engine.landmarkMap.values()) landmarks.set(lm.id, Array.from(lm.position));
  const poses = engine.keyframes.map((kf) => ({ id: kf.id, rotation: Array.from(kf.pose.rotation), translation: Array.from(kf.pose.translation) }));
  return { landmarks, poses };
}

function expectUnchanged(engine: VisionEngine, before: ReturnType<typeof snapshot>) {
  const after = snapshot(engine);
  expect(after.landmarks.size).toBe(before.landmarks.size);
  for (const [id, p] of before.landmarks) expect(after.landmarks.get(id)).toEqual(p);
  expect(after.poses).toEqual(before.poses);
}

describe("bundle adjustment commits atomically (指示書 §5.2)", () => {
  it("a rejected run (forced no_gain) changes no landmark and no keyframe pose; the next frame tracks as before", () => {
    // BA off during tracking so the run under test is the only one.
    const engine = new VisionEngine(W, H, resolveConfig({ bundleAdjustment: { enabled: false, minGainFraction: 1.0 } }), createRng(5));
    for (let f = 0; f < FRAMES; f++) engine.process(input(f));
    expect(engine.keyframes.length).toBeGreaterThanOrEqual(3);
    const before = snapshot(engine);
    expect(engine.runBundleAdjustment(TP_K)).toBe(true);
    const o = engine.process(input(FRAMES - 1));
    const b = o.bundleAdjustment;
    expect(b.runs).toBe(1);
    expect(b.rejected).toBe(true);
    expect(b.rejectReason).toBe("no_gain");
    expectUnchanged(engine, before);
    for (const lm of engine.landmarkMap.values()) expect(lm.baMode).toBe("none");
    expect(o.mapPose?.framesSinceTracked).toBe(0);
    // The world-anchor correction of a rejected run is the identity.
    expect(Array.from(b.correctionRotation)).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    expect(Array.from(b.correctionTranslation)).toEqual([0, 0, 0]);
  });

  it("an accepted run writes every solved landmark and leaves none half-written", () => {
    const engine = new VisionEngine(W, H, resolveConfig({ bundleAdjustment: { enabled: false } }), createRng(5));
    for (let f = 0; f < FRAMES; f++) engine.process(input(f));
    // Make the run worth accepting: push the free keyframes' landmarks off a little.
    const rng = createRng(3);
    for (const lm of engine.landmarkMap.values()) for (let a = 0; a < 3; a++) lm.position[a] += (rng() - 0.5) * 0.01;
    expect(engine.runBundleAdjustment(TP_K)).toBe(true);
    const o = engine.process(input(FRAMES - 1));
    const b = o.bundleAdjustment;
    expect(b.rejected).toBe(false);
    let adjusted = 0, propagated = 0, none = 0;
    for (const lm of engine.landmarkMap.values()) {
      for (const v of lm.position) expect(Number.isFinite(v)).toBe(true);
      if (lm.baMode === "adjusted") adjusted++;
      else if (lm.baMode === "propagated") propagated++;
      else none++;
    }
    // The three groups partition the map and the counts match the report (minus the purged).
    expect(adjusted + propagated + none).toBe(engine.landmarkMap.size);
    expect(adjusted).toBe(b.landmarks - b.purged);
    expect(propagated).toBe(b.propagated);
    expect(none).toBe(b.untouched);
    expect(b.errorAfterPx).toBeLessThan(b.errorBeforePx);
    for (const kf of engine.keyframes) {
      for (const v of kf.pose.rotation) expect(Number.isFinite(v)).toBe(true);
      for (const v of kf.pose.translation) expect(Number.isFinite(v)).toBe(true);
    }
  });

  it("a NaN landmark in the map is left out of the solve: the run stays finite and no good landmark or pose is poisoned", () => {
    const engine = new VisionEngine(W, H, resolveConfig({ bundleAdjustment: { enabled: false } }), createRng(5));
    for (let f = 0; f < FRAMES; f++) engine.process(input(f));
    const rng = createRng(3);
    for (const lm of engine.landmarkMap.values()) for (let a = 0; a < 3; a++) lm.position[a] += (rng() - 0.5) * 0.01;
    // A landmark observed by the newest keyframe (so it would be in the solve).
    const newest = engine.keyframes[engine.keyframes.length - 1];
    const victim = newest.observations.map((ob) => engine.landmarkMap.get(ob.landmarkId)).find((lm) => lm)!;
    expect(victim).toBeDefined();
    victim.position[0] = NaN;
    const before = snapshot(engine);
    expect(engine.runBundleAdjustment(TP_K)).toBe(true);
    const o = engine.process(input(FRAMES - 1));
    const b = o.bundleAdjustment;
    expect(b.rejectReason).not.toBe("invalid");
    expect(Number.isFinite(b.errorBeforePx) && Number.isFinite(b.errorAfterPx)).toBe(true);
    expect(b.errorAfterPx).toBeLessThan(b.errorBeforePx);
    for (const lm of engine.landmarkMap.values()) {
      if (lm.id === victim.id) continue;
      const wasFinite = before.landmarks.get(lm.id)?.every(Number.isFinite);
      if (wasFinite) for (const v of lm.position) expect(Number.isFinite(v)).toBe(true);
    }
    for (const kf of engine.keyframes) {
      for (const v of kf.pose.rotation) expect(Number.isFinite(v)).toBe(true);
      for (const v of kf.pose.translation) expect(Number.isFinite(v)).toBe(true);
    }
    // The victim was not part of the solve.
    expect(victim.baMode).not.toBe("adjusted");
  });
});
