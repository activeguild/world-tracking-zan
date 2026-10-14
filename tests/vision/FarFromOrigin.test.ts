import { describe, expect, it } from "vitest";
import { resolveConfig } from "../../src/ar/ARConfig";
import { TrackingState } from "../../src/ar/ARState";
import { createRng } from "../../src/vision/OutlierRejection";
import type { VisionOutput } from "../../src/vision/types";
import { VisionEngine } from "../../src/vision/VisionEngine";
import { TP_H, TP_K, TP_W, tpGravity, tpPose, tpRender } from "../helpers/twoPlaneScene";

/**
 * 改善指示書 §4 follow-up: the on-device PnP error grew with the camera's
 * distance from the initialization view and fell back when the camera
 * returned to it. The map unit is the initialization baseline, so a camera
 * half a metre from the origin is 5–15 map units away; with the previous
 * rotation Jacobian (−[X_cam]× instead of −[R X]×) the LM step was no longer
 * a descent direction there and PnP left a biased pose when a frame moved
 * more than a few pixels. This walks the camera far out at a medium speed
 * and reads the PnP error over the far half.
 */
const W = TP_W;
const H = TP_H;
const FRAMES = 140;
function poseAt(f: number) {
  // Slow start (so the map initializes with a short baseline = small map
  // unit), then a medium-speed walk to the side with a slow yaw.
  const s = f < 30 ? 0.004 * f : 0.12 + 0.012 * (f - 30);
  return tpPose([s, 0.1 * s, 0.3 * s], 0.002 * f, 0);
}
const sequence: Uint8Array[] = [];
for (let f = 0; f < FRAMES; f++) sequence.push(tpRender(poseAt(f)));

describe("PnP and BA far from the map origin (Jacobian fix)", () => {
  it("keeps the PnP error flat and the BA converging as the camera walks many map units away from the initialization view", () => {
    const engine = new VisionEngine(W, H, resolveConfig(), createRng(5));
    const runs: string[] = [];
    let unconverged = 0, runCount = 0, itSum = 0;
    const outs: VisionOutput[] = [];
    for (let f = 0; f < FRAMES; f++) {
      const o = engine.process({ frameId: f, timestamp: f * 33.3, width: W, height: H, gray: sequence[f], intrinsics: TP_K, gravity: tpGravity });
      outs.push(o);
      const ba = o.bundleAdjustment;
      if (ba.ranThisFrame) {
        runCount++;
        itSum += ba.iterations;
        if (!ba.converged) unconverged++;
        runs.push(`f${f}:${ba.errorBeforePx.toFixed(2)}→${ba.errorAfterPx.toFixed(2)} it${ba.iterations}${ba.converged ? "" : "!"}${ba.rejected ? " " + ba.rejectReason : ""}`);
      }
    }
    expect(outs.some((o) => o.state === TrackingState.PLANE_FOUND)).toBe(true);
    const err = (from: number, to: number) => {
      let s = 0, n = 0;
      for (let f = from; f < to; f++) {
        const mp = outs[f].mapPose;
        if (mp && mp.framesSinceTracked === 0) {
          s += mp.meanReprojectionErrorPx;
          n++;
        }
      }
      return { mean: n ? s / n : NaN, n };
    };
    const near = err(40, 70);
    const far = err(100, FRAMES);
    const lastC = outs[FRAMES - 1].mapPose!.cameraCenter;
    const dist = Math.hypot(lastC[0], lastC[1], lastC[2]);
    const lost = outs.filter((o) => o.mapPose && o.mapPose.framesSinceTracked > 0).length;
    console.log(`[far] BA runs ${runCount}, unconverged ${unconverged}, mean iterations ${(itSum / Math.max(1, runCount)).toFixed(1)}: ${runs.join("  ")}`);
    console.log(`[far] PnP error near ${near.mean.toFixed(2)} px (${near.n}) → far ${far.mean.toFixed(2)} px (${far.n}); final camera ${dist.toFixed(1)} map units from the origin; lost frames ${lost}`);
    expect(dist).toBeGreaterThan(5);
    expect(near.n).toBeGreaterThan(20);
    expect(far.n).toBeGreaterThan(20);
    expect(far.mean).toBeLessThan(1.0);
    expect(far.mean).toBeLessThan(near.mean + 0.5);
    expect(lost).toBeLessThanOrEqual(3);
    // Before the fix 8 of 11 runs hit the 15-iteration cap once the camera was
    // more than a few map units out (the on-device `it 15!`); now every run
    // converges in well under 10 iterations.
    expect(runCount).toBeGreaterThanOrEqual(6);
    expect(unconverged).toBe(0);
    expect(itSum / runCount).toBeLessThan(10);
  });
});
