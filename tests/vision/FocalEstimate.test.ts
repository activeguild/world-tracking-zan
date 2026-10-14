import { describe, expect, it } from "vitest";
import { resolveConfig } from "../../src/ar/ARConfig";
import { approximateIntrinsics } from "../../src/camera/CameraIntrinsics";
import { createRng } from "../../src/vision/OutlierRejection";
import type { VisionInput } from "../../src/vision/types";
import { VisionEngine } from "../../src/vision/VisionEngine";
import { TP_FOV, TP_H, TP_W, tpGravity, tpPose, tpRender } from "../helpers/twoPlaneScene";

/**
 * v16: what a wrong focal length does to the PnP error on a scene with depth
 * variation (floor + back wall), rendered with a 66° camera and tracked
 * with 60 / 66 / 72° assumed.
 *
 * Findings this test pins down:
 *  - During the sideways slide the map was built from, a 10% focal error
 *    changes the PnP error by well under 0.2 px: the map absorbs the error.
 *    The 0.98 → 2.40 px growth seen on Android during its scan is therefore
 *    not a focal-length problem.
 *  - The error of a wrong focal length appears only under large tilts
 *    (±25° here, ≈ +0.5 px) — the tilt test is what exposes a camera-model
 *    error, and even then the effect is modest.
 *  - A per-frame "PnP + f" re-solve on the existing map cannot recover the
 *    true focal length (the ratio f_est / f stays ≈ 1.00 with a 10% error,
 *    measured while developing this test): the map, built with the assumed f, is already a
 *    consistent reconstruction for it. Such a HUD row would be misleading,
 *    so it was not added.
 */
const W = TP_W;
const H = TP_H;
const TRUE_FOV = TP_FOV;
const SLIDE = 90;
const FRAMES = 210;
/**
 * Frames 0–89: a sideways slide with a slow yaw (landmarks on both surfaces
 * get triangulated). Frames 90–209: the on-device tilt test — pitch swings of
 * ±25° about the camera x axis from a fixed position, back to level at the
 * end.
 */
function poseAt(f: number) {
  const s = Math.min(f, SLIDE);
  const yaw = 0.0015 * s;
  const pitch = f > SLIDE ? ((25 * Math.PI) / 180) * Math.sin(((f - SLIDE) / (FRAMES - SLIDE)) * 2 * Math.PI) : 0;
  return tpPose([0.005 * s, 0.001 * s, 0.002 * s], yaw, pitch);
}
const sequence: Uint8Array[] = [];
for (let f = 0; f < FRAMES; f++) sequence.push(tpRender(poseAt(f)));

interface Run {
  fov: number;
  tracked: number;
  /** Mean PnP error (px) over the slide / over the tilt phase. */
  slideErrPx: number;
  tiltErrPx: number;
}

function run(assumedFov: number): Run {
  const k = approximateIntrinsics(W, H, assumedFov);
  // Bundle adjustment off: this test is about the raw map's response to the camera model.
  const engine = new VisionEngine(W, H, resolveConfig({ bundleAdjustment: { enabled: false } }), createRng(11));
  let tracked = 0;
  let slideErr = 0, slideN = 0, tiltErr = 0, tiltN = 0;
  for (let f = 0; f < FRAMES; f++) {
    const input: VisionInput = { frameId: f, timestamp: f * 33.3, width: W, height: H, gray: sequence[f], intrinsics: k, gravity: tpGravity };
    const mp = engine.process(input).mapPose;
    if (mp?.framesSinceTracked !== 0) continue;
    tracked++;
    if (f < SLIDE) {
      slideErr += mp.meanReprojectionErrorPx;
      slideN++;
    } else {
      tiltErr += mp.meanReprojectionErrorPx;
      tiltN++;
    }
  }
  expect(slideN, `fov ${assumedFov}: tracked during the slide`).toBeGreaterThan(30);
  expect(tiltN, `fov ${assumedFov}: tracked during the tilt`).toBeGreaterThan(60);
  return { fov: assumedFov, tracked, slideErrPx: slideErr / slideN, tiltErrPx: tiltErr / tiltN };
}

describe("focal length error signature (v16)", () => {
  it("a 10% focal error leaves the slide-phase PnP error unchanged and shows only under ±25° tilts", () => {
    const rows = [60, 66, 72].map(run);
    console.log(
      "[focal] " +
        rows.map((r) => `assumed ${r.fov}°: PnP err slide ${r.slideErrPx.toFixed(2)} px  tilt ${r.tiltErrPx.toFixed(2)} px  (tracked ${r.tracked}/${FRAMES})`).join("\n[focal] "),
    );
    const right = rows.find((r) => r.fov === 66)!;
    for (const r of rows) {
      // The map absorbs the focal error for the views it was built from.
      expect(Math.abs(r.slideErrPx - right.slideErrPx), `fov ${r.fov} slide`).toBeLessThan(0.2);
      expect(r.slideErrPx).toBeLessThan(0.6);
    }
    // Tilting exposes it: both wrong assumptions are measurably worse than the right one.
    for (const r of rows) {
      if (r.fov === TRUE_FOV) continue;
      expect(r.tiltErrPx - right.tiltErrPx, `fov ${r.fov} tilt`).toBeGreaterThan(0.3);
    }
    // Even with the right focal length the tilt phase is costlier than the slide
    // (views the map was not built from), but it stays near 1 px on this synthetic scene.
    expect(right.tiltErrPx).toBeLessThan(1.2);
  });
});
