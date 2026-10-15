import { describe, expect, it } from "vitest";
import { resolveConfig } from "../../src/ar/ARConfig";
import { approximateIntrinsics } from "../../src/camera/CameraIntrinsics";
import { type Mat3, mat3Invert, mat3Multiply, mat3TransformPoint } from "../../src/math/Matrix";
import { createRng } from "../../src/vision/OutlierRejection";
import type { ReanchorStats, VisionInput, VisionOutput } from "../../src/vision/types";
import { VisionEngine } from "../../src/vision/VisionEngine";
import { makeTexture, sampleBilinear } from "../helpers/synthetic";

/**
 * v24 re-anchor probe (Android recording 35: the PnP residual is isotropic,
 * so it is not a landmark depth error — it is either on the live tracks or
 * between the map and the corners as seen now). The probe re-observes a
 * sample of PnP inliers from the oldest keyframe observation of their
 * landmark and compares the residual of the live track with the residual
 * of where the keyframe patch lands.
 *
 * Synthetic floor slide (the v20 scene). Clean run: the tracks sit on the
 * keyframe patches (snap ≈ 0, landed ≈ track). Drift run: at frame 100 every
 * linked track is pushed 1.5 px in a random direction and keeps that offset
 * (LK follows the texture under the shifted window) — the on-device case of
 * tracks that locked onto a neighbouring pattern. The probe must read the
 * drift as a snap of ≈ 1.5 px with the landed residual well below the track
 * residual, and PnP on the landed observations must reach the clean error.
 */
const W = 640;
const H = 480;
const K = approximateIntrinsics(W, H, 66);
const Kmat = new Float64Array([K.fx, 0, K.cx, 0, K.fy, K.cy, 0, 0, 1]);
const KInv = mat3Invert(Kmat)!;
const I: Mat3 = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);

const base = makeTexture(W, H, createRng(4141), [12, 30, 70, 160]);
const n = [0, Math.SQRT1_2, Math.SQRT1_2];
const gravity = [0, Math.SQRT1_2, Math.SQRT1_2];
const FRAMES = 60;

function pixelHomography(r: Mat3, t: ArrayLike<number>, nn: ArrayLike<number>, d: number): Mat3 {
  const h = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) h[i * 3 + j] = r[i * 3 + j] + (t[i] * nn[j]) / d;
  return mat3Multiply(mat3Multiply(Kmat, h), KInv);
}

function render(f: number): Uint8Array {
  const t = [-0.004 * f, -0.001 * f, 0];
  const Hf = f === 0 ? I : pixelHomography(I, t, n, 1);
  const inv = mat3Invert(Hf)!;
  const out = new Uint8Array(W * H);
  const pt = new Float64Array(2);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      if (!mat3TransformPoint(inv, x, y, pt)) {
        out[y * W + x] = 128;
        continue;
      }
      out[y * W + x] = Math.round(sampleBilinear(base, W, H, pt[0], pt[1], 128));
    }
  }
  return out;
}
const frames: Uint8Array[] = [];
for (let f = 0; f < FRAMES; f++) frames.push(render(f));

function runEngine(drift: boolean) {
  const engine = new VisionEngine(W, H, resolveConfig(), createRng(9));
  const probes: ReanchorStats[] = [];
  const errs: number[] = [];
  let pushed = 0;
  const rng = createRng(77);
  for (let f = 0; f < 200; f++) {
    const input: VisionInput = { frameId: f, timestamp: f * 33.3, width: W, height: H, gray: frames[Math.min(f, FRAMES - 1)], intrinsics: K, gravity };
    const o: VisionOutput = engine.process(input);
    if (drift && f === 100) {
      for (const t of engine.tracksForTests()) {
        if (t.landmarkId < 0) continue;
        const a = rng() * Math.PI * 2;
        const dx = 1.5 * Math.cos(a), dy = 1.5 * Math.sin(a);
        t.x += dx;
        t.y += dy;
        t.prevX += dx;
        t.prevY += dy;
        pushed++;
      }
    }
    if (!o.mapPose || o.mapPose.framesSinceTracked !== 0) continue;
    if (f >= 110) {
      errs.push(o.mapPose.meanReprojectionErrorPx);
      const ra = o.mapPose.reanchor;
      if (ra && ra.frameId === f) probes.push(ra);
    }
  }
  if (drift) expect(pushed).toBeGreaterThan(80);
  expect(probes.length).toBeGreaterThan(10);
  const mean = (xs: number[]) => xs.reduce((a, c) => a + c, 0) / xs.length;
  return {
    probes,
    meanErr: mean(errs),
    sampled: mean(probes.map((p) => p.sampled)),
    landed: mean(probes.map((p) => p.landed)),
    trackRms: mean(probes.map((p) => p.trackRmsPx)),
    landedRms: mean(probes.map((p) => p.landedRmsPx)),
    snap: mean(probes.map((p) => p.meanSnapPx)),
    improved: mean(probes.map((p) => p.improved / Math.max(1, p.landed))),
    trackPnp: mean(probes.map((p) => p.trackPnpErrorPx)),
    landedPnp: mean(probes.map((p) => p.landedPnpErrorPx)),
    kfAge: mean(probes.map((p) => p.keyframeAgeMedianFrames)),
  };
}

describe("re-anchor probe (v24)", () => {
  it("reads drifted tracks as a snap with a much smaller landed residual, and a clean run as landed ≈ track", () => {
    const clean = runEngine(false);
    const drift = runEngine(true);
    const fmt = (r: ReturnType<typeof runEngine>) =>
      `PnP ${r.meanErr.toFixed(2)} px; probe n ${r.landed.toFixed(0)}/${r.sampled.toFixed(0)} trk ${r.trackRms.toFixed(2)} → kf ${r.landedRms.toFixed(2)} px (×${(r.landedRms / r.trackRms).toFixed(2)}) snap ${r.snap.toFixed(2)} px better ${(r.improved * 100).toFixed(0)}% kf age ${r.kfAge.toFixed(0)} f; PnP on sample ${r.trackPnp.toFixed(2)} → ${r.landedPnp.toFixed(2)} px`;
    console.log(`[reanchor] clean: ${fmt(clean)}\n[reanchor] drift: ${fmt(drift)}`);

    // Clean: the probe runs every 5 frames on a sample of the inliers, the
    // keyframe patches land where the tracks are, and the residual is the
    // same either way.
    expect(clean.sampled).toBeGreaterThan(20);
    expect(clean.landed).toBeGreaterThan(15);
    expect(clean.snap).toBeLessThan(0.5);
    expect(clean.landedRms / clean.trackRms).toBeGreaterThan(0.6);
    expect(clean.landedRms / clean.trackRms).toBeLessThan(1.6);
    expect(clean.landedRms).toBeLessThan(0.5);

    // Drift: the live tracks carry the 1.5 px push (PnP error up by an order
    // of magnitude), the keyframe patches land back on the corners the map
    // was built from (snap ≈ 1.5 px), the landed residual is a fraction of
    // the track residual, and PnP on the landed observations recovers the
    // clean error.
    expect(drift.meanErr).toBeGreaterThan(clean.meanErr * 4);
    expect(drift.landed).toBeGreaterThan(15);
    expect(drift.snap).toBeGreaterThan(1.0);
    expect(drift.snap).toBeLessThan(2.0);
    expect(drift.landedRms / drift.trackRms).toBeLessThan(0.5);
    expect(drift.improved).toBeGreaterThan(0.8);
    expect(drift.landedPnp).toBeLessThan(drift.trackPnp * 0.5);
    expect(drift.landedPnp).toBeLessThan(clean.landedPnp * 2 + 0.2);
  });

  it("is off without the keyframe check and with a zero cadence, and never touches the tracks", () => {
    for (const landmarks of [{ relinkVerify: false }, { reanchorProbeEveryFrames: 0 }]) {
      const engine = new VisionEngine(W, H, resolveConfig({ landmarks }), createRng(9));
      let last: VisionOutput | null = null;
      for (let f = 0; f < 120; f++) {
        last = engine.process({ frameId: f, timestamp: f * 33.3, width: W, height: H, gray: frames[Math.min(f, FRAMES - 1)], intrinsics: K, gravity });
      }
      expect(last!.mapPose).not.toBeNull();
      expect(last!.mapPose!.reanchor).toBeNull();
    }
    // Default: the probe reports, the tracks are exactly where LK put them
    // (the probe's keyframe-LK positions are never written back).
    const engine = new VisionEngine(W, H, resolveConfig(), createRng(9));
    let before: { id: number; x: number; y: number }[] = [];
    let after: { id: number; x: number; y: number }[] = [];
    for (let f = 0; f < 121; f++) {
      const o = engine.process({ frameId: f, timestamp: f * 33.3, width: W, height: H, gray: frames[Math.min(f, FRAMES - 1)], intrinsics: K, gravity });
      if (f === 119) before = engine.tracksForTests().map((t) => ({ id: t.id, x: t.x, y: t.y }));
      if (f === 120) {
        expect(o.mapPose!.reanchor?.frameId).toBe(120);
        after = engine.tracksForTests().map((t) => ({ id: t.id, x: t.x, y: t.y }));
      }
    }
    // Frames 119 → 120 is a repeated image (the sequence holds its last
    // frame), so every surviving track stays within LK precision of itself.
    const byId = new Map(before.map((t) => [t.id, t]));
    let moved = 0;
    for (const t of after) {
      const b = byId.get(t.id);
      if (b && Math.hypot(t.x - b.x, t.y - b.y) > 0.3) moved++;
    }
    expect(moved).toBeLessThan(after.length * 0.05);
  });
});
