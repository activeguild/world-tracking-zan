import { describe, expect, it } from "vitest";
import { resolveConfig } from "../../src/ar/ARConfig";
import { TrackingState } from "../../src/ar/ARState";
import { WorldAnchor } from "../../src/ar/WorldAnchor";
import { approximateIntrinsics } from "../../src/camera/CameraIntrinsics";
import { worldToMap } from "../../src/math/CoordinateSystem";
import { type Mat3, mat3Invert, mat3Multiply, mat3TransformPoint } from "../../src/math/Matrix";
import { downsampleToCoarse } from "../../src/vision/Keyframe";
import { createRng } from "../../src/vision/OutlierRejection";
import { coarseShift } from "../../src/vision/Relocalizer";
import type { VisionInput, VisionOutput } from "../../src/vision/types";
import { VisionEngine } from "../../src/vision/VisionEngine";
import { makeTexture, translateImage, warpImage } from "../helpers/synthetic";

/**
 * Phase 5 completion criterion: recover from a short tracking loss into the
 * same map, so that a placed object keeps its place.
 */
const W = 640;
const H = 480;
const K = approximateIntrinsics(W, H);
const Kmat = new Float64Array([K.fx, 0, K.cx, 0, K.fy, K.cy, 0, 0, 1]);
const KInv = mat3Invert(Kmat)!;
const I: Mat3 = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);

function pixelHomography(r: Mat3, t: ArrayLike<number>, n: ArrayLike<number>, d: number): Mat3 {
  const h = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) h[i * 3 + j] = r[i * 3 + j] + (t[i] * n[j]) / d;
  return mat3Multiply(mat3Multiply(Kmat, h), KInv);
}

function input(frameId: number, gray: Uint8Array, gravity: number[] | null): VisionInput {
  return { frameId, timestamp: frameId * 33.3, width: W, height: H, gray, intrinsics: K, gravity };
}

function projectWorld(anchor: WorldAnchor, mp: NonNullable<VisionOutput["mapPose"]>, pw: ArrayLike<number>): [number, number] {
  const pm = worldToMap(anchor.frame!, pw);
  const r = mp.rotation, t = mp.translation;
  const x = r[0] * pm[0] + r[1] * pm[1] + r[2] * pm[2] + t[0];
  const y = r[3] * pm[0] + r[4] * pm[1] + r[5] * pm[2] + t[1];
  const z = r[6] * pm[0] + r[7] * pm[1] + r[8] * pm[2] + t[2];
  return [(x / z) * K.fx + K.cx, (y / z) * K.fy + K.cy];
}

const base = makeTexture(W, H, createRng(3030), [12, 30, 70, 160]);
const n = [0, Math.SQRT1_2, Math.SQRT1_2];
const gravity = [0, Math.SQRT1_2, Math.SQRT1_2];

/** Desk frame f: camera center C = (0.004 f, 0.001 f, 0) over the plane. */
function deskFrame(f: number): { img: Uint8Array; H: Mat3 } {
  const t = [-0.004 * f, -0.001 * f, 0];
  const Hf = f === 0 ? I : pixelHomography(I, t, n, 1);
  return { img: f === 0 ? base : warpImage(base, W, H, Hf, 128), H: Hf };
}

describe("coarseShift", () => {
  it("recovers an integer shift between two low-resolution images", () => {
    const tex = makeTexture(160, 120, createRng(5), [8, 20, 40]);
    const shifted = translateImage(tex, 160, 120, 14, -9, 128);
    const a = downsampleToCoarse({ width: 160, height: 120, data: tex });
    const b = downsampleToCoarse({ width: 160, height: 120, data: shifted });
    const s = coarseShift(a, b, 12);
    expect(s.dx).toBe(7);
    expect([-4, -5]).toContain(s.dy); // −9 px → −4.5 after the 2× downsample
    expect(s.score).toBeGreaterThan(0.6);
  });
});

describe("Relocalization (Phase 5)", () => {
  it("recovers into the same map after a blank gap; a placed object keeps its pixel", () => {
    const engine = new VisionEngine(W, H, resolveConfig(), createRng(11));
    const anchor = new WorldAnchor({ assumedPlaneDistanceMeters: 0.5 });
    const outs: VisionOutput[] = [];
    const Hs: (Mat3 | null)[] = [];
    const blank = new Uint8Array(W * H).fill(96);
    let frameId = 0;
    // 1) normal motion, map + plane
    for (let f = 0; f < 30; f++, frameId++) {
      const { img, H: Hf } = deskFrame(f);
      Hs.push(Hf);
      outs.push(engine.process(input(frameId, img, gravity)));
    }
    const foundIdx = outs.findIndex((o) => o.state === TrackingState.PLANE_FOUND);
    expect(foundIdx).toBeGreaterThan(0);
    const A = 28;
    expect(outs[A].mapPose).not.toBeNull();
    const mapFrameId = outs[A].mapPose!.mapFrameId;
    expect(anchor.create(outs[A].plane!, outs[A].mapPose!)).toBe(true);
    const taps: [number, number][] = [[320, 260], [230, 300], [420, 210]];
    const placed = taps.map(([u, v]) => anchor.hitTest(u, v, K, outs[A].mapPose!)!);
    const keyframesBefore = outs[A].relocalization.keyframes;
    expect(keyframesBefore).toBeGreaterThanOrEqual(1);

    // 2) occlusion / blur: blank frames → tracking lost
    const gap = 8;
    for (let g = 0; g < gap; g++, frameId++) {
      Hs.push(null);
      outs.push(engine.process(input(frameId, blank, gravity)));
    }
    const lastBlank = outs[outs.length - 1];
    expect([TrackingState.TRACKING_LOST, TrackingState.RELOCALIZING]).toContain(lastBlank.state);
    expect(lastBlank.mapPose).not.toBeNull(); // the map is kept, not reset
    expect(lastBlank.mapPose!.framesSinceTracked).toBeGreaterThan(0);

    // 3) the camera comes back a little further along its path
    const resumeAt = 33; // frames 33.. (a jump of 5 frames ≈ 13 px vs the last tracked view)
    let relocFrame = -1;
    for (let f = resumeAt; f < resumeAt + 25; f++, frameId++) {
      const { img, H: Hf } = deskFrame(f);
      Hs.push(Hf);
      const o = engine.process(input(frameId, img, gravity));
      outs.push(o);
      if (relocFrame < 0 && o.relocalization.attempt === "success") relocFrame = frameId;
    }
    expect(relocFrame, "relocalization succeeded").toBeGreaterThan(0);
    const last = outs[outs.length - 1];
    expect(last.mapPose).not.toBeNull();
    expect(last.mapPose!.mapFrameId).toBe(mapFrameId); // same map → same world
    expect(last.mapPose!.framesSinceTracked).toBe(0);
    expect(last.relocalization.successCount).toBeGreaterThanOrEqual(1);
    expect([TrackingState.PLANE_FOUND, TrackingState.PLANE_DETECTING]).toContain(last.state);

    // 4) placed points reproject onto the same desk pixels after recovery
    const HAinv = mat3Invert(Hs[A]!)!;
    const pt = new Float64Array(2);
    const errs: number[] = [];
    for (let B = relocFrame + 1; B < outs.length; B++) {
      const Hb = Hs[B];
      if (!Hb) continue;
      const mp = outs[B].mapPose!;
      if (mp.framesSinceTracked !== 0) continue;
      const HBA = mat3Multiply(Hb, HAinv);
      for (let k = 0; k < taps.length; k++) {
        mat3TransformPoint(HBA, taps[k][0], taps[k][1], pt);
        const [pu, pv] = projectWorld(anchor, mp, placed[k].position);
        errs.push(Math.hypot(pu - pt[0], pv - pt[1]));
      }
    }
    expect(errs.length).toBeGreaterThan(30);
    errs.sort((a, b) => a - b);
    console.log(`[reloc] drift after recovery: median ${errs[errs.length >> 1].toFixed(2)} px, max ${errs[errs.length - 1].toFixed(2)} px`);
    expect(errs[errs.length >> 1]).toBeLessThan(2);
    expect(errs[errs.length - 1]).toBeLessThan(5);
  });

  it("recovers after looking at a different scene and coming back", () => {
    const engine = new VisionEngine(W, H, resolveConfig(), createRng(12));
    const other = makeTexture(W, H, createRng(4040), [10, 25, 60, 150]);
    const outs: VisionOutput[] = [];
    let frameId = 0;
    for (let f = 0; f < 30; f++, frameId++) outs.push(engine.process(input(frameId, deskFrame(f).img, gravity)));
    const mapFrameId = outs[outs.length - 1].mapPose!.mapFrameId;
    // Look at a different, static scene for a while (features track fine there,
    // but none of them belong to the map → RELOCALIZING).
    for (let g = 0; g < 20; g++, frameId++) outs.push(engine.process(input(frameId, other, gravity)));
    const away = outs[outs.length - 1];
    expect(away.mapPose!.mapFrameId).toBe(mapFrameId);
    expect(away.mapPose!.framesSinceTracked).toBeGreaterThan(10);
    expect(away.state).toBe(TrackingState.RELOCALIZING);
    // Come back to roughly where we were.
    let recovered = false;
    for (let f = 27; f < 45; f++, frameId++) {
      const o = engine.process(input(frameId, deskFrame(f).img, gravity));
      outs.push(o);
      if (o.mapPose && o.mapPose.framesSinceTracked === 0 && o.mapPose.mapFrameId === mapFrameId) recovered = true;
    }
    expect(recovered).toBe(true);
    expect(outs[outs.length - 1].relocalization.successCount).toBeGreaterThanOrEqual(1);
  });

  it("creates keyframes as the camera moves and keeps the count bounded", () => {
    const cfg = resolveConfig();
    const engine = new VisionEngine(W, H, cfg, createRng(13));
    let last: VisionOutput | null = null;
    for (let f = 0; f < 90; f++) last = engine.process(input(f, deskFrame(f).img, gravity));
    expect(last!.relocalization.keyframes).toBeGreaterThanOrEqual(3);
    expect(last!.relocalization.keyframes).toBeLessThanOrEqual(cfg.relocalization.maxKeyframes);
  });

  it("resets the map when relocalization never succeeds", () => {
    const cfg = resolveConfig({ landmarks: { lostResetFrames: 20 } });
    const engine = new VisionEngine(W, H, cfg, createRng(14));
    const other = makeTexture(W, H, createRng(5050), [10, 25, 60, 150]);
    let frameId = 0;
    let last: VisionOutput | null = null;
    for (let f = 0; f < 30; f++, frameId++) last = engine.process(input(frameId, deskFrame(f).img, gravity));
    const mapFrameId = last!.mapPose!.mapFrameId;
    for (let g = 0; g < 30; g++, frameId++) last = engine.process(input(frameId, other, gravity));
    // Either the map was reset (no map) or re-initialized on the new scene.
    expect(last!.mapPose === null || last!.mapPose.mapFrameId !== mapFrameId).toBe(true);
    expect(last!.relocalization.keyframes).toBeLessThanOrEqual(1);
  });
});
