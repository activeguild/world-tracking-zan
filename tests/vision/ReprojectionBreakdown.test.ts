import { describe, expect, it } from "vitest";
import { resolveConfig } from "../../src/ar/ARConfig";
import { TrackingState } from "../../src/ar/ARState";
import { approximateIntrinsics } from "../../src/camera/CameraIntrinsics";
import { type Mat3, mat3Invert, mat3Multiply, mat3TransformPoint } from "../../src/math/Matrix";
import { createRng } from "../../src/vision/OutlierRejection";
import { emptyReprojectionBreakdown, type ReprojectionBreakdown, type VisionInput, type VisionOutput } from "../../src/vision/types";
import { VisionEngine } from "../../src/vision/VisionEngine";
import { makeTexture, sampleBilinear } from "../helpers/synthetic";

/**
 * v16 diagnostics: the PnP inlier error split by image region (center disc
 * vs edge ring) and landmark kind. On device (Android) placed objects
 * shifted while the phone was tilted and came back when tilted back, with a
 * mean error of 2.4–3.0 px: a camera-model mismatch. The breakdown must
 * separate a pinhole-consistent scene (center ≈ edge) from a lens with
 * radial distortion (edge ≫ center).
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
const HALF_DIAG = Math.hypot(W, H) / 2;

function pixelHomography(r: Mat3, t: ArrayLike<number>, nn: ArrayLike<number>, d: number): Mat3 {
  const h = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) h[i * 3 + j] = r[i * 3 + j] + (t[i] * nn[j]) / d;
  return mat3Multiply(mat3Multiply(Kmat, h), KInv);
}

/**
 * Render frame f of a slow slide over the floor, then apply a radial lens
 * distortion to the *observed* image: an observed pixel p maps to the ideal
 * pinhole pixel c + (p − c)·(1 + k1·r²), r = |p − c| / half-diagonal. k1 = 0
 * is the pinhole the engine assumes.
 */
function render(f: number, k1: number): Uint8Array {
  const t = [-0.004 * f, -0.001 * f, 0];
  const Hf = f === 0 ? I : pixelHomography(I, t, n, 1);
  const inv = mat3Invert(Hf)!;
  const out = new Uint8Array(W * H);
  const pt = new Float64Array(2);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const dx = x - K.cx;
      const dy = y - K.cy;
      const r2 = (dx * dx + dy * dy) / (HALF_DIAG * HALF_DIAG);
      const s = 1 + k1 * r2;
      const ux = K.cx + dx * s;
      const uy = K.cy + dy * s;
      if (!mat3TransformPoint(inv, ux, uy, pt)) {
        out[y * W + x] = 128;
        continue;
      }
      out[y * W + x] = Math.round(sampleBilinear(base, W, H, pt[0], pt[1], 128));
    }
  }
  return out;
}

function run(k1: number): { breakdown: ReprojectionBreakdown; meanErrPx: number; frames: number } {
  const engine = new VisionEngine(W, H, resolveConfig(), createRng(9));
  const outs: VisionOutput[] = [];
  for (let f = 0; f < FRAMES; f++) {
    const input: VisionInput = { frameId: f, timestamp: f * 33.3, width: W, height: H, gray: render(f, k1), intrinsics: K, gravity };
    outs.push(engine.process(input));
  }
  const foundIdx = outs.findIndex((o) => o.state === TrackingState.PLANE_FOUND);
  expect(foundIdx, `k1 ${k1}: plane found`).toBeGreaterThan(0);
  // Average the breakdown over the tracked frames after the plane was found.
  const acc: ReprojectionBreakdown = { ...emptyReprojectionBreakdown(), centerRadius: 0 };
  let frames = 0;
  let meanErr = 0;
  for (let f = foundIdx; f < FRAMES; f++) {
    const mp = outs[f].mapPose;
    if (!mp || mp.framesSinceTracked !== 0) continue;
    const b = mp.reprojection;
    acc.centerCount += b.centerCount;
    acc.centerErrorPx += b.centerErrorPx * b.centerCount;
    acc.edgeCount += b.edgeCount;
    acc.edgeErrorPx += b.edgeErrorPx * b.edgeCount;
    acc.planeCount += b.planeCount;
    acc.planeErrorPx += b.planeErrorPx * b.planeCount;
    acc.otherCount += b.otherCount;
    acc.otherErrorPx += b.otherErrorPx * b.otherCount;
    acc.lowParallaxCount += b.lowParallaxCount;
    acc.lowParallaxErrorPx += b.lowParallaxErrorPx * b.lowParallaxCount;
    acc.midParallaxCount += b.midParallaxCount;
    acc.midParallaxErrorPx += b.midParallaxErrorPx * b.midParallaxCount;
    acc.highParallaxCount += b.highParallaxCount;
    acc.highParallaxErrorPx += b.highParallaxErrorPx * b.highParallaxCount;
    acc.youngTrackCount += b.youngTrackCount;
    acc.youngTrackErrorPx += b.youngTrackErrorPx * b.youngTrackCount;
    acc.midTrackCount += b.midTrackCount;
    acc.midTrackErrorPx += b.midTrackErrorPx * b.midTrackCount;
    acc.oldTrackCount += b.oldTrackCount;
    acc.oldTrackErrorPx += b.oldTrackErrorPx * b.oldTrackCount;
    acc.youngLandmarkCount += b.youngLandmarkCount;
    acc.youngLandmarkErrorPx += b.youngLandmarkErrorPx * b.youngLandmarkCount;
    acc.midLandmarkCount += b.midLandmarkCount;
    acc.midLandmarkErrorPx += b.midLandmarkErrorPx * b.midLandmarkCount;
    acc.oldLandmarkCount += b.oldLandmarkCount;
    acc.oldLandmarkErrorPx += b.oldLandmarkErrorPx * b.oldLandmarkCount;
    acc.centerRadius = b.centerRadius;
    meanErr += mp.meanReprojectionErrorPx;
    frames++;
  }
  acc.centerErrorPx /= Math.max(1, acc.centerCount);
  acc.edgeErrorPx /= Math.max(1, acc.edgeCount);
  acc.planeErrorPx /= Math.max(1, acc.planeCount);
  acc.otherErrorPx /= Math.max(1, acc.otherCount);
  acc.lowParallaxErrorPx /= Math.max(1, acc.lowParallaxCount);
  acc.midParallaxErrorPx /= Math.max(1, acc.midParallaxCount);
  acc.highParallaxErrorPx /= Math.max(1, acc.highParallaxCount);
  acc.youngTrackErrorPx /= Math.max(1, acc.youngTrackCount);
  acc.midTrackErrorPx /= Math.max(1, acc.midTrackCount);
  acc.oldTrackErrorPx /= Math.max(1, acc.oldTrackCount);
  acc.youngLandmarkErrorPx /= Math.max(1, acc.youngLandmarkCount);
  acc.midLandmarkErrorPx /= Math.max(1, acc.midLandmarkCount);
  acc.oldLandmarkErrorPx /= Math.max(1, acc.oldLandmarkCount);
  return { breakdown: acc, meanErrPx: meanErr / Math.max(1, frames), frames };
}

describe("PnP reprojection breakdown (v16 diagnostics)", () => {
  it("a pinhole-consistent scene has similar center and edge errors; a radially distorted lens shows edge ≫ center", () => {
    const pinhole = run(0);
    const distorted = run(-0.06); // ≈ 24 px of barrel at the corners, ≈ 6 px at the center/edge boundary
    const fmt = (r: ReturnType<typeof run>) =>
      `mean ${r.meanErrPx.toFixed(2)} px | center ${r.breakdown.centerErrorPx.toFixed(2)} (${r.breakdown.centerCount}) edge ${r.breakdown.edgeErrorPx.toFixed(2)} (${r.breakdown.edgeCount}) | plane ${r.breakdown.planeErrorPx.toFixed(2)} (${r.breakdown.planeCount}) other ${r.breakdown.otherErrorPx.toFixed(2)} (${r.breakdown.otherCount}) over ${r.frames} frames`;
    console.log(`[reproj] pinhole:   ${fmt(pinhole)}\n[reproj] distorted: ${fmt(distorted)}`);
    // Both regions populated, the split radius reported.
    expect(pinhole.breakdown.centerCount).toBeGreaterThan(0);
    expect(pinhole.breakdown.edgeCount).toBeGreaterThan(0);
    expect(pinhole.breakdown.centerRadius).toBeCloseTo(0.5, 5);
    // Pinhole: the edge ring is not systematically worse than the center.
    const pinholeRatio = pinhole.breakdown.edgeErrorPx / Math.max(1e-6, pinhole.breakdown.centerErrorPx);
    expect(pinholeRatio).toBeLessThan(1.5);
    // Distorted lens: the edge ring carries the model error.
    const distortedRatio = distorted.breakdown.edgeErrorPx / Math.max(1e-6, distorted.breakdown.centerErrorPx);
    expect(distortedRatio).toBeGreaterThan(1.5);
    expect(distorted.breakdown.edgeErrorPx).toBeGreaterThan(pinhole.breakdown.edgeErrorPx);
    // The per-kind and per-parallax counts add up to the per-region counts (same inlier set).
    expect(pinhole.breakdown.planeCount + pinhole.breakdown.otherCount).toBe(pinhole.breakdown.centerCount + pinhole.breakdown.edgeCount);
    const par = pinhole.breakdown.lowParallaxCount + pinhole.breakdown.midParallaxCount + pinhole.breakdown.highParallaxCount;
    expect(par).toBe(pinhole.breakdown.centerCount + pinhole.breakdown.edgeCount);
    console.log(
      `[reproj] parallax bins (pinhole): <2° ${pinhole.breakdown.lowParallaxErrorPx.toFixed(2)} (${pinhole.breakdown.lowParallaxCount})  2–5° ${pinhole.breakdown.midParallaxErrorPx.toFixed(2)} (${pinhole.breakdown.midParallaxCount})  >5° ${pinhole.breakdown.highParallaxErrorPx.toFixed(2)} (${pinhole.breakdown.highParallaxCount})`,
    );
  });

  it("splits the same inliers by track age and by landmark age (LK drift vs map inconsistency)", () => {
    // A longer slide so that tracks and landmarks reach the old bin (≥ 150 frames).
    const engine = new VisionEngine(W, H, resolveConfig(), createRng(9));
    const last: ReprojectionBreakdown[] = [];
    for (let f = 0; f < 200; f++) {
      const input: VisionInput = { frameId: f, timestamp: f * 33.3, width: W, height: H, gray: render(Math.min(f, FRAMES - 1), 0), intrinsics: K, gravity };
      const o = engine.process(input);
      if (o.mapPose && o.mapPose.framesSinceTracked === 0 && f >= 190) last.push(o.mapPose.reprojection);
    }
    expect(last.length).toBeGreaterThan(0);
    const b = last[last.length - 1];
    const total = b.centerCount + b.edgeCount;
    // Both splits cover exactly the PnP inlier set.
    expect(b.youngTrackCount + b.midTrackCount + b.oldTrackCount).toBe(total);
    expect(b.youngLandmarkCount + b.midLandmarkCount + b.oldLandmarkCount).toBe(total);
    expect(b.ageYoungFrames).toBe(30);
    expect(b.ageOldFrames).toBe(150);
    // After 200 frames of continuous tracking the oldest tracks and landmarks populate the old bin.
    expect(b.oldTrackCount).toBeGreaterThan(0);
    expect(b.oldLandmarkCount).toBeGreaterThan(0);
    // A landmark can never be older than the frame count, and a track observing it cannot be
    // older than the landmark's own track unless re-linked: on a pinhole-consistent synthetic
    // floor neither age carries extra error.
    const fmt = (y: number, yc: number, m: number, mc: number, o: number, oc: number) => `<30 ${y.toFixed(2)} (${yc})  30–150 ${m.toFixed(2)} (${mc})  >150 ${o.toFixed(2)} (${oc})`;
    console.log(
      `[reproj] age bins: track ${fmt(b.youngTrackErrorPx, b.youngTrackCount, b.midTrackErrorPx, b.midTrackCount, b.oldTrackErrorPx, b.oldTrackCount)}\n[reproj] age bins: lm    ${fmt(b.youngLandmarkErrorPx, b.youngLandmarkCount, b.midLandmarkErrorPx, b.midLandmarkCount, b.oldLandmarkErrorPx, b.oldLandmarkCount)}`,
    );
    if (b.youngTrackCount && b.oldTrackCount) expect(b.oldTrackErrorPx / Math.max(1e-6, b.youngTrackErrorPx)).toBeLessThan(1.5);
    if (b.youngLandmarkCount && b.oldLandmarkCount) expect(b.oldLandmarkErrorPx / Math.max(1e-6, b.youngLandmarkErrorPx)).toBeLessThan(1.5);
  });

  it("v19: splits the same inliers by how the track got its landmark (native / relink / reloc)", () => {
    const engine = new VisionEngine(W, H, resolveConfig(), createRng(9));
    let relinkSeen = 0;
    let last: ReprojectionBreakdown | null = null;
    let cut = 0;
    for (let f = 0; f < 200; f++) {
      const input: VisionInput = { frameId: f, timestamp: f * 33.3, width: W, height: H, gray: render(Math.min(f, FRAMES - 1), 0), intrinsics: K, gravity };
      const o = engine.process(input);
      if (f === 100) {
        // Tracks never die on the synthetic floor, so cut every other landmark link: the
        // re-association of the next frames re-links the same corners (within 4 px).
        for (const t of engine.tracksForTests()) {
          if (t.landmarkId < 0 || cut % 2 === 1) {
            if (t.landmarkId >= 0) cut++;
            continue;
          }
          const lm = engine.landmarkMap.get(t.landmarkId)!;
          lm.trackId = -1;
          t.landmarkId = -1;
          // Like a re-detected corner: no anchor, so the track cannot triangulate a duplicate.
          t.anchorFrame = -1;
          t.anchorPose = null;
          cut++;
        }
      }
      if (!o.mapPose || o.mapPose.framesSinceTracked !== 0) continue;
      const b = o.mapPose.reprojection;
      // The split covers exactly the PnP inlier set, every frame.
      expect(b.nativeCount + b.relinkCount + b.relocCount).toBe(b.centerCount + b.edgeCount);
      if (b.relinkCount) relinkSeen++;
      if (f < 100) expect(b.relinkCount).toBe(0);
      last = b;
    }
    expect(last).not.toBeNull();
    expect(cut).toBeGreaterThan(40);
    // Every landmark-linked track carries its link source.
    for (const t of engine.tracksForTests()) if (t.landmarkId >= 0) expect(t.linkSource === "native" || t.linkSource === "relink" || t.linkSource === "reloc").toBe(true);
    // The cut links come back as relinks; on a consistent synthetic floor the re-linked
    // group (same corners) is not worse than the native one.
    expect(relinkSeen).toBeGreaterThan(50);
    expect(last!.relinkCount).toBeGreaterThan(10);
    const b = last!;
    console.log(`[reproj] link source: native ${b.nativeErrorPx.toFixed(2)} (${b.nativeCount})  relink ${b.relinkErrorPx.toFixed(2)} (${b.relinkCount})  reloc ${b.relocErrorPx.toFixed(2)} (${b.relocCount}); frames with relinks ${relinkSeen}`);
    if (b.nativeCount && b.relinkCount) expect(b.relinkErrorPx / Math.max(1e-6, b.nativeErrorPx)).toBeLessThan(1.5);
    expect(b.relocCount).toBe(0);
  });
});
