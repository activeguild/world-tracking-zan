import { describe, expect, it } from "vitest";
import { resolveConfig } from "../../src/ar/ARConfig";
import { approximateIntrinsics } from "../../src/camera/CameraIntrinsics";
import { type Mat3, mat3Invert, mat3Multiply, mat3TransformPoint } from "../../src/math/Matrix";
import { rotationAxisAngle, type RigidTransform } from "../../src/math/Pose";
import { ImagePyramid } from "../../src/vision/ImagePyramid";
import { FeatureDetector } from "../../src/vision/FeatureDetector";
import { createRng } from "../../src/vision/OutlierRejection";
import { Relocalizer } from "../../src/vision/Relocalizer";
import type { ReprojectionBreakdown, Track, VisionInput, VisionOutput } from "../../src/vision/types";
import { VisionEngine } from "../../src/vision/VisionEngine";
import { makeTexture, sampleBilinear, translateImage } from "../helpers/synthetic";

/**
 * v20 (Android recording 28): re-linked tracks carried 0.5–1.5 px more PnP
 * error than native ones and were the majority of the inliers after 20 s.
 * A re-association by distance alone links a re-detected corner 1–2 px
 * beside the landmark's projection and keeps that offset for the life of
 * the track. The keyframe check tracks the landmark's keyframe observation
 * into the current frame and snaps the track onto it.
 */
const W = 640;
const H = 480;
const K = approximateIntrinsics(W, H, 66);
const Kmat = new Float64Array([K.fx, 0, K.cx, 0, K.fy, K.cy, 0, 0, 1]);
const KInv = mat3Invert(Kmat)!;
const I: Mat3 = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
const identity = (): RigidTransform => ({ rotation: Float64Array.from(I), translation: new Float64Array(3) });

function makeTrack(id: number, x: number, y: number, landmarkId: number): Track {
  return {
    id, x, y, prevX: x, prevY: y, age: 1, score: 0, inlier: true, outlierStreak: 0,
    refX: x, refY: y, refFrame: -1, landmarkId, anchorFrame: -1, anchorX: x, anchorY: y, anchorPose: null,
    planePoint: null, planeStreak: 0, planeOutliers: 0, offPlane: false,
  };
}

function pyramidOf(img: Uint8Array, levels: number): ImagePyramid {
  const p = new ImagePyramid(W, H, levels);
  p.build(img);
  return p;
}

describe("Relocalizer.verifyRelinks (v20)", () => {
  const cfg = resolveConfig();
  const levels = cfg.tracker.pyramidLevels;
  const rng = createRng(2020);
  const tex = makeTexture(W, H, rng, [12, 30, 70, 160]);
  // The current frame is the keyframe image shifted by (3, −2) px; a corner
  // observed at (x, y) in the keyframe is at (x + 3, y − 2) now.
  const DX = 3, DY = -2;
  const cur = translateImage(tex, W, H, DX, DY, 128);
  const kfPyr = pyramidOf(tex, levels);
  const curPyr = pyramidOf(cur, levels);
  // Keyframe observations on real corners (FAST), away from the border.
  const corners = new FeatureDetector(W, H, cfg.features)
    .detect(tex, { wanted: 200 })
    .filter((c) => c.x > 60 && c.x < W - 60 && c.y > 60 && c.y < H - 60)
    .sort((a, b) => b.score - a.score);
  expect(corners.length).toBeGreaterThan(4);
  const obs = corners.slice(0, 4).map((c, i) => ({ id: i + 1, x: c.x, y: c.y }));

  function relocalizerWithKeyframe(): Relocalizer {
    const r = new Relocalizer(cfg.relocalization, cfg.tracker);
    r.create(kfPyr, identity(), obs.map((o, i) => makeTrack(i + 1, o.x, o.y, o.id)), 0, 0);
    return r;
  }

  it("confirms a candidate within the radius and snaps it onto the keyframe-consistent position", () => {
    const r = relocalizerWithKeyframe();
    const radius = 2.5;
    // Candidate tracks 1.5 px beside the true position (a re-detected corner
    // that landed on the neighbouring pixel), one exactly on it, one 2.3 px off.
    const cands = [
      { landmarkId: 1, x: obs[0].x + DX + 1.5, y: obs[0].y + DY },
      { landmarkId: 2, x: obs[1].x + DX, y: obs[1].y + DY },
      { landmarkId: 3, x: obs[2].x + DX - 1.6, y: obs[2].y + DY + 1.6 },
    ];
    const v = r.verifyRelinks(cands, curPyr, identity(), radius);
    expect(v.map((x) => x.status)).toEqual(["verified", "verified", "verified"]);
    for (let i = 0; i < cands.length; i++) {
      const trueX = obs[i].x + DX, trueY = obs[i].y + DY;
      const err = Math.hypot(v[i].x - trueX, v[i].y - trueY);
      expect(err, `candidate ${i} snapped to the keyframe position`).toBeLessThan(0.3);
    }
  });

  it("rejects a candidate farther than the radius from where the keyframe patch lands, and one on changed texture", () => {
    const r = relocalizerWithKeyframe();
    // 4 px beside the true corner with a 2.5 px radius: the keyframe patch
    // lands outside the radius of the track → not the same corner.
    const far = r.verifyRelinks([{ landmarkId: 1, x: obs[0].x + DX + 4, y: obs[0].y + DY }], curPyr, identity(), 2.5);
    expect(far[0].status).toBe("rejected");
    // Occlusion: the current frame has noise where landmark 4 is.
    const occluded = Uint8Array.from(cur);
    const nrng = createRng(7);
    const ox = obs[3].x + DX, oy = obs[3].y + DY;
    for (let y = oy - 20; y < oy + 20; y++) for (let x = ox - 20; x < ox + 20; x++) occluded[y * W + x] = Math.floor(nrng() * 255);
    const occ = r.verifyRelinks([{ landmarkId: 4, x: ox, y: oy }], pyramidOf(occluded, levels), identity(), 2.5);
    expect(occ[0].status).toBe("rejected");
    // The same candidate on the intact frame passes (the rejection is the occlusion, not the point).
    expect(r.verifyRelinks([{ landmarkId: 4, x: ox, y: oy }], curPyr, identity(), 2.5)[0].status).toBe("verified");
  });

  it("leaves a landmark no keyframe observes unverified, and picks the keyframe nearest in rotation when several observe it", () => {
    const r = relocalizerWithKeyframe();
    const none = r.verifyRelinks([{ landmarkId: 99, x: 100, y: 100 }], curPyr, identity(), 2.5);
    expect(none[0].status).toBe("unverified");
    expect(none[0].x).toBe(100);
    // A second keyframe, 30° away, whose image is unrelated texture: with the
    // current pose near the first keyframe the check must use the first one.
    const other = makeTexture(W, H, createRng(99), [12, 30, 70, 160]);
    const rot30 = rotationAxisAngle([0, 1, 0], Math.PI / 6);
    r.create(pyramidOf(other, levels), { rotation: rot30, translation: new Float64Array(3) }, [makeTrack(10, obs[0].x, obs[0].y, 1)], 5, 5 * 33);
    expect(r.count).toBe(2);
    const near = r.verifyRelinks([{ landmarkId: 1, x: obs[0].x + DX + 1, y: obs[0].y + DY }], curPyr, identity(), 2.5);
    expect(near[0].status).toBe("verified");
    // With the current pose next to the unrelated keyframe, that one is used and fails.
    const farPose = r.verifyRelinks([{ landmarkId: 1, x: obs[0].x + DX + 1, y: obs[0].y + DY }], curPyr, { rotation: rot30, translation: new Float64Array(3) }, 2.5);
    expect(farPose[0].status).toBe("rejected");
  });
});

/**
 * Engine: a slow slide over a textured floor. At frame 100 every other
 * landmark link is cut and the freed track is pushed 1.5 px sideways — the
 * on-device case of a corner re-detected beside the landmark's projection.
 * Without the check the re-association links it as is and the relinked
 * group carries the offset; with the check it is snapped back onto the
 * keyframe observation and the two groups agree.
 */
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

function runEngine(relinkVerify: boolean) {
  const engine = new VisionEngine(W, H, resolveConfig({ landmarks: { relinkVerify } }), createRng(9));
  let last: ReprojectionBreakdown | null = null;
  let cut = 0;
  let verified = 0, rejected = 0, unverified = 0, candidates = 0, snapSum = 0, snapN = 0;
  const errs: number[] = [];
  for (let f = 0; f < 200; f++) {
    const input: VisionInput = { frameId: f, timestamp: f * 33.3, width: W, height: H, gray: frames[Math.min(f, FRAMES - 1)], intrinsics: K, gravity };
    const o: VisionOutput = engine.process(input);
    if (f === 100) {
      for (const t of engine.tracksForTests()) {
        if (t.landmarkId < 0 || cut % 2 === 1) {
          if (t.landmarkId >= 0) cut++;
          continue;
        }
        const lm = engine.landmarkMap.get(t.landmarkId)!;
        lm.trackId = -1;
        t.landmarkId = -1;
        t.anchorFrame = -1;
        t.anchorPose = null;
        // The re-detected corner sits 1.5 px beside the landmark's projection.
        t.x += 1.5;
        t.prevX += 1.5;
        cut++;
      }
    }
    if (!o.mapPose || o.mapPose.framesSinceTracked !== 0) continue;
    const rl = o.mapPose.relink;
    candidates += rl.candidates;
    verified += rl.verified;
    rejected += rl.rejected;
    unverified += rl.unverified;
    if (rl.verified) {
      snapSum += rl.meanSnapPx * rl.verified;
      snapN += rl.verified;
    }
    if (f >= 110) errs.push(o.mapPose.meanReprojectionErrorPx);
    last = o.mapPose.reprojection;
  }
  expect(cut).toBeGreaterThan(40);
  expect(last).not.toBeNull();
  const b = last!;
  return { b, candidates, verified, rejected, unverified, meanSnap: snapN ? snapSum / snapN : 0, meanErr: errs.reduce((a, c) => a + c, 0) / errs.length };
}

describe("re-association with the keyframe check in the engine (v20)", () => {
  it("snaps displaced re-detections back onto the keyframe observation: relink error ≈ native; without the check the offset stays", () => {
    const on = runEngine(true);
    const off = runEngine(false);
    const fmt = (r: ReturnType<typeof runEngine>) =>
      `native ${r.b.nativeErrorPx.toFixed(2)} (${r.b.nativeCount})  relink ${r.b.relinkErrorPx.toFixed(2)} (${r.b.relinkCount})  raw ${r.b.relinkRawErrorPx.toFixed(2)} (${r.b.relinkRawCount}); candidates ${r.candidates} verified ${r.verified} rejected ${r.rejected} unverified ${r.unverified}, mean snap ${r.meanSnap.toFixed(2)} px, PnP ${r.meanErr.toFixed(2)} px`;
    console.log(`[relink] check on:  ${fmt(on)}\n[relink] check off: ${fmt(off)}`);
    // With the check: the candidates were verified against keyframes, the
    // snap undid the 1.5 px push, and the relinked group is as good as the native one.
    expect(on.verified).toBeGreaterThan(40);
    expect(on.rejected).toBe(0);
    expect(on.meanSnap).toBeGreaterThan(1.0);
    expect(on.meanSnap).toBeLessThan(2.0);
    expect(on.b.relinkCount).toBeGreaterThan(10);
    expect(on.b.relinkErrorPx / Math.max(1e-6, on.b.nativeErrorPx)).toBeLessThan(1.5);
    expect(on.b.relinkErrorPx).toBeLessThan(0.3);
    // Without the check: linked by distance alone, the 1.5 px offset stays
    // on half of the observations and PnP settles on a compromise pose, so
    // the native half degrades too (the on-device "uniform 2–3 px") and the
    // frame's PnP error is an order of magnitude higher.
    expect(off.verified).toBe(0);
    expect(off.unverified).toBeGreaterThan(40);
    expect(off.b.relinkCount).toBeGreaterThan(10);
    expect(off.b.relinkErrorPx).toBeGreaterThan(on.b.relinkErrorPx * 3);
    expect(off.meanErr).toBeGreaterThan(on.meanErr * 5);
  });
});
