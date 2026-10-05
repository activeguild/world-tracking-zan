import { describe, expect, it } from "vitest";
import { resolveConfig } from "../../src/ar/ARConfig";
import { TrackingState } from "../../src/ar/ARState";
import { approximateIntrinsics } from "../../src/camera/CameraIntrinsics";
import { type Mat3, mat3Identity, mat3Invert, mat3Multiply } from "../../src/math/Matrix";
import type { RigidTransform } from "../../src/math/Pose";
import { FeatureDetector } from "../../src/vision/FeatureDetector";
import { ImagePyramid } from "../../src/vision/ImagePyramid";
import { LandmarkMap } from "../../src/vision/LandmarkMap";
import { createRng } from "../../src/vision/OutlierRejection";
import { Relocalizer, spatialCellCount } from "../../src/vision/Relocalizer";
import type { Track, VisionInput, VisionOutput } from "../../src/vision/types";
import { VisionEngine } from "../../src/vision/VisionEngine";
import { makeTexture, warpImage } from "../helpers/synthetic";

/**
 * 修正指示書 v5: a relocalization result is a *candidate* that has to pass a
 * global validation (inliers, error, match, inlier ratio, spatial
 * distribution) and, unless clearly high quality, a confirmation frame
 * before it re-seeds the canonical pose.
 */
const W = 640;
const H = 480;
const K = approximateIntrinsics(W, H);
const identity: RigidTransform = { rotation: mat3Identity(), translation: new Float64Array(3) };

function track(id: number, x: number, y: number, landmarkId: number): Track {
  return {
    id, x, y, prevX: x, prevY: y, age: 5, score: 10, inlier: true, outlierStreak: 0,
    refX: x, refY: y, refFrame: -1, landmarkId, anchorFrame: -1, anchorX: x, anchorY: y, anchorPose: null,
    planePoint: null, planeStreak: 0, planeOutliers: 0, offPlane: false,
  };
}

/** A keyframe of the identity camera looking at corners of `gray`, with landmarks at depth 2 (only those passing `keep`). */
function keyframeScene(gray: Uint8Array, keep: (x: number, y: number) => boolean) {
  const cfg = resolveConfig();
  const pyramid = new ImagePyramid(W, H, cfg.tracker.pyramidLevels);
  pyramid.build(gray);
  const corners = new FeatureDetector(W, H, cfg.features).detect(gray, { wanted: 300 });
  const map = new LandmarkMap();
  const tracks: Track[] = [];
  let id = 1;
  for (const c of corners) {
    if (!keep(c.x, c.y)) continue;
    const Z = 2;
    const lm = map.add([((c.x - K.cx) / K.fx) * Z, ((c.y - K.cy) / K.fy) * Z, Z], id, 0);
    lm.observations = 5;
    tracks.push(track(id++, c.x, c.y, lm.id));
  }
  const reloc = new Relocalizer(cfg.relocalization, cfg.tracker);
  reloc.create(pyramid, identity, tracks, 0, 0);
  return { cfg, pyramid, map, tracks, reloc };
}

describe("spatialCellCount", () => {
  it("counts the occupied cells of a 3×3 grid", () => {
    expect(spatialCellCount([10, 20], [10, 20], 2, W, H)).toBe(1);
    expect(spatialCellCount([10, 630], [10, 470], 2, W, H)).toBe(2);
    const xs = [100, 320, 540, 100, 320, 540, 100, 320, 540];
    const ys = [80, 80, 80, 240, 240, 240, 400, 400, 400];
    expect(spatialCellCount(xs, ys, 9, W, H)).toBe(9);
  });
});

describe("Relocalizer global validation (v5 §5–§7)", () => {
  const gray = makeTexture(W, H, createRng(77), [12, 30, 70, 160]);

  it("accepts a well-distributed, well-supported match and reports its quality", () => {
    const { cfg, pyramid, map, tracks, reloc } = keyframeScene(gray, () => true);
    expect(tracks.length).toBeGreaterThan(100);
    const r = reloc.relocalize(pyramid, map, K);
    expect(r.success).toBe(true);
    expect(r.rejectCode).toBeNull();
    expect(r.inlierCount).toBeGreaterThanOrEqual(cfg.relocalization.minInliers);
    expect(r.meanReprojectionErrorPx).toBeLessThanOrEqual(cfg.relocalization.maxMeanErrorPx);
    expect(r.matchScore).toBeGreaterThan(0.9);
    expect(r.inlierRatio).toBeGreaterThanOrEqual(cfg.relocalization.minInlierRatio);
    expect(r.spatialCells).toBeGreaterThanOrEqual(cfg.relocalization.minSpatialCells);
    expect(r.keyframeId).toBe(1);
  });

  it("rejects a match whose inliers sit in one corner of the image (poor_spatial_distribution), even with enough inliers", () => {
    const { cfg, pyramid, map, tracks, reloc } = keyframeScene(gray, (x, y) => x < 200 && y < 150);
    expect(tracks.length).toBeGreaterThanOrEqual(cfg.relocalization.minInliers);
    const r = reloc.relocalize(pyramid, map, K);
    expect(r.success).toBe(false);
    expect(r.rejectCode).toBe("poor_spatial_distribution");
    // It got as far as PnP with a good solve: only the distribution failed.
    expect(r.inlierCount).toBeGreaterThanOrEqual(cfg.relocalization.minInliers);
    expect(r.meanReprojectionErrorPx).toBeLessThanOrEqual(cfg.relocalization.maxMeanErrorPx);
    expect(r.spatialCells).toBeLessThan(cfg.relocalization.minSpatialCells);
    expect(r.pose).not.toBeNull();
    expect(r.reason).toMatch(/cells/);
  });

  it("rejects a view that does not match any keyframe (low_match_score) and a blank view", () => {
    const { pyramid, map, reloc } = keyframeScene(gray, () => true);
    const other = new ImagePyramid(W, H, resolveConfig().tracker.pyramidLevels);
    other.build(makeTexture(W, H, createRng(78), [10, 25, 60, 150]));
    const r = reloc.relocalize(other, map, K);
    expect(r.success).toBe(false);
    expect(["low_match_score", "lk_failed", "insufficient_inliers", "high_reprojection_error", "low_inlier_ratio"]).toContain(r.rejectCode);
    void pyramid;
  });

  it("reports where each keyframe dropped out and the best trial (v6 §1–§3)", () => {
    // Success: every stage counter accounts for the one keyframe tried.
    const ok = keyframeScene(gray, () => true);
    const r1 = ok.reloc.relocalize(ok.pyramid, ok.map, K);
    const d1 = r1.diagnostics;
    expect(d1.keyframes).toBe(1);
    expect(d1.candidatesTried).toBe(1);
    expect(d1.coarseTested).toBe(1);
    expect(d1.coarsePassed).toBe(1);
    expect(d1.lkTested).toBe(1);
    expect(d1.lkPassed).toBe(1);
    expect(d1.pnpTested).toBe(1);
    expect(d1.pnpPassed).toBe(1);
    expect(d1.validated).toBe(1);
    expect(d1.trials).toHaveLength(1);
    expect(d1.trials[0].stage).toBe("ok");
    expect(d1.best?.keyframeId).toBe(1);
    expect(d1.best?.inlierCount).toBe(r1.inlierCount);
    expect(d1.rejectCode).toBeNull();

    // Spatial rejection: PnP ran and passed the inlier / error checks, the
    // trial stops at "spatial" and that is the attempt's reject code.
    const corner = keyframeScene(gray, (x, y) => x < 200 && y < 150);
    const r2 = corner.reloc.relocalize(corner.pyramid, corner.map, K);
    const d2 = r2.diagnostics;
    expect(d2.pnpTested).toBe(1);
    expect(d2.pnpPassed).toBe(1);
    expect(d2.spatialRejected).toBe(1);
    expect(d2.validated).toBe(0);
    expect(d2.trials[0].stage).toBe("spatial");
    expect(d2.best?.stage).toBe("spatial");
    expect(d2.best?.spatialCells).toBe(r2.spatialCells);
    expect(d2.rejectCode).toBe("poor_spatial_distribution");

    // A different scene: the keyframe drops out early (coarse or LK) and the
    // counters say so; the best trial carries its coarse score.
    const other = new ImagePyramid(W, H, resolveConfig().tracker.pyramidLevels);
    other.build(makeTexture(W, H, createRng(78), [10, 25, 60, 150]));
    const r3 = ok.reloc.relocalize(other, ok.map, K);
    const d3 = r3.diagnostics;
    expect(d3.candidatesTried).toBe(1);
    expect(d3.validated).toBe(0);
    expect(d3.best).not.toBeNull();
    expect(d3.best!.stage).not.toBe("ok");
    expect(d3.bestCoarseScore).toBe(d3.trials[0].coarseScore);
    console.log(`[reloc-diag] other scene: stage ${d3.best!.stage} ncc ${d3.bestCoarseScore.toFixed(2)} lk ${d3.best!.lkRatio.toFixed(2)} ${d3.best!.inlierCount}i`);
  });
});

// ---- Engine-level confirmation and post-relocalization monitoring ----
const Kmat = new Float64Array([K.fx, 0, K.cx, 0, K.fy, K.cy, 0, 0, 1]);
const KInv = mat3Invert(Kmat)!;
const I: Mat3 = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
const base = makeTexture(W, H, createRng(3030), [12, 30, 70, 160]);
const n = [0, Math.SQRT1_2, Math.SQRT1_2];
const gravity = [0, Math.SQRT1_2, Math.SQRT1_2];

function pixelHomography(r: Mat3, t: ArrayLike<number>, nn: ArrayLike<number>, d: number): Mat3 {
  const h = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) h[i * 3 + j] = r[i * 3 + j] + (t[i] * nn[j]) / d;
  return mat3Multiply(mat3Multiply(Kmat, h), KInv);
}
function deskFrame(f: number): Uint8Array {
  const t = [-0.004 * f, -0.001 * f, 0];
  return f === 0 ? base : warpImage(base, W, H, pixelHomography(I, t, n, 1), 128);
}
function input(frameId: number, gray: Uint8Array): VisionInput {
  return { frameId, timestamp: frameId * 33.3, width: W, height: H, gray, intrinsics: K, gravity };
}

describe("Relocalization confirmation and consistency (v5 §10–§11, §16–§17)", () => {
  it("holds a validated candidate for one frame, applies it when reproduced, and the following map PnP agrees with it", () => {
    // Never "clearly high quality": every candidate goes through confirmation.
    const engine = new VisionEngine(W, H, resolveConfig({ relocalization: { immediateInliers: 100000 } }), createRng(11));
    const outs: VisionOutput[] = [];
    let frameId = 0;
    for (let f = 0; f < 30; f++, frameId++) outs.push(engine.process(input(frameId, deskFrame(f))));
    expect(outs[29].state).toBe(TrackingState.PLANE_FOUND);
    const mapFrameId = outs[29].mapPose!.mapFrameId;
    const blank = new Uint8Array(W * H).fill(96);
    // Longer than landmarks.reassociateMaxLostFrames: the remembered landmark
    // image positions expire, so the keyframe relocalization is the only way
    // back (re-association cannot pre-empt it).
    const gap = resolveConfig().landmarks.reassociateMaxLostFrames + 5;
    for (let g = 0; g < gap; g++, frameId++) outs.push(engine.process(input(frameId, blank)));
    expect(outs[outs.length - 1].mapPose!.framesSinceTracked).toBeGreaterThan(0);
    expect(outs[outs.length - 1].mapPose!.mapFrameId).toBe(mapFrameId); // not reset

    // While lost on blank frames, the last attempt's diagnostics stay on the
    // output between attempts (v6: the HUD must not read "none" with nothing
    // else two frames out of three) and say the keyframes dropped out early.
    const lastBlank = outs[outs.length - 1].relocalization;
    expect(lastBlank.diagnostics).not.toBeNull();
    expect(lastBlank.diagnostics!.validated).toBe(0);
    expect(lastBlank.diagnostics!.candidatesTried).toBeGreaterThan(0);
    expect(lastBlank.framesSinceAttempt).toBeGreaterThanOrEqual(0);
    expect(lastBlank.framesSinceAttempt).toBeLessThan(resolveConfig().relocalization.attemptEveryNFrames);
    expect(outs[29].relocalization.diagnostics).toBeNull(); // nothing while tracking

    const attempts: string[] = [];
    let successFrame = -1;
    for (let f = 36; f < 60; f++, frameId++) {
      const o = engine.process(input(frameId, deskFrame(f)));
      outs.push(o);
      attempts.push(o.relocalization.attempt);
      if (successFrame < 0 && o.relocalization.attempt === "success") successFrame = outs.length - 1;
    }
    console.log(`[reloc-v5] attempts after return: ${attempts.join(" ")}`);
    expect(successFrame).toBeGreaterThan(0);
    // A candidate frame precedes the applied frame (confirmation), never an
    // immediate apply.
    const idx = attempts.indexOf("success");
    expect(idx).toBeGreaterThan(0);
    expect(attempts.slice(0, idx)).toContain("candidate");
    expect(attempts[idx - 1]).toBe("candidate");

    const applied = outs[successFrame];
    expect(applied.mapPose!.mapFrameId).toBe(mapFrameId);
    expect(applied.mapPose!.relocalized).toBe(true);
    expect(applied.mapPose!.sourceHistory.endsWith("R")).toBe(true);
    expect(applied.relocalization.inlierCount).toBeGreaterThanOrEqual(resolveConfig().relocalization.minInliers);
    expect(applied.relocalization.spatialCells).toBeGreaterThanOrEqual(resolveConfig().relocalization.minSpatialCells);
    expect(applied.relocalization.jumpTranslation).toBeGreaterThan(0); // diagnostic: distance from the held pose

    // Post-relocalization monitor: the map PnP of the next frames stays close
    // to the relocalized pose (camera motion only) and never contradicts it.
    const after = outs.slice(successFrame + 1, successFrame + 1 + resolveConfig().relocalization.postRelocMonitorFrames);
    expect(after.length).toBeGreaterThan(0);
    for (const o of after) {
      expect(o.mapPose!.framesSinceTracked).toBe(0);
      expect(o.mapPose!.relocalized).toBe(true);
      expect(o.relocalization.postInconsistent).toBe(false);
    }
    const lastMon = after[after.length - 1].relocalization;
    console.log(`[reloc-v5] post-reloc map Δ ${lastMon.postDeltaTranslation.toFixed(3)} map units / ${lastMon.postDeltaRotationDeg.toFixed(2)}°`);
    expect(lastMon.postDeltaRotationDeg).toBeLessThan(3);
    expect(lastMon.postDeltaTranslation).toBeLessThan(1.0);
    // The monitoring window ends: the flag clears, and the lost-episode
    // diagnostics are gone once tracking resumed.
    const later = outs[outs.length - 1];
    expect(later.mapPose!.relocalized).toBe(false);
    expect(later.mapPose!.framesSinceTracked).toBe(0);
    expect(later.relocalization.diagnostics).toBeNull();
    expect(later.mapPose!.recoveryMode).toBe("tracking");
    expect(later.mapPose!.requiredInliers).toBe(resolveConfig().landmarks.minPnPInliers);
    // While lost the recovery rule was visible on the output (v6 §10).
    // (the rule is chosen from the lost count before this frame's increment)
    const lostOut = outs.find((o) => o.mapPose && o.mapPose.framesSinceTracked > resolveConfig().landmarks.longLostFrames + 1)!;
    expect(lostOut.mapPose!.recoveryMode).toBe("long");
    expect(lostOut.mapPose!.requiredInliers).toBe(resolveConfig().landmarks.minRecoveryInliersLong);
  });

  it("applies a clearly high-quality candidate at once (no confirmation delay) with the default config", () => {
    const engine = new VisionEngine(W, H, resolveConfig(), createRng(11));
    let frameId = 0;
    for (let f = 0; f < 30; f++, frameId++) engine.process(input(frameId, deskFrame(f)));
    const blank = new Uint8Array(W * H).fill(96);
    const gap = resolveConfig().landmarks.reassociateMaxLostFrames + 5;
    for (let g = 0; g < gap; g++, frameId++) engine.process(input(frameId, blank));
    const attempts: string[] = [];
    let applied: VisionOutput | null = null;
    for (let f = 36; f < 50 && !applied; f++, frameId++) {
      const o = engine.process(input(frameId, deskFrame(f)));
      attempts.push(o.relocalization.attempt);
      if (o.relocalization.attempt === "success") applied = o;
    }
    expect(applied).not.toBeNull();
    expect(applied!.relocalization.inlierCount).toBeGreaterThanOrEqual(resolveConfig().relocalization.immediateInliers);
    expect(attempts.filter((a) => a === "candidate").length).toBe(0);
  });
});
