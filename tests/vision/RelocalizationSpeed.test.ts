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
import { downsampleToCoarse } from "../../src/vision/Keyframe";
import { coarseShift, Relocalizer } from "../../src/vision/Relocalizer";
import {
  candidateAction,
  candidateStale,
  lkFailureReason,
  shouldAttemptRelocalization,
  shouldPrepareRelocalization,
} from "../../src/vision/RelocalizationSchedule";
import type { Track, VisionInput, VisionOutput } from "../../src/vision/types";
import { VisionEngine } from "../../src/vision/VisionEngine";
import { makeTexture, translateImage, warpImage } from "../helpers/synthetic";

/**
 * 修正指示書 v14: find the *right* keyframe faster — rank every keyframe by
 * coarse similarity, send only the top K to LK / PnP, do not retry a failed
 * keyframe on an unchanged view, stop at the first strong candidate, and
 * measure each stage. PnP, validation (v12) and confirmation are unchanged.
 */
const W = 640;
const H = 480;
const K = approximateIntrinsics(W, H);
const identity: RigidTransform = { rotation: mat3Identity(), translation: new Float64Array(3) };
const CFG = resolveConfig();
const RC = CFG.relocalization;

function track(id: number, x: number, y: number, landmarkId: number): Track {
  return {
    id, x, y, prevX: x, prevY: y, age: 5, score: 10, inlier: true, outlierStreak: 0,
    refX: x, refY: y, refFrame: -1, landmarkId, anchorFrame: -1, anchorX: x, anchorY: y, anchorPose: null,
    planePoint: null, planeStreak: 0, planeOutliers: 0, offPlane: false,
  };
}

function pyramidOf(gray: Uint8Array): ImagePyramid {
  const p = new ImagePyramid(W, H, CFG.tracker.pyramidLevels);
  p.build(gray);
  return p;
}

/**
 * A keyframe store with one identity-pose keyframe per texture; each
 * keyframe observes its own landmarks at depth 2 (only corners passing
 * `keep`). Keyframe ids are 1…n in texture order.
 */
function keyframeStore(textures: Uint8Array[], keep: (kf: number, x: number, y: number) => boolean = () => true) {
  const map = new LandmarkMap();
  const reloc = new Relocalizer(RC, CFG.tracker);
  const detector = new FeatureDetector(W, H, CFG.features);
  let id = 1;
  textures.forEach((gray, kf) => {
    const tracks: Track[] = [];
    for (const c of detector.detect(gray, { wanted: 300 })) {
      if (!keep(kf, c.x, c.y)) continue;
      const Z = 2;
      const lm = map.add([((c.x - K.cx) / K.fx) * Z, ((c.y - K.cy) / K.fy) * Z, Z], id, 0);
      lm.observations = 5;
      tracks.push(track(id++, c.x, c.y, lm.id));
    }
    reloc.create(pyramidOf(gray), identity, tracks, kf * 10, kf * 333);
  });
  return { map, reloc };
}

const textures = Array.from({ length: 8 }, (_, i) => makeTexture(W, H, createRng(100 + i), [12, 30, 70, 160]));

describe("v14 scheduling (Tests 1–4, 10–14, 17)", () => {
  it("Test 1 / 2 / 3: preparation follows the tracking quality, never a lost state; fast motion with a healthy PnP prepares only", () => {
    const th = { minInliers: RC.prepareMinInliers, maxErrorPx: RC.prepareMaxErrorPx };
    const healthy = { worldEstablished: true, tracked: true, inliers: 120, errorPx: 1.0, motionLevel: "normal" as const };
    expect(shouldPrepareRelocalization(healthy, th)).toBe(false);
    expect(shouldPrepareRelocalization({ ...healthy, inliers: 29, errorPx: 1.82 }, th)).toBe(true); // the on-device weak PnP (v14)
    expect(shouldPrepareRelocalization({ ...healthy, errorPx: 4.0 }, th)).toBe(true); // large error alone
    expect(shouldPrepareRelocalization({ ...healthy, motionLevel: "fast" }, th)).toBe(true); // fast motion, healthy PnP
    expect(shouldPrepareRelocalization({ ...healthy, motionLevel: "medium" }, th)).toBe(false);
    expect(shouldPrepareRelocalization({ ...healthy, tracked: false, inliers: 0 }, th)).toBe(false); // lost: attempt, not prepare
    expect(shouldPrepareRelocalization({ ...healthy, worldEstablished: false, inliers: 10 }, th)).toBe(false);
    // v16: Android-style healthy tracking (many inliers, 2.6–3.3 px error,
    // never "trusted" at 1.5 px) does not keep the preparation running.
    expect(shouldPrepareRelocalization({ ...healthy, inliers: 101, errorPx: 2.65 }, th)).toBe(false);
    expect(shouldPrepareRelocalization({ ...healthy, inliers: 64, errorPx: 3.27 }, th)).toBe(false);
  });

  it("Test 3 / 4 / 7: an attempt needs an established world and a map PnP that is not locating the camera", () => {
    const base = { worldEstablished: true, framesSinceTracked: 0, startAfterLostFrames: RC.startAfterLostFrames, attemptEveryNFrames: RC.attemptEveryNFrames, pending: false };
    // Fast motion, healthy PnP (framesSinceTracked 0): never an attempt.
    expect(shouldAttemptRelocalization(base)).toBe(false);
    // Lost: from the first lost frame on (startAfterLostFrames 1), every 3rd frame.
    expect(shouldAttemptRelocalization({ ...base, framesSinceTracked: 1 })).toBe(true);
    expect(shouldAttemptRelocalization({ ...base, framesSinceTracked: 2 })).toBe(false);
    expect(shouldAttemptRelocalization({ ...base, framesSinceTracked: 4 })).toBe(true);
    // A pending candidate is confirmed every frame.
    expect(shouldAttemptRelocalization({ ...base, framesSinceTracked: 2, pending: true })).toBe(true);
    // No world: scan continues, nothing to return to (v10).
    expect(shouldAttemptRelocalization({ ...base, framesSinceTracked: 10, worldEstablished: false })).toBe(false);
  });

  it("Test 12 / 13: a strong candidate is applied at once only when clearly high quality; an acceptable one is always confirmed", () => {
    expect(candidateAction("strong", RC.immediateInliers, RC.immediateMaxErrorPx, RC, 1)).toBe("apply");
    expect(candidateAction("strong", RC.immediateInliers - 1, RC.immediateMaxErrorPx, RC, 1)).toBe("confirm");
    expect(candidateAction("strong", 100, RC.immediateMaxErrorPx + 0.1, RC, 1)).toBe("confirm");
    expect(candidateAction("acceptable", 200, 0.5, RC, 1)).toBe("confirm");
    expect(candidateAction("strong", 30, 1.4, RC, 0)).toBe("apply"); // no confirmation configured for strong
  });

  it("Test 17: a pending candidate older than its age limit is stale", () => {
    expect(candidateStale(10, 11, RC.candidateMaxAgeFrames)).toBe(false);
    expect(candidateStale(10, 12, RC.candidateMaxAgeFrames)).toBe(false);
    expect(candidateStale(10, 13, RC.candidateMaxAgeFrames)).toBe(true);
    expect(candidateStale(10, 10, 0)).toBe(false);
    expect(candidateStale(10, 11, 0)).toBe(true);
  });

  it("§49: the LK failure reason is the dominant rejected status, insufficient_tracks without one", () => {
    const base = { ok: 10, outOfBounds: 0, diverged: 0, lowTexture: 0, highResidual: 0, fbError: 0, tooFar: 0 };
    expect(lkFailureReason(base)).toBe("insufficient_tracks");
    // Blur: starts inside the image, iterations ran out of it (on-device 154/179).
    expect(lkFailureReason({ ...base, diverged: 154, fbError: 15, highResidual: 3 })).toBe("diverged");
    expect(lkFailureReason({ ...base, fbError: 40, highResidual: 5 })).toBe("high_fb_error");
    expect(lkFailureReason({ ...base, highResidual: 30, tooFar: 4 })).toBe("high_lk_error");
    expect(lkFailureReason({ ...base, tooFar: 25 })).toBe("too_far");
    expect(lkFailureReason({ ...base, outOfBounds: 25, lowTexture: 2 })).toBe("out_of_bounds");
    expect(lkFailureReason({ ...base, lowTexture: 25 })).toBe("low_texture");
    // No majority: a mix of everything is just too few tracks.
    expect(lkFailureReason({ ...base, fbError: 10, highResidual: 10, tooFar: 10, outOfBounds: 10 })).toBe("insufficient_tracks");
  });
});

describe("v14 keyframe ranking and search budget (Tests 5–11, 14, AC-2 / AC-3)", () => {
  it("Test 5 / 6 / 8 / 10 / 11: 8 keyframes are ranked by coarse similarity, only the top K reach LK, the matching one goes LK → PnP → validation", () => {
    const { map, reloc } = keyframeStore(textures);
    expect(reloc.count).toBe(8);
    // The camera is back at the view of keyframe 6 (texture index 5), shifted a little.
    const current = pyramidOf(translateImage(textures[5], W, H, 18, -10, 128));
    const r = reloc.relocalize(current, map, K, null, 0, { frameId: 100 });
    // Warm timing (the first call includes JIT warm-up): the same attempt again.
    const t0 = performance.now();
    reloc.relocalize(current, map, K, null, 0, { frameId: 200 });
    const ms = performance.now() - t0;
    const t1 = performance.now();
    reloc.prepare(current, 300);
    const rankMs = performance.now() - t1;
    const d = r.diagnostics;
    expect(d.ranked).toHaveLength(8); // every keyframe ranked …
    for (let i = 1; i < d.ranked.length; i++) expect(d.ranked[i - 1].rankScore).toBeGreaterThanOrEqual(d.ranked[i].rankScore); // … best first
    expect(d.ranked[0].keyframeId).toBe(6); // AC-2: similarity, not keyframe order
    expect(d.ranked[0].rankScore).toBeGreaterThan(0.6);
    expect(d.lkCandidates).toBeLessThanOrEqual(RC.maxLkCandidatesPerFrame); // Test 6: top K only
    expect(d.ranked.filter((k) => k.selected)).toHaveLength(d.lkCandidates);
    expect(d.candidatesTried).toBeLessThanOrEqual(RC.maxLkCandidatesPerFrame);
    // Test 8 / 10 / 11: the matching keyframe passed coarse → LK → PnP → validation.
    expect(r.success).toBe(true);
    expect(r.keyframeId).toBe(6);
    expect(r.level).toBe("strong");
    expect(d.lkPassed).toBeGreaterThanOrEqual(1);
    expect(d.pnpPassed).toBeGreaterThanOrEqual(1);
    expect(d.validated).toBe(1);
    expect(d.searchStage).toBe("validation");
    const okTrial = d.trials.find((t) => t.keyframeId === 6)!;
    expect(okTrial.stage).toBe("ok");
    expect(okTrial.lkObservations).toBeGreaterThan(0);
    expect(okTrial.lkTracked).toBe(okTrial.lkStatus.ok);
    expect(okTrial.lkFailureReason).toBeNull();
    // §18: the search stopped at the first strong candidate — it was ranked first, so it was the only one tried.
    expect(d.candidatesTried).toBe(1);
    console.log(
      `[v14] ranking 8 keyframes + ${d.candidatesTried} trial(s): ${ms.toFixed(1)} ms (ranking alone ${rankMs.toFixed(1)} ms); top ${d.ranked.slice(0, 3).map((k) => `KF${k.keyframeId} ${k.rankScore.toFixed(2)}`).join(", ")}`,
    );
  });

  it("Test 7 / 14: keyframes with a low coarse score are not sent to LK; with no match the attempt is rejected and the keyframes tried enter a cooldown", () => {
    const { map, reloc } = keyframeStore(textures);
    const other = pyramidOf(makeTexture(W, H, createRng(999), [10, 25, 60, 150]));
    const r = reloc.relocalize(other, map, K, null, 0, { frameId: 50 });
    const d = r.diagnostics;
    expect(r.success).toBe(false);
    expect(d.ranked).toHaveLength(8);
    expect(d.ranked[0].rankScore).toBeLessThan(0.5); // nothing looks like this view
    expect(d.lkCandidates).toBe(RC.maxLkCandidatesPerFrame); // the budget was offered to the top K …
    // Test 7: a keyframe below the coarse gate never reaches LK; the coarse
    // gate (0.25) is a work-saver, so a random view may pass it now and then —
    // such a keyframe fails at LK, never at PnP / validation (Test 14).
    const belowGate = d.trials.filter((t) => t.coarseScore < RC.coarseMinScore);
    for (const t of belowGate) expect(t.stage).toBe("coarse");
    expect(d.lkTested).toBe(d.trials.length - belowGate.length);
    expect(d.lkPassed).toBe(0);
    expect(d.pnpTested).toBe(0);
    expect(d.validated).toBe(0);
    expect(["coarse", "lk"]).toContain(d.searchStage);
    expect(["low_match_score", "lk_failed"]).toContain(r.rejectCode);
    // Every keyframe tried enters a cooldown; the ones not selected do not.
    for (const t of d.trials) expect(reloc.retryStateOf(t.keyframeId)?.cooldownUntil).toBe(50 + RC.retryCooldownFrames);
    for (const k of d.ranked.filter((k) => !k.selected)) expect(reloc.retryStateOf(k.keyframeId)).toBeNull();
    console.log(`[v14] unrelated view: rank top ${d.ranked[0].rankScore.toFixed(2)}, refined best ${d.bestCoarseScore.toFixed(2)}, lk tried ${d.lkTested}, stage ${d.searchStage}`);
  });

  it("Test 9 / AC-4: an LK failure on the first keyframe does not end the attempt — the next keyframe is tried and succeeds", () => {
    // Keyframe 1: the same desk but its observations lie in a region that is
    // flat in the keyframe image (LK has no texture to track there) — coarse
    // similarity still passes. Keyframe 2: the normal desk view. Keyframe 3:
    // unrelated.
    const flat = Uint8Array.from(textures[0]);
    for (let y = 90; y < 390; y++) for (let x = 120; x < 520; x++) flat[y * W + x] = 128;
    const inFlat = (x: number, y: number) => x > 140 && x < 500 && y > 110 && y < 370;
    // Corners of the flat image lie outside the rectangle; use the original's
    // corners inside the rectangle as the observations instead.
    const map = new LandmarkMap();
    const reloc = new Relocalizer(RC, CFG.tracker);
    const detector = new FeatureDetector(W, H, CFG.features);
    let id = 1;
    const observe = (gray: Uint8Array, corners: { x: number; y: number }[], frame: number) => {
      const tracks: Track[] = [];
      for (const c of corners) {
        const Z = 2;
        const lm = map.add([((c.x - K.cx) / K.fx) * Z, ((c.y - K.cy) / K.fy) * Z, Z], id, 0);
        lm.observations = 5;
        tracks.push(track(id++, c.x, c.y, lm.id));
      }
      reloc.create(pyramidOf(gray), identity, tracks, frame, frame * 33);
    };
    const deskCorners = detector.detect(textures[0], { wanted: 300 });
    observe(flat, deskCorners.filter((c) => inFlat(c.x, c.y)), 0); // KF1: observations on flat pixels
    observe(textures[0], deskCorners, 10); // KF2: the real view
    observe(textures[7], detector.detect(textures[7], { wanted: 300 }), 20); // KF3: elsewhere
    expect(reloc.keyframes[0].observations.length).toBeGreaterThanOrEqual(RC.minInliers);

    const current = pyramidOf(translateImage(textures[0], W, H, 6, 4, 128));
    // Force KF1 first (the pending-candidate path uses the same preference).
    const r = reloc.relocalize(current, map, K, null, 0, { frameId: 60, preferKeyframeId: 1 });
    const d = r.diagnostics;
    expect(d.ranked[0].keyframeId).toBe(1);
    const t1 = d.trials.find((t) => t.keyframeId === 1)!;
    expect(t1.coarseScore).toBeGreaterThanOrEqual(RC.coarseMinScore); // coarse passed …
    expect(t1.stage).toBe("lk"); // … LK did not
    expect(t1.lkFailureReason).toBe("low_texture"); // §49: and we know why
    expect(t1.lkStatus.lowTexture).toBeGreaterThan(0);
    // AC-4: the search went on to KF2 and succeeded.
    const t2 = d.trials.find((t) => t.keyframeId === 2)!;
    expect(t2.stage).toBe("ok");
    expect(r.success).toBe(true);
    expect(r.keyframeId).toBe(2);
    expect(d.lkTested).toBe(2);
    expect(d.lkPassed).toBe(1);
    // The failed keyframe is in a cooldown, the successful one is not.
    expect(reloc.retryStateOf(1)!.cooldownUntil).toBe(60 + RC.retryCooldownFrames);
    expect(reloc.retryStateOf(1)!.lastStage).toBe("lk");
    expect(reloc.retryStateOf(2)!.cooldownUntil).toBe(-1);
    expect(reloc.retryStateOf(2)!.lastSuccessFrame).toBe(60);
  });

  it("Test 15 / 16 / AC-5: a keyframe that failed is not retried on the same view within its cooldown, the other keyframes get their turn, and a changed image retries at once", () => {
    const { map, reloc } = keyframeStore(textures);
    const other = pyramidOf(makeTexture(W, H, createRng(999), [10, 25, 60, 150]));
    const tried = (frameId: number, p: ImagePyramid) => {
      const r = reloc.relocalize(p, map, K, null, 0, { frameId });
      return { r, ids: r.diagnostics.trials.map((t) => t.keyframeId).sort((a, b) => a - b) };
    };
    const a = tried(10, other);
    expect(a.ids).toHaveLength(RC.maxLkCandidatesPerFrame);
    expect(a.r.diagnostics.retrySuppressed).toBe(0);
    // Same view next frame: the three that failed are skipped, three others are tried (AC-4).
    const b = tried(11, other);
    expect(b.r.diagnostics.retrySuppressed).toBe(RC.maxLkCandidatesPerFrame);
    expect(b.ids.filter((i) => a.ids.includes(i))).toHaveLength(0);
    expect(b.ids).toHaveLength(RC.maxLkCandidatesPerFrame);
    // Third frame: the remaining two, then nothing is left to try on this view.
    const c = tried(12, other);
    expect(c.ids).toHaveLength(8 - 2 * RC.maxLkCandidatesPerFrame);
    const dd = tried(13, other);
    expect(dd.r.rejectCode).toBe("retry_cooldown");
    expect(dd.r.diagnostics.candidatesTried).toBe(0);
    expect(dd.r.diagnostics.retrySuppressed).toBeGreaterThan(0);
    expect(dd.r.diagnostics.ranked).toHaveLength(8); // still ranked and visible
    // Test 16: the camera comes back to keyframe 3's view while every
    // keyframe is still inside its cooldown — the image changed, so it is
    // tried and relocalizes.
    const back = tried(14, pyramidOf(translateImage(textures[2], W, H, -12, 7, 128)));
    expect(back.r.diagnostics.retrySuppressed).toBe(0);
    expect(back.r.success).toBe(true);
    expect(back.r.keyframeId).toBe(3);
    expect(reloc.retryStateOf(3)!.lastSuccessFrame).toBe(14);
    // After the cooldown the same failed view is tried again (no permanent exclusion, §34–§35).
    const again = tried(10 + RC.retryCooldownFrames + 1, other);
    expect(again.r.diagnostics.retrySuppressed).toBe(0);
  });

  it("Test 12 / 13 (ranking): a strong candidate ranked first ends the search; preparation reuses the ranking on an unchanged view", () => {
    const { map, reloc } = keyframeStore(textures);
    const current = pyramidOf(translateImage(textures[3], W, H, 5, 3, 128));
    // Prepared on the previous (tracked) frame, used on the first lost frame.
    const prep = reloc.prepare(current, 40)!;
    expect(prep.ranked[0].keyframeId).toBe(4);
    const r = reloc.relocalize(current, map, K, null, 0, { frameId: 41, firstAttempt: true });
    expect(r.diagnostics.usedPreparedRanking).toBe(true);
    expect(r.success).toBe(true);
    expect(r.keyframeId).toBe(4);
    expect(r.diagnostics.candidatesTried).toBe(1); // §18 early success
    // A changed view does not reuse it.
    const elsewhere = pyramidOf(textures[6]);
    const r2 = reloc.relocalize(elsewhere, map, K, null, 0, { frameId: 42 });
    expect(r2.diagnostics.usedPreparedRanking).toBe(false);
    expect(r2.keyframeId).toBe(7);
    // Within the preparation period it is reused; older than that not.
    reloc.prepare(current, 50);
    const r3 = reloc.relocalize(current, map, K, null, 0, { frameId: 50 + RC.prepareEveryNFrames });
    expect(r3.diagnostics.usedPreparedRanking).toBe(true);
    reloc.prepare(current, 70);
    const r4 = reloc.relocalize(current, map, K, null, 0, { frameId: 70 + RC.prepareEveryNFrames + 1 });
    expect(r4.diagnostics.usedPreparedRanking).toBe(false);
  });

  it("follow-up (on-device `out_of_bounds`): a shift that leaves too few observations inside the image skips LK with that reason; the coarse search weighs the overlap", () => {
    // Keyframe observing only the right-hand band of the desk; the camera
    // then pans so the content moves 180 px to the right: the band leaves
    // the image, so no relocalization from this keyframe is possible.
    const { map, reloc } = keyframeStore([textures[0], textures[1]], (kf, x) => kf !== 0 || x > 430);
    expect(reloc.keyframes[0].observations.length).toBeGreaterThanOrEqual(RC.minInliers);
    const panned = pyramidOf(translateImage(textures[0], W, H, 180, 0, 128));
    const r = reloc.relocalize(panned, map, K, null, 0, { frameId: 10, preferKeyframeId: 1 });
    const t1 = r.diagnostics.trials.find((t) => t.keyframeId === 1)!;
    expect(t1.coarseScore).toBeGreaterThanOrEqual(RC.coarseMinScore); // the pan was found …
    expect(t1.stage).toBe("lk");
    expect(t1.lkFailureReason).toBe("out_of_bounds"); // … but the band is gone
    expect(t1.lkTracked).toBe(0);
    expect(t1.lkStatus.outOfBounds).toBeGreaterThan(t1.lkObservations - RC.minInliers);
    expect(r.diagnostics.lkTested).toBe(0); // no LK spent on it (keyframe 2 is unrelated and fails at coarse)
    expect(r.reason).toMatch(/inside the image/);
    expect(reloc.retryStateOf(1)!.lastStage).toBe("lk");

    // Overlap weighting: an identical image is matched at zero shift with
    // full overlap; a shifted copy is still found, with the overlap reported.
    const a = reloc.keyframes[0].coarse;
    const same = coarseShift(a, a, RC.coarseSearchRadius, RC.coarseMinOverlap);
    expect(same.dx).toBe(0);
    expect(same.dy).toBe(0);
    expect(same.overlap).toBe(1);
    expect(same.weightedScore).toBeCloseTo(same.score, 12);
    const shifted = downsampleToCoarse(pyramidOf(translateImage(textures[0], W, H, 64, -32, 128)).levels[2]);
    const s = coarseShift(a, shifted, RC.coarseSearchRadius, RC.coarseMinOverlap);
    expect(s.dx).toBe(8); // 64 px / 8
    expect(s.dy).toBe(-4);
    expect(s.overlap).toBeLessThan(1);
    expect(s.overlap).toBeGreaterThanOrEqual(RC.coarseMinOverlap * RC.coarseMinOverlap);
    expect(s.weightedScore).toBeLessThan(s.score);
    // Shifts with less than the minimum overlap per axis are never chosen.
    const far = coarseShift(a, shifted, 60, 0.5);
    expect(Math.abs(far.dx)).toBeLessThanOrEqual(a.width / 2);
    expect(Math.abs(far.dy)).toBeLessThanOrEqual(a.height / 2);
  });
});

describe("v15 keyframe coverage: no churn on a swinging camera, redundancy-based eviction", () => {
  const rotY = (deg: number): RigidTransform => {
    const a = (deg * Math.PI) / 180;
    const c = Math.cos(a), s = Math.sin(a);
    return { rotation: new Float64Array([c, 0, s, 0, 1, 0, -s, 0, c]), translation: new Float64Array([0, 0, 0]) };
  };
  const tracksFor = (n: number): Track[] => Array.from({ length: n }, (_, i) => track(i + 1, 100 + i, 100 + i, i + 1));
  const pyramid = pyramidOf(textures[0]);
  const DEPTH = 2;

  it("a camera swinging between two views creates two keyframes, not one every keyframeMinFrameGap frames", () => {
    const reloc = new Relocalizer(RC, CFG.tracker);
    let frame = 0;
    let created = 0;
    // 30 swings A (0°) ↔ B (+15°), each held for keyframeMinFrameGap frames.
    for (let swing = 0; swing < 30; swing++) {
      const pose = rotY(swing % 2 === 0 ? 0 : 15);
      for (let f = 0; f < RC.keyframeMinFrameGap; f++, frame++) {
        if (reloc.shouldCreate(pose, frame, 100, 0, DEPTH)) {
          reloc.create(pyramid, pose, tracksFor(100), frame, frame * 33, null, DEPTH);
          created++;
        }
      }
    }
    // The two views are stored once each; only the time-based refresh
    // (keyframeMaxFrameGap) adds to the count, never the swinging itself.
    const refreshes = Math.floor(frame / RC.keyframeMaxFrameGap);
    expect(created).toBeLessThanOrEqual(2 + refreshes);
    expect(reloc.count).toBeLessThanOrEqual(RC.maxKeyframes);
    expect(reloc.keyframes[0].id).toBe(1); // the origin view is never dropped
    // Both views are still covered.
    expect(reloc.nearestKeyframeDistance(rotY(0), DEPTH)).toBeLessThan(1);
    expect(reloc.nearestKeyframeDistance(rotY(15), DEPTH)).toBeLessThan(1);
    console.log(`[v15] 30 swings over ${frame} frames: ${created} keyframes created (old policy: ~${Math.floor(frame / RC.keyframeMinFrameGap)}), ${reloc.count} kept`);
  });

  it("when full, the most redundant keyframe is dropped — never the first or the newest — so distinct views survive", () => {
    const reloc = new Relocalizer(RC, CFG.tracker);
    // 8 distinct views 15° apart …
    for (let i = 0; i < RC.maxKeyframes; i++) reloc.create(pyramid, rotY(i * 15), tracksFor(50), i * 20, 0, null, DEPTH);
    expect(reloc.count).toBe(RC.maxKeyframes);
    // … then a view only 4° from view 3 (a near duplicate) is forced in: it
    // is the newest, so its twin (view 3, id 4) is the one evicted — not the
    // oldest non-origin view (id 2).
    reloc.create(pyramid, rotY(3 * 15 + 4), tracksFor(50), 200, 0, null, DEPTH);
    const ids = reloc.keyframes.map((k) => k.id);
    expect(reloc.count).toBe(RC.maxKeyframes);
    expect(ids).toContain(1);
    expect(ids).toContain(2);
    expect(ids).not.toContain(4);
    expect(ids).toContain(9);
    expect(reloc.evictions).toBe(1);
    expect(reloc.createdCount).toBe(9);
    // shouldCreate refuses that near-duplicate view in the first place (within
    // the min / max frame gaps, so neither time rule applies).
    const f = 200 + RC.keyframeMinFrameGap + 1;
    expect(f - 200).toBeLessThan(RC.keyframeMaxFrameGap);
    expect(reloc.shouldCreate(rotY(3 * 15 + 4), f, 100, 0, DEPTH)).toBe(false);
    // A genuinely new view (far from all) is accepted; so is a pure translation beyond the depth ratio.
    expect(reloc.shouldCreate(rotY(8 * 15), f, 100, 0, DEPTH)).toBe(true);
    const shifted: RigidTransform = { rotation: mat3Identity(), translation: new Float64Array([DEPTH * RC.keyframeTranslationDepthRatio * 1.5, 0, 0]) };
    expect(reloc.shouldCreate(shifted, f, 100, RC.keyframeParallaxPx + 1, DEPTH)).toBe(true);
    // Back at the origin view: 49° from the last keyframe (the old trigger
    // would fire) but identical to keyframe 1 → refused (coverage).
    expect(reloc.shouldCreate(rotY(0), f, 100, 0, DEPTH)).toBe(false);
    // A small translation from the origin view, below the depth ratio, is not a new view either.
    const near: RigidTransform = { rotation: mat3Identity(), translation: new Float64Array([DEPTH * RC.keyframeTranslationDepthRatio * 0.5, 0, 0]) };
    expect(reloc.shouldCreate(near, f, 100, RC.keyframeParallaxPx + 1, DEPTH)).toBe(false);
    // The time-based refresh still applies after keyframeMaxFrameGap frames.
    expect(reloc.shouldCreate(rotY(3 * 15 + 4), 200 + RC.keyframeMaxFrameGap, 100, 0, DEPTH)).toBe(true);
  });
});

describe("v15 keyframe landmarks: protected from the age prune, thin keyframes never spend an LK slot", () => {
  it("LandmarkMap.prune keeps protected landmarks past maxAge and removes unprotected ones first at the cap", () => {
    const map = new LandmarkMap();
    for (let i = 1; i <= 10; i++) {
      const lm = map.add([i, 0, 2], i, 0);
      lm.lastSeenFrame = i <= 5 ? 0 : 100; // 1–5 old, 6–10 recent
    }
    const protectedIds = new Set([1, 2]);
    // Age rule: old unprotected 3, 4, 5 go; protected 1, 2 stay although just as old.
    expect(map.prune(200, 150, 1000, protectedIds)).toBe(3);
    expect(map.get(1)).toBeDefined();
    expect(map.get(2)).toBeDefined();
    expect(map.get(3)).toBeUndefined();
    expect(map.size).toBe(7);
    // Cap: applies to the 5 unprotected landmarks only (5 → 4 removes one), never to 1 / 2,
    // so a large protected set cannot starve the current view of fresh landmarks.
    expect(map.prune(200, 150, 4, protectedIds)).toBe(1);
    expect(map.get(1)).toBeDefined();
    expect(map.get(2)).toBeDefined();
    expect(map.size).toBe(6);
    expect(map.countUnprotected(protectedIds)).toBe(4);
    expect(map.countUnprotected(null)).toBe(6);
    // Without a protected set the old behaviour: everything old goes.
    const plain = new LandmarkMap();
    for (let i = 1; i <= 3; i++) plain.add([i, 0, 2], i, 0).lastSeenFrame = 0;
    expect(plain.prune(200, 150, 1000)).toBe(3);
  });

  it("a keyframe whose landmarks left the map is ranked but not tried, is evicted first, and the engine protects keyframe landmarks", () => {
    // Two matching keyframes of the same view; the landmarks of the second
    // are removed from the map (as the age prune did on device).
    const { map, reloc } = keyframeStore([textures[0], textures[0]]);
    const thin = reloc.keyframes[1];
    for (const o of thin.observations) map.remove(o.landmarkId);
    expect(reloc.aliveObservations(thin, map)).toBe(0);
    expect(reloc.referencedLandmarkIds().has(thin.observations[0].landmarkId)).toBe(true);
    const current = pyramidOf(translateImage(textures[0], W, H, 6, -4, 128));
    const r = reloc.relocalize(current, map, K, null, 0, { frameId: 10, preferKeyframeId: thin.id });
    const d = r.diagnostics;
    const rankedThin = d.ranked.find((k) => k.keyframeId === thin.id)!;
    expect(rankedThin.unusable).toBe(true);
    expect(rankedThin.alive).toBe(0);
    expect(rankedThin.selected).toBe(false);
    expect(d.unusableKeyframes).toBe(1);
    expect(d.trials.some((t) => t.keyframeId === thin.id)).toBe(false); // no LK slot spent on it
    expect(r.success).toBe(true);
    expect(r.keyframeId).toBe(1);
    // Eviction prefers the thin keyframe over any viewpoint argument.
    const DEPTH = 2;
    const far: RigidTransform = { rotation: mat3Identity(), translation: new Float64Array([DEPTH * 2, 0, 0]) };
    const store = new Relocalizer(RC, CFG.tracker);
    const mapB = new LandmarkMap();
    const detector = new FeatureDetector(W, H, CFG.features);
    let id = 1;
    const mk = (pose: RigidTransform, frame: number, alive: boolean) => {
      const tracks: Track[] = [];
      for (const c of detector.detect(textures[frame % textures.length], { wanted: 300 })) {
        const lm = mapB.add([((c.x - K.cx) / K.fx) * 2, ((c.y - K.cy) / K.fy) * 2, 2], id, 0);
        lm.observations = 5;
        tracks.push(track(id++, c.x, c.y, lm.id));
      }
      store.create(pyramidOf(textures[frame % textures.length]), pose, tracks, frame * 20, 0, mapB, DEPTH);
      if (!alive) for (const t of tracks) mapB.remove(t.landmarkId);
    };
    for (let i = 0; i < RC.maxKeyframes; i++) mk({ rotation: mat3Identity(), translation: new Float64Array([i * 0.5, 0, 0]) }, i, i !== 3);
    mk(far, 20, true); // 9th: the thin keyframe (id 4) must go, although it is far from its neighbours
    expect(store.keyframes.map((k) => k.id)).not.toContain(4);
    expect(store.keyframes.map((k) => k.id)).toContain(2);
    expect(store.count).toBe(RC.maxKeyframes);

    // Engine: after 220 tracked frames of a moving camera every stored keyframe
    // still has at least minInliers landmarks in the map (the age prune alone
    // would have removed the first view's landmarks after maxLandmarkAgeFrames).
    const engine = new VisionEngine(W, H, resolveConfig(), createRng(13));
    let last: VisionOutput | null = null;
    for (let f = 0; f < 220; f++) last = engine.process(input(f, deskFrame(f)));
    expect(last!.relocalization.keyframes).toBeGreaterThanOrEqual(3);
    const summary = engine.keyframeSummary();
    expect(summary[0].frameId).toBeLessThan(220 - resolveConfig().landmarks.maxLandmarkAgeFrames);
    for (const kf of summary) expect(kf.alive).toBeGreaterThanOrEqual(RC.minInliers);
    console.log(`[v15] keyframes after 220 frames: ${summary.map((k) => `KF${k.id}@${k.frameId} ${k.alive}/${k.observations}`).join("  ")}`);
  });
});

// ---- Engine level: timeline, stages, no change to tracking / visibility inputs ----
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

describe("v14 engine: episode timeline, search stage, unchanged tracking (Tests 1, 4, 18, 19, §45–§46)", () => {
  it("measures lost → coarse → LK → PnP → validation → applied, and the stages read confirming / applied; tracking itself never attempts", () => {
    const engine = new VisionEngine(W, H, resolveConfig({ relocalization: { immediateInliers: 100000 } }), createRng(11));
    const outs: VisionOutput[] = [];
    let frameId = 0;
    for (let f = 0; f < 30; f++, frameId++) outs.push(engine.process(input(frameId, deskFrame(f))));
    expect(outs[29].state).toBe(TrackingState.PLANE_FOUND);
    // Test 1: healthy world tracking — no attempt, no relocalizing state.
    for (const o of outs) {
      expect(o.relocalization.attempt).toBe("none");
      expect(o.state).not.toBe(TrackingState.RELOCALIZING);
      expect(["idle", "prepare"]).toContain(o.relocalization.searchStage);
    }
    const mapFrameId = outs[29].mapPose!.mapFrameId;
    const blank = new Uint8Array(W * H).fill(96);
    const gap = resolveConfig().landmarks.reassociateMaxLostFrames + 5;
    const lostFrame = frameId;
    for (let g = 0; g < gap; g++, frameId++) outs.push(engine.process(input(frameId, blank)));
    const lastBlank = outs[outs.length - 1];
    expect(lastBlank.state).toBe(TrackingState.RELOCALIZING);
    // Test 4: attempts started right after the loss (startAfterLostFrames) and
    // the timeline is anchored at the first untracked frame.
    const tl0 = lastBlank.relocalization.timeline!;
    expect(tl0.lostFrame).toBe(lostFrame);
    expect(tl0.lostTimestamp).toBeCloseTo(lostFrame * 33.3, 6);
    expect(tl0.firstAttemptFrame).toBe(lostFrame + resolveConfig().relocalization.startAfterLostFrames);
    expect(tl0.attempts).toBeGreaterThan(0);
    expect(tl0.firstLkSuccessFrame).toBe(-1); // nothing to track on blank frames
    expect(["coarse", "lk"]).toContain(lastBlank.relocalization.searchStage);
    // Test 18: while lost the visibility input says so (v13 decides from these).
    expect(lastBlank.mapPose!.framesSinceTracked).toBeGreaterThan(resolveConfig().state.mapLostFrameTolerance);
    expect(lastBlank.mapPose!.relocalized).toBe(false);

    const stages: string[] = [];
    let applied: VisionOutput | null = null;
    for (let f = 36; f < 60 && !applied; f++, frameId++) {
      const o = engine.process(input(frameId, deskFrame(f)));
      outs.push(o);
      stages.push(o.relocalization.searchStage);
      if (o.relocalization.attempt === "success") applied = o;
    }
    expect(applied, "relocalized").not.toBeNull();
    console.log(`[v14] stages after return: ${stages.join(" ")}`);
    expect(stages).toContain("confirming"); // candidate held (immediate apply disabled)
    expect(stages[stages.length - 1]).toBe("applied");
    const tl = applied!.relocalization.timeline!;
    expect(tl.firstCoarseMatchMs).toBeGreaterThanOrEqual(0);
    expect(tl.firstLkSuccessMs).toBeGreaterThanOrEqual(tl.firstCoarseMatchMs);
    expect(tl.firstPnpSuccessMs).toBeGreaterThanOrEqual(tl.firstLkSuccessMs);
    expect(tl.validationSuccessMs).toBeGreaterThanOrEqual(tl.firstPnpSuccessMs);
    expect(tl.confirmationSuccessMs).toBeGreaterThan(tl.validationSuccessMs); // one confirmation frame later
    expect(tl.confirmationSuccessFrame).toBe(applied!.frameId);
    console.log(
      `[v14] timeline (ms since lost): attempts ${tl.attempts} coarse ${tl.firstCoarseMatchMs.toFixed(0)} lk ${tl.firstLkSuccessMs.toFixed(0)} pnp ${tl.firstPnpSuccessMs.toFixed(0)} val ${tl.validationSuccessMs.toFixed(0)} applied ${tl.confirmationSuccessMs.toFixed(0)}`,
    );
    // Test 19: the apply frame is the v13 CONFIRMING window (relocalized), the same map,
    // and the state is back in world tracking at once (v14 §60): on device the
    // state fell to PLANE_DETECTING until the plane detector re-found the plane,
    // which kept the cube hidden for seconds after a good map pose.
    expect(applied!.mapPose!.relocalized).toBe(true);
    expect(applied!.mapPose!.mapFrameId).toBe(mapFrameId);
    expect(applied!.mapPose!.framesSinceTracked).toBe(0);
    expect(applied!.state).toBe(TrackingState.PLANE_FOUND);
    // The following frames track normally on a fresh map pose (visible under v13 once the monitor window closes).
    let o: VisionOutput = applied!;
    for (let f = 60; f < 66; f++, frameId++) {
      o = engine.process(input(frameId, deskFrame(f)));
      expect(o.state).toBe(TrackingState.PLANE_FOUND);
    }
    expect(o.mapPose!.framesSinceTracked).toBe(0);
    expect(o.mapPose!.relocalized).toBe(false);
    expect(o.relocalization.searchStage === "idle" || o.relocalization.searchStage === "prepare").toBe(true);
    // The diagnostics of the attempt carried the ranking and the LK breakdown (§46).
    const d = applied!.relocalization.diagnostics!;
    expect(d.ranked.length).toBe(d.keyframes);
    expect(d.lkCandidates).toBeLessThanOrEqual(resolveConfig().relocalization.maxLkCandidatesPerFrame);
  });
});
