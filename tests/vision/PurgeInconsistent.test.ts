import { describe, expect, it } from "vitest";
import { resolveConfig } from "../../src/ar/ARConfig";
import { createRng } from "../../src/vision/OutlierRejection";
import type { VisionInput, VisionOutput } from "../../src/vision/types";
import { VisionEngine } from "../../src/vision/VisionEngine";
import { TP_H, TP_K, TP_W, tpGravity, tpPose, tpRender } from "../helpers/twoPlaneScene";

/**
 * v21 (Android recording 31): landmarks a stored keyframe contradicts are
 * removed from the map instead of being skipped per run / per frame.
 *
 * Two sources share one criterion (`bundleAdjustment.maxObservationErrorPx`,
 * 10 px): a keyframe observation the bundle adjustment leaves out as a gross
 * outlier, and a re-association candidate whose keyframe patch lands that
 * far from the landmark's projection. On device both groups were the same
 * population (`+22–32 out` per BA run, `relink ✗ far 7–14@9–15px`): a
 * landmark whose keyframe view and map position disagree by 10 px kept
 * feeding PnP compromises and relink candidates for the rest of the session.
 */
const W = TP_W;
const H = TP_H;
const FRAMES = 150;
function poseAt(f: number) {
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

const OFFSET_PX = 15;

describe("purge of inconsistent landmarks (v21)", () => {
  it("BA: a landmark whose keyframe observation is a gross outlier is removed, its track re-anchors, tracking continues", () => {
    const run = (purgeInconsistent: boolean) => {
      const engine = new VisionEngine(W, H, resolveConfig({ landmarks: { purgeInconsistent } }), createRng(5));
      const outs: VisionOutput[] = [];
      let f = 0;
      for (; f < 100; f++) outs.push(engine.process(input(f)));
      expect(engine.keyframeCount).toBeGreaterThanOrEqual(2);
      const baseline = meanErr(outs, 80, 100);
      // Control run on the clean map: what the solve leaves out by itself.
      expect(engine.runBundleAdjustment(TP_K)).toBe(true);
      const clean = { ...engine.bundleAdjustment };
      // Corrupt the newest keyframe's record of 20 landmarks that still have
      // a live track: the observation now points 15 px beside the corner
      // (a mis-association at keyframe time).
      const kf = engine.keyframes[engine.keyframes.length - 1];
      const corrupted: number[] = [];
      for (const o of kf.observations) {
        const lm = engine.landmarkMap.get(o.landmarkId);
        if (!lm || lm.trackId < 0) continue;
        o.x += OFFSET_PX;
        corrupted.push(o.landmarkId);
        if (corrupted.length === 20) break;
      }
      expect(corrupted.length).toBe(20);
      const sizeBefore = engine.landmarkMap.size;
      const trackIds = engine.tracksForTests().filter((t) => corrupted.includes(t.landmarkId)).map((t) => t.id);
      expect(trackIds.length).toBe(20);
      expect(engine.runBundleAdjustment(TP_K)).toBe(true);
      const ba = { ...engine.bundleAdjustment };
      const remaining = corrupted.filter((id) => engine.landmarkMap.get(id) !== undefined).length;
      const linkedAfter = engine.tracksForTests().filter((t) => corrupted.includes(t.landmarkId));
      const sizeAfter = engine.landmarkMap.size;
      // Tracking goes on; the unlinked tracks take a fresh anchor in the next
      // tracked frame (a new triangulation from current poses, not the old pair).
      let lost = 0;
      let reanchored = -1;
      for (; f < 140; f++) {
        const o = engine.process(input(f));
        outs.push(o);
        if (!o.mapPose || o.mapPose.framesSinceTracked !== 0) lost++;
        // Within the next frames (a v25 init stage in one of them resets the
        // anchors of unlinked tracks again, so the first frame alone may read 0).
        if (f >= 100 && f < 103) reanchored = Math.max(reanchored, engine.tracksForTests().filter((t) => trackIds.includes(t.id) && ((t.landmarkId < 0 && t.anchorFrame === f) || (t.landmarkId >= 0 && !corrupted.includes(t.landmarkId)))).length);
      }
      const after = meanErr(outs, 110, 140);
      return { clean, ba, remaining, linkedAfter, sizeBefore, sizeAfter, reanchored, lost, baseline, after, mapIds: new Set(outs.filter((o) => o.mapPose).map((o) => o.mapPose!.mapFrameId)).size };
    };
    const on = run(true);
    const off = run(false);
    console.log(
      `[purge] BA: clean run +${on.clean.outliers} out / ${on.clean.purged} purged; corrupted run +${on.ba.outliers} out / ${on.ba.purged} purged, map ${on.sizeBefore} → ${on.sizeAfter}, ${on.reanchored}/20 tracks re-anchored, lost ${on.lost}, PnP ${on.baseline.toFixed(2)} → ${on.after.toFixed(2)} px; purge off: +${off.ba.outliers} out / ${off.ba.purged} purged, ${off.remaining} of 20 remain, PnP ${off.after.toFixed(2)} px`,
    );
    // The clean synthetic map has no gross outliers, so nothing is purged by itself.
    expect(on.clean.outliers).toBe(0);
    expect(on.clean.purged).toBe(0);
    // Exactly the 20 contradicted landmarks go, their tracks are unlinked and re-anchor.
    expect(on.ba.outliers).toBe(20);
    expect(on.ba.purged).toBe(20);
    expect(on.remaining).toBe(0);
    expect(on.linkedAfter.length).toBe(0);
    expect(on.sizeAfter).toBe(on.sizeBefore - 20);
    expect(on.mapIds).toBe(1);
    expect(on.lost).toBe(0);
    // The freed tracks re-anchor right away and PnP stays at the clean level.
    expect(on.reanchored).toBeGreaterThanOrEqual(15);
    expect(on.after).toBeLessThan(on.baseline + 0.3);
    // Off: the outliers are only skipped and the landmarks stay in the map.
    expect(off.ba.outliers).toBe(20);
    expect(off.ba.purged).toBe(0);
    expect(off.remaining).toBe(20);
    expect(off.linkedAfter.length).toBe(20);
  });

  it("relink: a landmark whose keyframe patch lands ≥ 10 px from its projection is removed; off, it stays and keeps producing candidates", () => {
    const run = (purgeInconsistent: boolean) => {
      const engine = new VisionEngine(W, H, resolveConfig({ landmarks: { purgeInconsistent } }), createRng(5));
      const outs: VisionOutput[] = [];
      let f = 0;
      for (; f < 100; f++) outs.push(engine.process(input(f)));
      expect(engine.keyframeCount).toBeGreaterThanOrEqual(2);
      const k = TP_K;
      const pose = engine.currentMapPose;
      const r = pose.rotation, t = pose.translation;
      const observed = new Set<number>();
      for (const kf of engine.keyframes) for (const o of kf.observations) observed.add(o.landmarkId);
      // 30 polluted landmarks: the map position is moved so that it projects
      // 12 px beside the corner the keyframes recorded, and a landmark-less
      // track is put right there (a re-detected corner the map explains
      // wrongly). The keyframe patch, tracked from that candidate, lands on
      // the true corner 12 px away.
      const polluted: number[] = [];
      const cfg = resolveConfig();
      for (const tr of engine.tracksForTests()) {
        if (tr.landmarkId < 0 || !observed.has(tr.landmarkId)) continue;
        const lm = engine.landmarkMap.get(tr.landmarkId)!;
        if (lm.observations < cfg.landmarks.minObservationsForPose) continue;
        const p = lm.position;
        const xc = r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + t[0];
        const yc = r[3] * p[0] + r[4] * p[1] + r[5] * p[2] + t[1];
        const zc = r[6] * p[0] + r[7] * p[1] + r[8] * p[2] + t[2];
        const dx = (12 / k.fx) * zc;
        const nx = xc + dx - t[0], ny = yc - t[1], nz = zc - t[2];
        p[0] = r[0] * nx + r[3] * ny + r[6] * nz;
        p[1] = r[1] * nx + r[4] * ny + r[7] * nz;
        p[2] = r[2] * nx + r[5] * ny + r[8] * nz;
        lm.trackId = -1;
        tr.landmarkId = -1;
        tr.anchorFrame = -1;
        tr.anchorPose = null;
        tr.x += 12;
        tr.prevX += 12;
        polluted.push(lm.id);
        if (polluted.length === 30) break;
      }
      expect(polluted.length).toBe(30);
      let candidates = 0, tooFar = 0, purged = 0, lk = 0, verified = 0, farSum = 0;
      let lost = 0;
      for (; f < 130; f++) {
        const o = engine.process(input(f));
        outs.push(o);
        if (!o.mapPose || o.mapPose.framesSinceTracked !== 0) {
          lost++;
          continue;
        }
        const rl = o.mapPose.relink;
        candidates += rl.candidates;
        tooFar += rl.rejectedTooFar;
        farSum += rl.meanFarPx * rl.rejectedTooFar;
        lk += rl.rejectedLk;
        verified += rl.verified;
        purged += rl.purged;
      }
      const remaining = polluted.filter((id) => engine.landmarkMap.get(id) !== undefined).length;
      const linked = engine.tracksForTests().filter((tr) => polluted.includes(tr.landmarkId)).length;
      return { candidates, tooFar, meanFar: tooFar ? farSum / tooFar : 0, lk, verified, purged, purgedTotal: outs[outs.length - 1].mapPose!.relink.purgedTotal, remaining, linked, lost, after: meanErr(outs, 110, 130) };
    };
    const on = run(true);
    const off = run(false);
    console.log(
      `[purge] relink: on — candidates ${on.candidates} far ${on.tooFar}@${on.meanFar.toFixed(1)}px lk ${on.lk} verified ${on.verified} purged ${on.purged} (total ${on.purgedTotal}), ${on.remaining}/30 remain, ${on.linked} linked, lost ${on.lost}, PnP ${on.after.toFixed(2)} px; off — candidates ${off.candidates} far ${off.tooFar}@${off.meanFar.toFixed(1)}px purged ${off.purged}, ${off.remaining}/30 remain, ${off.linked} linked, PnP ${off.after.toFixed(2)} px`,
    );
    // The keyframe patch lands on the true corner, ≈ 12 px from the candidate: too far, and the landmark goes.
    expect(on.tooFar).toBeGreaterThanOrEqual(20);
    expect(on.meanFar).toBeGreaterThan(10);
    expect(on.meanFar).toBeLessThan(14);
    expect(on.purged).toBeGreaterThanOrEqual(20);
    expect(on.purgedTotal).toBe(on.purged);
    expect(on.remaining).toBeLessThanOrEqual(30 - on.purged);
    expect(on.linked).toBe(0);
    expect(on.lost).toBe(0);
    // Off: rejected only — the same landmarks come back as candidates frame after frame.
    expect(off.purged).toBe(0);
    expect(off.remaining).toBe(30);
    expect(off.candidates).toBeGreaterThan(on.candidates * 2);
  });
});
