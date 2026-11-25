import { describe, expect, it } from "vitest";
import {
  advanceTimeline,
  completeTimeline,
  emptyLkStatusCounts,
  emptyRelocalizationTimeline,
  lkFailureReason,
  shouldAttemptRelocalization,
  shouldPrepareRelocalization,
  type LkStatusCounts,
  type RelocalizationAttemptInput,
  type RelocalizationPrepareInput,
  type TimelineAttempt,
} from "../../src/vision/RelocalizationSchedule";

/**
 * Boundary and bookkeeping cases of the relocalization schedule (found by
 * mutation testing): exact thresholds, ties, and the "first success only"
 * rule of the timeline. The scenario tests live in RelocalizationSpeed.test.ts.
 */
const counts = (o: Partial<LkStatusCounts>): LkStatusCounts => ({ ...emptyLkStatusCounts(), ...o });

describe("lkFailureReason: majority, ties and the exact half", () => {
  it("a tie for the largest count names the first reason in the fixed order", () => {
    expect(lkFailureReason(counts({ fbError: 5, highResidual: 5 }))).toBe("high_fb_error");
    expect(lkFailureReason(counts({ diverged: 4, tooFar: 4 }))).toBe("diverged");
  });

  it("exactly half of the rejections is still a majority; one less is not", () => {
    expect(lkFailureReason(counts({ fbError: 5, highResidual: 3, tooFar: 2 }))).toBe("high_fb_error");
    expect(lkFailureReason(counts({ fbError: 5, highResidual: 3, tooFar: 3 }))).toBe("insufficient_tracks");
  });

  it("every rejection status counts towards the total the majority is measured against", () => {
    // 5 of 11: no majority. Dropping diverged or lowTexture from the total would make it one.
    expect(lkFailureReason(counts({ fbError: 5, diverged: 3, lowTexture: 3 }))).toBe("insufficient_tracks");
    expect(lkFailureReason(counts({ fbError: 5, outOfBounds: 3, highResidual: 3 }))).toBe("insufficient_tracks");
    expect(lkFailureReason(counts({ ok: 100 }))).toBe("insufficient_tracks");
  });
});

describe("shouldPrepareRelocalization: the weak-PnP thresholds are exclusive", () => {
  const input = (o: Partial<RelocalizationPrepareInput>): RelocalizationPrepareInput => ({
    worldEstablished: true,
    tracked: true,
    inliers: 40,
    errorPx: 3.5,
    motionLevel: "normal",
    ...o,
  });
  const t = { minInliers: 40, maxErrorPx: 3.5 };

  it("exactly at the limits is not weak; one step past either is", () => {
    expect(shouldPrepareRelocalization(input({}), t)).toBe(false);
    expect(shouldPrepareRelocalization(input({ inliers: 39 }), t)).toBe(true);
    expect(shouldPrepareRelocalization(input({ errorPx: 3.51 }), t)).toBe(true);
  });
});

describe("shouldAttemptRelocalization: located frames and the start delay", () => {
  const input = (o: Partial<RelocalizationAttemptInput>): RelocalizationAttemptInput => ({
    worldEstablished: true,
    framesSinceTracked: 1,
    startAfterLostFrames: 1,
    attemptEveryNFrames: 3,
    pending: false,
    ...o,
  });

  it("a frame the map located (0 frames since tracked) never attempts, even with a pending candidate", () => {
    expect(shouldAttemptRelocalization(input({ framesSinceTracked: 0, pending: true }))).toBe(false);
    expect(shouldAttemptRelocalization(input({ framesSinceTracked: 0, startAfterLostFrames: 0 }))).toBe(false);
  });

  it("before the start delay nothing is attempted, also where the frame count is a multiple of the interval away from it", () => {
    // (1 − 4) % 3 === −0: without the delay test this would attempt.
    expect(shouldAttemptRelocalization(input({ framesSinceTracked: 1, startAfterLostFrames: 4 }))).toBe(false);
    expect(shouldAttemptRelocalization(input({ framesSinceTracked: 4, startAfterLostFrames: 4 }))).toBe(true);
    expect(shouldAttemptRelocalization(input({ framesSinceTracked: 5, startAfterLostFrames: 4 }))).toBe(false);
    expect(shouldAttemptRelocalization(input({ framesSinceTracked: 7, startAfterLostFrames: 4 }))).toBe(true);
  });

  it("a pending candidate attempts every lost frame, before the delay too", () => {
    expect(shouldAttemptRelocalization(input({ framesSinceTracked: 1, startAfterLostFrames: 4, pending: true }))).toBe(true);
  });
});

describe("relocalization timeline", () => {
  const all: TimelineAttempt = { coarsePassed: 1, lkPassed: 1, pnpPassed: 1, validated: 1 };
  const none: TimelineAttempt = { coarsePassed: 0, lkPassed: 0, pnpPassed: 0, validated: 0 };

  it("an empty timeline is −1 everywhere (nothing happened yet) with no attempts", () => {
    expect(emptyRelocalizationTimeline()).toEqual({
      lostFrame: -1,
      lostTimestamp: -1,
      firstAttemptFrame: -1,
      firstAttemptMs: -1,
      firstCoarseMatchFrame: -1,
      firstCoarseMatchMs: -1,
      firstLkSuccessFrame: -1,
      firstLkSuccessMs: -1,
      firstPnpSuccessFrame: -1,
      firstPnpSuccessMs: -1,
      validationSuccessFrame: -1,
      validationSuccessMs: -1,
      confirmationSuccessFrame: -1,
      confirmationSuccessMs: -1,
      attempts: 0,
    });
    expect(emptyRelocalizationTimeline(12, 400)).toMatchObject({ lostFrame: 12, lostTimestamp: 400 });
  });

  it("milliseconds are measured from the loss, clamped at 0, and −1 when the loss time is unknown", () => {
    const t = emptyRelocalizationTimeline(10, 1000);
    advanceTimeline(t, all, 13, 1100);
    expect(t.firstAttemptMs).toBe(100);
    expect(t.validationSuccessMs).toBe(100);
    completeTimeline(t, 14, 1133);
    expect(t.confirmationSuccessMs).toBe(133);

    const early = emptyRelocalizationTimeline(10, 1000);
    advanceTimeline(early, all, 13, 900); // clock went backwards
    completeTimeline(early, 14, 950);
    expect([early.firstAttemptMs, early.confirmationSuccessMs]).toEqual([0, 0]);

    const unknown = emptyRelocalizationTimeline();
    advanceTimeline(unknown, all, 3, 500);
    completeTimeline(unknown, 4, 600);
    expect([unknown.firstAttemptMs, unknown.firstCoarseMatchMs, unknown.confirmationSuccessMs]).toEqual([-1, -1, -1]);
  });

  it("a loss at timestamp 0 is a known loss time", () => {
    const t = emptyRelocalizationTimeline(0, 0);
    advanceTimeline(t, all, 0, 250);
    completeTimeline(t, 0, 300);
    expect([t.firstAttemptMs, t.confirmationSuccessMs]).toEqual([250, 300]);
  });

  it("each stage keeps its first success — also when that was frame 0 — and counts every attempt", () => {
    const t = emptyRelocalizationTimeline(0, 0);
    advanceTimeline(t, all, 0, 10);
    advanceTimeline(t, all, 3, 110);
    completeTimeline(t, 0, 20);
    completeTimeline(t, 5, 170);
    expect(t).toMatchObject({
      firstAttemptFrame: 0,
      firstAttemptMs: 10,
      firstCoarseMatchFrame: 0,
      firstCoarseMatchMs: 10,
      firstLkSuccessFrame: 0,
      firstLkSuccessMs: 10,
      firstPnpSuccessFrame: 0,
      firstPnpSuccessMs: 10,
      validationSuccessFrame: 0,
      validationSuccessMs: 10,
      confirmationSuccessFrame: 0,
      confirmationSuccessMs: 20,
      attempts: 2,
    });
  });

  it("a stage that did not pass is not recorded; it is when it passes later", () => {
    const t = emptyRelocalizationTimeline(10, 1000);
    advanceTimeline(t, none, 11, 1033);
    expect(t).toMatchObject({ firstAttemptFrame: 11, firstCoarseMatchFrame: -1, firstLkSuccessFrame: -1, firstPnpSuccessFrame: -1, validationSuccessFrame: -1 });
    advanceTimeline(t, { ...none, coarsePassed: 2 }, 14, 1133);
    expect(t).toMatchObject({ firstAttemptFrame: 11, firstCoarseMatchFrame: 14, firstCoarseMatchMs: 133, firstLkSuccessFrame: -1 });
  });
});
