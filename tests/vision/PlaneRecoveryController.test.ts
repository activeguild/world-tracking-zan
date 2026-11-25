import { describe, expect, it } from "vitest";
import { PlaneRecovery, emptyPlaneRecovery, significantTwoViewMotion, type PlaneRecoveryObservation, type TwoViewMotionThresholds } from "../../src/vision/PlaneRecovery";

/**
 * Exact-threshold and bookkeeping cases of the plane-recovery controller
 * (found by mutation testing). The scenarios are in PlaneRecovery.test.ts.
 */
const obs = (o: Partial<PlaneRecoveryObservation> = {}): PlaneRecoveryObservation => ({
  motionLevel: "fast",
  twoViewMotion: false,
  mapInitialized: true,
  mapTracked: true,
  mapLost: false,
  mapInliers: 42,
  requiredInliers: 12,
  poseFinite: true,
  worldEstablished: false,
  timestamp: 1000,
  ...o,
});
const T: TwoViewMotionThresholds = { parallaxPx: 25, minConfidence: 0.5, minInliers: 20 };
const twoView = (o: Partial<NonNullable<Parameters<typeof significantTwoViewMotion>[0]>> = {}) => ({
  parallaxPx: 45,
  confidence: 1,
  translationConfidence: 1,
  inlierCount: 116,
  referenceFrameId: 7,
  model: "homography" as const,
  ...o,
});

describe("significantTwoViewMotion: thresholds are inclusive, the crossing is strict", () => {
  it("every measure exactly at its threshold still counts", () => {
    expect(significantTwoViewMotion(twoView({ parallaxPx: 25 }), null, T)).toBe(true);
    expect(significantTwoViewMotion(twoView({ confidence: 0.5 }), null, T)).toBe(true);
    expect(significantTwoViewMotion(twoView({ translationConfidence: 0.5 }), null, T)).toBe(true);
    expect(significantTwoViewMotion(twoView({ inlierCount: 20 }), null, T)).toBe(true);
  });

  it("a previous frame of the same reference already at the threshold means no crossing now", () => {
    expect(significantTwoViewMotion(twoView(), { parallaxPx: 25, referenceFrameId: 7 }, T)).toBe(false);
    expect(significantTwoViewMotion(twoView(), { parallaxPx: 24.9, referenceFrameId: 7 }, T)).toBe(true);
    expect(significantTwoViewMotion(twoView(), { parallaxPx: 25, referenceFrameId: 6 }, T)).toBe(true);
  });

  it("no model means no motion event, like a rotation-only model", () => {
    expect(significantTwoViewMotion(twoView({ model: "none" }), null, T)).toBe(false);
    expect(significantTwoViewMotion(twoView({ model: "rotation" }), null, T)).toBe(false);
  });
});

describe("PlaneRecovery bookkeeping", () => {
  it("a map with exactly the required inliers is healthy", () => {
    expect(PlaneRecovery.mapHealthy(obs({ mapInliers: 12 }))).toBe(true);
    expect(PlaneRecovery.mapHealthy(obs({ mapInliers: 11 }))).toBe(false);
  });

  it("while active, only frames with a trigger move the last-motion timestamp", () => {
    const r = new PlaneRecovery();
    expect(r.update(obs({ timestamp: 1000 }))).toBe(true);
    r.update(obs({ motionLevel: "normal", timestamp: 1100 }));
    expect(r.lastMotionTimestamp).toBe(1000);
    r.update(obs({ motionLevel: "normal", twoViewMotion: true, timestamp: 1200 }));
    expect(r.lastMotionTimestamp).toBe(1200);
  });

  it("reset ends the recovery and clears its counters", () => {
    const r = new PlaneRecovery();
    r.update(obs({ timestamp: 1000 }));
    expect([r.active, r.count, r.lastMotionTimestamp]).toEqual([true, 1, 1000]);
    r.reset();
    expect([r.active, r.reason, r.count, r.lastMotionTimestamp]).toEqual([false, null, 0, 0]);
    expect(r.elapsedMs(5000)).toBe(0);
  });

  it("the idle diagnostics report nothing found and nothing held", () => {
    expect(emptyPlaneRecovery()).toEqual({
      active: false,
      reason: null,
      state: "inactive",
      mapHealthy: false,
      mapInliers: 0,
      trackedFeatures: 0,
      seedCandidates: 0,
      seededPoints: 0,
      searchPoints: 0,
      bestInliers: 0,
      requiredInliers: 0,
      searchStage: "points",
      candidateFound: false,
      candidateCommitted: false,
      previousCandidateHeld: false,
      stableFrames: 0,
      requiredStableFrames: 0,
      recoveryElapsedMs: 0,
      recoveries: 0,
    });
  });
});
