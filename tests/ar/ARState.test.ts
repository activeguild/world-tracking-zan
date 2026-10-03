import { describe, expect, it } from "vitest";
import { ARError, ARErrorCode, TrackingState, TrackingStateMachine } from "../../src/ar/ARState";
import { DEFAULT_CONFIG, resolveConfig } from "../../src/ar/ARConfig";

const thresholds = { minTrackedForTracking: 40, lostBelow: 20, lostFrameTolerance: 3 };

describe("TrackingStateMachine", () => {
  it("starts in INITIALIZING and moves to SEARCHING_FEATURES on the first frame", () => {
    const sm = new TrackingStateMachine(thresholds);
    expect(sm.state).toBe(TrackingState.INITIALIZING);
    sm.update({ inlierCount: 0, featureCount: 300 });
    expect(sm.state).toBe(TrackingState.SEARCHING_FEATURES);
  });

  it("enters TRACKING once enough inliers are tracked", () => {
    const sm = new TrackingStateMachine(thresholds);
    sm.update({ inlierCount: 0, featureCount: 300 });
    sm.update({ inlierCount: 39, featureCount: 300 });
    expect(sm.state).toBe(TrackingState.SEARCHING_FEATURES);
    sm.update({ inlierCount: 40, featureCount: 300 });
    expect(sm.state).toBe(TrackingState.TRACKING);
  });

  it("tolerates short dropouts before declaring TRACKING_LOST", () => {
    const sm = new TrackingStateMachine(thresholds);
    sm.update({ inlierCount: 0, featureCount: 300 });
    sm.update({ inlierCount: 200, featureCount: 300 });
    sm.update({ inlierCount: 5, featureCount: 300 });
    sm.update({ inlierCount: 5, featureCount: 300 });
    expect(sm.state).toBe(TrackingState.TRACKING);
    sm.update({ inlierCount: 150, featureCount: 300 }); // recovered → counter resets
    sm.update({ inlierCount: 5, featureCount: 300 });
    sm.update({ inlierCount: 5, featureCount: 300 });
    expect(sm.state).toBe(TrackingState.TRACKING);
    sm.update({ inlierCount: 5, featureCount: 300 });
    expect(sm.state).toBe(TrackingState.TRACKING_LOST);
  });

  it("recovers from TRACKING_LOST straight into TRACKING or back to SEARCHING_FEATURES", () => {
    const sm = new TrackingStateMachine(thresholds);
    sm.update({ inlierCount: 0, featureCount: 300 });
    sm.update({ inlierCount: 200, featureCount: 300 });
    for (let i = 0; i < 3; i++) sm.update({ inlierCount: 0, featureCount: 0 });
    expect(sm.state).toBe(TrackingState.TRACKING_LOST);
    sm.update({ inlierCount: 0, featureCount: 300 });
    expect(sm.state).toBe(TrackingState.SEARCHING_FEATURES);
    sm.update({ inlierCount: 120, featureCount: 300 });
    expect(sm.state).toBe(TrackingState.TRACKING);

    for (let i = 0; i < 3; i++) sm.update({ inlierCount: 0, featureCount: 0 });
    expect(sm.state).toBe(TrackingState.TRACKING_LOST);
    sm.update({ inlierCount: 100, featureCount: 300 });
    expect(sm.state).toBe(TrackingState.TRACKING);
  });

  it("reset() returns to INITIALIZING", () => {
    const sm = new TrackingStateMachine(thresholds);
    sm.update({ inlierCount: 0, featureCount: 300 });
    sm.update({ inlierCount: 100, featureCount: 300 });
    sm.reset();
    expect(sm.state).toBe(TrackingState.INITIALIZING);
  });
});

describe("ARError", () => {
  it("carries a code", () => {
    const e = new ARError(ARErrorCode.CAMERA_PERMISSION_DENIED);
    expect(e.code).toBe("CAMERA_PERMISSION_DENIED");
    expect(e).toBeInstanceOf(Error);
  });
});

describe("resolveConfig", () => {
  it("returns defaults when no override is given", () => {
    const c = resolveConfig();
    expect(c).toEqual(DEFAULT_CONFIG);
    expect(c).not.toBe(DEFAULT_CONFIG);
  });

  it("merges partial overrides section by section", () => {
    const c = resolveConfig({ features: { maxFeatures: 123 }, useWorker: false, debug: { log: true } });
    expect(c.features.maxFeatures).toBe(123);
    expect(c.features.minFeatures).toBe(DEFAULT_CONFIG.features.minFeatures);
    expect(c.useWorker).toBe(false);
    expect(c.debug.log).toBe(true);
    expect(c.debug.overlay).toBe(DEFAULT_CONFIG.debug.overlay);
  });
});
