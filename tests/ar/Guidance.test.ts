import { describe, expect, it } from "vitest";
import { TrackingState } from "../../src/ar/ARState";
import { GUIDANCE_TEXT_JA, getGuidance, worldPhase, type GuidanceContext } from "../../src/ar/Guidance";

/**
 * 修正指示書 v10 §13–§17, §37: "go back to where you were" is produced only
 * for an established world that has been lost for the guidance delay;
 * before a world exists the user may scan anywhere.
 */
const DELAY = 2000;
function ctx(over: Partial<GuidanceContext>): GuidanceContext {
  return {
    state: TrackingState.TRACKING,
    worldEstablished: false,
    lowFeature: false,
    planeCandidate: false,
    lostMs: 0,
    relocGuidanceDelayMs: DELAY,
    ...over,
  };
}

describe("getGuidance (v10)", () => {
  it("Test 2 / 5: without an established world, no state produces RELOCALIZE or the return text", () => {
    for (const state of Object.values(TrackingState)) {
      for (const lowFeature of [false, true]) {
        for (const planeCandidate of [false, true]) {
          const key = getGuidance(ctx({ state, lowFeature, planeCandidate, lostMs: 60_000 }));
          expect(key, `${state} low=${lowFeature} cand=${planeCandidate}`).not.toBe("RELOCALIZE");
          expect(GUIDANCE_TEXT_JA[key]).not.toMatch(/先ほど見ていた場所/);
        }
      }
    }
  });

  it("initial / surface scan guidance: flat surface when features are low, plane detecting with a candidate, scan otherwise", () => {
    expect(getGuidance(ctx({ state: TrackingState.INITIALIZING }))).toBe("INITIALIZING");
    expect(getGuidance(ctx({ state: TrackingState.SEARCHING_FEATURES, lowFeature: true }))).toBe("SHOW_FLAT_SURFACE");
    expect(getGuidance(ctx({ state: TrackingState.SEARCHING_FEATURES }))).toBe("SCAN_SURFACE");
    expect(getGuidance(ctx({ state: TrackingState.TRACKING }))).toBe("SCAN_SURFACE");
    expect(getGuidance(ctx({ state: TrackingState.TRACKING, motionTooLow: true }))).toBe("MOVE_SLOWLY");
    expect(getGuidance(ctx({ state: TrackingState.PLANE_DETECTING }))).toBe("MOVE_SLOWLY");
    expect(getGuidance(ctx({ state: TrackingState.PLANE_DETECTING, planeCandidate: true }))).toBe("PLANE_DETECTING");
    // A lost map before the world exists reads as scanning, not as relocalizing.
    expect(getGuidance(ctx({ state: TrackingState.RELOCALIZING, lostMs: 10_000 }))).toBe("SCAN_SURFACE");
    expect(getGuidance(ctx({ state: TrackingState.TRACKING_LOST, lostMs: 10_000 }))).toBe("SCAN_SURFACE");
  });

  it("Test 6: established world, relocalizing, delay not elapsed → generic recovery, not the return text", () => {
    const key = getGuidance(ctx({ state: TrackingState.RELOCALIZING, worldEstablished: true, lostMs: DELAY - 1 }));
    expect(key).toBe("RECOVER");
    expect(GUIDANCE_TEXT_JA[key]).not.toMatch(/先ほど見ていた場所/);
    expect(getGuidance(ctx({ state: TrackingState.TRACKING_LOST, worldEstablished: true, lostMs: 0 }))).toBe("RECOVER");
  });

  it("Test 7: established world, relocalizing, delay elapsed → RELOCALIZE with the return text", () => {
    const key = getGuidance(ctx({ state: TrackingState.RELOCALIZING, worldEstablished: true, lostMs: DELAY }));
    expect(key).toBe("RELOCALIZE");
    expect(GUIDANCE_TEXT_JA[key]).toMatch(/先ほど見ていた場所にカメラを戻してください/);
    expect(getGuidance(ctx({ state: TrackingState.TRACKING_LOST, worldEstablished: true, lostMs: DELAY + 500 }))).toBe("RELOCALIZE");
  });

  it("established world while tracking: tap to place, then nothing", () => {
    expect(getGuidance(ctx({ state: TrackingState.PLANE_FOUND, worldEstablished: true }))).toBe("TAP_TO_PLACE");
    expect(getGuidance(ctx({ state: TrackingState.AR_ACTIVE, worldEstablished: true }))).toBe("NONE");
  });

  // ---- v11: plane recovery after fast motion ----
  it("v11 Test 9: plane recovery without a world → slow down / show a flat surface / detecting; never the return text", () => {
    for (const state of [TrackingState.TRACKING, TrackingState.PLANE_DETECTING]) {
      expect(getGuidance(ctx({ state, planeRecovery: "starting" }))).toBe("SLOW_DOWN");
      expect(GUIDANCE_TEXT_JA[getGuidance(ctx({ state, planeRecovery: "starting" }))]).toBe("スマホをゆっくり動かしてください");
      expect(getGuidance(ctx({ state, planeRecovery: "warmup" }))).toBe("PLANE_WARMUP");
      expect(GUIDANCE_TEXT_JA[getGuidance(ctx({ state, planeRecovery: "warmup" }))]).toBe("平らな場所をゆっくり映してください");
    }
    expect(getGuidance(ctx({ state: TrackingState.PLANE_DETECTING, planeRecovery: "candidate" }))).toBe("PLANE_DETECTING");
    expect(getGuidance(ctx({ state: TrackingState.PLANE_DETECTING, planeRecovery: "stable" }))).toBe("PLANE_DETECTING");
    // Too few features wins over the recovery wording.
    expect(getGuidance(ctx({ state: TrackingState.PLANE_DETECTING, planeRecovery: "warmup", lowFeature: true }))).toBe("SHOW_FLAT_SURFACE");
    // Whatever the recovery phase, no world → no RELOCALIZE and no "go back" text (v11 §46, AC-11).
    for (const state of Object.values(TrackingState)) {
      for (const planeRecovery of ["inactive", "starting", "warmup", "candidate", "stable"] as const) {
        const key = getGuidance(ctx({ state, planeRecovery, lostMs: 60_000 }));
        expect(key, `${state} ${planeRecovery}`).not.toBe("RELOCALIZE");
        expect(GUIDANCE_TEXT_JA[key]).not.toMatch(/先ほど見ていた場所/);
      }
    }
    // "none" / absent behaves exactly as before v11.
    expect(getGuidance(ctx({ state: TrackingState.PLANE_DETECTING, planeRecovery: "inactive" }))).toBe("MOVE_SLOWLY");
    expect(getGuidance(ctx({ state: TrackingState.TRACKING, planeRecovery: "inactive" }))).toBe("SCAN_SURFACE");
  });

  it("v11 Test 10: established world, lost past the delay → RELOCALIZE (unchanged by v11)", () => {
    for (const planeRecovery of ["inactive", "starting", "warmup", "candidate", "stable"] as const) {
      const key = getGuidance(ctx({ state: TrackingState.RELOCALIZING, worldEstablished: true, lostMs: DELAY, planeRecovery }));
      expect(key).toBe("RELOCALIZE");
      expect(GUIDANCE_TEXT_JA[key]).toMatch(/先ほど見ていた場所にカメラを戻してください/);
    }
    expect(getGuidance(ctx({ state: TrackingState.TRACKING_LOST, worldEstablished: true, lostMs: DELAY - 1, planeRecovery: "inactive" }))).toBe("RECOVER");
  });
});

describe("worldPhase (v10 §3, §26)", () => {
  it("maps engine states to the v10 phases, with the world flag deciding lost vs scan", () => {
    const p = (over: Partial<GuidanceContext>) => worldPhase(ctx(over));
    expect(p({ state: TrackingState.INITIALIZING })).toBe("INITIAL_SCAN");
    expect(p({ state: TrackingState.SEARCHING_FEATURES })).toBe("INITIAL_SCAN");
    expect(p({ state: TrackingState.TRACKING })).toBe("SURFACE_SCAN");
    expect(p({ state: TrackingState.PLANE_DETECTING })).toBe("SURFACE_SCAN");
    expect(p({ state: TrackingState.PLANE_DETECTING, planeCandidate: true })).toBe("PLANE_CANDIDATE");
    expect(p({ state: TrackingState.PLANE_FOUND, worldEstablished: true })).toBe("WORLD_TRACKING");
    expect(p({ state: TrackingState.AR_ACTIVE, worldEstablished: true })).toBe("WORLD_TRACKING");
    expect(p({ state: TrackingState.TRACKING_LOST })).toBe("SURFACE_SCAN");
    expect(p({ state: TrackingState.RELOCALIZING })).toBe("SURFACE_SCAN");
    expect(p({ state: TrackingState.TRACKING_LOST, worldEstablished: true, lostMs: 100 })).toBe("WORLD_LOST");
    expect(p({ state: TrackingState.RELOCALIZING, worldEstablished: true, lostMs: 100 })).toBe("WORLD_LOST");
    expect(p({ state: TrackingState.RELOCALIZING, worldEstablished: true, lostMs: DELAY })).toBe("RELOCALIZING");
  });

  it("v11 §28, §55: plane recovery phases while the map is tracked", () => {
    const p = (over: Partial<GuidanceContext>) => worldPhase(ctx(over));
    for (const state of [TrackingState.TRACKING, TrackingState.PLANE_DETECTING]) {
      expect(p({ state, planeRecovery: "starting" })).toBe("PLANE_RECOVERY");
      expect(p({ state, planeRecovery: "warmup" })).toBe("PLANE_WARMUP");
    }
    expect(p({ state: TrackingState.PLANE_DETECTING, planeRecovery: "candidate" })).toBe("PLANE_CANDIDATE");
    expect(p({ state: TrackingState.PLANE_DETECTING, planeRecovery: "stable" })).toBe("PLANE_CANDIDATE");
    expect(p({ state: TrackingState.PLANE_DETECTING, planeRecovery: "inactive" })).toBe("SURFACE_SCAN");
    expect(p({ state: TrackingState.TRACKING, planeRecovery: "candidate" })).toBe("SURFACE_SCAN");
    // Established world: the recovery phase never shows (the map tracking rides out the motion, §30).
    expect(p({ state: TrackingState.PLANE_FOUND, worldEstablished: true, planeRecovery: "inactive" })).toBe("WORLD_TRACKING");
    expect(p({ state: TrackingState.RELOCALIZING, worldEstablished: true, lostMs: DELAY, planeRecovery: "inactive" })).toBe("RELOCALIZING");
  });
});
