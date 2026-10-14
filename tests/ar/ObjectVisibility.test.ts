import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { DEFAULT_CONFIG } from "../../src/ar/ARConfig";
import { TrackingState } from "../../src/ar/ARState";
import { decideObjectVisibility, type ObjectVisibilityInput } from "../../src/ar/ObjectVisibility";
import { ARWorld } from "../../src/rendering/ARWorld";

/**
 * 修正指示書 v13: AR objects are shown only while the current camera pose is
 * trustworthy — world tracking on a fresh map pose, outside a relocalization
 * confirmation window. Lost / relocalizing / confirming → hidden, transforms
 * and the world kept.
 */
const TOL = DEFAULT_CONFIG.state.mapLostFrameTolerance;
const base = (o: Partial<ObjectVisibilityInput> = {}): ObjectVisibilityInput => ({
  state: TrackingState.AR_ACTIVE,
  worldReady: true,
  worldEstablished: true,
  framesSinceTracked: 0,
  lostFrameTolerance: TOL,
  relocalized: false,
  poseFinite: true,
  currentlyVisible: true,
  ...o,
});

describe("decideObjectVisibility (v13 §4–§11, Tests 1–9, 13–15)", () => {
  it("Test 1 / 6: world tracking on a fresh map pose, confirmed → visible (TRACKING_ACTIVE)", () => {
    for (const state of [TrackingState.PLANE_FOUND, TrackingState.AR_ACTIVE]) {
      expect(decideObjectVisibility(base({ state, currentlyVisible: false }))).toEqual({ visible: true, reason: "TRACKING_ACTIVE" });
      expect(decideObjectVisibility(base({ state }))).toEqual({ visible: true, reason: "TRACKING_ACTIVE" });
    }
  });

  it("Test 2: tracking lost → hidden (TRACKING_LOST)", () => {
    expect(decideObjectVisibility(base({ state: TrackingState.TRACKING_LOST, framesSinceTracked: 12 }))).toEqual({ visible: false, reason: "TRACKING_LOST" });
    expect(decideObjectVisibility(base({ state: TrackingState.SEARCHING_FEATURES, framesSinceTracked: 40 })).visible).toBe(false);
  });

  it("Test 3 / 7: relocalizing (also after a failed confirmation) → hidden (RELOCALIZING)", () => {
    expect(decideObjectVisibility(base({ state: TrackingState.RELOCALIZING, framesSinceTracked: 60 }))).toEqual({ visible: false, reason: "RELOCALIZING" });
    // A candidate that reached PnP / passed validation but was not applied is still RELOCALIZING.
    expect(decideObjectVisibility(base({ state: TrackingState.RELOCALIZING, framesSinceTracked: 0 }))).toEqual({ visible: false, reason: "RELOCALIZING" });
  });

  it("Test 4 / 5 / 13 / 14 / 15: an applied relocalization (strong or acceptable) is CONFIRMING until its monitor window closes — never a one-frame flash", () => {
    // Apply frame: state back to world tracking, map pose fresh, `relocalized` true.
    const applyFrame = base({ state: TrackingState.PLANE_FOUND, framesSinceTracked: 0, relocalized: true, currentlyVisible: false });
    expect(decideObjectVisibility(applyFrame)).toEqual({ visible: false, reason: "CONFIRMING" });
    // Monitor window frames (postRelocMonitorFrames): still hidden.
    for (let f = 0; f < DEFAULT_CONFIG.relocalization.postRelocMonitorFrames; f++) {
      expect(decideObjectVisibility(base({ state: TrackingState.AR_ACTIVE, relocalized: true, currentlyVisible: false })).visible).toBe(false);
    }
    // Window closed with the map still tracking → visible.
    expect(decideObjectVisibility(base({ state: TrackingState.AR_ACTIVE, relocalized: false, currentlyVisible: false }))).toEqual({ visible: true, reason: "TRACKING_ACTIVE" });
    // The level of the candidate (strong / acceptable) plays no role here: both go through the same window.
  });

  it("Test 9: a PnP dropout the state machine still tolerates keeps the objects shown, but never starts showing them on a held pose (§5, §11)", () => {
    for (let f = 1; f <= TOL; f++) {
      expect(decideObjectVisibility(base({ framesSinceTracked: f, currentlyVisible: true })).visible).toBe(true);
      expect(decideObjectVisibility(base({ framesSinceTracked: f, currentlyVisible: false }))).toEqual({ visible: false, reason: "TRACKING_LOST" });
    }
    expect(decideObjectVisibility(base({ framesSinceTracked: TOL + 1, currentlyVisible: true }))).toEqual({ visible: false, reason: "TRACKING_LOST" });
  });

  it("world not ready / not established, or a non-finite pose → hidden with that reason", () => {
    expect(decideObjectVisibility(base({ worldReady: false }))).toEqual({ visible: false, reason: "WORLD_NOT_READY" });
    expect(decideObjectVisibility(base({ worldEstablished: false }))).toEqual({ visible: false, reason: "WORLD_NOT_READY" });
    expect(decideObjectVisibility(base({ poseFinite: false }))).toEqual({ visible: false, reason: "POSE_INVALID" });
  });

  it("Test 8 / 30: visible → lost → hidden → relocalizing → confirming → visible, repeatedly", () => {
    let visible = false;
    const step = (o: Partial<ObjectVisibilityInput>) => {
      const d = decideObjectVisibility(base({ ...o, currentlyVisible: visible }));
      visible = d.visible;
      return d;
    };
    for (let cycle = 0; cycle < 3; cycle++) {
      expect(step({ state: TrackingState.AR_ACTIVE }).visible).toBe(true);
      expect(step({ state: TrackingState.AR_ACTIVE, framesSinceTracked: 1 }).visible).toBe(true); // tolerated dropout
      expect(step({ state: TrackingState.TRACKING_LOST, framesSinceTracked: 4 }).reason).toBe("TRACKING_LOST");
      expect(step({ state: TrackingState.RELOCALIZING, framesSinceTracked: 30 }).reason).toBe("RELOCALIZING");
      expect(step({ state: TrackingState.PLANE_FOUND, framesSinceTracked: 0, relocalized: true }).reason).toBe("CONFIRMING");
      expect(step({ state: TrackingState.AR_ACTIVE, framesSinceTracked: 0, relocalized: true }).reason).toBe("CONFIRMING");
      expect(step({ state: TrackingState.AR_ACTIVE, framesSinceTracked: 0 })).toEqual({ visible: true, reason: "TRACKING_ACTIVE" });
    }
  });
});

describe("ARWorld object visibility (v13 §12, §18–§20, §24, Tests 10–12)", () => {
  it("hiding keeps the transforms, the world root and the plane grid data; showing restores them; changes are idempotent", () => {
    const scene = new THREE.Scene();
    const world = new ARWorld();
    world.attach(scene, false);
    world.setWorldReady(true);
    world.showPlaneGrid(1, true);
    world.setPlaneGridVisible(true);
    const a = world.createCube(0.1);
    a.place([0.2, 0, -0.5]);
    const b = world.createCube(0.1);
    b.place([-0.1, 0, -0.8]);
    expect(a.root.visible).toBe(true);
    expect(world.isHidden).toBe(false);

    world.setObjectsVisible(false);
    expect(world.isHidden).toBe(true);
    expect(a.root.visible).toBe(false);
    expect(b.root.visible).toBe(false);
    // §12: transforms untouched; §13 / Test 10: the world root (anchor frame) is still there and ready.
    expect(a.position.toArray()).toEqual([0.2, 0, -0.5]);
    expect(b.position.toArray()).toEqual([-0.1, 0, -0.8]);
    expect(world.root.visible).toBe(true);
    expect(scene.children).toContain(world.root);
    expect(world.placedCount).toBe(2);
    // Idempotent: hiding again changes nothing.
    world.setObjectsVisible(false);
    expect(a.root.visible).toBe(false);
    // An object placed while hidden starts hidden too.
    const c = world.createCube(0.1);
    c.place([0, 0, -1]);
    expect(c.root.visible).toBe(false);

    world.setObjectsVisible(true);
    expect(world.isHidden).toBe(false);
    expect(a.root.visible).toBe(true);
    expect(b.root.visible).toBe(true);
    expect(c.root.visible).toBe(true);
    expect(a.position.toArray()).toEqual([0.2, 0, -0.5]);
    // Test 12: the objects' own animation (render loop) keeps running while hidden.
    world.setObjectsVisible(false);
    a.velocity.set(0.1, 0, 0);
    world.update(1);
    expect(a.position.x).toBeCloseTo(0.3, 9);
  });

  it("lostDurationMs follows markTracking, independent of visibility", () => {
    const world = new ARWorld();
    world.markTracking(true, 1000);
    expect(world.lostDurationMs(1500)).toBe(0);
    world.markTracking(false, 2000);
    world.markTracking(false, 2500);
    expect(world.lostDurationMs(2500)).toBe(500);
    world.markTracking(true, 3000);
    expect(world.lostDurationMs(3000)).toBe(0);
  });
});
