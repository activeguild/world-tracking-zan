import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TrackingState, TrackingStateMachine } from "../../src/ar/ARState";
import type { WorldAnchor } from "../../src/ar/WorldAnchor";
import type { FrameRequest, ResultResponse } from "../../src/worker/protocol";
import { FLOOR, mapFrameIdOf, mapPoseOf, runningSession, stubBrowser } from "./harness";

/**
 * The tracking state machine and the session's world / object visibility
 * against the Quint model of the tracking pipeline (quint/tracking.qnt).
 *
 * A walk is a sequence of frames. For each, the model says what the state
 * machine saw and which state it must be in; what the engine reports (map,
 * frames since the camera was located, relocalized, plane detected, world
 * established); and what the session must have afterwards (a world or not,
 * objects visible or not).
 *
 *   - the real `TrackingStateMachine` is fed the walk's observations
 *   - the real `ARSession` (fake browser, as in the other conformance tests)
 *     handles a real engine result with the walk's fields written in
 *
 * The engine's own relocalization bookkeeping (pending candidate, monitor
 * window, reset rules) is the model's reading of VisionEngine.updateMap; it
 * is not replayed here.
 *
 * Regenerate the walks with `npm run quint:export` after changing the model.
 */

interface TraceFile {
  transitions: number;
  traces: string[][];
}

const file = JSON.parse(
  readFileSync(fileURLToPath(new URL("../../quint/conformance/tracking-traces.json", import.meta.url)), "utf8"),
) as TraceFile;

/** The model's scaled configuration (quint/tracking.qnt). */
const MAP_LOST_TOLERANCE = 2;
const LOST_FRAME_TOLERANCE = 2;
const THRESHOLDS = { minTrackedForTracking: 40, lostBelow: 20, lostFrameTolerance: LOST_FRAME_TOLERANCE };
const INLIERS: Record<string, number> = { L: 10, M: 30, H: 50 };

const STATES: Record<string, TrackingState> = {
  Initializing: TrackingState.INITIALIZING,
  Searching: TrackingState.SEARCHING_FEATURES,
  Tracking: TrackingState.TRACKING,
  PlaneDetecting: TrackingState.PLANE_DETECTING,
  PlaneFound: TrackingState.PLANE_FOUND,
  TrackingLost: TrackingState.TRACKING_LOST,
  Relocalizing: TrackingState.RELOCALIZING,
};

interface Frame {
  inliers: number;
  saw: { mapInitialized: boolean; mapLost: boolean; planeFound: boolean; worldEstablished: boolean };
  state: TrackingState;
  mapId: number | null;
  framesSinceTracked: number;
  relocalized: boolean;
  planeDetected: boolean;
  worldEstablished: boolean;
  world: number | null;
  visible: boolean;
}

/** `inliers|saw|state|map|fst|relocalized|detected|established|world|visible` (scripts/quint-traces.mjs). */
function parse(view: string): Frame {
  const [inl, saw, state, map, fst, reloc, det, west, world, vis] = view.split("|");
  return {
    inliers: INLIERS[inl],
    saw: { mapInitialized: saw[0] === "1", mapLost: saw[1] === "1", planeFound: saw[2] === "1", worldEstablished: saw[3] === "1" },
    state: STATES[state],
    mapId: map === "-" ? null : Number(map),
    framesSinceTracked: Number(fst),
    relocalized: reloc === "1",
    planeDetected: det === "1",
    worldEstablished: west === "1",
    world: world === "-" ? null : Number(world),
    visible: vis === "1",
  };
}

describe("tracking state, world and object visibility conform to the Quint model", () => {
  beforeEach(stubBrowser);
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("has walks to replay", () => {
    expect(file.traces.length).toBeGreaterThan(100);
    expect(file.transitions).toBeGreaterThan(1000);
  });

  it("the state machine takes the model's transitions", () => {
    let steps = 0;
    let drops = 0;
    const reached = new Set<TrackingState>();
    for (let t = 0; t < file.traces.length; t++) {
      const machine = new TrackingStateMachine(THRESHOLDS);
      let hadMap = false;
      for (let i = 0; i < file.traces[t].length; i++) {
        const f = parse(file.traces[t][i]);
        // The engine gave up the map in this frame (before the state update).
        if (hadMap && f.mapId === null) {
          machine.mapDropped();
          drops++;
        }
        hadMap = f.mapId !== null;
        // The model's reading of the engine: the map counts as lost beyond the tolerance.
        expect(f.saw.mapLost, `walk ${t}, frame ${i}`).toBe(f.mapId !== null && f.framesSinceTracked > MAP_LOST_TOLERANCE);
        const got = machine.update({
          inlierCount: f.inliers,
          featureCount: 300,
          mapInitialized: f.saw.mapInitialized,
          mapLost: f.saw.mapLost,
          planeFound: f.saw.planeFound,
          worldEstablished: f.saw.worldEstablished,
        });
        if (got !== f.state) {
          throw new Error(`walk ${t}, frame ${i} (${file.traces[t][i]}): the model is in ${f.state}, TrackingStateMachine in ${got}`);
        }
        reached.add(got);
        steps++;
      }
    }
    expect(steps).toBeGreaterThan(5000);
    // Every engine state is on the walks (AR_ACTIVE is the session's view of PLANE_FOUND).
    expect(reached.size).toBe(6);
    expect(drops).toBeGreaterThan(20);
  });

  it("the session has the model's world and shows the objects exactly when the model does", async () => {
    let steps = 0;
    let shown = 0;
    let worlds = 0;
    // The session handles results one at a time and independently of the
    // engine, so the walks can share sessions: one per batch keeps this fast.
    const BATCH = 40;
    for (let b = 0; b < file.traces.length; b += BATCH) {
      const { h, worker, template } = await runningSession({ state: { mapLostFrameTolerance: MAP_LOST_TOLERANCE } });
      const session = h.session;
      const anchor = (session as unknown as { worldAnchor: WorldAnchor }).worldAnchor;
      // Map ids restart in every walk; keep them apart across the walks of a batch.
      let mapBase = 0;
      for (let t = b; t < Math.min(b + BATCH, file.traces.length); t++) {
        // Between walks: a frame without a map (drops the previous walk's world).
        const frames = ["H|0000|Searching|-|0|0|0|0|-|0", ...file.traces[t]];
        for (let i = 0; i < frames.length; i++) {
          const f = parse(frames[i]);
          await h.apply("videoFrame", "");
          const req = worker.requests.shift() as FrameRequest;
          worker.onmessage!({
            data: {
              ...template,
              frameId: req.frameId,
              gray: req.gray,
              state: f.state,
              worldEstablished: f.worldEstablished,
              mapPose: f.mapId === null ? null : mapPoseOf(mapBase + f.mapId, f.framesSinceTracked, f.relocalized),
              plane: f.planeDetected ? FLOOR : null,
              planeAnchor: null,
            } as ResultResponse,
          });
          const where = `walk ${t}, frame ${i - 1} (${frames[i]})`;
          if (session.isWorldReady !== (f.world !== null)) {
            throw new Error(`${where}: the model ${f.world !== null ? "has" : "has no"} world, the session ${session.isWorldReady ? "has one" : "has none"}`);
          }
          if (f.world !== null) expect(anchor.attachedMapFrameId, where).toBe(mapFrameIdOf(mapBase + f.world));
          const visible = session.getStats().objectVisibility.visible;
          if (visible !== f.visible) {
            throw new Error(`${where}: the model ${f.visible ? "shows" : "hides"} the objects, the session ${visible ? "shows" : "hides"} them (${session.getStats().objectVisibility.reason})`);
          }
          if (f.visible) shown++;
          if (f.world !== null) worlds++;
          steps++;
        }
        mapBase += 10;
      }
      h.session.stop();
    }
    expect(steps).toBeGreaterThan(5000);
    expect(worlds).toBeGreaterThan(500);
    expect(shown).toBeGreaterThan(100);
  }, 600_000);
});
