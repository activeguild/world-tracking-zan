import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { requiredConfirmations } from "../../src/vision/Relocalizer";
import { RelocalizationLedger, type RelocalizationLedgerConfig } from "../../src/vision/RelocalizationLedger";
import { candidateAction } from "../../src/vision/RelocalizationSchedule";

/**
 * The engine's relocalization bookkeeping (`RelocalizationLedger`, used by
 * VisionEngine.updateMap) against the Quint model of the tracking pipeline
 * (quint/tracking.qnt, `mapFrame`).
 *
 * A walk is a sequence of frames: whether the map PnP locates the camera,
 * what a relocalization attempt returns (if one is made) and whether it
 * lands where the pending candidate did; and the bookkeeping the model has
 * after the frame. The real ledger is driven in the engine's call order
 * (begin → settle → afterTracking → shouldResetMap) with the real
 * `requiredConfirmations` / `candidateAction`; the only thing imitated is the
 * tracker's "frames since the camera was located" counter.
 *
 * Regenerate the walks with `npm run quint:export` after changing the model.
 */

interface TraceFile {
  transitions: number;
  traces: string[][];
}

const file = JSON.parse(
  readFileSync(fileURLToPath(new URL("../../quint/conformance/relocalization-traces.json", import.meta.url)), "utf8"),
) as TraceFile;

/** The model's scaled configuration (quint/tracking.qnt). */
const CONFIRMATIONS = 1;
const MIN_ATTEMPTS = 2;
const CONFIG: RelocalizationLedgerConfig = {
  startAfterLostFrames: 1,
  attemptEveryNFrames: 2,
  candidateMaxAgeFrames: 2,
  postRelocMonitorFrames: 2,
  minAttemptsBeforeReset: MIN_ATTEMPTS,
  lostResetFrames: 4,
  preWorldLostResetFrames: 3,
};
const IMMEDIATE = { confirmationFrames: CONFIRMATIONS, immediateInliers: 60, immediateMaxErrorPx: 1 };

/** What a relocalization attempt of the model returns, as a candidate the real rules classify. */
const CANDIDATES: Record<string, { level: "strong" | "acceptable"; inliers: number; errorPx: number } | null> = {
  N: null,
  S: { level: "strong", inliers: 30, errorPx: 1.2 },
  I: { level: "strong", inliers: 80, errorPx: 0.5 },
  // Quality of an immediate candidate, but only inside the relaxed bound: must still be confirmed.
  A: { level: "acceptable", inliers: 80, errorPx: 0.5 },
};

interface Frame {
  located: boolean;
  reloc: string;
  near: boolean;
  map: boolean;
  framesSinceTracked: number;
  pending: { age: number; confirmations: number } | null;
  attempts: number;
  monitorLeft: number;
  relocalized: boolean;
  applied: boolean;
  worldEstablished: boolean;
}

/** `inputs|map|fst|pending|attempts|monitor|relocalized|applied|established` (scripts/quint-traces.mjs). */
function parse(view: string): Frame {
  const [inp, map, fst, pending, attempts, monitor, reloc, applied, west] = view.split("|");
  const [age, confirmations] = pending === "-" ? [0, 0] : pending.split(".").map(Number);
  return {
    located: inp[0] === "1",
    reloc: inp[1],
    near: inp[2] === "1",
    map: map !== "-",
    framesSinceTracked: Number(fst),
    pending: pending === "-" ? null : { age, confirmations },
    attempts: Number(attempts),
    monitorLeft: Number(monitor),
    relocalized: reloc === "1",
    applied: applied === "1",
    worldEstablished: west === "1",
  };
}

describe("the relocalization ledger conforms to the Quint model", () => {
  it("has walks to replay", () => {
    expect(file.traces.length).toBeGreaterThan(100);
    expect(file.transitions).toBeGreaterThan(1000);
  });

  it("every frame of every walk leaves the ledger in the model's state", () => {
    let frames = 0;
    const seen = { applied: 0, confirmed: 0, immediate: 0, moved: 0, lost: 0, resetWithWorld: 0, resetBeforeWorld: 0, monitored: 0, acceptableHeld: 0 };
    for (let t = 0; t < file.traces.length; t++) {
      const ledger = new RelocalizationLedger<string>(CONFIG);
      // Engine state the ledger is fed from, at the start of the frame.
      let map = false;
      let fst = 0;
      let world = false;
      for (let i = 0; i < file.traces[t].length; i++) {
        const f = parse(file.traces[t][i]);
        const where = `walk ${t}, frame ${i} (${file.traces[t][i]})`;
        const frameId = i;
        let applied = false;
        let relocalized = false;
        if (!map) {
          // No map: the engine does not touch the ledger (tryInitialize).
          fst = 0;
          map = f.map;
        } else {
          const start = ledger.begin(frameId, fst, world);
          // The model: a pending candidate never reaches its age limit.
          expect(start.stale, where).toBeNull();
          if (start.attempt) {
            const c = CANDIDATES[f.reloc];
            const hadPending = start.pending !== null;
            const needed = c ? requiredConfirmations(c.level, CONFIRMATIONS) : 0;
            const immediate = c ? candidateAction(c.level, c.inliers, c.errorPx, IMMEDIATE, needed) === "apply" : false;
            const outcome = ledger.settle(frameId, c ? { candidate: f.reloc, requiredConfirmations: needed, immediate, nearPending: f.near } : null);
            applied = outcome === "apply";
            if (applied) {
              seen.applied++;
              if (hadPending) seen.confirmed++;
              else seen.immediate++;
              // An acceptable candidate is never applied unconfirmed.
              if (!hadPending) expect(f.reloc, where).toBe("I");
            }
            // What happened to a pending candidate is told apart from a plain miss.
            if (!c) expect(outcome, where).toBe(hadPending ? "confirmation_lost" : "none");
            expect(outcome === "confirmation_moved", where).toBe(c !== null && hadPending && !f.near);
            if (outcome === "confirmation_moved") seen.moved++;
            if (outcome === "confirmation_lost") seen.lost++;
            if (outcome === "pending" && f.reloc === "A" && !hadPending) seen.acceptableHeld++;
          }
          // MapTracker: applyRelocalization re-seeds the pose, then the PnP runs.
          if (applied) fst = 0;
          fst = f.located ? 0 : fst + 1;
          const after = ledger.afterTracking(f.located);
          relocalized = applied || after.monitored;
          expect(after.watch, where).toBe(after.monitored && !applied);
          if (after.monitored) seen.monitored++;
          if (!f.located && ledger.shouldResetMap(fst, world)) {
            if (world) seen.resetWithWorld++;
            else seen.resetBeforeWorld++;
            ledger.clear();
            map = false;
            fst = 0;
            relocalized = false;
          }
        }

        const got = [
          map ? "map" : "-",
          fst,
          ledger.pending ? `${frameId + 1 - ledger.pending.frameId}.${ledger.pending.confirmations}` : "-",
          Math.min(ledger.failedAttempts, MIN_ATTEMPTS),
          ledger.monitoring ? "mon" : "-",
          relocalized,
          applied,
        ].join("|");
        const want = [
          f.map ? "map" : "-",
          f.framesSinceTracked,
          f.pending ? `${f.pending.age}.${f.pending.confirmations}` : "-",
          f.attempts,
          f.monitorLeft > 0 ? "mon" : "-",
          f.relocalized,
          f.applied,
        ].join("|");
        if (got !== want) throw new Error(`${where}\n  model  ${want}\n  ledger ${got}`);
        world = f.worldEstablished;
        frames++;
      }
    }
    expect(frames).toBeGreaterThan(5000);
    // Every rule of the bookkeeping is on the walks.
    for (const [what, n] of Object.entries(seen)) expect(n, what).toBeGreaterThan(10);
  });
});
