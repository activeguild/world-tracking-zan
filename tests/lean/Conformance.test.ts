import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { TrackingState, TrackingStateMachine } from "../../src/ar/ARState";

/**
 * The tracking state machine against its Lean model
 * (lean/Guardrails/TrackingState.lean).
 *
 * `lake exe conformance` exports the model's full transition table for the
 * shipped thresholds (every state × badFrames 0–2 × inlier counts around both
 * thresholds × the four flags); the TypeScript implementation must make the
 * same move in every row. The theorems proved about the model
 * (relocalizing_needs_world, located_map_leaves_lost, …) then hold for this
 * implementation on that domain. `npm run lean:check` rebuilds the proofs
 * and fails when the checked-in table is stale.
 */
interface Table {
  thresholds: { minTrackedForTracking: number; lostBelow: number; lostFrameTolerance: number };
  columns: string[];
  rows: [string, number, number, number, number, number, number, string, number][];
}

const table = JSON.parse(readFileSync(path.resolve(__dirname, "../../lean/conformance/tracking-state.json"), "utf8")) as Table;

describe("TrackingStateMachine conforms to the Lean model", () => {
  it("covers the whole enumerated domain", () => {
    // 8 states × 3 badFrames × 6 inlier counts × 2⁴ flags
    expect(table.rows.length).toBe(8 * 3 * 6 * 16);
  });

  it("makes the model's transition in every row", () => {
    const mismatches: string[] = [];
    for (const [state, bad, inliers, mi, pf, ml, we, nextState, nextBad] of table.rows) {
      const sm = new TrackingStateMachine({ ...table.thresholds, mapLostFrameTolerance: 3 } as ConstructorParameters<typeof TrackingStateMachine>[0]);
      const internals = sm as unknown as { _state: TrackingState; badFrames: number };
      internals._state = state as TrackingState;
      internals.badFrames = bad;
      const got = sm.update({
        inlierCount: inliers,
        featureCount: inliers,
        mapInitialized: mi === 1,
        planeFound: pf === 1,
        mapLost: ml === 1,
        worldEstablished: we === 1,
      });
      if (got !== nextState || internals.badFrames !== nextBad) {
        mismatches.push(`${state} bad=${bad} inl=${inliers} mi=${mi} pf=${pf} ml=${ml} we=${we}: model ${nextState}/${nextBad}, implementation ${got}/${internals.badFrames}`);
      }
    }
    expect(mismatches.slice(0, 10)).toEqual([]);
  });
});
