import { describe, expect, it } from "vitest";
import { RelocalizationLedger, type LedgerCandidate, type RelocalizationLedgerConfig } from "../../src/vision/RelocalizationLedger";

/**
 * Rules of the relocalization bookkeeping the Quint model's walks do not
 * reach (tests/quint/RelocalizationLedgerConformance.test.ts covers the
 * rest): more than one confirmation, and a candidate outliving its age limit
 * (in the engine a pending candidate is retried every frame, so that needs a
 * gap in the frame ids).
 */

const CONFIG: RelocalizationLedgerConfig = {
  startAfterLostFrames: 1,
  attemptEveryNFrames: 3,
  candidateMaxAgeFrames: 2,
  postRelocMonitorFrames: 3,
  minAttemptsBeforeReset: 10,
  lostResetFrames: 300,
  preWorldLostResetFrames: 30,
};

const candidate = (name: string, o: Partial<LedgerCandidate<string>> = {}): LedgerCandidate<string> => ({
  candidate: name,
  requiredConfirmations: 1,
  immediate: false,
  nearPending: true,
  ...o,
});

describe("RelocalizationLedger", () => {
  it("a candidate that needs two confirmations is applied on the second one, not the first", () => {
    const ledger = new RelocalizationLedger<string>(CONFIG);
    expect(ledger.begin(10, 1, true)).toEqual({ attempt: true, pending: null, stale: null });
    expect(ledger.settle(10, candidate("a", { requiredConfirmations: 2 }))).toBe("pending");
    ledger.afterTracking(false);

    // Pending: attempted in the very next frame although it is off the schedule.
    const second = ledger.begin(11, 2, true);
    expect(second.attempt).toBe(true);
    expect(second.pending).toEqual({ candidate: "a", frameId: 10, confirmations: 0 });
    expect(ledger.settle(11, candidate("b", { requiredConfirmations: 2 }))).toBe("pending");
    // Still the first candidate, one confirmation closer; its age keeps counting from frame 10.
    expect(ledger.pending).toEqual({ candidate: "a", frameId: 10, confirmations: 1 });
    expect(ledger.monitoring).toBe(false);
    ledger.afterTracking(false);

    expect(ledger.begin(12, 3, true).attempt).toBe(true);
    expect(ledger.settle(12, candidate("c", { requiredConfirmations: 2 }))).toBe("apply");
    expect(ledger.pending).toBeNull();
    expect(ledger.monitoring).toBe(true);
    expect(ledger.failedAttempts).toBe(0);
  });

  it("a pending candidate older than its age limit is dropped as a failed attempt and the search starts over", () => {
    const ledger = new RelocalizationLedger<string>(CONFIG);
    ledger.begin(10, 1, true);
    ledger.settle(10, candidate("a"));
    ledger.afterTracking(false);

    // Exactly at the limit (2 frames old): still pending, still attempted every frame.
    const atLimit = ledger.begin(12, 2, true);
    expect(atLimit.stale).toBeNull();
    expect(atLimit.pending?.candidate).toBe("a");
    expect(atLimit.attempt).toBe(true);
    expect(ledger.failedAttempts).toBe(0);

    // One frame later it is stale: gone, counted, and the attempt follows the schedule again
    // (frames since tracked 3 is off it: start 1, every 3 → 1, 4, 7, …).
    const stale = ledger.begin(13, 3, true);
    expect(stale.stale).toEqual({ candidate: "a", frameId: 10, confirmations: 0 });
    expect(stale.pending).toBeNull();
    expect(stale.attempt).toBe(false);
    expect(ledger.pending).toBeNull();
    expect(ledger.failedAttempts).toBe(1);

    expect(ledger.begin(14, 4, true)).toEqual({ attempt: true, pending: null, stale: null });
  });

  it("the monitor window stays open for the configured frames after the apply frame", () => {
    const ledger = new RelocalizationLedger<string>(CONFIG);
    ledger.begin(10, 1, true);
    expect(ledger.settle(10, candidate("a", { immediate: true }))).toBe("apply");
    // Apply frame: a relocalized pose, nothing to compare it with yet.
    expect(ledger.afterTracking(true)).toEqual({ monitored: true, watch: false });
    for (let f = 0; f < CONFIG.postRelocMonitorFrames; f++) {
      ledger.begin(11 + f, 0, true);
      expect(ledger.monitoring).toBe(true);
      expect(ledger.afterTracking(true)).toEqual({ monitored: true, watch: true });
    }
    expect(ledger.monitoring).toBe(false);
    ledger.begin(14, 0, true);
    expect(ledger.afterTracking(true)).toEqual({ monitored: false, watch: false });
  });

  it("the map is given up after a long loss with enough failed attempts in a world, quickly before one", () => {
    const ledger = new RelocalizationLedger<string>({ ...CONFIG, minAttemptsBeforeReset: 2 });
    expect(ledger.shouldResetMap(30, false)).toBe(false);
    expect(ledger.shouldResetMap(31, false)).toBe(true);
    // In a world the frame count alone is not enough …
    expect(ledger.shouldResetMap(301, true)).toBe(false);
    ledger.begin(1, 1, true);
    ledger.settle(1, null);
    expect(ledger.shouldResetMap(301, true)).toBe(false);
    ledger.begin(4, 4, true);
    ledger.settle(4, null);
    expect(ledger.failedAttempts).toBe(2);
    // … nor are the attempts without the long loss.
    expect(ledger.shouldResetMap(300, true)).toBe(false);
    expect(ledger.shouldResetMap(301, true)).toBe(true);
    // Locating the camera forgets the failures.
    ledger.afterTracking(true);
    expect(ledger.failedAttempts).toBe(0);
    expect(ledger.shouldResetMap(301, true)).toBe(false);
  });

  it("clear() forgets candidate, failures and monitor window", () => {
    const ledger = new RelocalizationLedger<string>(CONFIG);
    ledger.begin(10, 1, true);
    ledger.settle(10, candidate("a", { immediate: true }));
    ledger.afterTracking(false);
    ledger.begin(11, 1, true);
    ledger.settle(11, candidate("b"));
    ledger.begin(12, 2, true);
    ledger.settle(12, null);
    expect(ledger.monitoring).toBe(true);
    expect(ledger.failedAttempts).toBe(1);
    ledger.clear();
    expect(ledger.pending).toBeNull();
    expect(ledger.failedAttempts).toBe(0);
    expect(ledger.monitoring).toBe(false);
  });
});
