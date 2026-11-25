import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Harness, stubBrowser } from "./harness";

/**
 * The TypeScript session / worker client / worker handler against the Quint
 * model (quint/session.qnt).
 *
 * `quint/conformance/session-traces.json` holds walks of the model: the
 * action taken, the nondeterministic choice, and the model state after it.
 * Each walk is replayed on the real `ARSession`, `VisionWorkerClient` and
 * worker message handler (with a real `VisionEngine`); only the browser is
 * faked — camera, `Worker` transport, canvas. After every step the
 * implementation must be in the state the model says.
 *
 * Regenerate the walks with `npm run quint:export` after changing the model.
 */

interface TraceFile {
  transitions: number;
  traces: [action: string, pick: string, view: string][][];
}

const file = JSON.parse(
  readFileSync(fileURLToPath(new URL("../../quint/conformance/session-traces.json", import.meta.url)), "utf8"),
) as TraceFile;

describe("ARSession lifecycle × worker protocol conforms to the Quint model", () => {
  beforeEach(stubBrowser);
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("has walks to replay", () => {
    expect(file.traces.length).toBeGreaterThan(50);
    expect(file.transitions).toBeGreaterThan(300);
  });

  it("every step of every walk leaves the implementation in the model's state", async () => {
    let steps = 0;
    for (let t = 0; t < file.traces.length; t++) {
      const h = new Harness();
      expect(h.view()).toBe("Idle|-|nocam|NoBackend|free|-|-|-||");
      const done: string[] = [];
      for (const [action, pick, expected] of file.traces[t]) {
        await h.apply(action, pick);
        done.push(pick ? `${action}(${pick})` : action);
        const got = h.view();
        if (got !== expected) {
          throw new Error(
            `walk ${t}, step ${done.length}: after ${done.join(" → ")}\n  model          ${expected}\n  implementation ${got}`,
          );
        }
        steps++;
      }
      h.session.stop();
    }
    expect(steps).toBeGreaterThan(1000);
  }, 300_000);
});
