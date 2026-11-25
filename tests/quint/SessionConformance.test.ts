import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ARSession } from "../../src/ar/ARSession";
import { VisionEngine } from "../../src/vision/VisionEngine";
import { MainThreadVisionBackend, VisionWorkerClient } from "../../src/worker/VisionWorkerClient";
import { createWorkerHandler } from "../../src/worker/VisionWorkerHost";
import type { WorkerRequest, WorkerResponse } from "../../src/worker/protocol";

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

const UNCAUGHT = Symbol("uncaught worker error");
type Pending = WorkerResponse | typeof UNCAUGHT;

/** The transport of a `Worker`, with delivery in the test's hands. */
class FakeWorker {
  static live: FakeWorker | null = null;
  onmessage: ((ev: { data: WorkerResponse }) => void) | null = null;
  onerror: ((ev: { message: string }) => void) | null = null;
  requests: WorkerRequest[] = [];
  responses: Pending[] = [];
  readonly handle = createWorkerHandler((msg) => this.responses.push(msg));

  constructor() {
    FakeWorker.live = this;
  }
  postMessage(req: WorkerRequest): void {
    this.requests.push(req);
  }
  /** HTML "terminate a worker": both message queues are emptied. */
  terminate(): void {
    this.requests = [];
    this.responses = [];
    if (FakeWorker.live === this) FakeWorker.live = null;
  }
}

interface FakeStream {
  stopped: boolean;
  getTracks(): { stop(): void }[];
}

class Harness {
  readonly session: ARSession;
  private camera: { resolve(s: FakeStream): void; reject(e: unknown): void } | null = null;
  private streams: FakeStream[] = [];
  private applied = 0;
  private now = 0;

  constructor() {
    // A walk may end mid-start, with its worker still up.
    FakeWorker.live = null;
    vi.stubGlobal("navigator", {
      mediaDevices: {
        getUserMedia: () =>
          new Promise<FakeStream>((resolve, reject) => {
            this.camera = { resolve, reject };
          }),
      },
    });
    const video = {
      srcObject: null,
      muted: false,
      autoplay: false,
      readyState: 4,
      videoWidth: 160,
      videoHeight: 120,
      style: {} as Record<string, string>,
      setAttribute: () => undefined,
      play: () => Promise.resolve(),
      getBoundingClientRect: () => ({ width: 160, height: 120, left: 0, top: 0 }),
    };
    this.session = new ARSession({
      video: video as unknown as HTMLVideoElement,
      config: { processing: { width: 160, height: 120, grabber: "canvas2d", maxVisionFps: 0 } },
    });
    this.session.on("started", () => (this.applied = 0));
    this.session.on("frame", () => this.applied++);
  }

  private get internals(): {
    starting: Promise<void> | null;
    stopRequested: boolean;
    backend: VisionWorkerClient | MainThreadVisionBackend | null;
    nextFrameId: number;
    resetBeforeFrameId: number;
    onVideoFrame(now: number): void;
  } {
    return this.session as never;
  }

  /** Everything the awaits in start() chain on. */
  private settle(): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, 0));
  }

  async apply(action: string, pick: string): Promise<void> {
    const worker = FakeWorker.live;
    switch (action) {
      case "start":
        void this.session.start().catch(() => undefined);
        break;
      case "stop":
        this.session.stop();
        break;
      case "reset":
        this.session.reset();
        break;
      case "cameraSettles": {
        const camera = this.camera!;
        this.camera = null;
        if (pick === "ok") {
          const stream: FakeStream = {
            stopped: false,
            getTracks: () => [{ stop: () => (stream.stopped = true) }],
          };
          this.streams.push(stream);
          camera.resolve(stream);
        } else {
          camera.reject({ name: "NotAllowedError" });
        }
        break;
      }
      case "videoFrame":
        this.now += 1000;
        this.internals.onVideoFrame(this.now);
        break;
      case "workerHandles": {
        const req = worker!.requests.shift()!;
        if (pick === "Ok") {
          worker!.handle(req);
        } else if (req.type === "init" || (req.type === "frame" && pick === "Crash")) {
          // An exception outside the handler's try blocks: the worker's `error` event.
          worker!.responses.push(UNCAUGHT);
        } else {
          const method = req.type === "frame" ? "process" : "reset";
          const spy = vi.spyOn(VisionEngine.prototype, method).mockImplementationOnce(() => {
            throw new Error(`injected ${method} failure`);
          });
          worker!.handle(req);
          spy.mockRestore();
        }
        break;
      }
      case "mainHandles": {
        const res = worker!.responses.shift()!;
        if (res === UNCAUGHT) worker!.onerror!({ message: "injected uncaught error" });
        else worker!.onmessage!({ data: res });
        break;
      }
      default:
        throw new Error(`unknown model action ${action}`);
    }
    await this.settle();
  }

  /** The implementation's state, in the model's terms (scripts/quint-traces.mjs `view`). */
  view(): string {
    const s = this.internals;
    const running = this.session.isRunning;
    const starting = s.starting !== null;
    const cameraOn = this.streams.some((x) => !x.stopped);
    const phase = running ? "Running" : !starting ? "Idle" : cameraOn ? "StartingBackend" : "StartingCamera";
    const backend =
      s.backend === null ? "NoBackend" : s.backend instanceof VisionWorkerClient ? "WorkerBackend" : "MainBackend";
    const worker = FakeWorker.live;
    const requests = (worker?.requests ?? []).map((r) =>
      r.type === "init" ? "Init" : r.type === "reset" ? "Reset" : `Frame${r.frameId}`,
    );
    const responses = (worker?.responses ?? []).map((r) =>
      r === UNCAUGHT
        ? "Uncaught"
        : r.type === "ready"
          ? "Ready"
          : r.type === "result"
            ? `Result${r.frameId}`
            : r.frameId !== undefined
              ? `FrameError${r.frameId}`
              : "ResetError",
    );
    return [
      phase,
      starting ? (s.stopRequested ? "stop" : "go") : "-",
      cameraOn ? "cam" : "nocam",
      backend,
      (s.backend?.busy ?? false) ? "busy" : "free",
      running ? `n${s.nextFrameId}` : "-",
      running ? `rb${s.resetBeforeFrameId}` : "-",
      running ? `a${Math.min(this.applied, 3)}` : "-",
      requests.join(","),
      responses.join(","),
    ].join("|");
  }
}

class FakeCanvas {
  constructor(
    public width: number,
    public height: number,
  ) {}
  getContext(kind: string): unknown {
    if (kind !== "2d") return null;
    return {
      drawImage: () => undefined,
      getImageData: (_x: number, _y: number, w: number, h: number) => ({ data: new Uint8ClampedArray(w * h * 4) }),
    };
  }
}

describe("ARSession lifecycle × worker protocol conforms to the Quint model", () => {
  beforeEach(() => {
    FakeWorker.live = null;
    vi.stubGlobal("Worker", FakeWorker);
    vi.stubGlobal("OffscreenCanvas", FakeCanvas);
    vi.stubGlobal("HTMLMediaElement", { HAVE_METADATA: 1, HAVE_CURRENT_DATA: 2 });
    vi.stubGlobal("requestAnimationFrame", () => 1);
    vi.stubGlobal("cancelAnimationFrame", () => undefined);
  });
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
