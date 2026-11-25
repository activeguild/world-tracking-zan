import { vi } from "vitest";
import type { PartialARConfig } from "../../src/ar/ARConfig";
import { ARSession } from "../../src/ar/ARSession";
import type { MapPoseOutput, PlaneOutput } from "../../src/vision/types";
import { VisionEngine } from "../../src/vision/VisionEngine";
import { MainThreadVisionBackend, VisionWorkerClient } from "../../src/worker/VisionWorkerClient";
import { createWorkerHandler } from "../../src/worker/VisionWorkerHost";
import type { WorkerRequest, WorkerResponse } from "../../src/worker/protocol";

/**
 * The browser the Quint conformance tests fake: camera, `Worker` transport
 * (delivery in the test's hands) and canvas. Everything else is the real
 * `ARSession`, `VisionWorkerClient`, worker handler and `VisionEngine`.
 */

export const UNCAUGHT = Symbol("uncaught worker error");
export type Pending = WorkerResponse | typeof UNCAUGHT;

/** The transport of a `Worker`, with delivery in the test's hands. */
export class FakeWorker {
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

export class Harness {
  readonly session: ARSession;
  private camera: { resolve(s: FakeStream): void; reject(e: unknown): void } | null = null;
  private streams: FakeStream[] = [];
  private applied = 0;
  private now = 0;

  constructor(config: PartialARConfig = {}) {
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
      config: { ...config, processing: { width: 160, height: 120, grabber: "canvas2d", maxVisionFps: 0 } },
    });
    this.session.on("started", () => (this.applied = 0));
    this.session.on("frame", () => this.applied++);
  }

  get internals(): {
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

export class FakeCanvas {
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


/** Globals of a browser page; call in `beforeEach`, undo with `vi.unstubAllGlobals()`. */
export function stubBrowser(): void {
  FakeWorker.live = null;
  vi.stubGlobal("Worker", FakeWorker);
  vi.stubGlobal("OffscreenCanvas", FakeCanvas);
  vi.stubGlobal("HTMLMediaElement", { HAVE_METADATA: 1, HAVE_CURRENT_DATA: 2 });
  vi.stubGlobal("requestAnimationFrame", () => 1);
  vi.stubGlobal("cancelAnimationFrame", () => undefined);
}

/** Engine map ids of the walks → `mapFrameId`. */
export const mapFrameIdOf = (modelId: number): number => 100 + modelId;

/** A map pose for a fabricated result: camera at the map origin looking along +Z. */
export function mapPoseOf(modelMapId: number, framesSinceTracked = 0, relocalized = false): MapPoseOutput {
  return {
    rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
    translation: [0, 0, 0],
    inlierCount: 100,
    meanReprojectionErrorPx: 0.5,
    landmarkCount: 0,
    mapFrameId: mapFrameIdOf(modelMapId),
    framesSinceTracked,
    cameraCenter: [0, 0, 0],
    source: framesSinceTracked === 0 ? "map" : "propagated",
    relocalized,
    translationHeld: false,
    sourceDeltaTranslation: 0,
    sourceDeltaRotationDeg: 0,
    mapCandidate: null,
    planeCandidate: null,
  } as unknown as MapPoseOutput;
}

/** A found floor one unit below that camera (map Y is down). */
export const FLOOR = { normal: [0, -1, 0], d: 1, center: [0, 1, 3], found: true, areaEstimate: 4 } as unknown as PlaneOutput;

/** A running session on the worker backend, and one real engine result to build fabricated ones from. */
export async function runningSession(config: PartialARConfig = {}): Promise<{ h: Harness; worker: FakeWorker; template: import("../../src/worker/protocol").ResultResponse; firstFrameId: number }> {
  const h = new Harness(config);
  await h.apply("start", "");
  await h.apply("cameraSettles", "ok");
  await h.apply("workerHandles", "Ok");
  await h.apply("mainHandles", "");
  await h.apply("videoFrame", "");
  await h.apply("workerHandles", "Ok");
  const worker = FakeWorker.live!;
  const template = worker.responses[0] as import("../../src/worker/protocol").ResultResponse;
  if (template.type !== "result") throw new Error("the template frame did not produce a result");
  await h.apply("mainHandles", "");
  return { h, worker, template, firstFrameId: h.internals.nextFrameId };
}
