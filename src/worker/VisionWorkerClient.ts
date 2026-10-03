import type { ARConfig } from "../ar/ARConfig";
import type { TrackingState } from "../ar/ARState";
import type { GrayFrame } from "../camera/CameraFrame";
import type { TrackingQuality } from "../vision/TrackingQuality";
import type { MapPoseOutput, PlaneOutput, PoseOutput } from "../vision/types";
import { VisionEngine } from "../vision/VisionEngine";
import type { EngineTiming, WorkerRequest, WorkerResponse } from "./protocol";

/** Result delivered to the main thread for one processed frame. */
export interface VisionResult {
  frameId: number;
  timestamp: number;
  state: TrackingState;
  quality: TrackingQuality;
  pose: PoseOutput | null;
  mapPose: MapPoseOutput | null;
  plane: PlaneOutput | null;
  landmarks: Float32Array;
  landmarkCount: number;
  tracks: Float32Array;
  trackCount: number;
  processingMs: number;
  timing: EngineTiming;
  fastThreshold: number;
  /** The frame buffer, handed back for recycling. */
  grayBuffer: ArrayBuffer;
}

/**
 * Common interface for the worker-backed and the in-thread vision backends,
 * so ARSession never cares where the pixels are processed.
 */
export interface VisionBackend {
  init(width: number, height: number, config: ARConfig): Promise<void>;
  /** True while a frame is in flight; callers should drop frames meanwhile. */
  readonly busy: boolean;
  processFrame(frame: GrayFrame): void;
  reset(): void;
  dispose(): void;
  onResult: ((result: VisionResult) => void) | null;
  onError: ((message: string, grayBuffer?: ArrayBuffer) => void) | null;
}

/** Vision engine running in a dedicated Web Worker (spec §39–§41). */
export class VisionWorkerClient implements VisionBackend {
  private worker: Worker | null = null;
  private _busy = false;
  onResult: ((result: VisionResult) => void) | null = null;
  onError: ((message: string, grayBuffer?: ArrayBuffer) => void) | null = null;

  get busy(): boolean {
    return this._busy;
  }

  init(width: number, height: number, config: ARConfig): Promise<void> {
    return new Promise((resolve, reject) => {
      let worker: Worker;
      try {
        worker = new Worker(new URL("./VisionWorker.ts", import.meta.url), { type: "module" });
      } catch (e) {
        reject(e);
        return;
      }
      this.worker = worker;
      let ready = false;
      worker.onerror = (ev) => {
        if (!ready) reject(new Error(`VisionWorker failed to start: ${ev.message}`));
        else this.onError?.(ev.message);
      };
      worker.onmessage = (ev: MessageEvent<WorkerResponse>) => {
        const msg = ev.data;
        switch (msg.type) {
          case "ready":
            ready = true;
            resolve();
            return;
          case "result":
            this._busy = false;
            this.onResult?.({
              frameId: msg.frameId,
              timestamp: msg.timestamp,
              state: msg.state,
              quality: msg.quality,
              pose: msg.pose,
              mapPose: msg.mapPose,
              plane: msg.plane,
              landmarks: new Float32Array(msg.landmarks),
              landmarkCount: msg.landmarkCount,
              tracks: new Float32Array(msg.tracks),
              trackCount: msg.trackCount,
              processingMs: msg.processingMs,
              timing: msg.timing,
              fastThreshold: msg.fastThreshold,
              grayBuffer: msg.gray,
            });
            return;
          case "error":
            this._busy = false;
            this.onError?.(msg.message, msg.gray);
            return;
        }
      };
      const req: WorkerRequest = { type: "init", width, height, config };
      worker.postMessage(req);
    });
  }

  processFrame(frame: GrayFrame): void {
    if (!this.worker || this._busy) return;
    this._busy = true;
    const buffer = frame.data.buffer as ArrayBuffer;
    const req: WorkerRequest = {
      type: "frame",
      frameId: frame.frameId,
      timestamp: frame.timestamp,
      width: frame.width,
      height: frame.height,
      gray: buffer,
      intrinsics: frame.intrinsics,
      gravity: frame.gravity ?? null,
    };
    this.worker.postMessage(req, [buffer]);
  }

  reset(): void {
    const req: WorkerRequest = { type: "reset" };
    this.worker?.postMessage(req);
  }

  dispose(): void {
    this.worker?.terminate();
    this.worker = null;
    this._busy = false;
  }
}

/**
 * Fallback backend: same engine, main thread. Used when module workers are
 * unavailable or when `?worker=0` is given for debugging.
 */
export class MainThreadVisionBackend implements VisionBackend {
  private engine: VisionEngine | null = null;
  onResult: ((result: VisionResult) => void) | null = null;
  onError: ((message: string, grayBuffer?: ArrayBuffer) => void) | null = null;

  get busy(): boolean {
    return false;
  }

  async init(width: number, height: number, config: ARConfig): Promise<void> {
    this.engine = new VisionEngine(width, height, config);
  }

  processFrame(frame: GrayFrame): void {
    const engine = this.engine;
    if (!engine) return;
    try {
      const out = engine.process({
        frameId: frame.frameId,
        timestamp: frame.timestamp,
        width: frame.width,
        height: frame.height,
        gray: frame.data,
        intrinsics: frame.intrinsics,
        gravity: frame.gravity ?? null,
      });
      this.onResult?.({
        frameId: out.frameId,
        timestamp: out.timestamp,
        state: out.state,
        quality: out.quality,
        pose: out.pose,
        mapPose: out.mapPose,
        plane: out.plane,
        landmarks: out.landmarks,
        landmarkCount: out.landmarkCount,
        tracks: out.tracks,
        trackCount: out.trackCount,
        processingMs: out.processingMs,
        timing: { ...engine.timing },
        fastThreshold: engine.fastThreshold,
        grayBuffer: frame.data.buffer as ArrayBuffer,
      });
    } catch (e) {
      this.onError?.(e instanceof Error ? e.message : String(e), frame.data.buffer as ArrayBuffer);
    }
  }

  reset(): void {
    this.engine?.reset();
  }

  dispose(): void {
    this.engine = null;
  }
}
