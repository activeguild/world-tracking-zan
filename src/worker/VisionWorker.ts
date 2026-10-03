/// <reference lib="webworker" />
import { VisionEngine } from "../vision/VisionEngine";
import type { ErrorResponse, ResultResponse, WorkerRequest, WorkerResponse } from "./protocol";

/**
 * Web Worker entry point (spec §39).
 *
 * Owns a single VisionEngine. Receives grayscale frames, returns tracks +
 * quality + state. Every incoming buffer is sent back so the main thread can
 * recycle it.
 */
const ctx = self as unknown as DedicatedWorkerGlobalScope;

let engine: VisionEngine | null = null;

function post(msg: WorkerResponse, transfer: Transferable[] = []): void {
  ctx.postMessage(msg, transfer);
}

ctx.onmessage = (ev: MessageEvent<WorkerRequest>) => {
  const msg = ev.data;
  switch (msg.type) {
    case "init": {
      engine = new VisionEngine(msg.width, msg.height, msg.config);
      post({ type: "ready" });
      return;
    }
    case "reset": {
      engine?.reset();
      return;
    }
    case "frame": {
      if (!engine) {
        const err: ErrorResponse = {
          type: "error",
          message: "frame received before init",
          frameId: msg.frameId,
          gray: msg.gray,
        };
        post(err, [msg.gray]);
        return;
      }
      try {
        const out = engine.process({
          frameId: msg.frameId,
          timestamp: msg.timestamp,
          width: msg.width,
          height: msg.height,
          gray: new Uint8Array(msg.gray),
          intrinsics: msg.intrinsics,
          gravity: msg.gravity,
        });
        const tracksBuf = out.tracks.buffer as ArrayBuffer;
        const landmarksBuf = out.landmarks.buffer as ArrayBuffer;
        const res: ResultResponse = {
          type: "result",
          frameId: out.frameId,
          timestamp: out.timestamp,
          state: out.state,
          quality: out.quality,
          pose: out.pose,
          mapPose: out.mapPose,
          plane: out.plane,
          relocalization: out.relocalization,
          landmarks: landmarksBuf,
          landmarkCount: out.landmarkCount,
          tracks: tracksBuf,
          trackCount: out.trackCount,
          processingMs: out.processingMs,
          timing: { ...engine.timing },
          fastThreshold: engine.fastThreshold,
          gray: msg.gray,
        };
        post(res, landmarksBuf === tracksBuf ? [tracksBuf, msg.gray] : [tracksBuf, landmarksBuf, msg.gray]);
      } catch (e) {
        const err: ErrorResponse = {
          type: "error",
          message: e instanceof Error ? e.message : String(e),
          frameId: msg.frameId,
          gray: msg.gray,
        };
        post(err, [msg.gray]);
      }
      return;
    }
  }
};
