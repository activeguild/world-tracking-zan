import { VisionEngine } from "../vision/VisionEngine";
import { wallNow, type ErrorResponse, type ResultResponse, type WorkerRequest, type WorkerResponse } from "./protocol";

/**
 * The vision worker's message handler, without the worker globals: owns a
 * single VisionEngine, receives grayscale frames, returns tracks + quality +
 * state. Every incoming buffer is sent back so the main thread can recycle
 * it. `VisionWorker.ts` binds it to the worker scope; the Quint conformance
 * test (tests/quint) drives it directly.
 */
export function createWorkerHandler(
  post: (msg: WorkerResponse, transfer?: Transferable[]) => void,
): (msg: WorkerRequest) => void {
  let engine: VisionEngine | null = null;

  return (msg: WorkerRequest): void => {
    switch (msg.type) {
      case "init": {
        engine = new VisionEngine(msg.width, msg.height, msg.config);
        post({ type: "ready" });
        return;
      }
      case "reset": {
        // A failing reset is reported like a failing frame (no frame id, no
        // buffer) instead of escaping as the worker's `error` event, which
        // cannot say whether a frame died with it (quint/session.qnt).
        try {
          engine?.reset();
        } catch (e) {
          post({ type: "error", message: e instanceof Error ? e.message : String(e) });
        }
        return;
      }
      case "frame": {
        const receivedAt = wallNow();
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
            anchorPoints: msg.anchorPoints,
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
            planeSearch: out.planeSearch,
            planeAnchor: out.planeAnchor,
            planePose: out.planePose,
            relocalization: out.relocalization,
            motion: out.motion,
            bundleAdjustment: out.bundleAdjustment,
            initRefinement: out.initRefinement,
            worldEstablished: out.worldEstablished,
            planeRecovery: out.planeRecovery,
            landmarks: landmarksBuf,
            landmarkCount: out.landmarkCount,
            tracks: tracksBuf,
            trackCount: out.trackCount,
            processingMs: out.processingMs,
            timing: { ...engine.timing },
            fastThreshold: engine.fastThreshold,
            gray: msg.gray,
            receivedAt,
            postedAt: wallNow(),
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
}
