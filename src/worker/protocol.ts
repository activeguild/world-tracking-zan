import type { ARConfig } from "../ar/ARConfig";
import type { TrackingState } from "../ar/ARState";
import type { CameraIntrinsics } from "../camera/CameraIntrinsics";
import type { TrackingQuality } from "../vision/TrackingQuality";
import type {
  MapPoseOutput,
  PlaneAnchorOutput,
  PlaneOutput,
  PlanePoseOutput,
  PlaneRecoveryDiagnostics,
  PlaneSearchOutput,
  PoseOutput,
  MotionDiagnostics,
  RelocalizationOutput,
} from "../vision/types";

/**
 * Main thread ⇄ Vision Worker message protocol (spec §40, §41).
 *
 * Only TypedArray buffers cross the boundary per frame, always as
 * Transferables: the grayscale frame goes in, and comes back together with
 * the packed track array so the main thread can recycle both.
 */

export interface InitRequest {
  type: "init";
  width: number;
  height: number;
  config: ARConfig;
}

export interface FrameRequest {
  type: "frame";
  frameId: number;
  timestamp: number;
  width: number;
  height: number;
  /** Grayscale pixels (width*height bytes). Transferred. */
  gray: ArrayBuffer;
  intrinsics: CameraIntrinsics;
  /** Gravity direction in the camera frame, if known. */
  gravity: number[] | null;
}

export interface ResetRequest {
  type: "reset";
}

export type WorkerRequest = InitRequest | FrameRequest | ResetRequest;

export interface ReadyResponse {
  type: "ready";
}

export interface EngineTiming {
  pyramid: number;
  track: number;
  ransac: number;
  detect: number;
  pose: number;
  map: number;
  plane: number;
  reloc: number;
  total: number;
}

export interface ResultResponse {
  type: "result";
  frameId: number;
  timestamp: number;
  state: TrackingState;
  quality: TrackingQuality;
  pose: PoseOutput | null;
  mapPose: MapPoseOutput | null;
  plane: PlaneOutput | null;
  planeSearch: PlaneSearchOutput | null;
  planeAnchor: PlaneAnchorOutput | null;
  planePose: PlanePoseOutput | null;
  relocalization: RelocalizationOutput;
  motion: MotionDiagnostics;
  worldEstablished: boolean;
  planeRecovery: PlaneRecoveryDiagnostics;
  /** Float32 packed landmarks (see LANDMARK_STRIDE). Transferred. */
  landmarks: ArrayBuffer;
  landmarkCount: number;
  /** Float32 packed tracks (see TRACK_STRIDE). Transferred. */
  tracks: ArrayBuffer;
  trackCount: number;
  processingMs: number;
  timing: EngineTiming;
  fastThreshold: number;
  /** The input frame buffer, returned for reuse. Transferred. */
  gray: ArrayBuffer;
}

export interface ErrorResponse {
  type: "error";
  message: string;
  frameId?: number;
  /** Returned input buffer when the failure happened on a frame. */
  gray?: ArrayBuffer;
}

export type WorkerResponse = ReadyResponse | ResultResponse | ErrorResponse;
