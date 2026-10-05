import { TrackingState } from "./ARState";

/**
 * AR object visibility (修正指示書 v13): objects follow "is the current
 * camera pose trustworthy", never "the last pose we had".
 *
 * Lost → objects hidden (transform kept, map / world anchor / keyframes
 * kept, relocalization continues) → relocalizing hidden → candidate applied
 * and still being confirmed (post-relocalization monitor) hidden → world
 * tracking with a fresh map pose → visible. The decision is pure so it can
 * be unit-tested; `ARWorld` applies it idempotently.
 *
 * Hysteresis is the engine's own (v13 §5, §9): the state machine keeps
 * PLANE_FOUND / AR_ACTIVE for `state.mapLostFrameTolerance` frames of a
 * failed PnP before declaring the loss, and objects stay visible through
 * exactly those frames. Re-showing after a loss, however, needs a frame whose
 * pose came from the map (§11: a WORLD_TRACKING state with a held pose is
 * not enough).
 */
export type ObjectVisibilityReason =
  | "TRACKING_ACTIVE"
  | "TRACKING_LOST"
  | "RELOCALIZING"
  | "CONFIRMING"
  | "WORLD_NOT_READY"
  | "POSE_INVALID";

export interface ObjectVisibilityInput {
  state: TrackingState;
  /** The world frame exists (WorldAnchor created). */
  worldReady: boolean;
  /** World tracking established for the current map (v10). */
  worldEstablished: boolean;
  /** Frames since the map PnP last located the camera (0 = this frame). */
  framesSinceTracked: number;
  /** Frames the state machine tolerates before declaring the map lost (`state.mapLostFrameTolerance`). */
  lostFrameTolerance: number;
  /** The pose was re-seeded by a relocalization this frame or is still in its monitoring window (v5 §16–§17). */
  relocalized: boolean;
  /** Camera pose of this frame is finite. */
  poseFinite: boolean;
  /** Whether the objects are visible right now (for the asymmetric hysteresis). */
  currentlyVisible: boolean;
}

export interface ObjectVisibilityDecision {
  visible: boolean;
  reason: ObjectVisibilityReason;
}

export function decideObjectVisibility(i: ObjectVisibilityInput): ObjectVisibilityDecision {
  if (!i.worldReady || !i.worldEstablished) return { visible: false, reason: "WORLD_NOT_READY" };
  switch (i.state) {
    case TrackingState.RELOCALIZING:
      return { visible: false, reason: "RELOCALIZING" };
    case TrackingState.TRACKING_LOST:
    case TrackingState.SEARCHING_FEATURES:
    case TrackingState.INITIALIZING:
    case TrackingState.TRACKING:
    case TrackingState.PLANE_DETECTING:
      // Not world tracking (lost, or a lost map before the world was dropped).
      return { visible: false, reason: "TRACKING_LOST" };
    case TrackingState.PLANE_FOUND:
    case TrackingState.AR_ACTIVE:
      break;
  }
  if (!i.poseFinite) return { visible: false, reason: "POSE_INVALID" };
  // A relocalized pose is applied first and confirmed over the following
  // frames; nothing is shown until that window closes (§7–§9, §22, §26).
  if (i.relocalized) return { visible: false, reason: "CONFIRMING" };
  if (i.framesSinceTracked === 0) return { visible: true, reason: "TRACKING_ACTIVE" };
  // A short PnP dropout the state machine still tolerates: keep what is shown
  // (no flicker), but never *start* showing on a held pose.
  if (i.currentlyVisible && i.framesSinceTracked <= i.lostFrameTolerance) return { visible: true, reason: "TRACKING_ACTIVE" };
  return { visible: false, reason: "TRACKING_LOST" };
}
