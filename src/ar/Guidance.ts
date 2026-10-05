import { TrackingState } from "./ARState";

/**
 * User guidance (修正指示書 v10 §13–§17): decides *what the user should do*
 * from the tracking state and a few quality facts. Pure, no DOM, no engine
 * access, and it never touches engine thresholds.
 *
 * The one rule that matters: "go back to where you were" exists only for an
 * established world that was lost and has been failing to relocalize for a
 * while (`relocGuidanceDelayMs`). Before a world exists the user may scan
 * wherever they like, and a lost map is nothing to return to.
 */
export type GuidanceKey =
  | "INITIALIZING"
  | "SHOW_FLAT_SURFACE" // too few features: point at a textured flat surface
  | "SCAN_SURFACE" // scanning a surface, no map yet
  | "MOVE_SLOWLY" // map exists, needs parallax
  | "PLANE_DETECTING" // plane candidate, waiting for stability
  | "TAP_TO_PLACE" // world established, nothing placed
  | "NONE" // AR active
  | "RECOVER" // world lost, generic recovery
  | "RELOCALIZE"; // world lost for a while: return to the previous place

/** v10 phase names derived from the engine state (for the HUD and the guidance). */
export type WorldPhase = "INITIAL_SCAN" | "SURFACE_SCAN" | "PLANE_CANDIDATE" | "WORLD_TRACKING" | "WORLD_LOST" | "RELOCALIZING";

export interface GuidanceContext {
  state: TrackingState;
  /** A plane was found and the world anchored to it for the current map. */
  worldEstablished: boolean;
  /** Too few features in view. */
  lowFeature: boolean;
  /** A plane candidate exists (not yet stable). */
  planeCandidate: boolean;
  /** How long the established world has been lost (ms); 0 while tracking. */
  lostMs: number;
  /** Generic recovery guidance for this long before asking to return (ms). */
  relocGuidanceDelayMs: number;
  /** Camera barely moving while a map is needed (optional, v10 §13). */
  motionTooLow?: boolean;
}

/** Map the engine state to the v10 phase (v10 §3–§4, §26). */
export function worldPhase(ctx: Pick<GuidanceContext, "state" | "worldEstablished" | "planeCandidate" | "lostMs" | "relocGuidanceDelayMs">): WorldPhase {
  switch (ctx.state) {
    case TrackingState.INITIALIZING:
    case TrackingState.SEARCHING_FEATURES:
      return "INITIAL_SCAN";
    case TrackingState.TRACKING:
      return "SURFACE_SCAN";
    case TrackingState.PLANE_DETECTING:
      return ctx.planeCandidate ? "PLANE_CANDIDATE" : "SURFACE_SCAN";
    case TrackingState.PLANE_FOUND:
    case TrackingState.AR_ACTIVE:
      return "WORLD_TRACKING";
    case TrackingState.TRACKING_LOST:
    case TrackingState.RELOCALIZING:
      if (!ctx.worldEstablished) return "SURFACE_SCAN";
      return ctx.lostMs >= ctx.relocGuidanceDelayMs ? "RELOCALIZING" : "WORLD_LOST";
  }
}

export function getGuidance(ctx: GuidanceContext): GuidanceKey {
  if (ctx.state === TrackingState.INITIALIZING) return "INITIALIZING";
  if (!ctx.worldEstablished) {
    // Initial / surface scan (v10 §16): wherever the camera looks now is the
    // target. Never RELOCALIZE here, whatever the engine state says.
    if (ctx.lowFeature) return "SHOW_FLAT_SURFACE";
    if (ctx.state === TrackingState.PLANE_DETECTING) return ctx.planeCandidate ? "PLANE_DETECTING" : "MOVE_SLOWLY";
    if (ctx.motionTooLow) return "MOVE_SLOWLY";
    return "SCAN_SURFACE";
  }
  switch (ctx.state) {
    case TrackingState.AR_ACTIVE:
      return "NONE";
    case TrackingState.PLANE_FOUND:
      return "TAP_TO_PLACE";
    case TrackingState.TRACKING_LOST:
    case TrackingState.RELOCALIZING:
      return ctx.lostMs >= ctx.relocGuidanceDelayMs ? "RELOCALIZE" : "RECOVER";
    default:
      // Established world, camera tracking but the plane not re-detected
      // (should not persist): generic recovery wording, never "go back".
      return "RECOVER";
  }
}

/** Japanese wording for the demo (v10 §13, §46–§47). */
export const GUIDANCE_TEXT_JA: Record<GuidanceKey, string> = {
  INITIALIZING: "Initializing…",
  SHOW_FLAT_SURFACE: "模様のある平らな場所（机・床）にカメラを向けてください",
  SCAN_SURFACE: "平らな場所をゆっくりスキャンしてください（スマホをゆっくり左右に）",
  MOVE_SLOWLY: "スマホをゆっくり左右に動かしてください",
  PLANE_DETECTING: "平面を検出しています…",
  TAP_TO_PLACE: "平面をタップしてオブジェクトを置いてください",
  NONE: "",
  RECOVER: "カメラをゆっくり動かしてください",
  RELOCALIZE: "先ほど見ていた場所にカメラを戻してください",
};
