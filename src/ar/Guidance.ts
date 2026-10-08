import type { PlaneRecoveryState, PlaneSearchStage } from "../vision/types";
import { TrackingState } from "./ARState";

/**
 * User guidance (修正指示書 v10 §13–§17, v11 §45–§46): decides *what the
 * user should do* from the tracking state and a few quality facts. Pure, no
 * DOM, no engine access, and it never touches engine thresholds.
 *
 * The one rule that matters: "go back to where you were" exists only for an
 * established world that was lost and has been failing to relocalize for a
 * while (`relocGuidanceDelayMs`). Before a world exists the user may scan
 * wherever they like, and a lost map is nothing to return to — including
 * right after a fast motion, when the plane is simply searched again where
 * the camera looks now (v11 §15, §46).
 */
export type GuidanceKey =
  | "INITIALIZING"
  | "SHOW_FLAT_SURFACE" // too few features: point at a textured flat surface
  | "SCAN_SURFACE" // scanning a surface, no map yet
  | "MOVE_SLOWLY" // map exists, needs parallax
  | "SLOW_DOWN" // fast motion with the map still tracked (plane recovery, v11 §45)
  | "PLANE_WARMUP" // plane points re-collected after a fast motion (v11 §45)
  | "MOVE_SIDEWAYS" // plane points form a thin strip (extent test), needs a sideways move (v16)
  | "PLANE_DETECTING" // plane candidate, waiting for stability
  | "TAP_TO_PLACE" // world established, nothing placed
  | "NONE" // AR active
  | "RECOVER" // world lost, generic recovery
  | "RELOCALIZE"; // world lost for a while: return to the previous place

/** v10 / v11 phase names derived from the engine state (for the HUD and the guidance). */
export type WorldPhase =
  | "INITIAL_SCAN"
  | "SURFACE_SCAN"
  | "PLANE_RECOVERY"
  | "PLANE_WARMUP"
  | "PLANE_CANDIDATE"
  | "WORLD_TRACKING"
  | "WORLD_LOST"
  | "RELOCALIZING";

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
  /** Plane recovery state after a significant motion (v11 §28, v11.1 §27); absent / "inactive" when idle. */
  planeRecovery?: PlaneRecoveryState;
  /**
   * Where the plane search stopped this frame and for how long it has been
   * stopping there (ms), with the delay after which a persistent `extent`
   * stop asks for a sideways move (v16). Absent = not evaluated.
   */
  planeSearchStage?: PlaneSearchStage | null;
  planeSearchStageMs?: number;
  sidewaysGuidanceDelayMs?: number;
}

/** Map the engine state to the v10 / v11 phase (v10 §3–§4, §26; v11 §28, §55). */
export function worldPhase(
  ctx: Pick<GuidanceContext, "state" | "worldEstablished" | "planeCandidate" | "lostMs" | "relocGuidanceDelayMs" | "planeRecovery">,
): WorldPhase {
  switch (ctx.state) {
    case TrackingState.INITIALIZING:
    case TrackingState.SEARCHING_FEATURES:
      return "INITIAL_SCAN";
    case TrackingState.TRACKING:
    case TrackingState.PLANE_DETECTING:
      // Surface scan with the map tracked (v11 §28): a recovery after a fast
      // motion is its own phase until a plane candidate exists.
      if (ctx.planeRecovery === "starting") return "PLANE_RECOVERY";
      if (ctx.planeRecovery === "warmup") return "PLANE_WARMUP";
      if (ctx.state === TrackingState.PLANE_DETECTING && (ctx.planeCandidate || ctx.planeRecovery === "candidate" || ctx.planeRecovery === "stable")) {
        return "PLANE_CANDIDATE";
      }
      return "SURFACE_SCAN";
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
    // Plane recovery after a fast motion (v11 §15, §45): slow down, then show
    // a flat surface at the new place. Never "go back" (§46).
    if (ctx.planeRecovery === "starting") return "SLOW_DOWN";
    // v16: the landmarks are there but lie in a strip (the 2D-extent test
    // keeps failing): only a sideways move widens the floor coverage. Takes
    // precedence over the warmup / "move slowly" wording once it persists.
    const candidate = ctx.planeCandidate || ctx.planeRecovery === "candidate" || ctx.planeRecovery === "stable";
    if (
      !candidate &&
      ctx.planeSearchStage === "extent" &&
      ctx.sidewaysGuidanceDelayMs !== undefined &&
      (ctx.planeSearchStageMs ?? 0) >= ctx.sidewaysGuidanceDelayMs
    ) {
      return "MOVE_SIDEWAYS";
    }
    if (ctx.planeRecovery === "warmup") return "PLANE_WARMUP";
    if (ctx.state === TrackingState.PLANE_DETECTING) {
      return ctx.planeCandidate || ctx.planeRecovery === "candidate" || ctx.planeRecovery === "stable" ? "PLANE_DETECTING" : "MOVE_SLOWLY";
    }
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
  SLOW_DOWN: "スマホをゆっくり動かしてください",
  PLANE_WARMUP: "平らな場所をゆっくり映してください",
  MOVE_SIDEWAYS: "スマホを横にゆっくり動かしてください（床の広い範囲を映す）",
  PLANE_DETECTING: "平面を検出しています…",
  TAP_TO_PLACE: "平面をタップしてオブジェクトを置いてください",
  NONE: "",
  RECOVER: "カメラをゆっくり動かしてください",
  RELOCALIZE: "先ほど見ていた場所にカメラを戻してください",
};
