import type { WorldFrame } from "../math/CoordinateSystem";
import { rotationDistance } from "../math/Pose";
import type { PoseRejectCode, PoseSource } from "../vision/PoseValidation";
import type { TrackingQuality } from "../vision/TrackingQuality";
import type { BundleAdjustmentOutput, BundleAdjustmentRejectReason, MapPoseOutput, PlaneOutput, PlaneRecoveryDiagnostics } from "../vision/types";

/**
 * 修正指示書「World Tracking の空間固定性」§8: one record per processed
 * frame that says, for the same frameId, which pose was used and whether it
 * was accepted, whether the world transform (the anchor) changed and why,
 * what the plane did, and what the bundle adjustment did. Pure numbers,
 * computed from values the session already holds (a few vector operations
 * per frame); the HUD formats them only when it is shown and the log line
 * is written only when the world transform changed and logging is on.
 *
 * The world transform may change for exactly three reasons: it is created
 * on the first found plane, it follows an accepted bundle-adjustment
 * correction (the map moved under the camera, v18 / Phase 2), or it is
 * dropped with the map. Any other change is a bug; the anchor-invariance
 * test asserts `worldTransformChanged ⇒ reason ≠ "none"`.
 */
export type WorldTransformChangeReason = "none" | "created" | "ba_correction" | "map_reset";

export interface WorldTrackingDiagnostics {
  frameId: number;
  /** Which estimator produced this frame's camera pose ("none" before a map exists). */
  poseSource: PoseSource | "none";
  /** A fresh map-frame pose was solved and passed the temporal gate in this frame. */
  poseAccepted: boolean;
  /** Rejection code of the chosen candidate when it was not accepted (null otherwise). */
  poseRejectReason: PoseRejectCode | null;
  /** Tracking confidence 0–1 (quality.trackingConfidence). */
  poseConfidence: number;
  /** Mean PnP reprojection error of this frame (px). */
  reprojectionErrorPx: number;
  /** Camera motion since the previous frame: rotation (deg) and camera-centre move (meters once the world exists, map units before). */
  cameraRotationDeltaDeg: number;
  cameraTranslationDelta: number;
  /** World origin move (meters) and world-axes rotation (deg) since the previous frame; 0 when nothing changed. */
  anchorPositionDelta: number;
  anchorRotationDeltaDeg: number;
  worldTransformChanged: boolean;
  worldTransformChangeReason: WorldTransformChangeReason;
  /** Angle between this frame's and the previous frame's plane normal (deg); 0 without two consecutive planes. */
  planeNormalDeltaDeg: number;
  planeConfidence: number;
  /** The per-frame plane passed the PLANE_FOUND criterion. */
  planeAccepted: boolean;
  planeRecoveryActive: boolean;
  baExecuted: boolean;
  baAccepted: boolean;
  baRejectReason: BundleAdjustmentRejectReason;
  baErrorBefore: number;
  baErrorAfter: number;
  baMaxLandmarkShift: number;
  baMaxPoseShift: number;
  baMaxPoseRotationDeg: number;
}

/**
 * Rigid difference between two world frames as the move of the world origin
 * (map units × `scale` → meters) and the rotation between the world axes.
 * Both frames null, or the same object → 0 / 0. One of them null → NaN
 * (created / dropped: there is no "distance" to report).
 */
export function worldFrameDelta(prev: WorldFrame | null, cur: WorldFrame | null): { positionDelta: number; rotationDeltaDeg: number } {
  if (prev === cur) return { positionDelta: 0, rotationDeltaDeg: 0 };
  if (!prev || !cur) return { positionDelta: NaN, rotationDeltaDeg: NaN };
  const a = prev.plane.origin, b = cur.plane.origin;
  const scale = cur.scale > 0 ? cur.scale : 1;
  return {
    positionDelta: Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]) * scale,
    rotationDeltaDeg: (rotationDistance(prev.rotation, cur.rotation) * 180) / Math.PI,
  };
}

/** Angle (deg) between two unit normals; 0 when either is missing. */
export function planeNormalDeltaDeg(prev: ArrayLike<number> | null, cur: ArrayLike<number> | null): number {
  if (!prev || !cur) return 0;
  const dot = prev[0] * cur[0] + prev[1] * cur[1] + prev[2] * cur[2];
  const cx = prev[1] * cur[2] - prev[2] * cur[1];
  const cy = prev[2] * cur[0] - prev[0] * cur[2];
  const cz = prev[0] * cur[1] - prev[1] * cur[0];
  const na = Math.hypot(prev[0], prev[1], prev[2]);
  const nb = Math.hypot(cur[0], cur[1], cur[2]);
  if (na < 1e-12 || nb < 1e-12) return 0;
  // atan2 form: exact near 0 (acos loses the small angles).
  return (Math.atan2(Math.hypot(cx, cy, cz), dot) * 180) / Math.PI;
}

export interface WorldDiagnosticsInput {
  frameId: number;
  quality: TrackingQuality;
  mapPose: MapPoseOutput | null;
  plane: PlaneOutput | null;
  previousPlaneNormal: ArrayLike<number> | null;
  planeRecovery: PlaneRecoveryDiagnostics | null;
  bundleAdjustment: BundleAdjustmentOutput | null;
  previousWorld: WorldFrame | null;
  world: WorldFrame | null;
  /** Why the session changed the world frame in this frame (it knows; the frames alone do not say). */
  changeReason: WorldTransformChangeReason;
}

export function buildWorldDiagnostics(i: WorldDiagnosticsInput): WorldTrackingDiagnostics {
  const m = i.mapPose;
  const scale = i.world?.scale ?? 0;
  const frameDelta = worldFrameDelta(i.previousWorld, i.world);
  const changed = i.previousWorld !== i.world;
  const fresh = !!m && m.framesSinceTracked === 0 && m.source !== "propagated";
  // The candidate that was chosen (or would have been) and why it was not used.
  const reject = m ? (m.planeCandidate?.reject ?? m.mapCandidate?.reject ?? null) : null;
  const ba = i.bundleAdjustment;
  const ran = !!ba && ba.ranThisFrame;
  return {
    frameId: i.frameId,
    poseSource: m ? m.source : "none",
    poseAccepted: fresh && !m!.jumpRejected,
    poseRejectReason: fresh ? null : (reject?.code ?? null),
    poseConfidence: i.quality.trackingConfidence,
    reprojectionErrorPx: m ? m.meanReprojectionErrorPx : 0,
    cameraRotationDeltaDeg: m ? m.deltaRotationDeg : 0,
    cameraTranslationDelta: m ? m.deltaTranslation * (scale > 0 ? scale : 1) : 0,
    anchorPositionDelta: Number.isFinite(frameDelta.positionDelta) ? frameDelta.positionDelta : 0,
    anchorRotationDeltaDeg: Number.isFinite(frameDelta.rotationDeltaDeg) ? frameDelta.rotationDeltaDeg : 0,
    worldTransformChanged: changed,
    worldTransformChangeReason: changed ? i.changeReason : "none",
    planeNormalDeltaDeg: planeNormalDeltaDeg(i.previousPlaneNormal, i.plane?.normal ?? null),
    planeConfidence: i.plane?.confidence ?? 0,
    planeAccepted: !!i.plane?.found,
    planeRecoveryActive: !!i.planeRecovery?.active,
    baExecuted: ran,
    baAccepted: ran && !ba!.rejected,
    baRejectReason: ran ? ba!.rejectReason : "none",
    baErrorBefore: ran ? ba!.errorBeforePx : 0,
    baErrorAfter: ran ? ba!.errorAfterPx : 0,
    baMaxLandmarkShift: ran ? ba!.maxLandmarkShift : 0,
    baMaxPoseShift: ran ? ba!.maxKeyframeShift : 0,
    baMaxPoseRotationDeg: ran ? ba!.maxKeyframeRotationDeg : 0,
  };
}

/** One log line for a frame whose world transform changed (debug logging only). */
export function formatWorldChange(d: WorldTrackingDiagnostics): string {
  const delta = Number.isFinite(d.anchorPositionDelta) && d.worldTransformChangeReason === "ba_correction"
    ? `Δ ${(d.anchorPositionDelta * 100).toFixed(2)} cm / ${d.anchorRotationDeltaDeg.toFixed(2)}°`
    : "";
  return `WORLD TRANSFORM ${d.worldTransformChangeReason} at frame ${d.frameId}${delta ? ` ${delta}` : ""}${d.baExecuted ? `  (BA ${d.baAccepted ? "accepted" : `rejected ${d.baRejectReason}`} ${d.baErrorBefore.toFixed(2)} → ${d.baErrorAfter.toFixed(2)} px, kf ${d.baMaxPoseShift.toFixed(3)} u / ${d.baMaxPoseRotationDeg.toFixed(2)}°)` : ""}`;
}
