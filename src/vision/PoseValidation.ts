import { type RigidTransform, rotationDistance } from "../math/Pose";

/**
 * Common validation of a camera-pose candidate (修正指示書 v3 §2–§6, v4 §2–§9).
 *
 * Every estimator (landmark PnP, plane-relative PnP, propagation) produces a
 * *candidate*; only a candidate that passes this check against the
 * reference pose (the previous accepted pose, or another candidate) may
 * become the canonical camera pose. "A pose could be computed" is not
 * "this pose may be used".
 *
 * Three independent judgements (v4 §8–§9), never substituted for each other:
 *
 *   A. PnP quality        inlier count, reprojection error ("trusted")
 *   B. temporal continuity translation / rotation delta vs the current pose
 *   C. source agreement   map candidate vs plane candidate
 *
 * In particular a *trusted* candidate (good PnP) still has to pass B: a
 * well-supported solve can still be a different pose (planar ambiguity,
 * wrong links), and trusted ≠ continuous.
 */
export type PoseSource = "map" | "plane" | "propagated";

/** Structured rejection codes (v4 §11); the `reason` string is for the HUD. */
export type PoseRejectCode =
  | "translation_jump"
  | "rotation_jump"
  | "insufficient_observations"
  | "insufficient_inliers"
  | "high_reprojection_error"
  | "map_plane_disagreement"
  | "source_cooldown"
  | "invalid_pose";

export interface PoseRejection {
  code: PoseRejectCode;
  /** Human-readable reason with measured value and limit. */
  reason: string;
  /** Measured value that violated the limit (map units or degrees), when applicable. */
  delta: number;
  /** The limit it violated, when applicable. */
  limit: number;
}

export interface PoseCandidate {
  pose: RigidTransform;
  source: PoseSource;
  inlierCount: number;
  /** Mean reprojection error of the inliers (px). */
  reprojectionErrorPx: number;
}

export interface PoseValidationLimits {
  /** Largest allowed camera-center move (map units). */
  maxTranslation: number;
  /** Largest allowed rotation (degrees). */
  maxRotationDeg: number;
}

export interface PoseValidationResult {
  accepted: boolean;
  translationDelta: number;
  rotationDeltaDeg: number;
  /** Why it was rejected (null when accepted). */
  reason: string | null;
  rejection: PoseRejection | null;
}

/** Camera center of a pose: C = −Rᵀ t. */
export function cameraCenterOf(pose: RigidTransform, out = new Float64Array(3)): Float64Array {
  const r = pose.rotation;
  const t = pose.translation;
  out[0] = -(r[0] * t[0] + r[3] * t[1] + r[6] * t[2]);
  out[1] = -(r[1] * t[0] + r[4] * t[1] + r[7] * t[2]);
  out[2] = -(r[2] * t[0] + r[5] * t[1] + r[8] * t[2]);
  return out;
}

/** Translation (map units) and rotation (deg) between two poses. */
export function poseDelta(a: RigidTransform, b: RigidTransform): { translation: number; rotationDeg: number } {
  const ca = cameraCenterOf(a);
  const cb = cameraCenterOf(b);
  return {
    translation: Math.hypot(ca[0] - cb[0], ca[1] - cb[1], ca[2] - cb[2]),
    rotationDeg: (rotationDistance(a.rotation, b.rotation) * 180) / Math.PI,
  };
}

/** True for the codes that mean "the candidate is not continuous with the current pose". */
export function isJumpRejection(r: PoseRejection | null | undefined): boolean {
  return !!r && (r.code === "translation_jump" || r.code === "rotation_jump");
}

/**
 * Continuity check of `candidate` against `reference` (judgement B, or C
 * when `reference` is another candidate). `label` names the candidate in
 * the rejection reason (e.g. "plane"); `disagreement` switches the code to
 * `map_plane_disagreement`.
 */
export function validatePoseCandidate(
  candidate: RigidTransform,
  reference: RigidTransform,
  limits: PoseValidationLimits,
  label = "pose",
  disagreement = false,
): PoseValidationResult {
  const d = poseDelta(candidate, reference);
  const fail = (rejection: PoseRejection): PoseValidationResult => ({
    accepted: false,
    translationDelta: d.translation,
    rotationDeltaDeg: d.rotationDeg,
    reason: rejection.reason,
    rejection,
  });
  if (!Number.isFinite(d.translation) || !Number.isFinite(d.rotationDeg)) {
    return fail({ code: "invalid_pose", reason: `${label} invalid pose`, delta: NaN, limit: NaN });
  }
  if (d.translation > limits.maxTranslation) {
    return fail({
      code: disagreement ? "map_plane_disagreement" : "translation_jump",
      reason: `${label} translation jump ${d.translation.toFixed(3)} > ${limits.maxTranslation.toFixed(3)}`,
      delta: d.translation,
      limit: limits.maxTranslation,
    });
  }
  if (d.rotationDeg > limits.maxRotationDeg) {
    return fail({
      code: disagreement ? "map_plane_disagreement" : "rotation_jump",
      reason: `${label} rotation jump ${d.rotationDeg.toFixed(1)}° > ${limits.maxRotationDeg.toFixed(1)}°`,
      delta: d.rotationDeg,
      limit: limits.maxRotationDeg,
    });
  }
  return { accepted: true, translationDelta: d.translation, rotationDeltaDeg: d.rotationDeg, reason: null, rejection: null };
}
