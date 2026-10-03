import { type RigidTransform, rotationDistance } from "../math/Pose";

/**
 * Common validation of a camera-pose candidate (修正指示書 v3 §2–§6).
 *
 * Every estimator (landmark PnP, plane-relative PnP, propagation) produces a
 * *candidate*; only a candidate that passes this check against the
 * reference pose (the previous accepted pose, or another candidate) may
 * become the canonical camera pose. "A pose could be computed" is not
 * "this pose may be used".
 */
export type PoseSource = "map" | "plane" | "propagated";

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

/**
 * Continuity check of `candidate` against `reference`.
 * `label` names the candidate in the rejection reason (e.g. "plane").
 */
export function validatePoseCandidate(
  candidate: RigidTransform,
  reference: RigidTransform,
  limits: PoseValidationLimits,
  label = "pose",
): PoseValidationResult {
  const d = poseDelta(candidate, reference);
  if (d.translation > limits.maxTranslation) {
    return {
      accepted: false,
      translationDelta: d.translation,
      rotationDeltaDeg: d.rotationDeg,
      reason: `${label} translation jump ${d.translation.toFixed(3)} > ${limits.maxTranslation.toFixed(3)}`,
    };
  }
  if (d.rotationDeg > limits.maxRotationDeg) {
    return {
      accepted: false,
      translationDelta: d.translation,
      rotationDeltaDeg: d.rotationDeg,
      reason: `${label} rotation jump ${d.rotationDeg.toFixed(1)}° > ${limits.maxRotationDeg.toFixed(1)}°`,
    };
  }
  return { accepted: true, translationDelta: d.translation, rotationDeltaDeg: d.rotationDeg, reason: null };
}
