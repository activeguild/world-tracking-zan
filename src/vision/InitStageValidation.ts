/**
 * Acceptance of a multi-view init stage and of a keyframe re-pose after it
 * (修正指示書「座標系整合性と AR アンカー安定性」§4, §6). Pure functions over
 * measured numbers; the engine measures, these decide. A lower reprojection
 * error alone never accepts a solve (Android recording 41: a stage that cut
 * the error 1.36 → 0.36 px by rotating the current pose 12° and moving the
 * landmarks 1.8 u was "applied").
 */
export type InitStageRejectReason =
  | "none"
  /** A non-finite value in the solved landmarks, pose or errors. */
  | "invalid"
  /** Fewer init landmarks in the solve than `minLandmarks`. */
  | "few_landmarks"
  /** The current observations occupy fewer than `minSpatialCells` of the 3×3 grid. */
  | "spread"
  /** A solved landmark lies behind the reference or the current camera. */
  | "depth"
  /** The current pose moved more than the limits (rotation / camera-centre shift). */
  | "pose_jump"
  /** More than `maxShiftedFraction` of the landmarks moved beyond `maxShiftDepthRatio` × depth. */
  | "shift"
  /** The error did not drop by at least `minGainFraction`. */
  | "no_gain"
  /** The post-processing (keyframe poses, rebuilt landmarks, correction) produced a non-finite value. */
  | "postprocess";

export interface InitStageMeasures {
  finite: boolean;
  errorBeforePx: number;
  errorAfterPx: number;
  landmarks: number;
  /** Occupied cells of the 3×3 grid by the current observations. */
  spatialCells: number;
  /** Solved landmarks with a non-positive depth in the reference or the current view. */
  behindCamera: number;
  /** Rotation of the current pose by the solve (deg). */
  rotationDeg: number;
  /** Camera-centre move of the current pose by the solve (map units). */
  cameraShift: number;
  /** Median depth of the init landmarks in the reference view before the solve (map units). */
  medianDepth: number;
  /** Fraction of the landmarks that moved by more than `maxShiftDepthRatio` × their depth. */
  shiftedFraction: number;
}

export interface InitStageLimits {
  minGainFraction: number;
  minLandmarks: number;
  minSpatialCells: number;
  maxRotationDeg: number;
  /** Camera-centre move allowed, as a fraction of the median depth. */
  maxShiftDepthRatio: number;
  maxShiftedFraction: number;
}

export interface InitStageVerdict {
  accepted: boolean;
  reason: InitStageRejectReason;
  /** Camera-centre move limit that applied (map units). */
  maxShift: number;
}

/**
 * Decide a stage. Order: invalid → few_landmarks → spread → depth →
 * pose_jump → shift → no_gain, so the reason names the first hard failure.
 * The camera-centre move is compared as a length (not a direction angle: a
 * direction is meaningless when the translation is nearly zero).
 */
export function validateInitStage(m: InitStageMeasures, l: InitStageLimits): InitStageVerdict {
  const maxShift = l.maxShiftDepthRatio * Math.max(m.medianDepth, 1e-9);
  const fail = (reason: InitStageRejectReason): InitStageVerdict => ({ accepted: false, reason, maxShift });
  const nums = [m.errorBeforePx, m.errorAfterPx, m.rotationDeg, m.cameraShift, m.medianDepth, m.shiftedFraction];
  if (!m.finite || nums.some((v) => !Number.isFinite(v))) return fail("invalid");
  if (m.landmarks < l.minLandmarks) return fail("few_landmarks");
  if (m.spatialCells < l.minSpatialCells) return fail("spread");
  if (m.behindCamera > 0) return fail("depth");
  if (m.rotationDeg > l.maxRotationDeg || m.cameraShift > maxShift) return fail("pose_jump");
  if (m.shiftedFraction > l.maxShiftedFraction) return fail("shift");
  if (l.minGainFraction > 0 && !(m.errorAfterPx <= m.errorBeforePx * (1 - l.minGainFraction))) return fail("no_gain");
  if (l.minGainFraction <= 0 && !(m.errorAfterPx < m.errorBeforePx)) return fail("no_gain");
  return { accepted: true, reason: "none", maxShift };
}

/** Why a keyframe was not re-posed by PnP after a stage (it is carried by the rigid correction instead). */
export type KeyframeReposeSkip = "none" | "few_observations" | "few_inliers" | "error" | "spread" | "pose_jump" | "invalid";

export interface KeyframeReposeMeasures {
  /** Observations of *solved* landmarks the PnP ran on. */
  observations: number;
  inliers: number;
  meanErrorPx: number;
  spatialCells: number;
  /** Difference between the PnP pose and the keyframe's old pose carried by the rigid correction (deg / map units). */
  rotationDeg: number;
  shift: number;
  finite: boolean;
}

export interface KeyframeReposeLimits {
  minInliers: number;
  maxMeanErrorPx: number;
  minSpatialCells: number;
  maxRotationDeg: number;
  maxShift: number;
}

/**
 * A keyframe pose from PnP on the solved landmarks only (§4: never mixed
 * with landmarks still in the old frame) replaces the keyframe's pose when
 * it has enough inliers, a small error, spread observations and is close to
 * what the rigid correction predicts; otherwise the keyframe keeps its old
 * pose carried by that correction.
 */
export function validateKeyframeRepose(m: KeyframeReposeMeasures, l: KeyframeReposeLimits): KeyframeReposeSkip {
  if (!m.finite || !Number.isFinite(m.meanErrorPx) || !Number.isFinite(m.rotationDeg) || !Number.isFinite(m.shift)) return "invalid";
  if (m.observations < l.minInliers) return "few_observations";
  if (m.inliers < l.minInliers) return "few_inliers";
  if (m.meanErrorPx > l.maxMeanErrorPx) return "error";
  if (m.spatialCells < l.minSpatialCells) return "spread";
  if (m.rotationDeg > l.maxRotationDeg || m.shift > l.maxShift) return "pose_jump";
  return "none";
}
