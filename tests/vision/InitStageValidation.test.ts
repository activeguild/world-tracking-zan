import { describe, expect, it } from "vitest";
import { validateInitStage, validateKeyframeRepose, type InitStageLimits, type InitStageMeasures, type KeyframeReposeLimits, type KeyframeReposeMeasures } from "../../src/vision/InitStageValidation";

/**
 * 修正指示書「座標系整合性と AR アンカー安定性」§6 / §9.5: a lower
 * reprojection error alone never accepts an init stage; the pose change,
 * the landmark change, the spread, the depths and finiteness are judged too.
 */
const limits: InitStageLimits = { minGainFraction: 0.15, minLandmarks: 30, minSpatialCells: 4, maxRotationDeg: 3, maxShiftDepthRatio: 0.05, maxShiftedFraction: 0.1 };
const good: InitStageMeasures = {
  finite: true,
  errorBeforePx: 1.0,
  errorAfterPx: 0.4,
  landmarks: 120,
  spatialCells: 7,
  behindCamera: 0,
  rotationDeg: 0.3,
  cameraShift: 0.1,
  medianDepth: 15,
  shiftedFraction: 0.0,
};

describe("validateInitStage (指示書 §6)", () => {
  it("accepts a finite, well-spread solve that lowers the error with a small pose change", () => {
    const v = validateInitStage(good, limits);
    expect(v.accepted).toBe(true);
    expect(v.reason).toBe("none");
    expect(v.maxShift).toBeCloseTo(0.75, 9);
  });

  it("rotation only: a rotation beyond the limit is a pose jump even when the error improves a lot (recording 41: 12° for 1.36 → 0.36 px)", () => {
    expect(validateInitStage({ ...good, errorBeforePx: 1.36, errorAfterPx: 0.36, rotationDeg: 12.3, cameraShift: 0 }, limits).reason).toBe("pose_jump");
    expect(validateInitStage({ ...good, rotationDeg: 2.9, cameraShift: 0 }, limits).accepted).toBe(true);
  });

  it("translation only: a camera-centre move beyond 5 % of the median depth is a pose jump; the direction angle is never evaluated", () => {
    // 0.9 u on a 15 u scene = 6 % > 5 %.
    expect(validateInitStage({ ...good, rotationDeg: 0, cameraShift: 0.9 }, limits).reason).toBe("pose_jump");
    // 0.7 u = 4.7 %: fine, even though for a tiny |C| the same move would be a huge "direction" change.
    expect(validateInitStage({ ...good, rotationDeg: 0, cameraShift: 0.7 }, limits).accepted).toBe(true);
  });

  it("rotation and translation together: either one over its limit rejects", () => {
    expect(validateInitStage({ ...good, rotationDeg: 3.5, cameraShift: 0.2 }, limits).reason).toBe("pose_jump");
    expect(validateInitStage({ ...good, rotationDeg: 1.5, cameraShift: 0.8 }, limits).reason).toBe("pose_jump");
    expect(validateInitStage({ ...good, rotationDeg: 1.5, cameraShift: 0.5 }, limits).accepted).toBe(true);
  });

  it("biased observations (one corner of the image) are rejected as spread", () => {
    expect(validateInitStage({ ...good, spatialCells: 2 }, limits).reason).toBe("spread");
    expect(validateInitStage({ ...good, spatialCells: 4 }, limits).accepted).toBe(true);
  });

  it("too few landmarks are rejected before anything else is judged", () => {
    expect(validateInitStage({ ...good, landmarks: 29, rotationDeg: 10 }, limits).reason).toBe("few_landmarks");
  });

  it("a landmark behind either camera rejects the whole solve", () => {
    expect(validateInitStage({ ...good, behindCamera: 1 }, limits).reason).toBe("depth");
  });

  it("many landmarks moved far reject the solve (shift), as in the keyframe BA", () => {
    expect(validateInitStage({ ...good, shiftedFraction: 0.12 }, limits).reason).toBe("shift");
    expect(validateInitStage({ ...good, shiftedFraction: 0.1 }, limits).accepted).toBe(true);
  });

  it("an error that improves but not by the gain fraction is no_gain; with the fraction 0 any decrease counts", () => {
    expect(validateInitStage({ ...good, errorBeforePx: 1.0, errorAfterPx: 0.9 }, limits).reason).toBe("no_gain");
    expect(validateInitStage({ ...good, errorBeforePx: 1.0, errorAfterPx: 0.85 }, limits).accepted).toBe(true);
    expect(validateInitStage({ ...good, errorBeforePx: 1.0, errorAfterPx: 0.99 }, { ...limits, minGainFraction: 0 }).accepted).toBe(true);
    expect(validateInitStage({ ...good, errorBeforePx: 1.0, errorAfterPx: 1.0 }, { ...limits, minGainFraction: 0 }).reason).toBe("no_gain");
  });

  it("NaN / Infinity anywhere is invalid, whatever the other numbers say", () => {
    expect(validateInitStage({ ...good, finite: false }, limits).reason).toBe("invalid");
    expect(validateInitStage({ ...good, errorAfterPx: NaN }, limits).reason).toBe("invalid");
    expect(validateInitStage({ ...good, cameraShift: Infinity }, limits).reason).toBe("invalid");
    expect(validateInitStage({ ...good, medianDepth: NaN }, limits).reason).toBe("invalid");
  });

  it("small parallax / an unconstrained solve shows up as a pose jump or a shift, never as an acceptance by error alone", () => {
    // A degenerate two-view problem slides along the valley: large landmark
    // moves and a rotated pose for a tiny error gain.
    expect(validateInitStage({ ...good, errorBeforePx: 0.11, errorAfterPx: 0.1, rotationDeg: 4.7, shiftedFraction: 0.3 }, limits).reason).toBe("pose_jump");
    expect(validateInitStage({ ...good, errorBeforePx: 0.11, errorAfterPx: 0.1, rotationDeg: 0.5, shiftedFraction: 0.3 }, limits).reason).toBe("shift");
  });
});

describe("validateKeyframeRepose (指示書 §4)", () => {
  const l: KeyframeReposeLimits = { minInliers: 25, maxMeanErrorPx: 3.0, minSpatialCells: 4, maxRotationDeg: 3, maxShift: 0.75 };
  const ok: KeyframeReposeMeasures = { observations: 80, inliers: 70, meanErrorPx: 0.8, spatialCells: 6, rotationDeg: 0.4, shift: 0.1, finite: true };

  it("a keyframe with enough solved-landmark observations, inliers, spread and a pose near the rigid prediction is re-posed", () => {
    expect(validateKeyframeRepose(ok, l)).toBe("none");
  });

  it("too few observations of solved landmarks: the keyframe is not re-posed from a short list, nor from mixed landmarks (there is no fallback)", () => {
    expect(validateKeyframeRepose({ ...ok, observations: 10, inliers: 10 }, l)).toBe("few_observations");
    expect(validateKeyframeRepose({ ...ok, observations: 40, inliers: 20 }, l)).toBe("few_inliers");
  });

  it("error, spread, pose jump and non-finite values each keep the old pose (carried by the rigid correction)", () => {
    expect(validateKeyframeRepose({ ...ok, meanErrorPx: 3.5 }, l)).toBe("error");
    expect(validateKeyframeRepose({ ...ok, spatialCells: 3 }, l)).toBe("spread");
    expect(validateKeyframeRepose({ ...ok, rotationDeg: 3.5 }, l)).toBe("pose_jump");
    expect(validateKeyframeRepose({ ...ok, shift: 0.8 }, l)).toBe("pose_jump");
    expect(validateKeyframeRepose({ ...ok, finite: false }, l)).toBe("invalid");
    expect(validateKeyframeRepose({ ...ok, meanErrorPx: NaN }, l)).toBe("invalid");
  });
});
