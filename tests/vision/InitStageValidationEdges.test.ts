import { describe, expect, it } from "vitest";
import { validateInitStage, validateKeyframeRepose, type InitStageLimits, type InitStageMeasures, type KeyframeReposeLimits, type KeyframeReposeMeasures } from "../../src/vision/InitStageValidation";

/**
 * Exact-limit cases of the init-stage and keyframe re-pose verdicts (found
 * by mutation testing): every limit is inclusive — a value exactly at it
 * passes — and a rejection never reports `accepted`.
 */
const limits: InitStageLimits = { minGainFraction: 0.15, minLandmarks: 30, minSpatialCells: 4, maxRotationDeg: 3, maxShiftDepthRatio: 0.05, maxShiftedFraction: 0.1, maxReferenceErrorPx: 2.5 };
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
  referenceErrorPx: 0.3,
  converged: true,
};

describe("validateInitStage: exact limits", () => {
  it("a rejection is not accepted, whatever the reason", () => {
    expect(validateInitStage({ ...good, landmarks: 5 }, limits)).toMatchObject({ accepted: false, reason: "few_landmarks" });
    expect(validateInitStage({ ...good, rotationDeg: 12 }, limits)).toMatchObject({ accepted: false, reason: "pose_jump" });
    expect(validateInitStage({ ...good, finite: false }, limits)).toMatchObject({ accepted: false, reason: "invalid" });
  });

  it("exactly the minimum number of landmarks is enough", () => {
    expect(validateInitStage({ ...good, landmarks: 30 }, limits).accepted).toBe(true);
    expect(validateInitStage({ ...good, landmarks: 29 }, limits).reason).toBe("few_landmarks");
  });

  it("a rotation or a camera-centre move exactly at its limit is not a pose jump", () => {
    expect(validateInitStage({ ...good, rotationDeg: 3 }, limits).accepted).toBe(true);
    const maxShift = validateInitStage(good, limits).maxShift;
    expect(maxShift).toBeCloseTo(0.75, 12);
    expect(validateInitStage({ ...good, cameraShift: maxShift }, limits).accepted).toBe(true);
    expect(validateInitStage({ ...good, cameraShift: maxShift * 1.001 }, limits).reason).toBe("pose_jump");
  });
});

describe("validateKeyframeRepose: exact limits", () => {
  const l: KeyframeReposeLimits = { minInliers: 25, maxMeanErrorPx: 3.0, minSpatialCells: 4, maxRotationDeg: 3, maxShift: 0.75 };
  const ok: KeyframeReposeMeasures = { observations: 80, inliers: 70, meanErrorPx: 0.8, spatialCells: 6, rotationDeg: 0.4, shift: 0.1, finite: true };

  it("every measure exactly at its limit still re-poses the keyframe", () => {
    expect(validateKeyframeRepose({ ...ok, observations: 25, inliers: 25 }, l)).toBe("none");
    expect(validateKeyframeRepose({ ...ok, meanErrorPx: 3.0 }, l)).toBe("none");
    expect(validateKeyframeRepose({ ...ok, spatialCells: 4 }, l)).toBe("none");
    expect(validateKeyframeRepose({ ...ok, rotationDeg: 3 }, l)).toBe("none");
    expect(validateKeyframeRepose({ ...ok, shift: 0.75 }, l)).toBe("none");
  });

  it("one step past each limit names that limit", () => {
    expect(validateKeyframeRepose({ ...ok, observations: 24 }, l)).toBe("few_observations");
    expect(validateKeyframeRepose({ ...ok, inliers: 24 }, l)).toBe("few_inliers");
    expect(validateKeyframeRepose({ ...ok, meanErrorPx: 3.01 }, l)).toBe("error");
    expect(validateKeyframeRepose({ ...ok, spatialCells: 3 }, l)).toBe("spread");
    expect(validateKeyframeRepose({ ...ok, rotationDeg: 3.01 }, l)).toBe("pose_jump");
    expect(validateKeyframeRepose({ ...ok, shift: 0.76 }, l)).toBe("pose_jump");
  });
});
