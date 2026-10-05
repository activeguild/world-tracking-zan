import { describe, expect, it } from "vitest";
import { resolveConfig } from "../../src/ar/ARConfig";
import {
  spatialCoverageOf,
  stageOfValidation,
  validateRelocalizationCandidate,
  type RelocCandidateMeasures,
  type RelocValidationThresholds,
} from "../../src/vision/Relocalizer";

/**
 * 修正指示書 v9: the relocalization validation evaluates every condition
 * independently, keeps all PASS / FAIL flags, names the reject reason and
 * never uses the normal-tracking jump gate.
 */
const cfg = resolveConfig().relocalization;
const T: RelocValidationThresholds = {
  minInliers: cfg.minInliers,
  maxMeanErrorPx: cfg.maxMeanErrorPx,
  minInlierRatio: cfg.minInlierRatio,
  minSpatialCells: cfg.minSpatialCells,
  minSpatialCoverage: cfg.minSpatialCoverage,
};

function good(over: Partial<RelocCandidateMeasures> = {}): RelocCandidateMeasures {
  return {
    inliers: cfg.minInliers + 20,
    reprojectionErrorPx: cfg.maxMeanErrorPx / 2,
    inlierRatio: Math.min(1, cfg.minInlierRatio + 0.3),
    coveredCells: 9,
    spatialCoverage: 0.8,
    poseFinite: true,
    ...over,
  };
}

describe("validateRelocalizationCandidate (v9 §5–§8)", () => {
  it("Test 1: enough inliers and a low reprojection error pass both conditions, and the candidate passes", () => {
    const v = validateRelocalizationCandidate(good(), T);
    expect(v.inliersPassed).toBe(true);
    expect(v.reprojectionPassed).toBe(true);
    expect(v.ratioPassed).toBe(true);
    expect(v.spatialPassed).toBe(true);
    expect(v.coveragePassed).toBe(true);
    expect(v.posePassed).toBe(true);
    expect(v.passed).toBe(true);
    expect(v.rejectReason).toBeNull();
    expect(stageOfValidation(v)).toBe("ok");
    // Thresholds are echoed from the config, never hard-coded.
    expect(v.requiredInliers).toBe(cfg.minInliers);
    expect(v.maxReprojectionErrorPx).toBe(cfg.maxMeanErrorPx);
    expect(v.minInlierRatio).toBe(cfg.minInlierRatio);
    expect(v.minSpatialCells).toBe(cfg.minSpatialCells);
    expect(v.totalCells).toBe(9);
  });

  it("Test 2: too few inliers → rejectReason inliers", () => {
    const v = validateRelocalizationCandidate(good({ inliers: cfg.minInliers - 1 }), T);
    expect(v.inliersPassed).toBe(false);
    expect(v.passed).toBe(false);
    expect(v.rejectReason).toBe("inliers");
    expect(stageOfValidation(v)).toBe("pnp_inliers");
  });

  it("Test 3: reprojection error above the limit → rejectReason reprojection_error (the on-device 35i / 2.93px case)", () => {
    const v = validateRelocalizationCandidate(good({ inliers: 35, reprojectionErrorPx: 2.93, inlierRatio: 0.61, coveredCells: 4, spatialCoverage: 0.58 }), T);
    expect(v.inliersPassed).toBe(true);
    expect(v.reprojectionPassed).toBe(cfg.maxMeanErrorPx >= 2.93);
    expect(v.ratioPassed).toBe(0.61 >= cfg.minInlierRatio);
    expect(v.spatialPassed).toBe(4 >= cfg.minSpatialCells);
    // With the current thresholds (1.5 px) this candidate fails on the
    // reprojection error alone, and the breakdown says exactly that.
    expect(v.passed).toBe(false);
    expect(v.rejectReason).toBe("reprojection_error");
    expect(stageOfValidation(v)).toBe("reprojection");
    expect(v.reprojectionErrorPx).toBe(2.93);
    expect(v.maxReprojectionErrorPx).toBe(cfg.maxMeanErrorPx);
  });

  it("Test 4: low inlier ratio → ratioPassed false", () => {
    const v = validateRelocalizationCandidate(good({ inlierRatio: cfg.minInlierRatio - 0.05 }), T);
    expect(v.ratioPassed).toBe(false);
    expect(v.rejectReason).toBe("inlier_ratio");
    expect(stageOfValidation(v)).toBe("ratio");
  });

  it("Test 5: too few covered cells → spatialPassed false", () => {
    const v = validateRelocalizationCandidate(good({ coveredCells: cfg.minSpatialCells - 1 }), T);
    expect(v.spatialPassed).toBe(false);
    expect(v.rejectReason).toBe("spatial_distribution");
    expect(stageOfValidation(v)).toBe("spatial");
    // Coverage is a separate condition (v9 §12–§14): low coverage fails it on its own.
    const t2 = { ...T, minSpatialCoverage: 0.3 };
    const v2 = validateRelocalizationCandidate(good({ coveredCells: 4, spatialCoverage: 0.08 }), t2);
    expect(v2.spatialPassed).toBe(true);
    expect(v2.coveragePassed).toBe(false);
    expect(v2.rejectReason).toBe("spatial_distribution");
    const v3 = validateRelocalizationCandidate(good({ coveredCells: 4, spatialCoverage: 0.6 }), t2);
    expect(v3.coveragePassed).toBe(true);
    expect(v3.passed).toBe(true);
  });

  it("Test 6: several failing conditions keep every flag (not only the first failure)", () => {
    const v = validateRelocalizationCandidate(
      good({ inlierRatio: cfg.minInlierRatio - 0.1, coveredCells: cfg.minSpatialCells - 2, reprojectionErrorPx: cfg.maxMeanErrorPx * 2 }),
      T,
    );
    expect(v.inliersPassed).toBe(true);
    expect(v.reprojectionPassed).toBe(false);
    expect(v.ratioPassed).toBe(false);
    expect(v.spatialPassed).toBe(false);
    expect(v.passed).toBe(false);
    // The reason is the first in the fixed order; the others stay visible.
    expect(v.rejectReason).toBe("reprojection_error");
  });

  it("Test 7: a non-finite pose → rejectReason pose_invalid, whatever the other values", () => {
    const v = validateRelocalizationCandidate(good({ poseFinite: false }), T);
    expect(v.posePassed).toBe(false);
    expect(v.passed).toBe(false);
    expect(v.rejectReason).toBe("pose_invalid");
    expect(stageOfValidation(v)).toBe("invalid");
    const v2 = validateRelocalizationCandidate(good({ reprojectionErrorPx: NaN }), T);
    expect(v2.rejectReason).toBe("pose_invalid");
  });

  it("Test 8: a large pose jump from the held pose is diagnostic only and never rejects a relocalization candidate (v9 §16–§17)", () => {
    const v = validateRelocalizationCandidate(good({ translationJump: 25, rotationJumpDeg: 150 }), T);
    expect(v.passed).toBe(true);
    expect(v.rejectReason).toBeNull();
    expect(v.translationJump).toBe(25);
    expect(v.rotationJumpDeg).toBe(150);
  });

  it("spatialCoverageOf: bounding box over the image area", () => {
    expect(spatialCoverageOf([0, 640], [0, 480], 2, 640, 480)).toBeCloseTo(1, 9);
    expect(spatialCoverageOf([100, 420], [100, 340], 2, 640, 480)).toBeCloseTo((320 * 240) / (640 * 480), 9);
    expect(spatialCoverageOf([10, 12], [10, 11], 2, 640, 480)).toBeLessThan(0.001);
    expect(spatialCoverageOf([], [], 0, 640, 480)).toBe(0);
  });
});
