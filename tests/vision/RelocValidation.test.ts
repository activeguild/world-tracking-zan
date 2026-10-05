import { describe, expect, it } from "vitest";
import { resolveConfig } from "../../src/ar/ARConfig";
import {
  requiredConfirmations,
  spatialCoverageOf,
  stageOfValidation,
  validateRelocalizationCandidate,
  type RelocCandidateMeasures,
  type RelocValidationThresholds,
} from "../../src/vision/Relocalizer";

/**
 * 修正指示書 v9: the relocalization validation evaluates every condition
 * independently, keeps all PASS / FAIL flags, names the reject reason and
 * never uses the normal-tracking jump gate. v12 adds the acceptable error
 * tier: tested with the tier disabled (`T`, the v9 behaviour) and with the
 * default config (`TA`).
 */
const cfg = resolveConfig().relocalization;
/** v9 thresholds: acceptable tier disabled (equal to the strict bound). */
const T: RelocValidationThresholds = {
  minInliers: cfg.minInliers,
  maxMeanErrorPx: cfg.maxMeanErrorPx,
  acceptableMeanErrorPx: cfg.maxMeanErrorPx,
  minInlierRatio: cfg.minInlierRatio,
  minSpatialCells: cfg.minSpatialCells,
  minSpatialCoverage: cfg.minSpatialCoverage,
};
/** Default config: acceptable tier enabled (v12). */
const TA: RelocValidationThresholds = { ...T, acceptableMeanErrorPx: cfg.acceptableMeanErrorPx };

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
    expect(v.errorTier).toBe("strict");
    expect(v.reprojectionAcceptable).toBe(false);
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

  it("Test 3: reprojection error above the strict limit with the tier disabled → rejectReason reprojection_error (the on-device 35i / 2.93px case)", () => {
    const v = validateRelocalizationCandidate(good({ inliers: 35, reprojectionErrorPx: 2.93, inlierRatio: 0.61, coveredCells: 4, spatialCoverage: 0.58 }), T);
    expect(v.inliersPassed).toBe(true);
    expect(v.reprojectionPassed).toBe(cfg.maxMeanErrorPx >= 2.93);
    expect(v.ratioPassed).toBe(0.61 >= cfg.minInlierRatio);
    expect(v.spatialPassed).toBe(4 >= cfg.minSpatialCells);
    // With the strict threshold alone (1.5 px) this candidate fails on the
    // reprojection error, and the breakdown says exactly that.
    expect(v.passed).toBe(false);
    expect(v.rejectReason).toBe("reprojection_error");
    expect(v.errorTier).toBe("rejected");
    expect(stageOfValidation(v)).toBe("reprojection");
    expect(v.reprojectionErrorPx).toBe(2.93);
    expect(v.maxReprojectionErrorPx).toBe(cfg.maxMeanErrorPx);
    expect(v.acceptableReprojectionErrorPx).toBe(cfg.maxMeanErrorPx);
  });

  // ---- v12: acceptable error tier ----
  it("v12 Test 1: the on-device 2.44 px return passes as ACCEPTABLE when inliers, ratio, cells and pose all pass; strict stays 1.5 px", () => {
    expect(cfg.maxMeanErrorPx).toBe(1.5); // the strict bound is not moved
    expect(cfg.acceptableMeanErrorPx).toBeGreaterThanOrEqual(2.44);
    const v = validateRelocalizationCandidate(good({ inliers: 35, reprojectionErrorPx: 2.44, inlierRatio: 0.61, coveredCells: 4, spatialCoverage: 0.58 }), TA);
    expect(v.reprojectionPassed).toBe(false); // strict condition still fails …
    expect(v.reprojectionAcceptable).toBe(true); // … but the tier rescues it
    expect(v.errorTier).toBe("acceptable");
    expect(v.passed).toBe(true);
    expect(v.rejectReason).toBeNull();
    expect(stageOfValidation(v)).toBe("ok");
    expect(v.acceptableReprojectionErrorPx).toBe(cfg.acceptableMeanErrorPx);
    // A plainly good candidate is still strict.
    expect(validateRelocalizationCandidate(good(), TA).errorTier).toBe("strict");
  });

  it("v12 Test 2: within the acceptable bound the candidate is carried by the other conditions — any of them failing rejects it, named as the reason", () => {
    const base = { inliers: 35, reprojectionErrorPx: 2.44, inlierRatio: 0.61, coveredCells: 4, spatialCoverage: 0.58 };
    const ratio = validateRelocalizationCandidate(good({ ...base, inlierRatio: cfg.minInlierRatio - 0.05 }), TA);
    expect(ratio.passed).toBe(false);
    expect(ratio.errorTier).toBe("rejected");
    expect(ratio.reprojectionAcceptable).toBe(false);
    expect(ratio.rejectReason).toBe("inlier_ratio");
    const cells = validateRelocalizationCandidate(good({ ...base, coveredCells: cfg.minSpatialCells - 1 }), TA);
    expect(cells.passed).toBe(false);
    expect(cells.rejectReason).toBe("spatial_distribution");
    const inliers = validateRelocalizationCandidate(good({ ...base, inliers: cfg.minInliers - 1 }), TA);
    expect(inliers.passed).toBe(false);
    expect(inliers.rejectReason).toBe("inliers");
    const pose = validateRelocalizationCandidate(good({ ...base, poseFinite: false }), TA);
    expect(pose.passed).toBe(false);
    expect(pose.rejectReason).toBe("pose_invalid");
    // Jump stays diagnostic in the acceptable tier too (v9 §16–§17).
    const jump = validateRelocalizationCandidate(good({ ...base, translationJump: 25, rotationJumpDeg: 150 }), TA);
    expect(jump.passed).toBe(true);
    expect(jump.errorTier).toBe("acceptable");
  });

  it("v12 Test 3: beyond the acceptable bound the error itself rejects, whatever the other conditions", () => {
    const v = validateRelocalizationCandidate(good({ reprojectionErrorPx: cfg.acceptableMeanErrorPx + 0.01 }), TA);
    expect(v.passed).toBe(false);
    expect(v.errorTier).toBe("rejected");
    expect(v.rejectReason).toBe("reprojection_error");
    expect(stageOfValidation(v)).toBe("reprojection");
    // Exactly at the bound is still acceptable.
    expect(validateRelocalizationCandidate(good({ reprojectionErrorPx: cfg.acceptableMeanErrorPx }), TA).errorTier).toBe("acceptable");
    // A bound at or below the strict one disables the tier.
    const off = validateRelocalizationCandidate(good({ reprojectionErrorPx: 2.0 }), { ...TA, acceptableMeanErrorPx: 1.0 });
    expect(off.errorTier).toBe("rejected");
    expect(off.acceptableReprojectionErrorPx).toBe(cfg.maxMeanErrorPx);
    const undef = validateRelocalizationCandidate(good({ reprojectionErrorPx: 2.0 }), { ...T, acceptableMeanErrorPx: undefined });
    expect(undef.errorTier).toBe("rejected");
  });

  it("v12 Test 4: an acceptable-tier candidate always needs a confirmation frame; strict follows the configured count", () => {
    expect(requiredConfirmations("strict", cfg.confirmationFrames)).toBe(cfg.confirmationFrames);
    expect(requiredConfirmations("strict", 0)).toBe(0); // "always apply at once" config
    expect(requiredConfirmations("acceptable", 0)).toBe(1); // … except for the acceptable tier
    expect(requiredConfirmations("acceptable", cfg.confirmationFrames)).toBe(Math.max(1, cfg.confirmationFrames));
    expect(requiredConfirmations("acceptable", 2)).toBe(2);
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
