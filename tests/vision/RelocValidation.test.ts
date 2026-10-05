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
 * never uses the normal-tracking jump gate. v12 adds the strong /
 * acceptable / reject levels with a relaxed error range, the relaxed-range
 * NCC requirement and relocalization-specific jump limits: tested with the
 * relaxed range disabled (`T`, the v9 behaviour) and with the default
 * config (`TA`).
 */
const cfg = resolveConfig().relocalization;
/** v9 thresholds: relaxed range disabled (equal to the strict bound), no NCC / jump limits. */
const T: RelocValidationThresholds = {
  minInliers: cfg.minInliers,
  maxMeanErrorPx: cfg.maxMeanErrorPx,
  relaxedMeanErrorPx: cfg.maxMeanErrorPx,
  minInlierRatio: cfg.minInlierRatio,
  minSpatialCells: cfg.minSpatialCells,
  minSpatialCoverage: cfg.minSpatialCoverage,
};
/** Default config (v12): relaxed range, NCC requirement and jump limits (depth 10 map units). */
const DEPTH = 10;
const TA: RelocValidationThresholds = {
  ...T,
  relaxedMeanErrorPx: cfg.relaxedMeanErrorPx,
  relaxedMinMatchScore: cfg.relaxedMinMatchScore,
  maxTranslationJump: cfg.maxTranslationJumpDepthRatio * DEPTH,
  maxRotationJumpDeg: cfg.maxRotationJumpDeg,
};
/** The on-device return (v12 §18): 50i / 2.44 px / ratio 0.85 / 7 cells / NCC 0.67. */
const DEVICE: Partial<RelocCandidateMeasures> = { inliers: 50, reprojectionErrorPx: 2.44, inlierRatio: 0.85, coveredCells: 7, spatialCoverage: 0.7, matchScore: 0.67 };

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
    expect(v.level).toBe("strong");
    expect(v.reprojectionStrictOk).toBe(true);
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

  it("Test 3: reprojection error above the strict limit with the relaxed range disabled → rejectReason reprojection_error (the earlier on-device 35i / 2.93px case)", () => {
    const v = validateRelocalizationCandidate(good({ inliers: 35, reprojectionErrorPx: 2.93, inlierRatio: 0.61, coveredCells: 4, spatialCoverage: 0.58 }), T);
    expect(v.inliersPassed).toBe(true);
    expect(v.reprojectionPassed).toBe(cfg.maxMeanErrorPx >= 2.93);
    expect(v.ratioPassed).toBe(0.61 >= cfg.minInlierRatio);
    expect(v.spatialPassed).toBe(4 >= cfg.minSpatialCells);
    // With the strict threshold alone (1.5 px) this candidate fails on the
    // reprojection error, and the breakdown says exactly that.
    expect(v.passed).toBe(false);
    expect(v.rejectReason).toBe("reprojection_error");
    expect(v.level).toBe("reject");
    expect(stageOfValidation(v)).toBe("reprojection");
    expect(v.reprojectionErrorPx).toBe(2.93);
    expect(v.maxReprojectionErrorPx).toBe(cfg.maxMeanErrorPx);
    expect(v.relaxedReprojectionErrorPx).toBe(cfg.maxMeanErrorPx);
  });

  // ---- v12: strong / acceptable / reject ----
  it("v12 Test 1 / 11: a strong candidate (error within the strict bound) is accepted as before", () => {
    const v = validateRelocalizationCandidate(good({ ...DEVICE, reprojectionErrorPx: 1.2 }), TA);
    expect(v.level).toBe("strong");
    expect(v.passed).toBe(true);
    expect(v.reprojectionStrictOk).toBe(true);
    expect(v.reprojectionRelaxedOk).toBe(true);
    expect(v.rejectReason).toBeNull();
    expect(stageOfValidation(v)).toBe("ok");
    // Unchanged strict bound; the relaxed range never widens the strong one.
    expect(cfg.maxMeanErrorPx).toBe(1.5);
    expect(v.maxReprojectionErrorPx).toBe(1.5);
    // A strong candidate does not need the relaxed-range NCC (the coarse gate already applied).
    expect(validateRelocalizationCandidate(good({ ...DEVICE, reprojectionErrorPx: 1.2, matchScore: 0.3 }), TA).level).toBe("strong");
  });

  it("v12 Test 2: the on-device 50i / 2.44px / 0.85 / 7 cells / NCC 0.67 return → strict NG, relaxed OK, level ACCEPTABLE, accepted", () => {
    expect(cfg.relaxedMeanErrorPx).toBeGreaterThanOrEqual(2.44);
    expect(cfg.relaxedMeanErrorPx).toBeLessThan(5); // §24: never a large relaxation
    const v = validateRelocalizationCandidate(good(DEVICE), TA);
    expect(v.reprojectionStrictOk).toBe(false);
    expect(v.reprojectionRelaxedOk).toBe(true);
    expect(v.inliersPassed).toBe(true);
    expect(v.ratioPassed).toBe(true);
    expect(v.spatialPassed).toBe(true);
    expect(v.nccPassed).toBe(true);
    expect(v.posePassed).toBe(true);
    expect(v.level).toBe("acceptable");
    expect(v.passed).toBe(true);
    expect(v.rejectReason).toBeNull();
    expect(stageOfValidation(v)).toBe("ok");
    expect(v.relaxedReprojectionErrorPx).toBe(cfg.relaxedMeanErrorPx);
    expect(v.requiredNccScore).toBe(cfg.relaxedMinMatchScore);
  });

  it("v12 Test 3: error above the relaxed bound → reject, reason reprojection_error", () => {
    const v = validateRelocalizationCandidate(good({ ...DEVICE, reprojectionErrorPx: 3.5 }), TA);
    expect(v.level).toBe("reject");
    expect(v.rejectReason).toBe("reprojection_error");
    expect(stageOfValidation(v)).toBe("reprojection");
    expect(v.reprojectionRelaxedOk).toBe(false);
    // Exactly at the relaxed bound is still acceptable; a bound ≤ strict disables the range.
    expect(validateRelocalizationCandidate(good({ ...DEVICE, reprojectionErrorPx: cfg.relaxedMeanErrorPx }), TA).level).toBe("acceptable");
    const off = validateRelocalizationCandidate(good({ ...DEVICE, reprojectionErrorPx: 2.0 }), { ...TA, relaxedMeanErrorPx: 1.0 });
    expect(off.level).toBe("reject");
    expect(off.relaxedReprojectionErrorPx).toBe(cfg.maxMeanErrorPx);
    expect(validateRelocalizationCandidate(good({ ...DEVICE, reprojectionErrorPx: 2.0 }), { ...T, relaxedMeanErrorPx: undefined }).level).toBe("reject");
  });

  it("v12 Test 4: bad inlier ratio in the relaxed range → reject, reason inlier_ratio", () => {
    const v = validateRelocalizationCandidate(good({ ...DEVICE, reprojectionErrorPx: 2.4, inlierRatio: 0.35 }), TA);
    expect(v.level).toBe("reject");
    expect(v.rejectReason).toBe("inlier_ratio");
    expect(stageOfValidation(v)).toBe("ratio");
  });

  it("v12 Test 5: bad spatial distribution in the relaxed range → reject, reason spatial_distribution", () => {
    const v = validateRelocalizationCandidate(good({ ...DEVICE, reprojectionErrorPx: 2.4, coveredCells: 2 }), TA);
    expect(v.level).toBe("reject");
    expect(v.rejectReason).toBe("spatial_distribution");
    expect(stageOfValidation(v)).toBe("spatial");
  });

  it("v12 Test 6: bad NCC in the relaxed range → reject, reason ncc", () => {
    const v = validateRelocalizationCandidate(good({ ...DEVICE, reprojectionErrorPx: 2.4, matchScore: 0.3 }), TA);
    expect(v.level).toBe("reject");
    expect(v.nccPassed).toBe(false);
    expect(v.rejectReason).toBe("ncc");
    expect(stageOfValidation(v)).toBe("ncc");
    // Too few inliers is reported before the NCC (§17 order).
    expect(validateRelocalizationCandidate(good({ ...DEVICE, reprojectionErrorPx: 2.4, matchScore: 0.3, inliers: cfg.minInliers - 1 }), TA).rejectReason).toBe("inliers");
  });

  it("v12 Test 7: an invalid pose rejects whatever the other values, reason pose_invalid", () => {
    const v = validateRelocalizationCandidate(good({ ...DEVICE, poseFinite: false }), TA);
    expect(v.level).toBe("reject");
    expect(v.rejectReason).toBe("pose_invalid");
    expect(stageOfValidation(v)).toBe("invalid");
    expect(validateRelocalizationCandidate(good({ ...DEVICE, reprojectionErrorPx: NaN }), TA).rejectReason).toBe("pose_invalid");
  });

  it("v12 Test 8 / 9: the relocalization-specific jump limits reject in the relaxed range too (translation / rotation)", () => {
    const limitT = cfg.maxTranslationJumpDepthRatio * DEPTH;
    const t = validateRelocalizationCandidate(good({ ...DEVICE, translationJump: limitT + 0.01, rotationJumpDeg: 5 }), TA);
    expect(t.level).toBe("reject");
    expect(t.translationJumpPassed).toBe(false);
    expect(t.rejectReason).toBe("translation_jump");
    expect(stageOfValidation(t)).toBe("jump");
    expect(t.maxTranslationJump).toBe(limitT);
    const r = validateRelocalizationCandidate(good({ ...DEVICE, translationJump: 0.1, rotationJumpDeg: cfg.maxRotationJumpDeg + 1 }), TA);
    expect(r.level).toBe("reject");
    expect(r.rotationJumpPassed).toBe(false);
    expect(r.rejectReason).toBe("rotation_jump");
    // The same limits apply to a strong candidate (§9, §17).
    expect(validateRelocalizationCandidate(good({ ...DEVICE, reprojectionErrorPx: 1.0, translationJump: limitT + 1 }), TA).rejectReason).toBe("translation_jump");
    // Within the limits (or unknown) the jump passes; with no limits configured it is diagnostic only (v9 §16–§17 behaviour).
    expect(validateRelocalizationCandidate(good({ ...DEVICE, translationJump: limitT - 0.01, rotationJumpDeg: cfg.maxRotationJumpDeg - 1 }), TA).level).toBe("acceptable");
    expect(validateRelocalizationCandidate(good({ ...DEVICE, translationJump: NaN, rotationJumpDeg: NaN }), TA).level).toBe("acceptable");
    expect(validateRelocalizationCandidate(good({ translationJump: 25, rotationJumpDeg: 150 }), T).passed).toBe(true);
  });

  it("v12 §10–§11: an acceptable candidate always needs a confirmation frame; strong follows the configured count", () => {
    expect(requiredConfirmations("strong", cfg.confirmationFrames)).toBe(cfg.confirmationFrames);
    expect(requiredConfirmations("strong", 0)).toBe(0); // "always apply at once" config
    expect(requiredConfirmations("acceptable", 0)).toBe(1); // … except for an acceptable candidate
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
    // The reason is the first in the fixed order (v12 §17: ratio and spatial
    // come before the reprojection error); the others stay visible.
    expect(v.rejectReason).toBe("inlier_ratio");
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

  it("Test 8: without relocalization jump limits a large pose jump is diagnostic only (v9 §16–§17; v12 adds optional limits)", () => {
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
