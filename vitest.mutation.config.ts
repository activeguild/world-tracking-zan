import { defineConfig } from "vitest/config";

// Tests the mutation run may use. Only the fast unit tests of the decision
// modules in mutator.config.ts: the synthetic-video engine tests also reach
// these modules, but collecting per-test coverage through them takes longer
// than the whole mutation run is worth.
export default defineConfig({
  test: {
    environment: "node",
    include: [
      "tests/ar/ARState.test.ts",
      "tests/lean/Conformance.test.ts",
      "tests/ar/ObjectVisibility.test.ts",
      "tests/ar/Guidance.test.ts",
      "tests/ar/ObjectGestures.test.ts",
      "tests/vision/PoseValidation.test.ts",
      "tests/vision/InitStageValidation.test.ts",
      "tests/vision/PlaneRecovery.test.ts",
      "tests/vision/RelocalizationSpeed.test.ts",
    ],
    testTimeout: 30_000,
  },
});
