import { defineConfig } from "@mizchi/mutator";

export default defineConfig({
  include: [
    "src/ar/ARState.ts",
    "src/ar/ObjectVisibility.ts",
    "src/ar/Guidance.ts",
    "src/ar/ObjectGestures.ts",
    "src/vision/PoseValidation.ts",
    "src/vision/InitStageValidation.ts",
    "src/vision/RelocalizationSchedule.ts",
    "src/vision/PlaneRecovery.ts",
  ],
  runner: "vitest",
  config: "vitest.mutation.config.ts",
});
