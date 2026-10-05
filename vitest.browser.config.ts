import { defineConfig } from "vitest/config";

/**
 * Browser smoke test: builds the app, serves it, and drives headless Chromium
 * with a synthetic camera (fake video capture from a generated .y4m file).
 *
 *   npm run test:browser
 */
export default defineConfig({
  test: {
    environment: "node",
    include: ["tests/integration/**/*.btest.ts"],
    testTimeout: 180_000,
    hookTimeout: 180_000,
    fileParallelism: false,
  },
});
