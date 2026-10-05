/// <reference types="vitest/config" />
import { execSync } from "node:child_process";
import { defineConfig } from "vite";
import basicSsl from "@vitejs/plugin-basic-ssl";

function gitShortHash(): string {
  try {
    return execSync("git rev-parse --short HEAD", { stdio: ["ignore", "pipe", "ignore"] }).toString().trim();
  } catch {
    return process.env.VERCEL_GIT_COMMIT_SHA?.slice(0, 7) ?? "dev";
  }
}

/** Shown in the HUD so an on-device tester can tell which build is running. */
const BUILD_LABEL = `phase5 ${gitShortHash()} ${new Date().toISOString().slice(0, 16).replace("T", " ")}Z`;

// getUserMedia() requires a secure context. `localhost` is secure by default,
// but on-device testing over LAN (iPhone Safari / Android Chrome) needs HTTPS,
// so the dev server always serves a self-signed certificate.
export default defineConfig({
  define: {
    __BUILD_LABEL__: JSON.stringify(BUILD_LABEL),
  },
  plugins: [basicSsl()],
  server: {
    port: 5173,
    strictPort: false,
  },
  build: {
    target: "es2022",
    sourcemap: true,
  },
  worker: {
    format: "es",
  },
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
    testTimeout: 30_000,
  },
});
