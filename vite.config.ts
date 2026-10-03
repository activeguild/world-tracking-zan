/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import basicSsl from "@vitejs/plugin-basic-ssl";

// getUserMedia() requires a secure context. `localhost` is secure by default,
// but on-device testing over LAN (iPhone Safari / Android Chrome) needs HTTPS,
// so the dev server always serves a self-signed certificate.
export default defineConfig({
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
