import { execSync, spawn, type ChildProcess } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import path from "node:path";
import { chromium, type Browser, type ConsoleMessage } from "playwright";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { createRng, makeTexture } from "../helpers/synthetic";
import { writeGrayY4M } from "../helpers/y4m";

/**
 * End-to-end smoke test of the Phase 1 demo in headless Chromium.
 *
 *   1. generate a synthetic "camera" video (sliding crop of a texture)
 *   2. vite build + vite preview (https, self-signed)
 *   3. open the page with Chromium's fake camera fed from the video
 *   4. click Start, sample `window.__ar.stats()` for a few seconds
 *   5. assert that the worker pipeline tracks features and report numbers
 *
 * The video loops every 2 s, which produces a hard cut (large jump); the
 * engine must recover from it, so the test also exercises TRACKING_LOST.
 */

const ROOT = path.resolve(__dirname, "../..");
const OUT_DIR = path.join(ROOT, "test-results");
const PORT = 4173;
// The synthetic camera looks straight at a fronto-parallel plane; pretend the
// phone looks straight down at a floor by injecting gravity along +Z.
const URL = `https://localhost:${PORT}/?debug=1&gravity=0,0,1`;
const W = 640;
const H = 480;

const CHROMIUM_CANDIDATES = [
  process.env.PW_CHROMIUM_PATH,
  "/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
].filter((p): p is string => !!p && existsSync(p));

let server: ChildProcess | null = null;
let browser: Browser | null = null;

interface Stats {
  renderFps: number;
  visionFps: number;
  visionMs: number;
  quality: { featureCount: number; trackedCount: number; inlierCount: number; lowFeature: boolean };
  pose: {
    rotation: number[];
    translationDirection: number[];
    model: string;
    parallaxPx: number;
    confidence: number;
    translationConfidence: number;
    correspondences: number;
  } | null;
  mapPose: {
    rotation: number[];
    translation: number[];
    inlierCount: number;
    meanReprojectionErrorPx: number;
    landmarkCount: number;
    framesSinceTracked: number;
  } | null;
  plane: {
    normal: number[];
    inlierCount: number;
    horizontalness: number;
    horizontal: boolean;
    confidence: number;
    stableFrames: number;
    found: boolean;
    usedGravity: boolean;
  } | null;
  landmarkCount: number;
  state: string;
  framesProcessed: number;
  framesDropped: number;
  processingWidth: number;
  processingHeight: number;
  backend: string;
}

function makeVideo(file: string): void {
  const BW = W + 300;
  const BH = H + 200;
  const big = makeTexture(BW, BH, createRng(777), [12, 30, 70, 160]);
  const frames: Uint8Array[] = [];
  const n = 60; // 2 s at 30 fps
  for (let f = 0; f < n; f++) {
    const ox = 20 + Math.round(f * 3);
    const oy = 20 + Math.round(f * 2);
    const crop = new Uint8Array(W * H);
    for (let y = 0; y < H; y++) {
      const sy = oy + y;
      crop.set(big.subarray(sy * BW + ox, sy * BW + ox + W), y * W);
    }
    frames.push(crop);
  }
  writeGrayY4M(file, W, H, frames, 30);
}

function waitForServer(proc: ChildProcess): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("vite preview did not start")), 60_000);
    const onData = (chunk: Buffer) => {
      const text = chunk.toString();
      if (/localhost:\d+/.test(text)) {
        clearTimeout(timer);
        resolve();
      }
    };
    proc.stdout?.on("data", onData);
    proc.stderr?.on("data", onData);
    proc.on("exit", (code) => {
      clearTimeout(timer);
      reject(new Error(`vite preview exited with ${code}`));
    });
  });
}

describe("Phase 1 browser smoke test", () => {
  const videoFile = path.join(OUT_DIR, "synthetic-camera.y4m");
  const logs: string[] = [];

  beforeAll(async () => {
    mkdirSync(OUT_DIR, { recursive: true });
    makeVideo(videoFile);

    const viteBin = path.join(ROOT, "node_modules", "vite", "bin", "vite.js");
    execSync(`node "${viteBin}" build`, { cwd: ROOT, stdio: "pipe" });
    // Spawn the vite binary directly (not via npx) so that kill() reaches it.
    server = spawn(process.execPath, [viteBin, "preview", "--port", String(PORT), "--strictPort"], {
      cwd: ROOT,
      stdio: ["ignore", "pipe", "pipe"],
    });
    await waitForServer(server);

    browser = await chromium.launch({
      headless: true,
      executablePath: CHROMIUM_CANDIDATES[0],
      args: [
        "--use-fake-ui-for-media-stream",
        "--use-fake-device-for-media-stream",
        `--use-file-for-fake-video-capture=${videoFile}`,
        "--autoplay-policy=no-user-gesture-required",
        "--no-sandbox",
      ],
    });
  });

  afterAll(async () => {
    await browser?.close();
    server?.kill();
  });

  it("tracks features from the camera through the worker pipeline", async () => {
    const context = await browser!.newContext({ ignoreHTTPSErrors: true, permissions: ["camera"] });
    const page = await context.newPage();
    page.on("console", (m: ConsoleMessage) => {
      const t = m.text();
      if (t.startsWith("[AR]") || m.type() === "error") logs.push(t);
    });
    page.on("pageerror", (e) => logs.push(`pageerror: ${e.message}`));

    await page.goto(URL, { waitUntil: "load" });
    await page.click("#start");

    // Wait until a good number of frames went through the vision engine.
    await page.waitForFunction(() => window.__ar.stats().framesProcessed > 60, null, { timeout: 60_000 });

    const samples: Stats[] = [];
    for (let i = 0; i < 12; i++) {
      samples.push((await page.evaluate(() => window.__ar.stats())) as Stats);
      await page.waitForTimeout(500);
    }
    await context.close();

    const avg = (f: (s: Stats) => number) => samples.reduce((a, s) => a + f(s), 0) / samples.length;
    const max = (f: (s: Stats) => number) => Math.max(...samples.map(f));
    const states = samples.reduce<Record<string, number>>((acc, s) => {
      acc[s.state] = (acc[s.state] ?? 0) + 1;
      return acc;
    }, {});
    const report = {
      backend: samples[0].backend,
      processing: `${samples[0].processingWidth}x${samples[0].processingHeight}`,
      framesProcessed: samples[samples.length - 1].framesProcessed,
      framesDropped: samples[samples.length - 1].framesDropped,
      renderFps: avg((s) => s.renderFps).toFixed(1),
      visionFps: avg((s) => s.visionFps).toFixed(1),
      visionMs: avg((s) => s.visionMs).toFixed(1),
      featureCount: `${avg((s) => s.quality.featureCount).toFixed(0)} avg / ${max((s) => s.quality.featureCount)} max`,
      trackedCount: `${avg((s) => s.quality.trackedCount).toFixed(0)} avg / ${max((s) => s.quality.trackedCount)} max`,
      inlierCount: `${avg((s) => s.quality.inlierCount).toFixed(0)} avg / ${max((s) => s.quality.inlierCount)} max`,
      states,
      pose: samples
        .filter((s) => s.pose)
        .map((s) => ({
          model: s.pose!.model,
          t: s.pose!.translationDirection.map((v) => v.toFixed(2)).join(","),
          parallax: s.pose!.parallaxPx.toFixed(1),
          tConf: s.pose!.translationConfidence.toFixed(2),
          rotDeg: ((Math.acos(Math.max(-1, Math.min(1, (s.pose!.rotation[0] + s.pose!.rotation[4] + s.pose!.rotation[8] - 1) / 2))) * 180) / Math.PI).toFixed(2),
        })),
    };
    const phase3 = samples.map((s) => ({
      state: s.state,
      landmarks: s.landmarkCount,
      pnp: s.mapPose?.inlierCount ?? null,
      reproj: s.mapPose ? s.mapPose.meanReprojectionErrorPx.toFixed(2) : null,
      mapT: s.mapPose ? s.mapPose.translation.map((v) => v.toFixed(2)).join(",") : null,
      plane: s.plane
        ? `n=(${s.plane.normal.map((v) => v.toFixed(2)).join(",")}) in=${s.plane.inlierCount} hz=${s.plane.horizontalness.toFixed(2)} conf=${s.plane.confidence.toFixed(2)} found=${s.plane.found}`
        : null,
    }));
    console.log("[browser-smoke] " + JSON.stringify(report, null, 2));
    console.log("[browser-smoke] phase3 " + JSON.stringify(phase3, null, 2));

    // Phase 2: the synthetic camera translates over a fronto-parallel plane.
    // The crop offset grows by (3, 2) px/frame, so image content moves by
    // (−3, −2) px/frame: the camera moved toward (+X, +Y) in the CV frame,
    // C = (+, +, 0) and t = −R·C ∝ (−3, −2, 0).
    const poses = samples.map((s) => s.pose).filter((p): p is NonNullable<Stats["pose"]> => !!p);
    expect(poses.length).toBeGreaterThanOrEqual(samples.length / 2);
    for (const p of poses) {
      const rotDeg = (Math.acos(Math.max(-1, Math.min(1, (p.rotation[0] + p.rotation[4] + p.rotation[8] - 1) / 2))) * 180) / Math.PI;
      expect(rotDeg, "rotation should stay near identity for a pure translation").toBeLessThan(3);
    }
    const confident = poses.filter((p) => p.translationConfidence > 0.3);
    expect(confident.length).toBeGreaterThanOrEqual(1);
    for (const p of confident) {
      const [tx, ty] = p.translationDirection;
      const len = Math.hypot(tx, ty);
      const cos = (tx * -3 + ty * -2) / (len * Math.hypot(3, 2));
      expect(cos, `t direction ${p.translationDirection}`).toBeGreaterThan(Math.cos((20 * Math.PI) / 180));
    }
    console.log("[browser-smoke] console:\n" + logs.slice(0, 12).join("\n"));

    const errors = logs.filter((l) => l.startsWith("pageerror") || /error/i.test(l) && !l.startsWith("[AR]"));
    expect(errors, errors.join("\n")).toEqual([]);
    expect(samples[0].backend).toBe("worker");
    expect(samples[0].processingWidth).toBe(640);
    expect(samples[samples.length - 1].framesProcessed).toBeGreaterThan(100);
    expect(avg((s) => s.visionFps)).toBeGreaterThan(8);
    expect(max((s) => s.quality.featureCount)).toBeGreaterThanOrEqual(200);
    expect(max((s) => s.quality.trackedCount)).toBeGreaterThanOrEqual(100);
    const trackingLike = (states["TRACKING"] ?? 0) + (states["PLANE_DETECTING"] ?? 0) + (states["PLANE_FOUND"] ?? 0);
    expect(trackingLike).toBeGreaterThanOrEqual(samples.length / 2);

    // Phase 3: the map initializes, landmarks accumulate, and the (gravity-
    // injected) floor plane is found with a normal along −Z (toward the camera).
    const withMap = samples.filter((s) => s.mapPose);
    expect(withMap.length).toBeGreaterThanOrEqual(samples.length / 2);
    expect(max((s) => s.landmarkCount)).toBeGreaterThanOrEqual(50);
    const found = samples.filter((s) => s.plane?.found);
    expect(found.length).toBeGreaterThanOrEqual(1);
    for (const s of found) {
      expect(s.plane!.usedGravity).toBe(true);
      expect(s.plane!.horizontal).toBe(true);
      expect(s.plane!.normal[2]).toBeLessThan(-0.95);
    }
    expect(states["PLANE_FOUND"] ?? 0).toBeGreaterThanOrEqual(1);
  });
});
