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
    mapFrameId: number;
    framesSinceTracked: number;
    source: string;
  } | null;
  planePose: {
    tracked: boolean;
    inlierCount: number;
    candidateCount: number;
    inlierRatio: number;
    reprojectionErrorPx: number;
    confidence: number;
  } | null;
  planeAnchored: boolean;
  poseAgeMs: number;
  poseStale: boolean;
  cameraWorldPosition: number[] | null;
  objects: { id: number; position: number[]; yaw: number }[];
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
  worldReady: boolean;
  worldScale: number;
  placedObjects: number;
  relocalization: { keyframes: number; attempt: string; inlierCount: number; successCount: number } | null;
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

  it("debug HUD and debug console logs are off without ?debug=1 and on with it (v9 §23–§24, Tests 10–11)", async () => {
    const base = `https://localhost:${PORT}/`;
    // Plain URL: HUD hidden, feature overlay hidden, toggle present, no [AR] console output.
    const ctxOff = await browser!.newContext({ ignoreHTTPSErrors: true, permissions: ["camera"] });
    const pageOff = await ctxOff.newPage();
    const offLogs: string[] = [];
    pageOff.on("console", (m: ConsoleMessage) => {
      if (m.text().startsWith("[AR]")) offLogs.push(m.text());
    });
    await pageOff.goto(base, { waitUntil: "load" });
    await pageOff.click("#start");
    await pageOff.waitForFunction(() => window.__ar.stats().framesProcessed > 20, null, { timeout: 60_000 });
    const off = await pageOff.evaluate(() => ({
      hud: getComputedStyle(document.querySelector(".ar-hud")!).display,
      overlay: getComputedStyle(document.getElementById("overlay")!).display,
      toggle: getComputedStyle(document.getElementById("debug-toggle")!).display,
      toggleOn: document.getElementById("debug-toggle")!.classList.contains("on"),
      // Diagnostics are still produced with the HUD off.
      motion: window.__ar.stats().motion?.level ?? null,
    }));
    expect(off.hud).toBe("none");
    expect(off.overlay).toBe("none");
    expect(off.toggle).not.toBe("none");
    expect(off.toggleOn).toBe(false);
    expect(off.motion).not.toBeNull();
    expect(offLogs, offLogs.join("\n")).toEqual([]);
    // One tap turns the HUD on; another turns it off again.
    await pageOff.click("#debug-toggle");
    expect(await pageOff.evaluate(() => getComputedStyle(document.querySelector(".ar-hud")!).display)).toBe("block");
    await pageOff.click("#debug-toggle");
    expect(await pageOff.evaluate(() => getComputedStyle(document.querySelector(".ar-hud")!).display)).toBe("none");
    // `debug=true` is not debug mode either.
    await ctxOff.close();
    const ctxTrue = await browser!.newContext({ ignoreHTTPSErrors: true, permissions: ["camera"] });
    const pageTrue = await ctxTrue.newPage();
    await pageTrue.goto(`${base}?debug=true`, { waitUntil: "load" });
    expect(await pageTrue.evaluate(() => getComputedStyle(document.querySelector(".ar-hud")!).display)).toBe("none");
    await ctxTrue.close();

    // ?debug=1: HUD visible with the diagnostics rows.
    const ctxOn = await browser!.newContext({ ignoreHTTPSErrors: true, permissions: ["camera"] });
    const pageOn = await ctxOn.newPage();
    const onLogs: string[] = [];
    pageOn.on("console", (m: ConsoleMessage) => {
      if (m.text().startsWith("[AR]")) onLogs.push(m.text());
    });
    await pageOn.goto(`${base}?debug=1`, { waitUntil: "load" });
    await pageOn.click("#start");
    await pageOn.waitForFunction(() => window.__ar.stats().framesProcessed > 20, null, { timeout: 60_000 });
    const on = await pageOn.evaluate(() => ({
      hud: getComputedStyle(document.querySelector(".ar-hud")!).display,
      text: (document.querySelector(".ar-hud") as HTMLElement).innerText,
      toggleOn: document.getElementById("debug-toggle")!.classList.contains("on"),
    }));
    expect(on.hud).toBe("block");
    expect(on.toggleOn).toBe(true);
    expect(on.text).toMatch(/State/);
    expect(on.text).toMatch(/Motion/);
    // v16: engine per-stage and main-thread timing rows.
    expect(on.text).toMatch(/Vis\s+pyr/);
    expect(on.text).toMatch(/Main\s+grab/);
    expect(onLogs.length).toBeGreaterThan(0);
    await ctxOn.close();
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

    // Phase 4: once the world exists, a tap on the plane places the cube and
    // the session reports AR_ACTIVE. Wait for a moment where the world is ready.
    await page.waitForFunction(() => window.__ar.stats().worldReady, null, { timeout: 30_000 });
    const placement = (await page.evaluate(() => {
      const s = window.__ar.session;
      const rect = document.getElementById("video")!.getBoundingClientRect();
      const hit = s.hitTest(rect.width / 2, rect.height * 0.6);
      if (!hit) return { hit: null };
      s.placeCube(hit);
      const st = window.__ar.stats();
      const canvas = document.getElementById("three") as HTMLCanvasElement;
      return {
        hit: { x: hit.position.x, y: hit.position.y, z: hit.position.z, distance: hit.distance },
        placed: st.placedObjects,
        state: st.state,
        worldScale: st.worldScale,
        threeCanvas: canvas.width > 0 && canvas.height > 0,
        cameraY: s.threeCamera.position.y,
        objects: st.objects,
        poseAgeMs: st.poseAgeMs,
        planeAnchored: st.planeAnchored,
      };
    })) as {
      hit: { x: number; y: number; z: number; distance: number } | null;
      placed?: number;
      state?: string;
      worldScale?: number;
      threeCanvas?: boolean;
      cameraY?: number;
      objects?: { id: number; position: number[] }[];
      poseAgeMs?: number;
      planeAnchored?: boolean;
    };
    console.log("[browser-smoke] placement " + JSON.stringify(placement));
    // 修正指示書 §25–§26: while the camera keeps moving, the object's world
    // position must not change (Case A check) — sample it over a second.
    const objectTrace = (await page.evaluate(async () => {
      const out: { pos: number[]; camera: number[] | null; source: string | null; planeIn: number | null; age: number }[] = [];
      for (let i = 0; i < 10; i++) {
        const st = window.__ar.stats();
        out.push({
          pos: st.objects[0]?.position ?? [],
          camera: st.cameraWorldPosition,
          source: st.mapPose?.source ?? null,
          planeIn: st.planePose?.inlierCount ?? null,
          age: st.poseAgeMs,
        });
        await new Promise((r) => setTimeout(r, 100));
      }
      return out;
    })) as { pos: number[]; camera: number[] | null; source: string | null; planeIn: number | null; age: number }[];
    console.log("[browser-smoke] object trace " + JSON.stringify(objectTrace));
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
      mapFrame: s.mapPose?.mapFrameId ?? null,
      keyframes: s.relocalization?.keyframes ?? null,
      relocOk: s.relocalization?.successCount ?? null,
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

    // Phase 5: the test video loops every 2 s with a hard cut. Relocalization
    // must bring the camera back into the same map instead of resetting it,
    // so the map frame id never changes once the map exists.
    const mapFrames = new Set(samples.filter((s) => s.mapPose).map((s) => s.mapPose!.mapFrameId));
    expect(mapFrames.size, `map frame ids seen: ${[...mapFrames]}`).toBe(1);
    expect(samples[samples.length - 1].relocalization!.successCount).toBeGreaterThanOrEqual(1);
    expect(samples[samples.length - 1].relocalization!.keyframes).toBeGreaterThanOrEqual(1);

    // Phase 4
    expect(placement.hit, "hit test on the found plane").not.toBeNull();
    expect(Math.abs(placement.hit!.y)).toBeLessThan(1e-6);
    expect(placement.hit!.distance).toBeGreaterThan(0.1);
    expect(placement.placed).toBe(1);
    expect(placement.state).toBe("AR_ACTIVE");
    expect(placement.worldScale).toBeGreaterThan(0);
    expect(placement.threeCanvas).toBe(true);
    // The synthetic camera looks straight down at the plane 0.5 m away.
    expect(placement.cameraY).toBeGreaterThan(0.3);
    expect(placement.cameraY).toBeLessThan(0.7);

    // 修正指示書 v2: fixed map + PnP is the pose source (the experimental plane
    // estimator is off by default), the object's world position is constant
    // while the camera moves, pose age is bounded.
    expect(placement.planeAnchored).toBe(false);
    expect(placement.objects?.length).toBe(1);
    expect(placement.poseAgeMs).toBeGreaterThanOrEqual(0);
    expect(placement.poseAgeMs).toBeLessThan(1000);
    const first = objectTrace[0].pos;
    expect(first.length).toBe(3);
    for (const s of objectTrace) {
      expect(s.pos, "object world position must not change with camera motion").toEqual(first);
    }
    const cameraMoved = objectTrace.some(
      (s) => s.camera && objectTrace[0].camera && Math.hypot(s.camera[0] - objectTrace[0].camera![0], s.camera[2] - objectTrace[0].camera![2]) > 1e-4,
    );
    expect(cameraMoved, "the camera pose should change over the trace").toBe(true);
    expect(objectTrace.filter((s) => s.source === "map").length).toBeGreaterThanOrEqual(objectTrace.length / 2);
    for (const s of objectTrace) expect(["map", "propagated"]).toContain(s.source);
  });
});
