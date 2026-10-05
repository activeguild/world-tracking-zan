import { describe, expect, it } from "vitest";
import { resolveConfig } from "../../src/ar/ARConfig";
import { approximateIntrinsics } from "../../src/camera/CameraIntrinsics";
import { type Mat3, mat3Invert, mat3Multiply } from "../../src/math/Matrix";
import { angleBetween, rotationDistance, rotationToEulerDeg, rotationY } from "../../src/math/Pose";
import { createRng } from "../../src/vision/OutlierRejection";
import type { VisionInput, VisionOutput } from "../../src/vision/types";
import { VisionEngine } from "../../src/vision/VisionEngine";
import { makeTexture, warpImage } from "../helpers/synthetic";

/**
 * Phase 2 completion criterion: when the camera moves (rotate, translate),
 * the estimated pose changes continuously and matches the simulated motion.
 *
 * Image sequences are generated as homography warps of a textured plane:
 *   H_pix = K · (R + t·nᵀ/d) · K⁻¹      (planar scene, CV camera frame)
 */
const W = 640;
const H = 480;
const K = approximateIntrinsics(W, H);
const Kmat = new Float64Array([K.fx, 0, K.cx, 0, K.fy, K.cy, 0, 0, 1]);
const KInv = mat3Invert(Kmat)!;
const deg = (rad: number) => (rad * 180) / Math.PI;

function pixelHomography(r: Mat3, t: ArrayLike<number>, n: ArrayLike<number>, d: number): Mat3 {
  const h = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) h[i * 3 + j] = r[i * 3 + j] + (t[i] * n[j]) / d;
  return mat3Multiply(mat3Multiply(Kmat, h), KInv);
}

function input(frameId: number, gray: Uint8Array): VisionInput {
  return { frameId, timestamp: frameId * 33.3, width: W, height: H, gray, intrinsics: K };
}

const base = makeTexture(W, H, createRng(99), [12, 30, 70, 160]);

describe("VisionEngine pose (Phase 2)", () => {
  const config = resolveConfig();

  it("pure yaw rotation: accumulated yaw follows the simulated camera", () => {
    const engine = new VisionEngine(W, H, config, createRng(1));
    const stepDeg = 0.4;
    const frames = 24;
    const outs: VisionOutput[] = [];
    for (let f = 0; f < frames; f++) {
      // Camera rotates by +yaw about its Y axis: X_cur = R · X_ref, R = Ry(-yaw)?
      // A camera yawing left (positive rotation about Y, right-handed, Y down
      // → turning toward -X) sees the scene move... We simply define the
      // image warp with R and check the engine returns the same R.
      const R = rotationY((f * stepDeg * Math.PI) / 180);
      const img = f === 0 ? base : warpImage(base, W, H, pixelHomography(R, [0, 0, 0], [0, 0, 1], 1), 128);
      outs.push(engine.process(input(f, img)));
    }
    const yaws: number[] = [];
    for (let f = 2; f < frames; f++) {
      const p = outs[f].pose;
      expect(p, `frame ${f} pose`).not.toBeNull();
      const R = Float64Array.from(p!.rotation);
      const truth = rotationY((f * stepDeg * Math.PI) / 180);
      expect(deg(rotationDistance(R, truth)), `frame ${f} rotation error`).toBeLessThan(0.6);
      expect(["rotation", "homography"]).toContain(p!.model);
      yaws.push(rotationToEulerDeg(R).yaw);
    }
    // Monotonic, continuous growth.
    for (let i = 1; i < yaws.length; i++) {
      expect(yaws[i]).toBeGreaterThan(yaws[i - 1] - 0.2);
      expect(Math.abs(yaws[i] - yaws[i - 1])).toBeLessThan(1.5);
    }
    expect(yaws[yaws.length - 1]).toBeGreaterThan(stepDeg * (frames - 3));
  });

  it("lateral translation over a fronto-parallel plane: t direction ≈ −X, R ≈ I", () => {
    const engine = new VisionEngine(W, H, config, createRng(2));
    const frames = 20;
    const outs: VisionOutput[] = [];
    const I = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    for (let f = 0; f < frames; f++) {
      // Camera center moves +X by 0.004·f (plane at d = 1) → t = −C.
      const t = [-0.004 * f, 0, 0];
      const img = f === 0 ? base : warpImage(base, W, H, pixelHomography(I, t, [0, 0, 1], 1), 128);
      outs.push(engine.process(input(f, img)));
    }
    let confident = 0;
    for (let f = 2; f < frames; f++) {
      const p = outs[f].pose!;
      expect(p).not.toBeNull();
      expect(deg(rotationDistance(Float64Array.from(p.rotation), I)), `frame ${f}`).toBeLessThan(1.0);
      if (p.translationConfidence > 0.2) {
        confident++;
        expect(p.model).toBe("homography");
        expect(deg(angleBetween(p.translationDirection, [-1, 0, 0])), `frame ${f} t`).toBeLessThan(15);
        expect(p.planeNormal).not.toBeNull();
        expect(deg(angleBetween(p.planeNormal!, [0, 0, 1]))).toBeLessThan(15);
      }
    }
    expect(confident).toBeGreaterThanOrEqual(8);
    // Parallax grows continuously with the motion.
    const par = outs.slice(3).map((o) => o.pose!.parallaxPx);
    for (let i = 1; i < par.length; i++) expect(par[i]).toBeGreaterThan(par[i - 1] - 0.5);
  });

  it("renews the reference frame on large parallax and keeps the rotation consistent", () => {
    const cfg = resolveConfig({ pose: { maxReferenceParallaxPx: 30 } });
    const engine = new VisionEngine(W, H, cfg, createRng(3));
    const frames = 40;
    const refs = new Set<number>();
    const I = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    for (let f = 0; f < frames; f++) {
      const t = [-0.004 * f, -0.002 * f, 0];
      const img = f === 0 ? base : warpImage(base, W, H, pixelHomography(I, t, [0, 0, 1], 1), 128);
      const out = engine.process(input(f, img));
      if (out.pose) {
        refs.add(out.pose.referenceFrameId);
        expect(deg(rotationDistance(Float64Array.from(out.pose.rotation), I)), `frame ${f}`).toBeLessThan(1.5);
        expect(out.pose.parallaxPx).toBeLessThan(80);
      }
    }
    expect(refs.size).toBeGreaterThanOrEqual(2);
  });

  it("static camera: pose is identity with zero translation confidence", () => {
    const engine = new VisionEngine(W, H, config, createRng(4));
    let last: VisionOutput | null = null;
    for (let f = 0; f < 5; f++) last = engine.process(input(f, base));
    const p = last!.pose!;
    expect(p).not.toBeNull();
    expect(p.model).toBe("rotation");
    expect(deg(rotationDistance(Float64Array.from(p.rotation), new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1])))).toBeLessThan(0.1);
    expect(p.translationConfidence).toBe(0);
    expect(last!.quality.poseDelta).toBeLessThan(0.1);
  });

  it("pose estimation stays within the frame budget", () => {
    const engine = new VisionEngine(W, H, config, createRng(5));
    const I = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    const times: number[] = [];
    for (let f = 0; f < 12; f++) {
      const t = [-0.003 * f, 0, 0];
      const img = f === 0 ? base : warpImage(base, W, H, pixelHomography(I, t, [0, 0, 1], 1), 128);
      engine.process(input(f, img));
      if (f >= 2) times.push(engine.timing.pose);
    }
    times.sort((a, b) => a - b);
    const median = times[times.length >> 1];
    console.log(`[perf] pose step median ${median.toFixed(1)} ms, total ${engine.timing.total.toFixed(1)} ms`);
    expect(median).toBeLessThan(60);
  });
});
