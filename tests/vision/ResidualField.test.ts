import { describe, expect, it } from "vitest";
import { resolveConfig } from "../../src/ar/ARConfig";
import { approximateIntrinsics } from "../../src/camera/CameraIntrinsics";
import { type Mat3, mat3Invert, mat3Multiply, mat3TransformPoint } from "../../src/math/Matrix";
import { createRng } from "../../src/vision/OutlierRejection";
import { RESIDUAL_GROUP_ADJUSTED, RESIDUAL_GROUP_PLANE, fitResidualField, type ResidualField } from "../../src/vision/ResidualField";
import { cameraCenterOf } from "../../src/math/Pose";
import type { VisionInput, VisionOutput } from "../../src/vision/types";
import { VisionEngine } from "../../src/vision/VisionEngine";
import { makeTexture, sampleBilinear } from "../helpers/synthetic";
import { TP_FOV, TP_H, TP_W, tpGravity, tpPose, tpRender } from "../helpers/twoPlaneScene";

/**
 * v22 diagnostics: the PnP residual *vectors* fitted with an affine field.
 * On Android every magnitude split (region / kind / parallax / age / BA mode
 * / link source) came out uniform at 2–3 px; the direction of the residuals
 * tells noise (random) from a camera-model or pose bias (systematic) and
 * from a map split (groups pulling apart). The fit must read each of those
 * patterns off a synthetic field, and must stay quiet on a clean engine run.
 */
const W = 640;
const H = 480;
const HALF_DIAG = Math.hypot(W, H) / 2;

function field(rng: () => number, n: number, make: (x: number, y: number, i: number) => [number, number, number]) {
  const px = new Float64Array(n), py = new Float64Array(n), dx = new Float64Array(n), dy = new Float64Array(n);
  const g = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    px[i] = (rng() - 0.5) * W;
    py[i] = (rng() - 0.5) * H;
    const [ex, ey, group] = make(px[i], py[i], i);
    dx[i] = ex;
    dy[i] = ey;
    g[i] = group;
  }
  return fitResidualField(px, py, dx, dy, g, n, HALF_DIAG);
}
const gauss = (rng: () => number) => {
  let s = 0;
  for (let k = 0; k < 12; k++) s += rng();
  return s - 6;
};
const fmt = (r: ResidualField) =>
  `n ${r.count} mean (${r.meanX.toFixed(2)}, ${r.meanY.toFixed(2)}) scale ${(r.scale * 100).toFixed(3)}% (${r.scaleEdgePx.toFixed(2)} px@edge) rot ${r.rotationDeg.toFixed(3)}° shear ${(r.shearX * 100).toFixed(3)}/${(r.shearY * 100).toFixed(3)}% rms ${r.rmsPx.toFixed(2)} random ${r.randomPx.toFixed(2)} systematic ${(r.systematic * 100).toFixed(0)}%`;

describe("residual vector field (v22 diagnostics)", () => {
  it("pure noise: no shift, scale or rotation, nothing systematic, random ≈ rms", () => {
    const rng = createRng(1);
    const r = field(rng, 200, () => [2 * gauss(rng), 2 * gauss(rng), 0]);
    console.log(`[resfield] noise: ${fmt(r)}`);
    expect(Math.hypot(r.meanX, r.meanY)).toBeLessThan(0.5);
    expect(Math.abs(r.scaleEdgePx)).toBeLessThan(0.6);
    expect(Math.abs(r.rotationDeg)).toBeLessThan(0.1);
    expect(r.systematic).toBeLessThan(0.1);
    expect(r.randomPx).toBeGreaterThan(r.rmsPx * 0.9);
  });

  it("a common shift (pose bias / frame out of step) is read as the mean and is systematic", () => {
    const rng = createRng(2);
    const r = field(rng, 200, () => [2 + 0.3 * gauss(rng), -1 + 0.3 * gauss(rng), 0]);
    console.log(`[resfield] shift: ${fmt(r)}`);
    expect(r.meanX).toBeCloseTo(2, 0.1);
    expect(r.meanY).toBeCloseTo(-1, 0.1);
    expect(Math.abs(r.scaleEdgePx)).toBeLessThan(0.3);
    expect(r.systematic).toBeGreaterThan(0.9);
    expect(r.randomPx).toBeLessThan(0.5); // 2-D rms of σ = 0.3 per axis ≈ 0.42
  });

  it("a radial scale (focal length / depth scale off by 1%) is read as scale, not rotation; shear reads fx ≠ fy", () => {
    const rng = createRng(3);
    const r = field(rng, 300, (x, y) => [0.01 * x + 0.3 * gauss(rng), 0.01 * y + 0.3 * gauss(rng), 0]);
    console.log(`[resfield] scale: ${fmt(r)}`);
    expect(r.scale).toBeCloseTo(0.01, 3);
    expect(r.scaleEdgePx).toBeCloseTo(4, 0);
    expect(Math.abs(r.rotationDeg)).toBeLessThan(0.05);
    expect(Math.abs(r.shearX)).toBeLessThan(0.001);
    expect(r.systematic).toBeGreaterThan(0.8);
    // Only fx off by 1 %: half scale, half shear.
    const rng2 = createRng(4);
    const s = field(rng2, 300, (x) => [0.01 * x + 0.3 * gauss(rng2), 0.3 * gauss(rng2), 0]);
    console.log(`[resfield] fx only: ${fmt(s)}`);
    expect(s.scale).toBeCloseTo(0.005, 3);
    expect(s.shearX).toBeCloseTo(0.005, 3);
  });

  it("a rotation about the principal point (roll bias) is read as rotation with no scale", () => {
    const rng = createRng(5);
    const a = (0.5 * Math.PI) / 180;
    const r = field(rng, 300, (x, y) => [x * Math.cos(a) - y * Math.sin(a) - x + 0.2 * gauss(rng), x * Math.sin(a) + y * Math.cos(a) - y + 0.2 * gauss(rng), 0]);
    console.log(`[resfield] rotation: ${fmt(r)}`);
    expect(r.rotationDeg).toBeCloseTo(0.5, 1);
    expect(Math.abs(r.scaleEdgePx)).toBeLessThan(0.3);
    expect(r.systematic).toBeGreaterThan(0.8);
  });

  it("two landmark groups pulling in opposite directions: the global mean is zero, the group means are not, and the affine field explains little", () => {
    const rng = createRng(6);
    const r = field(rng, 300, (_x, _y, i) => {
      const adj = i % 2 === 0;
      const plane = i % 3 === 0;
      return [(adj ? 1.5 : -1.5) + 0.3 * gauss(rng), 0.3 * gauss(rng), (adj ? RESIDUAL_GROUP_ADJUSTED : 0) | (plane ? RESIDUAL_GROUP_PLANE : 0)];
    });
    console.log(`[resfield] split: ${fmt(r)} adj (${r.adjustedX.toFixed(2)}, ${r.adjustedY.toFixed(2)}) ×${r.adjustedCount} none (${r.untouchedX.toFixed(2)}, ${r.untouchedY.toFixed(2)}) ×${r.untouchedCount}`);
    expect(Math.abs(r.meanX)).toBeLessThan(0.2);
    expect(r.adjustedX).toBeCloseTo(1.5, 0);
    expect(r.untouchedX).toBeCloseTo(-1.5, 0);
    expect(r.adjustedCount + r.untouchedCount).toBe(300);
    expect(r.planeCount + r.otherCount).toBe(300);
    // The split is not an affine pattern: it reads as "random" to the field fit.
    expect(r.systematic).toBeLessThan(0.1);
    expect(r.randomPx).toBeGreaterThan(1.4);
  });

  it("v23: depth errors line the residuals up with the parallax direction; isotropic noise does not", () => {
    // Random parallax directions with 40–80 px sensitivity; residual = ±3 %
    // depth error × sensitivity along the direction, plus 0.3 px noise.
    const rng = createRng(8);
    const n = 300;
    const px = new Float64Array(n), py = new Float64Array(n), dx = new Float64Array(n), dy = new Float64Array(n);
    const ex = new Float64Array(n), ey = new Float64Array(n), sens = new Float64Array(n);
    const g = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      px[i] = (rng() - 0.5) * W;
      py[i] = (rng() - 0.5) * H;
      const a = rng() * Math.PI * 2;
      ex[i] = Math.cos(a);
      ey[i] = Math.sin(a);
      sens[i] = 40 + rng() * 40;
      g[i] = i % 3 === 0 ? RESIDUAL_GROUP_PLANE : 0;
      const depthErr = (i % 3 === 0 ? 0.03 : -0.03) + 0.005 * gauss(rng); // floor too close, the rest too far
      dx[i] = depthErr * sens[i] * ex[i] + 0.3 * gauss(rng);
      dy[i] = depthErr * sens[i] * ey[i] + 0.3 * gauss(rng);
    }
    const r = fitResidualField(px, py, dx, dy, g, n, HALF_DIAG, { ex, ey, sensitivity: sens });
    console.log(
      `[resfield] depth errors: ${fmt(r)} | ∥ ${r.parallelRmsPx.toFixed(2)} ⊥ ${r.perpRmsPx.toFixed(2)} ×${r.parallaxRatio.toFixed(1)} depth err med ${(r.depthErrorMedian * 100).toFixed(1)}% bias ${(r.depthErrorBias * 100).toFixed(1)}% plane ${(r.planeDepthBias * 100).toFixed(1)}% other ${(r.otherDepthBias * 100).toFixed(1)}% (n ${r.parallaxCount}, sens ${r.depthSensitivityMedianPx.toFixed(0)} px)`,
    );
    // Random directions: the affine field explains nothing, the parallax split everything.
    expect(r.systematic).toBeLessThan(0.1);
    expect(r.parallaxRatio).toBeGreaterThan(3);
    expect(r.depthErrorMedian).toBeCloseTo(0.03, 2);
    expect(r.planeDepthBias).toBeCloseTo(0.03, 2);
    expect(r.otherDepthBias).toBeCloseTo(-0.03, 2);
    expect(r.parallaxCount).toBe(n);
    // Isotropic noise with the same geometry: ratio ≈ 1, no bias, small implied error.
    const nx = new Float64Array(n), ny = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      nx[i] = 2 * gauss(rng);
      ny[i] = 2 * gauss(rng);
    }
    const noise = fitResidualField(px, py, nx, ny, g, n, HALF_DIAG, { ex, ey, sensitivity: sens });
    console.log(`[resfield] noise, parallax split: ∥ ${noise.parallelRmsPx.toFixed(2)} ⊥ ${noise.perpRmsPx.toFixed(2)} ×${noise.parallaxRatio.toFixed(2)} depth err med ${(noise.depthErrorMedian * 100).toFixed(1)}% bias ${(noise.depthErrorBias * 100).toFixed(2)}%`);
    expect(noise.parallaxRatio).toBeGreaterThan(0.8);
    expect(noise.parallaxRatio).toBeLessThan(1.25);
    expect(Math.abs(noise.depthErrorBias)).toBeLessThan(0.01);
    // Landmarks below the sensitivity floor are left out.
    const weak = Float64Array.from(sens);
    for (let i = 0; i < n; i += 2) weak[i] = 1;
    const half = fitResidualField(px, py, dx, dy, g, n, HALF_DIAG, { ex, ey, sensitivity: weak });
    expect(half.parallaxCount).toBe(n / 2);
  });

  it("fewer than six inliers: mean only", () => {
    const rng = createRng(7);
    const r = field(rng, 4, () => [1, 1, 0]);
    expect(r.count).toBe(4);
    expect(r.meanX).toBeCloseTo(1, 6);
    expect(r.scale).toBe(0);
    expect(r.randomPx).toBeCloseTo(r.rmsPx, 6);
  });
});

/** Floor slide of the breakdown test, with an optional radial lens on the observed image. */
const K = approximateIntrinsics(W, H, 66);
const Kmat = new Float64Array([K.fx, 0, K.cx, 0, K.fy, K.cy, 0, 0, 1]);
const KInv = mat3Invert(Kmat)!;
const I: Mat3 = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
const base = makeTexture(W, H, createRng(4141), [12, 30, 70, 160]);
const n = [0, Math.SQRT1_2, Math.SQRT1_2];
const gravity = [0, Math.SQRT1_2, Math.SQRT1_2];
function pixelHomography(r: Mat3, t: ArrayLike<number>, nn: ArrayLike<number>, d: number): Mat3 {
  const h = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) h[i * 3 + j] = r[i * 3 + j] + (t[i] * nn[j]) / d;
  return mat3Multiply(mat3Multiply(Kmat, h), KInv);
}
function render(f: number, k1: number): Uint8Array {
  const t = [-0.004 * f, -0.001 * f, 0];
  const Hf = f === 0 ? I : pixelHomography(I, t, n, 1);
  const inv = mat3Invert(Hf)!;
  const out = new Uint8Array(W * H);
  const pt = new Float64Array(2);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const dx = x - K.cx, dy = y - K.cy;
      const s = 1 + (k1 * (dx * dx + dy * dy)) / (HALF_DIAG * HALF_DIAG);
      if (!mat3TransformPoint(inv, K.cx + dx * s, K.cy + dy * s, pt)) {
        out[y * W + x] = 128;
        continue;
      }
      out[y * W + x] = Math.round(sampleBilinear(base, W, H, pt[0], pt[1], 128));
    }
  }
  return out;
}
function runEngine(k1: number) {
  const engine = new VisionEngine(W, H, resolveConfig(), createRng(9));
  const fields: ResidualField[] = [];
  for (let f = 0; f < 60; f++) {
    const input: VisionInput = { frameId: f, timestamp: f * 33.3, width: W, height: H, gray: render(f, k1), intrinsics: K, gravity };
    const o: VisionOutput = engine.process(input);
    if (o.mapPose && o.mapPose.framesSinceTracked === 0 && f >= 40) fields.push(o.mapPose.reprojection.field);
  }
  expect(fields.length).toBeGreaterThan(10);
  const avg = (sel: (r: ResidualField) => number) => fields.reduce((a, r) => a + sel(r), 0) / fields.length;
  return {
    count: avg((r) => r.count),
    mean: avg((r) => Math.hypot(r.meanX, r.meanY)),
    scaleEdgePx: avg((r) => Math.abs(r.scaleEdgePx)),
    rotationDeg: avg((r) => Math.abs(r.rotationDeg)),
    rmsPx: avg((r) => r.rmsPx),
    randomPx: avg((r) => r.randomPx),
    systematic: avg((r) => r.systematic),
    gap: avg((r) => (r.planeCount && r.otherCount ? Math.hypot(r.planeX - r.otherX, r.planeY - r.otherY) : 0)),
  };
}

describe("residual vector field in the engine", () => {
  it("is quiet on a pinhole-consistent floor and shows a radial pattern on a distorted lens", () => {
    const pinhole = runEngine(0);
    const distorted = runEngine(-0.06);
    const f = (r: ReturnType<typeof runEngine>) =>
      `inliers ${r.count.toFixed(0)} mean ${r.mean.toFixed(2)} px scale ${r.scaleEdgePx.toFixed(2)} px@edge rot ${r.rotationDeg.toFixed(3)}° rms ${r.rmsPx.toFixed(2)} random ${r.randomPx.toFixed(2)} systematic ${(r.systematic * 100).toFixed(0)}% plane/other gap ${r.gap.toFixed(2)} px`;
    console.log(`[resfield] engine pinhole:   ${f(pinhole)}\n[resfield] engine distorted: ${f(distorted)}`);
    expect(pinhole.count).toBeGreaterThan(50);
    expect(pinhole.mean).toBeLessThan(0.1);
    expect(pinhole.scaleEdgePx).toBeLessThan(0.3);
    expect(pinhole.rotationDeg).toBeLessThan(0.05);
    expect(distorted.scaleEdgePx).toBeGreaterThan(pinhole.scaleEdgePx * 2);
    expect(distorted.systematic).toBeGreaterThan(pinhole.systematic);
  });

  it("focal length off by 10 %: the map absorbs it during the slide, the field shows the model error only while tilting", () => {
    // The FocalEstimate scene: floor + wall rendered at 66°, 90 frames of
    // sideways slide, then a ±25° pitch swing. Bundle adjustment off.
    const SLIDE = 90, FRAMES = 210;
    const poseAt = (f: number) => {
      const s = Math.min(f, SLIDE);
      const pitch = f > SLIDE ? ((25 * Math.PI) / 180) * Math.sin(((f - SLIDE) / (FRAMES - SLIDE)) * 2 * Math.PI) : 0;
      return tpPose([0.005 * s, 0.001 * s, 0.002 * s], 0.0015 * s, pitch);
    };
    const seq: Uint8Array[] = [];
    for (let f = 0; f < FRAMES; f++) seq.push(tpRender(poseAt(f)));
    const run = (fov: number) => {
      const k = approximateIntrinsics(TP_W, TP_H, fov);
      // A diagnostic of a *deliberately* perturbed map: no BA and no init stage may repair it meanwhile.
      const engine = new VisionEngine(TP_W, TP_H, resolveConfig({ bundleAdjustment: { enabled: false }, landmarks: { initRefineStages: 0 } }), createRng(11));
      const slide: ResidualField[] = [], tilt: ResidualField[] = [];
      for (let f = 0; f < FRAMES; f++) {
        const mp = engine.process({ frameId: f, timestamp: f * 33.3, width: TP_W, height: TP_H, gray: seq[f], intrinsics: k, gravity: tpGravity }).mapPose;
        if (!mp || mp.framesSinceTracked !== 0) continue;
        (f < SLIDE ? slide : tilt).push(mp.reprojection.field);
      }
      const summarize = (fs: ResidualField[]) => ({
        n: fs.length,
        mean: fs.reduce((a, r) => a + Math.hypot(r.meanX, r.meanY), 0) / fs.length,
        scaleEdgePx: fs.reduce((a, r) => a + Math.abs(r.scaleEdgePx), 0) / fs.length,
        rotationDeg: fs.reduce((a, r) => a + Math.abs(r.rotationDeg), 0) / fs.length,
        rmsPx: fs.reduce((a, r) => a + r.rmsPx, 0) / fs.length,
        systematic: fs.reduce((a, r) => a + r.systematic, 0) / fs.length,
        gap: fs.reduce((a, r) => a + (r.planeCount && r.otherCount ? Math.hypot(r.planeX - r.otherX, r.planeY - r.otherY) : 0), 0) / fs.length,
        maxSystematic: Math.max(...fs.map((r) => r.systematic)),
      });
      return { slide: summarize(slide), tilt: summarize(tilt) };
    };
    const right = run(TP_FOV);
    const wrong = run(TP_FOV + 6);
    const f = (s: ReturnType<typeof run>["slide"]) =>
      `n ${s.n} mean ${s.mean.toFixed(2)} px scale ${s.scaleEdgePx.toFixed(2)} px@edge rot ${s.rotationDeg.toFixed(3)}° rms ${s.rmsPx.toFixed(2)} systematic ${(s.systematic * 100).toFixed(0)}% (max ${(s.maxSystematic * 100).toFixed(0)}%) plane/other gap ${s.gap.toFixed(2)} px`;
    console.log(`[resfield] fov 66 slide: ${f(right.slide)}\n[resfield] fov 66 tilt:  ${f(right.tilt)}\n[resfield] fov 72 slide: ${f(wrong.slide)}\n[resfield] fov 72 tilt:  ${f(wrong.tilt)}`);
    expect(right.slide.n).toBeGreaterThan(30);
    expect(wrong.tilt.n).toBeGreaterThan(60);
    // Right model: nothing systematic in either phase.
    expect(right.tilt.systematic).toBeLessThan(0.3);
    // Wrong model: the field carries the error while tilting, more than with the right model.
    expect(wrong.tilt.systematic).toBeGreaterThan(right.tilt.systematic * 1.5);
    expect(wrong.tilt.rmsPx).toBeGreaterThan(right.tilt.rmsPx * 1.5);
  });

  it("v23: landmark depths perturbed by ±3 % along their anchor rays read as parallax-aligned residuals with the right depth error", () => {
    // Floor + wall, 110 frames of slide (so the current view has a baseline
    // to the anchor views), BA off so nothing repairs the map.
    const FRAMES = 140;
    const poseAt = (f: number) => tpPose([0.005 * f, 0.001 * f, 0.002 * f], 0.0015 * f, 0);
    const seq: Uint8Array[] = [];
    for (let f = 0; f < FRAMES; f++) seq.push(tpRender(poseAt(f)));
    const run = (perturb: number) => {
      // A diagnostic of a *deliberately* perturbed map: no BA and no init stage may repair it meanwhile.
      const engine = new VisionEngine(TP_W, TP_H, resolveConfig({ bundleAdjustment: { enabled: false }, landmarks: { initRefineStages: 0 } }), createRng(11));
      const before: ResidualField[] = [], after: ResidualField[] = [];
      for (let f = 0; f < FRAMES; f++) {
        const mp = engine.process({ frameId: f, timestamp: f * 33.3, width: TP_W, height: TP_H, gray: seq[f], intrinsics: approximateIntrinsics(TP_W, TP_H, TP_FOV), gravity: tpGravity }).mapPose;
        if (f === 100 && perturb > 0) {
          // Move every anchored landmark along the ray from its anchor camera.
          const rng = createRng(77);
          let moved = 0;
          for (const lm of engine.landmarkMap.values()) {
            if (!lm.anchorPose) continue;
            const c = cameraCenterOf(lm.anchorPose);
            const s = 1 + (rng() < 0.5 ? -perturb : perturb);
            for (let a = 0; a < 3; a++) lm.position[a] = c[a] + (lm.position[a] - c[a]) * s;
            moved++;
          }
          expect(moved).toBeGreaterThan(100);
        }
        if (!mp || mp.framesSinceTracked !== 0) continue;
        (f < 100 ? before : after).push(mp.reprojection.field);
      }
      const summarize = (fs: ResidualField[]) => ({
        n: fs.length,
        ratio: fs.reduce((a, r) => a + r.parallaxRatio, 0) / fs.length,
        depthErr: fs.reduce((a, r) => a + r.depthErrorMedian, 0) / fs.length,
        count: fs.reduce((a, r) => a + r.parallaxCount, 0) / fs.length,
        sens: fs.reduce((a, r) => a + r.depthSensitivityMedianPx, 0) / fs.length,
        rms: fs.reduce((a, r) => a + r.rmsPx, 0) / fs.length,
        systematic: fs.reduce((a, r) => a + r.systematic, 0) / fs.length,
      });
      return { before: summarize(before.slice(60)), after: summarize(after.slice(2)) };
    };
    const clean = run(0);
    const bad = run(0.03);
    const f = (s: ReturnType<typeof run>["before"]) => `n ${s.n} rms ${s.rms.toFixed(2)} px sys ${(s.systematic * 100).toFixed(0)}% ∥/⊥ ×${s.ratio.toFixed(2)} depth err med ${(s.depthErr * 100).toFixed(2)}% (${s.count.toFixed(0)} landmarks with sens ${s.sens.toFixed(0)} px)`;
    console.log(`[resfield] clean before: ${f(clean.before)}\n[resfield] clean after:  ${f(clean.after)}\n[resfield] ±3% before:   ${f(bad.before)}\n[resfield] ±3% after:    ${f(bad.after)}`);
    expect(bad.after.n).toBeGreaterThan(20);
    expect(clean.after.count).toBeGreaterThan(30);
    // Clean map: even a correct synthetic map carries its small triangulation
    // error mostly in depth (×1.7–1.8 along the parallax direction, 0.2–0.3 %
    // implied), so ×1 is the signature of pure observation noise, not of a
    // healthy map.
    expect(clean.after.ratio).toBeLessThan(2.2);
    expect(clean.after.depthErr).toBeLessThan(0.005);
    // Perturbed map: the residual lines up with the parallax direction and the
    // implied depth error rises several-fold. It reads below the injected 3 %
    // because only PnP inliers (gate 6 px, sensitivity ~170 px → the worst
    // landmarks fall out) enter the statistic and the compromise pose absorbs
    // part of it.
    expect(bad.after.ratio).toBeGreaterThan(2.5);
    expect(bad.after.ratio).toBeGreaterThan(clean.after.ratio * 2);
    expect(bad.after.depthErr).toBeGreaterThan(clean.after.depthErr * 3);
    expect(bad.after.depthErr).toBeGreaterThan(0.01);
    expect(bad.after.depthErr).toBeLessThan(0.045);
    // Before the perturbation both runs agree (same engine, same seed).
    expect(Math.abs(bad.before.rms - clean.before.rms)).toBeLessThan(1e-9);
  });
});
