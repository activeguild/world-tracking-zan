import { describe, expect, it } from "vitest";
import { resolveConfig } from "../../src/ar/ARConfig";
import { createRng } from "../../src/vision/OutlierRejection";
import type { VisionInput } from "../../src/vision/types";
import { VisionEngine } from "../../src/vision/VisionEngine";
import { TP_H, TP_K, TP_W, tpFloorPoint, tpGravityAt, tpPose, tpProject, tpRender } from "../helpers/twoPlaneScene";

/**
 * 修正指示書 §6 (where does the drift start?): the anchor-invariance
 * scenario showed the map scale growing by ~4.5 % during the initial slide
 * with and without BA — a bias in the map growth itself. The candidate is a
 * systematic lag of the LK tracks behind the true image motion: a track
 * that trails the corner it was detected on by a fraction of a pixel per
 * frame shows less parallax than the truth, so it is triangulated too deep,
 * and every landmark generation triangulated from biased tracks inflates the
 * scale a little more.
 *
 * Measured directly: each track's position against the true projection of
 * the floor point it was detected on, binned by track age.
 */
const W = TP_W;
const H = TP_H;
const FRAMES = 90;
function poseAt(f: number) {
  return tpPose([0.005 * f, 0.001 * f, 0.002 * f], 0.0015 * f, 0);
}
const sequence: Uint8Array[] = [];
for (let f = 0; f < FRAMES; f++) sequence.push(tpRender(poseAt(f)));
function input(f: number): VisionInput {
  return { frameId: f, timestamp: f * 33.3, width: W, height: H, gray: sequence[f], intrinsics: TP_K, gravity: tpGravityAt(poseAt(f)) };
}

export interface BiasStats {
  /** Mean signed error (track − truth) per age bin, px. */
  bins: { age: string; n: number; dx: number; dy: number; abs: number }[];
  /** Linear fit of the error along the image motion vs age: px per frame of age. */
  lagPerFrame: number;
  meanAbs: number;
}

export function measureTrackBias(config: Parameters<typeof resolveConfig>[0] = {}): BiasStats {
  const engine = new VisionEngine(W, H, resolveConfig(config), createRng(5));
  const origin = new Map<number, { f: number; X: number[] }>();
  const samples: { age: number; dx: number; dy: number; along: number }[] = [];
  for (let f = 0; f < FRAMES; f++) {
    engine.process(input(f));
    const pose = poseAt(f);
    for (const t of engine.tracksForTests()) {
      let o = origin.get(t.id);
      if (!o) {
        // Floor only (the lower half of the image is the floor in this scene).
        if (t.y < 260) continue;
        const X = tpFloorPoint(pose, t.x, t.y);
        if (!X) continue;
        o = { f, X };
        origin.set(t.id, o);
        continue;
      }
      const p = tpProject(pose, o.X);
      if (!p) continue;
      // Image motion direction of this point since detection (truth).
      const p0 = tpProject(poseAt(o.f), o.X)!;
      const mx = p[0] - p0[0], my = p[1] - p0[1];
      const ml = Math.hypot(mx, my);
      const dx = t.x - p[0], dy = t.y - p[1];
      samples.push({ age: f - o.f, dx, dy, along: ml > 1e-6 ? -(dx * mx + dy * my) / ml : 0 });
    }
  }
  const edges = [1, 5, 10, 20, 40, 80, 1000];
  const bins: BiasStats["bins"] = [];
  for (let i = 0; i + 1 < edges.length; i++) {
    const s = samples.filter((x) => x.age >= edges[i] && x.age < edges[i + 1]);
    if (!s.length) continue;
    const m = (k: "dx" | "dy") => s.reduce((a, c) => a + c[k], 0) / s.length;
    bins.push({ age: `${edges[i]}–${edges[i + 1] - 1}`, n: s.length, dx: m("dx"), dy: m("dy"), abs: s.reduce((a, c) => a + Math.hypot(c.dx, c.dy), 0) / s.length });
  }
  // Lag = error component against the motion direction (positive = trailing), fitted linearly in age.
  let sa = 0, sl = 0, saa = 0, sal = 0;
  for (const x of samples) {
    sa += x.age;
    sl += x.along;
    saa += x.age * x.age;
    sal += x.age * x.along;
  }
  const n = samples.length;
  const lagPerFrame = (n * sal - sa * sl) / Math.max(1e-9, n * saa - sa * sa);
  return { bins, lagPerFrame, meanAbs: samples.reduce((a, c) => a + Math.hypot(c.dx, c.dy), 0) / n };
}

describe("LK track bias against the true image motion (指示書 §6)", () => {
  it("reports the signed track error by age and the lag per frame", () => {
    const s = measureTrackBias();
    console.log(`[trackbias] lag ${(s.lagPerFrame * 1000).toFixed(2)} mpx/frame of age, mean |err| ${s.meanAbs.toFixed(3)} px; by age: ${s.bins.map((b) => `${b.age}f n${b.n} (${b.dx.toFixed(3)}, ${b.dy.toFixed(3)}) |${b.abs.toFixed(3)}|`).join("  ")}`);
    expect(s.bins.length).toBeGreaterThan(3);
  });
});
