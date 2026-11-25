import { describe, expect, it } from "vitest";
import { GyroIntegrator, conjugateRotation, deviceToCameraMatrix } from "../../src/sensors/GyroIntegrator";
import { angleDeg, rotationVectorDeg } from "../../src/ar/GyroDiagnostics";

const DEG = Math.PI / 180;

describe("GyroIntegrator", () => {
  it("integrates a constant rate and interpolates between samples", () => {
    const g = new GyroIntegrator({ correctBias: false });
    for (let i = 0; i <= 120; i++) g.add(1000 + (i * 1000) / 60, 30 * DEG, 0, 0);
    const r = g.rotationBetween(1000, 2000)!;
    expect(angleDeg(r)).toBeCloseTo(30, 6);
    const v = rotationVectorDeg(r);
    expect(v[0]).toBeCloseTo(30, 6);
    expect(Math.abs(v[1]) + Math.abs(v[2])).toBeLessThan(1e-9);
    // Half way between two samples.
    const h = g.rotationBetween(1000, 1000 + 1000 / 120)!;
    expect(angleDeg(h)).toBeCloseTo(0.25, 6);
    // Composition: (t0→t1) then (t1→t2) = (t0→t2).
    const a = g.rotationBetween(1100, 1500)!.slice();
    const b = g.rotationBetween(1500, 1900)!.slice();
    expect(angleDeg(a) + angleDeg(b)).toBeCloseTo(angleDeg(g.rotationBetween(1100, 1900)!), 6);
    expect(g.sampleHz).toBeGreaterThan(55);
    expect(g.sampleHz).toBeLessThan(65);
  });

  it("refuses times outside the ring and across a gap", () => {
    const g = new GyroIntegrator({ correctBias: false });
    for (let i = 0; i <= 60; i++) g.add(i * 16, 0, 10 * DEG, 0);
    expect(g.rotationBetween(-50, 500)).toBeNull();
    expect(g.rotationBetween(100, 960 + 200)).toBeNull();
    expect(g.rotationBetween(100, 960 + 30)).not.toBeNull();
    // A one second gap (page hidden): nothing is known about it.
    for (let i = 0; i <= 60; i++) g.add(2000 + i * 16, 0, 10 * DEG, 0);
    expect(g.rotationBetween(500, 2500)).toBeNull();
    expect(angleDeg(g.rotationBetween(2100, 2600)!)).toBeCloseTo(5, 6);
    // Non-finite and out-of-order samples are ignored.
    const n = g.sampleCount;
    g.add(NaN, 0, 0, 0);
    g.add(2100, 0, 0, 0);
    expect(g.sampleCount).toBe(n);
  });

  it("estimates the bias while held still and removes it", () => {
    const bias = [0.4 * DEG, -0.3 * DEG, 0.2 * DEG];
    const g = new GyroIntegrator();
    let t = 0;
    let seed = 7;
    const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff - 0.5) * 2;
    // 2 s hold with zero-mean tremor (±0.5°/s).
    for (let i = 0; i < 120; i++, t += 1000 / 60) g.add(t, bias[0] + rnd() * 0.5 * DEG, bias[1] + rnd() * 0.5 * DEG, bias[2] + rnd() * 0.5 * DEG);
    expect(Math.hypot(g.bias[0] - bias[0], g.bias[1] - bias[1], g.bias[2] - bias[2]) / DEG).toBeLessThan(0.08);
    // 3 s of pure bias afterwards: the corrected integral stays near zero (raw: 1.6°).
    const t0 = t;
    for (let i = 0; i < 180; i++, t += 1000 / 60) g.add(t, bias[0], bias[1], bias[2]);
    expect(angleDeg(g.rotationBetween(t0, t - 20)!)).toBeLessThan(0.25);
    // A real rotation is not absorbed into the bias.
    const before = Float64Array.from(g.bias);
    for (let i = 0; i < 60; i++, t += 1000 / 60) g.add(t, 40 * DEG, 0, 0);
    expect(Math.hypot(g.bias[0] - before[0], g.bias[1] - before[1], g.bias[2] - before[2])).toBe(0);
    expect(g.maxRateBetween(t - 500, t) / DEG).toBeGreaterThan(39);
  });

  it("maps device axes to camera axes (portrait and landscape)", () => {
    const m = deviceToCameraMatrix(0);
    // Proper rotation.
    const det =
      m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
    expect(det).toBeCloseTo(1, 12);
    const about = (wx: number, wy: number, wz: number, angle = 0) => {
      const g = new GyroIntegrator({ correctBias: false });
      for (let i = 0; i <= 60; i++) g.add(i * 16, wx * DEG, wy * DEG, wz * DEG);
      return rotationVectorDeg(conjugateRotation(deviceToCameraMatrix(angle), g.rotationBetween(0, 960)!));
    };
    // Device X (right) = camera X; device Y (up) = −camera Y; device Z (out of the screen) = −camera Z.
    const x = about(10, 0, 0);
    expect(x[0]).toBeCloseTo(9.6, 6);
    const y = about(0, 10, 0);
    expect(y[1]).toBeCloseTo(-9.6, 6);
    const z = about(0, 0, 10);
    expect(z[2]).toBeCloseTo(-9.6, 6);
    // Landscape (90°): the device X axis is the video's vertical axis.
    const l = about(10, 0, 0, 90);
    expect(Math.abs(l[1])).toBeCloseTo(9.6, 6);
    expect(Math.abs(l[0])).toBeLessThan(1e-9);
  });

  it("reports the bias without applying it when correction is off (recording 46)", () => {
    const on = new GyroIntegrator();
    const off = new GyroIntegrator({ correctBias: false });
    // A slow steady roll of 0.7°/s for 3 s: under the stillness rate, so it is learned as "bias".
    for (let i = 0; i <= 180; i++) {
      on.add(i * (1000 / 60), 0, 0, 0.7 * DEG);
      off.add(i * (1000 / 60), 0, 0, 0.7 * DEG);
    }
    expect(Math.hypot(off.bias[0], off.bias[1], off.bias[2]) / DEG).toBeCloseTo(0.7, 2);
    // Applied, the real rotation of the last 2 s disappears; reported only, it is measured.
    expect(angleDeg(on.rotationBetween(1000, 2900)!)).toBeLessThan(0.1);
    expect(angleDeg(off.rotationBetween(1000, 2900)!)).toBeCloseTo(0.7 * 1.9, 2);
  });
});
