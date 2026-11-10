/**
 * Vision ↔ gyro rotation diagnostic (Phase 6 step 0).
 *
 * One physical motion drives both sensors: the device orientation Q(t)
 * (device → world) is integrated from body rates at 1 kHz; the gyro sees the
 * rates at 60 Hz, vision sees R(t) = M · Q(t)ᵀ (world → camera) at 30 fps.
 * With consistent conventions the two agree; every test then breaks one
 * thing and checks the diagnostic reports exactly that.
 */
import { describe, expect, it } from "vitest";
import { GyroComparer, angleDeg, type GyroDiagnostics, type GyroDiagnosticsConfig } from "../../src/ar/GyroDiagnostics";
import { GyroProvider, rotationRateToDevice } from "../../src/sensors/GyroProvider";
import { deviceToCameraMatrix } from "../../src/sensors/GyroIntegrator";
import { rotationAxisAngle, rotationZ } from "../../src/math/Pose";
import { mat3Multiply } from "../../src/math/Matrix";

const DEG = Math.PI / 180;
const CFG: GyroDiagnosticsConfig = {
  windowsMs: [1000, 3000],
  stillRateDegPerSec: 3,
  stillMinMs: 250,
  segmentMinDeg: 3,
  lagMaxMs: 200,
  lagStepMs: 10,
  lagWindowMs: 2000,
  lagMinMotionDeg: 5,
  lagEveryFrames: 1,
};

/** Smooth 0 → 1 → 0 bump of unit area over [a, b]. */
function bump(t: number, a: number, b: number): number {
  if (t <= a || t >= b) return 0;
  const u = (t - a) / (b - a);
  return (1 - Math.cos(2 * Math.PI * u)) / (b - a);
}

/** Body rates (rad/s, device frame): hold, pitch +15° with some yaw, hold, roll 8°, hold, pitch back, hold. */
function ratesAt(tMs: number): [number, number, number] {
  const t = tMs / 1000;
  const wx = 15 * DEG * bump(t, 0.6, 1.6) - 15 * DEG * bump(t, 4.4, 5.4);
  const wy = 6 * DEG * bump(t, 0.7, 1.5);
  const wz = 8 * DEG * bump(t, 2.4, 3.2);
  return [wx, wy, wz];
}

function transpose(m: ArrayLike<number>): Float64Array {
  return new Float64Array([m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]);
}

interface Sim {
  /** Vision rotation (world → camera) at time t (ms). */
  visionAt(tMs: number): Float64Array;
  endMs: number;
  gyro: GyroProvider;
}

function simulate(durationMs = 6400, gyroBias: [number, number, number] = [0, 0, 0], report: (x: number, y: number, z: number) => [number, number, number] = (x, y, z) => [x, y, z], leadMs = 0): Sim {
  durationMs += leadMs;
  const rates = (ms: number) => ratesAt(ms - leadMs);
  const M = deviceToCameraMatrix(0);
  const q: Float64Array[] = [];
  let Q = rotationAxisAngle([0.3, 1, 0.2], 0.4); // arbitrary start orientation
  const gyro = new GyroProvider();
  for (let ms = 0; ms <= durationMs; ms++) {
    q.push(Q);
    const [wx, wy, wz] = rates(ms + 0.5);
    const th = Math.hypot(wx, wy, wz) * 0.001;
    if (th > 0) Q = mat3Multiply(Q, rotationAxisAngle([wx, wy, wz], th));
  }
  // 60 Hz gyro: mean rate over the sample interval (what a real gyro integrates).
  const step = 1000 / 60;
  for (let t = 0; t <= durationMs; t += step) {
    let sx = 0, sy = 0, sz = 0, n = 0;
    for (let ms = Math.max(0, t - step); ms < t; ms += 1, n++) {
      const [wx, wy, wz] = rates(ms + 0.5);
      sx += wx; sy += wy; sz += wz;
    }
    n = Math.max(1, n);
    const [rx, ry, rz] = report(sx / n + gyroBias[0], sy / n + gyroBias[1], sz / n + gyroBias[2]);
    gyro.integrator.add(t, rx, ry, rz);
  }
  return {
    visionAt(tMs: number) {
      const i = Math.min(q.length - 1, Math.max(0, Math.round(tMs)));
      return mat3Multiply(M, transpose(q[i]));
    },
    gyro,
    endMs: durationMs - 100,
  };
}

function run(
  sim: Sim,
  opts: { stampOffsetMs?: number; distort?: (r: Float64Array, tMs: number, frame: number, cmp: GyroComparer) => Float64Array; tracked?: (frame: number) => boolean; mapId?: (frame: number) => number } = {},
): { records: GyroDiagnostics[]; cmp: GyroComparer } {
  const cmp = new GyroComparer(CFG);
  const records: GyroDiagnostics[] = [];
  let frame = 0;
  for (let t = 300; t <= sim.endMs; t += 1000 / 30, frame++) {
    let r = sim.visionAt(t);
    if (opts.distort) r = opts.distort(r, t, frame, cmp);
    records.push(cmp.update(frame, t + (opts.stampOffsetMs ?? 0), r, opts.tracked ? opts.tracked(frame) : true, opts.mapId ? opts.mapId(frame) : 1, sim.gyro));
  }
  return { records, cmp };
}

const maxOf = (xs: number[]) => xs.reduce((a, b) => Math.max(a, b), 0);

describe("GyroComparer", () => {
  it("agrees when vision and gyro see the same motion", () => {
    const { records } = run(simulate());
    const last = records[records.length - 1];
    expect(last.available).toBe(true);
    expect(last.sampleHz).toBeGreaterThan(55);
    // Every window, every frame: well under 0.1°.
    const diffs = records.flatMap((r) => r.windows.map((w) => w.diffDeg));
    expect(diffs.length).toBeGreaterThan(200);
    expect(maxOf(diffs)).toBeLessThan(0.05);
    expect(maxOf(records.map((r) => r.frame?.diffDeg ?? 0))).toBeLessThan(0.04);
    // The 1 s window during the pitch saw real rotation on both sides.
    const moving = records.filter((r) => r.windows.length && r.windows[0].gyroDeg > 10);
    expect(moving.length).toBeGreaterThan(5);
    for (const r of moving) expect(Math.abs(r.windows[0].visionDeg - r.windows[0].gyroDeg)).toBeLessThan(0.05);
    // Three hold → move → hold segments (pitch+yaw, roll, pitch back).
    expect(last.segmentCount).toBe(3);
    expect(last.segmentMaxDiffDeg).toBeLessThan(0.05);
    const first = records.find((r) => r.segmentCount === 1)!.segment!;
    expect(first.gyroDeg).toBeGreaterThan(15);
    // Pitch about the camera X axis; the device-Y yaw shows up as −camera Y.
    expect(Math.abs(first.gyroAxisDeg[0])).toBeGreaterThan(14);
    expect(first.gyroAxisDeg[1] * first.visionAxisDeg[1]).toBeGreaterThan(0);
    for (let k = 0; k < 3; k++) expect(first.visionAxisDeg[k]).toBeCloseTo(first.gyroAxisDeg[k], 1);
    // Zero lag found (within the smoothing / step).
    expect(Math.abs(last.lagMs)).toBeLessThan(10);
    expect(records.some((r) => r.still)).toBe(true);
    expect(records.some((r) => !r.still)).toBe(true);
    expect(last.axesBest).toBe("ok");
  });

  it("reports a pose-dependent vision rotation error on the hold → hold segment", () => {
    const sim = simulate();
    const R0 = sim.visionAt(300);
    // Vision rolls 1° too much per 15° of camera rotation away from the start pose
    // (the "tilt and the cube slides, tilt back and it returns" shape).
    const { records } = run(sim, {
      distort: (r) => {
        const away = angleDeg(mat3Multiply(R0, transpose(r)));
        return mat3Multiply(rotationZ((away / 15) * DEG), r);
      },
    });
    const first = records.find((r) => r.segmentCount === 1)!.segment!;
    expect(first.diffDeg).toBeGreaterThan(0.9);
    expect(first.diffDeg).toBeLessThan(1.25);
    // The error is about the camera Z axis (roll).
    expect(Math.abs(first.diffAxisDeg[2])).toBeGreaterThan(0.85);
    expect(Math.abs(first.diffAxisDeg[0]) + Math.abs(first.diffAxisDeg[1])).toBeLessThan(0.4);
    // Back at the start pose the error is gone: the last segment (pitch back) undoes it.
    const last = records[records.length - 1];
    expect(last.segmentCount).toBe(3);
    expect(last.segmentMaxDiffDeg).toBeGreaterThan(0.9);
  });

  it("finds the frame ↔ gyro time offset and the holds do not depend on it", () => {
    // Frames stamped 40 ms late (callback time instead of capture time).
    const { records } = run(simulate(), { stampOffsetMs: 40 });
    const last = records[records.length - 1];
    expect(last.lagMs).toBeGreaterThan(-50);
    expect(last.lagMs).toBeLessThan(-30);
    expect(last.lagRmsBestDeg).toBeLessThan(last.lagRmsZeroDeg);
    // While moving, the zero-lag window disagrees and the lag-corrected one does not.
    const moving = records.filter((r) => r.windows.length && r.windows[0].gyroDeg > 10 && Number.isFinite(r.windows[0].diffLagDeg));
    expect(moving.length).toBeGreaterThan(3);
    expect(maxOf(moving.map((r) => r.windows[0].diffDeg))).toBeGreaterThan(0.3);
    // Hold → hold: both ends stationary, the offset drops out.
    expect(last.segmentCount).toBe(3);
    expect(last.segmentMaxDiffDeg).toBeLessThan(0.1);
  });

  it("stays continuous across a map correction, and shows it when not told", () => {
    const sim = simulate();
    const rc = rotationAxisAngle([0.2, 1, -0.3], 1.5 * DEG);
    const rcT = transpose(rc);
    // From frame 60 on the pose is expressed in the corrected map: R' = R · Rcᵀ.
    const told = run(sim, {
      distort: (r, _t, frame, cmp) => {
        if (frame === 60) cmp.applyMapCorrection(rc);
        return frame >= 60 ? mat3Multiply(r, rcT) : r;
      },
    });
    expect(maxOf(told.records.flatMap((r) => r.windows.map((w) => w.diffDeg)))).toBeLessThan(0.05);
    expect(maxOf(told.records.map((r) => r.frame?.diffDeg ?? 0))).toBeLessThan(0.04);
    const untold = run(sim, { distort: (r, _t, frame) => (frame >= 60 ? mat3Multiply(r, rcT) : r) });
    expect(untold.records[60].frame!.diffDeg).toBeCloseTo(1.5, 1);
  });

  it("skips untracked frames, restarts on a new map, and reports a missing gyro", () => {
    const sim = simulate();
    // Lost for frames 40–70 (held pose = garbage), new map from frame 120.
    const garbage = rotationAxisAngle([1, 0, 0], 0.7);
    const { records } = run(sim, {
      tracked: (f) => f < 40 || f > 70,
      distort: (r, _t, f) => (f >= 40 && f <= 70 ? garbage : f >= 120 ? mat3Multiply(r, rotationAxisAngle([0, 1, 0], 0.5)) : r),
      mapId: (f) => (f >= 120 ? 2 : 1),
    });
    for (let f = 40; f <= 70; f++) {
      expect(records[f].frame).toBeNull();
      expect(records[f].windows.length).toBe(0);
    }
    // Nothing compares across the map change; within each map the two agree.
    expect(records[120].frame).toBeNull();
    expect(records[120].windows.length).toBe(0);
    expect(maxOf(records.flatMap((r) => r.windows.map((w) => w.diffDeg)))).toBeLessThan(0.05);
    expect(maxOf(records.map((r) => r.segmentMaxDiffDeg))).toBeLessThan(0.05);

    const cmp = new GyroComparer(CFG);
    expect(cmp.update(0, 0, sim.visionAt(300), true, 1, null).available).toBe(false);
    expect(cmp.update(1, 33, sim.visionAt(333), true, 1, new GyroProvider()).available).toBe(false);
  });

  it("a gyro bias is learned during the holds and does not show as disagreement", () => {
    // 1.5 s of holding still before the first move (the bias needs 0.5 s of stillness on both sides of a sample).
    const { records } = run(simulate(6400, [0.3 * DEG, -0.2 * DEG, 0.25 * DEG], undefined, 1500));
    const last = records[records.length - 1];
    expect(last.biasDegPerSec).toBeGreaterThan(0.3);
    expect(last.biasDegPerSec).toBeLessThan(0.55);
    // Uncorrected, 0.44°/s over a 2 s segment would be ~0.9°.
    expect(last.segmentCount).toBe(3);
    expect(last.segmentMaxDiffDeg).toBeLessThan(0.3);
  });

  it("names the axis mapping when the browser reports the rates about other axes (recording 45, iOS WebKit)", () => {
    // WebKit: alpha / beta / gamma are the rates about X / Y / Z. Read with
    // the W3C order (x = beta, y = gamma, z = alpha) the device sees (y, z, x).
    const wrong = run(simulate(6400, [0, 0, 0], (x, y, z) => rotationRateToDevice("w3c", x, y, z)));
    const seg = wrong.records.find((r) => r.segmentCount === 1)!.segment!;
    // Same amount of rotation, about a perpendicular axis: the disagreement is ~√2 × the angle.
    expect(Math.abs(seg.visionDeg - seg.gyroDeg)).toBeLessThan(0.5);
    expect(seg.diffDeg).toBeGreaterThan(seg.visionDeg);
    // The pitch (camera x) shows up on the camera z axis, as on the device.
    expect(Math.abs(seg.gyroAxisDeg[2])).toBeGreaterThan(12);
    expect(Math.abs(seg.visionAxisDeg[2])).toBeLessThan(2);
    expect(Math.abs(seg.visionAxisDeg[0])).toBeGreaterThan(12);
    const last = wrong.records[wrong.records.length - 1];
    expect(last.axesBest).not.toBe("ok");
    expect(last.axesBest).toMatch(/^\[[+−][xyz],[+−][xyz],[+−][xyz]\]( T)?$/);
    expect(last.axesBestRmsDeg).toBeLessThan(last.axesRmsDeg * 0.5);

    // Read with WebKit's order the same samples agree, and the check says so.
    const right = run(simulate(6400, [0, 0, 0], (x, y, z) => rotationRateToDevice("webkit", x, y, z)));
    const ok = right.records[right.records.length - 1];
    expect(ok.axesBest).toBe("ok");
    expect(ok.segmentCount).toBe(3);
    expect(ok.segmentMaxDiffDeg).toBeLessThan(0.05);
    expect(rotationRateToDevice("w3c", 1, 2, 3)).toEqual([2, 3, 1]);
    expect(rotationRateToDevice("webkit", 1, 2, 3)).toEqual([1, 2, 3]);
  });
});
