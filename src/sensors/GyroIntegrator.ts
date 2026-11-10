/**
 * Gyro integration for the vision ↔ gyro rotation diagnostic (Phase 6, step 0:
 * measure before fusing anything).
 *
 * Samples are body rates in the W3C device frame (X right, Y up, Z out of the
 * screen), rad/s, stamped in milliseconds on the `performance.now()` clock.
 * The integrator keeps a ring of cumulative orientations `q(t)` (device →
 * start frame, `q ← q ⊗ Exp(ω·dt)`), so the rotation between any two recent
 * times is `q(t0)⁻¹ ⊗ q(t1)`: it maps a vector expressed in the device frame
 * at t1 into the device frame at t0.
 *
 * Bias: the mean rate while the device is held still (|ω| below
 * `stillRateRad` from `stillMinMs` before a sample to `stillMinMs` after it)
 * is tracked with a slow EMA and subtracted before integration. Hand tremor
 * is zero-mean, so a hold gives the bias.
 *
 * Pure (no DOM): `GyroProvider` feeds it from DeviceMotion.
 */
import type { Mat3 } from "../math/Matrix";

export interface GyroIntegratorOptions {
  /** Ring capacity in samples (60 Hz × 20 s by default). */
  capacity?: number;
  /** Rate magnitude below which the device counts as still (rad/s). */
  stillRateRad?: number;
  /** How long the rate must stay below it (ms). */
  stillMinMs?: number;
  /** EMA weight per still sample for the bias. */
  biasAlpha?: number;
  /** Subtract the estimated bias before integrating. */
  correctBias?: boolean;
  /** Gaps longer than this are not integrated (ms): the ring restarts. */
  maxGapMs?: number;
}

export class GyroIntegrator {
  private readonly cap: number;
  private readonly times: Float64Array;
  private readonly quats: Float64Array; // w, x, y, z per sample
  private readonly rates: Float64Array; // |ω| per sample (rad/s, bias-corrected)
  private head = 0;
  private count = 0;
  private readonly q = new Float64Array([1, 0, 0, 0]);
  private lastT = NaN;
  private stillSince = NaN;
  /** Still samples waiting to be confirmed by what follows them: t, wx, wy, wz. */
  private readonly pending: number[] = [];
  private readonly stillRate: number;
  private readonly stillMinMs: number;
  private readonly biasAlpha: number;
  private readonly correctBias: boolean;
  private readonly maxGapMs: number;
  /** Estimated bias (rad/s, device frame). */
  readonly bias = new Float64Array(3);
  biasSamples = 0;
  /** Segment id: bumps whenever the ring restarts after a gap. */
  private segment = 0;
  private readonly segments: Int32Array;
  private hzEma = 0;

  constructor(opts: GyroIntegratorOptions = {}) {
    this.cap = opts.capacity ?? 1200;
    this.times = new Float64Array(this.cap);
    this.quats = new Float64Array(this.cap * 4);
    this.rates = new Float64Array(this.cap);
    this.segments = new Int32Array(this.cap);
    this.stillRate = opts.stillRateRad ?? (2 * Math.PI) / 180;
    this.stillMinMs = opts.stillMinMs ?? 250;
    this.biasAlpha = opts.biasAlpha ?? 0.02;
    this.correctBias = opts.correctBias ?? true;
    this.maxGapMs = opts.maxGapMs ?? 200;
  }

  get sampleCount(): number {
    return this.count;
  }
  /** Smoothed sample rate (Hz), 0 before two samples. */
  get sampleHz(): number {
    return this.hzEma;
  }
  get latestTime(): number {
    return this.lastT;
  }
  /** Latest bias-corrected rate magnitude (rad/s). */
  get latestRate(): number {
    return this.count ? this.rates[(this.head + this.cap - 1) % this.cap] : 0;
  }

  reset(): void {
    this.head = 0;
    this.count = 0;
    this.q.set([1, 0, 0, 0]);
    this.lastT = NaN;
    this.stillSince = NaN;
    this.segment++;
  }

  /** Add one sample: time (ms) and body rates (rad/s, device frame). */
  add(tMs: number, wx: number, wy: number, wz: number): void {
    if (!Number.isFinite(tMs) || !Number.isFinite(wx) || !Number.isFinite(wy) || !Number.isFinite(wz)) return;
    if (Number.isFinite(this.lastT)) {
      const dtMs = tMs - this.lastT;
      if (dtMs <= 0) return; // duplicate / out of order
      if (dtMs > this.maxGapMs) {
        // Page was hidden, events throttled: what happened in between is unknown.
        this.segment++;
        this.stillSince = NaN;
        this.pending.length = 0;
      } else {
        this.hzEma = this.hzEma === 0 ? 1000 / dtMs : this.hzEma * 0.95 + (1000 / dtMs) * 0.05;
        // Bias from holds, on the raw rate.
        const rawMag = Math.hypot(wx - this.bias[0], wy - this.bias[1], wz - this.bias[2]);
        if (rawMag < this.stillRate) {
          if (!Number.isFinite(this.stillSince)) this.stillSince = tMs;
          // A sample counts towards the bias only when the device was still
          // for `stillMinMs` before it *and* stays still for as long after
          // it: the slow start of a real rotation must not leak into the bias.
          this.pending.push(tMs, wx, wy, wz);
          while (this.pending.length && tMs - this.pending[0] >= this.stillMinMs) {
            const pt = this.pending[0];
            if (pt - this.stillSince >= this.stillMinMs) {
              const a = this.biasSamples < 1 / this.biasAlpha ? 1 / (this.biasSamples + 1) : this.biasAlpha;
              this.bias[0] += (this.pending[1] - this.bias[0]) * a;
              this.bias[1] += (this.pending[2] - this.bias[1]) * a;
              this.bias[2] += (this.pending[3] - this.bias[2]) * a;
              this.biasSamples++;
            }
            this.pending.splice(0, 4);
          }
        } else {
          this.stillSince = NaN;
          this.pending.length = 0;
        }
        const bx = this.correctBias ? this.bias[0] : 0;
        const by = this.correctBias ? this.bias[1] : 0;
        const bz = this.correctBias ? this.bias[2] : 0;
        const dt = dtMs / 1000;
        integrate(this.q, (wx - bx) * dt, (wy - by) * dt, (wz - bz) * dt);
      }
    }
    this.lastT = tMs;
    const i = this.head;
    this.times[i] = tMs;
    this.quats.set(this.q, i * 4);
    this.rates[i] = Math.hypot(
      wx - (this.correctBias ? this.bias[0] : 0),
      wy - (this.correctBias ? this.bias[1] : 0),
      wz - (this.correctBias ? this.bias[2] : 0),
    );
    this.segments[i] = this.segment;
    this.head = (this.head + 1) % this.cap;
    if (this.count < this.cap) this.count++;
  }

  /** Index (ring position) of the last sample with time ≤ t, or −1. */
  private find(tMs: number): number {
    if (this.count === 0) return -1;
    let lo = 0;
    let hi = this.count - 1;
    const base = (this.head + this.cap - this.count) % this.cap;
    if (this.times[base] > tMs) return -1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (this.times[(base + mid) % this.cap] <= tMs) lo = mid;
      else hi = mid - 1;
    }
    return lo;
  }

  /**
   * Cumulative orientation at time t (w, x, y, z), interpolated between the
   * two neighbouring samples, with the segment id it belongs to. Null when t
   * is outside the ring (older than the oldest sample, or more than
   * `toleranceMs` past the newest).
   */
  orientationAt(tMs: number, out: Float64Array, toleranceMs = 60): number {
    const k = this.find(tMs);
    if (k < 0) return -1;
    const base = (this.head + this.cap - this.count) % this.cap;
    const i0 = (base + k) % this.cap;
    if (k === this.count - 1) {
      if (tMs - this.times[i0] > toleranceMs) return -1;
      for (let j = 0; j < 4; j++) out[j] = this.quats[i0 * 4 + j];
      return this.segments[i0];
    }
    const i1 = (base + k + 1) % this.cap;
    if (this.segments[i1] !== this.segments[i0]) {
      // Across a gap: only the near side of it is usable.
      if (tMs - this.times[i0] > toleranceMs) return -1;
      for (let j = 0; j < 4; j++) out[j] = this.quats[i0 * 4 + j];
      return this.segments[i0];
    }
    const span = this.times[i1] - this.times[i0];
    const u = span > 0 ? (tMs - this.times[i0]) / span : 0;
    // Neighbouring samples are a fraction of a degree apart: nlerp is exact enough.
    let n = 0;
    for (let j = 0; j < 4; j++) {
      out[j] = this.quats[i0 * 4 + j] * (1 - u) + this.quats[i1 * 4 + j] * u;
      n += out[j] * out[j];
    }
    n = Math.sqrt(n) || 1;
    for (let j = 0; j < 4; j++) out[j] /= n;
    return this.segments[i0];
  }

  /**
   * Rotation between two times in the device frame, row-major: maps a vector
   * in the device frame at t1 into the device frame at t0. Null when either
   * time is outside the ring or a gap lies between them.
   */
  rotationBetween(t0Ms: number, t1Ms: number, out: Mat3 = new Float64Array(9)): Mat3 | null {
    const s0 = this.orientationAt(t0Ms, QA);
    const s1 = this.orientationAt(t1Ms, QB);
    if (s0 < 0 || s1 < 0 || s0 !== s1) return null;
    // q0⁻¹ ⊗ q1
    const w0 = QA[0], x0 = -QA[1], y0 = -QA[2], z0 = -QA[3];
    const w1 = QB[0], x1 = QB[1], y1 = QB[2], z1 = QB[3];
    const w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1;
    const x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1;
    const y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1;
    const z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1;
    quatToMat(w, x, y, z, out);
    return out;
  }

  /** Largest bias-corrected rate magnitude in [t0, t1] (rad/s); NaN when no sample falls inside. */
  maxRateBetween(t0Ms: number, t1Ms: number): number {
    const base = (this.head + this.cap - this.count) % this.cap;
    let max = NaN;
    for (let k = this.count - 1; k >= 0; k--) {
      const i = (base + k) % this.cap;
      const t = this.times[i];
      if (t > t1Ms) continue;
      if (t < t0Ms) break;
      if (!(max >= this.rates[i])) max = this.rates[i];
    }
    return max;
  }
}

const QA = new Float64Array(4);
const QB = new Float64Array(4);

/** q ← q ⊗ Exp(rotation vector), normalized. */
function integrate(q: Float64Array, rx: number, ry: number, rz: number): void {
  const th = Math.hypot(rx, ry, rz);
  let dw = 1, dx = rx * 0.5, dy = ry * 0.5, dz = rz * 0.5;
  if (th > 1e-9) {
    const s = Math.sin(th / 2) / th;
    dw = Math.cos(th / 2);
    dx = rx * s;
    dy = ry * s;
    dz = rz * s;
  }
  const w = q[0], x = q[1], y = q[2], z = q[3];
  let nw = w * dw - x * dx - y * dy - z * dz;
  let nx = w * dx + x * dw + y * dz - z * dy;
  let ny = w * dy - x * dz + y * dw + z * dx;
  let nz = w * dz + x * dy - y * dx + z * dw;
  const n = Math.hypot(nw, nx, ny, nz) || 1;
  nw /= n; nx /= n; ny /= n; nz /= n;
  q[0] = nw; q[1] = nx; q[2] = ny; q[3] = nz;
}

function quatToMat(w: number, x: number, y: number, z: number, m: Mat3): void {
  m[0] = 1 - 2 * (y * y + z * z);
  m[1] = 2 * (x * y - w * z);
  m[2] = 2 * (x * z + w * y);
  m[3] = 2 * (x * y + w * z);
  m[4] = 1 - 2 * (x * x + z * z);
  m[5] = 2 * (y * z - w * x);
  m[6] = 2 * (x * z - w * y);
  m[7] = 2 * (y * z + w * x);
  m[8] = 1 - 2 * (x * x + y * y);
}

/**
 * Device frame → camera frame (X right, Y down, Z forward) for the back
 * camera, following the upright video at the given screen orientation angle
 * (same mapping as GravityProvider): rotate (x, y) by the angle, then
 * camera = (x, −y, −z). A proper rotation, row-major.
 */
export function deviceToCameraMatrix(screenAngleDeg: number, out: Mat3 = new Float64Array(9)): Mat3 {
  const rad = (screenAngleDeg * Math.PI) / 180;
  const c = Math.cos(rad);
  const s = Math.sin(rad);
  out[0] = c; out[1] = -s; out[2] = 0;
  out[3] = -s; out[4] = -c; out[5] = 0;
  out[6] = 0; out[7] = 0; out[8] = -1;
  return out;
}

/** M · R · Mᵀ: a device-frame rotation expressed in camera axes. */
export function conjugateRotation(m: Mat3, r: Mat3, out: Mat3 = new Float64Array(9)): Mat3 {
  const t = TMP;
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) t[i * 3 + j] = m[i * 3] * r[j] + m[i * 3 + 1] * r[3 + j] + m[i * 3 + 2] * r[6 + j];
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) out[i * 3 + j] = t[i * 3] * m[j * 3] + t[i * 3 + 1] * m[j * 3 + 1] + t[i * 3 + 2] * m[j * 3 + 2];
  return out;
}
const TMP = new Float64Array(9);
