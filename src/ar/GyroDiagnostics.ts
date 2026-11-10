/**
 * Vision ↔ gyro rotation comparison (Phase 6 step 0, diagnostics only).
 *
 * The question it answers: when the phone is tilted and the placed object
 * slides, is the *rotation* of the vision pose wrong? The gyro measures the
 * relative rotation between two moments independently of the map, so
 *
 *   V = R(t0) · R(t1)ᵀ          (vision: camera at t1 → camera at t0)
 *   G = gyro rotation t1 → t0   (same convention, camera axes)
 *   D = V · Gᵀ                  (what vision rotated that the gyro did not)
 *
 * and the angle of D is the disagreement. Three views of it:
 *
 *  - sliding windows (1 s / 3 s): how the two diverge while moving. Sensitive
 *    to the frame ↔ gyro time offset (30°/s × 30 ms = 0.9°), so the offset is
 *    estimated from the last seconds of motion and the window is also
 *    reported at that lag;
 *  - still → still segments: the pose latched while the phone is held still,
 *    compared with the next hold after a move. Both ends are stationary, so
 *    the time offset drops out: this is the clean number;
 *  - frame to frame, for a glance.
 *
 * Map corrections (BA, init stage) re-express the pose in a moved map; they
 * are not camera motion. The comparer accumulates their rotation so that the
 * vision rotation it compares stays continuous across them.
 *
 * Nothing here feeds the tracker.
 */
import type { Mat3 } from "../math/Matrix";
import type { GyroSource } from "../sensors/GyroProvider";

export interface GyroDiagnosticsConfig {
  /** Sliding window lengths (ms). */
  windowsMs: number[];
  /** Gyro rate below which the phone counts as held still (deg/s) … */
  stillRateDegPerSec: number;
  /** … for this long (ms). */
  stillMinMs: number;
  /** A still → still segment is reported when the gyro turned at least this much in between (deg). */
  segmentMinDeg: number;
  /** Time-offset search: ± range and step (ms), over this much history (ms), needing this much motion (deg). */
  lagMaxMs: number;
  lagStepMs: number;
  lagWindowMs: number;
  lagMinMotionDeg: number;
  /** Re-estimate the lag every N results. */
  lagEveryFrames: number;
}

export interface GyroWindowRecord {
  windowMs: number;
  /** Actual span between the two frames compared (ms). */
  spanMs: number;
  visionDeg: number;
  gyroDeg: number;
  /** Angle of D at zero lag (deg). */
  diffDeg: number;
  /** Rotation vector of D in camera axes at t0 (deg): x = pitch, y = yaw, z = roll. */
  diffAxisDeg: [number, number, number];
  /** Angle of D with the gyro read at the estimated lag (deg); NaN without a lag estimate. */
  diffLagDeg: number;
}

export interface GyroSegmentRecord {
  /** Frame ids and times of the two holds. */
  fromFrameId: number;
  toFrameId: number;
  durationMs: number;
  visionDeg: number;
  gyroDeg: number;
  diffDeg: number;
  diffAxisDeg: [number, number, number];
  /** Rotation vectors of both (camera axes at the first hold, deg), to check the axis mapping. */
  visionAxisDeg: [number, number, number];
  gyroAxisDeg: [number, number, number];
}

export interface GyroDiagnostics {
  /** Gyro samples are arriving. */
  available: boolean;
  sampleHz: number;
  rateDegPerSec: number;
  biasDegPerSec: number;
  /** The phone is held still at this frame. */
  still: boolean;
  /** Previous tracked frame → this frame. */
  frame: { visionDeg: number; gyroDeg: number; diffDeg: number } | null;
  windows: GyroWindowRecord[];
  /** Estimated time offset: the gyro at (frame time + lagMs) matches the frame. NaN until estimated. */
  lagMs: number;
  /** RMS frame-to-frame disagreement at zero lag and at the best lag (deg). */
  lagRmsZeroDeg: number;
  lagRmsBestDeg: number;
  /**
   * Axis check: the frame-to-frame disagreement (RMS, deg) with the axis
   * mapping in use, and with the best of the 48 alternatives (24 signed axis
   * permutations × rate handedness). `axesBest` names the alternative
   * (camera axes: new x, y, z as ± old axes, `T` = opposite handedness) or is
   * "ok" when the mapping in use is the best one. NaN / "" until estimated.
   */
  axesRmsDeg: number;
  axesBestRmsDeg: number;
  axesBest: string;
  /** Convention of the gyro source ("w3c" / "webkit"), when it says. */
  axes: string;
  /** Last completed still → still segment, how long ago, and how many so far. */
  segment: GyroSegmentRecord | null;
  segmentAgeMs: number;
  segmentCount: number;
  /** Largest and mean disagreement over all segments (deg). */
  segmentMaxDiffDeg: number;
  segmentMeanDiffDeg: number;
}

interface Entry {
  t: number;
  frameId: number;
  r: Float64Array;
}

const HISTORY_MS = 8000;
const MAX_FRAME_GAP_MS = 250;

export class GyroComparer {
  private history: Entry[] = [];
  /** Accumulated map-correction rotation: continuous rotation = R_pose · acc. */
  private readonly acc = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  private mapFrameId = -1;
  private anchor: Entry | null = null;
  private movedSinceAnchor = false;
  private lagMs = NaN;
  private lagRmsZero = NaN;
  private lagRmsBest = NaN;
  private axesRms = NaN;
  private axesBestRms = NaN;
  private axesBest = "";
  private results = 0;
  private segment: GyroSegmentRecord | null = null;
  private segmentTime = 0;
  private segmentCount = 0;
  private segmentMax = 0;
  private segmentSum = 0;

  constructor(private readonly cfg: GyroDiagnosticsConfig) {}

  /** Forget everything that refers to the current map (map reset). */
  resetMap(): void {
    this.history = [];
    this.acc.set([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    this.anchor = null;
    this.movedSinceAnchor = false;
  }

  /**
   * The map was re-expressed by a rigid correction (old map → new map, row-major
   * rotation Rc): the pose rotation became R · Rcᵀ without the camera moving.
   * Call before `update` of the frame that carries the corrected pose.
   */
  applyMapCorrection(rc: ArrayLike<number>): void {
    mul(rc, this.acc, TMP_A);
    this.acc.set(TMP_A);
  }

  /**
   * One vision result. `rotation` is the map pose rotation (X_cam = R X_map,
   * row-major) of the frame captured at `tMs`; `tracked` = PnP located the
   * camera this frame (held / propagated poses are not compared).
   */
  update(frameId: number, tMs: number, rotation: ArrayLike<number> | null, tracked: boolean, mapFrameId: number, gyro: GyroSource | null): GyroDiagnostics {
    this.results++;
    if (mapFrameId !== this.mapFrameId) {
      this.resetMap();
      this.mapFrameId = mapFrameId;
    }
    const available = !!gyro && gyro.hasReading;
    const out: GyroDiagnostics = {
      available,
      sampleHz: gyro?.sampleHz ?? 0,
      rateDegPerSec: gyro?.rateDeg ?? 0,
      biasDegPerSec: gyro?.biasDeg ?? 0,
      still: false,
      frame: null,
      windows: [],
      lagMs: this.lagMs,
      lagRmsZeroDeg: this.lagRmsZero,
      lagRmsBestDeg: this.lagRmsBest,
      axesRmsDeg: this.axesRms,
      axesBestRmsDeg: this.axesBestRms,
      axesBest: this.axesBest,
      axes: gyro?.axes ?? "",
      segment: this.segment,
      segmentAgeMs: this.segment ? tMs - this.segmentTime : 0,
      segmentCount: this.segmentCount,
      segmentMaxDiffDeg: this.segmentMax,
      segmentMeanDiffDeg: this.segmentCount ? this.segmentSum / this.segmentCount : 0,
    };
    if (!available || !gyro) return out;
    const maxRate = gyro.maxRateDegBetween(tMs - this.cfg.stillMinMs, tMs);
    const still = maxRate < this.cfg.stillRateDegPerSec;
    out.still = still;
    if (!still) this.movedSinceAnchor = true;
    if (!rotation || !tracked) return out;

    const cur: Entry = { t: tMs, frameId, r: new Float64Array(9) };
    mul(rotation, this.acc, cur.r);
    const prev = this.history.length ? this.history[this.history.length - 1] : null;
    this.history.push(cur);
    while (this.history.length && tMs - this.history[0].t > HISTORY_MS) this.history.shift();

    // Frame to frame.
    if (prev && tMs - prev.t <= MAX_FRAME_GAP_MS) {
      const g = gyro.cameraRotationBetween(prev.t, tMs);
      if (g) {
        relative(prev.r, cur.r, TMP_V);
        mulT(TMP_V, g, TMP_D);
        out.frame = { visionDeg: angleDeg(TMP_V), gyroDeg: angleDeg(g), diffDeg: angleDeg(TMP_D) };
      }
    }

    // Time offset between the frame clock and the gyro clock.
    if (this.results % Math.max(1, this.cfg.lagEveryFrames) === 0) this.estimateLag(gyro, tMs);
    out.lagMs = this.lagMs;
    out.lagRmsZeroDeg = this.lagRmsZero;
    out.lagRmsBestDeg = this.lagRmsBest;
    out.axesRmsDeg = this.axesRms;
    out.axesBestRmsDeg = this.axesBestRms;
    out.axesBest = this.axesBest;

    // Sliding windows.
    for (const w of this.cfg.windowsMs) {
      const from = this.entryAt(tMs - w);
      if (!from || tMs - from.t < w * 0.6) continue;
      const g = gyro.cameraRotationBetween(from.t, tMs);
      if (!g) continue;
      relative(from.r, cur.r, TMP_V);
      mulT(TMP_V, g, TMP_D);
      const rec: GyroWindowRecord = {
        windowMs: w,
        spanMs: tMs - from.t,
        visionDeg: angleDeg(TMP_V),
        gyroDeg: angleDeg(g),
        diffDeg: angleDeg(TMP_D),
        diffAxisDeg: rotationVectorDeg(TMP_D),
        diffLagDeg: NaN,
      };
      if (Number.isFinite(this.lagMs)) {
        const gl = gyro.cameraRotationBetween(from.t + this.lagMs, tMs + this.lagMs);
        if (gl) {
          mulT(TMP_V, gl, TMP_D);
          rec.diffLagDeg = angleDeg(TMP_D);
        }
      }
      out.windows.push(rec);
    }

    // Still → still segments.
    if (still) {
      if (this.anchor && this.movedSinceAnchor) {
        const g = gyro.cameraRotationBetween(this.anchor.t, tMs);
        if (g && angleDeg(g) >= this.cfg.segmentMinDeg) {
          relative(this.anchor.r, cur.r, TMP_V);
          mulT(TMP_V, g, TMP_D);
          const seg: GyroSegmentRecord = {
            fromFrameId: this.anchor.frameId,
            toFrameId: frameId,
            durationMs: tMs - this.anchor.t,
            visionDeg: angleDeg(TMP_V),
            gyroDeg: angleDeg(g),
            diffDeg: angleDeg(TMP_D),
            diffAxisDeg: rotationVectorDeg(TMP_D),
            visionAxisDeg: rotationVectorDeg(TMP_V),
            gyroAxisDeg: rotationVectorDeg(g),
          };
          this.segment = seg;
          this.segmentTime = tMs;
          this.segmentCount++;
          this.segmentSum += seg.diffDeg;
          if (seg.diffDeg > this.segmentMax) this.segmentMax = seg.diffDeg;
          out.segment = seg;
          out.segmentAgeMs = 0;
          out.segmentCount = this.segmentCount;
          out.segmentMaxDiffDeg = this.segmentMax;
          out.segmentMeanDiffDeg = this.segmentSum / this.segmentCount;
        }
      }
      this.anchor = cur;
      this.movedSinceAnchor = false;
    }
    return out;
  }

  /** True on the update that completed a segment (for logging). */
  get lastSegmentTime(): number {
    return this.segmentTime;
  }

  /** History entry closest to time t (none older than the oldest kept). */
  private entryAt(t: number): Entry | null {
    let best: Entry | null = null;
    let bestD = Infinity;
    for (const e of this.history) {
      const d = Math.abs(e.t - t);
      if (d < bestD) {
        bestD = d;
        best = e;
      }
    }
    return best;
  }

  private estimateLag(gyro: GyroSource, tMs: number): void {
    const h = this.history;
    let first = h.length - 1;
    while (first > 0 && tMs - h[first - 1].t <= this.cfg.lagWindowMs) first--;
    let motion = 0;
    const pairs: [Entry, Entry][] = [];
    for (let i = Math.max(1, first + 1); i < h.length; i++) {
      if (h[i].t - h[i - 1].t > MAX_FRAME_GAP_MS) continue;
      relative(h[i - 1].r, h[i].r, TMP_V);
      motion += angleDeg(TMP_V);
      pairs.push([h[i - 1], h[i]]);
    }
    if (pairs.length < 8 || motion < this.cfg.lagMinMotionDeg) return;
    let bestLag = NaN;
    let bestCost = Infinity;
    let zeroCost = NaN;
    for (let lag = -this.cfg.lagMaxMs; lag <= this.cfg.lagMaxMs + 1e-9; lag += this.cfg.lagStepMs) {
      let sum = 0;
      let n = 0;
      for (const [a, b] of pairs) {
        const g = gyro.cameraRotationBetween(a.t + lag, b.t + lag);
        if (!g) continue;
        relative(a.r, b.r, TMP_V);
        mulT(TMP_V, g, TMP_D);
        const d = angleDeg(TMP_D);
        sum += d * d;
        n++;
      }
      if (n < pairs.length * 0.8) continue;
      const cost = sum / n;
      if (Math.abs(lag) < 1e-9) zeroCost = cost;
      if (cost < bestCost) {
        bestCost = cost;
        bestLag = lag;
      }
    }
    if (!Number.isFinite(bestLag)) return;
    // The cost curve is flat near its minimum for slow motion: smooth the estimate.
    this.lagMs = Number.isFinite(this.lagMs) ? this.lagMs * 0.8 + bestLag * 0.2 : bestLag;
    this.lagRmsBest = Math.sqrt(bestCost);
    this.lagRmsZero = Math.sqrt(zeroCost);
    this.checkAxes(gyro, pairs, bestLag);
  }

  /**
   * Is the device → camera axis mapping right? Re-express the gyro rotation
   * of each frame pair under every signed axis permutation (and the opposite
   * rate handedness) and see which one the vision rotation agrees with.
   * Browsers do not agree on what alpha / beta / gamma are the rates about.
   */
  private checkAxes(gyro: GyroSource, pairs: [Entry, Entry][], lag: number): void {
    const vs: Float64Array[] = [];
    const gs: Float64Array[] = [];
    for (const [a, b] of pairs) {
      const g = gyro.cameraRotationBetween(a.t + lag, b.t + lag);
      if (!g) continue;
      const v = new Float64Array(9);
      relative(a.r, b.r, v);
      vs.push(v);
      gs.push(Float64Array.from(g));
    }
    if (vs.length < 8) return;
    let best = Infinity;
    let bestName = "";
    let current = NaN;
    const pg = new Float64Array(9);
    const gt = new Float64Array(9);
    for (const perm of AXIS_PERMUTATIONS) {
      for (let flip = 0; flip < 2; flip++) {
        let sum = 0;
        for (let i = 0; i < vs.length; i++) {
          let g: ArrayLike<number> = gs[i];
          if (flip) {
            transposeInto(gs[i], gt);
            g = gt;
          }
          conjugate(perm.m, g, pg);
          mulT(vs[i], pg, TMP_D);
          const d = angleDeg(TMP_D);
          sum += d * d;
        }
        const rms = Math.sqrt(sum / vs.length);
        const identity = perm.identity && !flip;
        if (identity) current = rms;
        if (rms < best) {
          best = rms;
          bestName = identity ? "ok" : perm.name + (flip ? " T" : "");
        }
      }
    }
    this.axesRms = current;
    this.axesBestRms = best;
    // Another mapping has to be clearly better to be named (noise picks one at random otherwise).
    this.axesBest = bestName !== "ok" && best < current * 0.7 ? bestName : "ok";
  }
}

interface AxisPermutation {
  m: Float64Array;
  name: string;
  identity: boolean;
}

/** The 24 proper signed axis permutations (rotations of the cube). */
const AXIS_PERMUTATIONS: AxisPermutation[] = (() => {
  const out: AxisPermutation[] = [];
  const axes = ["x", "y", "z"];
  for (const p of [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]) {
    for (let signs = 0; signs < 8; signs++) {
      const s = [signs & 1 ? -1 : 1, signs & 2 ? -1 : 1, signs & 4 ? -1 : 1];
      const m = new Float64Array(9);
      for (let i = 0; i < 3; i++) m[i * 3 + p[i]] = s[i];
      const det =
        m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
      if (det < 0) continue;
      out.push({
        m,
        name: `[${p.map((k, i) => (s[i] < 0 ? "−" : "+") + axes[k]).join(",")}]`,
        identity: p[0] === 0 && p[1] === 1 && p[2] === 2 && signs === 0,
      });
    }
  }
  return out;
})();

const TMP_C = new Float64Array(9);
/** out = m · r · mᵀ */
function conjugate(m: ArrayLike<number>, r: ArrayLike<number>, out: Float64Array): void {
  mul(m, r, TMP_C);
  mulT(TMP_C, m, out);
}
function transposeInto(r: ArrayLike<number>, out: Float64Array): void {
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) out[i * 3 + j] = r[j * 3 + i];
}

const TMP_A = new Float64Array(9);
const TMP_V = new Float64Array(9);
const TMP_D = new Float64Array(9);

/** out = a · b */
function mul(a: ArrayLike<number>, b: ArrayLike<number>, out: Float64Array): void {
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) out[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
}
/** out = a · bᵀ */
function mulT(a: ArrayLike<number>, b: ArrayLike<number>, out: Float64Array): void {
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) out[i * 3 + j] = a[i * 3] * b[j * 3] + a[i * 3 + 1] * b[j * 3 + 1] + a[i * 3 + 2] * b[j * 3 + 2];
}
/** Camera at t1 → camera at t0: R0 · R1ᵀ. */
function relative(r0: ArrayLike<number>, r1: ArrayLike<number>, out: Float64Array): void {
  mulT(r0, r1, out);
}

export function angleDeg(r: ArrayLike<number>): number {
  const c = Math.max(-1, Math.min(1, (r[0] + r[4] + r[8] - 1) / 2));
  // Near identity acos loses precision: use the skew part.
  const sx = r[7] - r[5];
  const sy = r[2] - r[6];
  const sz = r[3] - r[1];
  const s = Math.hypot(sx, sy, sz) / 2;
  return (Math.atan2(s, c) * 180) / Math.PI;
}

/** Rotation vector (axis × angle) in degrees. */
export function rotationVectorDeg(r: ArrayLike<number>): [number, number, number] {
  const sx = r[7] - r[5];
  const sy = r[2] - r[6];
  const sz = r[3] - r[1];
  const s = Math.hypot(sx, sy, sz) / 2;
  const c = Math.max(-1, Math.min(1, (r[0] + r[4] + r[8] - 1) / 2));
  const th = Math.atan2(s, c);
  if (s < 1e-12) return [0, 0, 0];
  const k = ((th / (2 * s)) * 180) / Math.PI;
  return [sx * k, sy * k, sz * k];
}

export function formatGyroSegment(s: GyroSegmentRecord): string {
  const v = (a: [number, number, number]) => `(${a.map((x) => x.toFixed(2)).join(", ")})`;
  return (
    `GYRO SEGMENT frames ${s.fromFrameId} → ${s.toFrameId} (${(s.durationMs / 1000).toFixed(1)} s): vision ${s.visionDeg.toFixed(2)}° gyro ${s.gyroDeg.toFixed(2)}° ` +
    `diff ${s.diffDeg.toFixed(2)}° ${v(s.diffAxisDeg)} | vision ${v(s.visionAxisDeg)} gyro ${v(s.gyroAxisDeg)} (pitch, yaw, roll; camera axes)`
  );
}

export type { Mat3 };
