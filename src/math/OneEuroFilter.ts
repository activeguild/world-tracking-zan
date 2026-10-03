/**
 * One Euro filter (Casiez et al. 2012) for pose smoothing (spec §34).
 *
 * Low cutoff at rest removes jitter; the cutoff rises with speed so fast
 * motion is passed through with little latency.
 */
export interface OneEuroConfig {
  /** Minimum cutoff frequency (Hz). Lower = smoother at rest. */
  minCutoff: number;
  /** Speed coefficient. Higher = less lag during fast motion. */
  beta: number;
  /** Cutoff for the derivative estimate (Hz). */
  dCutoff: number;
}

export const DEFAULT_ONE_EURO: OneEuroConfig = { minCutoff: 1.0, beta: 0.05, dCutoff: 1.0 };

class LowPass {
  private y = 0;
  private initialized = false;

  filter(x: number, alpha: number): number {
    if (!this.initialized) {
      this.y = x;
      this.initialized = true;
      return x;
    }
    this.y = alpha * x + (1 - alpha) * this.y;
    return this.y;
  }

  get last(): number {
    return this.y;
  }

  reset(): void {
    this.initialized = false;
  }
}

function alphaFor(cutoff: number, dt: number): number {
  const tau = 1 / (2 * Math.PI * cutoff);
  return 1 / (1 + tau / dt);
}

/** Scalar One Euro filter. */
export class OneEuroFilter {
  private readonly x = new LowPass();
  private readonly dx = new LowPass();
  private lastTime: number | null = null;

  constructor(public config: OneEuroConfig = DEFAULT_ONE_EURO) {}

  reset(): void {
    this.x.reset();
    this.dx.reset();
    this.lastTime = null;
  }

  /** @param t timestamp in seconds */
  filter(value: number, t: number): number {
    if (this.lastTime === null) {
      this.lastTime = t;
      this.dx.filter(0, 1);
      return this.x.filter(value, 1);
    }
    const dt = Math.max(1e-4, t - this.lastTime);
    this.lastTime = t;
    const prev = this.x.last;
    const d = (value - prev) / dt;
    const edx = this.dx.filter(d, alphaFor(this.config.dCutoff, dt));
    const cutoff = this.config.minCutoff + this.config.beta * Math.abs(edx);
    return this.x.filter(value, alphaFor(cutoff, dt));
  }
}

/** Vector One Euro filter (independent per component). */
export class OneEuroVector {
  private readonly filters: OneEuroFilter[];

  constructor(size: number, config: OneEuroConfig = DEFAULT_ONE_EURO) {
    this.filters = Array.from({ length: size }, () => new OneEuroFilter(config));
  }

  reset(): void {
    for (const f of this.filters) f.reset();
  }

  filter(values: ArrayLike<number>, t: number, out?: Float64Array): Float64Array {
    const o = out ?? new Float64Array(this.filters.length);
    for (let i = 0; i < this.filters.length; i++) o[i] = this.filters[i].filter(values[i], t);
    return o;
  }
}

/**
 * Quaternion smoothing with a speed-adaptive slerp factor (same idea as
 * the One Euro filter applied to the rotation angle). Quaternions as
 * [x, y, z, w].
 */
export class QuaternionSmoother {
  private q: Float64Array | null = null;
  private lastTime: number | null = null;
  private readonly speed = new LowPass();

  constructor(public config: OneEuroConfig = DEFAULT_ONE_EURO) {}

  reset(): void {
    this.q = null;
    this.lastTime = null;
    this.speed.reset();
  }

  filter(input: ArrayLike<number>, t: number): Float64Array {
    let target: Float64Array = Float64Array.from([input[0], input[1], input[2], input[3]]);
    if (!this.q || this.lastTime === null) {
      this.q = target;
      this.lastTime = t;
      return Float64Array.from(this.q);
    }
    const dt = Math.max(1e-4, t - this.lastTime);
    this.lastTime = t;
    // Shortest arc
    let dot = this.q[0] * target[0] + this.q[1] * target[1] + this.q[2] * target[2] + this.q[3] * target[3];
    if (dot < 0) {
      target = Float64Array.from(target, (v) => -v);
      dot = -dot;
    }
    const angle = 2 * Math.acos(Math.min(1, dot));
    const speed = this.speed.filter(angle / dt, alphaFor(this.config.dCutoff, dt));
    const cutoff = this.config.minCutoff + this.config.beta * speed;
    const alpha = alphaFor(cutoff, dt);
    this.q = slerp(this.q, target, alpha);
    return Float64Array.from(this.q);
  }
}

function slerp(a: Float64Array, b: Float64Array, t: number): Float64Array {
  let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  dot = Math.max(-1, Math.min(1, dot));
  const out = new Float64Array(4);
  if (dot > 0.9995) {
    for (let i = 0; i < 4; i++) out[i] = a[i] + t * (b[i] - a[i]);
  } else {
    const theta = Math.acos(dot);
    const s = Math.sin(theta);
    const wa = Math.sin((1 - t) * theta) / s;
    const wb = Math.sin(t * theta) / s;
    for (let i = 0; i < 4; i++) out[i] = wa * a[i] + wb * b[i];
  }
  const len = Math.hypot(out[0], out[1], out[2], out[3]) || 1;
  for (let i = 0; i < 4; i++) out[i] /= len;
  return out;
}
