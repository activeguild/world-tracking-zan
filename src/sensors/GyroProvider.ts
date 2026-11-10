/**
 * Gyro (DeviceMotion `rotationRate`) → rotation of the camera between two
 * frame times, for the vision ↔ gyro diagnostic. Nothing here feeds the
 * tracker.
 *
 * W3C: `rotationRate.alpha / beta / gamma` are deg/s about the device Z / X /
 * Y axes (Chrome follows this). WebKit on iOS reports them about X / Y / Z
 * instead (recording 45, iPhone: a pure pitch of 15.6° by vision came out of
 * the W3C mapping as 14.9° about the camera Z axis, i.e. `alpha` carried the
 * device X rate). `GyroAxes` selects the convention; `detectGyroAxes()`
 * picks WebKit's on iOS. Event times are `event.timeStamp` (same clock as
 * `performance.now()` and the video frame callback); when a browser reports
 * something else (not within a second of now) the arrival time is used.
 */
import type { Mat3 } from "../math/Matrix";
import { GyroIntegrator, conjugateRotation, deviceToCameraMatrix, type GyroIntegratorOptions } from "./GyroIntegrator";

/** What the diagnostic needs from a gyro (implemented by GyroProvider; faked in tests). */
export interface GyroSource {
  /** Rotation mapping camera-frame vectors at t1 into the camera frame at t0 (row-major), or null. */
  cameraRotationBetween(t0Ms: number, t1Ms: number): Mat3 | null;
  /** Largest rate magnitude in [t0, t1] (deg/s); NaN when unknown. */
  maxRateDegBetween(t0Ms: number, t1Ms: number): number;
  /** Latest rate magnitude (deg/s). */
  readonly rateDeg: number;
  /** Estimated bias magnitude (deg/s). */
  readonly biasDeg: number;
  readonly sampleHz: number;
  readonly hasReading: boolean;
  /** The bias estimate is subtracted before integration (false: only reported). */
  readonly biasCorrected?: boolean;
  /** Axis convention in use (for the HUD); undefined for fakes. */
  readonly axes?: string;
}

const DEG = Math.PI / 180;

/** Which device axis each of alpha / beta / gamma is the rate about. */
export type GyroAxes = "w3c" | "webkit";

/** iOS (every browser there is WebKit) → "webkit", otherwise the W3C order. */
export function detectGyroAxes(): GyroAxes {
  if (typeof navigator === "undefined") return "w3c";
  const ua = navigator.userAgent || "";
  const iOS = /iPad|iPhone|iPod/.test(ua) || (/Macintosh/.test(ua) && (navigator.maxTouchPoints ?? 0) > 1);
  return iOS ? "webkit" : "w3c";
}

/** alpha / beta / gamma (any unit) → rates about the device X / Y / Z axes. */
export function rotationRateToDevice(axes: GyroAxes, alpha: number, beta: number, gamma: number): [number, number, number] {
  return axes === "webkit" ? [alpha, beta, gamma] : [beta, gamma, alpha];
}

export class GyroProvider implements GyroSource {
  readonly integrator: GyroIntegrator;
  readonly biasCorrected: boolean;
  private listening = false;
  private readonly m = new Float64Array(9);
  private readonly r = new Float64Array(9);
  private readonly onMotion = (ev: DeviceMotionEvent) => {
    const w = ev.rotationRate;
    if (!w || w.alpha == null || w.beta == null || w.gamma == null) return;
    const now = performance.now();
    const t = Math.abs(ev.timeStamp - now) < 1000 ? ev.timeStamp : now;
    const [x, y, z] = rotationRateToDevice(this.axes, w.alpha, w.beta, w.gamma);
    this.integrator.add(t, x * DEG, y * DEG, z * DEG);
  };

  constructor(
    opts?: GyroIntegratorOptions,
    readonly axes: GyroAxes = detectGyroAxes(),
  ) {
    this.integrator = new GyroIntegrator(opts);
    this.biasCorrected = opts?.correctBias ?? true;
  }

  /** Start listening. The DeviceMotion permission (iOS) is requested by GravityProvider.start(). */
  start(): boolean {
    if (typeof window === "undefined" || !("DeviceMotionEvent" in window)) return false;
    if (!this.listening) window.addEventListener("devicemotion", this.onMotion);
    this.listening = true;
    return true;
  }

  stop(): void {
    if (this.listening) window.removeEventListener("devicemotion", this.onMotion);
    this.listening = false;
  }

  get hasReading(): boolean {
    return this.integrator.sampleCount > 1;
  }
  get sampleHz(): number {
    return this.integrator.sampleHz;
  }
  get rateDeg(): number {
    return this.integrator.latestRate / DEG;
  }
  get biasDeg(): number {
    const b = this.integrator.bias;
    return Math.hypot(b[0], b[1], b[2]) / DEG;
  }

  cameraRotationBetween(t0Ms: number, t1Ms: number): Mat3 | null {
    const r = this.integrator.rotationBetween(t0Ms, t1Ms, this.r);
    if (!r) return null;
    deviceToCameraMatrix(screenAngleDeg(), this.m);
    return conjugateRotation(this.m, r);
  }

  maxRateDegBetween(t0Ms: number, t1Ms: number): number {
    return this.integrator.maxRateBetween(t0Ms, t1Ms) / DEG;
  }
}

function screenAngleDeg(): number {
  if (typeof screen !== "undefined" && screen.orientation && typeof screen.orientation.angle === "number") {
    return screen.orientation.angle;
  }
  const w = (typeof window !== "undefined" ? window : {}) as { orientation?: number };
  return typeof w.orientation === "number" ? w.orientation : 0;
}
