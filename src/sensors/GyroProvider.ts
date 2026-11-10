/**
 * Gyro (DeviceMotion `rotationRate`) → rotation of the camera between two
 * frame times, for the vision ↔ gyro diagnostic. Nothing here feeds the
 * tracker.
 *
 * W3C: `rotationRate.alpha / beta / gamma` are deg/s about the device Z / X /
 * Y axes. Event times are `event.timeStamp` (same clock as
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
}

const DEG = Math.PI / 180;

export class GyroProvider implements GyroSource {
  readonly integrator: GyroIntegrator;
  private listening = false;
  private readonly m = new Float64Array(9);
  private readonly r = new Float64Array(9);
  private readonly onMotion = (ev: DeviceMotionEvent) => {
    const w = ev.rotationRate;
    if (!w || w.alpha == null || w.beta == null || w.gamma == null) return;
    const now = performance.now();
    const t = Math.abs(ev.timeStamp - now) < 1000 ? ev.timeStamp : now;
    this.integrator.add(t, w.beta * DEG, w.gamma * DEG, w.alpha * DEG);
  };

  constructor(opts?: GyroIntegratorOptions) {
    this.integrator = new GyroIntegrator(opts);
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
