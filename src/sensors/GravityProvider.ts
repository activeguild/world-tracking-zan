/**
 * Gravity direction from DeviceMotion, expressed in the camera frame
 * (X right, Y down, Z forward).
 *
 * Phase 3 uses this only to decide whether a detected plane is horizontal
 * (spec §22). Full IMU fusion is Phase 6. The sign of
 * `accelerationIncludingGravity` differs between iOS and Android; the plane
 * test is sign-agnostic (|cos|), so no attempt is made to normalize it.
 */
export class GravityProvider {
  private raw: Float64Array | null = null;
  private filtered: Float64Array | null = null;
  private listening = false;
  private readonly onMotion = (ev: DeviceMotionEvent) => {
    const a = ev.accelerationIncludingGravity;
    if (!a || a.x == null || a.y == null || a.z == null) return;
    this.raw = new Float64Array([a.x, a.y, a.z]);
    if (!this.filtered) {
      this.filtered = Float64Array.from(this.raw);
    } else {
      const k = this.alpha;
      for (let i = 0; i < 3; i++) this.filtered[i] = this.filtered[i] * (1 - k) + this.raw[i] * k;
    }
  };

  /**
   * @param alpha EMA weight per DeviceMotion sample (~60 Hz). 0.06 ≈ 0.3 s
   *        time constant: hand accelerations average out, slow tilts follow.
   */
  constructor(private readonly alpha = 0.06) {}

  static isSupported(): boolean {
    return typeof window !== "undefined" && "DeviceMotionEvent" in window;
  }

  /**
   * Request permission (iOS 13+ requires a user gesture) and start
   * listening. Resolves to false when unsupported or denied.
   */
  async start(): Promise<boolean> {
    if (!GravityProvider.isSupported()) return false;
    const DME = DeviceMotionEvent as unknown as { requestPermission?: () => Promise<"granted" | "denied"> };
    if (typeof DME.requestPermission === "function") {
      try {
        const res = await DME.requestPermission();
        if (res !== "granted") return false;
      } catch {
        return false;
      }
    }
    window.addEventListener("devicemotion", this.onMotion);
    this.listening = true;
    return true;
  }

  stop(): void {
    if (this.listening) window.removeEventListener("devicemotion", this.onMotion);
    this.listening = false;
  }

  get hasReading(): boolean {
    return this.filtered !== null;
  }

  /**
   * Gravity in the camera frame of the displayed video, or null.
   *
   * Device frame (W3C): X right, Y up, Z out of the screen. The back camera
   * looks along −Z, so camera = (x_s, −y_s, −z) after rotating (x, y) by the
   * screen orientation angle so that the vector follows the upright video.
   */
  get gravityCamera(): Float64Array | null {
    if (!this.filtered) return null;
    const [x, y, z] = this.filtered;
    const angle = screenAngleDeg();
    const rad = (angle * Math.PI) / 180;
    const c = Math.cos(rad);
    const s = Math.sin(rad);
    const xs = c * x - s * y;
    const ys = s * x + c * y;
    return new Float64Array([xs, -ys, -z]);
  }
}

function screenAngleDeg(): number {
  if (typeof screen !== "undefined" && screen.orientation && typeof screen.orientation.angle === "number") {
    return screen.orientation.angle;
  }
  const w = window as unknown as { orientation?: number };
  return typeof w.orientation === "number" ? w.orientation : 0;
}

/** Parse a `?gravity=x,y,z` override (camera frame) for testing. */
export function parseGravityOverride(value: string | null): Float64Array | null {
  if (!value) return null;
  const parts = value.split(",").map((v) => Number(v));
  if (parts.length !== 3 || parts.some((v) => !Number.isFinite(v))) return null;
  return Float64Array.from(parts);
}
