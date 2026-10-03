/**
 * `[AR]` debug logger (spec §72).
 *
 * Disabled by default; enabled through `ARConfig.debug.log` (or `?debug=1`
 * in the demo). Production builds keep the logger but never enable it.
 */
export interface LogFields {
  state: string;
  features: number;
  tracked: number;
  inliers: number;
  planeConfidence: number;
  visionFPS: number;
  visionMs: number;
  /** Plane-relative pose quality (修正指示書 §9) and pose source, once anchored. */
  planeInliers?: number;
  planeErrorPx?: number;
  poseSource?: string;
  /** Render time − frame capture time (§17–§18). */
  poseAgeMs?: number;
}

export class ARLogger {
  private lastLog = 0;

  constructor(
    public enabled: boolean,
    private readonly intervalMs: number,
  ) {}

  /** Log at most once per interval. */
  periodic(now: number, fields: LogFields): void {
    if (!this.enabled) return;
    if (now - this.lastLog < this.intervalMs) return;
    this.lastLog = now;
    console.log(
      `[AR]\nstate=${fields.state}\nfeatures=${fields.features}\ntracked=${fields.tracked}\n` +
        `inliers=${fields.inliers}\nplaneConfidence=${fields.planeConfidence.toFixed(2)}\n` +
        `visionFPS=${fields.visionFPS.toFixed(0)}\nvisionMs=${fields.visionMs.toFixed(1)}` +
        (fields.poseSource !== undefined
          ? `\nposeSource=${fields.poseSource}\nplaneInliers=${fields.planeInliers ?? 0}\nplaneErrorPx=${(fields.planeErrorPx ?? 0).toFixed(2)}`
          : "") +
        (fields.poseAgeMs !== undefined ? `\nposeAgeMs=${fields.poseAgeMs.toFixed(0)}` : ""),
    );
  }

  info(message: string): void {
    if (this.enabled) console.log(`[AR] ${message}`);
  }

  warn(message: string): void {
    if (this.enabled) console.warn(`[AR] ${message}`);
  }
}
