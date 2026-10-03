/**
 * On-screen HUD (spec §42):
 *
 *   FPS / Vision FPS / Feature Count / Tracked Count / Inlier Count /
 *   Plane Confidence / Tracking State / Pose (reserved) / Vision ms
 */
export interface HudStats {
  renderFps: number;
  visionFps: number;
  featureCount: number;
  trackedCount: number;
  inlierCount: number;
  planeConfidence: number;
  state: string;
  visionMs: number;
  fastThreshold: number;
  processingSize: string;
  backend: string;
  message?: string;
}

export class DebugOverlay {
  private readonly el: HTMLElement;
  private readonly lines: HTMLElement[] = [];

  constructor(parent: HTMLElement) {
    const el = document.createElement("div");
    el.className = "ar-hud";
    parent.appendChild(el);
    this.el = el;
  }

  set visible(v: boolean) {
    this.el.style.display = v ? "block" : "none";
  }

  update(s: HudStats): void {
    const rows: string[] = [
      `FPS         ${s.renderFps.toFixed(0)}`,
      `Vision FPS  ${s.visionFps.toFixed(0)}  (${s.visionMs.toFixed(1)} ms)`,
      `Features    ${s.featureCount}`,
      `Tracked     ${s.trackedCount}`,
      `Inliers     ${s.inlierCount}`,
      `Plane conf  ${s.planeConfidence.toFixed(2)}`,
      `State       ${s.state}`,
      `Pose        (phase 2)`,
      `FAST thr    ${s.fastThreshold}`,
      `Proc size   ${s.processingSize}  [${s.backend}]`,
    ];
    if (s.message) rows.push("", s.message);
    this.setLines(rows);
  }

  private setLines(rows: string[]): void {
    while (this.lines.length < rows.length) {
      const d = document.createElement("div");
      this.el.appendChild(d);
      this.lines.push(d);
    }
    for (let i = 0; i < this.lines.length; i++) {
      const text = rows[i] ?? "";
      if (this.lines[i].textContent !== text) this.lines[i].textContent = text;
    }
  }
}
