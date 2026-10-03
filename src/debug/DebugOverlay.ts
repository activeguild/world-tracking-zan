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
  /** Phase 2 pose (null until available). */
  pose?: {
    yaw: number;
    pitch: number;
    roll: number;
    translationDirection: number[];
    model: string;
    parallaxPx: number;
    confidence: number;
    translationConfidence: number;
    correspondences: number;
  } | null;
  /** Phase 3 map + plane. */
  map?: {
    landmarks: number;
    pnpInliers: number;
    reprojPx: number;
    translation: number[];
    framesSinceTracked: number;
  } | null;
  plane?: {
    normal: number[];
    inliers: number;
    horizontalness: number;
    horizontal: boolean;
    stableFrames: number;
    confidence: number;
    found: boolean;
    usedGravity: boolean;
  } | null;
  gravityAvailable?: boolean;
  world?: { ready: boolean; scale: number; placed: number } | null;
  reloc?: { keyframes: number; attempt: string; inliers: number; successes: number } | null;
  /** Build identifier (phase + commit + time) so testers can confirm the deployed version. */
  build?: string;
}

function fmt(deg: number): string {
  return (deg >= 0 ? "+" : "") + deg.toFixed(1);
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
      ...(s.pose
        ? [
            `Pose R      yaw ${fmt(s.pose.yaw)}°  pitch ${fmt(s.pose.pitch)}°  roll ${fmt(s.pose.roll)}°`,
            `Pose t      (${s.pose.translationDirection.map((v) => v.toFixed(2)).join(", ")})`,
            `Pose model  ${s.pose.model}  parallax ${s.pose.parallaxPx.toFixed(1)}px  n=${s.pose.correspondences}`,
            `Pose conf   ${s.pose.confidence.toFixed(2)}  t-conf ${s.pose.translationConfidence.toFixed(2)}`,
          ]
        : [`Pose        —`]),
      ...(s.map
        ? [
            `Map         ${s.map.landmarks} lm  pnp ${s.map.pnpInliers}  err ${s.map.reprojPx.toFixed(2)}px  lost ${s.map.framesSinceTracked}`,
            `Map t       (${s.map.translation.map((v) => v.toFixed(2)).join(", ")})`,
          ]
        : [`Map         —`]),
      ...(s.plane
        ? [
            `Plane n     (${s.plane.normal.map((v) => v.toFixed(2)).join(", ")})  in ${s.plane.inliers}`,
            `Plane       hz ${s.plane.horizontalness.toFixed(2)} ${s.plane.horizontal ? "H" : "-"}  stable ${s.plane.stableFrames}  conf ${s.plane.confidence.toFixed(2)}  ${s.plane.found ? "FOUND" : ""}`,
          ]
        : [`Plane       —`]),
      `Gravity     ${s.gravityAvailable ? "yes" : "no (fallback up = −Y)"}`,
      `World       ${s.world?.ready ? `ready  scale ${s.world.scale.toFixed(3)} m/unit  objects ${s.world.placed}` : "—"}`,
      `Keyframes   ${s.reloc ? `${s.reloc.keyframes}  reloc ${s.reloc.attempt}${s.reloc.attempt === "success" ? ` (${s.reloc.inliers})` : ""}  ok×${s.reloc.successes}` : "—"}`,
      `FAST thr    ${s.fastThreshold}`,
      `Proc size   ${s.processingSize}  [${s.backend}]`,
      ...(s.build ? [`Build       ${s.build}`] : []),
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
