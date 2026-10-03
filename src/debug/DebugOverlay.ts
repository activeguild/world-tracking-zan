/**
 * On-screen HUD (spec §42, 修正指示書 v2 §23):
 *
 *   TRACKING  features / tracked / inliers / PnP quality
 *   CAMERA    map-frame and world-frame camera centers, per-frame Δ
 *   WORLD     scale, landmark distribution (plane / non-plane), plane
 *   OBJECT    world positions of the placed objects (must not follow the camera)
 *   TIMING    frame timestamp, pose timestamp, pose age
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
  /** Camera frames skipped because the vision backend was still busy. */
  framesDropped?: number;
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
  /** Phase 3 map + PnP telemetry (v2 §7). */
  map?: {
    landmarks: number;
    pnpInliers: number;
    reprojPx: number;
    cameraCenter: number[];
    framesSinceTracked: number;
    /** Camera center displacement since the previous frame, world meters (NaN before the world exists). */
    deltaTranslationM: number;
    deltaRotationDeg: number;
    translationHeld: boolean;
    jumpRejected: boolean;
    source: string;
  } | null;
  plane?: {
    normal: number[];
    inliers: number;
    rms: number;
    horizontalness: number;
    horizontal: boolean;
    stableFrames: number;
    confidence: number;
    found: boolean;
    usedGravity: boolean;
  } | null;
  gravityAvailable?: boolean;
  planeSearch?: { points: number; bestInliers: number; minInliers: number; threshold: number; horizontalness: number } | null;
  world?: { ready: boolean; scale: number; placed: number } | null;
  /** Plane-relative pose quality (experimental estimator). */
  planePose?: { tracked: boolean; inliers: number; candidates: number; ratio: number; errorPx: number; confidence: number } | null;
  /** Camera world position (m) once the world exists. */
  cameraWorld?: number[] | null;
  /** Placed objects' world positions (m); must not move with the camera. */
  objects?: { id: number; position: number[] }[];
  /** Frame capture time, pose arrival time, render-time pose age (ms). */
  timing?: { frameMs: number; poseMs: number; ageMs: number; stale: boolean } | null;
  reloc?: { keyframes: number; attempt: string; inliers: number; successes: number } | null;
  /** Build identifier (phase + commit + time) so testers can confirm the deployed version. */
  build?: string;
}

function fmt(deg: number): string {
  return (deg >= 0 ? "+" : "") + deg.toFixed(1);
}

function xyz(p: ArrayLike<number>): string {
  const f = (v: number) => (v >= 0 ? " " : "") + v.toFixed(3);
  return `X ${f(p[0])}  Y ${f(p[1])}  Z ${f(p[2])}`;
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
    const m = s.map;
    const p = s.plane;
    const nonPlane = m && p ? Math.max(0, m.landmarks - p.inliers) : null;
    const rows: string[] = [
      `FPS ${s.renderFps.toFixed(0)}  Vision ${s.visionFps.toFixed(0)} (${s.visionMs.toFixed(1)} ms)${s.framesDropped !== undefined ? `  drop ${s.framesDropped}` : ""}`,
      `State       ${s.state}`,
      `=== TRACKING ===`,
      `Features    ${s.featureCount}  tracked ${s.trackedCount}  inliers ${s.inlierCount}`,
      m
        ? `PnP         inliers ${m.pnpInliers}  err ${m.reprojPx.toFixed(2)}px  lost ${m.framesSinceTracked}  [${m.source}]${m.translationHeld ? "  t HELD" : ""}${m.jumpRejected ? "  JUMP" : ""}`
        : `PnP         —`,
      ...(s.pose
        ? [
            `Pose R      yaw ${fmt(s.pose.yaw)}°  pitch ${fmt(s.pose.pitch)}°  roll ${fmt(s.pose.roll)}°`,
            `Pose 2view  ${s.pose.model}  t (${s.pose.translationDirection.map((v) => v.toFixed(2)).join(", ")})  parallax ${s.pose.parallaxPx.toFixed(1)}px  conf ${s.pose.confidence.toFixed(2)}/${s.pose.translationConfidence.toFixed(2)}  n=${s.pose.correspondences}`,
          ]
        : [`Pose        —`]),
      `Keyframes   ${s.reloc ? `${s.reloc.keyframes}  reloc ${s.reloc.attempt}${s.reloc.attempt === "success" ? ` (${s.reloc.inliers})` : ""}  ok×${s.reloc.successes}` : "—"}`,
      `=== CAMERA ===`,
      `map C       ${m ? xyz(m.cameraCenter) : "—"}`,
      `world C     ${s.cameraWorld ? xyz(s.cameraWorld) : "—"}`,
      m
        ? `Δ           t ${Number.isFinite(m.deltaTranslationM) ? `${(m.deltaTranslationM * 100).toFixed(1)} cm` : "—"}  rot ${m.deltaRotationDeg.toFixed(2)}°`
        : `Δ           —`,
      `=== WORLD ===`,
      `World       ${s.world?.ready ? `ready  scale ${s.world.scale.toFixed(3)} m/unit  objects ${s.world.placed}` : "—"}`,
      `Landmarks   ${m ? `${m.landmarks}  plane ${p ? p.inliers : 0}  non-plane ${nonPlane ?? m.landmarks}` : "—"}`,
      ...(p
        ? [
            `Plane n     (${p.normal.map((v) => v.toFixed(2)).join(", ")})  rms ${p.rms.toFixed(4)}`,
            `Plane       hz ${p.horizontalness.toFixed(2)} ${p.horizontal ? "H" : "-"}  stable ${p.stableFrames}  conf ${p.confidence.toFixed(2)}  ${p.found ? "FOUND" : ""}`,
          ]
        : [
            s.planeSearch
              ? `Plane       ${s.world?.ready ? "fixed (world)" : "—"} search: ${s.planeSearch.points} pts, best ${s.planeSearch.bestInliers}/${s.planeSearch.minInliers}, thr ${s.planeSearch.threshold.toFixed(3)}`
              : `Plane       —`,
          ]),
      ...(s.planePose
        ? [
            `Plane pose  ${
              s.planePose.tracked
                ? `in ${s.planePose.inliers}/${s.planePose.candidates} (${s.planePose.ratio.toFixed(2)})  err ${s.planePose.errorPx.toFixed(2)}px  conf ${s.planePose.confidence.toFixed(2)}`
                : `— (${s.planePose.candidates} pts)`
            }`,
          ]
        : []),
      `Gravity     ${s.gravityAvailable ? "yes" : "no (fallback up = −Y)"}`,
      `=== OBJECT ===`,
      ...(s.objects && s.objects.length
        ? s.objects.slice(0, 2).map((o) => `Object ${o.id}    ${xyz(o.position)}`)
        : [`Object      —`]),
      `=== TIMING ===`,
      s.timing
        ? `frame t ${s.timing.frameMs.toFixed(0)}  pose t ${s.timing.poseMs.toFixed(0)}  age ${s.timing.ageMs.toFixed(0)} ms${s.timing.stale ? "  POSE STALE" : ""}`
        : `Timing      —`,
      `FAST thr ${s.fastThreshold}  proc ${s.processingSize}  [${s.backend}]`,
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
