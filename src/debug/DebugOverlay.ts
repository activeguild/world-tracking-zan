/**
 * On-screen HUD (spec §42, 修正指示書 v2 §23):
 *
 *   TRACKING  features / tracked / inliers / PnP quality
 *   CAMERA    map-frame and world-frame camera centers, per-frame Δ
 *   WORLD     scale, landmark distribution (plane / non-plane), plane
 *   OBJECT    world positions of the placed objects (must not follow the camera)
 *   TIMING    frame timestamp, pose timestamp, pose age
 */
/** One pose candidate on the HUD (v4 §25). */
export interface HudCandidate {
  inliers: number;
  errorPx: number;
  /** Continuity vs the previous canonical pose (world meters, NaN before the world exists / deg). */
  deltaM: number;
  deltaDeg: number;
  trusted: boolean;
  /** Rejection code (null when accepted) and its measured value / limit in display units. */
  rejectCode: string | null;
  rejectDetail: string | null;
}

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
    translationPredicted: boolean;
    jumpRejected: boolean;
    source: string;
    /** Pose-source diagnostics (v3 §19–§21, v4 §25). */
    mapInliers: number;
    planeInliers: number;
    rejectReason: string | null;
    /** Per-candidate diagnostics; deltas in world meters (NaN before the world exists). */
    mapCandidate: HudCandidate | null;
    planeCandidate: HudCandidate | null;
    /** Temporal-gate limits in force (world meters / deg). */
    gateMaxM: number;
    gateMaxDeg: number;
    /** Map vs plane candidate difference in world meters (NaN before the world exists). */
    sourceDeltaM: number;
    sourceDeltaDeg: number;
    history: string;
    /** Landmarks re-linked to tracks this frame. */
    relinked: number;
    /** Pose re-seeded by a relocalization (this frame or its monitoring window). */
    relocalized: boolean;
  } | null;
  /** How long tracking has been lost (ms). */
  lostMs?: number;
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
  reloc?: {
    keyframes: number;
    /** none / candidate / success / fail (v5 §18). */
    attempt: string;
    inliers: number;
    errorPx: number;
    match: number;
    inlierRatio: number;
    spatialCells: number;
    keyframeId: number;
    successes: number;
    reason: string | null;
    rejectCode: string | null;
    /** Candidate / applied pose distance from the held pose (world meters, NaN before the world exists) / deg. */
    jumpM: number;
    jumpDeg: number;
    /** Map PnP vs relocalized pose over the frames after the last relocalization. */
    postM: number;
    postDeg: number;
    postInconsistent: boolean;
  } | null;
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

/** `in 42  err 1.80px  Δ 1.2 cm / 0.8°  TRUSTED  reject translation_jump (31.0 > 8.0 cm)` */
function candidate(c: HudCandidate | null): string {
  if (!c) return "—";
  const d = Number.isFinite(c.deltaM) ? `${(c.deltaM * 100).toFixed(1)} cm` : "—";
  return (
    `in ${c.inliers}  err ${c.errorPx.toFixed(2)}px  Δ ${d} / ${c.deltaDeg.toFixed(1)}°` +
    (c.trusted ? "  TRUSTED" : "") +
    `  reject ${c.rejectCode ? `${c.rejectCode}${c.rejectDetail ? ` (${c.rejectDetail})` : ""}` : "-"}`
  );
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
        ? `PnP         inliers ${m.pnpInliers}  err ${m.reprojPx.toFixed(2)}px  lost ${m.framesSinceTracked}${m.relinked ? `  relink ${m.relinked}` : ""}${m.translationPredicted ? "  t PRED" : m.translationHeld ? "  t HELD" : ""}${m.jumpRejected ? "  JUMP" : ""}`
        : `PnP         —`,
      `SOURCE      ${m ? `${m.framesSinceTracked > 0 ? "LOST" : m.source.toUpperCase()}  ${m.history.slice(-24)}` : "—"}`,
      `MAP cand    ${m ? candidate(m.mapCandidate) : "—"}`,
      `PLANE cand  ${m ? candidate(m.planeCandidate) : "—"}`,
      m
        ? `Gate        ≤ ${Number.isFinite(m.gateMaxM) ? `${(m.gateMaxM * 100).toFixed(1)} cm` : "—"} / ${m.gateMaxDeg.toFixed(0)}°  map↔plane Δ ${Number.isFinite(m.sourceDeltaM) && m.planeInliers ? `${(m.sourceDeltaM * 100).toFixed(1)} cm / ${m.sourceDeltaDeg.toFixed(1)}°` : "—"}`
        : `Gate        —`,
      `LOST        ${s.lostMs !== undefined ? `${s.lostMs.toFixed(0)} ms` : "—"}`,
      ...(s.pose
        ? [
            `Pose R      yaw ${fmt(s.pose.yaw)}°  pitch ${fmt(s.pose.pitch)}°  roll ${fmt(s.pose.roll)}°`,
            `Pose 2view  ${s.pose.model}  t (${s.pose.translationDirection.map((v) => v.toFixed(2)).join(", ")})  parallax ${s.pose.parallaxPx.toFixed(1)}px  conf ${s.pose.confidence.toFixed(2)}/${s.pose.translationConfidence.toFixed(2)}  n=${s.pose.correspondences}`,
          ]
        : [`Pose        —`]),
      `RELOC       ${s.reloc ? `${s.reloc.attempt}  kf ${s.reloc.keyframes}  ok×${s.reloc.successes}${m?.relocalized ? "  RELOCALIZED" : ""}` : "—"}`,
      ...(s.reloc && s.reloc.attempt !== "none"
        ? [
            `  cand      in ${s.reloc.inliers}  err ${s.reloc.errorPx.toFixed(2)}px  match ${s.reloc.match.toFixed(2)}  ratio ${s.reloc.inlierRatio.toFixed(2)}  cells ${s.reloc.spatialCells}/9  kf ${s.reloc.keyframeId}`,
            `  jump      ${Number.isFinite(s.reloc.jumpM) ? `${(s.reloc.jumpM * 100).toFixed(1)} cm` : "—"} / ${s.reloc.jumpDeg.toFixed(1)}°`,
          ]
        : []),
      ...(s.reloc?.attempt === "fail" && (s.reloc.rejectCode || s.reloc.reason)
        ? [`  REJECT    ${s.reloc.rejectCode ?? ""}${s.reloc.reason ? `  ${s.reloc.reason}` : ""}`]
        : []),
      ...(s.reloc && (s.reloc.postM > 0 || s.reloc.postInconsistent)
        ? [
            `  post      map Δ ${Number.isFinite(s.reloc.postM) ? `${(s.reloc.postM * 100).toFixed(1)} cm` : "—"} / ${s.reloc.postDeg.toFixed(1)}°${s.reloc.postInconsistent ? "  INCONSISTENT" : ""}`,
          ]
        : []),
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
