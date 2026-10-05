/**
 * On-screen HUD (spec §42, 修正指示書 v2 §23, v4 §25, v5 §18, v6 §11–§16).
 *
 * Laid out for a phone in portrait: one short "Label  value" row per item,
 * sections TRACK / MAP / RELOC / WORLD / OBJECT / TIMING. The RELOC details
 * (stage counters of the last attempt, best candidate) are shown only while
 * the camera is lost or was just relocalized, so normal tracking stays small.
 *
 * Every number is the engine's own value under an unambiguous label: `MAP`
 * rows are the map PnP candidate of the current frame, `RELOC` rows are the
 * relocalization attempt / its best keyframe candidate.
 */
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

/** Stage counters of the last relocalization attempt (v6 §1, §5). */
export interface HudRelocDiagnostics {
  keyframes: number;
  tried: number;
  coarsePassed: number;
  lkPassed: number;
  pnpTested: number;
  pnpPassed: number;
  validated: number;
  bestCoarseScore: number;
  best: {
    keyframeId: number;
    stage: string;
    inliers: number;
    errorPx: number;
    inlierRatio: number;
    spatialCells: number;
    coarseScore: number;
  } | null;
  /** Short reject code of the best candidate's stage (v6 §6), null when it passed. */
  fail: string | null;
  /** Frames since the attempt (0 = this frame). */
  age: number;
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
    /** Map PnP recovery status (v6 §10). */
    observations: number;
    requiredInliers: number;
    recoveryMode: string;
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
    /** Last attempt of the current lost episode (null while tracking). */
    diag: HudRelocDiagnostics | null;
  } | null;
  /** Frame-to-frame motion level and LK diagnostics (v7 §12–§13). */
  motion?: {
    level: string;
    medianPx: number;
    maxPx: number;
    before: number;
    after: number;
    fbRejects: number;
    tooFar: number;
    prediction: string;
    searchScale: number;
  } | null;
  /** Build identifier (phase + commit + time) so testers can confirm the deployed version. */
  build?: string;
}

interface Row {
  text: string;
  cls?: string;
}

function fmt(deg: number): string {
  return (deg >= 0 ? "+" : "") + deg.toFixed(1);
}

function xyz(p: ArrayLike<number>): string {
  const f = (v: number) => (v >= 0 ? " " : "") + v.toFixed(3);
  return `${f(p[0])} ${f(p[1])} ${f(p[2])}`;
}

/** World length in cm, "—" before the world exists. */
function cm(m: number): string {
  return Number.isFinite(m) ? `${(m * 100).toFixed(1)}cm` : "—";
}

/** `14i 3.18px Δ 1.2cm/0.8° TRUSTED` */
function candidate(c: HudCandidate): string {
  return `${c.inliers}i ${c.errorPx.toFixed(2)}px  Δ ${cm(c.deltaM)}/${c.deltaDeg.toFixed(1)}°${c.trusted ? "  TRUSTED" : ""}`;
}

function row(label: string, value: string, cls?: string): Row {
  return { text: `${label.padEnd(6)} ${value}`, cls };
}

function section(name: string): Row {
  return { text: `=== ${name} ===`, cls: "hud-section" };
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
    const r = s.reloc;
    const lost = !!m && m.framesSinceTracked > 0;
    const rows: Row[] = [];

    // ---- always ----
    rows.push({ text: `State  ${s.state}`, cls: "hud-state" });
    rows.push(
      row("FPS", `${s.renderFps.toFixed(0)} / vis ${s.visionFps.toFixed(0)} (${s.visionMs.toFixed(0)}ms)${s.framesDropped ? `  drop ${s.framesDropped}` : ""}`),
    );

    // ---- TRACK ----
    rows.push(section("TRACK"));
    rows.push(row("Feat", `${s.featureCount} / ${s.trackedCount} / ${s.inlierCount}`));
    if (m) {
      const flags = `${m.relinked ? `  relink ${m.relinked}` : ""}${m.translationPredicted ? "  t PRED" : m.translationHeld ? "  t HELD" : ""}${m.jumpRejected ? "  JUMP" : ""}`;
      rows.push(row("PnP", `${m.pnpInliers}i ${m.reprojPx.toFixed(2)}px${flags}`, m.jumpRejected ? "hud-warn" : undefined));
      rows.push(row("Source", `${lost ? "LOST" : m.source.toUpperCase()}${m.relocalized ? " (RELOC)" : ""}  ${m.history.slice(-20)}`));
    } else {
      rows.push(row("PnP", "—"));
    }
    rows.push(row("Lost", s.lostMs !== undefined && s.lostMs > 0 ? `${s.lostMs.toFixed(0)}ms` : "0ms", lost ? "hud-warn" : undefined));
    if (s.motion) {
      const mo = s.motion;
      const lvl = mo.level === "fast" ? "FAST" : mo.level === "medium" ? "MED" : "N";
      rows.push(
        row(
          "Motion",
          `${lvl}  ${mo.medianPx.toFixed(1)}px (max ${mo.maxPx.toFixed(0)})  ${mo.prediction === "homography" ? "H" : mo.prediction === "velocity" ? "V" : "-"}${mo.searchScale !== 1 ? ` ×${mo.searchScale}` : ""}`,
          mo.level === "fast" ? "hud-warn" : undefined,
        ),
      );
      rows.push(row("LK", `${mo.after}/${mo.before}${mo.fbRejects ? `  fb ${mo.fbRejects}` : ""}${mo.tooFar ? `  far ${mo.tooFar}` : ""}`));
    }
    if (s.pose) {
      rows.push(row("Rot", `yaw ${fmt(s.pose.yaw)} pitch ${fmt(s.pose.pitch)} roll ${fmt(s.pose.roll)}`));
      rows.push(
        row("2view", `${s.pose.model} par ${s.pose.parallaxPx.toFixed(0)}px conf ${s.pose.confidence.toFixed(2)}/${s.pose.translationConfidence.toFixed(2)} n${s.pose.correspondences}`),
      );
    }
    if (r) {
      rows.push(
        row(
          "Reloc",
          `${r.attempt}  kf ${r.keyframes}  ok×${r.successes}${r.diag && r.diag.age > 0 && r.attempt === "none" ? `  (last ${r.diag.age}f ago)` : ""}`,
          r.attempt === "fail" ? "hud-warn" : undefined,
        ),
      );
    }

    // ---- MAP (current frame's map PnP candidate) ----
    if (m) {
      rows.push(section("MAP"));
      if (m.mapCandidate) {
        const c = m.mapCandidate;
        rows.push(row("Cand", candidate(c)));
        if (c.rejectCode) rows.push(row("Reject", `${c.rejectCode}${c.rejectDetail ? ` (${c.rejectDetail})` : ""}`, "hud-warn"));
      } else {
        rows.push(row("Cand", "—"));
      }
      // Recovery rule in force (v6 §10): which inlier count the candidate needs.
      rows.push(
        row(
          "Need",
          `${m.mapCandidate ? m.mapCandidate.inliers : 0}/${m.requiredInliers}i  obs ${m.observations}  (${m.recoveryMode})`,
          lost ? "hud-warn" : undefined,
        ),
      );
      rows.push(row("Gate", `${cm(m.gateMaxM)} / ${m.gateMaxDeg.toFixed(0)}°`));
      if (m.planeCandidate) {
        const c = m.planeCandidate;
        rows.push(row("Plane", candidate(c)));
        if (c.rejectCode) rows.push(row("PRej", `${c.rejectCode}${c.rejectDetail ? ` (${c.rejectDetail})` : ""}`));
        rows.push(row("M↔P", `${cm(m.sourceDeltaM)} / ${m.sourceDeltaDeg.toFixed(1)}°`));
      }
    }

    // ---- RELOC (only while lost, during an attempt, or right after a relocalization) ----
    if (r && (lost || r.attempt !== "none" || r.diag || m?.relocalized)) {
      rows.push(section("RELOC"));
      const d = r.diag;
      if (d) {
        rows.push(row("KF", `${d.keyframes} / try ${d.tried}${d.age > 0 ? `  (${d.age}f ago)` : ""}`));
        rows.push(row("NCC", `${d.coarsePassed}  best ${d.bestCoarseScore.toFixed(2)}`));
        rows.push(row("LK", `${d.lkPassed}`));
        rows.push(row("PnP", `${d.pnpTested} ran / ${d.pnpPassed} ok`));
        rows.push(row("VAL", `${d.validated}`, d.validated === 0 && d.tried > 0 ? "hud-warn" : undefined));
        if (d.best) {
          const b = d.best;
          rows.push(row("Best", `KF${b.keyframeId} ${b.inliers}i ${b.errorPx.toFixed(2)}px`));
          rows.push(row("Ratio", `${b.inlierRatio.toFixed(2)}  cells ${b.spatialCells}/9  ncc ${b.coarseScore.toFixed(2)}`));
          rows.push(row("Stage", b.stage));
        }
        if (d.fail) rows.push(row("Fail", d.fail, "hud-warn"));
      }
      if (r.attempt === "candidate" || r.attempt === "success") {
        rows.push(row(r.attempt === "success" ? "Apply" : "Cand", `KF${r.keyframeId} ${r.inliers}i ${r.errorPx.toFixed(2)}px match ${r.match.toFixed(2)}`));
        rows.push(row("Jump", `${cm(r.jumpM)} / ${r.jumpDeg.toFixed(1)}°`));
      }
      if (r.attempt === "fail" && r.rejectCode === "confirmation_failed") rows.push(row("Fail", "confirmation", "hud-warn"));
      if (r.postM > 0 || r.postInconsistent) {
        rows.push(row("Post", `map Δ ${cm(r.postM)} / ${r.postDeg.toFixed(1)}°${r.postInconsistent ? "  INCONSISTENT" : ""}`, r.postInconsistent ? "hud-warn" : undefined));
      }
    }

    // ---- WORLD ----
    rows.push(section("WORLD"));
    rows.push(row("World", s.world?.ready ? `ready  ${s.world.scale.toFixed(3)} m/u  obj ${s.world.placed}` : "—"));
    if (s.cameraWorld) rows.push(row("Cam", xyz(s.cameraWorld)));
    if (m) {
      rows.push(row("mapC", xyz(m.cameraCenter)));
      rows.push(row("Δcam", `${cm(m.deltaTranslationM)} / ${m.deltaRotationDeg.toFixed(2)}°`));
      rows.push(row("LM", `${m.landmarks}  plane ${p ? p.inliers : 0}  other ${p ? Math.max(0, m.landmarks - p.inliers) : m.landmarks}`));
    }
    if (p) {
      rows.push(row("Plane", `n(${p.normal.map((v) => v.toFixed(2)).join(",")}) rms ${p.rms.toFixed(3)}`));
      rows.push(row("", `hz ${p.horizontalness.toFixed(2)}${p.horizontal ? " H" : ""}  st ${p.stableFrames}  conf ${p.confidence.toFixed(2)}${p.found ? "  FOUND" : ""}`));
    } else if (s.planeSearch) {
      rows.push(row("Plane", `${s.world?.ready ? "fixed" : "search"} ${s.planeSearch.points}pt best ${s.planeSearch.bestInliers}/${s.planeSearch.minInliers} thr ${s.planeSearch.threshold.toFixed(3)}`));
    }
    if (s.planePose) {
      const pp = s.planePose;
      rows.push(row("PPose", pp.tracked ? `${pp.inliers}/${pp.candidates} (${pp.ratio.toFixed(2)}) ${pp.errorPx.toFixed(2)}px conf ${pp.confidence.toFixed(2)}` : `— (${pp.candidates}pt)`));
    }
    rows.push(row("Grav", s.gravityAvailable ? "yes" : "no (up = −Y)"));

    // ---- OBJECT ----
    rows.push(section("OBJECT"));
    if (s.objects && s.objects.length) {
      for (const o of s.objects.slice(0, 2)) rows.push(row(`Obj${o.id}`, xyz(o.position)));
    } else {
      rows.push(row("Obj", "—"));
    }

    // ---- TIMING ----
    rows.push(section("TIMING"));
    if (s.timing) {
      rows.push(row("Pose", `age ${s.timing.ageMs.toFixed(0)}ms${s.timing.stale ? "  STALE" : ""}`, s.timing.stale ? "hud-warn" : undefined));
    }
    rows.push(row("Proc", `FAST ${s.fastThreshold}  ${s.processingSize}  [${s.backend}]`));
    if (s.build) rows.push(row("Build", s.build));
    if (s.message) rows.push({ text: s.message });
    this.setLines(rows);
  }

  private setLines(rows: Row[]): void {
    while (this.lines.length < rows.length) {
      const d = document.createElement("div");
      this.el.appendChild(d);
      this.lines.push(d);
    }
    for (let i = 0; i < this.lines.length; i++) {
      const r = rows[i];
      const text = r?.text ?? "";
      const cls = r?.cls ?? "";
      const el = this.lines[i];
      if (el.textContent !== text) el.textContent = text;
      if (el.className !== cls) el.className = cls;
      const show = i < rows.length ? "" : "none";
      if (el.style.display !== show) el.style.display = show;
    }
  }
}
