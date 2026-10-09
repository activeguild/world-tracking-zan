/**
 * On-screen HUD (spec §42, 修正指示書 v2 §23, v4 §25, v5 §18, v6 §11–§16, v11 §24).
 *
 * Laid out for a phone in portrait: one short "Label  value" row per item,
 * sections TRACK / MAP / RELOC / PLANE / WORLD / OBJECT / TIMING. The RELOC
 * details (stage counters of the last attempt, best candidate) are shown
 * only while the camera is lost or was just relocalized, and the PLANE
 * pipeline only until the world is established, so normal tracking stays small.
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

/** Stage counters of the last relocalization attempt (v6 §1, §5; v14 §23, §46–§48). */
export interface HudRelocDiagnostics {
  keyframes: number;
  tried: number;
  coarsePassed: number;
  lkPassed: number;
  pnpTested: number;
  pnpPassed: number;
  validated: number;
  bestCoarseScore: number;
  /** v14: ranking of all keyframes (best first), the LK / PnP budget used and the keyframes held back by the retry cooldown. */
  ranked: { keyframeId: number; score: number; selected: boolean; suppressed: boolean; unusable: boolean; alive: number }[];
  lkCandidates: number;
  pnpCandidates: number;
  retrySuppressed: number;
  /** Keyframes with too few live landmarks to relocalize (v15). */
  unusableKeyframes: number;
  usedPreparedRanking: boolean;
  /** The jump limits vs the held pose were in force (v16); false after a long loss. */
  jumpLimitsActive: boolean;
  best: {
    keyframeId: number;
    stage: string;
    inliers: number;
    errorPx: number;
    inlierRatio: number;
    spatialCells: number;
    coarseScore: number;
    /** v14 §49: LK breakdown of the best candidate. */
    lk: { observations: number; tracked: number; fb: number; residual: number; far: number; diverged: number; oob: number; texture: number; reason: string | null };
    /** Per-condition validation of the best candidate (v9 §21), null before PnP. */
    validation: {
      inliers: number;
      requiredInliers: number;
      errorPx: number;
      maxErrorPx: number;
      /** Relaxed error bound and the validation level (v12 §7, §14). */
      relaxedErrorPx: number;
      level: string;
      /** Coarse NCC vs the relaxed-range requirement, jump vs the relocalization limits (v12 §5, §9). */
      ncc: number;
      requiredNcc: number;
      nccPassed: boolean;
      translationJumpPassed: boolean;
      rotationJumpPassed: boolean;
      /** Jump vs the held pose (map units / deg) and the limits in force (Infinity = not checked, v16). */
      translationJump: number;
      rotationJumpDeg: number;
      maxTranslationJump: number;
      maxRotationJumpDeg: number;
      ratio: number;
      minRatio: number;
      cells: number;
      totalCells: number;
      minCells: number;
      coverage: number;
      minCoverage: number;
      inliersPassed: boolean;
      reprojectionPassed: boolean;
      ratioPassed: boolean;
      spatialPassed: boolean;
      coveragePassed: boolean;
      posePassed: boolean;
      rejectReason: string | null;
    } | null;
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
  /** v10 phase (INITIAL_SCAN / SURFACE_SCAN / PLANE_CANDIDATE / WORLD_TRACKING / WORLD_LOST / RELOCALIZING). */
  phase?: string;
  /** World tracking established for the current map (v10 §27). */
  worldEstablished?: boolean;
  visionMs: number;
  /** Camera frames skipped because the vision backend was still busy. */
  framesDropped?: number;
  /** Per-stage vision engine time of the last frame (ms, v16). */
  engineTiming?: { pyramid: number; track: number; ransac: number; detect: number; pose: number; map: number; plane: number; reloc: number; ba: number; total: number } | null;
  /** Phase 7: last local bundle adjustment on the current map (null before the first frame). */
  bundleAdjustment?: {
    runs: number;
    lastFrameId: number;
    keyframes: number;
    freeKeyframes: number;
    landmarks: number;
    observations: number;
    outliers: number;
    reverted: number;
    errorBeforePx: number;
    errorAfterPx: number;
    iterations: number;
    converged: boolean;
    ms: number;
    maxLandmarkShift: number;
    maxKeyframeShift: number;
    maxKeyframeRotationDeg: number;
  } | null;
  /** Main-thread / transport time of the last frame (ms, v16). */
  mainTiming?: {
    grabMs: number;
    captureMs: number;
    queueInMs: number;
    engineMs: number;
    queueOutMs: number;
    roundTripMs: number;
    worldMs: number;
    presentMs: number;
    overlayMs: number;
    resultIntervalMs: number;
  } | null;
  /** Time of the previous HUD refresh (formatting + DOM, ms). */
  hudMs?: number;
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
    /** v16: inlier error by image region / landmark kind (null before a tracked frame). */
    reprojection: {
      centerCount: number;
      centerErrorPx: number;
      edgeCount: number;
      edgeErrorPx: number;
      planeCount: number;
      planeErrorPx: number;
      otherCount: number;
      otherErrorPx: number;
      lowParallaxCount: number;
      lowParallaxErrorPx: number;
      midParallaxCount: number;
      midParallaxErrorPx: number;
      highParallaxCount: number;
      highParallaxErrorPx: number;
      parallaxLowDeg: number;
      parallaxHighDeg: number;
      youngTrackCount: number;
      youngTrackErrorPx: number;
      midTrackCount: number;
      midTrackErrorPx: number;
      oldTrackCount: number;
      oldTrackErrorPx: number;
      youngLandmarkCount: number;
      youngLandmarkErrorPx: number;
      midLandmarkCount: number;
      midLandmarkErrorPx: number;
      oldLandmarkCount: number;
      oldLandmarkErrorPx: number;
      ageYoungFrames: number;
      ageOldFrames: number;
    } | null;
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
  planeSearch?: {
    points: number;
    bestInliers: number;
    minInliers: number;
    threshold: number;
    horizontalness: number;
    /** Stage the search stopped at (v11 §22) and the 2D-extent test values. */
    stage: string;
    inliers: number;
    extentMajor: number;
    extentMinor: number;
    extentRequired: number;
  } | null;
  /** Plane recovery after fast motion (v11 §23–§24). */
  planeRecovery?: {
    active: boolean;
    reason: string | null;
    state: string;
    mapHealthy: boolean;
    mapInliers: number;
    seedCandidates: number;
    seededPoints: number;
    /** This frame's search result (v11.1 §21–§24), not a held earlier candidate. */
    candidateFound: boolean;
    candidateCommitted: boolean;
    previousCandidateHeld: boolean;
    stableFrames: number;
    requiredStableFrames: number;
    elapsedMs: number;
    recoveries: number;
  } | null;
  /** New-landmark triangulation of this frame (why the map does / does not grow). */
  triangulation?: {
    candidates: number;
    parallaxRejected: number;
    cheiralityRejected: number;
    angleRejected: number;
    errorRejected: number;
    depthRejected: number;
    added: number;
  } | null;
  world?: { ready: boolean; scale: number; placed: number } | null;
  /** Plane-relative pose quality (experimental estimator). */
  planePose?: { tracked: boolean; inliers: number; candidates: number; ratio: number; errorPx: number; confidence: number } | null;
  /** Camera world position (m) once the world exists. */
  cameraWorld?: number[] | null;
  /** Placed objects' world positions (m); must not move with the camera. */
  objects?: { id: number; position: number[] }[];
  /** Whether the objects are shown and why not (v13 §15–§16). */
  objectVisibility?: { visible: boolean; reason: string } | null;
  /** Frame capture time, pose arrival time, render-time pose age (ms). */
  timing?: { frameMs: number; poseMs: number; ageMs: number; stale: boolean } | null;
  reloc?: {
    keyframes: number;
    /** Keyframes created / evicted so far (v15 churn). */
    keyframesCreated: number;
    keyframesEvicted: number;
    /** none / candidate / success / fail (v5 §18). */
    attempt: string;
    inliers: number;
    errorPx: number;
    /** strong / acceptable / reject / null (v12 §7). */
    level: string | null;
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
    /** v14 §24, §47: search stage, preparation, pending candidate age and the episode timeline (ms since the loss, −1 = not yet). */
    searchStage: string;
    preparing: boolean;
    pendingKeyframeId: number;
    candidateAgeFrames: number;
    timeline: {
      attempts: number;
      firstCoarseMs: number;
      firstLkMs: number;
      firstPnpMs: number;
      validationMs: number;
      confirmationMs: number;
    } | null;
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
    if (s.phase) {
      rows.push({ text: `Phase  ${s.phase}`, cls: s.phase === "WORLD_LOST" || s.phase === "RELOCALIZING" ? "hud-state hud-warn" : "hud-state" });
    }
    if (s.worldEstablished !== undefined) rows.push(row("World", `Established ${s.worldEstablished ? "YES" : "NO"}`));
    rows.push(
      row("FPS", `${s.renderFps.toFixed(0)} / vis ${s.visionFps.toFixed(0)} (${s.visionMs.toFixed(0)}ms)${s.framesDropped ? `  drop ${s.framesDropped}` : ""}`),
    );
    // v16: where the vision time goes (the engine measured it all along; the
    // 17 → 120 ms climb before the world existed on device had no breakdown).
    // Right under FPS: at the end of the TIMING section it was below the fold
    // of the on-device recording.
    const et = s.engineTiming;
    if (et) {
      const ms = (v: number) => v.toFixed(v >= 10 ? 0 : 1);
      rows.push(
        row(
          "Vis",
          `pyr ${ms(et.pyramid)} lk ${ms(et.track)} rsc ${ms(et.ransac)} fast ${ms(et.detect)} 2view ${ms(et.pose)} map ${ms(et.map)} plane ${ms(et.plane)} reloc ${ms(et.reloc)}${et.ba > 0 ? ` ba ${ms(et.ba)}` : ""} = ${ms(et.total)}ms`,
          et.total > 33 ? "hud-warn" : undefined,
        ),
      );
    }
    // v16: the rest of the frame's wall time (main thread + worker transport).
    // On Android the engine took 22–31 ms but vision ran at 10–14 fps; this
    // row says whether the grab, the display copy, the result waiting for a
    // busy main thread (q↓), the Three.js update or the HUD itself is to blame.
    const mt = s.mainTiming;
    if (mt) {
      const ms = (v: number) => v.toFixed(v >= 10 ? 0 : 1);
      const overhead = mt.roundTripMs - mt.engineMs;
      rows.push(
        row(
          "Main",
          `grab ${ms(mt.grabMs)} cap ${ms(mt.captureMs)} q↑ ${ms(mt.queueInMs)} eng ${ms(mt.engineMs)} q↓ ${ms(mt.queueOutMs)} rtt ${ms(mt.roundTripMs)}  world ${ms(mt.worldMs)} show ${ms(mt.presentMs)} ovl ${ms(mt.overlayMs)} hud ${ms(s.hudMs ?? 0)}  every ${ms(mt.resultIntervalMs)}ms`,
          overhead > 20 ? "hud-warn" : undefined,
        ),
      );
    }

    // ---- TRACK ----
    rows.push(section("TRACK"));
    rows.push(row("Feat", `${s.featureCount} / ${s.trackedCount} / ${s.inlierCount}`));
    if (m) {
      const flags = `${m.relinked ? `  relink ${m.relinked}` : ""}${m.translationPredicted ? "  t PRED" : m.translationHeld ? "  t HELD" : ""}${m.jumpRejected ? "  JUMP" : ""}`;
      rows.push(row("PnP", `${m.pnpInliers}i ${m.reprojPx.toFixed(2)}px${flags}`, m.jumpRejected ? "hud-warn" : undefined));
      // v16: where the error sits. Edge ≫ center → focal length / lens
      // distortion (objects shift while tilting and come back); plane ≠
      // other → biased landmark depths.
      const rb = m.reprojection;
      if (rb && (rb.centerCount || rb.edgeCount)) {
        const ratio = rb.centerCount && rb.edgeCount && rb.centerErrorPx > 0 ? rb.edgeErrorPx / rb.centerErrorPx : 0;
        rows.push(
          row(
            "Err",
            `center ${rb.centerErrorPx.toFixed(2)}px (${rb.centerCount})  edge ${rb.edgeErrorPx.toFixed(2)}px (${rb.edgeCount})${ratio ? `  ×${ratio.toFixed(1)}` : ""}  |  plane ${rb.planeErrorPx.toFixed(2)} (${rb.planeCount})  other ${rb.otherErrorPx.toFixed(2)} (${rb.otherCount})`,
            ratio >= 1.5 ? "hud-warn" : undefined,
          ),
        );
        // By the landmark's triangulation parallax: error concentrated in the
        // low bin = noisy depths (multi-view refinement); uniform = observation noise.
        const lowHigh = rb.lowParallaxCount && rb.highParallaxCount && rb.highParallaxErrorPx > 0 ? rb.lowParallaxErrorPx / rb.highParallaxErrorPx : 0;
        rows.push(
          row(
            "Par",
            `<${rb.parallaxLowDeg}° ${rb.lowParallaxErrorPx.toFixed(2)}px (${rb.lowParallaxCount})  ${rb.parallaxLowDeg}–${rb.parallaxHighDeg}° ${rb.midParallaxErrorPx.toFixed(2)} (${rb.midParallaxCount})  >${rb.parallaxHighDeg}° ${rb.highParallaxErrorPx.toFixed(2)} (${rb.highParallaxCount})${lowHigh ? `  ×${lowHigh.toFixed(1)}` : ""}`,
            lowHigh >= 1.5 ? "hud-warn" : undefined,
          ),
        );
        // By track age vs landmark age: error growing with track age = LK
        // drift of the observation; growing with landmark age only = map
        // inconsistency (bundle adjustment).
        const trackOldYoung = rb.youngTrackCount && rb.oldTrackCount && rb.youngTrackErrorPx > 0 ? rb.oldTrackErrorPx / rb.youngTrackErrorPx : 0;
        const lmOldYoung = rb.youngLandmarkCount && rb.oldLandmarkCount && rb.youngLandmarkErrorPx > 0 ? rb.oldLandmarkErrorPx / rb.youngLandmarkErrorPx : 0;
        const bins = (y: number, yc: number, m: number, mc: number, o: number, oc: number) =>
          `<${rb.ageYoungFrames} ${y.toFixed(2)} (${yc})  ${rb.ageYoungFrames}–${rb.ageOldFrames} ${m.toFixed(2)} (${mc})  >${rb.ageOldFrames} ${o.toFixed(2)} (${oc})`;
        rows.push(
          row(
            "Age",
            `track ${bins(rb.youngTrackErrorPx, rb.youngTrackCount, rb.midTrackErrorPx, rb.midTrackCount, rb.oldTrackErrorPx, rb.oldTrackCount)}${trackOldYoung ? `  ×${trackOldYoung.toFixed(1)}` : ""}`,
            trackOldYoung >= 1.5 ? "hud-warn" : undefined,
          ),
        );
        rows.push(
          row(
            "",
            `lm    ${bins(rb.youngLandmarkErrorPx, rb.youngLandmarkCount, rb.midLandmarkErrorPx, rb.midLandmarkCount, rb.oldLandmarkErrorPx, rb.oldLandmarkCount)}${lmOldYoung ? `  ×${lmOldYoung.toFixed(1)}` : ""}`,
            lmOldYoung >= 1.5 ? "hud-warn" : undefined,
          ),
        );
      }
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
          `${r.attempt}  kf ${r.keyframes} (${r.keyframesCreated} made, ${r.keyframesEvicted} out)  ok×${r.successes}${r.diag && r.diag.age > 0 && r.attempt === "none" ? `  (last ${r.diag.age}f ago)` : ""}`,
          r.attempt === "fail" ? "hud-warn" : undefined,
        ),
      );
    }
    // Phase 7: what the last bundle adjustment did to the map. The error is
    // the keyframe-observation error (not this frame's PnP); a run that does
    // not converge or leaves > 2 px is orange.
    const ba = s.bundleAdjustment;
    if (ba && ba.runs > 0) {
      rows.push(
        row(
          "BA",
          `×${ba.runs}  kf ${ba.freeKeyframes}+1/${ba.keyframes} lm ${ba.landmarks} obs ${ba.observations}${ba.outliers ? ` (+${ba.outliers} out)` : ""}${ba.reverted ? ` rev ${ba.reverted}` : ""}  ${ba.errorBeforePx.toFixed(2)} → ${ba.errorAfterPx.toFixed(2)}px  it ${ba.iterations}${ba.converged ? "" : "!"}  ${ba.ms.toFixed(0)}ms  Δlm ${ba.maxLandmarkShift.toFixed(3)} kf ${ba.maxKeyframeShift.toFixed(3)}u/${ba.maxKeyframeRotationDeg.toFixed(1)}°`,
          !ba.converged || ba.errorAfterPx > 2 ? "hud-warn" : undefined,
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
      // v10 §28: why we are here — only an established world is relocalized into.
      if (lost) rows.push(row("Reason", s.worldEstablished ? "WORLD_LOST" : "scan continues (no world)"));
      // v14 §24, §47: where the search stands (prepare / coarse / lk / pnp /
      // validation / confirming / applied) before the stage counters.
      rows.push(
        row(
          "Stage",
          `${r.searchStage.toUpperCase()}${r.preparing ? " (tracking)" : ""}${r.pendingKeyframeId >= 0 ? `  cand KF${r.pendingKeyframeId} age ${r.candidateAgeFrames}f` : ""}`,
          r.searchStage === "confirming" || r.searchStage === "applied" ? undefined : lost ? "hud-warn" : undefined,
        ),
      );
      const d = r.diag;
      // v14 §48–§49: the LK breakdown of the best candidate says whether the
      // keyframe, the initial guess or the window fails. Right under Stage so
      // it is above the fold on a phone (the on-device recording cut it off).
      const lk = d?.best?.lk;
      if (lk && lk.observations > 0) {
        rows.push(
          row(
            "LKpts",
            `${lk.tracked}/${lk.observations}${lk.reason ? `  ${lk.reason}` : ""}  fb ${lk.fb} res ${lk.residual} far ${lk.far} div ${lk.diverged} oob ${lk.oob} tex ${lk.texture}`,
            lk.reason ? "hud-warn" : undefined,
          ),
        );
      }
      // v16: the verdict on the best candidate right under the LK row, so
      // the reason is above the fold (on device the Best / Reject rows at the
      // end of the section were cut off while a full turn kept failing).
      if (d?.best) {
        const b = d.best;
        const v = b.validation;
        const verdict = v
          ? v.rejectReason
            ? `✗ ${v.rejectReason}`
            : `✓ ${v.level.toUpperCase()}`
          : `✗ ${d.fail ?? b.stage}`;
        rows.push(
          row("Best", `KF${b.keyframeId} ${b.inliers}i ${b.errorPx.toFixed(2)}px  ncc ${b.coarseScore.toFixed(2)}  ${verdict}`, v && !v.rejectReason ? undefined : "hud-warn"),
        );
      }
      if (d) {
        // v14 §23, §36: total keyframes, how many were ranked (all), sent to LK
        // / PnP (the budget), validated, held back by the retry cooldown.
        rows.push(
          row(
            "KF",
            `${d.keyframes}  ranked ${d.ranked.length}  lk ${d.lkCandidates}  pnp ${d.pnpCandidates}  val ${d.validated}${d.retrySuppressed > 0 ? `  retry ${d.retrySuppressed}` : ""}${d.unusableKeyframes > 0 ? `  thin ${d.unusableKeyframes}` : ""}${d.age > 0 ? `  (${d.age}f ago)` : ""}`,
            d.unusableKeyframes > 0 ? "hud-warn" : undefined,
          ),
        );
        if (d.ranked.length > 0) {
          // × = too few live landmarks (v15), ~ = retry cooldown, · = not selected.
          const top = d.ranked
            .slice(0, 4)
            .map((k) => `KF${k.keyframeId} ${k.score.toFixed(2)}${k.unusable ? `×${k.alive}` : k.suppressed ? "~" : k.selected ? "" : "·"}`)
            .join("  ");
          rows.push(row("Rank", `${top}${d.usedPreparedRanking ? "  (prepared)" : ""}`));
        }
        rows.push(row("NCC", `${d.coarsePassed}/${d.tried}  best ${d.bestCoarseScore.toFixed(2)}`));
        rows.push(row("LK", `${d.lkPassed}/${d.lkCandidates}`));
        rows.push(row("PnP", `${d.pnpTested} ran / ${d.pnpPassed} ok`));
        rows.push(row("VAL", `${d.validated}`, d.validated === 0 && d.tried > 0 ? "hud-warn" : undefined));
        if (d.best) {
          const b = d.best;
          const v = b.validation;
          if (v) {
            // v9 §21: every condition with its value, its threshold and PASS / FAIL.
            const ok = (p: boolean) => (p ? "OK" : "NG");
            rows.push(row("Inlier", `${v.inliers}/${v.requiredInliers}  ${ok(v.inliersPassed)}`, v.inliersPassed ? undefined : "hud-warn"));
            // v12 §14: strict bound, then the relaxed bound on its own row when it exists.
            rows.push(row("Error", `${v.errorPx.toFixed(2)}/${v.maxErrorPx.toFixed(2)}px  strict ${ok(v.reprojectionPassed)}`, v.reprojectionPassed ? undefined : "hud-warn"));
            if (v.relaxedErrorPx > v.maxErrorPx) {
              const relaxedOk = v.errorPx <= v.relaxedErrorPx;
              rows.push(row("", `${v.errorPx.toFixed(2)}/${v.relaxedErrorPx.toFixed(2)}px  relaxed ${ok(relaxedOk)}`, relaxedOk ? undefined : "hud-warn"));
            }
            rows.push(row("Ratio", `${v.ratio.toFixed(2)}/${v.minRatio.toFixed(2)}  ${ok(v.ratioPassed)}`, v.ratioPassed ? undefined : "hud-warn"));
            rows.push(row("Cells", `${v.cells}/${v.totalCells} (min ${v.minCells})  ${ok(v.spatialPassed)}`, v.spatialPassed ? undefined : "hud-warn"));
            rows.push(
              row("Cover", `${v.coverage.toFixed(2)}${v.minCoverage > 0 ? `/${v.minCoverage.toFixed(2)}  ${ok(v.coveragePassed)}` : ""}`, v.coveragePassed ? undefined : "hud-warn"),
            );
            // NCC matters for the relaxed range (v12 §5); jump limits are the relocalization-specific ones (§9).
            if (Number.isFinite(v.requiredNcc)) {
              rows.push(row("NCC", `${v.ncc.toFixed(2)}/${v.requiredNcc.toFixed(2)}  ${ok(v.nccPassed)}`, v.nccPassed ? undefined : "hud-warn"));
            }
            // v16: the jump vs the held pose with the limits in force; after a
            // long loss the limits are off and the jump is information only.
            if (Number.isFinite(v.rotationJumpDeg) || Number.isFinite(v.translationJump)) {
              const lim = (x: number) => (Number.isFinite(x) ? x.toFixed(x >= 10 ? 0 : 1) : "∞");
              const ng = !v.translationJumpPassed || !v.rotationJumpPassed;
              rows.push(
                row(
                  "Jump",
                  `${v.translationJump.toFixed(2)}/${lim(v.maxTranslationJump)} u  ${v.rotationJumpDeg.toFixed(0)}/${lim(v.maxRotationJumpDeg)}°  ${ng ? "NG (reloc limit)" : d.jumpLimitsActive ? "OK" : "no limit (long loss)"}`,
                  ng ? "hud-warn" : undefined,
                ),
              );
            }
            if (!v.posePassed) rows.push(row("Pose", "invalid", "hud-warn"));
            rows.push(row("Level", v.level.toUpperCase(), v.level === "reject" ? "hud-warn" : undefined));
            rows.push(row("Reject", v.rejectReason ?? "-", v.rejectReason ? "hud-warn" : undefined));
          } else {
            // Dropped out before PnP: the stage says where.
            rows.push(row("Stage", b.stage));
            if (d.fail) rows.push(row("Reject", d.fail, "hud-warn"));
          }
        } else if (d.fail) {
          rows.push(row("Reject", d.fail, "hud-warn"));
        }
      }
      if (r.attempt === "candidate" || r.attempt === "success") {
        rows.push(
          row(
            r.attempt === "success" ? "Apply" : "Cand",
            `KF${r.keyframeId} ${r.inliers}i ${r.errorPx.toFixed(2)}px${r.level === "acceptable" ? " ACCEPTABLE" : r.level === "strong" ? " STRONG" : ""} match ${r.match.toFixed(2)}`,
          ),
        );
        rows.push(row("Jump", `${cm(r.jumpM)} / ${r.jumpDeg.toFixed(1)}°`));
      }
      if (r.attempt === "fail" && r.rejectCode === "confirmation_failed") rows.push(row("Fail", "confirmation", "hud-warn"));
      if (r.attempt === "fail" && r.rejectCode === "stale_candidate") rows.push(row("Fail", "stale candidate", "hud-warn"));
      if (r.attempt === "fail" && r.rejectCode === "retry_cooldown") rows.push(row("Fail", "retry cooldown (view unchanged)", "hud-warn"));
      if (r.postM > 0 || r.postInconsistent) {
        rows.push(row("Post", `map Δ ${cm(r.postM)} / ${r.postDeg.toFixed(1)}°${r.postInconsistent ? "  INCONSISTENT" : ""}`, r.postInconsistent ? "hud-warn" : undefined));
      }
      // v14 §45–§47, §55: ms since the loss at which each stage first succeeded.
      const tl = r.timeline;
      if (tl) {
        const t = (ms: number) => (ms < 0 ? "-" : `${Math.round(ms)}`);
        rows.push(row("Time", `try ${tl.attempts}  ncc ${t(tl.firstCoarseMs)}  lk ${t(tl.firstLkMs)}  pnp ${t(tl.firstPnpMs)}  val ${t(tl.validationMs)}  apply ${t(tl.confirmationMs)} ms`));
      }
    }

    // ---- PLANE (v11 §22, §24): the detection pipeline stage by stage,
    // search → best → candidate → commit → stable, plus the recovery state.
    // Shown while a map exists and no world has been established yet.
    const ps = s.planeSearch;
    const pr = s.planeRecovery;
    if (ps && m && !s.world?.ready) {
      rows.push(section("PLANE"));
      if (pr) {
        const reason = pr.reason === "fast_motion" ? "FAST" : pr.reason === "two_view_motion" ? "TWO_VIEW" : pr.reason === "insufficient_plane_points" ? "PLANE_POINTS" : pr.reason === "manual" ? "MANUAL" : "";
        rows.push(
          row(
            "Recov",
            pr.active
              ? `${reason}  ${pr.state}  ${(pr.elapsedMs / 1000).toFixed(1)}s${pr.recoveries > 1 ? `  ×${pr.recoveries}` : ""}`
              : `—${pr.recoveries ? `  (done ×${pr.recoveries})` : ""}`,
            pr.active ? "hud-warn" : undefined,
          ),
        );
        rows.push(row("Map", `${pr.mapInliers}i  ${pr.mapHealthy ? "healthy" : "not located"}`, pr.mapHealthy ? undefined : "hud-warn"));
        rows.push(row("Seed", pr.active ? `${pr.seededPoints} of ${pr.seedCandidates} recent (window)` : `${pr.seededPoints}  (${pr.seedCandidates} recent)`));
      }
      rows.push(row("Search", `${ps.points}pt  best ${ps.bestInliers}/${ps.minInliers}  thr ${ps.threshold.toFixed(3)}`));
      const stageText =
        ps.stage === "candidate"
          ? `candidate  ${ps.inliers}i  s2 ${ps.extentMinor.toFixed(2)} ≥ ${ps.extentRequired.toFixed(2)}`
          : ps.stage === "extent"
            ? `extent  s2 ${ps.extentMinor.toFixed(2)} < ${ps.extentRequired.toFixed(2)}  (s1 ${ps.extentMajor.toFixed(2)})`
            : ps.stage === "reclassify"
              ? `reclassify  ${ps.inliers} < ${ps.minInliers}  (window ${ps.bestInliers})`
              : ps.stage === "support"
                ? `support  ${ps.bestInliers} < ${ps.minInliers}`
                : `points  ${ps.points} < ${ps.minInliers}`;
      rows.push(row("Stage", stageText, ps.stage === "candidate" ? undefined : "hud-warn"));
      // This frame's candidate (v11.1 §21–§24); a candidate only *held* from
      // an earlier frame is shown as such, not as found.
      const foundNow = pr ? pr.candidateFound : ps.stage === "candidate" && !!p;
      const committed = pr ? pr.candidateCommitted : foundNow && !!p && p.horizontal;
      const heldOnly = pr ? pr.previousCandidateHeld : !foundNow && !!p;
      rows.push(
        row(
          "Cand",
          `${foundNow ? "YES" : "NO"}  commit ${committed ? "YES" : "NO"}${foundNow && p && !p.horizontal ? `  (hz ${p.horizontalness.toFixed(2)})` : ""}${heldOnly ? "  (held)" : ""}`,
        ),
      );
      rows.push(
        row(
          "Stable",
          `${pr ? pr.stableFrames : p ? p.stableFrames : 0}/${pr ? pr.requiredStableFrames : "?"}${p ? `  conf ${p.confidence.toFixed(2)}` : ""}${p?.found ? "  FOUND" : ""}`,
        ),
      );
      if (s.triangulation) {
        const t = s.triangulation;
        rows.push(
          row(
            "Tri",
            `cand ${t.candidates}  +${t.added}  par ${t.parallaxRejected} ang ${t.angleRejected} err ${t.errorRejected} dep ${t.depthRejected}${t.cheiralityRejected ? ` chi ${t.cheiralityRejected}` : ""}`,
          ),
        );
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
    } else if (ps && s.world?.ready) {
      rows.push(row("Plane", `fixed  ${ps.points}pt best ${ps.bestInliers}/${ps.minInliers} thr ${ps.threshold.toFixed(3)}`));
    }
    if (s.planePose) {
      const pp = s.planePose;
      rows.push(row("PPose", pp.tracked ? `${pp.inliers}/${pp.candidates} (${pp.ratio.toFixed(2)}) ${pp.errorPx.toFixed(2)}px conf ${pp.confidence.toFixed(2)}` : `— (${pp.candidates}pt)`));
    }
    rows.push(row("Grav", s.gravityAvailable ? "yes" : "no (up = −Y)"));

    // ---- OBJECT ----
    rows.push(section("OBJECT"));
    const ov = s.objectVisibility;
    if (ov) {
      rows.push(row("Visible", ov.visible ? "YES" : "NO", ov.visible ? undefined : "hud-warn"));
      rows.push(row("Reason", ov.reason));
    }
    if (s.objects && s.objects.length) {
      // Transforms are kept while hidden (v13 §12); the HUD says "hidden" instead of a stale-looking position.
      for (const o of s.objects.slice(0, 2)) rows.push(row(`Obj${o.id}`, ov && !ov.visible ? "hidden" : xyz(o.position)));
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
