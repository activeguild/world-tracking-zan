#!/usr/bin/env node
// Model-based test input: random walks of the Quint models (fixed seed, so
// the output is reproducible), reduced to the walks that exercise a model
// transition no earlier walk did. tests/quint replays them against the
// TypeScript implementation.
//
//   node scripts/quint-traces.mjs            → writes quint/conformance/*-traces.json
//   node scripts/quint-traces.mjs --check    → exit 1 when a checked-in file is stale
import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const SEED = "0x5e55101";

const tag = (v) => v.tag;
const int = (v) => Number(v["#bigint"]);
const some = (o) => (o.tag === "Some" ? o.value : null);

// ---- session.qnt ----------------------------------------------------------
const request = (r) => (r.tag === "Frame" ? `Frame${int(r.value)}` : r.tag);
const response = (r) =>
  r.tag === "Result" ? `Result${int(r.value.id)}` : r.tag === "FrameError" ? `FrameError${int(r.value)}` : r.tag;
// The part of the model state the implementation can be asked about.
// Counters only mean something during a run, stopRequested only during a start.
const sessionView = (s) => {
  const phase = tag(s.phase);
  const run = phase === "Running";
  const starting = phase === "StartingCamera" || phase === "StartingBackend";
  return [
    phase,
    starting ? (s.stopRequested ? "stop" : "go") : "-",
    s.cameraOn ? "cam" : "nocam",
    tag(s.backend),
    s.busy ? "busy" : "free",
    run ? `n${int(s.nextFrameId)}` : "-",
    run ? `rb${int(s.resetBeforeFrame)}` : "-",
    run ? `a${int(s.applied)}` : "-",
    s.toWorker.map(request).join(","),
    s.toMain.map(response).join(","),
  ].join("|");
};
const sessionPick = (p) => {
  if (p.ok.tag === "Some") return p.ok.value ? "ok" : "fail";
  if (p.outcome.tag === "Some") return p.outcome.value.tag;
  return "";
};

// ---- mapcorrection.qnt ----------------------------------------------------
const tags = (l) => `[${l.map(int).join(",")}]`;
const anchorOf = (o) => {
  const a = some(o);
  return a ? `m${int(a.mapId)}${tags(a.view)}` : "-";
};
const mapOf = (o) => {
  const m = some(o);
  return m ? `m${int(m.id)}${tags(m.corr)}` : "-";
};
const tagOf = (o) => {
  const t = some(o);
  return t === null ? "-" : String(int(t));
};
// anchor | requests "id+" (with anchor points) / "id-" | results "id;map;stage tag;ba tag;plane"
const mapCorrectionView = (s) =>
  [
    anchorOf(s.anchor),
    s.toEngine.map((r) => `${int(r.id)}${some(r.anchor) ? "+" : "-"}`).join(","),
    s.toSession
      .map((r) => [int(r.id), mapOf(r.map), tagOf(r.stage), tagOf(r.ba), r.planeFound ? "p" : "-"].join(";"))
      .join(","),
  ].join("|");
// The engine's event is fully described by the result it appends; the walk carries no pick.
const mapCorrectionPick = () => "";

// ---- tracking.qnt ---------------------------------------------------------
const bit = (b) => (b ? "1" : "0");
const idOf = (o) => {
  const v = some(o);
  return v === null ? "-" : String(int(v));
};
// inliers | what the state machine saw: mapInitialized,mapLost,planeFound,worldEstablished
// | state | map | framesSinceTracked | relocalized | plane detected | world established
// | session world | objects visible
const trackingView = (s) =>
  [
    tag(s.inliers)[0],
    [bit(some(s.mapId) !== null), bit(s.obsMapLost), bit(s.obsPlaneFound), bit(s.obsWorldEstablished)].join(""),
    tag(s.state),
    idOf(s.mapId),
    int(s.framesSinceTracked),
    bit(s.relocalized),
    bit(s.planeDetected),
    bit(s.worldEstablished),
    idOf(s.world),
    bit(s.visible),
  ].join("|");

// The engine's relocalization bookkeeping (RelocalizationLedger), same model.
// frame inputs: located, relocalization result N/S/I/A, near
// | map | framesSinceTracked | pending "age.confirmations" | failed attempts (capped)
// | monitor frames left | relocalized | applied this frame | world established
// | frames since the PnP located the camera (capped)
const pickOf = (o) => (o && o.tag === "Some" ? o.value : null);
const ledgerView = (s, picks) => {
  const p = some(s.pending);
  const reloc = pickOf(picks?.reloc);
  return [
    picks ? [bit(pickOf(picks.located)), reloc ? { NoCandidate: "N", Strong: "S", StrongImmediate: "I", Acceptable: "A" }[tag(reloc)] : "N", bit(pickOf(picks.near))].join("") : "---",
    idOf(s.mapId),
    int(s.framesSinceTracked),
    p ? `${int(p.age)}.${int(p.confirmations)}` : "-",
    int(s.attempts),
    int(s.monitorLeft),
    bit(s.relocalized),
    bit(int(s.sinceApply) === 0),
    bit(s.worldEstablished),
    int(s.lostFrames),
  ].join("|");
};

const MODELS = [
  {
    name: "session",
    samples: 1500,
    maxSteps: 30,
    // Two mixes of the same model: every action, and the walks without stop()
    // (a sub-behaviour of `step`) that get deep into a run.
    steps: ["step", "stepWithoutStop"],
    view: sessionView,
    pick: sessionPick,
    legend: "phase|stopRequested|camera|backend|busy|nextFrameId|resetBeforeFrame|applied|toWorker|toMain",
  },
  {
    name: "mapcorrection",
    samples: 1500,
    maxSteps: 16,
    steps: ["step"],
    view: mapCorrectionView,
    pick: mapCorrectionPick,
    legend: "anchor map[corrections]|requests id+/- (anchor points)|results id;map[corrections];stage tag;ba tag;plane found",
  },
  {
    name: "tracking",
    samples: 700,
    maxSteps: 24,
    steps: ["step"],
    view: trackingView,
    // One action, and the frame's inputs that matter are in the state it
    // leads to: a step is just that state.
    compact: true,
    pick: () => "",
    legend: "inliers L/M/H|state machine saw mapInitialized,mapLost,planeFound,worldEstablished|state|map|framesSinceTracked|relocalized|plane detected|world established|session world|visible",
  },
  {
    name: "relocalization",
    spec: "tracking",
    samples: 700,
    maxSteps: 24,
    steps: ["step"],
    view: ledgerView,
    compact: true,
    pick: () => "",
    legend: "inputs located,reloc N/S/I/A,near|map|framesSinceTracked|pending age.confirmations|failed attempts (capped)|monitor frames left|relocalized|applied|world established|lost frames (capped)",
  },
];

function generate(model) {
  const dir = mkdtempSync(join(tmpdir(), "quint-traces-"));
  try {
    model.steps.forEach((step, i) => {
      execFileSync(
        join(root, "node_modules/.bin/quint"),
        [
          "run", join(root, `quint/${model.spec ?? model.name}.qnt`),
          "--backend", "typescript", "--mbt", "--invariant", "safety", "--step", step,
          "--max-steps", String(model.maxSteps), "--max-samples", String(model.samples), "--n-traces", String(model.samples),
          "--seed", SEED, "--out-itf", join(dir, `${"abcdef"[i]}_{seq}.itf.json`),
        ],
        { stdio: ["ignore", "ignore", "inherit"] },
      );
    });
    const files = readdirSync(dir).filter((f) => f.endsWith(".itf.json"));
    const order = (f) => (f.charCodeAt(0) - 97) * 1e9 + Number(f.match(/\d+/)[0]);
    files.sort((a, b) => order(a) - order(b));
    const seen = new Set();
    const traces = [];
    for (const f of files) {
      const states = JSON.parse(readFileSync(join(dir, f), "utf8")).states;
      const steps = [];
      let last = -1;
      for (let i = 1; i < states.length; i++) {
        const picks = states[i]["mbt::nondetPicks"];
        const step = [states[i]["mbt::actionTaken"], model.pick(picks), model.view(states[i].s, picks)];
        const key = `${model.view(states[i - 1].s, states[i - 1]["mbt::nondetPicks"])} ${step.join(" ")}`;
        if (!seen.has(key)) {
          seen.add(key);
          last = i - 1;
        }
        steps.push(model.compact ? step[2] : step);
      }
      if (last >= 0) traces.push(steps.slice(0, last + 1));
    }
    const head = {
      model: `quint/${model.spec ?? model.name}.qnt`,
      seed: SEED,
      samples: model.samples,
      maxSteps: model.maxSteps,
      view: model.legend,
      transitions: seen.size,
    };
    const text =
      `{\n` +
      Object.entries(head).map(([k, v]) => `  ${JSON.stringify(k)}: ${JSON.stringify(v)},\n`).join("") +
      `  "traces": [\n${traces.map((t) => `    ${JSON.stringify(t)}`).join(",\n")}\n  ]\n}\n`;
    const steps = traces.reduce((n, t) => n + t.length, 0);
    return { text, summary: `${model.name}: ${traces.length} traces, ${steps} steps, ${seen.size} distinct transitions` };
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

let stale = false;
for (const model of MODELS) {
  const { text, summary } = generate(model);
  const file = join(root, `quint/conformance/${model.name}-traces.json`);
  if (process.argv.includes("--check")) {
    if (!existsSync(file) || readFileSync(file, "utf8") !== text) {
      console.error(`quint-traces: quint/conformance/${model.name}-traces.json is stale. Regenerate with: npm run quint:export`);
      stale = true;
    }
  } else {
    writeFileSync(file, text);
    console.error(`quint-traces: ${summary}`);
  }
}
if (stale) process.exit(1);
