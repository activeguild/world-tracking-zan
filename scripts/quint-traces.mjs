#!/usr/bin/env node
// Model-based test input: random walks of quint/session.qnt (fixed seed, so
// the output is reproducible), reduced to the walks that exercise a model
// transition no earlier walk did. tests/quint replays them against the
// TypeScript session / worker client / worker handler.
//
//   node scripts/quint-traces.mjs            → writes quint/conformance/session-traces.json
//   node scripts/quint-traces.mjs --stdout   → prints it (used by quint-check.sh)
import { execFileSync } from "node:child_process";
import { mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const SEED = "0x5e55101";
const SAMPLES = 1500;
const MAX_STEPS = 30;

const dir = mkdtempSync(join(tmpdir(), "quint-traces-"));
try {
  // Two mixes of the same model: every action, and the walks without stop()
  // (a sub-behaviour of `step`) that get deep into a run.
  for (const [prefix, step] of [["a", "step"], ["b", "stepWithoutStop"]]) {
    execFileSync(
      join(root, "node_modules/.bin/quint"),
      [
        "run", join(root, "quint/session.qnt"),
        "--backend", "typescript", "--mbt", "--invariant", "safety", "--step", step,
        "--max-steps", String(MAX_STEPS), "--max-samples", String(SAMPLES), "--n-traces", String(SAMPLES),
        "--seed", SEED, "--out-itf", join(dir, `${prefix}_{seq}.itf.json`),
      ],
      { stdio: ["ignore", "ignore", "inherit"] },
    );
  }

  const tag = (v) => v.tag;
  const int = (v) => Number(v["#bigint"]);
  const request = (r) => (r.tag === "Frame" ? `Frame${int(r.value)}` : r.tag);
  const response = (r) =>
    r.tag === "Result" ? `Result${int(r.value.id)}` : r.tag === "FrameError" ? `FrameError${int(r.value)}` : r.tag;
  // The part of the model state the implementation can be asked about.
  // Counters only mean something during a run, stopRequested only during a start.
  const view = (s) => {
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
  const pick = (p) => {
    if (p.ok.tag === "Some") return p.ok.value ? "ok" : "fail";
    if (p.outcome.tag === "Some") return p.outcome.value.tag;
    return "";
  };

  const files = readdirSync(dir).filter((f) => f.endsWith(".itf.json"));
  const order = (f) => (f[0] === "a" ? 0 : 1e9) + Number(f.match(/\d+/)[0]);
  files.sort((a, b) => order(a) - order(b));
  const seen = new Set();
  const traces = [];
  for (const f of files) {
    const states = JSON.parse(readFileSync(join(dir, f), "utf8")).states;
    const steps = [];
    let last = -1;
    for (let i = 1; i < states.length; i++) {
      const step = [states[i]["mbt::actionTaken"], pick(states[i]["mbt::nondetPicks"]), view(states[i].s)];
      const key = `${view(states[i - 1].s)} ${step.join(" ")}`;
      if (!seen.has(key)) {
        seen.add(key);
        last = i - 1;
      }
      steps.push(step);
    }
    if (last >= 0) traces.push(steps.slice(0, last + 1));
  }
  const out = {
    model: "quint/session.qnt",
    seed: SEED,
    samples: SAMPLES,
    maxSteps: MAX_STEPS,
    view: "phase|stopRequested|camera|backend|busy|nextFrameId|resetBeforeFrame|applied|toWorker|toMain",
    transitions: seen.size,
    traces,
  };
  const text =
    `{\n` +
    Object.entries(out)
      .filter(([k]) => k !== "traces")
      .map(([k, v]) => `  ${JSON.stringify(k)}: ${JSON.stringify(v)},\n`)
      .join("") +
    `  "traces": [\n${traces.map((t) => `    ${JSON.stringify(t)}`).join(",\n")}\n  ]\n}\n`;
  if (process.argv.includes("--stdout")) process.stdout.write(text);
  else {
    writeFileSync(join(root, "quint/conformance/session-traces.json"), text);
    const steps = traces.reduce((n, t) => n + t.length, 0);
    console.error(`quint-traces: ${traces.length} traces, ${steps} steps, ${seen.size} distinct transitions`);
  }
} finally {
  rmSync(dir, { recursive: true, force: true });
}
