import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { WorldAnchor } from "../../src/ar/WorldAnchor";
import type { WorldFrame } from "../../src/math/CoordinateSystem";
import type { RigidTransform } from "../../src/math/Pose";
import type { MapPoseOutput, PlaneOutput } from "../../src/vision/types";
import type { FrameRequest, ResultResponse } from "../../src/worker/protocol";
import { FakeWorker, Harness, stubBrowser } from "./harness";

/**
 * `ARSession` and `WorldAnchor` against the Quint model of map corrections
 * (quint/mapcorrection.qnt).
 *
 * The model's walks say which frames are sent, what the engine's map does in
 * each (init, reset, an accepted init stage and / or keyframe BA, a found
 * plane), and where the anchor must be after each result. Here the real
 * session runs on the fake browser of the lifecycle test; the engine's
 * results are a real result with the map fields of the walk written in.
 *
 * A correction tag of the model is a concrete rigid transform (large, about
 * different axes, so that order matters). The world frame the session holds
 * must equal a reference anchor that was created from the same result and
 * moved by exactly the model's tags, in the model's order.
 *
 * Regenerate the walks with `npm run quint:export` after changing the model.
 */

interface TraceFile {
  transitions: number;
  traces: [action: string, pick: string, view: string][][];
}

const file = JSON.parse(
  readFileSync(fileURLToPath(new URL("../../quint/conformance/mapcorrection-traces.json", import.meta.url)), "utf8"),
) as TraceFile;

/** The rigid correction a model tag stands for. */
function correctionOf(tag: number): RigidTransform {
  const a = ((20 + 7 * tag) * Math.PI) / 180;
  const c = Math.cos(a);
  const s = Math.sin(a);
  const rotation =
    tag % 3 === 0
      ? [1, 0, 0, 0, c, -s, 0, s, c]
      : tag % 3 === 1
        ? [c, 0, s, 0, 1, 0, -s, 0, c]
        : [c, -s, 0, s, c, 0, 0, 0, 1];
  return { rotation: new Float64Array(rotation), translation: new Float64Array([0.1 * (tag + 1), -0.05, 0.2 + 0.03 * tag]) };
}

const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];
const mapFrameIdOf = (modelId: number) => 100 + modelId;

/** Camera at the map origin looking along +Z; the floor one unit below it (map Y is down). */
function mapPoseOf(modelMapId: number): MapPoseOutput {
  return {
    rotation: [...IDENTITY],
    translation: [0, 0, 0],
    inlierCount: 100,
    meanReprojectionErrorPx: 0.5,
    landmarkCount: 0,
    mapFrameId: mapFrameIdOf(modelMapId),
    framesSinceTracked: 0,
    cameraCenter: [0, 0, 0],
    source: "map",
    relocalized: false,
    translationHeld: false,
    sourceDeltaTranslation: 0,
    sourceDeltaRotationDeg: 0,
    mapCandidate: null,
    planeCandidate: null,
  } as unknown as MapPoseOutput;
}

const FLOOR = { normal: [0, -1, 0], d: 1, center: [0, 1, 3], found: true, areaEstimate: 4 } as unknown as PlaneOutput;

interface ModelAnchor {
  mapId: number;
  view: number[];
}
interface ModelResult {
  id: number;
  mapId: number | null;
  stage: number | null;
  ba: number | null;
  planeFound: boolean;
}

function parseAnchor(text: string): ModelAnchor | null {
  if (text === "-") return null;
  const m = /^m(\d+)\[([\d,]*)\]$/.exec(text)!;
  return { mapId: Number(m[1]), view: m[2] ? m[2].split(",").map(Number) : [] };
}

/** `id;map;stage;ba;plane` (scripts/quint-traces.mjs). The corrections in a map's list never contain `;`. */
function parseResult(text: string): ModelResult {
  const [id, map, stage, ba, plane] = text.split(";");
  return {
    id: Number(id),
    mapId: parseAnchor(map)?.mapId ?? null,
    stage: stage === "-" ? null : Number(stage),
    ba: ba === "-" ? null : Number(ba),
    planeFound: plane === "p",
  };
}

/** Results are separated by `,`, and so are the tags inside `[...]`: split outside brackets. */
function splitResults(text: string): string[] {
  if (!text) return [];
  const out: string[] = [];
  let depth = 0;
  let cur = "";
  for (const ch of text) {
    if (ch === "[") depth++;
    if (ch === "]") depth--;
    if (ch === "," && depth === 0) {
      out.push(cur);
      cur = "";
    } else cur += ch;
  }
  out.push(cur);
  return out;
}

function expectSameFrame(got: WorldFrame, want: WorldFrame, where: string): void {
  const close = (a: ArrayLike<number>, b: ArrayLike<number>, what: string) => {
    for (let i = 0; i < b.length; i++) {
      if (Math.abs(a[i] - b[i]) > 1e-9) throw new Error(`${where}: world ${what}[${i}] is ${a[i]}, the model's corrections give ${b[i]}`);
    }
  };
  close(got.plane.origin, want.plane.origin, "origin");
  close(got.rotation, want.rotation, "rotation");
  close(got.plane.matrix, want.plane.matrix, "matrix");
  if (Math.abs(got.scale - want.scale) > 1e-12) throw new Error(`${where}: world scale ${got.scale} ≠ ${want.scale}`);
}

describe("map corrections × world anchor conform to the Quint model", () => {
  beforeEach(stubBrowser);
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("has walks to replay", () => {
    expect(file.traces.length).toBeGreaterThan(50);
    expect(file.transitions).toBeGreaterThan(300);
  });

  it("after every result the anchor is the model's: same map, same corrections, same order", async () => {
    let steps = 0;
    let corrected = 0;
    let both = 0;
    for (let t = 0; t < file.traces.length; t++) {
      // A running session on the worker backend, and one real result as the template.
      const h = new Harness();
      await h.apply("start", "");
      await h.apply("cameraSettles", "ok");
      await h.apply("workerHandles", "Ok");
      await h.apply("mainHandles", "");
      await h.apply("videoFrame", "");
      await h.apply("workerHandles", "Ok");
      const worker = FakeWorker.live!;
      const template = worker.responses[0] as ResultResponse;
      expect(template.type).toBe("result");
      await h.apply("mainHandles", "");
      const firstFrameId = h.internals.nextFrameId;

      const session = h.session as unknown as { worldAnchor: WorldAnchor; config: { world: { assumedPlaneDistanceMeters: number } } };
      const reference = new WorldAnchor({ assumedPlaneDistanceMeters: session.config.world.assumedPlaneDistanceMeters });
      let anchor: ModelAnchor | null = null;
      const pending: ResultResponse[] = [];
      const done: string[] = [];

      for (const [action, , view] of file.traces[t]) {
        const [anchorText, requestsText, resultsText] = view.split("|");
        const where = `walk ${t}, step ${done.length + 1} (${[...done, action].join(" → ")})`;
        if (action === "sendFrame") {
          await h.apply("videoFrame", "");
          const sent = worker.requests[worker.requests.length - 1] as FrameRequest;
          const want = requestsText.split(",").pop()!;
          expect(`${sent.frameId - firstFrameId}${sent.anchorPoints ? "+" : "-"}`, where).toBe(want);
          // The anchor points are the world origin in the map as the anchor has it now.
          if (sent.anchorPoints) {
            const o = reference.frame!.plane.origin;
            for (let i = 0; i < 3; i++) expect(Math.abs(sent.anchorPoints[i] - o[i]), where).toBeLessThan(1e-9);
          }
        } else if (action === "engineProcesses") {
          const req = worker.requests.shift() as FrameRequest;
          const r = parseResult(splitResults(resultsText).pop()!);
          expect(req.frameId - firstFrameId, where).toBe(r.id);
          const stage = r.stage !== null ? correctionOf(r.stage) : correctionOf(7);
          const ba = r.ba !== null ? correctionOf(r.ba) : correctionOf(8);
          pending.push({
            ...template,
            frameId: req.frameId,
            gray: req.gray,
            mapPose: r.mapId === null ? null : mapPoseOf(r.mapId),
            plane: r.planeFound ? FLOOR : null,
            planeAnchor: null,
            // A BA that did not move the map either did not run or was rejected.
            // The rejected run carries a correction here that must not be applied.
            bundleAdjustment: {
              ...template.bundleAdjustment,
              ranThisFrame: r.ba !== null || req.frameId % 2 === 0,
              rejected: r.ba === null,
              rejectReason: r.ba === null ? "no_gain" : "",
              errorBeforePx: 1,
              errorAfterPx: 1,
              maxKeyframeShift: 0,
              maxKeyframeRotationDeg: 0,
              maxLandmarkShift: 0,
              correctionRotation: ba.rotation,
              correctionTranslation: ba.translation,
            },
            // Without a stage in this frame the record is an older frame's
            // (accepted, with a correction): it must not be applied again.
            initRefinement: {
              ...(template.initRefinement ?? {}),
              lastStage: {
                stage: 1,
                frameId: r.stage !== null ? req.frameId : req.frameId - 1,
                applied: true,
                rejectReason: "",
                rotationDeg: 0,
                cameraShift: 0,
                correctionRotation: stage.rotation,
                correctionTranslation: stage.translation,
              },
            },
          } as unknown as ResultResponse);
          if (r.stage !== null && r.ba !== null) both++;
        } else if (action === "handleResult") {
          const res = pending.shift()!;
          worker.onmessage!({ data: res });
          const next = parseAnchor(anchorText);
          if (next === null) {
            reference.reset();
          } else if (anchor !== null && anchor.mapId === next.mapId && reference.isReady) {
            // Kept: moved by the tags the model appended, in its order.
            expect(next.view.slice(0, anchor.view.length), where).toEqual(anchor.view);
            for (const tag of next.view.slice(anchor.view.length)) {
              reference.applyMapCorrection(correctionOf(tag));
              corrected++;
            }
          } else {
            // Created on this result's plane and pose (after the frame's corrections).
            reference.reset();
            reference.create(FLOOR, res.mapPose!);
          }
          anchor = next;
          expect(session.worldAnchor.isReady, where).toBe(next !== null);
          if (next !== null) {
            expect(session.worldAnchor.attachedMapFrameId, where).toBe(mapFrameIdOf(next.mapId));
            expectSameFrame(session.worldAnchor.frame!, reference.frame!, where);
          }
        } else {
          throw new Error(`unknown model action ${action}`);
        }
        done.push(action);
        steps++;
      }
      h.session.stop();
    }
    expect(steps).toBeGreaterThan(1000);
    expect(corrected).toBeGreaterThan(50);
    // The case the ordering matters for is in the walks.
    expect(both).toBeGreaterThan(3);
  }, 300_000);
});
