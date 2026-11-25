import { describe, expect, it } from "vitest";
import { WorldAnchor } from "../../src/ar/WorldAnchor";
import { buildWorldDiagnostics, planeNormalDeltaDeg, worldFrameDelta, worldFrameValuesDiffer } from "../../src/ar/WorldDiagnostics";
import { rotationAxisAngle } from "../../src/math/Pose";
import { emptyBundleAdjustment, emptyRelinkStats, emptyReprojectionBreakdown, type MapPoseOutput, type PlaneOutput } from "../../src/vision/types";
import { emptyQuality } from "../../src/vision/TrackingQuality";
import { poseFromCenter } from "../helpers/scene";

/** 指示書 §8: the per-frame world-tracking record and the anchor delta it reports. */
function mapPose(mapFrameId = 0): MapPoseOutput {
  const p = poseFromCenter(new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]), [0, 0, 0]);
  return {
    rotation: Array.from(p.rotation),
    translation: Array.from(p.translation),
    inlierCount: 100,
    meanReprojectionErrorPx: 0.4,
    landmarkCount: 300,
    mapFrameId,
    framesSinceTracked: 0,
    cameraCenter: [0, 0, 0],
    deltaTranslation: 0.01,
    deltaRotationDeg: 0.5,
    translationHeld: false,
    translationPredicted: false,
    jumpRejected: false,
    reassociated: 0,
    relink: emptyRelinkStats(),
    reanchor: null,
    mapInlierCount: 100,
    planeInlierCount: 0,
    rejectReason: null,
    mapCandidate: null,
    planeCandidate: null,
    gateMaxTranslation: 0,
    gateMaxRotationDeg: 0,
    observations: 100,
    requiredInliers: 12,
    recoveryMode: "tracking",
    sourceDeltaTranslation: 0,
    sourceDeltaRotationDeg: 0,
    sourceHistory: "M",
    relocalized: false,
    triangulation: { candidates: 0, parallaxRejected: 0, cheiralityRejected: 0, angleRejected: 0, errorRejected: 0, depthRejected: 0, added: 0 },
    reprojection: emptyReprojectionBreakdown(),
    source: "map",
  };
}

const planeNormal = [0, -Math.SQRT1_2, -Math.SQRT1_2];
const planeCenter = [0, 0.85, 0.85];
const plane: PlaneOutput = {
  normal: planeNormal,
  d: -(planeNormal[1] * planeCenter[1] + planeNormal[2] * planeCenter[2]),
  center: planeCenter,
  inlierCount: 200, rmsResidual: 0.001, areaEstimate: 1, horizontalness: 1, horizontal: true, confidence: 1, stableFrames: 10, found: true, usedGravity: true,
};

describe("world-tracking diagnostics (指示書 §8)", () => {
  it("worldFrameDelta: same object → 0, created / dropped → NaN, a known correction → its translation (scaled) and rotation", () => {
    const anchor = new WorldAnchor({ assumedPlaneDistanceMeters: 0.5 });
    expect(worldFrameDelta(null, null)).toEqual({ positionDelta: 0, rotationDeltaDeg: 0 });
    anchor.create(plane, mapPose());
    const w0 = anchor.frame!;
    expect(worldFrameDelta(w0, w0)).toEqual({ positionDelta: 0, rotationDeltaDeg: 0 });
    expect(Number.isNaN(worldFrameDelta(null, w0).positionDelta)).toBe(true);
    // Pure translation of the map by (0.02, 0, 0) u → origin moves 0.02 u × scale meters, no rotation.
    anchor.applyMapCorrection({ rotation: new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]), translation: new Float64Array([0.02, 0, 0]) });
    const w1 = anchor.frame!;
    const d1 = worldFrameDelta(w0, w1);
    expect(d1.positionDelta).toBeCloseTo(0.02 * w0.scale, 12);
    expect(d1.rotationDeltaDeg).toBeCloseTo(0, 9);
    // Pure rotation of 1.5° about the origin of the map: the axes rotate by 1.5°.
    anchor.applyMapCorrection({ rotation: rotationAxisAngle([0, 1, 0], (1.5 * Math.PI) / 180), translation: new Float64Array(3) });
    const d2 = worldFrameDelta(w1, anchor.frame!);
    expect(d2.rotationDeltaDeg).toBeCloseTo(1.5, 9);
  });

  it("planeNormalDeltaDeg: 0 without two normals, the angle between them otherwise", () => {
    expect(planeNormalDeltaDeg(null, planeNormal)).toBe(0);
    expect(planeNormalDeltaDeg(planeNormal, planeNormal)).toBeCloseTo(0, 9);
    const tilted = [Math.sin((3 * Math.PI) / 180), -Math.SQRT1_2 * Math.cos((3 * Math.PI) / 180), -Math.SQRT1_2 * Math.cos((3 * Math.PI) / 180)];
    expect(planeNormalDeltaDeg(planeNormal, tilted)).toBeCloseTo(3, 6);
  });

  it("§9: change detection is by value — a replaced frame object with identical values is not a change, a changed value always is", () => {
    const anchor = new WorldAnchor({ assumedPlaneDistanceMeters: 0.5 });
    anchor.create(plane, mapPose());
    const w0 = anchor.frame!;
    const base = { frameId: 3, quality: emptyQuality(), mapPose: mapPose(), plane, previousPlaneNormal: planeNormal, planeRecovery: null, bundleAdjustment: emptyBundleAdjustment() };
    // Identity correction: the session replaces the object, the values are the same.
    anchor.applyMapCorrection({ rotation: new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]), translation: new Float64Array(3) });
    const w1 = anchor.frame!;
    expect(w1).not.toBe(w0);
    expect(worldFrameValuesDiffer(w0, w1)).toBe(false);
    const same = buildWorldDiagnostics({ ...base, previousWorld: w0, world: w1, changeReason: "ba_correction" });
    expect(same.worldTransformChanged).toBe(false);
    expect(same.worldFrameReplaced).toBe(true);
    expect(same.worldTransformChangeReason).toBe("none");
    expect(same.worldChangeReasonWithoutChange).toBe(true);
    expect(same.anchorPositionDelta).toBe(0);
    // A real move is a change whatever the object identity; without a reason it is the unexplained case.
    anchor.applyMapCorrection({ rotation: new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]), translation: new Float64Array([1e-6, 0, 0]) });
    const w2 = anchor.frame!;
    expect(worldFrameValuesDiffer(w1, w2)).toBe(true);
    const moved = buildWorldDiagnostics({ ...base, previousWorld: w1, world: w2, changeReason: "ba_correction" });
    expect(moved.worldTransformChanged).toBe(true);
    expect(moved.worldTransformChangeReason).toBe("ba_correction");
    expect(moved.worldChangeReasonWithoutChange).toBe(false);
    const unexplained = buildWorldDiagnostics({ ...base, previousWorld: w1, world: w2, changeReason: "none" });
    expect(unexplained.worldTransformChanged).toBe(true);
    expect(unexplained.worldTransformChangeReason).toBe("none");
    // A scale change alone is a change too; rotation noise below 1e-12 is not.
    expect(worldFrameValuesDiffer(w2, { ...w2, scale: w2.scale * (1 + 1e-9) })).toBe(true);
    const jitter = { ...w2, rotation: Float64Array.from(w2.rotation) };
    jitter.rotation[0] += 1e-15;
    expect(worldFrameValuesDiffer(w2, jitter)).toBe(false);
    // Null ↔ frame is created / dropped.
    expect(worldFrameValuesDiffer(null, w2)).toBe(true);
    expect(worldFrameValuesDiffer(w2, null)).toBe(true);
    expect(worldFrameValuesDiffer(null, null)).toBe(false);
  });

  it("buildWorldDiagnostics: changed only when the frame values differ, with the session's reason; a reason without a change is dropped", () => {
    const anchor = new WorldAnchor({ assumedPlaneDistanceMeters: 0.5 });
    anchor.create(plane, mapPose());
    const w = anchor.frame!;
    const base = {
      frameId: 7,
      quality: { ...emptyQuality(), trackingConfidence: 0.9 },
      mapPose: mapPose(),
      plane,
      previousPlaneNormal: planeNormal,
      planeRecovery: null,
      bundleAdjustment: emptyBundleAdjustment(),
    };
    const same = buildWorldDiagnostics({ ...base, previousWorld: w, world: w, changeReason: "ba_correction" });
    expect(same.worldTransformChanged).toBe(false);
    expect(same.worldTransformChangeReason).toBe("none");
    expect(same.anchorPositionDelta).toBe(0);
    expect(same.poseSource).toBe("map");
    expect(same.poseAccepted).toBe(true);
    expect(same.poseRejectReason).toBeNull();
    expect(same.cameraTranslationDelta).toBeCloseTo(0.01 * w.scale, 12);
    expect(same.planeAccepted).toBe(true);
    expect(same.baExecuted).toBe(false);

    const created = buildWorldDiagnostics({ ...base, previousWorld: null, world: w, changeReason: "created" });
    expect(created.worldTransformChanged).toBe(true);
    expect(created.worldTransformChangeReason).toBe("created");
    expect(created.anchorPositionDelta).toBe(0); // no distance to a frame that did not exist

    anchor.applyMapCorrection({ rotation: new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]), translation: new Float64Array([0, 0.01, 0]) });
    const ba = { ...emptyBundleAdjustment(), runs: 1, ranThisFrame: true, errorBeforePx: 1.2, errorAfterPx: 0.8, maxKeyframeShift: 0.02, maxKeyframeRotationDeg: 0.3 };
    const moved = buildWorldDiagnostics({ ...base, bundleAdjustment: ba, previousWorld: w, world: anchor.frame, changeReason: "ba_correction" });
    expect(moved.worldTransformChanged).toBe(true);
    expect(moved.worldTransformChangeReason).toBe("ba_correction");
    expect(moved.anchorPositionDelta).toBeCloseTo(0.01 * w.scale, 12);
    expect(moved.baExecuted && moved.baAccepted).toBe(true);
    expect(moved.baErrorBefore).toBe(1.2);
    expect(moved.baMaxPoseShift).toBe(0.02);

    // A held pose (no fresh PnP) is not "accepted"; the chosen candidate's rejection is the reason.
    const held = { ...mapPose(), framesSinceTracked: 2, source: "propagated" as const, mapCandidate: { inlierCount: 30, reprojectionErrorPx: 0.5, deltaTranslation: 0.3, deltaRotationDeg: 1, trusted: false, reject: { code: "translation_jump" as const, reason: "map translation jump", delta: 0.3, limit: 0.1 } } };
    const lost = buildWorldDiagnostics({ ...base, mapPose: held, previousWorld: anchor.frame, world: anchor.frame, changeReason: "none" });
    expect(lost.poseAccepted).toBe(false);
    expect(lost.poseRejectReason).toBe("translation_jump");
    expect(lost.poseSource).toBe("propagated");

    // Dropped with the map.
    const dropped = buildWorldDiagnostics({ ...base, mapPose: null, previousWorld: anchor.frame, world: null, changeReason: "map_reset" });
    expect(dropped.worldTransformChanged).toBe(true);
    expect(dropped.worldTransformChangeReason).toBe("map_reset");
    expect(dropped.poseSource).toBe("none");
  });
});
