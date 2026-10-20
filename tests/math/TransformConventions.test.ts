import { describe, expect, it } from "vitest";
import { WorldAnchor } from "../../src/ar/WorldAnchor";
import { worldFrameValuesDiffer } from "../../src/ar/WorldDiagnostics";
import { cameraPoseToThree, mapToWorld, threePoseToCamera, worldToMap } from "../../src/math/CoordinateSystem";
import { applyTransform, cameraCenterOf, composeTransforms, invertTransform, rotationAxisAngle, rotationDistance, type RigidTransform } from "../../src/math/Pose";
import { translationFit } from "../../src/math/RigidFit";
import { emptyRelinkStats, emptyReprojectionBreakdown, type MapPoseOutput, type PlaneOutput } from "../../src/vision/types";

/**
 * 修正指示書「座標系整合性と空間固定性」§5, §10-B: the transform conventions,
 * proven on known points rather than read off function names.
 *
 *   X_cam = R · X_map + t                     (applyTransform, RigidTransform)
 *   composeTransforms(B, A) = B ∘ A           (apply A first, then B)
 *   invertTransform(T) ∘ T = identity
 *   correction C = T_new⁻¹ ∘ T_old            (old map → new map: the camera point is the same)
 *   rebased pose T_new = T_cur ∘ C⁻¹          (same camera-frame coordinates for the corrected landmark)
 *   WorldAnchor.applyMapCorrection(C)         (world coordinates of a corrected map point are unchanged)
 */
const deg = (d: number) => (d * Math.PI) / 180;
const T = (axis: number[], angleDeg: number, t: number[]): RigidTransform => ({ rotation: rotationAxisAngle(axis, deg(angleDeg)), translation: Float64Array.from(t) });
const near = (a: ArrayLike<number>, b: ArrayLike<number>, digits = 10) => {
  for (let i = 0; i < a.length; i++) expect(a[i]).toBeCloseTo(b[i], digits);
};

describe("rigid transform conventions (§5)", () => {
  it("X_cam = R X + t: a 90° yaw about +Y maps +Z (ahead) to +X, then the translation adds", () => {
    const yaw90 = T([0, 1, 0], 90, [0.5, -0.2, 1]);
    // R_y(90°) · (0, 0, 1) = (sin 90, 0, cos 90) = (1, 0, 0)
    near(applyTransform(yaw90, [0, 0, 1]), [1.5, -0.2, 1], 9);
    // The camera centre C = −Rᵀ t maps to the origin of the camera frame.
    near(applyTransform(yaw90, cameraCenterOf(yaw90)), [0, 0, 0], 9);
  });

  it("composeTransforms(B, A) applies A first, then B — not the other way round", () => {
    const A = T([1, 0, 0], 30, [0.1, 0.2, 0.3]);
    const B = T([0, 0, 1], -50, [-0.4, 0.5, 0.6]);
    const X = [0.3, -0.7, 2.1];
    near(applyTransform(composeTransforms(B, A), X), applyTransform(B, applyTransform(A, X)), 12);
    // The other order gives a different point (the transforms do not commute).
    const other = applyTransform(composeTransforms(A, B), X);
    const expected = applyTransform(B, applyTransform(A, X));
    expect(Math.hypot(other[0] - expected[0], other[1] - expected[1], other[2] - expected[2])).toBeGreaterThan(0.1);
  });

  it("invertTransform round-trips points and composes to the identity", () => {
    const A = T([0.2, 1, 0.3], 37, [1.5, -2, 0.7]);
    const inv = invertTransform(A);
    const X = [0.3, -0.7, 2.1];
    near(applyTransform(inv, applyTransform(A, X)), X, 12);
    near(applyTransform(A, applyTransform(inv, X)), X, 12);
    const id = composeTransforms(inv, A);
    near(id.rotation, [1, 0, 0, 0, 1, 0, 0, 0, 1], 12);
    near(id.translation, [0, 0, 0], 12);
  });

  it("the map correction C = T_new⁻¹ T_old maps old-map coordinates to new-map coordinates of the same camera point", () => {
    const tOld = T([0, 1, 0], 12, [0.2, -0.1, 3]);
    // The solver moved this view's pose a little.
    const tNew = composeTransforms(T([1, 0.2, 0], 0.8, [0.01, -0.02, 0.015]), tOld);
    const C = composeTransforms(invertTransform(tNew), tOld);
    const xOld = [0.4, 0.6, 5.2];
    // Same point in the camera: T_new(C X_old) = T_old X_old.
    near(applyTransform(tNew, applyTransform(C, xOld)), applyTransform(tOld, xOld), 12);
    // Rebased pose T_cur ∘ C⁻¹ sees the corrected landmark where T_cur saw the old one.
    const tCur = T([0.1, 1, 0], 15, [0.25, -0.05, 3.1]);
    const rebased = composeTransforms(tCur, invertTransform(C));
    near(applyTransform(rebased, applyTransform(C, xOld)), applyTransform(tCur, xOld), 12);
    // And the inverse direction (new → old) is C⁻¹, not Cᵀ-with-the-same-t.
    near(applyTransform(invertTransform(C), applyTransform(C, xOld)), xOld, 12);
  });

  it("WorldAnchor.applyMapCorrection(C): a corrected map point keeps its world coordinates and the rebased camera keeps its Three.js pose; an identity C changes no value", () => {
    const normal = [0, -Math.SQRT1_2, -Math.SQRT1_2];
    const center = [0, 0.85, 0.85];
    const plane: PlaneOutput = { normal, d: -(normal[1] * center[1] + normal[2] * center[2]), center, inlierCount: 100, rmsResidual: 0, areaEstimate: 1, horizontalness: 1, horizontal: true, confidence: 1, stableFrames: 5, found: true, usedGravity: true };
    const cam = T([0, 1, 0], 5, [0.1, 0, 0.2]);
    const mp = (p: RigidTransform): MapPoseOutput => ({
      rotation: Array.from(p.rotation), translation: Array.from(p.translation), inlierCount: 50, meanReprojectionErrorPx: 0.3, landmarkCount: 100, mapFrameId: 0, framesSinceTracked: 0,
      cameraCenter: Array.from(cameraCenterOf(p)), deltaTranslation: 0, deltaRotationDeg: 0, translationHeld: false, translationPredicted: false, jumpRejected: false, reassociated: 0,
      relink: emptyRelinkStats(), reanchor: null, mapInlierCount: 50, planeInlierCount: 0, rejectReason: null, mapCandidate: null, planeCandidate: null, gateMaxTranslation: 0, gateMaxRotationDeg: 0,
      observations: 50, requiredInliers: 12, recoveryMode: "tracking", sourceDeltaTranslation: 0, sourceDeltaRotationDeg: 0, sourceHistory: "M", relocalized: false,
      triangulation: { candidates: 0, parallaxRejected: 0, cheiralityRejected: 0, angleRejected: 0, errorRejected: 0, depthRejected: 0, added: 0 }, reprojection: emptyReprojectionBreakdown(), source: "map",
    });
    const anchor = new WorldAnchor({ assumedPlaneDistanceMeters: 0.5 });
    expect(anchor.create(plane, mp(cam))).toBe(true);
    const before = anchor.frame!;
    const xOld = [0.2, 0.6, 1.1];
    const wOld = Array.from(mapToWorld(before, xOld));
    const threeOld = cameraPoseToThree(before, cam);
    const C = T([0.3, 1, 0.1], 1.2, [0.02, -0.01, 0.03]);
    anchor.applyMapCorrection(C);
    const after = anchor.frame!;
    expect(worldFrameValuesDiffer(before, after)).toBe(true);
    // The map moved under the world: the corrected point has the same world coordinates …
    near(mapToWorld(after, applyTransform(C, xOld)), wOld, 10);
    // … the world → map matrix inverts it …
    near(worldToMap(after, wOld), applyTransform(C, xOld), 10);
    // … and the rebased camera has the same Three.js pose (position, and the rotation up to quaternion sign).
    const rebased = composeTransforms(cam, invertTransform(C));
    const threeNew = cameraPoseToThree(after, rebased);
    near(threeNew.position, threeOld.position, 10);
    const q = threeNew.quaternion, q0 = threeOld.quaternion;
    expect(Math.abs(q[0] * q0[0] + q[1] * q0[1] + q[2] * q0[2] + q[3] * q0[3])).toBeCloseTo(1, 10);
    // Three.js → CV round trip through the corrected frame lands on the rebased pose.
    const back = threePoseToCamera(after, threeNew.position, threeNew.quaternion);
    // (rotationDistance goes through acos: ~1e-7 rad resolution near 0; the matrices agree to 1e-9.)
    expect(rotationDistance(back.rotation, rebased.rotation)).toBeLessThan(1e-6);
    near(back.rotation, rebased.rotation, 9);
    near(back.translation, rebased.translation, 9);
    // anchorCorrection "none" / an identity correction: a new frame object with the same values is not a change.
    const snapshot = anchor.frame!;
    anchor.applyMapCorrection({ rotation: new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]), translation: new Float64Array(3) });
    expect(anchor.frame).not.toBe(snapshot);
    expect(worldFrameValuesDiffer(snapshot, anchor.frame)).toBe(false);
  });

  it("translationFit recovers the translation of a rigid move when given the true rotation; with the wrong rotation it absorbs the rotation into a biased translation", () => {
    const n = 12;
    const before = new Float64Array(n * 3);
    for (let i = 0; i < n; i++) {
      before[i * 3] = 2 + 0.5 * Math.cos((i * 2 * Math.PI) / n);
      before[i * 3 + 1] = -1 + 0.5 * Math.sin((i * 2 * Math.PI) / n);
      before[i * 3 + 2] = 6 + 0.1 * (i % 3);
    }
    const move = T([0.2, 1, 0], 2, [0.03, -0.02, 0.05]);
    const after = new Float64Array(n * 3);
    for (let i = 0; i < n; i++) after.set(applyTransform(move, before.subarray(i * 3, i * 3 + 3)), i * 3);
    const fit = translationFit(before, after, n, move.rotation)!;
    near(fit.transform.translation, move.translation, 10);
    near(fit.transform.rotation, move.rotation, 12);
    expect(fit.inliers).toBe(n);
    // Wrong (identity) rotation: the 2° of the cloud at ~6 u from the origin become ~0.2 u of translation.
    const wrong = translationFit(before, after, n, [1, 0, 0, 0, 1, 0, 0, 0, 1])!;
    const err = Math.hypot(wrong.transform.translation[0] - move.translation[0], wrong.transform.translation[1] - move.translation[1], wrong.transform.translation[2] - move.translation[2]);
    expect(err).toBeGreaterThan(0.1);
  });
});
