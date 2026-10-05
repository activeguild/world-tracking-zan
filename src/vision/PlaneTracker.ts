import type { PlaneTrackingConfig } from "../ar/ARConfig";
import type { CameraIntrinsics } from "../camera/CameraIntrinsics";
import { refinePosePnP } from "../math/PnP";
import type { RigidTransform } from "../math/Pose";
import { intersectRayPlane, pixelRay, transformRay } from "../math/Ray";
import { transpose3 } from "../math/Decomposition";
import type { Track } from "./types";

/**
 * Plane-anchored camera tracking (修正指示書 §5–§9, §20–§22).
 *
 * Once a horizontal plane has been found it becomes the fixed reference of
 * the AR world: the plane equation n·X + d = 0 (map frame) is frozen here and
 * never re-estimated. From then on the camera pose is computed *relative to
 * that plane* and no longer depends on triangulated landmark depths:
 *
 *   1. every tracked feature assumed to lie on the plane is "lifted": its
 *      pixel ray (from a frame whose pose is known) is intersected with the
 *      plane, giving a 3D point whose depth comes from the plane, not from
 *      triangulation
 *   2. per frame, the pose is the PnP solution (LM + Huber) of those plane
 *      points against the current pixel positions. For points on a plane
 *      this is exactly the pose encoded by the plane-induced homography
 *      H = K (R + t nᵀ/d) K⁻¹ between the lifting view and the current view,
 *      solved directly for (R, t) with n, d known — no decomposition
 *      ambiguity, and off-plane features are rejected by the robust loss
 *   3. features that keep violating the plane (objects standing on the desk,
 *      the background) are marked off-plane after a few frames and ignored
 *      by this estimator (the landmark map still uses them)
 *
 * Quality (§9): inlier count, inlier ratio, mean reprojection error,
 * confidence. Failure: fewer than `minInliers` plane inliers → the caller
 * falls back to the landmark-map pose (plane out of view).
 */
export interface PlaneAnchor {
  /** Unit normal, oriented toward the camera that found the plane. */
  normal: Float64Array;
  /** n·X + d = 0 */
  d: number;
  center: Float64Array;
  frameId: number;
  /** Distance of the anchoring camera to the plane (map units). */
  cameraDistance: number;
}

export interface PlaneTrackingResult {
  /** The plane PnP produced a pose candidate. */
  tracked: boolean;
  /**
   * The candidate was validated by the caller and the plane bookkeeping was
   * updated with the canonical pose (`commit`). A rejected candidate leaves
   * every streak / off-plane flag untouched (v4 §12–§13).
   */
  accepted: boolean;
  pose: RigidTransform | null;
  /** Plane points whose reprojection passed the gate. */
  inlierCount: number;
  /** Lifted, not-yet-rejected plane points offered to the estimator. */
  candidateCount: number;
  /** Candidates that have passed probation. */
  confirmedCount: number;
  inlierRatio: number;
  meanErrorPx: number;
  confidence: number;
}

const NOT_TRACKED: PlaneTrackingResult = {
  tracked: false,
  accepted: false,
  pose: null,
  inlierCount: 0,
  candidateCount: 0,
  confirmedCount: 0,
  inlierRatio: 0,
  meanErrorPx: 0,
  confidence: 0,
};

export class PlaneTracker {
  private _anchor: PlaneAnchor | null = null;
  private pts3 = new Float64Array(0);
  private obsX = new Float64Array(0);
  private obsY = new Float64Array(0);
  private lastResult: PlaneTrackingResult = NOT_TRACKED;

  constructor(private readonly config: PlaneTrackingConfig) {}

  get anchored(): boolean {
    return this._anchor !== null;
  }

  get anchor(): PlaneAnchor | null {
    return this._anchor;
  }

  get result(): PlaneTrackingResult {
    return this.lastResult;
  }

  /** Forget the anchor and every lifted point. */
  reset(tracks: readonly Track[]): void {
    this._anchor = null;
    this.lastResult = NOT_TRACKED;
    for (const t of tracks) clearPlaneState(t);
  }

  /**
   * Fix the plane. `cameraPose` is the map-frame pose (X_cam = R X_map + t)
   * of the frame in which the plane was found.
   */
  setAnchor(plane: { normal: ArrayLike<number>; d: number; center: ArrayLike<number> }, cameraPose: RigidTransform, frameId: number): void {
    const n = new Float64Array([plane.normal[0], plane.normal[1], plane.normal[2]]);
    const len = Math.hypot(n[0], n[1], n[2]) || 1;
    n[0] /= len;
    n[1] /= len;
    n[2] /= len;
    const c = cameraCenter(cameraPose);
    const dist = Math.abs(n[0] * c[0] + n[1] * c[1] + n[2] * c[2] + plane.d / len);
    this._anchor = {
      normal: n,
      d: plane.d / len,
      center: new Float64Array([plane.center[0], plane.center[1], plane.center[2]]),
      frameId,
      cameraDistance: dist,
    };
  }

  /**
   * Lift tracks that have no plane point yet: intersect their pixel ray
   * (through `pose`, assumed accurate for this frame) with the plane.
   * `confirmed(track)` says whether the track is already known to lie on the
   * plane (e.g. it observes a plane-inlier landmark); such tracks skip
   * probation. Returns the number of tracks lifted.
   */
  lift(tracks: readonly Track[], pose: RigidTransform, k: CameraIntrinsics, confirmed: (t: Track) => boolean): number {
    const a = this._anchor;
    if (!a) return 0;
    const cfg = this.config;
    const rt = transpose3(pose.rotation);
    const c = cameraCenter(pose);
    const minSin = Math.sin((cfg.minRayAngleDeg * Math.PI) / 180);
    const maxT = cfg.maxLiftDistanceRatio * a.cameraDistance;
    let lifted = 0;
    for (const t of tracks) {
      if (t.planePoint || t.offPlane) continue;
      const rayCam = pixelRay(t.x, t.y, k.fx, k.fy, k.cx, k.cy);
      const ray = transformRay(rayCam, rt, c);
      const dir = ray.direction;
      // Grazing rays give unstable intersections.
      if (Math.abs(a.normal[0] * dir[0] + a.normal[1] * dir[1] + a.normal[2] * dir[2]) < minSin) continue;
      const hit = intersectRayPlane(ray, a.normal, a.d);
      if (!hit || hit.t > maxT) continue;
      t.planePoint = hit.point;
      t.planeOutliers = 0;
      t.planeStreak = confirmed(t) ? cfg.probationFrames : 0;
      lifted++;
    }
    return lifted;
  }

  /** Lifted, not-yet-rejected plane points. */
  private candidatesOf(tracks: readonly Track[]): { candidates: Track[]; confirmedCount: number } {
    const candidates: Track[] = [];
    let confirmedCount = 0;
    for (const t of tracks) {
      if (!t.planePoint || t.offPlane) continue;
      candidates.push(t);
      if (t.planeStreak >= this.config.probationFrames) confirmedCount++;
    }
    return { candidates, confirmedCount };
  }

  /**
   * Plane-relative pose *candidate* for the current frame from the lifted
   * tracks. `prior` is the pose prediction (previous pose, optionally
   * rotated by the frame-to-frame rotation).
   *
   * This only solves; it mutates no track. The caller validates the
   * candidate (temporal gate, agreement with the map) and, when it was
   * accepted, calls `commit()` with the canonical pose. A candidate that
   * was rejected must not strengthen any plane point's standing (v4 §12–§13).
   */
  update(tracks: readonly Track[], prior: RigidTransform, k: CameraIntrinsics): PlaneTrackingResult {
    const a = this._anchor;
    if (!a) return (this.lastResult = NOT_TRACKED);
    const cfg = this.config;
    const f = (k.fx + k.fy) / 2;

    const { candidates, confirmedCount } = this.candidatesOf(tracks);
    const useConfirmedOnly = confirmedCount >= cfg.minInliers;
    const solveSet = useConfirmedOnly ? candidates.filter((t) => t.planeStreak >= cfg.probationFrames) : candidates;
    if (solveSet.length < 6) {
      return (this.lastResult = { ...NOT_TRACKED, candidateCount: candidates.length, confirmedCount });
    }

    this.ensure(solveSet.length);
    for (let i = 0; i < solveSet.length; i++) {
      const t = solveSet[i];
      const p = t.planePoint!;
      this.pts3[i * 3] = p[0];
      this.pts3[i * 3 + 1] = p[1];
      this.pts3[i * 3 + 2] = p[2];
      this.obsX[i] = (t.x - k.cx) / k.fx;
      this.obsY[i] = (t.y - k.cy) / k.fy;
    }
    const res = refinePosePnP(prior, this.pts3, this.obsX, this.obsY, solveSet.length, {
      huber: cfg.pnpHuberPx / f,
      inlierThreshold: cfg.pnpInlierPx / f,
      maxIterations: cfg.pnpMaxIterations,
      epsilon: 1e-6,
    });
    if (res.inlierCount < cfg.minInliers) {
      return (this.lastResult = { ...NOT_TRACKED, candidateCount: candidates.length, confirmedCount });
    }
    const meanErrorPx = res.meanError * f;
    this.lastResult = {
      tracked: true,
      accepted: false,
      pose: res.pose,
      inlierCount: res.inlierCount,
      candidateCount: candidates.length,
      confirmedCount,
      inlierRatio: candidates.length ? res.inlierCount / candidates.length : 0,
      meanErrorPx,
      confidence: Math.min(1, res.inlierCount / cfg.goodInliers) * Math.max(0, 1 - meanErrorPx / cfg.pnpInlierPx),
    };
    return this.lastResult;
  }

  /**
   * The candidate of this frame was accepted: gate every plane point with
   * the *canonical* pose (the plane pose itself, or the map pose it agreed
   * with) and update probation / off-plane bookkeeping. Only this method
   * changes the plane state, so a rejected candidate never does.
   */
  commit(tracks: readonly Track[], pose: RigidTransform, k: CameraIntrinsics): PlaneTrackingResult {
    const a = this._anchor;
    if (!a) return (this.lastResult = NOT_TRACKED);
    const cfg = this.config;
    const f = (k.fx + k.fy) / 2;
    const { candidates, confirmedCount } = this.candidatesOf(tracks);
    const gateSq = (cfg.pnpInlierPx / f) ** 2;
    const r = pose.rotation;
    const tt = pose.translation;
    let inliers = 0;
    let errSum = 0;
    for (const t of candidates) {
      const p = t.planePoint!;
      const z = r[6] * p[0] + r[7] * p[1] + r[8] * p[2] + tt[2];
      let ok = false;
      if (z > 1e-6) {
        const u = (r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + tt[0]) / z;
        const v = (r[3] * p[0] + r[4] * p[1] + r[5] * p[2] + tt[1]) / z;
        const e2 = (u - (t.x - k.cx) / k.fx) ** 2 + (v - (t.y - k.cy) / k.fy) ** 2;
        if (e2 <= gateSq) {
          ok = true;
          inliers++;
          errSum += Math.sqrt(e2);
        }
      }
      if (ok) {
        t.planeStreak++;
        t.planeOutliers = 0;
      } else {
        t.planeStreak = 0;
        t.planeOutliers++;
        if (t.planeOutliers > cfg.maxOutlierStreak) {
          // Not on the plane (or lifted from a bad pose): stop using it here.
          t.offPlane = true;
          t.planePoint = null;
        }
      }
    }
    const meanErrorPx = inliers ? (errSum / inliers) * f : 0;
    const inlierRatio = candidates.length ? inliers / candidates.length : 0;
    const confidence = Math.min(1, inliers / cfg.goodInliers) * Math.max(0, 1 - meanErrorPx / cfg.pnpInlierPx);
    this.lastResult = {
      tracked: this.lastResult.tracked,
      accepted: true,
      pose: this.lastResult.pose,
      inlierCount: inliers,
      candidateCount: candidates.length,
      confirmedCount,
      inlierRatio,
      meanErrorPx,
      confidence,
    };
    return this.lastResult;
  }

  private ensure(n: number): void {
    if (this.pts3.length < n * 3) {
      this.pts3 = new Float64Array(n * 3);
      this.obsX = new Float64Array(n);
      this.obsY = new Float64Array(n);
    }
  }
}

export function clearPlaneState(t: Track): void {
  t.planePoint = null;
  t.planeStreak = 0;
  t.planeOutliers = 0;
  t.offPlane = false;
}

/** Camera center in the map frame: C = −Rᵀ t. */
function cameraCenter(pose: RigidTransform): Float64Array {
  const r = pose.rotation;
  const t = pose.translation;
  return new Float64Array([
    -(r[0] * t[0] + r[3] * t[1] + r[6] * t[2]),
    -(r[1] * t[0] + r[4] * t[1] + r[7] * t[2]),
    -(r[2] * t[0] + r[5] * t[1] + r[8] * t[2]),
  ]);
}
