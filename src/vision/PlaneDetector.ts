import type { PlaneConfig } from "../ar/ARConfig";
import { symmetricEigen } from "../math/Decomposition";
import { horizontalness as planeHorizontalness, ransacPlane, type PlaneModel } from "../math/Plane";
import { angleBetween } from "../math/Pose";
import type { Rng } from "./OutlierRejection";
import type { PlaneOutput } from "./types";

/**
 * Plane detection on the landmark map (spec §21–§24, §51).
 *
 *   landmarks → RANSAC plane → PCA refit → horizontality (gravity or
 *   fallback up) → temporal stability → confidence → PLANE_FOUND
 *
 * One active plane at a time (spec §51). A plane is reported as `found`
 * only after `stableFramesRequired` consecutive consistent detections, and
 * stays found while it keeps being re-detected (with a short grace period).
 */
export interface PlaneCandidate extends PlaneOutput {
  /** Landmark ids that are inliers of the plane. */
  inlierIds: number[];
}

export class PlaneDetector {
  private previous: PlaneCandidate | null = null;
  private stableFrames = 0;
  private found = false;
  private missedFrames = 0;

  constructor(
    private readonly config: PlaneConfig,
    private readonly rng: Rng = Math.random,
  ) {}

  reset(): void {
    this.previous = null;
    this.stableFrames = 0;
    this.found = false;
    this.missedFrames = 0;
  }

  get isFound(): boolean {
    return this.found;
  }

  get current(): PlaneCandidate | null {
    return this.previous;
  }

  /**
   * @param points landmark positions (x,y,z interleaved) in the map frame
   * @param ids    landmark ids, parallel to `points`
   * @param up     up direction in the map frame (from gravity), or null
   */
  update(points: Float64Array, ids: number[], n: number, up: Float64Array | null): PlaneCandidate | null {
    const cfg = this.config;
    if (n < cfg.minInliers) {
      return this.miss();
    }

    // Scale reference: median distance of landmarks from the map origin.
    const dists = new Float64Array(n);
    for (let i = 0; i < n; i++) dists[i] = Math.hypot(points[i * 3], points[i * 3 + 1], points[i * 3 + 2]);
    dists.sort();
    const scale = Math.max(1e-9, dists[n >> 1]);
    const threshold = cfg.inlierThresholdRatio * scale;

    const res = ransacPlane(points, n, {
      threshold,
      confidence: cfg.confidence,
      maxIterations: cfg.maxIterations,
      minInliers: cfg.minInliers,
    }, this.rng);
    if (!res.plane || res.inlierCount < cfg.minInliers) {
      return this.miss();
    }

    const plane = orientTowardCamera(res.plane);
    const inlierIds: number[] = [];
    for (let i = 0; i < n; i++) if (res.inliers[i]) inlierIds.push(ids[i]);
    const area = estimateArea(points, res.inliers, n, plane);

    const upVec = up ?? new Float64Array([0, -1, 0]);
    const hz = planeHorizontalness(plane.normal, upVec);
    const horizontal = hz >= (up ? cfg.horizontalThreshold : cfg.fallbackHorizontalThreshold);

    // Temporal stability against the previous candidate.
    let stable = false;
    if (this.previous) {
      const angle = (angleBetween(plane.normal, this.previous.normal, true) * 180) / Math.PI;
      const shift = Math.hypot(
        plane.center[0] - this.previous.center[0],
        plane.center[1] - this.previous.center[1],
        plane.center[2] - this.previous.center[2],
      );
      // Center shift is measured along the normal (in-plane drift of the
      // centroid is expected as landmarks come and go).
      const nShift = Math.abs(
        plane.normal[0] * (plane.center[0] - this.previous.center[0]) +
          plane.normal[1] * (plane.center[1] - this.previous.center[1]) +
          plane.normal[2] * (plane.center[2] - this.previous.center[2]),
      );
      stable = angle < cfg.stableAngleDeg && nShift < cfg.stableCenterRatio * scale && shift < scale;
    }
    this.stableFrames = stable ? this.stableFrames + 1 : 0;
    this.missedFrames = 0;

    if (horizontal && this.stableFrames >= cfg.stableFramesRequired && res.inlierCount >= cfg.minInliers) {
      this.found = true;
    } else if (!horizontal || !stable) {
      // A clearly different or non-horizontal plane breaks the found state.
      if (!stable && this.previous && this.found) {
        // keep `found` through a single inconsistent frame (grace), drop after
        this.missedFrames = 1;
      } else if (!horizontal) {
        this.found = false;
      }
    }

    const inlierScore = Math.min(1, res.inlierCount / cfg.goodInlierCount);
    const residualScore = Math.max(0, 1 - res.rmsResidual / threshold);
    const stabilityScore = Math.min(1, this.stableFrames / cfg.stableFramesRequired);
    const confidence = inlierScore * residualScore * hz * (0.5 + 0.5 * stabilityScore);

    const candidate: PlaneCandidate = {
      normal: Array.from(plane.normal),
      d: plane.d,
      center: Array.from(plane.center),
      inlierCount: res.inlierCount,
      rmsResidual: res.rmsResidual,
      areaEstimate: area,
      horizontalness: hz,
      horizontal,
      confidence,
      stableFrames: this.stableFrames,
      found: this.found,
      usedGravity: up !== null,
      inlierIds,
    };
    this.previous = candidate;
    return candidate;
  }

  private miss(): PlaneCandidate | null {
    this.missedFrames++;
    this.stableFrames = 0;
    if (this.missedFrames > this.config.lostFrames) {
      this.found = false;
      this.previous = null;
      return null;
    }
    if (this.previous) {
      this.previous = { ...this.previous, stableFrames: 0, found: this.found };
    }
    return this.previous;
  }
}

/** Flip the normal so that it points toward the map-frame camera (the origin). */
function orientTowardCamera(p: PlaneModel): PlaneModel {
  // Camera at origin: signed distance of the origin is d. We want the normal
  // to point from the plane toward the camera, i.e. n·(0 − c) > 0 ⇔ d > 0.
  if (p.d < 0) {
    return { normal: new Float64Array([-p.normal[0], -p.normal[1], -p.normal[2]]), d: -p.d, center: p.center };
  }
  return p;
}

/** Bounding extent of the inliers in the plane's own 2D basis (PCA). */
function estimateArea(points: Float64Array, inliers: Uint8Array, n: number, plane: PlaneModel): number {
  const cov = new Float64Array(9);
  let count = 0;
  const c = plane.center;
  for (let i = 0; i < n; i++) {
    if (!inliers[i]) continue;
    const x = points[i * 3] - c[0];
    const y = points[i * 3 + 1] - c[1];
    const z = points[i * 3 + 2] - c[2];
    cov[0] += x * x; cov[1] += x * y; cov[2] += x * z;
    cov[4] += y * y; cov[5] += y * z; cov[8] += z * z;
    count++;
  }
  if (count < 3) return 0;
  cov[3] = cov[1]; cov[6] = cov[2]; cov[7] = cov[5];
  const eig = symmetricEigen(cov, 3);
  // Two largest eigenvalues span the plane; 2σ extents on each axis.
  const s1 = Math.sqrt(Math.max(0, eig.values[2] / count));
  const s2 = Math.sqrt(Math.max(0, eig.values[1] / count));
  return 4 * s1 * 4 * s2;
}
