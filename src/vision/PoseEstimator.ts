import type { PoseConfig, RansacConfig } from "../ar/ARConfig";
import type { CameraIntrinsics } from "../camera/CameraIntrinsics";
import { ransacEssential, recoverPose } from "../math/EssentialMatrix";
import { decomposeHomography, homographySolutionSupport, type HomographySolution } from "../math/HomographyDecomposition";
import { type Mat3, mat3Identity, mat3Invert, mat3Multiply } from "../math/Matrix";
import { angleBetween, normalize3, projectToRotation } from "../math/Pose";
import { ransacHomography, type Rng } from "./OutlierRejection";

/**
 * Two-view relative pose (spec §15–§18, Phase 2).
 *
 * Input: pixel correspondences between a reference frame and the current
 * frame plus intrinsics. Output: rotation + unit translation direction
 * (scale-free), the geometric model that produced it, and confidences.
 *
 * Model selection follows the ORB-SLAM idea: both a homography and an
 * essential matrix are fitted with RANSAC; when the homography explains
 * (nearly) as many correspondences as E, the scene is planar or the motion
 * is a pure rotation and E is degenerate, so the homography is decomposed
 * instead. With too little parallax only the rotation is trusted.
 */
export type PoseModel = "essential" | "homography" | "rotation" | "none";

export interface RelativePose {
  /** X_cur = R · X_ref + t */
  rotation: Mat3;
  /** Unit translation direction in the current camera frame (zero when unknown). */
  translationDirection: Float64Array;
  inlierCount: number;
  model: PoseModel;
  /** Median image-space displacement of the correspondences (pixels). */
  parallaxPx: number;
  /** 0–1: how much the translation direction can be trusted. */
  translationConfidence: number;
  /** 0–1: overall confidence of the estimate. */
  confidence: number;
  /** Plane normal in the reference camera frame, when a homography model was used. */
  planeNormal: Float64Array | null;
  /** Per-correspondence inlier flags of the chosen model (length n). */
  inlierMask: Uint8Array;
  /** Diagnostics */
  homographyInliers: number;
  essentialInliers: number;
}

export class PoseEstimator {
  private readonly homographyRansac: RansacConfig;
  private nx1 = new Float64Array(0);
  private ny1 = new Float64Array(0);
  private nx2 = new Float64Array(0);
  private ny2 = new Float64Array(0);

  constructor(
    private readonly config: PoseConfig,
    ransac: RansacConfig,
    private readonly rng: Rng = Math.random,
  ) {
    this.homographyRansac = ransac;
  }

  /**
   * @param x1,y1 reference-frame pixel positions
   * @param x2,y2 current-frame pixel positions
   * @param previousNormal plane normal of the previous estimate (reference frame),
   *        used to disambiguate the two homography solutions.
   */
  estimate(
    x1: Float32Array | Float64Array,
    y1: Float32Array | Float64Array,
    x2: Float32Array | Float64Array,
    y2: Float32Array | Float64Array,
    n: number,
    k: CameraIntrinsics,
    previousNormal: Float64Array | null = null,
    /**
     * Cap on the RANSAC iterations of both models (v16). The engine lowers
     * it once the world exists: the map PnP is then the canonical pose and
     * the two-view result only a frame-to-frame rotation prior, while a
     * low inlier ratio made the essential RANSAC run to its 300-iteration
     * cap (6–17 ms on Android). Undefined = the configured maxima.
     */
    maxIterations?: number,
  ): RelativePose {
    const cfg = this.config;
    if (n < cfg.minCorrespondences) return noPose(n);
    const hIter = maxIterations === undefined ? this.homographyRansac.maxIterations : Math.min(this.homographyRansac.maxIterations, maxIterations);
    const eIter = maxIterations === undefined ? cfg.maxIterations : Math.min(cfg.maxIterations, maxIterations);

    // Parallax in pixels (median displacement).
    const disp = new Float64Array(n);
    for (let i = 0; i < n; i++) disp[i] = Math.hypot(x2[i] - x1[i], y2[i] - y1[i]);
    const sorted = Float64Array.from(disp).sort();
    const parallaxPx = sorted[n >> 1];

    // Normalized coordinates.
    this.ensure(n);
    const { nx1, ny1, nx2, ny2 } = this;
    for (let i = 0; i < n; i++) {
      nx1[i] = (x1[i] - k.cx) / k.fx;
      ny1[i] = (y1[i] - k.cy) / k.fy;
      nx2[i] = (x2[i] - k.cx) / k.fx;
      ny2[i] = (y2[i] - k.cy) / k.fy;
    }
    const f = (k.fx + k.fy) / 2;

    // Homography (pixel space) — reuse the Phase 1 RANSAC.
    const hRes = ransacHomography(
      x1, y1, x2, y2, n,
      { ...this.homographyRansac, inlierThreshold: cfg.ransacThresholdPx, maxIterations: hIter },
      this.rng,
    );
    // Essential (normalized space).
    const eRes = ransacEssential(
      nx1, ny1, nx2, ny2, n,
      {
        threshold: cfg.ransacThresholdPx / f,
        confidence: this.homographyRansac.confidence,
        maxIterations: eIter,
        minCorrespondences: cfg.minCorrespondences,
      },
      this.rng,
    );

    const hIn = hRes.homography ? hRes.inlierCount : 0;
    const eIn = eRes.essential ? eRes.inlierCount : 0;
    if (hIn === 0 && eIn === 0) return noPose(n);

    // Calibrated homography Hn = K⁻¹ H K.
    let hn: Mat3 | null = null;
    if (hRes.homography) {
      const K = new Float64Array([k.fx, 0, k.cx, 0, k.fy, k.cy, 0, 0, 1]);
      const kInv = mat3Invert(K);
      if (kInv) hn = mat3Multiply(mat3Multiply(kInv, hRes.homography), K);
    }

    // Too little parallax → translation is not observable; rotation from H.
    if (parallaxPx < cfg.minParallaxPx) {
      if (hn) {
        return {
          rotation: projectToRotation(hn),
          translationDirection: new Float64Array(3),
          inlierCount: hIn,
          model: "rotation",
          parallaxPx,
          translationConfidence: 0,
          confidence: Math.min(1, hIn / cfg.goodInlierCount),
          planeNormal: null,
          inlierMask: hRes.inliers,
          homographyInliers: hIn,
          essentialInliers: eIn,
        };
      }
      return noPose(n, parallaxPx);
    }

    const ratio = hIn / Math.max(1, hIn + eIn);
    const preferHomography = hn !== null && ratio > cfg.homographyRatioThreshold;

    let result: RelativePose | null = null;
    if (preferHomography && hn) {
      result = this.fromHomography(hn, hRes.inliers, n, parallaxPx, previousNormal, hIn, eIn);
    }
    if (!result && eRes.essential) {
      result = this.fromEssential(eRes.essential, eRes.inliers, n, f, parallaxPx, hIn, eIn);
    }
    if (!result && hn) {
      result = this.fromHomography(hn, hRes.inliers, n, parallaxPx, previousNormal, hIn, eIn);
    }
    return result ?? noPose(n, parallaxPx);
  }

  private fromEssential(
    e: Mat3,
    inliers: Uint8Array,
    n: number,
    f: number,
    parallaxPx: number,
    hIn: number,
    eIn: number,
  ): RelativePose | null {
    const cfg = this.config;
    const rec = recoverPose(e, this.nx1, this.ny1, this.nx2, this.ny2, inliers, n, cfg.maxTriangulationErrorPx / f);
    if (!rec || rec.good < 8) return null;
    const support = rec.good / Math.max(1, eIn);
    if (support < cfg.minCheiralityRatio) return null;
    // Ambiguity: the runner-up candidate should be clearly worse.
    const ambiguity = rec.secondBest / Math.max(1, rec.good);
    const translationConfidence = clamp01(parallaxPx / cfg.fullConfidenceParallaxPx) * (1 - ambiguity) * support;
    return {
      rotation: rec.pose.rotation,
      translationDirection: normalize3(rec.pose.translation),
      inlierCount: eIn,
      model: "essential",
      parallaxPx,
      translationConfidence,
      confidence: Math.min(1, eIn / cfg.goodInlierCount) * support,
      planeNormal: null,
      inlierMask: inliers,
      homographyInliers: hIn,
      essentialInliers: eIn,
    };
  }

  private fromHomography(
    hn: Mat3,
    inliers: Uint8Array,
    n: number,
    parallaxPx: number,
    previousNormal: Float64Array | null,
    hIn: number,
    eIn: number,
  ): RelativePose | null {
    const cfg = this.config;
    const sols = decomposeHomography(hn);
    if (sols.length === 0) return null;
    const idxArr: number[] = [];
    for (let i = 0; i < n; i++) if (inliers[i]) idxArr.push(i);
    const idx = Int32Array.from(idxArr);

    if (sols[0].pureRotation) {
      return {
        rotation: sols[0].rotation,
        translationDirection: new Float64Array(3),
        inlierCount: hIn,
        model: "rotation",
        parallaxPx,
        translationConfidence: 0,
        confidence: Math.min(1, hIn / cfg.goodInlierCount),
        planeNormal: null,
        inlierMask: inliers,
        homographyInliers: hIn,
        essentialInliers: eIn,
      };
    }

    // Positive-depth support for each of the 4 candidates.
    let best: { sol: HomographySolution; support: number } | null = null;
    let second: { sol: HomographySolution; support: number } | null = null;
    for (const sol of sols) {
      const support = homographySolutionSupport(sol, this.nx1, this.ny1, idx, idx.length);
      const entry = { sol, support };
      if (!best || support > best.support) {
        second = best;
        best = entry;
      } else if (!second || support > second.support) {
        second = entry;
      }
    }
    if (!best || best.support < 8) return null;
    const total = idx.length;
    const bestRatio = best.support / total;
    if (bestRatio < cfg.minCheiralityRatio) return null;

    // Two solutions typically survive the depth test. Prefer the one whose
    // normal is consistent with the previous estimate; otherwise prefer the
    // normal facing the camera (largest n_z) — a desk/floor seen from a phone
    // held above it. The alternative solution swaps the roles of t and n.
    let chosen = best;
    let ambiguous = false;
    if (second && second.support / total >= cfg.minCheiralityRatio && second.support >= best.support * 0.9) {
      ambiguous = true;
      if (previousNormal) {
        const aBest = angleBetween(best.sol.normal, previousNormal);
        const aSecond = angleBetween(second.sol.normal, previousNormal);
        chosen = aSecond < aBest ? second : best;
        ambiguous = Math.abs(aSecond - aBest) < 0.1;
      } else {
        chosen = second.sol.normal[2] > best.sol.normal[2] ? second : best;
      }
    }
    const sol = chosen.sol;
    const translationConfidence =
      clamp01(parallaxPx / cfg.fullConfidenceParallaxPx) * (ambiguous ? 0.5 : 1) * bestRatio;
    return {
      rotation: sol.rotation,
      translationDirection: normalize3(sol.translation),
      inlierCount: hIn,
      model: "homography",
      parallaxPx,
      translationConfidence,
      confidence: Math.min(1, hIn / cfg.goodInlierCount) * bestRatio,
      planeNormal: Float64Array.from(sol.normal),
      inlierMask: inliers,
      homographyInliers: hIn,
      essentialInliers: eIn,
    };
  }

  private ensure(n: number): void {
    if (this.nx1.length < n) {
      this.nx1 = new Float64Array(n);
      this.ny1 = new Float64Array(n);
      this.nx2 = new Float64Array(n);
      this.ny2 = new Float64Array(n);
    }
  }
}

function noPose(n: number, parallaxPx = 0): RelativePose {
  return {
    rotation: mat3Identity(),
    translationDirection: new Float64Array(3),
    inlierCount: 0,
    model: "none",
    parallaxPx,
    translationConfidence: 0,
    confidence: 0,
    planeNormal: null,
    inlierMask: new Uint8Array(Math.max(0, n)),
    homographyInliers: 0,
    essentialInliers: 0,
  };
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}
