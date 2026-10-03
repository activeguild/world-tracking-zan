import type { RansacConfig } from "../ar/ARConfig";
import {
  estimateHomography,
  homographyTransferErrorSq,
  isDegenerateQuad,
} from "../math/Homography";
import type { Mat3 } from "../math/Matrix";

/**
 * Homography + RANSAC outlier rejection on frame-to-frame correspondences
 * (spec §14, §15).
 *
 * Between consecutive frames (≈33 ms apart) the parallax of a hand-held
 * camera is tiny, so a single homography explains essentially all correct
 * tracks even for non-planar scenes; anything it does not explain is a
 * tracking failure (drift, occlusion, repeated texture) and is dropped.
 */
export interface RansacResult {
  /** Best homography (prev → cur), or null when RANSAC could not run. */
  homography: Mat3 | null;
  /** 1 = inlier, 0 = outlier, per correspondence. */
  inliers: Uint8Array;
  inlierCount: number;
  /** Mean reprojection error of inliers in pixels. */
  meanError: number;
  iterations: number;
}

export type Rng = () => number;

/** Deterministic PRNG (mulberry32) for reproducible tests. */
export function createRng(seed: number): Rng {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function ransacHomography(
  x1: Float32Array | Float64Array,
  y1: Float32Array | Float64Array,
  x2: Float32Array | Float64Array,
  y2: Float32Array | Float64Array,
  n: number,
  config: RansacConfig,
  rng: Rng = Math.random,
): RansacResult {
  const inliers = new Uint8Array(n);
  if (n < Math.max(4, config.minCorrespondences)) {
    // Not enough data to reject anything: keep everything, report no model.
    inliers.fill(1, 0, n);
    return { homography: null, inliers, inlierCount: n, meanError: 0, iterations: 0 };
  }

  const threshSq = config.inlierThreshold * config.inlierThreshold;
  const sample = new Int32Array(4);
  const bestInliers = new Uint8Array(n);
  let bestCount = 0;
  let bestErr = Number.POSITIVE_INFINITY;
  let bestH: Mat3 | null = null;
  let maxIter = config.maxIterations;
  let iter = 0;

  for (; iter < maxIter; iter++) {
    if (!drawSample(sample, n, rng)) break;
    if (isDegenerateQuad(x1, y1, sample) || isDegenerateQuad(x2, y2, sample)) continue;
    const h = estimateHomography(x1, y1, x2, y2, sample, 4);
    if (!h) continue;

    let count = 0;
    let errSum = 0;
    for (let i = 0; i < n; i++) {
      const e = homographyTransferErrorSq(h, x1[i], y1[i], x2[i], y2[i]);
      if (e < threshSq) {
        inliers[i] = 1;
        count++;
        errSum += e;
      } else {
        inliers[i] = 0;
      }
    }
    if (count > bestCount || (count === bestCount && errSum < bestErr)) {
      bestCount = count;
      bestErr = errSum;
      bestH = h;
      bestInliers.set(inliers);
      // Adaptive termination (Hartley & Zisserman 4.18).
      const w = count / n;
      const pNoOutlier = Math.pow(w, 4);
      if (pNoOutlier >= 1 - 1e-9) {
        maxIter = iter + 1;
      } else if (pNoOutlier > 0) {
        const needed = Math.ceil(Math.log(1 - config.confidence) / Math.log(1 - pNoOutlier));
        maxIter = Math.min(config.maxIterations, Math.max(iter + 1, needed));
      }
    }
  }

  if (!bestH || bestCount < 4) {
    inliers.fill(0, 0, n);
    return { homography: null, inliers, inlierCount: 0, meanError: 0, iterations: iter };
  }

  // Refit on all inliers, then re-classify.
  const idx = new Int32Array(bestCount);
  let k = 0;
  for (let i = 0; i < n; i++) if (bestInliers[i]) idx[k++] = i;
  const refined = estimateHomography(x1, y1, x2, y2, idx, bestCount) ?? bestH;

  let count = 0;
  let errSum = 0;
  for (let i = 0; i < n; i++) {
    const e = homographyTransferErrorSq(refined, x1[i], y1[i], x2[i], y2[i]);
    if (e < threshSq) {
      inliers[i] = 1;
      count++;
      errSum += Math.sqrt(e);
    } else {
      inliers[i] = 0;
    }
  }
  // If the refit somehow lost inliers (rare numerical case), keep the sample model.
  if (count < bestCount) {
    inliers.set(bestInliers);
    count = bestCount;
    errSum = 0;
    for (let i = 0; i < n; i++) {
      if (inliers[i]) errSum += Math.sqrt(homographyTransferErrorSq(bestH, x1[i], y1[i], x2[i], y2[i]));
    }
    return { homography: bestH, inliers, inlierCount: count, meanError: errSum / count, iterations: iter };
  }
  return {
    homography: refined,
    inliers,
    inlierCount: count,
    meanError: count > 0 ? errSum / count : 0,
    iterations: iter,
  };
}

function drawSample(out: Int32Array, n: number, rng: Rng): boolean {
  if (n < 4) return false;
  for (let i = 0; i < 4; i++) {
    let v: number;
    let tries = 0;
    do {
      v = Math.min(n - 1, (rng() * n) | 0);
      tries++;
    } while (contains(out, i, v) && tries < 32);
    if (tries >= 32) return false;
    out[i] = v;
  }
  return true;
}

function contains(arr: Int32Array, len: number, v: number): boolean {
  for (let i = 0; i < len; i++) if (arr[i] === v) return true;
  return false;
}
