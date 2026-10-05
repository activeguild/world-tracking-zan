import type { TrackerConfig } from "../ar/ARConfig";
import type { ImagePyramid, PyramidLevel } from "./ImagePyramid";

/**
 * Pyramidal Lucas-Kanade optical flow with forward-backward verification
 * (spec §12, §13).
 *
 * Algorithm (Bouguet):
 *   for each point, from the coarsest level to level 0:
 *     - sample the template patch T and its gradients at the point
 *     - build the 2×2 structure tensor G and reject if poorly conditioned
 *     - iterate: sample J at the current estimate, compute b = Σ∇T·(T−J),
 *       update d += G⁻¹ b until |Δd| < ε
 *     - propagate d to the next finer level (×2)
 *
 * Forward-backward check: track prev→cur, then cur→prev starting at the found
 * position; the point is accepted only if it returns within
 * `forwardBackwardThreshold` pixels of its origin.
 */

export const enum TrackStatus {
  /** Tracked and verified. */
  OK = 0,
  /** Window left the image. */
  OUT_OF_BOUNDS = 1,
  /** Structure tensor too weak (texture-less patch). */
  LOW_TEXTURE = 2,
  /** Residual after convergence too large. */
  HIGH_RESIDUAL = 3,
  /** Forward-backward error too large. */
  FB_ERROR = 4,
  /** Moved further than `maxDisplacement`. */
  TOO_FAR = 5,
}

export interface TrackResult {
  /** Output positions (x,y interleaved), valid where status === OK. */
  positions: Float32Array;
  status: Uint8Array;
  /** Mean absolute residual at level 0 per point. */
  residual: Float32Array;
  /** Forward-backward error per point (pixels). */
  fbError: Float32Array;
  okCount: number;
}

export class FeatureTracker {
  private readonly half: number;
  private readonly winArea: number;
  // Template buffers (reused)
  private readonly tI: Float32Array;
  private readonly tX: Float32Array;
  private readonly tY: Float32Array;

  constructor(private readonly config: TrackerConfig) {
    const win = config.windowSize | 1; // force odd
    this.half = (win - 1) / 2;
    this.winArea = win * win;
    this.tI = new Float32Array(this.winArea);
    this.tX = new Float32Array(this.winArea);
    this.tY = new Float32Array(this.winArea);
  }

  /**
   * Track `count` points (x,y interleaved in `points`, level-0 pixels of
   * `prev`) into `cur`. Both pyramids must have identical dimensions.
   */
  track(
    prev: ImagePyramid,
    cur: ImagePyramid,
    points: Float32Array,
    count: number,
    out?: TrackResult,
    /** Optional initial guesses in `cur` (x,y interleaved); defaults to `points`. */
    guesses: Float32Array | null = null,
    /** Override of `maxDisplacement` (e.g. for relocalization with a known coarse shift). */
    maxDisplacement = this.config.maxDisplacement,
  ): TrackResult {
    const res = out && out.status.length >= count ? out : allocResult(count);
    const cfg = this.config;
    const fbThreshSq = cfg.forwardBackwardThreshold * cfg.forwardBackwardThreshold;
    const maxDispSq = maxDisplacement * maxDisplacement;
    const tmp = new Float32Array(3); // x, y, residual
    let okCount = 0;

    for (let i = 0; i < count; i++) {
      const px = points[i * 2];
      const py = points[i * 2 + 1];
      const gx = guesses ? guesses[i * 2] : px;
      const gy = guesses ? guesses[i * 2 + 1] : py;
      res.fbError[i] = 0;
      res.residual[i] = 0;

      const status = this.trackOne(prev, cur, px, py, gx, gy, tmp);
      if (status !== TrackStatus.OK) {
        res.status[i] = status;
        continue;
      }
      const nx = tmp[0];
      const ny = tmp[1];
      res.residual[i] = tmp[2];
      // Displacement is measured from the initial guess so that a coarse
      // pre-alignment (relocalization) does not count against the gate.
      const dx = nx - gx;
      const dy = ny - gy;
      if (dx * dx + dy * dy > maxDispSq) {
        res.status[i] = TrackStatus.TOO_FAR;
        continue;
      }

      // Backward pass: cur → prev, starting from the original position guess.
      const back = this.trackOne(cur, prev, nx, ny, px, py, tmp);
      if (back !== TrackStatus.OK) {
        res.status[i] = TrackStatus.FB_ERROR;
        continue;
      }
      const bx = tmp[0] - px;
      const by = tmp[1] - py;
      const fbSq = bx * bx + by * by;
      res.fbError[i] = Math.sqrt(fbSq);
      if (fbSq > fbThreshSq) {
        res.status[i] = TrackStatus.FB_ERROR;
        continue;
      }

      res.positions[i * 2] = nx;
      res.positions[i * 2 + 1] = ny;
      res.status[i] = TrackStatus.OK;
      okCount++;
    }
    res.okCount = okCount;
    return res;
  }

  /**
   * Track a single point from `src` (template) to `dst`.
   * (sx, sy): point in src; (gx, gy): initial guess in dst (level 0).
   * Writes [x, y, residual] into `out`.
   */
  private trackOne(
    src: ImagePyramid,
    dst: ImagePyramid,
    sx: number,
    sy: number,
    gx: number,
    gy: number,
    out: Float32Array,
  ): TrackStatus {
    const cfg = this.config;
    const half = this.half;
    const levels = Math.min(cfg.pyramidLevels, src.levels.length);
    const topScale = 1 / (1 << (levels - 1));

    // Displacement estimate at the coarsest level.
    let dX = (gx - sx) * topScale;
    let dY = (gy - sy) * topScale;
    let residual = 0;

    for (let L = levels - 1; L >= 0; L--) {
      const scale = 1 / (1 << L);
      const srcL = src.levels[L];
      const dstL = dst.levels[L];
      const lx = sx * scale;
      const ly = sy * scale;

      if (!this.sampleTemplate(srcL, lx, ly)) {
        // Near the border the window does not fit at coarse levels; skip the
        // level and let the finer ones do the work (OpenCV behaviour).
        if (L === 0) return TrackStatus.OUT_OF_BOUNDS;
        dX *= 2;
        dY *= 2;
        continue;
      }

      // Structure tensor
      const tX = this.tX;
      const tY = this.tY;
      let gxx = 0;
      let gxy = 0;
      let gyy = 0;
      for (let k = 0; k < this.winArea; k++) {
        const ix = tX[k];
        const iy = tY[k];
        gxx += ix * ix;
        gxy += ix * iy;
        gyy += iy * iy;
      }
      const tr = gxx + gyy;
      const det = gxx * gyy - gxy * gxy;
      const minEig = tr / 2 - Math.sqrt(Math.max(0, (tr * tr) / 4 - det));
      if (minEig / this.winArea < cfg.minEigenvalue || det < 1e-9) {
        return TrackStatus.LOW_TEXTURE;
      }
      const invDet = 1 / det;
      const iGxx = gyy * invDet;
      const iGxy = -gxy * invDet;
      const iGyy = gxx * invDet;

      // Gauss-Newton iterations
      for (let iter = 0; iter < cfg.maxIterations; iter++) {
        const cx = lx + dX;
        const cy = ly + dY;
        if (
          cx - half < 1 ||
          cy - half < 1 ||
          cx + half >= dstL.width - 2 ||
          cy + half >= dstL.height - 2
        ) {
          if (L === 0) return TrackStatus.OUT_OF_BOUNDS;
          break; // keep the current estimate, refine at the next level
        }
        const r = this.computeMismatch(dstL, cx, cy);
        const bx = r[0];
        const by = r[1];
        residual = r[2];
        const ddx = iGxx * bx + iGxy * by;
        const ddy = iGxy * bx + iGyy * by;
        dX += ddx;
        dY += ddy;
        if (ddx * ddx + ddy * ddy < cfg.epsilon * cfg.epsilon) break;
      }

      if (L > 0) {
        dX *= 2;
        dY *= 2;
      }
    }

    if (residual > cfg.maxResidual) return TrackStatus.HIGH_RESIDUAL;
    out[0] = sx + dX;
    out[1] = sy + dY;
    out[2] = residual;
    return TrackStatus.OK;
  }

  /** Sample template intensities and gradients around (x, y) in `level`. */
  private sampleTemplate(level: PyramidLevel, x: number, y: number): boolean {
    const half = this.half;
    const { width: w, height: h, data, gradX, gradY } = level;
    if (x - half < 1 || y - half < 1 || x + half >= w - 2 || y + half >= h - 2) return false;

    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    const fx = x - x0;
    const fy = y - y0;
    const w00 = (1 - fx) * (1 - fy);
    const w10 = fx * (1 - fy);
    const w01 = (1 - fx) * fy;
    const w11 = fx * fy;

    const tI = this.tI;
    const tX = this.tX;
    const tY = this.tY;
    let k = 0;
    for (let dy = -half; dy <= half; dy++) {
      let p = (y0 + dy) * w + (x0 - half);
      for (let dx = -half; dx <= half; dx++, p++, k++) {
        tI[k] = w00 * data[p] + w10 * data[p + 1] + w01 * data[p + w] + w11 * data[p + w + 1];
        // gradients are stored ×2 → scale by 0.5
        tX[k] =
          0.5 * (w00 * gradX[p] + w10 * gradX[p + 1] + w01 * gradX[p + w] + w11 * gradX[p + w + 1]);
        tY[k] =
          0.5 * (w00 * gradY[p] + w10 * gradY[p + 1] + w01 * gradY[p + w] + w11 * gradY[p + w + 1]);
      }
    }
    return true;
  }

  private readonly mismatch = new Float64Array(3);

  /**
   * Compute b = Σ ∇T · (T − J) with J sampled at (x, y) in `level`, plus the
   * mean absolute residual. Caller guarantees bounds.
   */
  private computeMismatch(level: PyramidLevel, x: number, y: number): Float64Array {
    const half = this.half;
    const { width: w, data } = level;
    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    const fx = x - x0;
    const fy = y - y0;
    const w00 = (1 - fx) * (1 - fy);
    const w10 = fx * (1 - fy);
    const w01 = (1 - fx) * fy;
    const w11 = fx * fy;

    const tI = this.tI;
    const tX = this.tX;
    const tY = this.tY;
    let bx = 0;
    let by = 0;
    let absSum = 0;
    let k = 0;
    for (let dy = -half; dy <= half; dy++) {
      let p = (y0 + dy) * w + (x0 - half);
      for (let dx = -half; dx <= half; dx++, p++, k++) {
        const j = w00 * data[p] + w10 * data[p + 1] + w01 * data[p + w] + w11 * data[p + w + 1];
        const e = tI[k] - j;
        bx += tX[k] * e;
        by += tY[k] * e;
        absSum += e < 0 ? -e : e;
      }
    }
    const out = this.mismatch;
    out[0] = bx;
    out[1] = by;
    out[2] = absSum / this.winArea;
    return out;
  }
}

export function allocResult(count: number): TrackResult {
  return {
    positions: new Float32Array(count * 2),
    status: new Uint8Array(count),
    residual: new Float32Array(count),
    fbError: new Float32Array(count),
    okCount: 0,
  };
}
