import type { FeatureConfig } from "../ar/ARConfig";
import type { Corner } from "./types";

/**
 * FAST-9 corner detector with non-maximum suppression and grid-based
 * spatial distribution (spec §11, §49).
 *
 * Pure TypeScript on TypedArrays. The hot loop is structured so that a later
 * WASM/SIMD port can replace `detectRaw()` without changing the selection
 * logic in `select()`.
 */

/** Bresenham circle of radius 3 (16 pixels), clockwise from the top. */
const CIRCLE_DX = [0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1];
const CIRCLE_DY = [-3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3];
const ARC_LENGTH = 9;
const RADIUS = 3;

export interface DetectOptions {
  /**
   * Optional occupancy mask (width*height, non-zero = blocked). Pixels inside
   * blocked regions are not reported. Used to avoid re-detecting corners next
   * to features already being tracked.
   */
  mask?: Uint8Array | null;
  /** How many new corners the caller wants (drives adaptive thresholding). */
  wanted: number;
  /** Per-grid-cell occupancy from existing tracks (gridCols*gridRows). */
  cellCounts?: Uint16Array | null;
}

export class FeatureDetector {
  /** Current FAST threshold (adaptive). */
  threshold: number;

  private readonly offsets = new Int32Array(16);
  private readonly scoreMap: Int32Array;
  private readonly candidates: Int32Array;
  private candidateCount = 0;
  /** Number of raw corners found by the last `detect()` call (before NMS). */
  lastRawCount = 0;
  /** Number of corners after NMS in the last `detect()` call. */
  lastNmsCount = 0;

  constructor(
    readonly width: number,
    readonly height: number,
    private readonly config: FeatureConfig,
  ) {
    this.threshold = config.fastThreshold;
    for (let i = 0; i < 16; i++) this.offsets[i] = CIRCLE_DY[i] * width + CIRCLE_DX[i];
    this.scoreMap = new Int32Array(width * height);
    // Worst case every 2nd pixel is a corner; cap to keep memory bounded.
    this.candidates = new Int32Array(Math.min(width * height, 65536));
  }

  /**
   * Detect corners, apply NMS, enforce `minDistance` and per-cell quotas and
   * return at most `opts.wanted` corners sorted by descending score.
   */
  detect(gray: Uint8Array, opts: DetectOptions): Corner[] {
    const cfg = this.config;
    let selected: Corner[] = [];
    // When the current threshold yields too few corners, retry immediately
    // with a lower one (bounded) instead of waiting for the next frame.
    for (let attempt = 0; attempt < 3; attempt++) {
      this.detectRaw(gray, opts.mask ?? null);
      const nms = this.nonMaxSuppression();
      selected = this.select(nms, opts);
      if (selected.length >= opts.wanted || this.threshold <= cfg.fastThresholdMin) break;
      this.threshold = Math.max(cfg.fastThresholdMin, this.threshold - 4);
    }
    this.adaptThreshold(opts.wanted, selected.length);
    return selected;
  }

  /**
   * FAST-9 segment test over the whole image. Fills `scoreMap` at corner
   * positions and records their indices in `candidates`.
   */
  private detectRaw(gray: Uint8Array, mask: Uint8Array | null): void {
    const { width: w, height: h } = this;
    const t = this.threshold;
    const off = this.offsets;
    const scoreMap = this.scoreMap;
    const cands = this.candidates;
    const margin = Math.max(RADIUS, this.config.borderMargin);
    let count = 0;
    const maxCands = cands.length;

    for (let y = margin; y < h - margin && count < maxCands; y++) {
      const row = y * w;
      for (let x = margin; x < w - margin; x++) {
        const p = row + x;
        if (mask !== null && mask[p] !== 0) continue;
        const c = gray[p];
        const hi = c + t;
        const lo = c - t;

        // Fast rejection on the 4 compass points. A run of 9 contiguous circle
        // pixels always contains one of {0, 8} and at least two of {0,4,8,12}.
        let bright = 0;
        let dark = 0;
        let v = gray[p + off[0]];
        if (v > hi) bright++;
        else if (v < lo) dark++;
        v = gray[p + off[8]];
        if (v > hi) bright++;
        else if (v < lo) dark++;
        if (bright === 0 && dark === 0) continue;
        v = gray[p + off[4]];
        if (v > hi) bright++;
        else if (v < lo) dark++;
        v = gray[p + off[12]];
        if (v > hi) bright++;
        else if (v < lo) dark++;
        if (bright < 2 && dark < 2) continue;

        const score = segmentTest(gray, p, off, hi, lo, c, t);
        if (score > 0) {
          scoreMap[p] = score;
          cands[count++] = p;
          if (count >= maxCands) break;
        }
      }
    }
    this.candidateCount = count;
    this.lastRawCount = count;
  }

  /** 3×3 non-maximum suppression over the candidate set. Clears scoreMap. */
  private nonMaxSuppression(): Corner[] {
    const w = this.width;
    const scoreMap = this.scoreMap;
    const cands = this.candidates;
    const n = this.candidateCount;
    const out: Corner[] = [];
    for (let k = 0; k < n; k++) {
      const p = cands[k];
      const s = scoreMap[p];
      if (
        s > scoreMap[p - 1] &&
        s >= scoreMap[p + 1] &&
        s > scoreMap[p - w - 1] &&
        s > scoreMap[p - w] &&
        s > scoreMap[p - w + 1] &&
        s >= scoreMap[p + w - 1] &&
        s >= scoreMap[p + w] &&
        s >= scoreMap[p + w + 1]
      ) {
        out.push({ x: p % w, y: (p / w) | 0, score: s });
      }
    }
    // Reset only the touched cells so the map is clean for the next frame.
    for (let k = 0; k < n; k++) scoreMap[cands[k]] = 0;
    this.lastNmsCount = out.length;
    return out;
  }

  /**
   * Grid-based selection: strongest corners first, respecting `minDistance`
   * (via a temporary stamp into the caller-provided mask or a local one) and
   * the per-cell quota.
   */
  private select(corners: Corner[], opts: DetectOptions): Corner[] {
    const cfg = this.config;
    const wanted = Math.max(0, opts.wanted);
    if (wanted === 0 || corners.length === 0) return [];
    corners.sort((a, b) => b.score - a.score);

    const cellW = this.width / cfg.gridCols;
    const cellH = this.height / cfg.gridRows;
    const cellCounts = opts.cellCounts
      ? Uint16Array.from(opts.cellCounts)
      : new Uint16Array(cfg.gridCols * cfg.gridRows);

    const minDist = cfg.minDistance;
    const minDistSq = minDist * minDist;
    // Simple spatial hash of accepted points for the min-distance test.
    const hashCell = Math.max(1, minDist);
    const hashCols = Math.ceil(this.width / hashCell) + 1;
    const hashRows = Math.ceil(this.height / hashCell) + 1;
    const buckets: Corner[][] = new Array(hashCols * hashRows);

    const selected: Corner[] = [];
    for (const c of corners) {
      if (selected.length >= wanted) break;
      const cx = Math.min(cfg.gridCols - 1, (c.x / cellW) | 0);
      const cy = Math.min(cfg.gridRows - 1, (c.y / cellH) | 0);
      const cell = cy * cfg.gridCols + cx;
      if (cellCounts[cell] >= cfg.maxPerCell) continue;

      // min-distance check against accepted corners in neighbouring buckets
      const hx = (c.x / hashCell) | 0;
      const hy = (c.y / hashCell) | 0;
      let tooClose = false;
      for (let by = hy - 1; by <= hy + 1 && !tooClose; by++) {
        if (by < 0 || by >= hashRows) continue;
        for (let bx = hx - 1; bx <= hx + 1; bx++) {
          if (bx < 0 || bx >= hashCols) continue;
          const b = buckets[by * hashCols + bx];
          if (!b) continue;
          for (const o of b) {
            const dx = o.x - c.x;
            const dy = o.y - c.y;
            if (dx * dx + dy * dy < minDistSq) {
              tooClose = true;
              break;
            }
          }
          if (tooClose) break;
        }
      }
      if (tooClose) continue;

      (buckets[hy * hashCols + hx] ??= []).push(c);
      cellCounts[cell]++;
      selected.push(c);
    }
    return selected;
  }

  /**
   * Adaptive threshold: too few raw corners → lower, far too many → raise.
   * Keeps FAST cheap on busy textures and sensitive on low-texture surfaces.
   */
  private adaptThreshold(wanted: number, selectedCount: number): void {
    const cfg = this.config;
    if (wanted <= 0) return;
    if (selectedCount < wanted) {
      this.threshold = Math.max(cfg.fastThresholdMin, this.threshold - 2);
    } else if (this.lastNmsCount > wanted * 6) {
      this.threshold = Math.min(cfg.fastThresholdMax, this.threshold + 2);
    }
  }
}

/**
 * Full 16-pixel segment test. Returns a score > 0 when at least ARC_LENGTH
 * contiguous circle pixels are all brighter than `hi` or all darker than `lo`.
 * Score = sum over the qualifying pixels of (|I − c| − t).
 */
function segmentTest(
  gray: Uint8Array,
  p: number,
  off: Int32Array,
  hi: number,
  lo: number,
  c: number,
  t: number,
): number {
  let brightRun = 0;
  let darkRun = 0;
  let brightMax = 0;
  let darkMax = 0;
  let brightScore = 0;
  let darkScore = 0;
  // Walk 16 + (ARC_LENGTH - 1) pixels to handle wrap-around.
  for (let k = 0; k < 16 + ARC_LENGTH - 1; k++) {
    const v = gray[p + off[k & 15]];
    if (v > hi) {
      brightRun++;
      darkRun = 0;
      if (brightRun > brightMax) brightMax = brightRun;
    } else if (v < lo) {
      darkRun++;
      brightRun = 0;
      if (darkRun > darkMax) darkMax = darkRun;
    } else {
      brightRun = 0;
      darkRun = 0;
    }
    if (k < 16) {
      if (v > hi) brightScore += v - c - t;
      else if (v < lo) darkScore += c - v - t;
    }
  }
  if (brightMax >= ARC_LENGTH) return brightScore + 1;
  if (darkMax >= ARC_LENGTH) return darkScore + 1;
  return 0;
}
