/**
 * Grayscale image pyramid with per-level gradients (spec §12).
 *
 *   Level 0: W × H          (e.g. 640 × 480)
 *   Level 1: W/2 × H/2      (320 × 240)
 *   Level 2: W/4 × H/4      (160 × 120)
 *
 * Downsampling is a 2×2 box filter. Gradients are central differences stored
 * as Int16 (not halved: Ix = I(x+1) − I(x−1)). Buffers are allocated once and
 * reused across frames.
 */
export interface PyramidLevel {
  width: number;
  height: number;
  data: Uint8Array;
  /** Central-difference gradient along x, ×2 (Int16). Zero on the border. */
  gradX: Int16Array;
  /** Central-difference gradient along y, ×2 (Int16). Zero on the border. */
  gradY: Int16Array;
}

export class ImagePyramid {
  readonly levels: PyramidLevel[] = [];

  constructor(
    readonly width: number,
    readonly height: number,
    readonly numLevels: number,
  ) {
    let w = width;
    let h = height;
    for (let i = 0; i < numLevels; i++) {
      this.levels.push({
        width: w,
        height: h,
        data: new Uint8Array(w * h),
        gradX: new Int16Array(w * h),
        gradY: new Int16Array(w * h),
      });
      w = w >> 1;
      h = h >> 1;
      if (w < 8 || h < 8) break;
    }
  }

  /** Fill the pyramid from a level-0 grayscale image (copied). */
  build(gray: Uint8Array): void {
    const l0 = this.levels[0];
    l0.data.set(gray);
    computeGradients(l0);
    for (let i = 1; i < this.levels.length; i++) {
      downsample2x(this.levels[i - 1], this.levels[i]);
      computeGradients(this.levels[i]);
    }
  }
}

export function downsample2x(src: PyramidLevel, dst: PyramidLevel): void {
  const sw = src.width;
  const s = src.data;
  const d = dst.data;
  const dw = dst.width;
  const dh = dst.height;
  for (let y = 0; y < dh; y++) {
    let si = y * 2 * sw;
    let di = y * dw;
    for (let x = 0; x < dw; x++, si += 2, di++) {
      d[di] = (s[si] + s[si + 1] + s[si + sw] + s[si + sw + 1] + 2) >> 2;
    }
  }
}

export function computeGradients(level: PyramidLevel): void {
  const { width: w, height: h, data, gradX, gradY } = level;
  // Borders stay zero (never sampled by LK: the window keeps a 1px margin).
  for (let y = 1; y < h - 1; y++) {
    const row = y * w;
    for (let x = 1; x < w - 1; x++) {
      const i = row + x;
      gradX[i] = data[i + 1] - data[i - 1];
      gradY[i] = data[i + w] - data[i - w];
    }
  }
}
