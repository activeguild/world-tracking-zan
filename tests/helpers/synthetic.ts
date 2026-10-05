import { type Mat3, mat3Invert, mat3TransformPoint } from "../../src/math/Matrix";
import { createRng, type Rng } from "../../src/vision/OutlierRejection";

/**
 * Synthetic grayscale image generators shared by unit and browser tests.
 */

export { createRng };

/**
 * Multi-scale smooth random texture: several layers of low-resolution noise,
 * bilinearly upsampled and summed. Has gradients at every pyramid level,
 * unlike white noise.
 */
export function makeTexture(width: number, height: number, rng: Rng, layers = [10, 24, 60, 140]): Uint8Array {
  const acc = new Float32Array(width * height);
  let weight = 1;
  let totalWeight = 0;
  for (const cells of layers) {
    const gw = cells + 2;
    const gh = Math.max(2, Math.round((cells * height) / width)) + 2;
    const grid = new Float32Array(gw * gh);
    for (let i = 0; i < grid.length; i++) grid[i] = rng();
    const sx = (gw - 1) / width;
    const sy = (gh - 1) / height;
    for (let y = 0; y < height; y++) {
      const gy = y * sy;
      const y0 = Math.min(gh - 2, Math.floor(gy));
      const fy = gy - y0;
      for (let x = 0; x < width; x++) {
        const gx = x * sx;
        const x0 = Math.min(gw - 2, Math.floor(gx));
        const fx = gx - x0;
        const v =
          grid[y0 * gw + x0] * (1 - fx) * (1 - fy) +
          grid[y0 * gw + x0 + 1] * fx * (1 - fy) +
          grid[(y0 + 1) * gw + x0] * (1 - fx) * fy +
          grid[(y0 + 1) * gw + x0 + 1] * fx * fy;
        acc[y * width + x] += v * weight;
      }
    }
    totalWeight += weight;
    weight *= 0.8;
  }
  // Sharp-edged "objects" (rectangles and discs) on top of the smooth base,
  // so the image has real corners like a desk or floor with items on it.
  const shapeCount = Math.round((width * height) / 2500);
  for (let s = 0; s < shapeCount; s++) {
    const v = rng();
    const alpha = 0.35 + rng() * 0.5;
    if (rng() < 0.6) {
      const rw = 6 + Math.floor(rng() * Math.min(80, width / 6));
      const rh = 6 + Math.floor(rng() * Math.min(80, height / 6));
      const rx = Math.floor(rng() * (width - rw));
      const ry = Math.floor(rng() * (height - rh));
      for (let y = ry; y < ry + rh; y++) {
        for (let x = rx; x < rx + rw; x++) {
          const i = y * width + x;
          acc[i] = acc[i] * (1 - alpha) + v * totalWeight * alpha;
        }
      }
    } else {
      const r = 3 + rng() * Math.min(30, width / 16);
      const cx = r + rng() * (width - 2 * r);
      const cy = r + rng() * (height - 2 * r);
      const r2 = r * r;
      for (let y = Math.floor(cy - r); y <= Math.ceil(cy + r); y++) {
        if (y < 0 || y >= height) continue;
        for (let x = Math.floor(cx - r); x <= Math.ceil(cx + r); x++) {
          if (x < 0 || x >= width) continue;
          const dx = x - cx;
          const dy = y - cy;
          if (dx * dx + dy * dy > r2) continue;
          const i = y * width + x;
          acc[i] = acc[i] * (1 - alpha) + v * totalWeight * alpha;
        }
      }
    }
  }

  // Light 3×3 box blur: real camera edges are never perfectly sharp.
  const blurred = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    const y0 = Math.max(0, y - 1);
    const y1 = Math.min(height - 1, y + 1);
    for (let x = 0; x < width; x++) {
      const x0 = Math.max(0, x - 1);
      const x1 = Math.min(width - 1, x + 1);
      let sum = 0;
      let cnt = 0;
      for (let yy = y0; yy <= y1; yy++) {
        for (let xx = x0; xx <= x1; xx++) {
          sum += acc[yy * width + xx];
          cnt++;
        }
      }
      blurred[y * width + x] = sum / cnt;
    }
  }

  const out = new Uint8Array(width * height);
  for (let i = 0; i < out.length; i++) {
    out[i] = Math.max(0, Math.min(255, Math.round((blurred[i] / totalWeight) * 255)));
  }
  return out;
}

/** Bilinear sample with constant fill outside the image. */
export function sampleBilinear(img: Uint8Array, w: number, h: number, x: number, y: number, fill = 0): number {
  if (x < 0 || y < 0 || x >= w - 1 || y >= h - 1) return fill;
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const fx = x - x0;
  const fy = y - y0;
  const p = y0 * w + x0;
  return (
    img[p] * (1 - fx) * (1 - fy) +
    img[p + 1] * fx * (1 - fy) +
    img[p + w] * (1 - fx) * fy +
    img[p + w + 1] * fx * fy
  );
}

/**
 * Warp `src` by homography H (src → dst): dst(p) = src(H⁻¹ p).
 */
export function warpImage(src: Uint8Array, w: number, h: number, H: Mat3, fill = 0): Uint8Array {
  const inv = mat3Invert(H);
  if (!inv) throw new Error("non-invertible homography");
  const out = new Uint8Array(w * h);
  const pt = new Float64Array(2);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!mat3TransformPoint(inv, x, y, pt)) {
        out[y * w + x] = fill;
        continue;
      }
      out[y * w + x] = Math.round(sampleBilinear(src, w, h, pt[0], pt[1], fill));
    }
  }
  return out;
}

export function translationH(dx: number, dy: number): Mat3 {
  return new Float64Array([1, 0, dx, 0, 1, dy, 0, 0, 1]);
}

/** Similarity transform about the image center. */
export function similarityH(cx: number, cy: number, angleRad: number, scale: number, dx: number, dy: number): Mat3 {
  const c = Math.cos(angleRad) * scale;
  const s = Math.sin(angleRad) * scale;
  // T(c) · R·S · T(-c) · T(d)
  return new Float64Array([
    c, -s, cx - c * cx + s * cy + dx,
    s, c, cy - s * cx - c * cy + dy,
    0, 0, 1,
  ]);
}

export function translateImage(src: Uint8Array, w: number, h: number, dx: number, dy: number, fill = 0): Uint8Array {
  return warpImage(src, w, h, translationH(dx, dy), fill);
}

/**
 * White image with black axis-aligned squares. Returns the image and the
 * exact corner coordinates (pixel centers of the outermost black pixels).
 */
export function makeSquares(
  width: number,
  height: number,
  squares: { x: number; y: number; size: number }[],
): { image: Uint8Array; corners: { x: number; y: number }[] } {
  const image = new Uint8Array(width * height).fill(230);
  const corners: { x: number; y: number }[] = [];
  for (const s of squares) {
    for (let y = s.y; y < s.y + s.size; y++) {
      image.fill(20, y * width + s.x, y * width + s.x + s.size);
    }
    corners.push(
      { x: s.x, y: s.y },
      { x: s.x + s.size - 1, y: s.y },
      { x: s.x, y: s.y + s.size - 1 },
      { x: s.x + s.size - 1, y: s.y + s.size - 1 },
    );
  }
  return { image, corners };
}

/** Add uniform noise in [-amp, amp] to a copy of the image. */
export function addNoise(img: Uint8Array, amp: number, rng: Rng): Uint8Array {
  const out = new Uint8Array(img.length);
  for (let i = 0; i < img.length; i++) {
    out[i] = Math.max(0, Math.min(255, Math.round(img[i] + (rng() * 2 - 1) * amp)));
  }
  return out;
}

export function median(values: number[]): number {
  if (values.length === 0) return NaN;
  const s = [...values].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}
