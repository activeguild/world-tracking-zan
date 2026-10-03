import { TRACK_STRIDE } from "../vision/types";

/**
 * Draws tracked features and their motion vectors on a 2D overlay canvas
 * (spec §43).
 *
 *   ● inlier (green), age-coded alpha
 *   ● new / unverified (yellow)
 *   ● ───→ ●  motion since the previous frame
 *
 * Processing-image coordinates are mapped onto the video's `object-fit: cover`
 * display rectangle so that dots land on the pixels they describe.
 */
export interface ViewMapping {
  scale: number;
  offsetX: number;
  offsetY: number;
}

export function computeCoverMapping(
  srcWidth: number,
  srcHeight: number,
  dstWidth: number,
  dstHeight: number,
): ViewMapping {
  const scale = Math.max(dstWidth / srcWidth, dstHeight / srcHeight);
  return {
    scale,
    offsetX: (dstWidth - srcWidth * scale) / 2,
    offsetY: (dstHeight - srcHeight * scale) / 2,
  };
}

export class FeatureRenderer {
  private readonly ctx: CanvasRenderingContext2D;
  showVectors = true;

  constructor(readonly canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("2D context unavailable for overlay");
    this.ctx = ctx;
  }

  /** Match the canvas backing store to its CSS size (device pixel ratio aware). */
  resize(): void {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width * dpr));
    const h = Math.max(1, Math.round(rect.height * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
  }

  clear(): void {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  draw(tracks: Float32Array, trackCount: number, srcWidth: number, srcHeight: number): void {
    const ctx = this.ctx;
    const { width: W, height: H } = this.canvas;
    ctx.clearRect(0, 0, W, H);
    const m = computeCoverMapping(srcWidth, srcHeight, W, H);
    const r = Math.max(2, 2.5 * (W / 640));

    // Motion vectors first so dots sit on top.
    if (this.showVectors) {
      ctx.lineWidth = Math.max(1, W / 640);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.55)";
      ctx.beginPath();
      for (let i = 0; i < trackCount; i++) {
        const o = i * TRACK_STRIDE;
        const age = tracks[o + 5];
        if (age < 1) continue;
        const x = tracks[o + 1] * m.scale + m.offsetX;
        const y = tracks[o + 2] * m.scale + m.offsetY;
        const px = tracks[o + 3] * m.scale + m.offsetX;
        const py = tracks[o + 4] * m.scale + m.offsetY;
        ctx.moveTo(px, py);
        ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // Dots
    for (let i = 0; i < trackCount; i++) {
      const o = i * TRACK_STRIDE;
      const x = tracks[o + 1] * m.scale + m.offsetX;
      const y = tracks[o + 2] * m.scale + m.offsetY;
      const age = tracks[o + 5];
      const inlier = tracks[o + 6] > 0.5;
      if (age < 1) {
        ctx.fillStyle = "rgba(255, 220, 0, 0.9)";
      } else if (inlier) {
        const a = Math.min(1, 0.5 + age / 30);
        ctx.fillStyle = `rgba(40, 255, 120, ${a.toFixed(2)})`;
      } else {
        ctx.fillStyle = "rgba(255, 60, 60, 0.9)";
      }
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}
