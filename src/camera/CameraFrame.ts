import type { CameraIntrinsics } from "./CameraIntrinsics";

/**
 * A single grayscale processing frame (spec §9, §41).
 *
 * `data` is a plain Uint8Array over an ArrayBuffer so that it can be moved to
 * the vision worker with zero copy (Transferable) and returned for reuse.
 */
export interface GrayFrame {
  frameId: number;
  /** performance.now() based timestamp in ms. */
  timestamp: number;
  width: number;
  height: number;
  data: Uint8Array;
  intrinsics: CameraIntrinsics;
}

/**
 * Convert RGBA pixels to grayscale (Rec.601 luma, integer arithmetic).
 * Exported for unit tests and for the worker fallback path.
 */
export function rgbaToGray(rgba: Uint8ClampedArray | Uint8Array, out: Uint8Array): void {
  const n = out.length;
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    // 0.299 R + 0.587 G + 0.114 B  ≈ (77 R + 150 G + 29 B) >> 8
    out[i] = (77 * rgba[j] + 150 * rgba[j + 1] + 29 * rgba[j + 2]) >> 8;
  }
}

/**
 * Grabs the current video frame into a small canvas at processing resolution
 * and converts it to grayscale.
 *
 * Buffers are pooled: callers hand buffers back via `release()` after the
 * worker has transferred them back, so steady-state operation allocates
 * nothing per frame.
 */
export class FrameGrabber {
  private readonly canvas: HTMLCanvasElement | OffscreenCanvas;
  private readonly ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
  private readonly pool: ArrayBuffer[] = [];
  private nextFrameId = 0;

  constructor(
    public readonly width: number,
    public readonly height: number,
  ) {
    if (typeof OffscreenCanvas !== "undefined") {
      this.canvas = new OffscreenCanvas(width, height);
    } else {
      const c = document.createElement("canvas");
      c.width = width;
      c.height = height;
      this.canvas = c;
    }
    const ctx = this.canvas.getContext("2d", { willReadFrequently: true }) as
      | CanvasRenderingContext2D
      | OffscreenCanvasRenderingContext2D
      | null;
    if (!ctx) throw new Error("2D canvas context unavailable");
    this.ctx = ctx;
  }

  /**
   * Compute a processing size that keeps the video aspect ratio and whose
   * larger side equals `targetWidth` (e.g. 1280×720 → 640×360, 640×480 → 640×480).
   * Dimensions are rounded to multiples of 4.
   */
  static fitProcessingSize(
    videoWidth: number,
    videoHeight: number,
    targetWidth: number,
    targetHeight: number,
  ): { width: number; height: number } {
    const landscape = videoWidth >= videoHeight;
    const longTarget = Math.max(targetWidth, targetHeight);
    const scale = longTarget / (landscape ? videoWidth : videoHeight);
    const w = Math.max(4, Math.round((videoWidth * scale) / 4) * 4);
    const h = Math.max(4, Math.round((videoHeight * scale) / 4) * 4);
    return { width: w, height: h };
  }

  grab(video: CanvasImageSource, timestamp: number, intrinsics: CameraIntrinsics): GrayFrame {
    const { width, height } = this;
    this.ctx.drawImage(video, 0, 0, width, height);
    const imageData = this.ctx.getImageData(0, 0, width, height);
    const buffer = this.pool.pop() ?? new ArrayBuffer(width * height);
    const gray = new Uint8Array(buffer);
    rgbaToGray(imageData.data, gray);
    return {
      frameId: this.nextFrameId++,
      timestamp,
      width,
      height,
      data: gray,
      intrinsics,
    };
  }

  /** Return a buffer to the pool for reuse. */
  release(buffer: ArrayBuffer): void {
    if (buffer.byteLength === this.width * this.height && this.pool.length < 4) {
      this.pool.push(buffer);
    }
  }
}
