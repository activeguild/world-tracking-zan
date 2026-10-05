/**
 * Pose-synchronized camera frame display.
 *
 * The live <video> runs ~2 frames ahead of the pose the vision engine returns
 * for it, so rendered objects would lag behind the image during motion. The
 * presenter keeps a small ring of canvases: every frame handed to the vision
 * engine is also copied (GPU blit, synchronous) into the next ring slot, and
 * when that frame's pose arrives the slot is drawn onto the visible canvas.
 *
 * Deliberately not `createImageBitmap(video)`: on iOS Safari that call is
 * slow (full-resolution readback, tens of ms) and asynchronous, so the copies
 * piled up, starved the main thread and made the capture loop drop frames,
 * which in turn broke feature tracking.
 *
 * Only the vision engine's processed frames are shown, so the display runs at
 * vision rate (15–30 fps) rather than camera rate; that is the price of
 * time alignment and what ARKit-style pipelines do as well.
 */
export class FramePresenter {
  private readonly ring: { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D; frameId: number }[] = [];
  private readonly targetCtx: CanvasRenderingContext2D | null;
  private width = 0;
  private height = 0;
  private next = 0;

  constructor(
    private readonly target: HTMLCanvasElement,
    ringSize = 4,
    /** Device pixel ratio cap for the display canvas. */
    private readonly maxDpr = 2,
    /** Long-side cap of the display canvas in device pixels. */
    private readonly maxLongSide = 1440,
  ) {
    this.targetCtx = target.getContext("2d");
    for (let i = 0; i < ringSize; i++) {
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      if (!ctx) break;
      this.ring.push({ canvas, ctx, frameId: -1 });
    }
  }

  /** False when 2D canvases are unavailable; the caller then shows the live video. */
  get available(): boolean {
    return this.targetCtx !== null && this.ring.length > 0;
  }

  /** Copy the current video image into the ring slot for `frameId` (object-fit: cover). */
  capture(video: HTMLVideoElement, frameId: number): void {
    if (!this.available) return;
    this.updateSize();
    const slot = this.ring[this.next];
    this.next = (this.next + 1) % this.ring.length;
    slot.frameId = frameId;
    const vw = video.videoWidth || 1;
    const vh = video.videoHeight || 1;
    const scale = Math.max(this.width / vw, this.height / vh);
    const dw = vw * scale;
    const dh = vh * scale;
    slot.ctx.drawImage(video, (this.width - dw) / 2, (this.height - dh) / 2, dw, dh);
  }

  /** Show the frame captured for `frameId`. Returns false when it is no longer in the ring. */
  present(frameId: number): boolean {
    if (!this.targetCtx) return false;
    const slot = this.ring.find((s) => s.frameId === frameId);
    if (!slot || slot.canvas.width === 0) return false;
    if (this.target.width !== slot.canvas.width || this.target.height !== slot.canvas.height) {
      this.target.width = slot.canvas.width;
      this.target.height = slot.canvas.height;
    }
    this.targetCtx.drawImage(slot.canvas, 0, 0);
    return true;
  }

  clear(): void {
    for (const s of this.ring) s.frameId = -1;
  }

  private updateSize(): void {
    const rect = this.target.getBoundingClientRect();
    const dpr = Math.min(this.maxDpr, window.devicePixelRatio || 1);
    let w = Math.max(1, Math.round(rect.width * dpr));
    let h = Math.max(1, Math.round(rect.height * dpr));
    const long = Math.max(w, h);
    if (long > this.maxLongSide) {
      const k = this.maxLongSide / long;
      w = Math.max(1, Math.round(w * k));
      h = Math.max(1, Math.round(h * k));
    }
    if (w === this.width && h === this.height) return;
    this.width = w;
    this.height = h;
    for (const s of this.ring) {
      s.canvas.width = w;
      s.canvas.height = h;
      s.frameId = -1;
    }
  }
}
