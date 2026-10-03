import * as THREE from "three";

/**
 * Transparent WebGL renderer laid over the camera video. Sized to the
 * canvas' CSS box, device-pixel-ratio aware (capped for mobile GPUs).
 */
export class ARRenderer {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene: THREE.Scene;
  private width = 1;
  private height = 1;

  constructor(
    readonly canvas: HTMLCanvasElement,
    scene?: THREE.Scene,
    maxPixelRatio = 2,
  ) {
    this.renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, powerPreference: "high-performance" });
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.setPixelRatio(Math.min(maxPixelRatio, window.devicePixelRatio || 1));
    this.scene = scene ?? new THREE.Scene();
    this.resize();
  }

  /** Match the drawing buffer to the canvas' CSS size. Returns true when the size changed. */
  resize(): boolean {
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width));
    const h = Math.max(1, Math.round(rect.height));
    if (w === this.width && h === this.height) return false;
    this.width = w;
    this.height = h;
    this.renderer.setSize(w, h, false);
    return true;
  }

  get viewportWidth(): number {
    return this.width;
  }

  get viewportHeight(): number {
    return this.height;
  }

  render(camera: THREE.Camera): void {
    this.renderer.render(this.scene, camera);
  }

  dispose(): void {
    this.renderer.dispose();
  }
}
