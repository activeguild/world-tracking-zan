import type { CameraIntrinsics } from "../camera/CameraIntrinsics";
import { LANDMARK_STRIDE, type MapPoseOutput, type PlaneOutput } from "../vision/types";
import { computeCoverMapping } from "./FeatureRenderer";

/**
 * 2D debug rendering of the landmark map and the detected plane (spec §44).
 *
 * Landmarks and a grid on the plane are projected with the map-frame camera
 * pose and drawn on the same overlay canvas as the features. Phase 4 adds
 * the Three.js version; this one exists so Phase 3 can be verified on a
 * phone without a 3D renderer.
 */
export class PlaneRenderer {
  private readonly ctx: CanvasRenderingContext2D;

  constructor(readonly canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("2D context unavailable for plane overlay");
    this.ctx = ctx;
  }

  draw(
    mapPose: MapPoseOutput | null,
    plane: PlaneOutput | null,
    landmarks: Float32Array,
    landmarkCount: number,
    k: CameraIntrinsics,
    /** When the Three.js world exists it draws the plane itself; skip the 2D grid then. */
    drawGrid = true,
  ): void {
    if (!mapPose) return;
    const ctx = this.ctx;
    const { width: W, height: H } = this.canvas;
    const m = computeCoverMapping(k.width, k.height, W, H);
    const r = mapPose.rotation;
    const t = mapPose.translation;
    const px = (X: number, Y: number, Z: number): [number, number] | null => {
      const z = r[6] * X + r[7] * Y + r[8] * Z + t[2];
      if (z <= 1e-6) return null;
      const u = ((r[0] * X + r[1] * Y + r[2] * Z + t[0]) / z) * k.fx + k.cx;
      const v = ((r[3] * X + r[4] * Y + r[5] * Z + t[1]) / z) * k.fy + k.cy;
      return [u * m.scale + m.offsetX, v * m.scale + m.offsetY];
    };

    // Landmarks: small squares, cyan when on the plane, grey otherwise.
    const s = Math.max(2, 2 * (W / 640));
    for (let i = 0; i < landmarkCount; i++) {
      const o = i * LANDMARK_STRIDE;
      const p = px(landmarks[o], landmarks[o + 1], landmarks[o + 2]);
      if (!p) continue;
      ctx.fillStyle = landmarks[o + 3] > 0.5 ? "rgba(0, 220, 255, 0.9)" : "rgba(200, 200, 200, 0.5)";
      ctx.fillRect(p[0] - s, p[1] - s, 2 * s, 2 * s);
    }

    if (!plane || !drawGrid) return;

    // Plane grid: centred on the plane centroid, spanning ~ the inlier
    // extent, capped by the camera's distance to the plane.
    const n = plane.normal;
    const c = plane.center;
    const ref = Math.abs(n[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    const u = normalize(cross(n, ref));
    const v = cross(n, u);
    const camC = [
      -(r[0] * t[0] + r[3] * t[1] + r[6] * t[2]),
      -(r[1] * t[0] + r[4] * t[1] + r[7] * t[2]),
      -(r[2] * t[0] + r[5] * t[1] + r[8] * t[2]),
    ];
    const camDist = Math.abs(n[0] * camC[0] + n[1] * camC[1] + n[2] * camC[2] + plane.d);
    const half = Math.min(Math.max(0.05, Math.sqrt(Math.max(plane.areaEstimate, 1e-6)) / 2), Math.max(0.05, camDist));
    const cells = 8;
    const step = (2 * half) / cells;

    ctx.lineWidth = Math.max(1, W / 800);
    ctx.strokeStyle = plane.found
      ? "rgba(0, 255, 140, 0.8)"
      : plane.horizontal
        ? "rgba(255, 220, 0, 0.7)"
        : "rgba(255, 80, 80, 0.6)";
    ctx.beginPath();
    for (let i = 0; i <= cells; i++) {
      const a = -half + i * step;
      // lines along v
      const p0 = px(c[0] + u[0] * a + v[0] * -half, c[1] + u[1] * a + v[1] * -half, c[2] + u[2] * a + v[2] * -half);
      const p1 = px(c[0] + u[0] * a + v[0] * half, c[1] + u[1] * a + v[1] * half, c[2] + u[2] * a + v[2] * half);
      if (p0 && p1) {
        ctx.moveTo(p0[0], p0[1]);
        ctx.lineTo(p1[0], p1[1]);
      }
      // lines along u
      const q0 = px(c[0] + v[0] * a + u[0] * -half, c[1] + v[1] * a + u[1] * -half, c[2] + v[2] * a + u[2] * -half);
      const q1 = px(c[0] + v[0] * a + u[0] * half, c[1] + v[1] * a + u[1] * half, c[2] + v[2] * a + u[2] * half);
      if (q0 && q1) {
        ctx.moveTo(q0[0], q0[1]);
        ctx.lineTo(q1[0], q1[1]);
      }
    }
    ctx.stroke();

    // Normal arrow from the centre.
    const base = px(c[0], c[1], c[2]);
    const tip = px(c[0] + n[0] * half * 0.5, c[1] + n[1] * half * 0.5, c[2] + n[2] * half * 0.5);
    if (base && tip) {
      ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
      ctx.beginPath();
      ctx.moveTo(base[0], base[1]);
      ctx.lineTo(tip[0], tip[1]);
      ctx.stroke();
    }
  }
}

function cross(a: ArrayLike<number>, b: ArrayLike<number>): number[] {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function normalize(v: number[]): number[] {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}
