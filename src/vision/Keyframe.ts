import type { RigidTransform } from "../math/Pose";
import { ImagePyramid } from "./ImagePyramid";

/**
 * Keyframe (spec §35): a stored view of the map with its camera pose, the
 * image pyramid (for LK re-tracking) and the landmark observations.
 */
export interface KeyframeObservation {
  landmarkId: number;
  x: number;
  y: number;
}

export interface Keyframe {
  id: number;
  frameId: number;
  timestamp: number;
  /** X_cam = R X_map + t */
  pose: RigidTransform;
  pyramid: ImagePyramid;
  /** Coarse (level-3 equivalent) image used for the global shift search. */
  coarse: { width: number; height: number; data: Uint8Array; mean: number; norm: number };
  observations: KeyframeObservation[];
}

/**
 * Downsample a pyramid level by 2 (box filter) into a fresh buffer.
 */
export function downsampleToCoarse(src: { width: number; height: number; data: Uint8Array }): {
  width: number;
  height: number;
  data: Uint8Array;
  mean: number;
  norm: number;
} {
  const w = src.width >> 1;
  const h = src.height >> 1;
  const out = new Uint8Array(w * h);
  const s = src.data;
  const sw = src.width;
  let sum = 0;
  for (let y = 0; y < h; y++) {
    let si = y * 2 * sw;
    let di = y * w;
    for (let x = 0; x < w; x++, si += 2, di++) {
      const v = (s[si] + s[si + 1] + s[si + sw] + s[si + sw + 1] + 2) >> 2;
      out[di] = v;
      sum += v;
    }
  }
  const mean = sum / (w * h);
  let norm = 0;
  for (let i = 0; i < out.length; i++) {
    const d = out[i] - mean;
    norm += d * d;
  }
  return { width: w, height: h, data: out, mean, norm: Math.sqrt(norm) };
}

/** Deep-copy a pyramid (the engine reuses its buffers every frame). */
export function clonePyramid(src: ImagePyramid): ImagePyramid {
  const p = new ImagePyramid(src.width, src.height, src.numLevels);
  for (let i = 0; i < p.levels.length && i < src.levels.length; i++) {
    p.levels[i].data.set(src.levels[i].data);
    p.levels[i].gradX.set(src.levels[i].gradX);
    p.levels[i].gradY.set(src.levels[i].gradY);
  }
  return p;
}
