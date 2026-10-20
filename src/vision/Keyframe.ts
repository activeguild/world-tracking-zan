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
  /**
   * The origin keyframe (指示書「座標系整合性と空間固定性」§3): the two-view
   * *reference* camera of the map initialization, stored with its image and
   * the init landmarks at their reference pixels. Its pose is the identity —
   * it *is* the map frame — and it is immutable: the init stages and the
   * keyframe bundle adjustment hold it fixed, the stage post-processing never
   * re-poses it, eviction never drops it. Explicit flag rather than "index 0".
   */
  origin: boolean;
  /** X_cam = R X_map + t */
  pose: RigidTransform;
  pyramid: ImagePyramid;
  /** Coarse (level-3 equivalent, 1/8) image used for the global shift search. */
  coarse: CoarseImage;
  /** Even coarser (1/16) image used to rank all keyframes cheaply every attempt (v14 §10). */
  rank: CoarseImage;
  observations: KeyframeObservation[];
}

export interface CoarseImage {
  width: number;
  height: number;
  data: Uint8Array;
  mean: number;
  norm: number;
}

/**
 * Downsample a pyramid level by 2 (box filter) into a fresh buffer.
 */
export function downsampleToCoarse(src: { width: number; height: number; data: Uint8Array }): CoarseImage {
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
  copyPyramidInto(p, src);
  return p;
}

/** Copy `src` into the already allocated `dst` of the same geometry (no allocation). */
export function copyPyramidInto(dst: ImagePyramid, src: ImagePyramid): void {
  for (let i = 0; i < dst.levels.length && i < src.levels.length; i++) {
    dst.levels[i].data.set(src.levels[i].data);
    dst.levels[i].gradX.set(src.levels[i].gradX);
    dst.levels[i].gradY.set(src.levels[i].gradY);
  }
}
