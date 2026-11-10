/**
 * Height of the floor the map knows under a tap (recordings 45 / 46).
 *
 * The world plane is fixed once, from the first plane the detector accepts,
 * about a second after the map is initialized, when the map's depth
 * structure is still poorly determined. On device the landmarks around the
 * placed cube sat 7–16 % of the depth *below* that plane: the cube floated
 * 5–11 cm above the floor the map tracks, so it slid against the floor when
 * the phone tilted (and came back when the view was restored) and changed
 * size differently from the floor texture.
 *
 * A hit test therefore asks the landmarks near the tap how high the floor is
 * there, relative to the world plane, and intersects the ray with the plane
 * shifted by that height (same normal: the object still stands upright in
 * the world). The height is the median of the neighbours inside a
 * horizontal radius and a height window; because shifting the plane moves
 * the hit point along the ray, the neighbourhood is re-collected and the
 * height re-estimated a few times. Too few neighbours, or neighbours that
 * do not agree on a height (not a floor: clutter, an edge), fall back to
 * the world plane.
 *
 * Pure geometry in the map frame; nothing in the engine or the anchor is
 * touched.
 */
import { LANDMARK_STRIDE } from "../vision/types";

export interface LocalFloorConfig {
  /** Horizontal neighbourhood radius = ratio × depth of the hit (tried at 1× and 2×). */
  radiusDepthRatio: number;
  /** Neighbours farther above / below the current height than ratio × depth are not this floor. */
  heightWindowDepthRatio: number;
  /** Fewer neighbours than this: world plane. */
  minLandmarks: number;
  /** Median absolute deviation of the neighbour heights above ratio × depth: not a floor, world plane. */
  maxSpreadDepthRatio: number;
}

export interface LocalFloorHit {
  /** Hit point in the map frame. */
  point: Float64Array;
  /** Ray parameter of the hit (map units for a unit direction). */
  t: number;
  /** Height of the surface used above the world plane (map units; 0 = the world plane). */
  height: number;
  /** "local" when the neighbours decided the height, "plane" for the world plane fallback, "fixed" for a caller-given height. */
  surface: "local" | "plane" | "fixed";
  /** Neighbours behind the decision (0 on a fallback without any). */
  neighbors: number;
  /** Their height spread (MAD, map units; NaN without neighbours). */
  spread: number;
  /** Depth of the hit along the optical axis (map units). */
  depth: number;
  /** Why the world plane was used ("" when local). */
  fallback: "" | "few_landmarks" | "spread" | "miss";
}

function median(values: number[]): number {
  const s = values.slice().sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : 0.5 * (s[m - 1] + s[m]);
}

/** Ray ∩ {X : up·(X − origin) = h}; null when parallel or behind the origin of the ray. */
function intersect(c: ArrayLike<number>, d: ArrayLike<number>, up: ArrayLike<number>, origin: ArrayLike<number>, h: number): { t: number; p: Float64Array } | null {
  const denom = up[0] * d[0] + up[1] * d[1] + up[2] * d[2];
  if (Math.abs(denom) < 1e-9) return null;
  const num = h - (up[0] * (c[0] - origin[0]) + up[1] * (c[1] - origin[1]) + up[2] * (c[2] - origin[2]));
  const t = num / denom;
  if (!(t > 1e-9) || !Number.isFinite(t)) return null;
  return { t, p: new Float64Array([c[0] + t * d[0], c[1] + t * d[1], c[2] + t * d[2]]) };
}

/**
 * @param rayOrigin  camera centre (map frame)
 * @param rayDir     ray direction (map frame, any length)
 * @param forward    camera optical axis in the map frame (third row of R)
 * @param up, origin the world plane in the map frame
 * @param landmarks  packed `LANDMARK_STRIDE` floats per landmark (map frame)
 * @returns null when the ray misses the world plane too
 */
export function localFloorHit(
  rayOrigin: ArrayLike<number>,
  rayDir: ArrayLike<number>,
  forward: ArrayLike<number>,
  up: ArrayLike<number>,
  origin: ArrayLike<number>,
  landmarks: Float32Array | null,
  landmarkCount: number,
  cfg: LocalFloorConfig | null,
  fixedHeight?: number,
): LocalFloorHit | null {
  // A caller-given height (map units above the world plane): the ray meets
  // that plane and nothing is estimated. Used while an object is dragged, so
  // that it slides at the height it was placed at (recording 48: re-estimating
  // the floor at every drag event flipped the cube between two heights).
  if (fixedHeight !== undefined && Number.isFinite(fixedHeight)) {
    const f = intersect(rayOrigin, rayDir, up, origin, fixedHeight);
    if (!f) return null;
    const depth = forward[0] * (f.p[0] - rayOrigin[0]) + forward[1] * (f.p[1] - rayOrigin[1]) + forward[2] * (f.p[2] - rayOrigin[2]);
    return { point: f.p, t: f.t, height: fixedHeight, surface: "fixed", neighbors: 0, spread: NaN, depth, fallback: "" };
  }
  const base = intersect(rayOrigin, rayDir, up, origin, 0);
  if (!base) return null;
  const depthOf = (p: ArrayLike<number>) =>
    forward[0] * (p[0] - rayOrigin[0]) + forward[1] * (p[1] - rayOrigin[1]) + forward[2] * (p[2] - rayOrigin[2]);
  const plane = (fallback: LocalFloorHit["fallback"], neighbors = 0, spread = NaN): LocalFloorHit => ({
    point: base.p,
    t: base.t,
    height: 0,
    surface: "plane",
    neighbors,
    spread,
    depth: depthOf(base.p),
    fallback,
  });
  if (!cfg || !landmarks || landmarkCount <= 0) return plane("few_landmarks");

  // Heights above the world plane once; the neighbourhood moves with the hit.
  const hs = new Float64Array(landmarkCount);
  for (let i = 0; i < landmarkCount; i++) {
    const k = i * LANDMARK_STRIDE;
    const x = landmarks[k], y = landmarks[k + 1], z = landmarks[k + 2];
    hs[i] = Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z) ? up[0] * (x - origin[0]) + up[1] * (y - origin[1]) + up[2] * (z - origin[2]) : NaN;
  }

  let best: LocalFloorHit | null = null;
  let lastN = 0;
  let lastSpread = NaN;
  let reason: LocalFloorHit["fallback"] = "few_landmarks";
  for (const mult of [1, 2]) {
    let cur = base;
    let h = 0;
    let ok = false;
    let n = 0;
    let spread = NaN;
    for (let it = 0; it < 4; it++) {
      const depth = depthOf(cur.p);
      if (!(depth > 1e-9)) break;
      const radius = cfg.radiusDepthRatio * depth * mult;
      const window = cfg.heightWindowDepthRatio * depth;
      const heights: number[] = [];
      for (let i = 0; i < landmarkCount; i++) {
        const hi = hs[i];
        if (!(Math.abs(hi - h) <= window)) continue;
        const k = i * LANDMARK_STRIDE;
        const vx = landmarks[k] - cur.p[0], vy = landmarks[k + 1] - cur.p[1], vz = landmarks[k + 2] - cur.p[2];
        const vh = up[0] * vx + up[1] * vy + up[2] * vz;
        if (Math.hypot(vx - vh * up[0], vy - vh * up[1], vz - vh * up[2]) > radius) continue;
        heights.push(hi);
      }
      n = heights.length;
      if (n < cfg.minLandmarks) {
        ok = false;
        reason = "few_landmarks";
        break;
      }
      const med = median(heights);
      spread = median(heights.map((v) => Math.abs(v - med)));
      if (spread > cfg.maxSpreadDepthRatio * depth) {
        ok = false;
        reason = "spread";
        break;
      }
      const next = intersect(rayOrigin, rayDir, up, origin, med);
      if (!next) {
        ok = false;
        reason = "miss";
        break;
      }
      const moved = Math.abs(med - h);
      h = med;
      cur = next;
      ok = true;
      if (moved < 1e-3 * depth) break;
    }
    lastN = n;
    lastSpread = spread;
    if (ok) {
      best = { point: cur.p, t: cur.t, height: h, surface: "local", neighbors: n, spread, depth: depthOf(cur.p), fallback: "" };
      break;
    }
    // A neighbourhood that disagrees on the height does not get better by widening it.
    if (reason === "spread") break;
  }
  return best ?? plane(reason, lastN, lastSpread);
}
