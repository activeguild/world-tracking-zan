import { worldToMap, type WorldFrame } from "../math/CoordinateSystem";
import { fitPlaneLeastSquares } from "../math/Plane";
import { LANDMARK_STRIDE } from "../vision/types";

/**
 * Where does the placed object sit relative to the floor the map actually
 * knows around it? (Android recording 43: "the cube's size changes
 * differently from the floor texture when I move closer / farther".)
 *
 * The apparent size of the cube changes as 1 / Z_cube and the floor texture
 * under it as 1 / Z_floor. They can only disagree when the cube is not at
 * the floor's depth — the world plane (Y = 0, fixed once from the first
 * found plane) passes above or below the real floor at the object's spot,
 * or is tilted against it. A point floating h above the floor also slides
 * against the texture by h / tan(elevation) when the camera tilts and comes
 * back when the view is restored, which is the long-standing "傾けるとずれる、
 * 戻すと戻る". The PnP residual says nothing about this: the pose can be
 * perfect while the placement surface is wrong.
 *
 * Per anchor point (the world origin and every placed object) this takes
 * the landmarks within a horizontal radius (a fraction of the object's
 * depth, map units) and not far above / below the plane (shelves and walls
 * are out), and reports their height above the world plane — median, young
 * / old generations separately, as a fraction of the object's depth — the
 * tilt of the local least-squares plane against the world up, and the
 * depth of the local surface along the camera → object ray relative to the
 * object's depth (`localDepthRatio`: 1 = the object is on the surface the
 * map knows there, > 1 = the floor is behind the object = it floats).
 * Diagnostics only: nothing in the engine, the anchor or the objects is
 * touched. A few vector operations per landmark per object.
 */
export interface ObjectFloorConfig {
  /** Horizontal neighbourhood radius = ratio × the object's depth from the camera (map units). */
  radiusDepthRatio: number;
  /** Landmarks farther above / below the world plane than ratio × depth are not floor (shelves, walls). */
  heightWindowDepthRatio: number;
  /** Fewer neighbours than this: the record carries counts only (NaN statistics). */
  minLandmarks: number;
  /** Landmark age (frames since triangulation) below which it counts as "young". */
  youngFrames: number;
}

export interface ObjectFloorRecord {
  /** 0 = the world origin, otherwise the placed object's id. */
  id: number;
  /** Object depth from the camera (map units; NaN when behind the camera). */
  depth: number;
  /** Object depth in meters (world scale). */
  depthMeters: number;
  neighbors: number;
  /** Neighbours the per-frame plane detector flagged as plane inliers. */
  planeFlagged: number;
  young: number;
  old: number;
  /** Median height of the neighbours above the world plane (map units; + = above). */
  heightMedian: number;
  /** heightMedian / depth (the size-mismatch scale: 0.03 = the floor is 3 % of the depth away from the plane). */
  heightRatio: number;
  /** Median absolute deviation of the heights / depth (how flat the neighbourhood is). */
  heightSpread: number;
  /** Height ratios of the young / old neighbours (NaN when either group has < 3). */
  youngHeightRatio: number;
  oldHeightRatio: number;
  /** Local least-squares plane normal vs the world up (deg; NaN without a fit). */
  tiltDeg: number;
  /** Depth of the local plane along the camera → object ray over the object's depth (NaN without a fit / parallel). */
  localDepthRatio: number;
}

const MIN_GROUP = 3;

function median(values: number[]): number {
  if (values.length === 0) return NaN;
  const s = values.slice().sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : 0.5 * (s[m - 1] + s[m]);
}

/**
 * @param landmarks packed `[x, y, z, planeInlier, age]` (map frame, `LANDMARK_STRIDE`)
 * @param rotation  camera rotation (row-major, X_cam = R X_map + t)
 * @param cameraCenter camera centre in the map frame
 * @param objects  world positions (meters) with ids; the world origin is added as id 0
 */
export function objectFloorDiagnostics(
  landmarks: Float32Array,
  landmarkCount: number,
  world: WorldFrame,
  rotation: ArrayLike<number>,
  cameraCenter: ArrayLike<number>,
  objects: readonly { id: number; position: ArrayLike<number> }[],
  cfg: ObjectFloorConfig,
): ObjectFloorRecord[] {
  const up = world.plane.up;
  const origin = world.plane.origin;
  const r = rotation;
  const C = cameraCenter;
  const out: ObjectFloorRecord[] = [];
  const targets: { id: number; map: Float64Array }[] = [{ id: 0, map: Float64Array.from(origin) }];
  for (const o of objects) targets.push({ id: o.id, map: worldToMap(world, o.position) });
  const pts = new Float64Array(landmarkCount * 3);
  for (const tg of targets) {
    const o = tg.map;
    const dx = o[0] - C[0], dy = o[1] - C[1], dz = o[2] - C[2];
    const depth = r[6] * dx + r[7] * dy + r[8] * dz;
    const rec: ObjectFloorRecord = {
      id: tg.id,
      depth: depth > 1e-9 ? depth : NaN,
      depthMeters: depth > 1e-9 ? depth * world.scale : NaN,
      neighbors: 0,
      planeFlagged: 0,
      young: 0,
      old: 0,
      heightMedian: NaN,
      heightRatio: NaN,
      heightSpread: NaN,
      youngHeightRatio: NaN,
      oldHeightRatio: NaN,
      tiltDeg: NaN,
      localDepthRatio: NaN,
    };
    out.push(rec);
    if (!(depth > 1e-9)) continue;
    const radius = cfg.radiusDepthRatio * depth;
    const window = cfg.heightWindowDepthRatio * depth;
    // Height of the object itself above the plane (0 for a hit-test placement; the
    // neighbours are measured relative to the plane, not to the object).
    const heights: number[] = [];
    const young: number[] = [];
    const old: number[] = [];
    let n = 0;
    for (let i = 0; i < landmarkCount; i++) {
      const k = i * LANDMARK_STRIDE;
      const x = landmarks[k], y = landmarks[k + 1], z = landmarks[k + 2];
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
      const ex = x - origin[0], ey = y - origin[1], ez = z - origin[2];
      const h = up[0] * ex + up[1] * ey + up[2] * ez;
      if (Math.abs(h) > window) continue;
      // Horizontal distance to the object: remove the up component of (X − o).
      const vx = x - o[0], vy = y - o[1], vz = z - o[2];
      const vh = up[0] * vx + up[1] * vy + up[2] * vz;
      const hx = vx - vh * up[0], hy = vy - vh * up[1], hz = vz - vh * up[2];
      if (Math.hypot(hx, hy, hz) > radius) continue;
      pts[n * 3] = x;
      pts[n * 3 + 1] = y;
      pts[n * 3 + 2] = z;
      n++;
      heights.push(h);
      if (landmarks[k + 3] > 0.5) rec.planeFlagged++;
      const age = landmarks[k + 4];
      if (age < cfg.youngFrames) {
        rec.young++;
        young.push(h);
      } else {
        rec.old++;
        old.push(h);
      }
    }
    rec.neighbors = n;
    if (n < cfg.minLandmarks) continue;
    const med = median(heights);
    rec.heightMedian = med;
    rec.heightRatio = med / depth;
    rec.heightSpread = median(heights.map((h) => Math.abs(h - med))) / depth;
    if (young.length >= MIN_GROUP) rec.youngHeightRatio = median(young) / depth;
    if (old.length >= MIN_GROUP) rec.oldHeightRatio = median(old) / depth;
    const plane = fitPlaneLeastSquares(pts, null, n);
    if (!plane) continue;
    const nn = plane.normal;
    let cosTilt = nn[0] * up[0] + nn[1] * up[1] + nn[2] * up[2];
    if (cosTilt < 0) {
      cosTilt = -cosTilt;
      nn[0] = -nn[0];
      nn[1] = -nn[1];
      nn[2] = -nn[2];
      plane.d = -plane.d;
    }
    rec.tiltDeg = (Math.acos(Math.min(1, cosTilt)) * 180) / Math.PI;
    // Camera → object ray: X = C + t (o − C); the local plane at t_local.
    const denom = nn[0] * dx + nn[1] * dy + nn[2] * dz;
    if (Math.abs(denom) > 1e-12) {
      const t = -(nn[0] * C[0] + nn[1] * C[1] + nn[2] * C[2] + plane.d) / denom;
      rec.localDepthRatio = t > 0 ? t : NaN;
    }
  }
  return out;
}
