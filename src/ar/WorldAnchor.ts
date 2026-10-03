import type { CameraIntrinsics } from "../camera/CameraIntrinsics";
import {
  cameraPoseToThree,
  mapDirToWorld,
  mapToWorld,
  worldFromPlane,
  type ThreeCameraPose,
  type WorldFrame,
} from "../math/CoordinateSystem";
import type { RigidTransform } from "../math/Pose";
import { intersectRayPlane, pixelRay, transformRay } from "../math/Ray";
import { transpose3 } from "../math/Decomposition";
import type { MapPoseOutput, PlaneOutput } from "../vision/types";

/**
 * World anchor (spec §25–§28): fixes the world frame on the first found
 * plane and answers hit tests against that plane.
 *
 * The world frame is defined once from the plane estimate at the time it
 * was found; later plane re-estimates do not move the world (that would
 * make placed objects jitter). The world only becomes invalid when the
 * landmark map is reset (new mapFrameId).
 */
export interface HitResult {
  /** World position on the plane (Three.js frame, meters). */
  position: Float64Array;
  /** World plane normal (= +Y). */
  normal: Float64Array;
  /** Distance from the camera along the ray (meters). */
  distance: number;
}

export interface WorldAnchorConfig {
  /**
   * Assumed distance (m) from the camera to the plane when the world is
   * created; fixes the monocular scale (spec §18). 0.5 m suits a desk,
   * ~1.3 m a floor.
   */
  assumedPlaneDistanceMeters: number;
}

export class WorldAnchor {
  private world: WorldFrame | null = null;
  private mapFrameId = -1;

  constructor(private readonly config: WorldAnchorConfig) {}

  get isReady(): boolean {
    return this.world !== null;
  }

  get frame(): WorldFrame | null {
    return this.world;
  }

  /** Frame id of the landmark map the world is attached to. */
  get attachedMapFrameId(): number {
    return this.mapFrameId;
  }

  reset(): void {
    this.world = null;
    this.mapFrameId = -1;
  }

  /**
   * Create the world from a found plane and the current map-frame camera
   * pose. No-op when already created for this map.
   */
  create(plane: PlaneOutput, mapPose: MapPoseOutput): boolean {
    if (this.world && this.mapFrameId === mapPose.mapFrameId) return false;
    const cam = toRigid(mapPose);
    // Camera center in map and its distance to the plane (map units).
    const r = cam.rotation;
    const t = cam.translation;
    const c = [
      -(r[0] * t[0] + r[3] * t[1] + r[6] * t[2]),
      -(r[1] * t[0] + r[4] * t[1] + r[7] * t[2]),
      -(r[2] * t[0] + r[5] * t[1] + r[8] * t[2]),
    ];
    const n = plane.normal;
    const distMap = Math.abs(n[0] * c[0] + n[1] * c[1] + n[2] * c[2] + plane.d);
    const scale = distMap > 1e-9 ? this.config.assumedPlaneDistanceMeters / distMap : 1;
    this.world = worldFromPlane(plane.normal, plane.center, cam, scale);
    this.mapFrameId = mapPose.mapFrameId;
    return true;
  }

  /** Drop the world when the map it was attached to is gone. */
  checkMap(mapPose: MapPoseOutput | null): boolean {
    if (!this.world) return false;
    if (!mapPose || mapPose.mapFrameId !== this.mapFrameId) {
      this.reset();
      return true;
    }
    return false;
  }

  /** Three.js camera pose for a map-frame camera pose. */
  cameraPose(mapPose: MapPoseOutput): ThreeCameraPose | null {
    if (!this.world) return null;
    return cameraPoseToThree(this.world, toRigid(mapPose));
  }

  /**
   * Hit test: pixel in processing-image coordinates → world point on the
   * plane Y = 0.
   */
  hitTest(u: number, v: number, k: CameraIntrinsics, mapPose: MapPoseOutput): HitResult | null {
    if (!this.world) return null;
    const cam = toRigid(mapPose);
    // Ray in camera frame → map frame (X_map = Rᵀ (X_cam − t)).
    const rayCam = pixelRay(u, v, k.fx, k.fy, k.cx, k.cy);
    const rt = transpose3(cam.rotation);
    const t = cam.translation;
    const cMap = [
      -(rt[0] * t[0] + rt[1] * t[1] + rt[2] * t[2]),
      -(rt[3] * t[0] + rt[4] * t[1] + rt[5] * t[2]),
      -(rt[6] * t[0] + rt[7] * t[1] + rt[8] * t[2]),
    ];
    const rayMap = transformRay(rayCam, rt, cMap);
    // Plane in the map frame: up · (X − origin) = 0  →  n = up, d = −up·origin
    const up = this.world.plane.up;
    const o = this.world.plane.origin;
    const d = -(up[0] * o[0] + up[1] * o[1] + up[2] * o[2]);
    const hit = intersectRayPlane(rayMap, up, d);
    if (!hit) return null;
    const position = mapToWorld(this.world, hit.point);
    position[1] = 0; // exactly on the plane
    const normal = mapDirToWorld(this.world, up);
    return { position, normal, distance: hit.t * this.world.scale };
  }

  /** Map point → world point (for debug rendering of landmarks). */
  toWorld(p: ArrayLike<number>, out?: Float64Array): Float64Array | null {
    if (!this.world) return null;
    return mapToWorld(this.world, p, out);
  }
}

function toRigid(mp: MapPoseOutput): RigidTransform {
  return { rotation: Float64Array.from(mp.rotation), translation: Float64Array.from(mp.translation) };
}
