import * as THREE from "three";
import type { CameraIntrinsics } from "../camera/CameraIntrinsics";
import { projectionMatrixFromIntrinsics, viewportIntrinsics, type ThreeCameraPose } from "../math/CoordinateSystem";
import { OneEuroVector, QuaternionSmoother, type OneEuroConfig } from "../math/OneEuroFilter";

/**
 * Three.js camera driven by the vision pose (spec §29, §34).
 *
 * - projection from the camera intrinsics, matching the `object-fit: cover`
 *   video underneath (so rendered objects sit on the pixels they belong to)
 * - position / rotation smoothing with One Euro filters (low latency under
 *   motion, no jitter at rest)
 */
export class ARCamera {
  readonly camera: THREE.PerspectiveCamera;
  private readonly posFilter: OneEuroVector;
  private readonly rotFilter: QuaternionSmoother;
  private smoothing = true;

  constructor(
    private readonly near: number,
    private readonly far: number,
    positionSmoothing: OneEuroConfig,
    rotationSmoothing: OneEuroConfig,
    existing?: THREE.PerspectiveCamera,
  ) {
    this.camera = existing ?? new THREE.PerspectiveCamera(60, 1, near, far);
    this.camera.near = near;
    this.camera.far = far;
    this.camera.matrixAutoUpdate = true;
    this.posFilter = new OneEuroVector(3, positionSmoothing);
    this.rotFilter = new QuaternionSmoother(rotationSmoothing);
  }

  setSmoothing(enabled: boolean): void {
    this.smoothing = enabled;
    if (!enabled) this.resetSmoothing();
  }

  resetSmoothing(): void {
    this.posFilter.reset();
    this.rotFilter.reset();
  }

  /** Update the projection to match intrinsics `k` shown in a viewport of the given CSS size. */
  updateProjection(k: CameraIntrinsics, viewportWidth: number, viewportHeight: number): void {
    const v = viewportIntrinsics(k, viewportWidth, viewportHeight);
    const m = projectionMatrixFromIntrinsics(v.fx, v.fy, v.cx, v.cy, viewportWidth, viewportHeight, this.near, this.far);
    this.camera.projectionMatrix.fromArray(Array.from(m));
    this.camera.projectionMatrixInverse.copy(this.camera.projectionMatrix).invert();
    // Keep the convenience fields roughly meaningful.
    this.camera.aspect = viewportWidth / viewportHeight;
    this.camera.fov = (2 * Math.atan(viewportHeight / (2 * v.fy)) * 180) / Math.PI;
  }

  /** Apply a world pose (optionally smoothed). `timeSec` drives the filters. */
  setPose(pose: ThreeCameraPose, timeSec: number): void {
    let p: ArrayLike<number> = pose.position;
    let q: ArrayLike<number> = pose.quaternion;
    if (this.smoothing) {
      p = this.posFilter.filter(pose.position, timeSec);
      q = this.rotFilter.filter(pose.quaternion, timeSec);
    }
    this.camera.position.set(p[0], p[1], p[2]);
    this.camera.quaternion.set(q[0], q[1], q[2], q[3]);
    this.camera.updateMatrixWorld(true);
  }
}
