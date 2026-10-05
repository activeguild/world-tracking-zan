import * as THREE from "three";

/**
 * A placed AR object (spec §29–§30, 修正指示書 §11–§15, §29).
 *
 * Its transform is a *world* pose (`objectWorldPose`): it changes only by
 * user placement (`place` / `setPosition`), by its own motion (`moveBy`,
 * `velocity`, `angularVelocityY`, `update`) — never because the camera
 * moved. The camera pose lives in ARCamera and is never read here.
 *
 * The world frame is the plane frame (plane = Y 0, origin = plane center),
 * so "plane-local" coordinates are simply world (x, z).
 */
export class ARObject {
  readonly root = new THREE.Group();
  private _placed = false;

  /** Own motion in world units per second (plane-local when y = 0). */
  readonly velocity = new THREE.Vector3();
  /** Own rotation about the world Y axis, radians per second. */
  angularVelocityY = 0;
  /** Visibility requested by the world (tracking trust); rendered only when also placed. */
  private shown = true;

  constructor(
    readonly id: number,
    readonly object: THREE.Object3D,
  ) {
    this.root.add(object);
    this.root.visible = false;
  }

  get placed(): boolean {
    return this._placed;
  }

  /** Current world position (copy). */
  get position(): THREE.Vector3 {
    return this.root.position.clone();
  }

  /** Current yaw about world Y (radians). */
  get yaw(): number {
    return this.root.rotation.y;
  }

  /**
   * Put the object on the plane at a world position (Y = 0), optionally
   * facing `yawRad`. Visibility still follows the tracking trust (v13): an
   * object placed while the objects are hidden stays hidden until shown.
   */
  place(position: ArrayLike<number>, yawRad = 0): void {
    this.root.position.set(position[0], position[1], position[2]);
    this.root.rotation.set(0, yawRad, 0);
    this._placed = true;
    this.root.visible = this.shown;
  }

  /** Set the world position directly (object motion, not camera motion). */
  setPosition(x: number, y: number, z: number): void {
    this.root.position.set(x, y, z);
  }

  /** Translate in world coordinates. */
  moveBy(dx: number, dy: number, dz: number): void {
    this.root.position.x += dx;
    this.root.position.y += dy;
    this.root.position.z += dz;
  }

  setYaw(yawRad: number): void {
    this.root.rotation.y = yawRad;
  }

  /**
   * Advance the object's own animation by `dtSec`. Pure world-space
   * integration; the camera is not an input.
   */
  update(dtSec: number): void {
    if (!this._placed || dtSec <= 0) return;
    if (this.velocity.lengthSq() > 0) this.root.position.addScaledVector(this.velocity, dtSec);
    if (this.angularVelocityY !== 0) this.root.rotation.y += this.angularVelocityY * dtSec;
  }

  /** Show / hide (tracking trust, v13). The transform is untouched; an unplaced object is never rendered. */
  setVisible(v: boolean): void {
    this.shown = v;
    this.root.visible = v && this._placed;
  }

  dispose(): void {
    this.root.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (mesh.geometry) mesh.geometry.dispose();
      const mat = mesh.material as THREE.Material | THREE.Material[] | undefined;
      if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
      else mat?.dispose();
    });
  }

  /**
   * Wrap a loaded model (e.g. `gltf.scene`): uniformly scaled so that its
   * largest horizontal extent equals `targetSize` meters (when > 0), centred
   * on X/Z and standing on the plane (min Y = 0).
   */
  static fromModel(id: number, model: THREE.Object3D, targetSize = 0): ARObject {
    const box = new THREE.Box3().setFromObject(model);
    const size = new THREE.Vector3();
    box.getSize(size);
    const holder = new THREE.Group();
    let scale = 1;
    if (targetSize > 0) {
      const extent = Math.max(size.x, size.z, 1e-6);
      scale = targetSize / extent;
    }
    model.scale.multiplyScalar(scale);
    const center = new THREE.Vector3();
    box.getCenter(center);
    model.position.set(-center.x * scale, -box.min.y * scale, -center.z * scale);
    holder.add(model);
    return new ARObject(id, holder);
  }

  /** Demo cube standing on the plane (its bottom face at Y = 0). */
  static cube(id: number, size: number, color = 0x3fa9f5): ARObject {
    const geometry = new THREE.BoxGeometry(size, size, size);
    const material = new THREE.MeshStandardMaterial({ color, roughness: 0.5, metalness: 0.1 });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.y = size / 2;
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(geometry),
      new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.6 }),
    );
    mesh.add(edges);
    return new ARObject(id, mesh);
  }
}
