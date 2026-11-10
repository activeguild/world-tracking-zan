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
/** What a placed object is (HUD / stats / tests); it does not change how it is tracked. */
export type ARObjectKind = "cube" | "model" | "video" | "object";

/**
 * The minimum of HTMLVideoElement a video panel needs (so it can be built
 * and unit-tested without a DOM). `videoWidth / videoHeight` are 0 until
 * the metadata has loaded; the panel listens for `loadedmetadata` then.
 */
export interface VideoLike {
  videoWidth: number;
  videoHeight: number;
  addEventListener(type: "loadedmetadata", listener: () => void, options?: { once?: boolean }): void;
}

/** Yaw (about world Y) that turns an object's +Z face toward a point, e.g. the camera. */
export function yawFacing(objectX: number, objectZ: number, targetX: number, targetZ: number): number {
  const dx = targetX - objectX;
  const dz = targetZ - objectZ;
  if (!Number.isFinite(dx) || !Number.isFinite(dz) || dx * dx + dz * dz < 1e-12) return 0;
  return Math.atan2(dx, dz);
}

export class ARObject {
  readonly root = new THREE.Group();
  private _placed = false;
  /** Media behind a video panel (paused on dispose); null for other objects. */
  private media: { pause(): void } | null = null;

  /** Own motion in world units per second (plane-local when y = 0). */
  readonly velocity = new THREE.Vector3();
  /** Own rotation about the world Y axis, radians per second. */
  angularVelocityY = 0;
  /** Visibility requested by the world (tracking trust); rendered only when also placed. */
  private shown = true;

  constructor(
    readonly id: number,
    readonly object: THREE.Object3D,
    readonly kind: ARObjectKind = "object",
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

  /** Uniform scale about the object's own origin (its footprint on the plane). 1 = as placed. */
  get scale(): number {
    return this.root.scale.x;
  }

  setScale(s: number): void {
    if (!Number.isFinite(s) || s <= 0) return;
    this.root.scale.setScalar(s);
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
    this.media?.pause();
    this.media = null;
    this.root.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (mesh.geometry) mesh.geometry.dispose();
      const mat = mesh.material as THREE.Material | THREE.Material[] | undefined;
      const mats = Array.isArray(mat) ? mat : mat ? [mat] : [];
      for (const m of mats) {
        (m as THREE.MeshBasicMaterial).map?.dispose();
        m.dispose();
      }
    });
  }

  /**
   * A video panel standing upright on the plane: a `width` m wide screen
   * (height from the video's aspect, 16:9 until the metadata is known)
   * showing `video` through a VideoTexture, with a thin dark back so it
   * reads as a screen from behind. Its +Z face shows the picture; the bottom
   * edge is at Y = 0. Playback (`play()`, muted / playsinline) is the
   * caller's job — it must start from a user gesture on phones.
   */
  static videoPanel(id: number, video: VideoLike, width: number, fallbackAspect = 16 / 9): ARObject {
    const holder = new THREE.Group();
    const texture = new THREE.VideoTexture(video as unknown as HTMLVideoElement);
    texture.colorSpace = THREE.SRGBColorSpace;
    const screen = new THREE.Mesh(
      new THREE.PlaneGeometry(1, 1),
      new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide, toneMapped: false }),
    );
    screen.name = "screen";
    const depth = Math.max(0.002, width * 0.015);
    const back = new THREE.Mesh(
      new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshStandardMaterial({ color: 0x202020, roughness: 0.8, metalness: 0.1 }),
    );
    back.name = "back";
    holder.add(back, screen);
    const layout = () => {
      const aspect = video.videoWidth > 0 && video.videoHeight > 0 ? video.videoWidth / video.videoHeight : fallbackAspect;
      const height = width / aspect;
      screen.scale.set(width, height, 1);
      screen.position.set(0, height / 2, depth / 2 + 1e-4);
      back.scale.set(width, height, depth);
      back.position.set(0, height / 2, 0);
    };
    layout();
    // The aspect is only known once the metadata has loaded; relayout then.
    video.addEventListener("loadedmetadata", layout, { once: true });
    const obj = new ARObject(id, holder, "video");
    obj.media = typeof (video as { pause?: unknown }).pause === "function" ? (video as unknown as { pause(): void }) : null;
    return obj;
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
    return new ARObject(id, holder, "model");
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
    return new ARObject(id, mesh, "cube");
  }
}
