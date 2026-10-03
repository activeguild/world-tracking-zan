import * as THREE from "three";

/**
 * A placed AR object (spec §29–§30). Lives at a fixed world position; it is
 * never moved to follow the camera — only the camera moves.
 */
export class ARObject {
  readonly root = new THREE.Group();
  private _placed = false;

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

  /** Put the object on the plane at a world position (Y = 0), optionally facing `yawRad`. */
  place(position: ArrayLike<number>, yawRad = 0): void {
    this.root.position.set(position[0], position[1], position[2]);
    this.root.rotation.set(0, yawRad, 0);
    this.root.visible = true;
    this._placed = true;
  }

  setVisible(v: boolean): void {
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
