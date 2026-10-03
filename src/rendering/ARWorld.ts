import * as THREE from "three";
import { ARObject } from "./ARObject";

/**
 * World root of the AR scene (spec §29, §33, §44).
 *
 * Holds the placed objects and the plane visualization at Y = 0. Objects
 * are fixed in world space; when tracking is lost they stay where they are
 * for a grace period and are then hidden until tracking returns.
 */
export class ARWorld {
  readonly root = new THREE.Group();
  readonly objects: ARObject[] = [];
  private readonly planeGroup = new THREE.Group();
  private nextId = 1;
  private lostSince: number | null = null;
  private hidden = false;

  constructor(private readonly holdPoseOnLostMs: number) {
    this.root.name = "ARWorld";
    this.root.add(this.planeGroup);
    this.planeGroup.visible = false;
    this.root.visible = false;
  }

  /** Attach to a scene (and add neutral lighting if the scene has none). */
  attach(scene: THREE.Scene, addLights = true): void {
    scene.add(this.root);
    if (addLights) {
      const hemi = new THREE.HemisphereLight(0xffffff, 0x666666, 1.0);
      const dir = new THREE.DirectionalLight(0xffffff, 1.2);
      dir.position.set(0.5, 1.5, 0.8);
      this.root.add(hemi, dir);
    }
  }

  /** Called when the world frame becomes available. */
  setWorldReady(ready: boolean): void {
    this.root.visible = ready;
    if (!ready) {
      for (const o of this.objects) this.remove(o);
      this.lostSince = null;
      this.hidden = false;
    }
  }

  /** Transparent grid on the plane (spec §44). `extent` in meters. */
  showPlaneGrid(extent: number, visible = true): void {
    this.planeGroup.clear();
    if (!visible) {
      this.planeGroup.visible = false;
      return;
    }
    const divisions = Math.max(4, Math.round(extent / 0.1));
    const grid = new THREE.GridHelper(extent, divisions, 0x66ffcc, 0x66ffcc);
    const mat = grid.material as THREE.Material;
    mat.transparent = true;
    mat.opacity = 0.35;
    mat.depthWrite = false;
    this.planeGroup.add(grid);
    const fill = new THREE.Mesh(
      new THREE.PlaneGeometry(extent, extent),
      new THREE.MeshBasicMaterial({ color: 0x33ccaa, transparent: true, opacity: 0.08, side: THREE.DoubleSide, depthWrite: false }),
    );
    fill.rotation.x = -Math.PI / 2;
    this.planeGroup.add(fill);
    this.planeGroup.visible = true;
  }

  createCube(size: number): ARObject {
    const obj = ARObject.cube(this.nextId++, size);
    this.objects.push(obj);
    this.root.add(obj.root);
    return obj;
  }

  add(object3d: THREE.Object3D): ARObject {
    const obj = new ARObject(this.nextId++, object3d);
    this.objects.push(obj);
    this.root.add(obj.root);
    return obj;
  }

  /** Add a loaded model normalized to `targetSize` meters (see ARObject.fromModel). */
  addModel(model: THREE.Object3D, targetSize: number): ARObject {
    const obj = ARObject.fromModel(this.nextId++, model, targetSize);
    this.objects.push(obj);
    this.root.add(obj.root);
    return obj;
  }

  remove(obj: ARObject): void {
    const i = this.objects.indexOf(obj);
    if (i >= 0) this.objects.splice(i, 1);
    this.root.remove(obj.root);
    obj.dispose();
  }

  get placedCount(): number {
    return this.objects.filter((o) => o.placed).length;
  }

  /**
   * Tracking status update (spec §33): keep objects at their last pose for
   * `holdPoseOnLostMs`, then hide them until tracking resumes.
   */
  updateTracking(tracking: boolean, nowMs: number): void {
    if (tracking) {
      this.lostSince = null;
      if (this.hidden) {
        this.hidden = false;
        for (const o of this.objects) o.setVisible(true);
        this.planeGroup.visible = this.planeGroup.children.length > 0;
      }
      return;
    }
    if (this.lostSince === null) this.lostSince = nowMs;
    if (!this.hidden && nowMs - this.lostSince > this.holdPoseOnLostMs) {
      this.hidden = true;
      for (const o of this.objects) o.setVisible(false);
      this.planeGroup.visible = false;
    }
  }

  get isHidden(): boolean {
    return this.hidden;
  }
}
