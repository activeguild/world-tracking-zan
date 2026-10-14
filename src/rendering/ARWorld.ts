import * as THREE from "three";
import { ARObject } from "./ARObject";

/**
 * World root of the AR scene (spec §29, §33, §44; 修正指示書 v13).
 *
 * Holds the placed objects and the plane visualization at Y = 0. Objects
 * are fixed in world space. Their *visibility* follows the trust in the
 * current camera pose (`ObjectVisibility.ts`): hidden while tracking is
 * lost, relocalizing or confirming a relocalization, shown again once world
 * tracking is back on a fresh map pose. Hiding never touches the objects'
 * transforms, the world root or the plane grid data.
 */
export class ARWorld {
  readonly root = new THREE.Group();
  readonly objects: ARObject[] = [];
  private readonly planeGroup = new THREE.Group();
  private nextId = 1;
  private lostSince: number | null = null;
  private hidden = false;
  private planeGridWanted = false;

  constructor() {
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

  /** Toggle the plane grid without rebuilding it (debug HUD on/off, v7 §17). */
  setPlaneGridVisible(visible: boolean): void {
    this.planeGridWanted = visible;
    this.planeGroup.visible = visible && !this.hidden && this.planeGroup.children.length > 0;
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
    this.planeGridWanted = true;
    this.planeGroup.visible = !this.hidden;
  }

  createCube(size: number): ARObject {
    return this.register(ARObject.cube(this.nextId++, size));
  }

  add(object3d: THREE.Object3D): ARObject {
    return this.register(new ARObject(this.nextId++, object3d));
  }

  /** Add a loaded model normalized to `targetSize` meters (see ARObject.fromModel). */
  addModel(model: THREE.Object3D, targetSize: number): ARObject {
    return this.register(ARObject.fromModel(this.nextId++, model, targetSize));
  }

  private register(obj: ARObject): ARObject {
    this.objects.push(obj);
    this.root.add(obj.root);
    // An object placed while the pose is not trusted starts hidden too.
    if (this.hidden) obj.setVisible(false);
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
   * Advance the objects' own animations (修正指示書 §14, §29). Called once per
   * rendered frame with the elapsed seconds; the camera pose is not involved.
   */
  update(dtSec: number): void {
    for (const o of this.objects) o.update(dtSec);
  }

  /**
   * Plane-local → world (修正指示書 §15). The world frame *is* the plane frame
   * (origin = plane center, plane = Y 0), so the transform is the identity:
   * plane-local (x, z) is world (x, 0, z).
   */
  planeToWorld(x: number, z: number): [number, number, number] {
    return [x, 0, z];
  }

  /**
   * Show or hide the placed objects (and the plane grid) as one, idempotently
   * (v13 §24): only a change touches the Three.js objects. Transforms are
   * never altered (§12); the world root stays attached and the render loop
   * keeps running (§20).
   */
  setObjectsVisible(visible: boolean): void {
    if (visible === !this.hidden) return;
    this.hidden = !visible;
    for (const o of this.objects) o.setVisible(visible);
    this.planeGroup.visible = visible && this.planeGridWanted && this.planeGroup.children.length > 0;
  }

  /** Objects are currently hidden because the camera pose is not trusted. */
  get isHidden(): boolean {
    return this.hidden;
  }

  /** Record whether the camera is tracked this frame (for `lostDurationMs`). */
  markTracking(tracking: boolean, nowMs: number): void {
    if (tracking) {
      this.lostSince = null;
    } else if (this.lostSince === null) {
      this.lostSince = nowMs;
    }
  }

  /** How long tracking has been lost (ms), 0 while tracking. */
  lostDurationMs(nowMs: number): number {
    return this.lostSince === null ? 0 : nowMs - this.lostSince;
  }
}
