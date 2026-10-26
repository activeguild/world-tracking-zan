import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { ARObject, yawFacing, type VideoLike } from "../../src/rendering/ARObject";
import { ARWorld } from "../../src/rendering/ARWorld";

/**
 * Sample video placement (ユーザー要望): a video panel is an ARObject like the
 * cube — it stands on the plane, is sized from the video's aspect, faces the
 * camera when placed and is manipulated with the same drag / twist / pinch.
 * Built here with a DOM-less stand-in for HTMLVideoElement.
 */
class FakeVideo implements VideoLike {
  videoWidth = 0;
  videoHeight = 0;
  paused = true;
  private listeners: (() => void)[] = [];
  addEventListener(_type: "loadedmetadata", listener: () => void): void {
    this.listeners.push(listener);
  }
  pause(): void {
    this.paused = true;
  }
  /** Simulate the metadata arriving (aspect known from now on). */
  loadMetadata(w: number, h: number): void {
    this.videoWidth = w;
    this.videoHeight = h;
    const l = this.listeners;
    this.listeners = [];
    for (const f of l) f();
  }
}

function bounds(obj: ARObject): { size: THREE.Vector3; min: THREE.Vector3 } {
  obj.root.updateWorldMatrix(true, true);
  const box = new THREE.Box3().setFromObject(obj.root);
  return { size: box.getSize(new THREE.Vector3()), min: box.min.clone() };
}

describe("video panel object", () => {
  it("stands on the plane with the requested width; 16:9 until the metadata says otherwise", () => {
    const video = new FakeVideo();
    const obj = ARObject.videoPanel(7, video, 0.3);
    expect(obj.kind).toBe("video");
    expect(obj.id).toBe(7);
    obj.place([0, 0, 0]);
    let b = bounds(obj);
    expect(b.size.x).toBeCloseTo(0.3, 9);
    expect(b.size.y).toBeCloseTo(0.3 * 9 / 16, 9);
    expect(b.min.y).toBeCloseTo(0, 9); // bottom edge on the floor
    expect(b.size.z).toBeLessThan(0.02); // a thin screen
    // A 4:3 clip: the panel keeps its width and grows in height.
    video.loadMetadata(320, 240);
    b = bounds(obj);
    expect(b.size.x).toBeCloseTo(0.3, 9);
    expect(b.size.y).toBeCloseTo(0.3 * 3 / 4, 9);
    expect(b.min.y).toBeCloseTo(0, 9);
    // The screen is a VideoTexture on the +Z face.
    const screen = obj.root.getObjectByName("screen") as THREE.Mesh;
    expect(screen).toBeDefined();
    expect((screen.material as THREE.MeshBasicMaterial).map).toBeInstanceOf(THREE.VideoTexture);
  });

  it("is a regular placed object: same manipulation API, the ray picks its screen, dispose pauses the video", () => {
    const video = new FakeVideo();
    video.loadMetadata(320, 180);
    const world = new ARWorld();
    const obj = world.addVideo(video, 0.3);
    obj.place([0.1, 0, -0.5], 0.3);
    expect(world.placedCount).toBe(1);
    expect(obj.yaw).toBeCloseTo(0.3, 12);
    obj.setYaw(obj.yaw + 0.5);
    expect(obj.yaw).toBeCloseTo(0.8, 12);
    obj.setYaw(0);
    obj.setScale(2);
    expect(obj.scale).toBe(2);
    expect(bounds(obj).size.x).toBeCloseTo(0.6, 9); // pinch scales about the footprint
    expect(bounds(obj).min.y).toBeCloseTo(0, 9);
    expect(obj.position.toArray()).toEqual([0.1, 0, -0.5]);
    // A ray from above the floor toward the panel centre hits it (what pickObject uses).
    obj.setScale(1);
    obj.root.updateWorldMatrix(true, true);
    const ray = new THREE.Raycaster(new THREE.Vector3(0.1, 0.3 * 9 / 32, 1), new THREE.Vector3(0, 0, -1));
    expect(ray.intersectObject(obj.root, true).length).toBeGreaterThan(0);
    // ... and from behind too (the back is a box, the screen double-sided).
    const back = new THREE.Raycaster(new THREE.Vector3(0.1, 0.3 * 9 / 32, -2), new THREE.Vector3(0, 0, 1));
    expect(back.intersectObject(obj.root, true).length).toBeGreaterThan(0);
    video.paused = false;
    world.remove(obj);
    expect(video.paused).toBe(true);
    expect(world.placedCount).toBe(0);
  });

  it("yawFacing turns the +Z face toward the camera", () => {
    const face = (yaw: number) => new THREE.Vector3(0, 0, 1).applyAxisAngle(new THREE.Vector3(0, 1, 0), yaw);
    expect(yawFacing(0, 0, 0, 1)).toBeCloseTo(0, 12); // camera straight in front (+Z)
    expect(yawFacing(0, 0, 1, 0)).toBeCloseTo(Math.PI / 2, 12); // camera to the +X side
    expect(yawFacing(0, 0, 0, -1)).toBeCloseTo(Math.PI, 12);
    for (const [cx, cz] of [[0.3, 0.4], [-1, 0.2], [0.5, -2], [-0.7, -0.1]]) {
      const yaw = yawFacing(0.1, -0.2, cx, cz);
      const f = face(yaw);
      const toCam = new THREE.Vector3(cx - 0.1, 0, cz + 0.2).normalize();
      expect(f.dot(toCam)).toBeCloseTo(1, 9);
    }
    expect(yawFacing(1, 2, 1, 2)).toBe(0); // camera right above: no preference
    expect(yawFacing(0, 0, NaN, 1)).toBe(0);
  });
});
