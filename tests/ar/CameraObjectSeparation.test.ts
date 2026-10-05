import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { ARCamera } from "../../src/rendering/ARCamera";
import { ARWorld } from "../../src/rendering/ARWorld";
import { TEST_K } from "../helpers/scene";

/**
 * 修正指示書 §24 (coordinate tests) and §32 Test D / E (object motion):
 * the camera pose and the object poses are independent. Moving or rotating
 * the camera changes only where objects appear on screen, never their world
 * position; objects move only through their own motion API.
 */
const NO_SMOOTHING = { minCutoff: 1000, beta: 0, dCutoff: 1 };

function setup() {
  const scene = new THREE.Scene();
  const world = new ARWorld();
  world.attach(scene, false);
  world.setWorldReady(true);
  const cam = new ARCamera(0.01, 50, NO_SMOOTHING, NO_SMOOTHING);
  cam.setSmoothing(false);
  cam.updateProjection(TEST_K, TEST_K.width, TEST_K.height);
  return { scene, world, cam };
}

function quatYaw(rad: number): Float64Array {
  const q = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), rad);
  return new Float64Array([q.x, q.y, q.z, q.w]);
}

function ndcOf(obj: THREE.Object3D, camera: THREE.Camera): THREE.Vector3 {
  camera.updateMatrixWorld(true);
  return obj.getWorldPosition(new THREE.Vector3()).project(camera);
}

describe("camera pose and object pose are independent (修正指示書 §24)", () => {
  it("Test 1: camera at the origin sees an object at (0, 0, −1) straight ahead", () => {
    const { world, cam } = setup();
    const obj = world.createCube(0.1);
    obj.place([0, 0, -1]);
    cam.setPose({ position: new Float64Array([0, 0, 0]), quaternion: quatYaw(0) }, 0);
    const ndc = ndcOf(obj.root, cam.camera);
    expect(Math.abs(ndc.x)).toBeLessThan(1e-9);
    expect(Math.abs(ndc.y)).toBeLessThan(1e-9);
  });

  it("Test 2 / 3: moving the camera right / left leaves the object's world position unchanged and shifts it on screen the other way", () => {
    const { world, cam } = setup();
    const obj = world.createCube(0.1);
    obj.place([0, 0, -1]);
    const before = obj.position;

    cam.setPose({ position: new Float64Array([0.1, 0, 0]), quaternion: quatYaw(0) }, 0.1);
    expect(obj.position.equals(before)).toBe(true);
    const right = ndcOf(obj.root, cam.camera);
    expect(right.x).toBeLessThan(-0.05); // appears to the left

    cam.setPose({ position: new Float64Array([-0.1, 0, 0]), quaternion: quatYaw(0) }, 0.2);
    expect(obj.position.equals(before)).toBe(true);
    const left = ndcOf(obj.root, cam.camera);
    expect(left.x).toBeGreaterThan(0.05); // appears to the right
    expect(left.x).toBeCloseTo(-right.x, 9);

    // The amount matches the pinhole model: Δx_ndc = 2 · fx · (0.1 / 1) / width.
    expect(-right.x).toBeCloseTo((2 * TEST_K.fx * 0.1) / TEST_K.width, 6);
  });

  it("Test 4: rotating the camera leaves the object's world position unchanged", () => {
    const { world, cam } = setup();
    const obj = world.createCube(0.1);
    obj.place([0, 0, -1]);
    const before = obj.position;
    cam.setPose({ position: new Float64Array([0, 0, 0]), quaternion: quatYaw(0.1) }, 0);
    expect(obj.position.equals(before)).toBe(true);
    const ndc = ndcOf(obj.root, cam.camera);
    // Turning the camera left (+yaw about +Y) moves the object to the right on screen.
    expect(ndc.x).toBeGreaterThan(0.05);
    expect(Math.abs(ndc.y)).toBeLessThan(1e-9);
  });

  it("the Three.js camera matrices are consistent (matrixWorldInverse · matrixWorld = I)", () => {
    const { cam } = setup();
    cam.setPose({ position: new Float64Array([0.3, 0.5, 0.2]), quaternion: quatYaw(0.7) }, 0);
    const m = new THREE.Matrix4().multiplyMatrices(cam.camera.matrixWorldInverse, cam.camera.matrixWorld);
    const e = m.elements;
    for (let i = 0; i < 16; i++) expect(e[i]).toBeCloseTo(i % 5 === 0 ? 1 : 0, 9);
  });
});

describe("object motion is world-space and camera-independent (修正指示書 Test D / E)", () => {
  it("Test D: moveBy translates in world X regardless of the camera", () => {
    const { world, cam } = setup();
    const obj = world.createCube(0.1);
    obj.place([0.2, 0, -0.5]);
    cam.setPose({ position: new Float64Array([0.4, 0.3, 0.1]), quaternion: quatYaw(0.4) }, 0);
    obj.moveBy(0.1, 0, 0);
    expect(obj.position.toArray()).toEqual([0.30000000000000004, 0, -0.5]);
    cam.setPose({ position: new Float64Array([-0.4, 0.3, 0.1]), quaternion: quatYaw(-0.4) }, 0.1);
    expect(obj.position.x).toBeCloseTo(0.3, 12);
    expect(obj.position.z).toBeCloseTo(-0.5, 12);
  });

  it("Test E: a walking character advances by its own velocity while the camera moves", () => {
    const { world, cam } = setup();
    const character = world.createCube(0.1);
    character.place([0, 0, -0.8]);
    character.velocity.set(0.05, 0, 0); // 5 cm/s along world X
    character.angularVelocityY = 0.5;
    const other = world.createCube(0.1);
    other.place([0.3, 0, -0.8]);

    let t = 0;
    for (let i = 0; i < 40; i++) {
      // Camera sweeps left/right and turns while the character walks.
      const x = 0.3 * Math.sin(i / 6);
      cam.setPose({ position: new Float64Array([x, 0.1, 0.2]), quaternion: quatYaw(0.3 * Math.cos(i / 6)) }, t);
      world.update(0.05);
      t += 0.05;
    }
    // 2 s × 0.05 m/s = 0.10 m along X; nothing else changed.
    expect(character.position.x).toBeCloseTo(0.1, 9);
    expect(character.position.y).toBeCloseTo(0, 12);
    expect(character.position.z).toBeCloseTo(-0.8, 12);
    expect(character.yaw).toBeCloseTo(1.0, 9);
    // The static object did not move at all.
    expect(other.position.toArray()).toEqual([0.3, 0, -0.8]);
    // Plane-local (x, z) is world (x, 0, z).
    expect(world.planeToWorld(0.1, -0.8)).toEqual([0.1, 0, -0.8]);
  });

  it("does not integrate motion before placement or for non-positive dt", () => {
    const { world } = setup();
    const obj = world.createCube(0.1);
    obj.velocity.set(1, 0, 0);
    world.update(1);
    expect(obj.position.x).toBe(0);
    obj.place([0, 0, 0]);
    world.update(0);
    world.update(-1);
    expect(obj.position.x).toBe(0);
  });
});
