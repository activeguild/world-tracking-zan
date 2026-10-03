import { describe, expect, it } from "vitest";
import { FrameGrabber, rgbaToGray } from "../../src/camera/CameraFrame";
import { approximateIntrinsics, scaleIntrinsics } from "../../src/camera/CameraIntrinsics";
import { ImagePyramid } from "../../src/vision/ImagePyramid";

describe("rgbaToGray", () => {
  it("converts RGBA to Rec.601 luma", () => {
    const rgba = new Uint8ClampedArray([255, 255, 255, 255, 0, 0, 0, 255, 255, 0, 0, 255, 0, 255, 0, 255]);
    const out = new Uint8Array(4);
    rgbaToGray(rgba, out);
    expect(out[0]).toBe(255);
    expect(out[1]).toBe(0);
    expect(out[2]).toBe((77 * 255) >> 8);
    expect(out[3]).toBe((150 * 255) >> 8);
  });
});

describe("FrameGrabber.fitProcessingSize", () => {
  it("keeps the aspect ratio and uses the long side as the target", () => {
    expect(FrameGrabber.fitProcessingSize(1280, 720, 640, 480)).toEqual({ width: 640, height: 360 });
    expect(FrameGrabber.fitProcessingSize(640, 480, 640, 480)).toEqual({ width: 640, height: 480 });
    // Portrait camera (iPhone held upright reports 720x1280).
    expect(FrameGrabber.fitProcessingSize(720, 1280, 640, 480)).toEqual({ width: 360, height: 640 });
  });
  it("rounds to multiples of 4", () => {
    const s = FrameGrabber.fitProcessingSize(1234, 777, 640, 480);
    expect(s.width % 4).toBe(0);
    expect(s.height % 4).toBe(0);
  });
});

describe("CameraIntrinsics", () => {
  it("approximates and rescales", () => {
    const k = approximateIntrinsics(1280, 720);
    expect(k.fx).toBe(1280);
    expect(k.cx).toBe(640);
    const s = scaleIntrinsics(k, 640, 360);
    expect(s.fx).toBe(640);
    expect(s.cx).toBe(320);
    expect(s.cy).toBe(180);
  });
});

describe("ImagePyramid", () => {
  it("halves dimensions per level and box-filters", () => {
    const w = 32;
    const h = 16;
    const img = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) img[y * w + x] = x * 5;
    const p = new ImagePyramid(w, h, 3);
    p.build(img);
    expect(p.levels.length).toBe(2); // 32x16 → 16x8 (next would be 8x4 < 8 → stop)
    expect(p.levels[1].width).toBe(16);
    expect(p.levels[1].height).toBe(8);
    // Pixel (0,0) at level 1 averages x=0,1 → (0+5+0+5+2)/4 = 3
    expect(p.levels[1].data[0]).toBe(3);
    // Gradient along x at level 0 interior: I(x+1)-I(x-1) = 10
    expect(p.levels[0].gradX[1 * w + 5]).toBe(10);
    expect(p.levels[0].gradY[1 * w + 5]).toBe(0);
  });
});
