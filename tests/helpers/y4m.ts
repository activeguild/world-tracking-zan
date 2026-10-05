import { writeFileSync } from "node:fs";

/**
 * Write grayscale frames as a YUV4MPEG2 (I420) file, the format Chromium's
 * `--use-file-for-fake-video-capture` accepts. Chroma planes are neutral.
 */
export function writeGrayY4M(path: string, width: number, height: number, frames: Uint8Array[], fps = 30): void {
  const header = Buffer.from(`YUV4MPEG2 W${width} H${height} F${fps}:1 Ip A1:1 C420jpeg\n`, "ascii");
  const frameTag = Buffer.from("FRAME\n", "ascii");
  const chromaSize = (width >> 1) * (height >> 1);
  const chroma = Buffer.alloc(chromaSize * 2, 128);
  const parts: Buffer[] = [header];
  for (const f of frames) {
    if (f.length !== width * height) throw new Error("frame size mismatch");
    parts.push(frameTag, Buffer.from(f.buffer, f.byteOffset, f.byteLength), chroma);
  }
  writeFileSync(path, Buffer.concat(parts));
}
