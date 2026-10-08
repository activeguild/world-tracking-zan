import type { CameraIntrinsics } from "./CameraIntrinsics";
import type { FrameSource, GrayFrame } from "./CameraFrame";

/**
 * WebGL frame grabber (v16): the video frame is uploaded as a texture, a
 * fragment shader downsamples it to the processing size and converts it to
 * luma, and `readPixels` returns the grayscale bytes directly into the
 * pooled buffer.
 *
 * Why: on Android Chrome the 2D-canvas path (`drawImage(video)` +
 * `getImageData`) took 31–41 ms per frame — the camera frame lives in a GPU
 * buffer and the 2D readback is the slow road — which halved the vision
 * rate although the engine itself took 22–31 ms. Through WebGL the upload
 * stays on the GPU side and only the small processing frame comes back.
 *
 * Packing: each RGBA texel of the render target holds four consecutive
 * horizontal gray pixels (R = x, G = x+1, B = x+2, A = x+3), so the target is
 * width/4 × height and `readPixels` transfers exactly width×height bytes —
 * the gray buffer itself, no conversion loop in JS. The processing width is a
 * multiple of 4 (`FrameGrabber.fitProcessingSize`).
 *
 * Output orientation matches the 2D grabber: row 0 is the top image row.
 * `readPixels` returns rows bottom-up, so the fragment at framebuffer row r
 * samples video row r counted from the top (texture v = 0 is the first,
 * i.e. top, row of the uploaded video).
 */
export class WebGLFrameGrabber implements FrameSource {
  readonly kind = "webgl" as const;
  private readonly canvas: HTMLCanvasElement | OffscreenCanvas;
  private readonly gl: WebGLRenderingContext;
  private readonly texture: WebGLTexture;
  private readonly pool: ArrayBuffer[] = [];
  private nextFrameId = 0;
  private readonly packedWidth: number;

  /** True when a WebGL context can be created here (not in Node, not without GPU). */
  static available(): boolean {
    try {
      const c = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(4, 4) : document.createElement("canvas");
      const gl = c.getContext("webgl", { failIfMajorPerformanceCaveat: false });
      if (!gl) return false;
      (gl as WebGLRenderingContext).getExtension("WEBGL_lose_context")?.loseContext();
      return true;
    } catch {
      return false;
    }
  }

  constructor(
    public readonly width: number,
    public readonly height: number,
  ) {
    if (width % 4 !== 0) throw new Error(`WebGLFrameGrabber: width ${width} must be a multiple of 4`);
    this.packedWidth = width / 4;
    this.canvas =
      typeof OffscreenCanvas !== "undefined"
        ? new OffscreenCanvas(this.packedWidth, height)
        : Object.assign(document.createElement("canvas"), { width: this.packedWidth, height });
    const gl = this.canvas.getContext("webgl", {
      alpha: true,
      antialias: false,
      depth: false,
      stencil: false,
      premultipliedAlpha: false,
      preserveDrawingBuffer: false,
    }) as WebGLRenderingContext | null;
    if (!gl) throw new Error("WebGL context unavailable");
    this.gl = gl;

    const vs = compile(
      gl,
      gl.VERTEX_SHADER,
      `attribute vec2 a_pos;
       void main() { gl_Position = vec4(a_pos, 0.0, 1.0); }`,
    );
    const fs = compile(
      gl,
      gl.FRAGMENT_SHADER,
      `#ifdef GL_FRAGMENT_PRECISION_HIGH
       precision highp float;
       #else
       precision mediump float;
       #endif
       // highp matters: with mediump a texel index near 640 only carries
       // ~0.5 px of precision, the samples land between pixels and the
       // bilinear blend differs from the 2D path by ~10 gray levels on edges.
       uniform sampler2D u_video;
       uniform vec2 u_size; // processing width, height
       const vec3 LUMA = vec3(0.299, 0.587, 0.114);
       float luma(float px, float py) {
         return dot(texture2D(u_video, vec2((px + 0.5) / u_size.x, (py + 0.5) / u_size.y)).rgb, LUMA);
       }
       void main() {
         // Framebuffer row r (bottom-up) samples image row r (top-down):
         // texture v = 0 is the top row of the uploaded video.
         float px = (gl_FragCoord.x - 0.5) * 4.0;
         float py = gl_FragCoord.y - 0.5;
         gl_FragColor = vec4(luma(px, py), luma(px + 1.0, py), luma(px + 2.0, py), luma(px + 3.0, py));
       }`,
    );
    const program = gl.createProgram();
    if (!program) throw new Error("WebGL program creation failed");
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(`WebGL program link failed: ${gl.getProgramInfoLog(program) ?? ""}`);
    }
    gl.useProgram(program);

    const quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const aPos = gl.getAttribLocation(program, "a_pos");
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
    gl.uniform2f(gl.getUniformLocation(program, "u_size"), width, height);
    gl.uniform1i(gl.getUniformLocation(program, "u_video"), 0);

    const texture = gl.createTexture();
    if (!texture) throw new Error("WebGL texture creation failed");
    this.texture = texture;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    // Linear sampling averages the 2×2 neighbourhood when downscaling by two
    // (the common 720p → 360 case); NPOT video textures need clamp + no mips.
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.viewport(0, 0, this.packedWidth, height);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
  }

  /** The context was lost (GPU reset, background tab); the caller should fall back. */
  get lost(): boolean {
    return this.gl.isContextLost();
  }

  grab(video: TexImageSource, timestamp: number, intrinsics: CameraIntrinsics, gravity: number[] | null = null): GrayFrame {
    const gl = this.gl;
    const { width, height } = this;
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    const buffer = this.pool.pop() ?? new ArrayBuffer(width * height);
    const gray = new Uint8Array(buffer);
    gl.readPixels(0, 0, this.packedWidth, height, gl.RGBA, gl.UNSIGNED_BYTE, gray);
    return { frameId: this.nextFrameId++, timestamp, width, height, data: gray, intrinsics, gravity };
  }

  release(buffer: ArrayBuffer): void {
    if (buffer.byteLength === this.width * this.height && this.pool.length < 4) this.pool.push(buffer);
  }

  dispose(): void {
    this.gl.getExtension("WEBGL_lose_context")?.loseContext();
  }
}

function compile(gl: WebGLRenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new Error("WebGL shader creation failed");
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader) ?? "";
    gl.deleteShader(shader);
    throw new Error(`WebGL shader compile failed: ${log}`);
  }
  return shader;
}
