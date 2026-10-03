import { ARError, ARErrorCode } from "../ar/ARState";

export interface CameraOptions {
  /** Preferred capture width. The browser may deliver something else. */
  idealWidth?: number;
  /** Preferred capture height. */
  idealHeight?: number;
  /** Preferred frame rate. */
  idealFrameRate?: number;
  facingMode?: "environment" | "user";
}

/**
 * Wraps getUserMedia() and the HTMLVideoElement (spec §9, §53).
 *
 * Responsibilities:
 *  - request the rear camera
 *  - map browser errors onto ARErrorCode
 *  - expose the actual video dimensions once metadata is available
 */
export class CameraManager {
  private stream: MediaStream | null = null;
  private _width = 0;
  private _height = 0;

  constructor(
    public readonly video: HTMLVideoElement,
    private readonly options: CameraOptions = {},
  ) {}

  /** Native video width after start() resolved. */
  get width(): number {
    return this._width;
  }

  /** Native video height after start() resolved. */
  get height(): number {
    return this._height;
  }

  get isRunning(): boolean {
    return this.stream !== null;
  }

  async start(): Promise<void> {
    if (this.stream) return;

    if (!navigator.mediaDevices?.getUserMedia) {
      throw new ARError(ARErrorCode.CAMERA_UNAVAILABLE, "getUserMedia() is not available");
    }

    const constraints: MediaStreamConstraints = {
      audio: false,
      video: {
        facingMode: this.options.facingMode ?? "environment",
        width: { ideal: this.options.idealWidth ?? 1280 },
        height: { ideal: this.options.idealHeight ?? 720 },
        frameRate: { ideal: this.options.idealFrameRate ?? 30 },
      },
    };

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (err) {
      throw mapGetUserMediaError(err);
    }

    this.stream = stream;
    const video = this.video;
    video.srcObject = stream;
    // iOS Safari requires these for inline autoplay without user gesture.
    video.setAttribute("playsinline", "true");
    video.muted = true;
    video.autoplay = true;

    await waitForMetadata(video);
    try {
      await video.play();
    } catch (err) {
      this.stop();
      throw new ARError(ARErrorCode.CAMERA_UNAVAILABLE, "video.play() failed", { cause: err });
    }

    this._width = video.videoWidth;
    this._height = video.videoHeight;
    if (this._width === 0 || this._height === 0) {
      this.stop();
      throw new ARError(ARErrorCode.CAMERA_UNAVAILABLE, "video reported zero dimensions");
    }
  }

  stop(): void {
    if (this.stream) {
      for (const track of this.stream.getTracks()) track.stop();
      this.stream = null;
    }
    this.video.srcObject = null;
    this._width = 0;
    this._height = 0;
  }
}

function waitForMetadata(video: HTMLVideoElement): Promise<void> {
  if (video.readyState >= HTMLMediaElement.HAVE_METADATA && video.videoWidth > 0) {
    return Promise.resolve();
  }
  return new Promise((resolve, reject) => {
    const onLoaded = () => {
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(new ARError(ARErrorCode.CAMERA_UNAVAILABLE, "video element error"));
    };
    const cleanup = () => {
      video.removeEventListener("loadedmetadata", onLoaded);
      video.removeEventListener("error", onError);
    };
    video.addEventListener("loadedmetadata", onLoaded);
    video.addEventListener("error", onError);
  });
}

export function mapGetUserMediaError(err: unknown): ARError {
  const name = (err as { name?: string } | null)?.name ?? "";
  switch (name) {
    case "NotAllowedError":
    case "PermissionDeniedError":
    case "SecurityError":
      return new ARError(ARErrorCode.CAMERA_PERMISSION_DENIED, `Camera permission denied (${name})`, {
        cause: err,
      });
    case "NotFoundError":
    case "DevicesNotFoundError":
    case "NotReadableError":
    case "TrackStartError":
    case "OverconstrainedError":
    case "ConstraintNotSatisfiedError":
    case "AbortError":
    case "TypeError":
    default:
      return new ARError(ARErrorCode.CAMERA_UNAVAILABLE, `Camera unavailable (${name || "unknown"})`, {
        cause: err,
      });
  }
}
