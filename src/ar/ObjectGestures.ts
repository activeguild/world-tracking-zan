/**
 * Touch / pointer gesture recognizer for manipulating placed objects
 * (spec §27–§30: objects move only by user placement or their own motion,
 * never with the camera).
 *
 * Pure state machine over pointer ids and positions — no DOM, so it is unit
 * tested directly. The caller (main.ts) feeds pointer events and decides what
 * each gesture means:
 *
 *   tap          one pointer down and up within the slop / time → place or move
 *   drag         one pointer moved beyond the slop → dragStart at the press
 *                position, drag on every move, dragEnd on release
 *   rotate       two pointers: the change of the angle between them, in
 *                screen radians (y down, so clockwise on screen is positive)
 *   pinch        two pointers: the ratio of their distance to the previous
 *                one (reported alongside rotate on every two-finger move)
 *
 * A second pointer ends a drag and starts a twist; after a twist the pointer
 * that remains is dead (no tap, no drag) until everything is released, so a
 * two-finger gesture never ends in an accidental placement.
 */
export type GestureEvent =
  | { kind: "tap"; x: number; y: number }
  | { kind: "dragStart"; x: number; y: number }
  | { kind: "drag"; x: number; y: number; dx: number; dy: number }
  | { kind: "dragEnd" }
  | { kind: "rotateStart"; cx: number; cy: number }
  | { kind: "rotate"; deltaRad: number; cx: number; cy: number }
  /** Two-finger pinch: the ratio of the current finger distance to the previous one (> 1 = spread). */
  | { kind: "pinch"; scale: number; cx: number; cy: number }
  | { kind: "rotateEnd" };

export interface GestureOptions {
  /** Movement (px) a single pointer may make and still count as a tap. */
  tapSlopPx: number;
  /** Longest press (ms) that still counts as a tap. */
  tapMaxMs: number;
}

export const DEFAULT_GESTURE_OPTIONS: GestureOptions = { tapSlopPx: 12, tapMaxMs: 400 };

interface Pointer {
  x: number;
  y: number;
  startX: number;
  startY: number;
  startT: number;
}

type Mode = "idle" | "pending" | "drag" | "rotate" | "dead";

export class GestureRecognizer {
  private readonly pointers = new Map<number, Pointer>();
  /** Insertion order of the active pointers (the first two define a twist). */
  private order: number[] = [];
  private mode: Mode = "idle";
  private lastAngle = 0;
  private lastDistance = 0;
  private readonly opts: GestureOptions;

  constructor(opts: Partial<GestureOptions> = {}) {
    this.opts = { ...DEFAULT_GESTURE_OPTIONS, ...opts };
  }

  get currentMode(): Mode {
    return this.mode;
  }

  down(id: number, x: number, y: number, t: number): GestureEvent[] {
    const out: GestureEvent[] = [];
    if (this.pointers.has(id)) return out;
    this.pointers.set(id, { x, y, startX: x, startY: y, startT: t });
    this.order.push(id);
    const n = this.pointers.size;
    if (n === 1) {
      this.mode = "pending";
    } else if (n === 2 && this.mode !== "dead") {
      if (this.mode === "drag") out.push({ kind: "dragEnd" });
      this.mode = "rotate";
      const [a, b] = this.pair();
      this.lastAngle = angleOf(a, b);
      this.lastDistance = distanceOf(a, b);
      out.push({ kind: "rotateStart", cx: (a.x + b.x) / 2, cy: (a.y + b.y) / 2 });
    }
    // A third pointer changes nothing: the first two keep defining the twist.
    return out;
  }

  move(id: number, x: number, y: number, t: number): GestureEvent[] {
    void t;
    const out: GestureEvent[] = [];
    const p = this.pointers.get(id);
    if (!p) return out;
    const dx = x - p.x;
    const dy = y - p.y;
    p.x = x;
    p.y = y;
    switch (this.mode) {
      case "pending":
        if (Math.hypot(x - p.startX, y - p.startY) > this.opts.tapSlopPx) {
          this.mode = "drag";
          out.push({ kind: "dragStart", x: p.startX, y: p.startY });
          out.push({ kind: "drag", x, y, dx: x - p.startX, dy: y - p.startY });
        }
        break;
      case "drag":
        out.push({ kind: "drag", x, y, dx, dy });
        break;
      case "rotate": {
        if (this.order.indexOf(id) > 1) break; // a third pointer does not steer
        const [a, b] = this.pair();
        const cx = (a.x + b.x) / 2, cy = (a.y + b.y) / 2;
        const angle = angleOf(a, b);
        const delta = wrapAngle(angle - this.lastAngle);
        this.lastAngle = angle;
        if (delta !== 0) out.push({ kind: "rotate", deltaRad: delta, cx, cy });
        const distance = distanceOf(a, b);
        // Fingers (nearly) on top of each other define neither a scale nor an angle: skip until they separate.
        if (distance > MIN_PINCH_DISTANCE_PX && this.lastDistance > MIN_PINCH_DISTANCE_PX && distance !== this.lastDistance) {
          out.push({ kind: "pinch", scale: distance / this.lastDistance, cx, cy });
        }
        if (distance > MIN_PINCH_DISTANCE_PX) this.lastDistance = distance;
        break;
      }
      default:
        break;
    }
    return out;
  }

  up(id: number, x: number, y: number, t: number): GestureEvent[] {
    return this.release(id, x, y, t, true);
  }

  cancel(id: number): GestureEvent[] {
    const p = this.pointers.get(id);
    if (!p) return [];
    return this.release(id, p.x, p.y, p.startT, false);
  }

  /** Forget every pointer (e.g. the session stopped). */
  reset(): void {
    this.pointers.clear();
    this.order = [];
    this.mode = "idle";
  }

  private release(id: number, x: number, y: number, t: number, mayTap: boolean): GestureEvent[] {
    const out: GestureEvent[] = [];
    const p = this.pointers.get(id);
    if (!p) return out;
    this.pointers.delete(id);
    this.order = this.order.filter((o) => o !== id);
    switch (this.mode) {
      case "pending":
        if (mayTap && t - p.startT <= this.opts.tapMaxMs && Math.hypot(x - p.startX, y - p.startY) <= this.opts.tapSlopPx) {
          out.push({ kind: "tap", x: p.startX, y: p.startY });
        }
        this.mode = "idle";
        break;
      case "drag":
        out.push({ kind: "dragEnd" });
        this.mode = "idle";
        break;
      case "rotate":
        out.push({ kind: "rotateEnd" });
        this.mode = this.pointers.size > 0 ? "dead" : "idle";
        break;
      case "dead":
        if (this.pointers.size === 0) this.mode = "idle";
        break;
      default:
        break;
    }
    return out;
  }

  private pair(): [Pointer, Pointer] {
    return [this.pointers.get(this.order[0])!, this.pointers.get(this.order[1])!];
  }
}

/** Below this finger distance (px) a pinch ratio would be noise. */
const MIN_PINCH_DISTANCE_PX = 8;

function angleOf(a: Pointer, b: Pointer): number {
  return Math.atan2(b.y - a.y, b.x - a.x);
}

function distanceOf(a: Pointer, b: Pointer): number {
  return Math.hypot(b.x - a.x, b.y - a.y);
}

export function wrapAngle(rad: number): number {
  let a = rad;
  while (a > Math.PI) a -= 2 * Math.PI;
  while (a <= -Math.PI) a += 2 * Math.PI;
  return a;
}
