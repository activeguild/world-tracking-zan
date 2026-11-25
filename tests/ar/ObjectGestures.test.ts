import { describe, expect, it } from "vitest";
import { GestureRecognizer, wrapAngle, type GestureEvent } from "../../src/ar/ObjectGestures";

/** Object manipulation gestures: tap / drag / two-finger twist, as a pure state machine. */
const kinds = (events: GestureEvent[]) => events.map((e) => e.kind);

describe("GestureRecognizer (object manipulation)", () => {
  it("a short press without movement is a tap at the press position", () => {
    const g = new GestureRecognizer({ tapSlopPx: 12, tapMaxMs: 400 });
    expect(g.down(1, 100, 200, 0)).toEqual([]);
    expect(g.move(1, 104, 203, 50)).toEqual([]); // within the slop
    const up = g.up(1, 104, 203, 120);
    expect(up).toEqual([{ kind: "tap", x: 100, y: 200 }]);
    expect(g.currentMode).toBe("idle");
  });

  it("a long press or a press that moved beyond the slop is not a tap", () => {
    const g = new GestureRecognizer({ tapSlopPx: 12, tapMaxMs: 400 });
    g.down(1, 100, 200, 0);
    expect(g.up(1, 100, 200, 900)).toEqual([]);
    g.down(2, 10, 10, 1000);
    expect(kinds(g.move(2, 40, 10, 1010))).toEqual(["dragStart", "drag"]);
    expect(kinds(g.up(2, 40, 10, 1020))).toEqual(["dragEnd"]);
  });

  it("drag: dragStart carries the press position, drag the current position and the delta, dragEnd on release", () => {
    const g = new GestureRecognizer({ tapSlopPx: 12, tapMaxMs: 400 });
    g.down(1, 100, 200, 0);
    const first = g.move(1, 130, 205, 20);
    expect(first[0]).toEqual({ kind: "dragStart", x: 100, y: 200 });
    expect(first[1]).toEqual({ kind: "drag", x: 130, y: 205, dx: 30, dy: 5 });
    expect(g.move(1, 140, 215, 40)).toEqual([{ kind: "drag", x: 140, y: 215, dx: 10, dy: 10 }]);
    expect(g.up(1, 140, 215, 60)).toEqual([{ kind: "dragEnd" }]);
    expect(g.currentMode).toBe("idle");
  });

  it("twist: two pointers report the change of their angle (screen radians, clockwise positive), and the remaining pointer is dead afterwards", () => {
    const g = new GestureRecognizer();
    g.down(1, 60, 100, 0);
    const start = g.down(2, 140, 100, 10);
    expect(start).toEqual([{ kind: "rotateStart", cx: 100, cy: 100 }]);
    // Rotate the pair by 90° clockwise on screen (y down): pointer 2 moves to the bottom, pointer 1 to the top.
    // (the distance changes too, so a pinch accompanies each rotate; only the rotation is read here)
    const a = g.move(2, 100, 140, 20).filter((e) => e.kind === "rotate");
    expect(a.length).toBe(1);
    const b = g.move(1, 100, 60, 30).filter((e) => e.kind === "rotate");
    expect(b.length).toBe(1);
    const total = [a[0], b[0]].reduce((s, e) => s + (e.kind === "rotate" ? e.deltaRad : 0), 0);
    expect(total).toBeCloseTo(Math.PI / 2, 9);
    expect((b[0] as { cx: number; cy: number }).cx).toBeCloseTo(100, 9);
    expect(g.up(2, 100, 140, 40)).toEqual([{ kind: "rotateEnd" }]);
    expect(g.currentMode).toBe("dead");
    // The pointer still down neither drags nor taps.
    expect(g.move(1, 200, 60, 50)).toEqual([]);
    expect(g.up(1, 200, 60, 60)).toEqual([]);
    expect(g.currentMode).toBe("idle");
    // A fresh tap works again.
    g.down(3, 5, 5, 100);
    expect(g.up(3, 5, 5, 150)).toEqual([{ kind: "tap", x: 5, y: 5 }]);
  });

  it("a second pointer ends a drag in progress and starts a twist; the angle wraps across ±π", () => {
    const g = new GestureRecognizer();
    g.down(1, 100, 100, 0);
    g.move(1, 150, 100, 10);
    const second = g.down(2, 50, 100, 20); // pointer 2 to the left: angle π
    expect(kinds(second)).toEqual(["dragEnd", "rotateStart"]);
    // Move pointer 2 slightly below the line: the angle goes from π to −(π − ε), a small clockwise step, not −2π.
    const r = g.move(2, 50, 102, 30).filter((e) => e.kind === "rotate");
    expect(r.length).toBe(1);
    expect(Math.abs((r[0] as { deltaRad: number }).deltaRad)).toBeLessThan(0.05);
    expect(wrapAngle(Math.PI + 0.1)).toBeCloseTo(-Math.PI + 0.1, 12);
    expect(wrapAngle(-Math.PI - 0.1)).toBeCloseTo(Math.PI - 0.1, 12);
  });

  it("pinch: two pointers report the ratio of their distance on every move, alongside any rotation; a pure spread has no rotation", () => {
    const g = new GestureRecognizer();
    g.down(1, 80, 100, 0);
    g.down(2, 120, 100, 0); // 40 px apart
    const spread = g.move(2, 160, 100, 10); // 80 px apart, same angle
    expect(kinds(spread)).toEqual(["pinch"]);
    expect((spread[0] as { scale: number }).scale).toBeCloseTo(2, 12);
    const shrink = g.move(1, 100, 100, 20); // 60 px apart
    expect((shrink[0] as { scale: number }).scale).toBeCloseTo(60 / 80, 12);
    // Rotate and spread together: both events, the product of the ratios is the total ratio.
    const both = g.move(2, 100, 220, 30); // pointer 1 at (100,100), pointer 2 straight below at 120 px
    expect(kinds(both)).toEqual(["rotate", "pinch"]);
    expect((both[0] as { deltaRad: number }).deltaRad).toBeCloseTo(Math.PI / 2, 12);
    expect((both[1] as { scale: number }).scale).toBeCloseTo(120 / 60, 12);
    // Fingers on top of each other: no pinch (and no bogus ratio when they separate again).
    const collapse = g.move(2, 100, 102, 40);
    expect(kinds(collapse).includes("pinch")).toBe(false);
    const reopen = g.move(2, 100, 220, 50);
    expect(kinds(reopen).includes("pinch")).toBe(false);
    expect(g.up(2, 100, 220, 60)).toEqual([{ kind: "rotateEnd" }]);
  });

  it("cancel never produces a tap; a third pointer neither steers nor breaks the twist; reset forgets everything", () => {
    const g = new GestureRecognizer();
    g.down(1, 10, 10, 0);
    expect(g.cancel(1)).toEqual([]);
    expect(g.currentMode).toBe("idle");
    g.down(1, 0, 0, 0);
    g.down(2, 100, 0, 0);
    g.down(3, 50, 50, 0);
    expect(g.move(3, 60, 60, 5)).toEqual([]);
    expect(kinds(g.move(2, 100, 10, 10))).toEqual(["rotate", "pinch"]);
    g.reset();
    expect(g.currentMode).toBe("idle");
    expect(g.up(1, 0, 0, 20)).toEqual([]);
  });
});
