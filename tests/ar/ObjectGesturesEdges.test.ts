import { describe, expect, it } from "vitest";
import { GestureRecognizer, wrapAngle, type GestureEvent } from "../../src/ar/ObjectGestures";

/**
 * Edge cases of the gesture state machine (found by mutation testing):
 * exact slop / time limits, duplicate and extra pointers, the dead state
 * after a twist, and the pinch distance floor (8 px).
 */
const kinds = (events: GestureEvent[]) => events.map((e) => e.kind);
const make = () => new GestureRecognizer({ tapSlopPx: 12, tapMaxMs: 400 });

describe("GestureRecognizer: tap limits are inclusive", () => {
  it("a release exactly at the slop and exactly at the time limit is a tap; the time is measured from the press", () => {
    const g = make();
    g.down(1, 100, 100, 1000);
    expect(g.up(1, 112, 100, 1400)).toEqual([{ kind: "tap", x: 100, y: 100 }]);
    g.down(1, 100, 100, 5000);
    expect(g.up(1, 100, 100, 5401)).toEqual([]);
    g.down(1, 100, 100, 6000);
    expect(g.up(1, 113, 100, 6100)).toEqual([]);
  });

  it("a move of exactly the slop does not start a drag", () => {
    const g = make();
    g.down(1, 100, 100, 0);
    expect(g.move(1, 112, 100, 10)).toEqual([]);
    expect(g.currentMode).toBe("pending");
    expect(kinds(g.move(1, 113, 100, 20))).toEqual(["dragStart", "drag"]);
  });

  it("a second down for a pointer already down is ignored: the press position and time stay", () => {
    const g = make();
    g.down(1, 100, 100, 0);
    expect(g.down(1, 300, 300, 200)).toEqual([]);
    expect(g.up(1, 100, 100, 350)).toEqual([{ kind: "tap", x: 100, y: 100 }]);
  });
});

describe("GestureRecognizer: extra pointers and the dead state", () => {
  it("a third pointer starts nothing", () => {
    const g = make();
    g.down(1, 100, 100, 0);
    expect(kinds(g.down(2, 200, 100, 10))).toEqual(["rotateStart"]);
    expect(g.down(3, 150, 300, 20)).toEqual([]);
    expect(g.currentMode).toBe("rotate");
  });

  it("after a twist the remaining finger is dead: a new second finger does not start another twist", () => {
    const g = make();
    g.down(1, 100, 100, 0);
    g.down(2, 200, 100, 10);
    expect(kinds(g.up(2, 200, 100, 20))).toEqual(["rotateEnd"]);
    expect(g.currentMode).toBe("dead");
    expect(g.down(3, 220, 120, 30)).toEqual([]);
    expect(g.currentMode).toBe("dead");
    expect(g.move(3, 260, 180, 40)).toEqual([]);
  });

  it("dead lasts until every finger is up, not until the first of several", () => {
    const g = make();
    g.down(1, 100, 100, 0);
    g.down(2, 200, 100, 10);
    g.down(3, 150, 300, 20);
    g.up(1, 100, 100, 30); // twist over, two fingers remain
    expect(g.currentMode).toBe("dead");
    g.up(2, 200, 100, 40); // one remains
    expect(g.currentMode).toBe("dead");
    expect(g.down(4, 50, 50, 50)).toEqual([]);
    g.up(3, 150, 300, 60);
    g.up(4, 50, 50, 70);
    expect(g.currentMode).toBe("idle");
  });

  it("reset forgets the pointers that were down: the next press is a fresh single press", () => {
    const g = make();
    g.down(1, 100, 100, 0);
    g.reset();
    g.down(2, 50, 60, 100);
    expect(g.currentMode).toBe("pending");
    expect(g.up(2, 50, 60, 150)).toEqual([{ kind: "tap", x: 50, y: 60 }]);
  });
});

describe("GestureRecognizer: twist centre and the pinch distance floor", () => {
  it("rotate and pinch carry the midpoint of the two fingers", () => {
    const g = make();
    g.down(1, 100, 100, 0);
    g.down(2, 200, 100, 10);
    const ev = g.move(2, 200, 300, 20);
    expect(ev.map((e) => e.kind)).toEqual(["rotate", "pinch"]);
    for (const e of ev) expect(e).toMatchObject({ cx: 150, cy: 200 });
  });

  it("fingers that start closer than the floor give no ratio on the first move apart, then pinch from there", () => {
    const g = make();
    g.down(1, 100, 100, 0);
    g.down(2, 104, 100, 10); // 4 px apart
    expect(kinds(g.move(2, 116, 100, 20))).toEqual([]); // 16 px: no previous distance to compare with
    expect(g.move(2, 132, 100, 30)).toEqual([{ kind: "pinch", scale: 2, cx: 116, cy: 100 }]);
  });

  it("exactly at the floor counts as too close, as a start and as a destination", () => {
    const start = make();
    start.down(1, 100, 100, 0);
    start.down(2, 108, 100, 10); // exactly 8 px
    expect(start.move(2, 116, 100, 20)).toEqual([]);

    const g = make();
    g.down(1, 100, 100, 0);
    g.down(2, 140, 100, 10); // 40 px
    expect(g.move(2, 108, 100, 20)).toEqual([]); // to exactly 8 px: no pinch, the 40 px stays the reference
    expect(g.move(2, 120, 100, 30)).toEqual([{ kind: "pinch", scale: 0.5, cx: 110, cy: 100 }]);
  });

  it("passing below the floor keeps the last usable distance as the reference", () => {
    const g = make();
    g.down(1, 100, 100, 0);
    g.down(2, 140, 100, 10); // 40 px
    expect(g.move(2, 104, 100, 20)).toEqual([]); // 4 px
    expect(g.move(2, 180, 100, 30)).toEqual([{ kind: "pinch", scale: 2, cx: 140, cy: 100 }]);
  });
});

describe("wrapAngle", () => {
  it("maps into (−π, π]: −π wraps to +π", () => {
    expect(wrapAngle(-Math.PI)).toBe(Math.PI);
    expect(wrapAngle(Math.PI)).toBe(Math.PI);
  });
});
