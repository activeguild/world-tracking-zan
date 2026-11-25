# Quint models

## Session lifecycle × worker protocol (`session.qnt`)

`session.qnt` models what happens when `ARSession.start()` / `stop()` /
`reset()` are called at any moment while the camera and the vision worker
start up, one frame is in flight, and responses arrive later. Lean
(`lean/`) proves single rules for every parameter value; this checks the
*interleavings* of several components, exhaustively, within small bounds
(3 frames and 2 resets per run).

| Invariant | Meaning |
|---|---|
| `busyIffOneInFlight` | the client is busy iff exactly one frame is between `processFrame()` and its result / error |
| `idleHoldsNothing` | not running and not starting ⇒ no camera, no backend, no messages |
| `cameraMatchesPhase`, `backendMatchesPhase` | the camera / backend exist exactly in the phases that own them |
| `lastCallWins` | once nothing is pending, the session is in the state asked for last (unless that start failed) |
| `noStaleResult`, `filterIsExact` | after `reset()`, nothing the engine computed before it is applied; the frame-id filter drops exactly those |

## Map corrections × world anchor (`mapcorrection.qnt`)

The engine refines its map in place (init stage, keyframe BA); the session's
world anchor must follow every correction exactly once, in the engine's
order. A correction is an opaque tag, "the anchor is where the map is" is
equality of tag lists (rigid transforms do not commute). Bounds: 5 frames,
3 corrections.

| Invariant | Meaning |
|---|---|
| `anchorFollowsMap` | after a result is handled, the anchor is attached to that result's map and reflects exactly its corrections, in order |
| `anchorPointsFresh` | a correction is never fitted around anchor points of another map or of an earlier state of this one (needs one frame in flight) |

## Tracking pipeline (`tracking.qnt`)

One step is one frame: the engine (pending relocalization candidate →
confirmation → apply → monitor window, map reset rules, plane recovery, the
tracking state machine, `worldEstablished`), then the session (world created
/ dropped, object visibility). The frame counts of the configuration are
scaled down; the rules are the same.

| Invariant | Meaning |
|---|---|
| `worldAgrees` | the session has a world exactly when the engine has established one, for the same map |
| `visibleOnlyWhenTrusted` | objects are shown only in world tracking, located within the tolerance; on a relocalized pose being confirmed only if they were shown already |
| `showStartsOnFreshPose` | showing starts on a frame whose pose came from the map |
| `noShowStartWhileConfirming` | showing never starts in the apply frame nor while the monitor window is open (was `hiddenWhileConfirming`: nothing shown at all) |
| `shownThroughRelocalization` | a relocalization applied inside the tolerated dropout does not hide what is shown |
| `pendingIsYoung`, `acceptableIsConfirmed` | a pending candidate is settled in the next frame; a relaxed-bound candidate is never applied unconfirmed |
| `recoveryOnlyBeforeWorld` | a plane recovery runs only before the world exists and while the map is not lost |
| `recoversWhenTrackingIsGood` | bounded liveness: after enough good frames in an established world the objects are visible |

The state machine and the session side are replayed against the code; the
engine's relocalization bookkeeping is the model's reading of
`VisionEngine.updateMap` and has no automatic check.

## Commands

- `npm run quint:check` — exhaustive model check (TLC), each repaired bug is
  still detectable when its repair is switched off, conformance walks up to
  date. Needs Java 17+; Quint downloads the checker on first use.
- `npm run quint:export` — regenerate `conformance/*-traces.json`.
- `npx vitest run tests/quint` — replay the walks on the real `ARSession`,
  `VisionWorkerClient`, worker handler, `WorldAnchor` and `TrackingStateMachine`
  (part of `npm test`;
  no Java, no Quint).
- Explore a counterexample:
  `npx quint run quint/session.qnt --backend typescript --init initNoWorkerErrorFix --invariant busyIffOneInFlight --verbosity 3`

## Changing the code

When `start` / `stop` / `reset` / the worker protocol / how map corrections
are reported and applied / the relocalization flow, the tracking states, world
creation or object visibility changes: change the
model first, run `quint:check`, then `quint:export` and the conformance
test. An invariant that no longer holds is a rule being dropped — say so in
the change, do not delete it silently.
