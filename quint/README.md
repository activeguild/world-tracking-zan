# Quint model: session lifecycle × worker protocol

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

## Commands

- `npm run quint:check` — exhaustive model check (TLC), each repaired bug is
  still detectable when its repair is switched off, conformance walks up to
  date. Needs Java 17+; Quint downloads the checker on first use.
- `npm run quint:export` — regenerate `conformance/session-traces.json`.
- `npx vitest run tests/quint` — replay the walks on the real `ARSession`,
  `VisionWorkerClient` and worker handler (part of `npm test`; no Java, no Quint).
- Explore a counterexample:
  `npx quint run quint/session.qnt --backend typescript --init initNoWorkerErrorFix --invariant busyIffOneInFlight --verbosity 3`

## Changing the code

When `start` / `stop` / `reset` / the worker protocol changes: change the
model first, run `quint:check`, then `quint:export` and the conformance
test. An invariant that no longer holds is a rule being dropped — say so in
the change, do not delete it silently.
