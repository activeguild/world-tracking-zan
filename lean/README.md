# Formal guardrails (Lean 4)

Small Lean models of the decision logic that keeps breaking silently when the
engine is changed, with the rules they must keep proved as theorems. No
Mathlib; the toolchain is pinned in `lean-toolchain`.

| Model | TypeScript it mirrors | Proved |
|---|---|---|
| `Guardrails/TrackingState.lean` | `TrackingStateMachine.update` (`src/ar/ARState.ts`) | RELOCALIZING only with an established world; a located map leaves the lost states at `lostBelow` inliers and stays there; lost needs `lostFrameTolerance` bad frames |
| `Guardrails/RetryCooldown.lean` | `Relocalizer.suppressed` / `noteFailure` | a failed keyframe is skipped for exactly the next two attempts at the shipped cadence |
| `Guardrails/LandmarkPurge.lean` | relink purge in `MapTracker.reassociateWith` | no purge without a tracked pose, on a first strike, or beyond the distance bound |
| `Guardrails/PlaneFound.lean` | `found` flag of `PlaneDetector.update` | one odd candidate keeps `found`, a third inconsistent one drops it, and it returns only through the stable-frames rule |

## How the model and the implementation are tied

Only the state machine is tied mechanically: `lake exe conformance` exports
the model's full transition table (`conformance/tracking-state.json`, checked
in) and `tests/lean/Conformance.test.ts` replays the TypeScript class against
every row in the normal unit run — no Lean needed for that. The other three
models are specifications kept next to unit tests of the same rules; they are
not replayed automatically.

## Commands

```bash
npm run lean:check
```

builds the proofs, rejects `sorry` / `admit` / `axiom` / `unsafe` /
`native_decide`, and fails when the checked-in table is stale.

```bash
npm run lean:export
```

regenerates the table after a deliberate change of the model.

Lean is installed with [elan](https://github.com/leanprover/elan) or by
unpacking the release archive named in `lean-toolchain` and putting its
`bin/` on `PATH`.

## Rules when changing the mirrored code

1. Change the Lean model first, so that it states the new behaviour.
2. Make the theorems build again. A theorem that no longer holds is a rule
   being dropped: say so in the change, do not delete it quietly.
3. `npm run lean:export`, then make the TypeScript pass the conformance test.
