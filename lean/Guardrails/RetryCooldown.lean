/-
Model of the relocalization retry cooldown (`Relocalizer.suppressed` /
`noteFailure`, src/vision/Relocalizer.ts). A keyframe that failed at frame
`f` is skipped on an unchanged view while `frame ≤ f + cooldown`.
-/
namespace Guardrails.RetryCooldown

/-- Frame up to which a keyframe that failed at `failedAt` is skipped. -/
def cooldownUntil (failedAt cooldown : Nat) : Nat := failedAt + cooldown

/-- Skipped at `frame` (the view is assumed unchanged). -/
def suppressed (failedAt cooldown frame : Nat) : Bool :=
  decide (frame ≤ cooldownUntil failedAt cooldown)

/-- The k-th attempt after the failure is skipped exactly while `k · period ≤ cooldown`. -/
theorem suppressed_iff (f c p k : Nat) :
    suppressed f c (f + k * p) = true ↔ k * p ≤ c := by
  simp [suppressed, cooldownUntil]

/--
Review finding 14, with the shipped values (attempt every 3 frames,
cooldown 6): the two attempts after a failure skip the keyframe, the third
may try it again — so three attempts of three keyframes cover all eight.
-/
theorem two_attempts_skipped (f : Nat) :
    suppressed f 6 (f + 3) = true ∧ suppressed f 6 (f + 6) = true ∧ suppressed f 6 (f + 9) = false := by
  simp [suppressed, cooldownUntil]

end Guardrails.RetryCooldown
