/-
Model of the `found` flag of `PlaneDetector.update` for a frame that
produced a candidate (src/vision/PlaneDetector.ts).
-/
namespace Guardrails.PlaneFound

structure S where
  found : Bool
  stableFrames : Nat
  inconsistent : Nat
  deriving DecidableEq, Repr

/--
`stable`: the candidate agrees with the previous one. `horizontal`,
`enough` (inlier count): properties of this candidate. `hasPrev`: a previous
candidate exists. `required`: `stableFramesRequired`.
-/
def step (required : Nat) (s : S) (hasPrev stable horizontal enough : Bool) : S :=
  let sf := if stable then s.stableFrames + 1 else 0
  let inc := if stable then 0 else s.inconsistent
  if horizontal && decide (required ≤ sf) && enough then
    { found := true, stableFrames := sf, inconsistent := inc }
  else if !horizontal || !stable then
    if !stable && hasPrev && s.found then
      { found := decide (inc + 1 ≤ 2), stableFrames := sf, inconsistent := inc + 1 }
    else if !horizontal then { found := false, stableFrames := sf, inconsistent := inc }
    else { found := s.found, stableFrames := sf, inconsistent := inc }
  else { found := s.found, stableFrames := sf, inconsistent := inc }

/-- An unstable horizontal candidate while found (the only kind the theorems below feed). -/
abbrev odd (required : Nat) (s : S) : S := step required s true false true true

/-- One odd candidate (two inconsistent comparisons) is tolerated. -/
theorem one_odd_candidate_keeps_found (required : Nat) (s : S)
    (hf : s.found = true) (hi : s.inconsistent = 0) (hr : 0 < required) :
    (odd required (odd required s)).found = true := by
  have h0 : ¬ required ≤ 0 := by omega
  simp [odd, step, hf, hi, h0]

/-- Review finding 13: a third inconsistent candidate in a row drops `found`. -/
theorem third_inconsistent_drops_found (required : Nat) (s : S)
    (hf : s.found = true) (hi : s.inconsistent = 0) (hr : 0 < required) :
    (odd required (odd required (odd required s))).found = false := by
  have h0 : ¬ required ≤ 0 := by omega
  simp [odd, step, hf, hi, h0]

/-- After that, `found` comes back only through the stable-frames rule. -/
theorem found_needs_stable_frames (required : Nat) (s : S) (hasPrev stable horizontal enough : Bool)
    (hr : 0 < required)
    (hnf : s.found = false) (h : (step required s hasPrev stable horizontal enough).found = true) :
    stable = true ∧ horizontal = true ∧ enough = true ∧ required ≤ s.stableFrames + 1 := by
  unfold step at h
  cases stable <;> cases horizontal <;> cases enough <;> simp [hnf] at h ⊢ <;> (try split at h) <;> simp_all <;> omega

end Guardrails.PlaneFound
