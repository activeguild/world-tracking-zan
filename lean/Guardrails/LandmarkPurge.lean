/-
Model of the relink purge decision (`MapTracker.reassociateWith`,
src/vision/MapTracker.ts): what one far landing of a keyframe patch does to
a landmark.
-/
namespace Guardrails.LandmarkPurge

inductive Outcome where
  | keep (strikes : Nat)
  | purge
  deriving DecidableEq, Repr

/--
`canPurge`: the predicted position came from the canonical pose of a tracked
frame. `landed`: distance of the forward-backward verified landing from the
candidate. `lo`/`hi`: `maxObservationErrorPx` / `lkMaxDisplacementPx`
(`hi = 0` means no upper bound). `strikes`: far landings seen before.
-/
def onFarLanding (canPurge : Bool) (lo hi landed strikes : Nat) : Outcome :=
  if canPurge && decide (0 < lo) && decide (lo ≤ landed) && (decide (hi = 0) || decide (landed ≤ hi)) then
    if strikes + 1 < 2 then .keep (strikes + 1) else .purge
  else .keep strikes

/-- Review finding 2: nothing is purged on a lost frame or under a guided-recovery seed pose. -/
theorem no_purge_without_tracked_pose (lo hi landed strikes : Nat) :
    onFarLanding false lo hi landed strikes = .keep strikes := by
  simp [onFarLanding]

/-- A first far landing never removes a landmark. -/
theorem first_strike_keeps (canPurge : Bool) (lo hi landed : Nat) :
    onFarLanding canPurge lo hi landed 0 ≠ .purge := by
  unfold onFarLanding; split <;> simp

/-- A purge implies every condition held: tracked pose, landing in range, an earlier strike. -/
theorem purge_conditions (canPurge : Bool) (lo hi landed strikes : Nat)
    (h : onFarLanding canPurge lo hi landed strikes = .purge) :
    canPurge = true ∧ lo ≤ landed ∧ (hi = 0 ∨ landed ≤ hi) ∧ 1 ≤ strikes := by
  unfold onFarLanding at h
  split at h
  · rename_i hc
    split at h
    · simp at h
    · rename_i hs
      simp at hc
      exact ⟨hc.1.1.1, hc.1.2, hc.2, by omega⟩
  · simp at h

/-- A runaway landing beyond the upper bound never counts. -/
theorem beyond_bound_keeps (canPurge : Bool) (lo hi landed strikes : Nat)
    (hhi : 0 < hi) (hfar : hi < landed) :
    onFarLanding canPurge lo hi landed strikes = .keep strikes := by
  unfold onFarLanding
  have h1 : ¬ hi = 0 := by omega
  have h2 : ¬ landed ≤ hi := by omega
  simp [h1, h2]

end Guardrails.LandmarkPurge
