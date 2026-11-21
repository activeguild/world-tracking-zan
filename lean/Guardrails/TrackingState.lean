/-
Model of `TrackingStateMachine.update` (src/ar/ARState.ts).

The TypeScript class is the implementation; this file is the specification
it is checked against (tests/lean/Conformance.test.ts replays the table
exported by `lake exe conformance`). The theorems below are the rules the
state machine must keep whatever else changes in it.
-/
namespace Guardrails.TrackingState

inductive State where
  | initializing | searchingFeatures | tracking | planeDetecting
  | planeFound | arActive | trackingLost | relocalizing
  deriving DecidableEq, Repr

open State

structure Thresholds where
  minTracked : Nat
  lostBelow : Nat
  lostFrameTolerance : Nat

structure Obs where
  inliers : Nat
  mapInitialized : Bool
  planeFound : Bool
  mapLost : Bool
  worldEstablished : Bool

structure Machine where
  state : State
  badFrames : Nat

/-- "Lost" group of the switch in `update`. -/
def State.isLost : State → Bool
  | searchingFeatures | trackingLost | relocalizing => true
  | _ => false

/-- "Tracking" group of the switch in `update`. -/
def State.isTracking : State → Bool
  | tracking | planeDetecting | planeFound | arActive => true
  | _ => false

def stepLost (t : Thresholds) (m : Machine) (o : Obs) : Machine :=
  let relocatable := o.mapInitialized && o.worldEstablished
  if t.minTracked ≤ o.inliers then
    { state :=
        if relocatable && o.mapLost then relocalizing
        else if relocatable && o.planeFound then planeFound
        else tracking
      badFrames := 0 }
  else if relocatable && !o.mapLost && o.planeFound && decide (t.lostBelow ≤ o.inliers) then
    { state := planeFound, badFrames := 0 }
  else if m.state = trackingLost then
    { m with state := if relocatable then relocalizing else searchingFeatures }
  else if m.state = relocalizing && !relocatable then
    { m with state := searchingFeatures }
  else m

def stepTracking (t : Thresholds) (m : Machine) (o : Obs) : Machine :=
  if o.inliers < t.lostBelow then
    if t.lostFrameTolerance ≤ m.badFrames + 1 then { state := trackingLost, badFrames := 0 }
    else { m with badFrames := m.badFrames + 1 }
  else if o.mapInitialized && o.mapLost then
    { state := if o.worldEstablished then relocalizing else tracking, badFrames := 0 }
  else if m.state = arActive then
    { state := if o.mapInitialized then arActive else tracking, badFrames := 0 }
  else if o.planeFound then { state := planeFound, badFrames := 0 }
  else if o.mapInitialized then
    { state := if m.state = planeFound then planeFound else planeDetecting, badFrames := 0 }
  else { state := tracking, badFrames := 0 }

def step (t : Thresholds) (m : Machine) (o : Obs) : Machine :=
  match m.state with
  | initializing => { state := searchingFeatures, badFrames := 0 }
  | searchingFeatures | trackingLost | relocalizing => stepLost t m o
  | tracking | planeDetecting | planeFound | arActive => stepTracking t m o

/-- v10: RELOCALIZING exists only to return to an established world. -/
theorem relocalizing_needs_world (t : Thresholds) (m : Machine) (o : Obs)
    (h : (step t m o).state = relocalizing) :
    o.mapInitialized = true ∧ o.worldEstablished = true := by
  obtain ⟨st, bad⟩ := m
  obtain ⟨inl, mi, pf, ml, we⟩ := o
  cases st <;> cases mi <;> cases we <;> cases ml <;> cases pf <;>
    simp [step, stepLost, stepTracking] at h ⊢ <;>
    (repeat' split at h) <;> simp_all

/--
Review finding 8: once the map locates the camera again in an established
world and the frame-to-frame inliers are at least `lostBelow`, a lost state
goes back to world tracking — it does not wait for `minTracked` inliers.
-/
theorem located_map_leaves_lost (t : Thresholds) (m : Machine) (o : Obs)
    (hl : m.state.isLost = true)
    (hmi : o.mapInitialized = true) (hwe : o.worldEstablished = true)
    (hml : o.mapLost = false) (hpf : o.planeFound = true)
    (hin : t.lostBelow ≤ o.inliers) :
    (step t m o).state = planeFound := by
  obtain ⟨st, bad⟩ := m
  obtain ⟨inl, mi, pf, ml, we⟩ := o
  simp only at hmi hwe hml hpf hin hl
  subst hmi hwe hml hpf
  cases st <;> simp [State.isLost] at hl <;>
    simp [step, stepLost, hin] <;> (try split) <;> rfl

/-- …and it holds there: no flapping between world tracking and a lost state at that inlier level. -/
theorem located_map_stays_found (t : Thresholds) (m : Machine) (o : Obs)
    (hs : m.state = planeFound)
    (hml : o.mapLost = false) (hpf : o.planeFound = true)
    (hin : t.lostBelow ≤ o.inliers) :
    step t m o = { state := planeFound, badFrames := 0 } := by
  unfold step stepTracking
  have : ¬ o.inliers < t.lostBelow := by omega
  simp [hs, hml, hpf, this]

/-- A tracking state is declared lost only after `lostFrameTolerance` consecutive bad frames. -/
theorem lost_needs_tolerance (t : Thresholds) (m : Machine) (o : Obs)
    (htr : m.state.isTracking = true)
    (h : (step t m o).state = trackingLost) :
    o.inliers < t.lostBelow ∧ t.lostFrameTolerance ≤ m.badFrames + 1 := by
  obtain ⟨st, bad⟩ := m
  obtain ⟨inl, mi, pf, ml, we⟩ := o
  cases st <;> simp [State.isTracking] at htr <;>
    cases mi <;> cases we <;> cases ml <;> cases pf <;>
    simp [step, stepTracking] at h ⊢ <;>
    (repeat' split at h) <;> simp_all <;> omega

end Guardrails.TrackingState
