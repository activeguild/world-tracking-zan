import Guardrails

/-!
`lake exe conformance` prints the transition table of the tracking state
machine model for the shipped thresholds as JSON. The table is checked in
(lean/conformance/tracking-state.json) and the TypeScript implementation is
replayed against it by tests/lean/Conformance.test.ts.
-/
open Guardrails.TrackingState

def stateName : State → String
  | .initializing => "INITIALIZING"
  | .searchingFeatures => "SEARCHING_FEATURES"
  | .tracking => "TRACKING"
  | .planeDetecting => "PLANE_DETECTING"
  | .planeFound => "PLANE_FOUND"
  | .arActive => "AR_ACTIVE"
  | .trackingLost => "TRACKING_LOST"
  | .relocalizing => "RELOCALIZING"

def allStates : List State :=
  [.initializing, .searchingFeatures, .tracking, .planeDetecting, .planeFound, .arActive, .trackingLost, .relocalizing]

def bools : List Bool := [false, true]

def b (x : Bool) : String := if x then "1" else "0"

def main : IO Unit := do
  let t : Thresholds := { minTracked := 40, lostBelow := 20, lostFrameTolerance := 3 }
  let mut rows : Array String := #[]
  for s in allStates do
    for bad in [0, 1, 2] do
      for inl in [0, 19, 20, 39, 40, 100] do
        for mi in bools do
          for pf in bools do
            for ml in bools do
              for we in bools do
                let r := step t { state := s, badFrames := bad }
                  { inliers := inl, mapInitialized := mi, planeFound := pf, mapLost := ml, worldEstablished := we }
                rows := rows.push
                  s!"[\"{stateName s}\",{bad},{inl},{b mi},{b pf},{b ml},{b we},\"{stateName r.state}\",{r.badFrames}]"
  IO.println "{"
  IO.println s!"  \"thresholds\": \{ \"minTrackedForTracking\": {t.minTracked}, \"lostBelow\": {t.lostBelow}, \"lostFrameTolerance\": {t.lostFrameTolerance} },"
  IO.println "  \"columns\": [\"state\", \"badFrames\", \"inlierCount\", \"mapInitialized\", \"planeFound\", \"mapLost\", \"worldEstablished\", \"nextState\", \"nextBadFrames\"],"
  IO.println "  \"rows\": ["
  IO.println ("    " ++ ",\n    ".intercalate rows.toList)
  IO.println "  ]"
  IO.println "}"
