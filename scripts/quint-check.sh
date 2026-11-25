#!/usr/bin/env bash
# Quint guardrails (quint/*.qnt): exhaustive model check, proof that the
# models are not vacuous, and that the checked-in conformance walks are what
# the models produce. Needs Java 17+ (TLC; Quint downloads it on first use).
set -euo pipefail
cd "$(dirname "$0")/.."
quint="node_modules/.bin/quint"

if [ ! -x "$quint" ]; then
  echo "quint-check: $quint not found. Run npm ci." >&2
  exit 1
fi
if ! command -v java >/dev/null 2>&1; then
  echo "quint-check: java not found (the model checker needs Java 17+)." >&2
  exit 1
fi

log="$(mktemp)"
trap 'rm -f "$log"; rm -rf _apalache-out' EXIT

# Every reachable state of the repaired model satisfies every invariant, and
# the interesting states are reachable.
check_model() {
  local spec="quint/$1.qnt"
  shift
  "$quint" typecheck "$spec" >/dev/null
  if ! "$quint" verify "$spec" --backend tlc --invariant safety --search-observations >"$log" 2>&1; then
    cat "$log" >&2
    echo "quint-check: $spec violates an invariant." >&2
    exit 1
  fi
  for obs in "$@"; do
    if ! grep -q "^$obs was observed" "$log"; then
      cat "$log" >&2
      echo "quint-check: $spec: observation $obs is not reachable (the model became vacuous)." >&2
      exit 1
    fi
  done
  echo "quint-check: $spec: $(grep -E "distinct states found, 0 states left" "$log" | head -1)"
}

# A repair (or an assumption) is needed: without it, the invariant it
# protects has a counterexample. A model that cannot find the known bugs
# proves nothing.
expect_violation() {
  local spec="quint/$1.qnt" init="$2" invariant="$3"
  if "$quint" verify "$spec" --backend tlc --init "$init" --invariant "$invariant" >"$log" 2>&1; then
    echo "quint-check: $spec: $invariant holds from $init — the model no longer detects that bug." >&2
    exit 1
  fi
  if ! grep -q "found a counterexample" "$log"; then
    cat "$log" >&2
    echo "quint-check: $spec: $init / $invariant failed for another reason than a counterexample." >&2
    exit 1
  fi
  echo "quint-check: $spec: $init violates $invariant (as it must)."
}

# Temporal properties (liveness under weak fairness), over the whole state graph.
check_liveness() {
  local spec="quint/$1.qnt" props="$2"
  if ! "$quint" verify "$spec" --backend tlc --no-invariant --temporal "$props" >"$log" 2>&1; then
    cat "$log" >&2
    echo "quint-check: $spec violates a temporal property ($props)." >&2
    exit 1
  fi
  echo "quint-check: $spec: $props hold."
}

# … and the same for a liveness property: without the repair there is a run
# in which the awaited thing never happens.
expect_liveness_violation() {
  local spec="quint/$1.qnt" init="$2" prop="$3"
  if "$quint" verify "$spec" --backend tlc --no-invariant --init "$init" --temporal "$prop" >"$log" 2>&1; then
    echo "quint-check: $spec: $prop holds from $init — the model no longer detects that bug." >&2
    exit 1
  fi
  if ! grep -q "found a counterexample" "$log"; then
    cat "$log" >&2
    echo "quint-check: $spec: $init / $prop failed for another reason than a counterexample." >&2
    exit 1
  fi
  echo "quint-check: $spec: $init violates $prop (as it must)."
}

# Session lifecycle × worker protocol.
check_model session appliesThreeFrames runsOnMainBackend dropsStaleResult abortsStart
expect_violation session initNoWorkerErrorFix busyIffOneInFlight
expect_violation session initNoRestartFix lastCallWins
expect_violation session initNoStaleResetFix noStaleResult
check_liveness session busyEnds,startSettles,lastStartRuns
expect_liveness_violation session initNoWorkerErrorFix busyEnds
expect_liveness_violation session initNoRestartFix lastStartRuns

# Map corrections × world anchor.
check_model mapcorrection anchorAfterTwoCorrections stageAndBaInOneFrame worldDroppedAndRecreated mapIdReused
expect_violation mapcorrection initNoOrderFix anchorFollowsMap
expect_violation mapcorrection initNoAnchorPointsFix anchorPointsFresh
# Not a repair but the assumption the anchor points rest on: one frame in flight.
expect_violation mapcorrection initTwoInFlight anchorPointsFresh
check_liveness mapcorrection everyResultHandled,anchorCatchesUp
expect_liveness_violation mapcorrection initNoOrderFix anchorCatchesUp

# Tracking pipeline: map loss, relocalization, tracking state, world, visibility, plane recovery.
check_model tracking relocalizedByConfirmation worldLostAndRebuilt planeRecoveryRan resetInEstablishedWorld relocalizedWhileShown
expect_violation tracking initNoWorldGate worldAgrees
expect_violation tracking initNoFreshShow showStartsOnFreshPose
expect_violation tracking initNoKeepShown shownThroughRelocalization
expect_violation tracking initNoLossByPnp lossEnds
expect_violation tracking initNoStateDrop worldAgrees

# The walks the TypeScript conformance tests replay are the models'.
node scripts/quint-traces.mjs --check
echo "quint-check: models checked, bugs detectable, conformance walks up to date."
