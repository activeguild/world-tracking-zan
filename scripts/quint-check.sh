#!/usr/bin/env bash
# Quint guardrail for the session lifecycle × worker protocol
# (quint/session.qnt): exhaustive model check, proof that the model is not
# vacuous, and that the checked-in conformance walks are what the model
# produces. Needs Java 17+ (TLC; Quint downloads it on first use).
set -euo pipefail
cd "$(dirname "$0")/.."
quint="node_modules/.bin/quint"
spec="quint/session.qnt"

if [ ! -x "$quint" ]; then
  echo "quint-check: $quint not found. Run npm ci." >&2
  exit 1
fi
if ! command -v java >/dev/null 2>&1; then
  echo "quint-check: java not found (the model checker needs Java 17+)." >&2
  exit 1
fi

"$quint" typecheck "$spec" >/dev/null

log="$(mktemp)"
tmp="$(mktemp)"
trap 'rm -f "$log" "$tmp"; rm -rf _apalache-out' EXIT

# 1. Every reachable state of the repaired model satisfies every invariant,
#    and the interesting states are reachable.
if ! "$quint" verify "$spec" --backend tlc --invariant safety --search-observations >"$log" 2>&1; then
  cat "$log" >&2
  echo "quint-check: the model violates an invariant." >&2
  exit 1
fi
for obs in appliesThreeFrames runsOnMainBackend dropsStaleResult abortsStart; do
  if ! grep -q "^$obs was observed" "$log"; then
    cat "$log" >&2
    echo "quint-check: observation $obs is not reachable (the model became vacuous)." >&2
    exit 1
  fi
done
grep -E "distinct states found, 0 states left" "$log" | head -1

# 2. Each repair is needed: without it, the invariant it protects has a
#    counterexample. (A model that cannot find the known bugs proves nothing.)
expect_violation() {
  local init="$1" invariant="$2"
  if "$quint" verify "$spec" --backend tlc --init "$init" --invariant "$invariant" >"$log" 2>&1; then
    echo "quint-check: $invariant holds from $init — the model no longer detects that bug." >&2
    exit 1
  fi
  if ! grep -q "found a counterexample" "$log"; then
    cat "$log" >&2
    echo "quint-check: $init / $invariant failed for another reason than a counterexample." >&2
    exit 1
  fi
  echo "quint-check: $init violates $invariant (as it must)."
}
expect_violation initNoWorkerErrorFix busyIffOneInFlight
expect_violation initNoRestartFix lastCallWins
expect_violation initNoStaleResetFix noStaleResult

# 3. The walks the TypeScript conformance test replays are the model's.
node scripts/quint-traces.mjs --stdout >"$tmp"
if ! diff -q "$tmp" quint/conformance/session-traces.json >/dev/null; then
  echo "quint-check: quint/conformance/session-traces.json is stale. Regenerate with: npm run quint:export" >&2
  exit 1
fi
echo "quint-check: model checked, bugs detectable, conformance walks up to date."
