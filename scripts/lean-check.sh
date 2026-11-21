#!/usr/bin/env bash
# Formal guardrails: build the Lean models and proofs, refuse escape hatches,
# and make sure the checked-in conformance table is what the model produces.
# Needs `lake` (Lean 4, version pinned in lean/lean-toolchain) on PATH.
set -euo pipefail
cd "$(dirname "$0")/../lean"

if ! command -v lake >/dev/null 2>&1; then
  echo "lean-check: lake not found. Install Lean 4 (see lean/README.md)." >&2
  exit 1
fi

# A proof with a hole is not a proof; an added axiom is an unchecked claim.
if grep -rnE '\b(sorry|admit)\b|^[[:space:]]*(axiom|unsafe)\b|native_decide' --include='*.lean' Guardrails Guardrails.lean Main.lean; then
  echo "lean-check: sorry / admit / axiom / unsafe / native_decide are not allowed." >&2
  exit 1
fi

lake build Guardrails conformance

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
lake exe conformance > "$tmp"
if ! diff -q "$tmp" conformance/tracking-state.json >/dev/null; then
  echo "lean-check: lean/conformance/tracking-state.json is stale. Regenerate with: npm run lean:export" >&2
  exit 1
fi
echo "lean-check: proofs built, no escape hatches, conformance table up to date."
