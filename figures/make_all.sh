#!/usr/bin/env bash
# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#
# Draw the data figures and check the main numbers against their reference values, from a RESULTS root.
#   RESULTS   result root (default: the shipped results, $REPRO/results; use $WORK/results for a re-run)
#   OUT       output directory (default: $REPRO/work/figures)
#   TAIL_SOURCE  auto | raw | derived -- inputs of fig-shard-tail and of the share of shard searches that ran
#                to the ef limit (default auto: per-query dumps if present under RESULTS, otherwise results/derived/)
# Figures: fig-waste, fig-overhead-same-recall, fig-adaptive-savings-both, fig-shard-tail, fig-verdicts (paper_figs.py).
# Any missing input stops the script with a non-zero exit.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO="$(cd "$HERE/.." && pwd)"
RESULTS="${RESULTS:-$REPRO/results}"; OUT="${OUT:-$REPRO/work/figures}"
RESULTS="$(cd "$RESULTS" && pwd)"                       # absolute before the cd below (relative paths are taken
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"               # from the caller's directory)
export RESULTS OUT
PY="${PY:-python3}"

cd "$HERE"
echo "== figures (RESULTS=$RESULTS) -> $OUT"
$PY paper_figs.py
echo "== numbers vs reference values"
$PY text_numbers.py
echo "done: $OUT"
