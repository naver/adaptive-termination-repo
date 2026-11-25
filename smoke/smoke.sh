#!/usr/bin/env bash
# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#
# Smoke test of the package. Two levels:
#
#   smoke/smoke.sh          quick: no compiled code, no data. Seconds to a minute.
#       1. lib/shardlib.py: k_i and w_j against reference values
#       2. figures/make_all.sh on the shipped results: the five data figures are drawn and the main numbers are
#          recomputed (text_numbers.py --strict: every recomputed one must equal its reference value; the script lists
#          the few it cannot recompute)
#
#   smoke/smoke.sh --full   also the whole experiment pipeline on a tiny synthetic dataset (20,000 x 32, 2,000 queries),
#       S in {1, 5}, ef limits {32, 64}: ground truth, shard split, HNSW build, calibration, gamma-grid replay, a direct
#       run, rank-K termination, per-query dumps, bootstrap verdicts, the approximate ground truth (calibration and,
#       where it selects other gamma_i, verdicts again), the Patience comparison (records with insertion counts, replay
#       against direct runs), then invariant checks (smoke/check_synth.py). Needs the bench
#       binary ($BIN; faiss/build_faiss.sh + bench/build_bench.sh, or the Docker image of faiss/Dockerfile). A few minutes.
#       Then calibration before serving on the same data (experiments/run_corpus.sh: held-out shard rows, partial and
#       final graphs, corpus/rule/nat dumps, verdicts) and its invariant checks (smoke/check_corpus.py).
#
# Outputs go to $WORK/smoke (default repro/work/smoke); nothing under results/ is written.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/../experiments/config.sh"
FULL=0; [ "${1:-}" = "--full" ] && FULL=1
SM=$WORK/smoke; mkdir -p "$SM"
ok() { echo "  [ok] $*"; }
step() { echo "=== $*"; }

step "1. shardlib constants"
$PY - <<'EOF'
import shardlib as L
assert [L.k_i_of(20, S) for S in (1, 5, 15, 30, 50)] == [20, 12, 7, 5, 4], "k_i"
assert L.k_i_of(5, 15) == 3, "k_i at K=5, S=15"
top3 = sum(L.w_j(20, 15)[:3])
assert abs(top3 - 0.96) < 0.005, top3                  # w_1 + w_2 + w_3 at S=15
assert L.CALIB_GAMMAS[0] == 0.005 and L.CALIB_GAMMAS[-1] == 0.1 and len(L.CALIB_GAMMAS) == 20
print("  [ok] k_i = 20/12/7/5/4, w_1..3 at S=15 = %.3f, calibration grid 0.005..0.1 (20)" % top3)
EOF

step "2. figures and numbers from the shipped results"
rm -rf "$SM/figures"                                   # no figure of an earlier run may stand in
RESULTS="$REPRO/results" OUT="$SM/figures" "$REPRO/figures/make_all.sh" > "$SM/figures.log" 2>&1 \
  || { tail -n 30 "$SM/figures.log"; die "figures/make_all.sh failed (log: $SM/figures.log)"; }
(cd "$REPRO/figures" && RESULTS="$REPRO/results" $PY text_numbers.py --strict > "$SM/text_numbers.log" 2>&1) \
  || { grep -v '^OK' "$SM/text_numbers.log" | tail -n 20; die "a recomputed number differs from its reference value (log: $SM/text_numbers.log)"; }
ok "$(tail -n 1 "$SM/text_numbers.log")"
for f in fig-waste fig-overhead-same-recall fig-adaptive-savings-both fig-shard-tail fig-verdicts; do
  [ -s "$SM/figures/$f.pdf" ] || die "missing $f.pdf"
done
if grep -q '\] source: raw' "$SM/figures.log"; then TS="per-query dumps (DUMPS_DIR)"; else TS="results/derived (rounded report values)"; fi
ok "five data figures drawn in $SM/figures; fig-shard-tail and the ef-limit share from $TS"

[ $FULL = 1 ] || { echo "quick smoke test passed (run with --full for the synthetic pipeline)"; exit 0; }

# ---------------------------------------------------------------- synthetic end-to-end
need_bin
step "3. synthetic pipeline (bench: $BIN)"
export RESULTS=$SM/results RAW_DIR=$SM/raw IDX_DIR=$SM/idx TRACE_DIR=$SM/trace SYNTH_SS="1 5"
EG="32 64"
[ -s "$RAW_DIR/synth/base.fbin" ] || $PY "$REPRO/data/make_synthetic.py" --out "$RAW_DIR/synth"
export N_SYNTH=$(ds_n synth) NORM_SYNTH=0
RAW_DIR=$RAW_DIR IDX_DIR=$IDX_DIR "$REPRO/data/prepare_all.sh" --steps gt,index --S "1 5" synth
for S in 1 5; do
  for E in $EG; do
    "$REPRO/experiments/cell.sh" calib synth $S $E
    "$REPRO/experiments/cell.sh" grid synth $S $E
    "$REPRO/experiments/cell.sh" direct synth $S $E 0.05
    "$REPRO/experiments/calib_dabs.sh" synth $S $E          # at S = 1 this copies adaptive termination's entry (same rank K)
    [ $S -gt 1 ] && RANK_MODE=dabs "$REPRO/experiments/cell.sh" grid synth $S $E
    for arm in nat rule dabs; do
      for blk in a b; do "$REPRO/experiments/dump_cell.sh" synth $S $E $arm $blk; done
    done
  done
done
# the driver of the main experiments on the same cells (skips what the loop above made; adds same-recall matching)
CORE_E="$EG" JOBS=2 "$REPRO/experiments/run_core.sh" synth
DUMPS_DIR="$RESULTS" $PY "$REPRO/lib/bootstrap.py" --results "$RESULTS" --B 2000
# approximate ground truth: calibrate again, then judge again the configurations whose gamma_i differ (B as above, since
# approx_verdicts.py judge checks its recomputed rule rows against verdicts/bootstrap.json)
for S in 1 5; do for E in $EG; do "$REPRO/experiments/approx_cell.sh" synth $S $E; done; done
CONFS=$($PY "$REPRO/experiments/approx_verdicts.py" list --results "$RESULTS")
if [ -n "$CONFS" ]; then
  echo "$CONFS" | while read -r DS S E; do for BLK in a b; do "$REPRO/experiments/dump_cell.sh" "$DS" "$S" "$E" approx "$BLK"; done; done
  DUMPS_DIR="$RESULTS" $PY "$REPRO/experiments/approx_verdicts.py" judge --results "$RESULTS" --B 2000 || die "approx_verdicts judge failed"
  ok "approximate ground truth: $(echo "$CONFS" | wc -l | tr -d ' ') configuration(s) judged again"
else
  ok "approximate ground truth: the same gamma_i in every synthetic shard (nothing to judge again)"
fi
# queue saturation (Patience) on the same cells: records with insertion counts, replays checked against direct runs
PATIENCE_DS=synth PATIENCE_E="$EG" PATIENCE_VERIFY="5 64" JOBS=2 "$REPRO/experiments/run_extras.sh" patience > "$SM/patience.log" 2>&1 \
  || { tail -n 30 "$SM/patience.log"; die "run_extras.sh patience failed (log: $SM/patience.log)"; }
$PY "$HERE/check_synth.py" "$RESULTS" "synth_N$N_SYNTH" "1 5" "$EG"
ok "synthetic pipeline: invariants hold; verdicts in $RESULTS/verdicts/bootstrap.json"

step "4. calibration before serving (held-out shard rows)"
HOLDOUT_IDX_DIR=$SM/idx_holdout HOLDOUT_TRACE_DIR=$SM/trace_holdout HOLDOUT_RESULTS=$SM/results_holdout \
  CORPUS_E="$EG" JOBS=2 "$REPRO/experiments/run_corpus.sh" synth > "$SM/corpus.log" 2>&1 \
  || { tail -n 30 "$SM/corpus.log"; die "run_corpus.sh failed (log: $SM/corpus.log)"; }
$PY "$HERE/check_corpus.py" "$SM/idx_holdout" "$SM/results_holdout" "synth_N$N_SYNTH" "1 5" "$EG"
ok "calibration before serving: invariants hold; summary in $SM/results_holdout/corpus_summary.md"
echo "full smoke test passed"
