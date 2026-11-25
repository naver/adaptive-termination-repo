#!/usr/bin/env bash
# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#
# Calibration before serving, one dataset, in order. Every shard calibrates on 500 of its own rows held out of the
# graph (the `corpus` arm); the comparison arms run on the same final graphs: nat (fixed ef) and rule (calibrated on
# the benchmark calibration queries #1000-1499, as in the main experiment). Every step skips what exists, so a rerun
# resumes.
#
#   1. partial graphs and held-out rows                          data/build_holdout.sh part
#   2. corpus calibration on the partial graphs, (S, E) cells    corpus_cell.sh
#   3. final graphs (held-out rows inserted); the partial graphs are then deleted (KEEP_PART=1 keeps them)
#                                                                data/build_holdout.sh final
#   4. rule calibration on the final graphs                      cell.sh table
#   5. tolerance dumps nat, rule, corpus x blocks a, b           dump_cell.sh
#   6. diagnostics of the held-out rows (holdout.json), verdicts, summary
#                                                                lib/holdout.py diag, lib/bootstrap.py, corpus_report.py
#
# Locations: the exact ground truth is read from $IDX_DIR/<dsn>/gt300*.ivec (data/ground_truth.sh; linked, not copied).
# Everything this driver writes goes elsewhere, so the main experiment's indexes and results are never touched:
#   HOLDOUT_IDX_DIR (default $WORK/idx_holdout), HOLDOUT_TRACE_DIR ($WORK/trace_holdout), HOLDOUT_RESULTS ($WORK/results_holdout)
# CORPUS_SS / CORPUS_E override the S and E lists (default: 1 5 15 for T2I and MS MARCO, config.sh ds_ss otherwise;
# $E_LIMITS). Independent cells run $JOBS at a time.
#
# Usage: run_corpus.sh <t2i|msmarco|glove|sift|synth>
set -uo pipefail
EXP=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$EXP/config.sh"
[ $# -eq 1 ] || die "usage: run_corpus.sh <t2i|msmarco|glove|sift|synth>"
DS=$1
N=$(ds_n "$DS") || exit 1
DSN=${DS}_N${N}
GT_IDX=$IDX_DIR/$DSN
export IDX_DIR=${HOLDOUT_IDX_DIR:-$WORK/idx_holdout}
export TRACE_DIR=${HOLDOUT_TRACE_DIR:-$WORK/trace_holdout}
export RESULTS=${HOLDOUT_RESULTS:-$WORK/results_holdout}
export REPRO WORK RAW_DIR BIN PY JOBS K
export "N_$(echo "$DS" | tr '[:lower:]' '[:upper:]')=$N"    # shardlib.ds_N for datasets other than the six (synth)
case $DS in t2i|msmarco) SS=${CORPUS_SS:-1 5 15};; *) SS=${CORPUS_SS:-$(ds_ss "$DS")};; esac
E_LIST=${CORPUS_E:-$E_LIMITS}
IDX=$IDX_DIR/$DSN
log() { echo "[corpus $DS $(date +%m-%d\ %H:%M:%S)] $*"; }
par() { xargs -P "$JOBS" -L 1 "$@" || die "a cell of this step failed (see the messages above); rerun to resume"; }

# ---- inputs: ground truth of the full row set (the final graphs hold every row)
need_bin
[ -s "$RAW_DIR/$DS/query.fbin" ] || die "query vectors not found: $RAW_DIR/$DS/query.fbin (see data/)"
mkdir -p "$IDX"
for g in gt300.ivec gt300_proxy.ivec gt300_q1500.ivec; do
  [ -s "$IDX/$g" ] && continue
  [ -s "$GT_IDX/$g" ] || die "ground truth not found: $GT_IDX/$g (data/ground_truth.sh $DS)"
  ln -s "$(cd "$GT_IDX" && pwd)/$g" "$IDX/$g"
done
log "start: S = $SS, E = $E_LIST, indexes -> $IDX, results -> $RESULTS"

# ---- 1. partial graphs
"$REPRO/data/build_holdout.sh" part "$DS" $SS || die "partial graphs failed"
log "1. partial graphs done"

# ---- 2. corpus calibration
for S in $SS; do for E in $E_LIST; do echo "$DS $S $E"; done; done | par "$EXP/corpus_cell.sh"
log "2. corpus calibration done"

# ---- 3. final graphs
"$REPRO/data/build_holdout.sh" final "$DS" $SS || die "final graphs failed"
if [ "${KEEP_PART:-0}" != 1 ]; then
  for S in $SS; do rm -f "$IDX/S$S"/part*/shard0.index; done
fi
log "3. final graphs done"

# ---- 4. rule calibration on the final graphs
for S in $SS; do for E in $E_LIST; do echo "table $DS $S $E"; done; done | par env RANK_MODE=rule "$EXP/cell.sh"
log "4. rule calibration done"

# ---- 5. tolerance dumps
for S in $SS; do for E in $E_LIST; do for A in nat rule corpus; do for B in a b; do echo "$DS $S $E $A $B"; done; done; done; done \
  | par "$EXP/dump_cell.sh"
log "5. tolerance dumps done"

# ---- 6. diagnostics, verdicts, summary
for S in $SS; do
  $PY "$REPRO/lib/holdout.py" diag --ds "$DS" --N "$N" --idx "$IDX" --base "$RAW_DIR/$DS/base.fbin" \
    --query "$RAW_DIR/$DS/query.fbin" --gt_proxy "$IDX/gt300_proxy.ivec" --S "$S" --K "$K" --out "$RESULTS/holdout.json" \
    || die "holdout diagnostics failed (S=$S)"
done
$PY "$REPRO/lib/bootstrap.py" --results "$RESULTS" > "$RESULTS/bootstrap.log" || die "bootstrap failed"
$PY "$EXP/corpus_report.py" --results "$RESULTS" || die "summary failed"
log "6. done: $RESULTS/verdicts/bootstrap.json, $RESULTS/corpus_summary.md"
