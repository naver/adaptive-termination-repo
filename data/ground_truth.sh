#!/usr/bin/env bash
# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#
# Exact top-300 ground truth (global row ids, nearest first) with the bench `gt` subcommand, written to $IDX_DIR/<ds>_N<N>/:
#   gt300.ivec        queries #0..     (T2I: #0-999; the other datasets: every query, of which the benches read #0-999)
#   gt300_proxy.ivec  queries #1000-1499   (calibration)
#   gt300_q1500.ivec  queries #1500-       (test block b: T2I #1500-20499, MS MARCO #1500-6979, GloVe/SIFT/DEEP #1500-9999)
# Exact search over the first N base rows; with --normalize 1 both sides are scaled to unit length first (squared L2 on
# unit vectors = cosine order). SIFT and DEEP use raw L2. T2I 100M (t2i100m) is split like T2I. Idempotent: existing files are kept.
#
# Usage: data/ground_truth.sh [t2i] [msmarco] [glove] [sift] [deep] [t2i100m] [synth]   (default: the four 10M/1M datasets)
#   THREADS (config.sh) sets OMP/OpenBLAS threads. N=... / NORM=... override the dataset defaults (e.g. for synth).
set -euo pipefail
. "$(dirname "$0")/common.sh"
need_bin

GT_K=300

run_gt() {   # run_gt OUT NQ QOFF
  local out=$1 nq=$2 qoff=$3
  if [ -s "$out" ]; then log "have $(basename "$out")"; return 0; fi
  log "gt $(basename "$out"): queries #$qoff-#$((qoff + nq - 1)) against $N rows (normalize=$NORM)"
  OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS \
    "$BIN" gt --base "$RAW/base.fbin" --limit "$N" --query "$RAW/query.fbin" --nq "$nq" --qoff "$qoff" \
              --k $GT_K --normalize "$NORM" --out "$out.part"
  mv "$out.part" "$out"
}

for ds in ${*:-$DATASETS_ALL}; do
  RAW="$RAW_DIR/$ds"
  [ -s "$RAW/base.fbin" ] && [ -s "$RAW/query.fbin" ] || die "$RAW/{base,query}.fbin missing (data/convert.sh $ds)"
  N=$(data_n "$ds"); NORM=$(data_norm "$ds")
  IDX="$IDX_DIR/${ds}_N${N}"; mkdir -p "$IDX"
  NQ_ALL=$(fbin_rows "$RAW/query.fbin")
  log "=== ground truth $ds N=$N normalize=$NORM queries=$NQ_ALL -> $IDX"
  if [ "$ds" = t2i ] || [ "$ds" = t2i100m ]; then
    # Four separate runs, as in the original pipeline (row i of each file = query qoff+i)
    run_gt "$IDX/gt300.ivec"        1000  0
    run_gt "$IDX/gt300_proxy.ivec"  500   1000
    run_gt "$IDX/gt300_q1500.ivec"  19000 1500
  else
    [ "$NQ_ALL" -gt 1500 ] || die "$ds has $NQ_ALL queries; the query split needs more than 1,500"
    # One run over every query, then the calibration and block-b rows are cut out of it
    run_gt "$IDX/gt300.ivec" "$NQ_ALL" 0
    expect_b=$((NQ_ALL - 1500))
    case $ds in msmarco|glove|sift|deep) [ "$(ds_nb "$ds")" = "$expect_b" ] || die "$ds: $expect_b block-b queries, expected $(ds_nb "$ds")";; esac
    [ -s "$IDX/gt300_proxy.ivec" ] || "$PY" "$DATA_DIR/gt_slice.py" --src "$IDX/gt300.ivec" --rows 1000:1500 --out "$IDX/gt300_proxy.ivec" --expect-rows 500
    [ -s "$IDX/gt300_q1500.ivec" ] || "$PY" "$DATA_DIR/gt_slice.py" --src "$IDX/gt300.ivec" --rows 1500: --out "$IDX/gt300_q1500.ivec" --expect-rows "$expect_b"
  fi
done
