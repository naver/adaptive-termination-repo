#!/usr/bin/env bash
# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#
# How many calibration queries the table needs: recalibrate one cell (dataset, S, E) of adaptive termination
# on the first P of its 500 calibration queries. It reuses the calibration record of the main experiment
# ($TRACE_DIR/<dsn>/rule/proxy_S{S}_E{E}.bin, written by `cell.sh table`), so no search is rerun here; the dumps of
# the arm p<P> (dump_cell.sh) and lib/bootstrap.py then judge it on the test queries like the rule.
#
# Output: key S{S}_E{E} of $RESULTS/cells/<dsn>/ncalib/calib_cells_p<P>.json; skipped if present.
# Usage: experiments/ncalib_cell.sh <ds> <S> <E> <P>      (the original runs: P in {125, 250}, S in {1, 5, 15}, four datasets)
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/config.sh"
[ $# -eq 4 ] || die "usage: ncalib_cell.sh <ds> <S> <E> <P>"
DS=$1; S=$2; E=$3; P=$4
N=$(ds_n "$DS") || exit 1
DSN=${DS}_N${N}; KEY=S${S}_E${E}
OUT=$RESULTS/cells/$DSN/ncalib; mkdir -p "$OUT"
CELLS=$OUT/calib_cells_p$P.json
TB=$TRACE_DIR/$DSN/rule/proxy_$KEY.bin
GT=$IDX_DIR/$DSN/gt300_proxy.ivec
$PY -c "import json,sys;sys.exit(0 if sys.argv[2] in json.load(open(sys.argv[1])) else 1)" "$CELLS" "$KEY" 2>/dev/null && exit 0
[ -s "$TB" ] || die "calibration record not found: $TB (run experiments/run_core.sh $DS first)"
[ -s "$GT" ] || die "ground truth of the calibration queries not found: $GT"
# shared output file: one writer at a time when cells run in parallel
exec 9> "$CELLS.lock"
command -v flock > /dev/null 2>&1 && flock 9
$PY "$REPRO/lib/pred_calib.py" cell --ds "$DS" --N "$N" --trace "$TB" --gt "$GT" --key "$KEY" --out "$CELLS" \
  --gammas "$CALIB_GAMMAS" --K "$K" --nq_limit "$P" > /dev/null || die "pred_calib failed: $DSN $KEY P=$P"
