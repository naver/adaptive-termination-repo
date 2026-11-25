#!/usr/bin/env bash
# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#
# K axis: does the calibration table need one row per K?  S=15, ef=200, adaptive termination (rank = k_i of that K).
# One cell = (K, arm, block):
#   nat  natural termination (gamma 0)
#   dep  deployed: the per-shard gammas calibrated at K=20 ($CELLS_ROOT/<dsn>/calib_cells.json, S15_E200), used unchanged;
#        only K, k_i and the comparison rank change
#   ref  recalibrated: record the calibration queries at this K and calibrate again (pred_calib.py cell --K)
# Blocks a = #0-999, b = #1500- (the ground truth is top-300, so K up to 100 works). Verdicts: experiments/analyze.py.
#
# Output: $RESULTS/kaxis/<ds>_N<N>/K{K}_{arm}_{blk}.{tsv,json}; recalibration in calib_cells_ref.json (key K{K});
#         calibration records in $TRACE_DIR/<dsn>/kaxis/proxy_K{K}.bin. Skipped if they exist.
# Usage:  experiments/kaxis_cell.sh <ds> <K> <nat|dep|ref> <a|b>
set -uo pipefail
. "$(dirname "$0")/config.sh"
[ $# -eq 4 ] || die "usage: $0 <ds> <K> <nat|dep|ref> <a|b>"
DS=$1; KK=$2; ARM=$3; BLK=$4; S=15; E=200
N=$(ds_n "$DS") || exit 1
DSN=${DS}_N$N; OUT=$RESULTS/kaxis/$DSN; mkdir -p "$OUT"
CELLS_ROOT=${CELLS_ROOT:-$RESULTS/cells}
f=$OUT/K${KK}_${ARM}_${BLK}
[ -s "$f.tsv" ] && [ -s "$f.json" ] && exit 0
need_bin
NORM=$(ds_norm "$DS"); IDX=$IDX_DIR/$DSN; KI=$(k_i "$S" "$KK")
LOCKS=$WORK/locks; mkdir -p "$LOCKS" "$TRACE_DIR/$DSN/kaxis"
locked() { local l=$1; shift; if command -v flock > /dev/null 2>&1; then flock "$l" "$@"; else "$@"; fi; }
glist() { $PY -c "
import json, sys
c = json.load(open(sys.argv[1]))[sys.argv[2]]
print(','.join('0' if g is None else f'{g:g}' for g in c['gamma_rank_shard']))" "$1" "$2"; }

case $ARM in
  nat) G=0 ;;
  dep) G=$(glist "$CELLS_ROOT/$DSN/calib_cells.json" S15_E200) || die "no K=20 gammas for S15_E200 in $CELLS_ROOT/$DSN/calib_cells.json" ;;
  ref) CELLS=$OUT/calib_cells_ref.json; key=K$KK
       if ! $PY -c "import json,sys; sys.exit(0 if sys.argv[2] in json.load(open(sys.argv[1])) else 1)" "$CELLS" "$key" 2>/dev/null; then
         tb=$TRACE_DIR/$DSN/kaxis/proxy_K$KK.bin
         locked "$LOCKS/${DSN}_kaxis_K$KK.lock" bash -c '[ -s "$1" ] && exit 0
           "$2" search --shards "$3/S15" --S 15 --query "$4" --normalize "$5" --check_norm "$5" --nq 500 --qoff 1000 \
             --gt "$3/gt300_proxy.ivec" --K "$6" --ki "$7" --rule_k "$7" --ef 200 --gamma 0 --repeat 1 --warmup 100 --trace "$1.part" \
             > /dev/null 2> "$1.err" && mv "$1.part" "$1"' \
           _ "$tb" "$BIN" "$IDX" "$RAW_DIR/$DS/query.fbin" "$NORM" "$KK" "$KI" \
           || { echo "!! calibration record failed $DS K=$KK: $(tail -n 2 "$tb.err")" >&2; exit 1; }
         locked "$LOCKS/${DSN}_kaxis_cells.lock" $PY "$REPRO/lib/pred_calib.py" cell --ds "$DS" --N "$N" --trace "$tb" \
           --gt "$IDX/gt300_proxy.ivec" --key "$key" --out "$CELLS" --gammas "$CALIB_GAMMAS" --K "$KK" > /dev/null \
           || die "pred_calib failed $DS K=$KK"
       fi
       G=$(glist "$CELLS" "$key") || exit 1 ;;
  *) die "arm must be nat, dep or ref" ;;
esac
case $BLK in
  a) Q=(--nq 1000 --qoff 0 --gt "$IDX/gt300.ivec") ;;
  b) Q=(--nq "$(ds_nb "$DS")" --qoff 1500 --gt "$IDX/gt300_q1500.ivec") ;;
  *) die "block must be a or b" ;;
esac
"$BIN" search --shards "$IDX/S$S" --S "$S" --query "$RAW_DIR/$DS/query.fbin" --normalize "$NORM" --check_norm "$NORM" "${Q[@]}" \
  --K "$KK" --ki "$KI" --rule_k "$KI" --ef "$E" --gamma "$G" --repeat 1 --warmup 100 --dump "$f.tsv.part" > "$f.json.part" 2> "$f.err" \
  && $PY -c "import json,sys; json.load(open(sys.argv[1]))" "$f.json.part" 2>/dev/null \
  || { echo "!! run failed $f: $(tail -n 2 "$f.err")" >&2; rm -f "$f.json.part" "$f.tsv.part"; exit 1; }
mv "$f.tsv.part" "$f.tsv" && mv "$f.json.part" "$f.json"
$PY -c "
import json, sys
d = json.load(open(sys.argv[1]))[0]
print('  kaxis %-8s K=%-3s k_i=%-2s %-3s %s recall=%.5f ndis=%.1f  gamma=[%s]' % (sys.argv[2], d['K'], d['ki'], sys.argv[3], sys.argv[4],
      d['recall'], d['ndis_per_query'], sys.argv[5][:30]))" "$f.json" "$DS" "$ARM" "$BLK" "$G"
