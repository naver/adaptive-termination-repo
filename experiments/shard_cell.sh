#!/usr/bin/env bash
# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#
# Per-query x per-shard distance computations for one configuration (the shares of shard searches that reach the ef
# limit for easy and hard queries; ndis rows of fig-shard-tail; the max/mean imbalance ratio of the slowest shard).
#
# Arms: fixed    = fixed ef (gamma 0)
#       adaptive = adaptive termination (rank k_i) with the calibrated per-shard gammas of $CELLS_ROOT/<dsn>/calib_cells.json
# Blocks: a = queries #0-999 (gt300.ivec), b = #1500- (gt300_q1500.ivec). Same queries and ground truth as the tolerance dumps,
# so the per-query hits equal those of the rule / nat dumps; what this adds is ndis per shard.
#
# CELLS_ROOT defaults to $RESULTS/cells (your own calibration). Point it at $REPRO/results/cells to reuse the shipped gammas.
# Output: $RESULTS/shard/<ds>_N<N>/S{S}_ef{E}_{fixed|adaptive}_{a|b}.{tsv,json}; skipped if both exist.
# Usage:  experiments/shard_cell.sh <ds> <S> <ef> <fixed|adaptive> <a|b>
set -uo pipefail
. "$(dirname "$0")/config.sh"
[ $# -eq 5 ] || die "usage: $0 <ds> <S> <ef> <fixed|adaptive> <a|b>"
DS=$1; S=$2; E=$3; ARM=$4; BLK=$5
N=$(ds_n "$DS") || exit 1
DSN=${DS}_N$N; OUT=$RESULTS/shard/$DSN; mkdir -p "$OUT"
CELLS_ROOT=${CELLS_ROOT:-$RESULTS/cells}
f=$OUT/S${S}_ef${E}_${ARM}_${BLK}
[ -s "$f.tsv" ] && [ -s "$f.json" ] && exit 0
need_bin
NORM=$(ds_norm "$DS"); KI=$(k_i "$S"); IDX=$IDX_DIR/$DSN
case $ARM in
  fixed)    G=0 ;;
  adaptive) G=$($PY -c "
import json, sys
c = json.load(open(sys.argv[1]))[sys.argv[2]]
print(','.join('0' if g is None else f'{g:g}' for g in c['gamma_rank_shard']))" "$CELLS_ROOT/$DSN/calib_cells.json" "S${S}_E${E}") \
              || die "no calibrated gammas for S${S}_E${E} in $CELLS_ROOT/$DSN/calib_cells.json"
            [ "$S" = 1 ] && G=${G%%,*} ;;
  *) die "arm must be fixed or adaptive" ;;
esac
case $BLK in
  a) Q=(--nq 1000 --qoff 0 --gt "$IDX/gt300.ivec") ;;
  b) Q=(--nq "$(ds_nb "$DS")" --qoff 1500 --gt "$IDX/gt300_q1500.ivec") ;;
  *) die "block must be a or b" ;;
esac
# The stopping test compares against rank k_i (adaptive termination). With gamma 0 the rank has no effect.
"$BIN" search --shards "$IDX/S$S" --S "$S" --query "$RAW_DIR/$DS/query.fbin" --normalize "$NORM" --check_norm "$NORM" "${Q[@]}" \
  --K "$K" --ki "$KI" --rule_k "$KI" --ef "$E" --gamma "$G" --repeat 1 --warmup 100 --dump_shard "$f.tsv.part" > "$f.json.part" 2> "$f.err" \
  && $PY -c "import json,sys; json.load(open(sys.argv[1]))" "$f.json.part" 2>/dev/null \
  || { echo "!! run failed $f: $(tail -n 2 "$f.err")" >&2; rm -f "$f.json.part" "$f.tsv.part"; exit 1; }
mv "$f.tsv.part" "$f.tsv" && mv "$f.json.part" "$f.json"
$PY -c "
import json, sys
d = json.load(open(sys.argv[1]))[0]
print('  shard %-8s S=%-2s ef=%-3s %-8s %s recall=%.5f ndis=%.1f' % (sys.argv[2], d['S'], d['ef'], sys.argv[3], sys.argv[4], d['recall'], d['ndis_per_query']))" \
  "$f.json" "$DS" "$ARM" "$BLK"
