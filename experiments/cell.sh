#!/usr/bin/env bash
# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#
# One cell (dataset, S, E) of the main experiment. Results go to $RESULTS/cells/<dsn>/ (rank mode rule) or
# $RESULTS/cells/<dsn>/dabs/; an existing result is skipped.
#
#   grid    test block a (#0-999, exact ground truth): one gamma = 0 run with --trace, replayed over the gamma grid
#           (lib/grid_replay.py). grid/S{S}_E{E}.json = {S, nq, ki, rule_k, ef, curve: [{gamma, recall, ndis}],
#           bench: the gamma = 0 run itself = the fixed-ef point}
#   table   calibration queries (#1000-1499): gamma = 0 record -> lib/pred_calib.py cell -> per-shard gamma_i,
#           stored under key S{S}_E{E} in calib_cells.json (no test-query run)
#   calib   table, then a direct run on block a with the calibrated per-shard gammas -> calib/S{S}_E{E}.json
#   direct  a direct run on block a with one given gamma (replay check) -> direct/S{S}_E{E}_g{gamma}.json
#
# RANK_MODE is the rank the rule compares against: rule = k_i (adaptive termination, default), dabs = K (rank-K termination;
# calib_dabs.sh runs its table).
# Traversal records go to $TRACE_DIR/<dsn>/<mode>/ (eval_* = block a, proxy_* = calibration queries).
# K (config.sh) other than 20 writes to $RESULTS/cells_K<K>/ and $TRACE_DIR/K<K>/ and replays over RANK_GAMMAS
# (run_extras.sh rank100 runs the grid cells at K=100).
#
# Usage: [RANK_MODE=rule|dabs] cell.sh <grid|table|calib|direct> <ds> <S> <E> [gamma]
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/config.sh"
[ $# -ge 4 ] || die "usage: [RANK_MODE=rule|dabs] cell.sh <grid|table|calib|direct> <ds> <S> <E> [gamma]"
MODE=$1; DS=$2; S=$3; E=$4; G=${5:-}
RANK_MODE=${RANK_MODE:-rule}
N=$(ds_n "$DS") || exit 1
DSN=${DS}_N${N}
BASE=$RESULTS/cells/$DSN
TROOT=$TRACE_DIR; RG=()
if [ "$K" != 20 ]; then BASE=$RESULTS/cells_K$K/$DSN; TROOT=$TRACE_DIR/K$K; RG=(--gammas "$RANK_GAMMAS"); fi
case $RANK_MODE in
  rule) OUT=$BASE ;;
  dabs) OUT=$BASE/$RANK_MODE ;;
  *) die "RANK_MODE must be rule or dabs (got '$RANK_MODE')" ;;
esac
TD=$TROOT/$DSN/$RANK_MODE
IDX=$IDX_DIR/$DSN
QF=$RAW_DIR/$DS/query.fbin
NORM=$(ds_norm "$DS")
KI=$(k_i "$S"); RK=$(rank_of "$RANK_MODE" "$S")
CELLS=$OUT/calib_cells.json; KEY=S${S}_E${E}
mkdir -p "$OUT" "$TD"

need_file() { [ -s "$1" ] || die "$2 not found: $1${3:+ ($3)}"; }
need_inputs() {
  need_bin
  need_file "$QF" "query vectors" "see data/"
  local s
  for ((s = 0; s < S; s++)); do
    need_file "$IDX/S$S/shard$s.index" "shard index" "build the S=$S shards with data/"
    need_file "$IDX/S$S/shard$s.sel" "shard id list" "build the S=$S shards with data/"
  done
}
# calibrated per-shard gammas of this cell as the bench --gamma argument (None = rule off -> 0)
gammas_of() {
  $PY -c "import json,sys;c=json.load(open(sys.argv[1]))[sys.argv[2]];print(','.join('0' if g is None else f'{g:g}' for g in c['gamma_rank_shard']))" "$1" "$2"
}
has_key() { [ -s "$1" ] && $PY -c "import json,sys;sys.exit(0 if sys.argv[2] in json.load(open(sys.argv[1])) else 1)" "$1" "$2"; }
common=(--shards "$IDX/S$S" --S "$S" --query "$QF" --normalize "$NORM" --check_norm "$NORM"
        --K "$K" --ki "$KI" --rule_k "$RK" --ef "$E")

# direct run on test block a -> $1 (bench JSON); remaining args go to the bench
run_block_a() {
  local f=$1; shift
  need_file "$IDX/gt300.ivec" "ground truth of test block a"
  "$BIN" search "${common[@]}" --nq 1000 --qoff 0 --gt "$IDX/gt300.ivec" "$@" --repeat 1 --warmup 100 \
    > "$f.part" 2> "$f.err" && $PY -c "import json,sys;json.load(open(sys.argv[1]))" "$f.part" 2>/dev/null \
    || { echo "!! bench failed: $f" >&2; tail -n 3 "$f.err" >&2; rm -f "$f.part"; return 1; }
  mv "$f.part" "$f"; rm -f "$f.err"
}

# calibration table entry of this cell (calibration queries #1000-1499)
table() {
  has_key "$CELLS" "$KEY" && return 0
  need_inputs
  need_file "$IDX/gt300_proxy.ivec" "ground truth of the calibration queries"
  local tb=$TD/proxy_S${S}_E${E}.bin
  if [ ! -s "$tb" ]; then
    "$BIN" search "${common[@]}" --nq 500 --qoff 1000 --gt "$IDX/gt300_proxy.ivec" --gamma 0 --repeat 1 --warmup 100 \
      --trace "$tb.part" > /dev/null 2> "$tb.err" && mv "$tb.part" "$tb" \
      || { echo "!! calibration record failed: $RANK_MODE $DSN $KEY" >&2; tail -n 3 "$tb.err" >&2; rm -f "$tb.part"; return 1; }
    rm -f "$tb.err"
  fi
  $PY "$REPRO/lib/pred_calib.py" cell --ds "$DS" --N "$N" --trace "$tb" --gt "$IDX/gt300_proxy.ivec" \
    --key "$KEY" --out "$CELLS" --gammas "$CALIB_GAMMAS" --K "$K" > /dev/null \
    || { echo "!! pred_calib failed: $RANK_MODE $DSN $KEY" >&2; return 1; }
}

case $MODE in
grid)
  f=$OUT/grid/S${S}_E${E}.json; [ -s "$f" ] && exit 0
  mkdir -p "$OUT/grid"
  need_inputs
  need_file "$IDX/gt300.ivec" "ground truth of test block a"
  t=$TD/eval_S${S}_E${E}.bin; b=$TD/eval_S${S}_E${E}.json
  if ! { [ -s "$b" ] && [ -s "$t" ]; }; then
    "$BIN" search "${common[@]}" --nq 1000 --qoff 0 --gt "$IDX/gt300.ivec" --gamma 0 --repeat 1 --warmup 100 \
      --trace "$t.part" > "$b.part" 2> "$b.err" && mv "$t.part" "$t" && mv "$b.part" "$b" \
      || { echo "!! record failed: $RANK_MODE $DSN S=$S E=$E" >&2; tail -n 3 "$b.err" >&2; rm -f "$t.part" "$b.part"; exit 1; }
    rm -f "$b.err"
  fi
  $PY "$REPRO/lib/grid_replay.py" --trace "$t" --gt "$IDX/gt300.ivec" --K "$K" --bench "$b" ${RG[@]+"${RG[@]}"} > "$f.part" && mv "$f.part" "$f" \
    || { echo "!! replay failed: $RANK_MODE $DSN S=$S E=$E" >&2; rm -f "$f.part"; exit 1; }
  $PY -c "import json,sys;d=json.load(open(sys.argv[1]));c=d['curve'];print('  grid %s %s S=%s E=%s  gamma=0 %.4f/%.0f  gamma=.005 %.4f/%.0f' % (sys.argv[2],sys.argv[3],d['S'],d['ef'],c[0]['recall'],c[0]['ndis'],c[1]['recall'],c[1]['ndis']))" "$f" "$RANK_MODE" "$DSN"
  ;;
table)
  table || exit 1
  ;;
calib)
  f=$OUT/calib/S${S}_E${E}.json; [ -s "$f" ] && exit 0
  mkdir -p "$OUT/calib"
  table || exit 1
  GL=$(gammas_of "$CELLS" "$KEY") || die "no calibration entry $KEY in $CELLS"
  [ "$S" = 1 ] && GL=${GL%%,*}            # one shard: a single value
  run_block_a "$f" --gamma "$GL" || exit 1
  $PY -c "import json,sys;d=json.load(open(sys.argv[1]))[0];print('  calib %s %s %s gamma_i=[%s] %.4f/%.0f' % (sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5][:60],d['recall'],d['ndis_per_query']))" "$f" "$RANK_MODE" "$DSN" "$KEY" "$GL"
  ;;
direct)
  [ -n "$G" ] || die "direct needs a gamma"
  f=$OUT/direct/S${S}_E${E}_g${G}.json; [ -s "$f" ] && exit 0
  mkdir -p "$OUT/direct"
  need_inputs
  run_block_a "$f" --gamma "$G" || exit 1
  ;;
*) die "unknown mode '$MODE' (grid | table | calib | direct)" ;;
esac
