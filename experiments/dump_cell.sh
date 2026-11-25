#!/usr/bin/env bash
# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#
# One configuration x arm x query block, run with --dump (per-query hits and ndis) for the tolerance verdicts
# (lib/bootstrap.py). Output: $RESULTS/dumps/<ds>_S{S}_ef{E}_{arm}_{blk}.{tsv,json}; skipped when both exist.
#
# Arms (rank = the rank the rule compares against):
#   nat     gamma = 0: fixed ef at ef = E (the reference every other arm is paired with)
#   rule    adaptive termination, rank k_i, per-shard gamma_i from cells/<dsn>/calib_cells.json
#   dabs    rank K, gamma_i from cells/<dsn>/dabs/calib_cells.json
#   corpus  rank k_i, gamma_i from cells/<dsn>/corpus/calib_cells.json (calibrated before serving on held-out shard
#           rows, experiments/corpus_cell.sh; run on the final graphs of data/build_holdout.sh)
#   p125, p250  rank k_i, gamma_i from cells/<dsn>/ncalib/calib_cells_p<P>.json (calibrated on the first P of the 500
#           calibration queries; experiments/ncalib_cell.sh)
#   approx  rank k_i, gamma_i from approx/<dsn>/calib_cells_approx.json (calibrated against the approximate ground
#           truth, experiments/approx_cell.sh; judged by experiments/approx_verdicts.py)
#   s1g     rank k_i, the single gamma calibrated on the unsharded index (entry S1_E{E} of cells/<dsn>/calib_cells.json)
#           applied unchanged to every shard (experiments/run_extras.sh transfer)
# Blocks: a = #0-999 (gt300.ivec), b = #1500-... (gt300_q1500.ivec).
#
# At S = 1, rule and dabs are the same configuration (rank k_i = K and the same calibration entry, see
# calib_dabs.sh). If the dump of one of them already exists with the same gammas it is copied.
#
# Usage: dump_cell.sh <ds> <S> <E> <nat|rule|dabs|corpus|p125|p250|approx|s1g> <a|b>
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/config.sh"
[ $# -eq 5 ] || die "usage: dump_cell.sh <ds> <S> <E> <nat|rule|dabs|corpus|p125|p250|approx> <a|b>"
DS=$1; S=$2; E=$3; ARM=$4; BLK=$5
N=$(ds_n "$DS") || exit 1
DSN=${DS}_N${N}; KEY=S${S}_E${E}
DD=$RESULTS/dumps; mkdir -p "$DD"
f=$DD/${DS}_S${S}_ef${E}_${ARM}_${BLK}
[ -s "$f.tsv" ] && [ -s "$f.json" ] && exit 0
IDX=$IDX_DIR/$DSN; QF=$RAW_DIR/$DS/query.fbin; CB=$RESULTS/cells/$DSN
NORM=$(ds_norm "$DS"); KI=$(k_i "$S")

need_file() { [ -s "$1" ] || die "$2 not found: $1${3:+ ($3)}"; }
gammas_of() {   # <cells.json> <key> -> per-shard gammas as the bench --gamma argument (None = rule off -> 0)
  [ -s "$1" ] || die "calibration table not found: $1 (run the calibration first)"
  $PY -c "import json,sys;c=json.load(open(sys.argv[1])).get(sys.argv[2]);sys.exit('!! no entry %s in %s' % (sys.argv[2], sys.argv[1])) if c is None else print(','.join('0' if g is None else f'{g:g}' for g in c['gamma_rank_shard']))" "$1" "$2"
}
cells_of() { case $1 in rule) echo "$CB/calib_cells.json";; dabs|corpus) echo "$CB/$1/calib_cells.json";;
  p125|p250) echo "$CB/ncalib/calib_cells_$1.json";; approx) echo "$RESULTS/approx/$DSN/calib_cells_approx.json";; esac; }

case $ARM in
  nat)    RK=$(rank_of rule "$S"); G=0 ;;
  rule|dabs) RK=$(rank_of "$ARM" "$S"); G=$(gammas_of "$(cells_of "$ARM")" "$KEY") || exit 1 ;;
  corpus|p125|p250|approx) RK=$(rank_of rule "$S"); G=$(gammas_of "$(cells_of "$ARM")" "$KEY") || exit 1 ;;
  s1g)    RK=$(rank_of rule "$S"); G=$(gammas_of "$CB/calib_cells.json" "S1_E${E}") || exit 1; G=${G%%,*} ;;
  *) die "unknown arm '$ARM' (nat | rule | dabs | corpus | p125 | p250 | approx | s1g)" ;;
esac
[ "$S" = 1 ] && G=${G%%,*}            # one shard: a single value

# S = 1: rule and dabs coincide -- reuse an existing dump of the same configuration
if [ "$S" = 1 ] && case $ARM in rule|dabs) true;; *) false;; esac; then
  for other in rule dabs; do
    [ "$other" = "$ARM" ] && continue
    o=$DD/${DS}_S${S}_ef${E}_${other}_${BLK}
    [ -s "$o.tsv" ] && [ -s "$o.json" ] || continue
    [ "$(rank_of "$other" 1)" = "$RK" ] || continue
    OG=$(gammas_of "$(cells_of "$other")" "$KEY" 2>/dev/null) || continue
    [ "${OG%%,*}" = "$G" ] || continue
    cp "$o.tsv" "$f.tsv.part" && cp "$o.json" "$f.json.part" && mv "$f.tsv.part" "$f.tsv" && mv "$f.json.part" "$f.json"
    echo "  $DS S=1 ef=$E $ARM $BLK = $other (same configuration at S=1, copied)"
    exit 0
  done
fi

case $BLK in
  a) Q=(--nq 1000 --qoff 0 --gt "$IDX/gt300.ivec") ;;
  b) Q=(--nq "$(ds_nb "$DS")" --qoff 1500 --gt "$IDX/gt300_q1500.ivec") ;;
  *) die "unknown block '$BLK' (a | b)" ;;
esac
need_bin
need_file "$QF" "query vectors" "see data/"
need_file "${Q[5]}" "ground truth of block $BLK" "see data/"
for ((s = 0; s < S; s++)); do
  need_file "$IDX/S$S/shard$s.index" "shard index" "build the S=$S shards with data/"
  need_file "$IDX/S$S/shard$s.sel" "shard id list" "build the S=$S shards with data/"
done

# $PIN is empty by default; the per-query time columns of the dump are not used by the verdicts
$PIN "$BIN" search --shards "$IDX/S$S" --S "$S" --query "$QF" --normalize "$NORM" --check_norm "$NORM" "${Q[@]}" \
  --K "$K" --ki "$KI" --rule_k "$RK" --ef "$E" --gamma "$G" --repeat 1 --warmup 100 --dump "$f.tsv.part" \
  > "$f.json.part" 2> "$f.err" && $PY -c "import json,sys;json.load(open(sys.argv[1]))" "$f.json.part" 2>/dev/null \
  && [ -s "$f.tsv.part" ] || { echo "!! bench failed: $f" >&2; tail -n 3 "$f.err" >&2; rm -f "$f.tsv.part" "$f.json.part"; exit 1; }
mv "$f.tsv.part" "$f.tsv" && mv "$f.json.part" "$f.json"; rm -f "$f.err"
$PY -c "import json,sys;d=json.load(open(sys.argv[1]))[0];n=sum(1 for _ in open(sys.argv[2]))-1;print('  %-8s S=%-2s ef=%-3s %-6s %s  n=%d recall=%.5f ndis=%.1f  gamma=[%s]' % (sys.argv[3],d['S'],d['ef'],sys.argv[4],sys.argv[5],n,d['recall'],d['ndis_per_query'],sys.argv[6][:40]))" \
  "$f.json" "$f.tsv" "$DS" "$ARM" "$BLK" "$G"
