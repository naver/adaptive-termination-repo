#!/usr/bin/env bash
# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#
# Arrival-time oracle for one configuration (fig-waste): test block a (queries #0-999), exact ground truth.
#
# The oracle replays the fixed-ef trajectory and records, per shard, the distance computations spent up to the instant
# the last kept ground-truth neighbour entered the result heap. The JSON carries waste_split (empty vs confirm) and
# empty_shards, which fig-waste plots; the TSV holds per-query ref_ndis / oracle_ndis / hits.
# The oracle is deterministic, so the machine it runs on does not matter.
#
# Output: $RESULTS/oracle/<ds>_N<N>/S{S}_ef{ef}.{json,tsv}; skipped if both exist.
# Usage:  experiments/oracle_cell.sh <ds> <S> <ef>
set -uo pipefail
. "$(dirname "$0")/config.sh"
[ $# -eq 3 ] || die "usage: $0 <ds> <S> <ef>"
DS=$1; S=$2; E=$3
N=$(ds_n "$DS") || exit 1
DSN=${DS}_N$N; OUT=$RESULTS/oracle/$DSN; mkdir -p "$OUT"
f=$OUT/S${S}_ef${E}
[ -s "$f.json" ] && [ -s "$f.tsv" ] && exit 0
need_bin
NORM=$(ds_norm "$DS"); KI=$(k_i "$S"); IDX=$IDX_DIR/$DSN
"$BIN" oracle --shards "$IDX/S$S" --S "$S" --query "$RAW_DIR/$DS/query.fbin" --nq 1000 --normalize "$NORM" --check_norm "$NORM" \
  --gt "$IDX/gt300.ivec" --K "$K" --ki "$KI" --ef "$E" --dump "$f.tsv.part" > "$f.json.part" 2> "$f.err" \
  && $PY -c "import json,sys; json.load(open(sys.argv[1]))" "$f.json.part" 2>/dev/null \
  || { echo "!! oracle failed $DS S=$S ef=$E: $(tail -n 2 "$f.err")" >&2; rm -f "$f.json.part" "$f.tsv.part"; exit 1; }
[ "$(wc -l < "$f.tsv.part" | tr -d ' ')" = 1001 ] || { echo "!! oracle dump has the wrong length: $f.tsv.part" >&2; exit 1; }
mv "$f.tsv.part" "$f.tsv" && mv "$f.json.part" "$f.json"
$PY -c "
import json, sys
d = json.load(open(sys.argv[1]))
print('  oracle %-8s S=%-2s ef=%-4s recall %.4f  fixed %9.0f  oracle %9.0f  ratio %.3f' % (sys.argv[2], d['S'], d['ef'], d['recall'],
      d['ref_ndis_per_query'], d['oracle_ndis_per_query'], d['ratio']))" "$f.json" "$DS"
