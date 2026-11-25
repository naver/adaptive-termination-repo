#!/usr/bin/env bash
# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#
# Calibration before serving (the `corpus` arm) of one cell (dataset, S, E): every shard searches its own 500 held-out
# rows against its partial graph (data/build_holdout.sh part), alone, with the rank and result count of the real S
# (k_i(S)), and chooses its gamma_i from that record (lib/pred_calib.py shard). The entry goes under key S{S}_E{E} in
# $RESULTS/cells/<dsn>/corpus/calib_cells.json, in the form of the rule's table (gamma_rank_shard), which dump_cell.sh
# reads for the arm `corpus`. Records: $TRACE_DIR/<dsn>/corpus/S{S}_E{E}_s{s}.bin. An existing entry is skipped.
#
# Usage: corpus_cell.sh <ds> <S> <E>
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/config.sh"
[ $# -eq 3 ] || die "usage: corpus_cell.sh <ds> <S> <E>"
DS=$1; S=$2; E=$3
N=$(ds_n "$DS") || exit 1
DSN=${DS}_N${N}; KEY=S${S}_E${E}
D=$IDX_DIR/$DSN/S$S
OUT=$RESULTS/cells/$DSN/corpus; CELLS=$OUT/calib_cells.json
TD=$TRACE_DIR/$DSN/corpus
NORM=$(ds_norm "$DS"); KI=$(k_i "$S")
mkdir -p "$OUT" "$TD"
[ -s "$CELLS" ] && $PY -c "import json,sys;sys.exit(0 if sys.argv[2] in json.load(open(sys.argv[1])) else 1)" "$CELLS" "$KEY" && exit 0
need_bin

traces=(); gts=()
for ((s = 0; s < S; s++)); do
  P=$D/part$s
  for f in shard0.index shard0.sel hold.fbin hold_gt.ivec; do
    [ -s "$P/$f" ] || die "$P/$f not found (data/build_holdout.sh part $DS $S)"
  done
  t=$TD/S${S}_E${E}_s$s.bin
  if [ ! -s "$t" ]; then
    # the held-out rows are already normalized (cut from the split's shard vectors)
    "$BIN" search --shards "$P" --S 1 --query "$P/hold.fbin" --nq 500 --qoff 0 --gt "$P/hold_gt.ivec" \
      --normalize 0 --check_norm "$NORM" --K "$K" --ki "$KI" --rule_k "$KI" --ef "$E" --gamma 0 \
      --repeat 1 --warmup 100 --trace "$t.part" > /dev/null 2> "$t.err" && mv "$t.part" "$t" \
      || { echo "!! held-out record failed: $DSN $KEY shard $s" >&2; tail -n 3 "$t.err" >&2; rm -f "$t.part"; exit 1; }
    rm -f "$t.err"
  fi
  traces+=("$t"); gts+=("$P/hold_gt.ivec")
done
$PY "$REPRO/lib/pred_calib.py" shard --trace "${traces[@]}" --gt "${gts[@]}" --key "$KEY" --out "$CELLS" \
  --gammas "$CALIB_GAMMAS" --K "$K" | sed "s/^/  $DS /" \
  || die "pred_calib shard failed: $DSN $KEY"
