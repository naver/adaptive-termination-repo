#!/usr/bin/env bash
# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#
# Shard indexes for calibration before serving (the `corpus` arm, lib/holdout.py), in two phases per S:
#
#   part   shard split (same seed-0 split as build_indexes.sh), then per shard s: 500 held-out rows set aside, the
#          partial graph built from the rest, the held-out rows' exact ground truth against the rest (bench gt, global ids)
#            $IDX_DIR/<ds>_N<N>/S<S>/part<s>/{shard0.index, shard0.sel}   partial graph as a one-shard directory
#            $IDX_DIR/<ds>_N<N>/S<S>/part<s>/{hold.fbin, hold.sel, hold_gt.ivec, hold.json}
#            $IDX_DIR/<ds>_N<N>/S<S>/.part
#   final  the held-out rows inserted into the partial graph (bench add): the served graph, rows = rest, then held out
#            $IDX_DIR/<ds>_N<N>/S<S>/shard<s>.{index,sel}, .done        (the layout every other script reads)
#
# Build times go to S<S>/build.json (bench build and bench add lines, one per shard). Run `final` after the held-out
# records (experiments/corpus_cell.sh) are written: the partial graph is what they search. Point IDX_DIR at a directory
# of its own (e.g. $WORK/idx_holdout): the final graphs differ from those of build_indexes.sh (insertion order).
#
# Usage: data/build_holdout.sh <part|final> DS [S ...]      (default S list: config.sh ds_ss)
set -euo pipefail
. "$(dirname "$0")/common.sh"
need_bin

PHASE=${1:?usage: data/build_holdout.sh <part|final> DS [S ...]}; shift
DS=${1:?usage: data/build_holdout.sh <part|final> DS [S ...]}; shift
SS=${*:-$(ds_ss "$DS")}
N=$(data_n "$DS"); NORM=$(data_norm "$DS")
RAW="$RAW_DIR/$DS"; OUT="$IDX_DIR/${DS}_N${N}"
HOLD="$REPRO/lib/holdout.py"
mkdir -p "$OUT"

part() {
  local S=$1 D="$OUT/S$1" s P
  [ -f "$D/.part" ] && { log "S=$S partial graphs already built"; return 0; }
  [ -s "$RAW/base.fbin" ] || die "$RAW/base.fbin missing (data/convert.sh $DS)"
  mkdir -p "$D"
  # the split's id lists are kept as shard<s>.orig.sel: shard<s>.sel is the final graph's (rest, then held out)
  local need_split=0
  for s in $(seq 0 $((S - 1))); do
    [ -f "$D/part$s/hold.json" ] && continue
    { [ -f "$D/shard$s.fbin" ] && [ -f "$D/shard$s.orig.sel" ]; } || need_split=1
  done
  if [ "$need_split" = 1 ]; then
    log "S=$S shard split"
    "$PY" "$REPRO/lib/shard_split.py" --base "$RAW/base.fbin" --N "$N" --S "$S" --out "$D" --chunk "${SPLIT_CHUNK:-2000000}" \
        $([ "$NORM" = 1 ] && echo --normalize)
    for s in $(seq 0 $((S - 1))); do mv "$D/shard$s.sel" "$D/shard$s.orig.sel"; done
  fi
  for s in $(seq 0 $((S - 1))); do
    P="$D/part$s"
    if [ ! -f "$P/hold.json" ]; then
      "$PY" "$HOLD" split --fbin "$D/shard$s.fbin" --sel "$D/shard$s.orig.sel" --S "$S" --s "$s" --out "$P.tmp"
      rm -rf "$P"; mv "$P.tmp" "$P"
    fi
    rm -f "$D/shard$s.fbin"
    if [ ! -f "$P/shard0.index" ]; then
      log "S=$S shard$s partial graph"
      OMP_NUM_THREADS=$THREADS "$BIN" build --base "$P/shard0.fbin" --out "$P/shard0.index.part" \
          --M "$M_HNSW" --efc "$EFC_HNSW" | sed 's/^{/{"step": "part", "shard": '"$s"', /' >> "$D/build.json"
      mv "$P/shard0.index.part" "$P/shard0.index"
    fi
    if [ ! -s "$P/hold_gt.ivec" ]; then
      OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS \
        "$BIN" gt --base "$P/shard0.fbin" --query "$P/hold.fbin" --k 300 --normalize 0 --out "$P/hold_gt.local.ivec"
      "$PY" "$HOLD" gtmap --gt "$P/hold_gt.local.ivec" --sel "$P/shard0.sel" --out "$P/hold_gt.ivec.part"
      mv "$P/hold_gt.ivec.part" "$P/hold_gt.ivec"; rm -f "$P/hold_gt.local.ivec"
    fi
    rm -f "$P/shard0.fbin"                 # the partial graph holds the rest's vectors
  done
  touch "$D/.part"
  log "S=$S partial graphs done: $(du -sh "$D" | cut -f1)"
}

final() {
  local S=$1 D="$OUT/S$1" s P
  [ -f "$D/.done" ] && { log "S=$S final graphs already built"; return 0; }
  [ -f "$D/.part" ] || die "S=$S: no partial graphs (data/build_holdout.sh part $DS $S)"
  for s in $(seq 0 $((S - 1))); do
    P="$D/part$s"
    [ -f "$D/shard$s.index" ] && [ -f "$D/shard$s.sel" ] && continue
    log "S=$S shard$s final graph (+ held-out rows)"
    OMP_NUM_THREADS=$THREADS "$BIN" add --index "$P/shard0.index" --base "$P/hold.fbin" --out "$D/shard$s.index.part" \
        | sed 's/^{/{"step": "add", "shard": '"$s"', /' >> "$D/build.json"
    "$PY" "$HOLD" finalsel --part "$P" --out "$D/shard$s.sel.part"
    mv "$D/shard$s.index.part" "$D/shard$s.index"; mv "$D/shard$s.sel.part" "$D/shard$s.sel"
  done
  touch "$D/.done"
  log "S=$S final graphs done: $(du -sh "$D" | cut -f1)"
}

log "=== holdout $PHASE $DS N=$N S=[$SS] M=$M_HNSW efC=$EFC_HNSW normalize=$NORM -> $OUT"
for S in $SS; do
  case $PHASE in part) part "$S" ;; final) final "$S" ;; *) die "unknown phase '$PHASE' (part | final)" ;; esac
done
