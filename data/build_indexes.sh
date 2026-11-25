#!/usr/bin/env bash
# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#
# Shard split + one HNSW index per shard (M=40, efConstruction=200), for each S of a dataset:
#   $IDX_DIR/<ds>_N<N>/S<S>/shard<s>.sel     global row ids of the shard, in shard order (lib/shard_split.py)
#   $IDX_DIR/<ds>_N<N>/S<S>/shard<s>.index   Faiss IndexHNSWFlat over the shard's rows (bench build)
#   $IDX_DIR/<ds>_N<N>/S<S>/build.json       one JSON line per shard build (build_s, n, d, M, efc)
#   $IDX_DIR/<ds>_N<N>/S<S>/.done
# Rows are dealt to shards by a seed-0 random permutation (equal sizes). For every dataset but SIFT and DEEP the shard rows are
# normalized to unit length during the split, so the build itself does not normalize again. The shard vector copies
# (shard<s>.fbin) are deleted once their index is written: the index holds the vectors, and keeping the copies would
# double the disk use (36 GB per S on MS MARCO).
#
# Usage: data/build_indexes.sh DS [S ...]          (default S list: config.sh ds_ss -- t2i/msmarco 1 5 15 30 50, glove/sift 1 5 15,
#                                                   deep/t2i100m 5 15)
#   THREADS (config.sh) sets OpenMP threads for the build. N=... / NORM=... override the dataset defaults.
#   SPLIT_CHUNK=2000000 rows per read in the split (lower it to save memory: ~3 x chunk x d x 4 bytes).
set -euo pipefail
. "$(dirname "$0")/common.sh"
need_bin

DS=${1:?usage: data/build_indexes.sh DS [S ...]}; shift
SS=${*:-$(ds_ss "$DS")}
N=$(data_n "$DS"); NORM=$(data_norm "$DS")
RAW="$RAW_DIR/$DS"; OUT="$IDX_DIR/${DS}_N${N}"
[ -s "$RAW/base.fbin" ] || die "$RAW/base.fbin missing (data/convert.sh $DS)"
SPLIT="$REPRO/lib/shard_split.py"
[ -f "$SPLIT" ] || die "$SPLIT not found"
mkdir -p "$OUT"
log "=== $DS N=$N S=[$SS] M=$M_HNSW efC=$EFC_HNSW normalize=$NORM -> $OUT"

for S in $SS; do
  D="$OUT/S$S"
  if [ -f "$D/.done" ]; then log "S=$S already built"; continue; fi
  mkdir -p "$D"
  # Split (again) if no split finished here, or if some shard lacks both its index and its vector copy. The split is
  # deterministic, so redoing it rewrites identical files.
  need_split=0
  [ -f "$D/.split" ] || need_split=1
  for s in $(seq 0 $((S - 1))); do
    if [ ! -f "$D/shard$s.index" ] && [ ! -f "$D/shard$s.fbin" ]; then need_split=1; break; fi
    [ -f "$D/shard$s.sel" ] || need_split=1
  done
  if [ "$need_split" = 1 ]; then
    log "S=$S shard split"
    "$PY" "$SPLIT" --base "$RAW/base.fbin" --N "$N" --S "$S" --out "$D" --chunk "${SPLIT_CHUNK:-2000000}" \
        $([ "$NORM" = 1 ] && echo --normalize)
    touch "$D/.split"
  fi
  for s in $(seq 0 $((S - 1))); do
    if [ -f "$D/shard$s.index" ]; then rm -f "$D/shard$s.fbin"; continue; fi
    log "S=$S shard$s build"
    OMP_NUM_THREADS=$THREADS "$BIN" build --base "$D/shard$s.fbin" --out "$D/shard$s.index.part" \
        --M "$M_HNSW" --efc "$EFC_HNSW" >> "$D/build.json"
    mv "$D/shard$s.index.part" "$D/shard$s.index"
    rm -f "$D/shard$s.fbin"
  done
  touch "$D/.done"
  log "S=$S done: $(du -sh "$D" | cut -f1)"
done
du -sh "$OUT" >&2
