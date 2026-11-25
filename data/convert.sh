#!/usr/bin/env bash
# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#
# Convert the downloads in $DL_DIR/<ds>/ into $RAW_DIR/<ds>/{base,query}.fbin (raw, not normalized). Idempotent.
# Needs: numpy; pyarrow + pandas for msmarco; h5py for glove and sift.
#
# Usage: data/convert.sh [t2i] [msmarco] [glove] [sift] [deep] [t2i100m]   (default: the four 10M/1M datasets)
#   After a successful run the downloads are no longer needed: rm -r "$WORK/download/<ds>"
#   (keep $WORK/download/msmarco/qrels.dev.small.tsv if you want to run verify_msmarco_qrels.py).
set -euo pipefail
. "$(dirname "$0")/common.sh"

for ds in ${*:-$DATASETS_ALL}; do
  out="$RAW_DIR/$ds"
  if [ -s "$out/base.fbin" ] && [ -s "$out/query.fbin" ]; then log "$ds: $out/{base,query}.fbin exist, skip"; continue; fi
  log "=== convert $ds -> $out"
  d="$DL_DIR/$ds"
  case $ds in
    t2i)
      "$PY" "$DATA_DIR/convert.py" t2i --src "$d/base.1B.first10M.fbin" --query "$d/query.public.100K.fbin" --out "$out" ;;
    t2i100m)
      "$PY" "$DATA_DIR/convert.py" t2i --src "$d/base.1B.first100M.fbin" --query "$d/query.public.100K.fbin" --out "$out" \
        --n 100000000 --dataset t2i100m ;;
    deep)
      "$PY" "$DATA_DIR/convert.py" t2i --src "$d/base.1B.first100M.fbin" --query "$d/query.public.10K.fbin" --out "$out" \
        --n 100000000 --d 96 --dataset deep ;;
    msmarco)
      [ -f "$d/corpus/.extracted" ] && [ -f "$d/queries/.extracted" ] || die "run data/download.sh msmarco first"
      pkl=$(find "$d/queries" -name embedding.pkl | head -1)
      [ -n "$pkl" ] || die "embedding.pkl not found under $d/queries"
      "$PY" "$DATA_DIR/convert.py" msmarco --parquet-dir "$d/corpus" --query-pkl "$pkl" --out "$out" ;;
    glove)
      "$PY" "$DATA_DIR/convert.py" hdf5 --src "$d/glove-200-angular.hdf5" --expect-distance angular --out "$out" ;;
    sift)
      "$PY" "$DATA_DIR/convert.py" hdf5 --src "$d/sift-128-euclidean.hdf5" --expect-distance euclidean --out "$out" ;;
    *) die "unknown dataset '$ds'" ;;
  esac
done
