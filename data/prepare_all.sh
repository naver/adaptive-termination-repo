#!/usr/bin/env bash
# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#
# Prepare datasets end to end: download -> convert -> exact ground truth -> shard split + HNSW indexes.
# Every step skips what already exists, so the same command resumes after an interruption.
# Disk, memory and time per dataset: data/DATASETS.md. Needs the bench binary (bench/build_bench.sh) for gt and index.
#
# Usage: data/prepare_all.sh [--steps download,convert,gt,index] [--S "1 5"] [DS ...]
#   DS: t2i msmarco glove sift (default: all four), deep t2i100m (the 100M sets, ~40 / 80 GB downloads), or synth (the tiny smoke-test set from make_synthetic.py; not part of the experiments)
#   --S overrides the S list of every selected dataset (default: config.sh ds_ss)
#   Environment: WORK / RAW_DIR / IDX_DIR / DL_DIR (locations), THREADS, CLEAN_DOWNLOADS=1 (delete $DL_DIR/<ds> after convert)
#
# Example (the two 1M datasets, a few GB):   data/prepare_all.sh glove sift
set -euo pipefail
. "$(dirname "$0")/common.sh"

STEPS="download,convert,gt,index"
SLIST=""
DSS=()
while [ $# -gt 0 ]; do
  case $1 in
    --steps) STEPS=$2; shift 2 ;;
    --S) SLIST=$2; shift 2 ;;
    -h|--help) sed -n '6,16p' "$0"; exit 0 ;;
    *) DSS+=("$1"); shift ;;
  esac
done
[ ${#DSS[@]} -gt 0 ] || DSS=($DATASETS_ALL)
has_step() { case ",$STEPS," in *",$1,"*) return 0;; *) return 1;; esac; }

for ds in "${DSS[@]}"; do
  case $ds in t2i|msmarco|glove|sift|deep|t2i100m|synth) ;; *) die "unknown dataset '$ds'";; esac
  log "##### $ds (steps: $STEPS)"
  if [ "$ds" = synth ]; then
    if has_step download || has_step convert; then "$PY" "$DATA_DIR/make_synthetic.py" --out "$RAW_DIR/synth"; fi
  else
    if has_step download; then "$DATA_DIR/download.sh" "$ds"; fi
    if has_step convert; then
      "$DATA_DIR/convert.sh" "$ds"
      if [ "${CLEAN_DOWNLOADS:-0}" = 1 ] && [ -d "$DL_DIR/$ds" ]; then
        # keep the 143 KB qrels file for the optional MS MARCO check
        find "$DL_DIR/$ds" -mindepth 1 -maxdepth 1 ! -name qrels.dev.small.tsv -exec rm -rf {} +
      fi
    fi
  fi
  if has_step gt; then "$DATA_DIR/ground_truth.sh" "$ds"; fi
  if has_step index; then "$DATA_DIR/build_indexes.sh" "$ds" $SLIST; fi
done
log "all done"
