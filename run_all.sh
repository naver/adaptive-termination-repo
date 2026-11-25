#!/usr/bin/env bash
# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#
# Reproduce the experiments end to end, or any part of it.
#
#   run_all.sh [--steps build,data,core,extras,cpu,figures] [DS ...]
#
#   build    faiss/build_faiss.sh + bench/build_bench.sh  (patched Faiss v1.15.0 and the benchmark driver)
#   data     data/prepare_all.sh DS...                    (download, convert, exact ground truth, shard split, HNSW indexes)
#   core     experiments/run_core.sh DS                   (calibration, gamma grids, same-recall matching, rank-K termination,
#                                                          tolerance dumps, bootstrap verdicts)
#   extras   experiments/run_extras.sh                    (oracle, per-shard dumps, approximate ground truth and its verdicts,
#                                                          K axis, fewer calibration queries, single-index gamma on shards,
#                                                          rank at K=100, Patience; T2I and
#                                                          MS MARCO plus the four-dataset approx and ncalib cells)
#   cpu      experiments/run_cpu.sh                       (CPU-time sessions; T2I and MS MARCO; needs a quiet dedicated machine)
#   figures  figures/make_all.sh with RESULTS=$RESULTS     (figures and checked numbers from the new results)
#
# DS defaults to all four datasets (t2i msmarco glove sift). Every step skips outputs that already exist.
# Default steps: build,data,core,extras,cpu,figures. Settings (locations, JOBS, PIN ...): experiments/config.sh.
# Expect days of machine time and several hundred GB of disk for the 10M datasets; see README.md "Requirements".
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/experiments/config.sh"

STEPS="build,data,core,extras,cpu,figures"
DSS=()
while [ $# -gt 0 ]; do
  case $1 in
    --steps) STEPS=$2; shift 2;;
    -h|--help) sed -n '6,23p' "$0"; exit 0;;
    *) DSS+=("$1"); shift;;
  esac
done
[ ${#DSS[@]} -gt 0 ] || DSS=(t2i msmarco glove sift)
has() { case ",$STEPS," in *",$1,"*) return 0;; *) return 1;; esac; }
log() { echo "=== $(date '+%F %T') $*"; }

if has build; then
  log "build"
  "$REPRO/faiss/build_faiss.sh"
  "$REPRO/bench/build_bench.sh"
fi
if has data; then
  log "data ${DSS[*]}"
  "$REPRO/data/prepare_all.sh" "${DSS[@]}"
fi
if has core; then
  for ds in "${DSS[@]}"; do log "core $ds"; "$REPRO/experiments/run_core.sh" "$ds"; done
fi
if has extras; then
  log "extras"
  "$REPRO/experiments/run_extras.sh"
fi
if has cpu; then
  cpu_dss=()
  for ds in "${DSS[@]}"; do case $ds in t2i|msmarco) cpu_dss+=("$ds");; esac; done
  if [ ${#cpu_dss[@]} -gt 0 ]; then log "cpu ${cpu_dss[*]}"; "$REPRO/experiments/run_cpu.sh" "${cpu_dss[@]}"; fi
fi
if has figures; then
  log "figures from $RESULTS"
  RESULTS="$RESULTS" CHECK=0 "$REPRO/figures/make_all.sh"
fi
log "done"
