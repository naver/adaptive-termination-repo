#!/usr/bin/env bash
# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#
# Main experiment of one dataset, in order. Every step skips the outputs that already exist, so a rerun resumes.
# CPU-time measurements are not part of this driver (experiments/cpu.py).
#
#   1. adaptive termination (rank k_i): calibration + calibrated run on block a, E in $E_LIMITS          cell.sh calib
#      gamma-grid replay cells over the E grid                                                  cell.sh grid
#        T2I, MS MARCO (same-recall matching needs a wide E range):
#          S=1   32 48 64 100 150 200 300 400 600 800 1200 1600
#          S=5   16 24 32 48 64 100 150 200 300 400
#          S>=15 8 12 16 24 32 48 64 100 150 200 300 400
#        GloVe, SIFT: $E_LIMITS
#   2. same-recall matching (R 0.95, and 0.90 for the E list of experiments/patience.py; T2I and MS MARCO) and
#      fixed ef matched to the calibrated cells' recall for S in {1, 5, 15} (the deployed comparison; all datasets,
#      E in $E_LIMITS, E = 200 on the 100M sets)                                               match.py
#   3. rank-K termination, S > 1: calibration and gamma-grid cells, E in $E_LIMITS                    calib_dabs.sh, cell.sh grid
#      (at S = 1 rank-K termination is adaptive termination)
#   4. tolerance dumps, E in $E_LIMITS, blocks a and b: nat, rule; dabs (S > 1)                   dump_cell.sh
#   5. paired-bootstrap verdicts over all dumps in $RESULTS/dumps                                lib/bootstrap.py
# Independent cells run $JOBS at a time (xargs -P). Each cell loads its shards, so size JOBS to the memory.
#
# deep, t2i100m (100M scale): only adaptive termination against fixed ef -- steps 1 (E grid = $E_LIMITS), 2 (fixed ef
# matched to the deployed recall at E = 200), 4 (nat, rule) and 5 (verdicts); no rank-K termination, no same-recall matching.
#
# CORE_E overrides the E values of steps 1, 2, 3, 4 (default $E_LIMITS), e.g. CORE_E="100 200" for a smoke test.
# CORE_S overrides the S values (default config.sh ds_ss), e.g. to run the S values of one dataset on separate machines
# that each hold one S's shards; RESULTS of such runs can be merged afterwards (calib_cells.json keys are per S and E).
#
# Usage: [CORE_E="..."] [CORE_S="..."] run_core.sh <t2i|msmarco|glove|sift|deep|t2i100m|synth>
set -uo pipefail
EXP=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$EXP/config.sh"
[ $# -eq 1 ] || die "usage: [CORE_E=\"...\"] [CORE_S=\"...\"] run_core.sh <t2i|msmarco|glove|sift|deep|t2i100m|synth>"
DS=$1
N=$(ds_n "$DS") || exit 1
DSN=${DS}_N${N}; SS=${CORE_S:-$(ds_ss "$DS")}; IDX=$IDX_DIR/$DSN
E_LIMITS=${CORE_E:-$E_LIMITS}
export REPRO WORK RAW_DIR IDX_DIR TRACE_DIR RESULTS BIN PY JOBS K PIN
export "N_$(echo "$DS" | tr '[:lower:]' '[:upper:]')=$N"    # shardlib.ds_N for datasets other than the six (synth)
# deployed comparison: every E on the datasets up to 10M vectors, E = 200 on the 100M sets
if ds_rule_only "$DS"; then case " $E_LIMITS " in *" 200 "*) TAB_E=200;; *) TAB_E=${E_LIMITS%% *};; esac
else TAB_E=$E_LIMITS; fi

case $DS in t2i|msmarco) WIDE=1;; *) WIDE=0;; esac      # same-recall matching over a wide E grid
egrid() {
  if [ $WIDE = 0 ]; then echo "$E_LIMITS"; return; fi
  case $1 in
    1) echo "32 48 64 100 150 200 300 400 600 800 1200 1600" ;;
    5) echo "16 24 32 48 64 100 150 200 300 400" ;;
    *) echo "8 12 16 24 32 48 64 100 150 200 300 400" ;;
  esac
}
log() { echo "[$DS $(date +%m-%d\ %H:%M:%S)] $*"; }
# run lines of arguments through a script, $JOBS at a time; abort the driver if any cell failed
par() { xargs -P "$JOBS" -L 1 "$@" || die "a cell of this step failed (see the messages above); rerun to resume"; }

# ---- inputs
need() { [ -s "$1" ] || die "$2 not found: $1 (see data/)"; }
need_bin
need "$RAW_DIR/$DS/query.fbin" "query vectors"
need "$IDX/gt300.ivec" "ground truth of test block a (#0-999)"
need "$IDX/gt300_proxy.ivec" "ground truth of the calibration queries (#1000-1499)"
need "$IDX/gt300_q1500.ivec" "ground truth of test block b (#1500-)"
for S in $SS; do need "$IDX/S$S/shard0.index" "S=$S shard index"; done
log "start: S = $SS, results -> $RESULTS"

# ---- 1. adaptive termination: calibration and grid cells
{ for S in $SS; do for E in $E_LIMITS; do echo "calib $DS $S $E"; done; done
  for S in $SS; do for E in $(egrid "$S"); do echo "grid $DS $S $E"; done; done; } | par env RANK_MODE=rule "$EXP/cell.sh"
log "1. adaptive termination: calibration and grid cells done"

# ---- 2. matching (one process per S; inside an S the search is sequential)
# shellcheck disable=SC2016
if [ $WIDE = 1 ]; then
  for S in $SS; do case $S in 1|5|15) echo "$S --tab";; *) echo "$S";; esac; done \
    | par env RANK_MODE=rule TAB_E="$TAB_E" bash -c '"$0" "$1" "$2" "$3" ${4:+"$4"} --tab-E "$TAB_E"' "$PY" "$EXP/match.py" "$DS"
else
  RANK_MODE=rule $PY "$EXP/match.py" "$DS" "$SS" --tab-only --tab-E "$TAB_E" || die "matching failed"
fi
log "2. matching done"

# ---- 3. rank-K termination (S > 1)
DABS_SS=""; ds_rule_only "$DS" || for S in $SS; do [ "$S" != 1 ] && DABS_SS="$DABS_SS $S"; done
if [ -n "$DABS_SS" ]; then
  for S in $DABS_SS; do for E in $E_LIMITS; do echo "$DS $S $E"; done; done | par "$EXP/calib_dabs.sh"
  for S in $DABS_SS; do for E in $E_LIMITS; do echo "grid $DS $S $E"; done; done | par env RANK_MODE=dabs "$EXP/cell.sh"
  log "3. rank-K termination calibration and grid cells done"
fi

# ---- 4. tolerance dumps
{ for S in $SS; do
    arms="nat rule"; case " $DABS_SS " in *" $S "*) arms="$arms dabs";; esac
    for E in $E_LIMITS; do for A in $arms; do for B in a b; do echo "$DS $S $E $A $B"; done; done; done
  done; } | par "$EXP/dump_cell.sh"
log "4. tolerance dumps done"

# ---- 5. verdicts
$PY "$REPRO/lib/bootstrap.py" --results "$RESULTS" || die "bootstrap failed"
log "5. verdicts done: $RESULTS/verdicts/bootstrap.json"
