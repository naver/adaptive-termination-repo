#!/usr/bin/env bash
# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#
# Distance-computation experiments outside the main calibration/tolerance grid (run experiments/run_core.sh first: the shard,
# approx and kaxis cells read its calibration, $CELLS_ROOT/<dsn>/calib_cells.json, default $RESULTS/cells).
# None of these is a CPU measurement, so cells run JOBS at a time.
#
#   oracle  arrival-time oracle, T2I and MS MARCO, S in {1,5,15,30,50}, at two ef per S                          20 cells
#           fig-waste: the ef at which fixed ef reaches a recall of 0.95, read from
#           $CELLS_ROOT/<dsn>/match/R0.950_S<S>_fixed.json (run_core.sh writes it); and ef = 200
#   shard   per-query x per-shard ndis, T2I and MS MARCO, S {1,5,15} x ef {100,200,300} x {fixed, adaptive} x blocks
#           {a,b}: the shares of shard searches that reach the ef limit for easy and hard queries (S 1,5,15 x ef 200,300),
#           the ndis rows of fig-shard-tail and the max/mean imbalance (S 5,15 x ef 100,200,300)                  72 cells
#   approx  calibration with the approximate ground truth (same gamma_i or not), four datasets x S {1,5,15} x E {100,200,300}
#           plus T2I and MS MARCO x S {30,50} x E {100,200,300}                                                  48 cells
#   approxjudge  the configurations in which the approximate ground truth selects a different gamma_i in at least one
#           shard (experiments/approx_verdicts.py list; 8 in the shipped results): dumps of the arm approx (rank k_i,
#           approximate-ground-truth gammas) x blocks {a,b}, then approx_verdicts.py judge against the nat and rule dumps
#           of run_core.sh -> $RESULTS/approx_verdicts/verdicts.json                                          16 cells
#   kaxis   S=15, ef=200, T2I and MS MARCO, K {5,10,20,50,100} x {nat, dep, ref} x blocks {a,b}   60 cells
#   ncalib  recalibrate on the first 125 and 250 calibration queries (ncalib_cell.sh, no search), then
#           dumps of the arms p125, p250 (four datasets x S {1,5,15} x E {100,200,300} x blocks {a,b}) and
#           lib/bootstrap.py over all dumps                                                           72 + 144 cells
#   rank100 comparison rank at K=100: gamma-grid cells (cell.sh grid with K=100) of the rule (rank k_i) and rank-K termination
#           (rank K), T2I and MS MARCO, S {5,15,30,50} x E {200,300} (ef >= K), results in $RESULTS/cells_K100/; then
#           rank_k100.py summary (cost at 1pp below fixed ef, against the K=20 grids of run_core.sh) and rank_k100.py mech
#           (when the rank is filled and where each arm stops, from the records of both K)                    32 cells
#   transfer the gamma calibrated on the unsharded index (S = 1), applied unchanged to every shard: dumps of the arm
#           s1g (rank k_i, entry S1_E{E} of calib_cells.json), T2I and MS MARCO x S {5,15,30,50} x E {100,200,300} x blocks
#           {a,b}, paired with the nat dumps of run_core.sh; then lib/bootstrap.py over all dumps                48 cells
#   patience queue-saturation early termination (Patience), T2I and MS MARCO (PATIENCE_DS), S {1,5,15,30,50}: one record
#           with insertion counts per E of experiments/patience.py elist (E_LIMITS and the E the adaptive same-recall
#           matching tried; patience_cell.sh), then patience.py check, verify (S=15, E=200: replay = direct run),
#           match (same recall 0.95, 0.90), default (Lucene's defaults at E_LIMITS), mech and summary;
#           results in $RESULTS/patience/                                                                     91 cells
#
# Rough cost on the reference machine (one search thread per cell, AMD EPYC 9354): oracle and approx are minutes per cell;
# the 10M-scale shard and kaxis cells take about 1-10 minutes each for block b (19,000 T2I / 5,480 MS MARCO queries at
# about 1-30 ms of CPU per query), dominated by S=15 and ef=300. Index loading adds up to a minute per cell for the 10M sets.
#
# Usage: experiments/run_extras.sh [oracle] [shard] [approx] [approxjudge] [kaxis] [ncalib] [transfer] [rank100] [patience]
#        (no argument: all nine; approxjudge needs approx)
set -uo pipefail
X="$(cd "$(dirname "$0")" && pwd)"
. "$X/config.sh"
need_bin
CELLS_ROOT=${CELLS_ROOT:-$RESULTS/cells}
PARTS=${*:-oracle shard approx approxjudge kaxis ncalib transfer rank100 patience}
if [ "$JOBS" -gt 1 ] && ! command -v flock > /dev/null 2>&1; then
  echo "!! flock not found: approx and kaxis write shared calibration files and run one at a time" >&2; LJOBS=1
else LJOBS=$JOBS; fi
par() { xargs -P "$1" -L 1 bash -c '"$0" "$@"' "$2"; }   # par <jobs> <script>  (reads argument lines on stdin)

for part in $PARTS; do
  echo "[extras $part] start $(date '+%m-%d %H:%M:%S')"
  case $part in
    oracle)
      for DS in t2i msmarco; do DSN=${DS}_N$(ds_n $DS); for S in $(ds_ss $DS); do
        M=$CELLS_ROOT/$DSN/match/R0.950_S${S}_fixed.json
        [ -s "$M" ] || die "missing $M (run experiments/run_core.sh $DS first)"
        echo "$DS $S $($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['ef'])" "$M")"; echo "$DS $S 200"
      done; done | par "$JOBS" "$X/oracle_cell.sh" ;;
    shard)
      for DS in t2i msmarco; do for S in 1 5 15; do for E in $E_LIMITS; do for ARM in fixed adaptive; do for B in a b; do
        echo "$DS $S $E $ARM $B"; done; done; done; done; done | par "$JOBS" "$X/shard_cell.sh" ;;
    approx)
      { for DS in t2i msmarco glove sift; do for S in 1 5 15; do for E in $E_LIMITS; do echo "$DS $S $E"; done; done; done
        for DS in t2i msmarco; do for S in 30 50; do for E in $E_LIMITS; do echo "$DS $S $E"; done; done; done; } \
        | par "$LJOBS" "$X/approx_cell.sh" ;;
    approxjudge)
      CONFS=$($PY "$X/approx_verdicts.py" list --results "$RESULTS") || die "approx_verdicts list failed"
      [ -n "$CONFS" ] || die "no configuration with a differing gamma_i (run the approx part first)"
      echo "$CONFS" | while read -r DS S E; do for B in a b; do echo "$DS $S $E approx $B"; done; done | par "$JOBS" "$X/dump_cell.sh"
      $PY "$X/approx_verdicts.py" judge --results "$RESULTS" || die "approx_verdicts judge failed" ;;
    kaxis)
      # recalibrate each K first (the two blocks of a K share one calibration), then all dumps
      for DS in t2i msmarco; do for KK in 5 10 20 50 100; do echo "$DS $KK ref a"; done; done | par "$LJOBS" "$X/kaxis_cell.sh"
      for DS in t2i msmarco; do for KK in 5 10 20 50 100; do for ARM in nat dep ref; do for B in a b; do
        echo "$DS $KK $ARM $B"; done; done; done; done | par "$JOBS" "$X/kaxis_cell.sh" ;;
    ncalib)
      for DS in t2i msmarco glove sift; do for P in 125 250; do for S in 1 5 15; do for E in $E_LIMITS; do
        echo "$DS $S $E $P"; done; done; done; done | par "$LJOBS" "$X/ncalib_cell.sh"
      for DS in t2i msmarco glove sift; do for P in 125 250; do for S in 1 5 15; do for E in $E_LIMITS; do for B in a b; do
        echo "$DS $S $E p$P $B"; done; done; done; done; done | par "$JOBS" "$X/dump_cell.sh"
      $PY "$REPRO/lib/bootstrap.py" --results "$RESULTS" || die "bootstrap failed" ;;
    transfer)
      for DS in t2i msmarco; do for S in 5 15 30 50; do for E in $E_LIMITS; do for B in a b; do
        echo "$DS $S $E s1g $B"; done; done; done; done | par "$JOBS" "$X/dump_cell.sh"
      $PY "$REPRO/lib/bootstrap.py" --results "$RESULTS" || die "bootstrap failed" ;;
    rank100)
      for DS in t2i msmarco; do for S in 5 15 30 50; do for E in 200 300; do for M in rule dabs; do
        echo "$M $DS $S $E"; done; done; done; done | xargs -P "$JOBS" -L 1 bash -c 'K=100 RANK_MODE="$0" "'"$X"'/cell.sh" grid "$@"'
      $PY "$X/rank_k100.py" summary --results "$RESULTS" || die "rank_k100 summary failed"
      $PY "$X/rank_k100.py" mech --results "$RESULTS" --trace-dir "$TRACE_DIR" --idx-dir "$IDX_DIR" || die "rank_k100 mech failed" ;;
    patience)
      export REPRO WORK RAW_DIR IDX_DIR TRACE_DIR RESULTS CELLS_ROOT BIN PY K
      PDS=${PATIENCE_DS:-t2i msmarco}
      for DS in $PDS; do for S in $(ds_ss $DS); do for E in $($PY "$X/patience.py" elist $DS $S); do
        echo "$DS $S $E"; done; done; done | par "$JOBS" "$X/patience_cell.sh"
      $PY "$X/patience.py" check $PDS || die "patience check failed"
      for DS in $PDS; do
        read -r VS VE <<< "${PATIENCE_VERIFY:-15 200}"           # the cell whose replays are checked by direct runs
        $PY "$X/patience.py" verify $DS "$VS" "$VE" || die "patience verify failed ($DS S=$VS E=$VE)"
        for c in match default mech; do $PY "$X/patience.py" $c $DS || die "patience $c failed ($DS)"; done
      done
      $PY "$X/patience.py" summary "$PDS" || die "patience summary failed" ;;
    *) die "unknown part '$part' (oracle | shard | approx | approxjudge | kaxis | ncalib | transfer | rank100 | patience)" ;;
  esac
  echo "[extras $part] done $(date '+%m-%d %H:%M:%S')"
done
