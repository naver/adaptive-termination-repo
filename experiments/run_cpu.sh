#!/usr/bin/env bash
# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#
# CPU-time measurements, one session per dataset (run experiments/run_core.sh first; the shard dumps of
# experiments/run_extras.sh shard are the reference of the cpu_tail determinism check).
#
# Per dataset, strictly one cell at a time on an otherwise idle machine:
#   1. experiments/cpu.py <ds>        fig-overhead-same-recall, fig-adaptive-savings-both, CPU vs ndis ratio  T2I 46 cells, MS MARCO 50 cells
#   2. experiments/cpu_tail.sh <ds>   fig-shard-tail CPU rows                              24 cells
#   3. experiments/buildcost.sh <ds>  cost of the calibration table, parts b c e a, S 1,5,15,30,50
# Both searches of every ratio come from the same session. The original runs measured on one socket of an AMD EPYC 9354
# (PIN="numactl --cpunodebind=1 --membind=1"); set PIN accordingly, and EXPECT_CPU="<model substring>" to make the bench
# refuse a different CPU model. Repetitions that fail the bench's noise or throttle gate are discarded by the analysis;
# the throttle gate reads cgroup v2 cpu.stat and is a no-op where no CPU quota applies.
# Each session first writes the measurement conditions to $RESULTS/env_<ds>.txt.
#
# Duration of the original sessions (EPYC 9354): steps 1+2 about 1.5 h (T2I) and 1.7 h (MS MARCO); step 3 about 1.5 h (T2I)
# and 3-4 h (MS MARCO), most of it the two index builds of (a).
#
# Usage: experiments/run_cpu.sh [t2i] [msmarco]          (no argument: both, T2I first)
set -uo pipefail
X="$(cd "$(dirname "$0")" && pwd)"
. "$X/config.sh"
need_bin
[ -n "$PIN" ] || echo "note: PIN is empty; CPU times are taken wherever the scheduler places the search thread" >&2
mkdir -p "$RESULTS"
for DS in ${*:-t2i msmarco}; do
  case $DS in t2i|msmarco) ;; *) die "CPU sessions exist for t2i and msmarco only (got $DS)";; esac
  echo "[cpu session $DS] start $(date '+%m-%d %H:%M:%S')"
  {
    echo "date: $(date)"; echo "bench: $BIN"; (md5sum "$BIN" 2>/dev/null || md5 "$BIN") | head -1
    grep -m1 'model name' /proc/cpuinfo 2>/dev/null; echo "PIN: ${PIN:-<none>}"
    [ -n "$PIN" ] && $PIN bash -c 'grep Cpus_allowed_list /proc/self/status' 2>/dev/null
    command -v numactl > /dev/null && numactl -H 2>/dev/null | grep -E 'node [0-9]+ (size|free)'
    for f in /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor /sys/devices/system/cpu/cpufreq/boost \
             /sys/kernel/mm/transparent_hugepage/enabled /sys/fs/cgroup/cpu.max; do [ -r "$f" ] && echo "$f: $(cat "$f")"; done
    echo "nproc: $(getconf _NPROCESSORS_ONLN)"
  } > "$RESULTS/env_$DS.txt" 2>&1
  $PY "$X/cpu.py" "$DS" || die "cpu.py $DS failed"
  "$X/cpu_tail.sh" "$DS" || die "cpu_tail.sh $DS failed"
  SS="$(ds_ss "$DS")" PARTS="b c e a" "$X/buildcost.sh" "$DS" || die "buildcost.sh $DS failed"
  echo "[cpu session $DS] done $(date '+%m-%d %H:%M:%S')"
done
