#!/usr/bin/env bash
# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#
# Cost of the calibration table: what building it costs, against building the index. One dataset, parts one at a time.
#
#   (a) control: plain HNSW index build = shard split (timed separately) + per-shard build (M=40, efC=200,
#       $BUILD_THREADS OpenMP threads, default 28), 2 repetitions
#   (b) exact ground truth of the 500 calibration queries (bench gt, exhaustive; $GT_THREADS OpenMP + OpenBLAS threads,
#       default 16 + 16), 5 repetitions
#   (c) approximate ground truth: search every shard at ef_gt = min(1000, 4E) and merge a top-300 (--gt_out), 5 repetitions
#   (e) one recorded traversal of the calibration queries (--trace, gamma 0, rank k_i) + its replay (pred_calib.py cell over
#       the 20-value gamma grid), 3 repetitions
# Searches and replays run on one thread (the bench searches the shards of a query one after the other); state this
# when quoting wall-clock ratios. The thread counts above assume a 32-core socket; override BUILD_THREADS / GT_THREADS.
#
# Two kinds of time are recorded (both are kept):
#   times.jsonl  per process: wall_s, cpu_s (user+sys of the process tree, getrusage), throttled (growth of the cgroup v2
#                nr_throttled counter, 0 when there is no CPU quota), rc, and cgroup_cpu_s when /sys/fs/cgroup/cpu.stat exists
#                -- used for (a), (b) and the replay of (e)
#   json/*.json  the bench's own search-loop timing -- used for (c) and the record of (e)
# A timed step that fails or is throttled is retried up to three times (every attempt stays in times.jsonl).
# Every command is prefixed with $PIN.
#
# Output: $RESULTS/buildcost/<ds>_<RUN>/{times.jsonl, json/}; RUN defaults to run1. Steps already recorded are skipped.
# Temporary files: $WORK/buildcost/<ds>/ (the build of (a) needs room for one full set of shard indexes; BUILD_DIR overrides).
# Usage:  [SS="1 5 15 30 50"] [PARTS="b c e a"] [RUN=run1] experiments/buildcost.sh <ds>
set -uo pipefail
. "$(dirname "$0")/config.sh"
[ $# -eq 1 ] || die "usage: $0 <ds>"
DS=$1
N=$(ds_n "$DS") || exit 1
DSN=${DS}_N$N; SS=${SS:-$(ds_ss "$DS")}; PARTS=${PARTS:-b c e a}; RUN=${RUN:-run1}
BUILD_THREADS=${BUILD_THREADS:-28}; GT_THREADS=${GT_THREADS:-16}
O=$RESULTS/buildcost/${DS}_$RUN; TMP=$WORK/buildcost/$DS; BUILD_DIR=${BUILD_DIR:-$TMP/build}
mkdir -p "$O/json" "$TMP"
T=$O/times.jsonl; touch "$T"
need_bin
export OMP_NUM_THREADS=$BUILD_THREADS OPENBLAS_NUM_THREADS=$BUILD_THREADS
NORM=$(ds_norm "$DS"); IDX=$IDX_DIR/$DSN; RAW=$RAW_DIR/$DS
GAMMAS=$CALIB_GAMMAS
EXPECT=(); [ -n "${EXPECT_CPU:-}" ] && EXPECT=(--expect_cpu "$EXPECT_CPU")

# timeit <name> <cmd...>: run cmd (stdout/stderr pass through) and append one line to times.jsonl
TIMEIT_PY='
import json, resource, subprocess, sys, time
def cgroup():
    d = {}
    try:
        for line in open("/sys/fs/cgroup/cpu.stat"):
            k, v = line.split()
            d[k] = int(v)
    except (OSError, ValueError):
        pass
    return d
log, name, cmd = sys.argv[1], sys.argv[2], sys.argv[3:]
c0, r0, w0 = cgroup(), resource.getrusage(resource.RUSAGE_CHILDREN), time.time()
rc = subprocess.call(cmd)
w1, r1, c1 = time.time(), resource.getrusage(resource.RUSAGE_CHILDREN), cgroup()
row = {"name": name, "wall_s": round(w1 - w0, 3),
       "cpu_s": round((r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime), 3),
       "throttled": c1.get("nr_throttled", 0) - c0.get("nr_throttled", 0), "rc": rc}
if "usage_usec" in c0 and "usage_usec" in c1:
    row["cgroup_cpu_s"] = round((c1["usage_usec"] - c0["usage_usec"]) / 1e6, 3)
with open(log, "a") as f:
    f.write(json.dumps(row) + "\n")
sys.exit(rc)
'
timeit() { local name=$1; shift; $PY -c "$TIMEIT_PY" "$T" "$name" "$@"; }
done_()   { grep "\"name\": \"$1\"" "$T" 2>/dev/null | grep -q '"throttled": 0, "rc": 0'; }
last_ok() { tail -1 "$T" | grep -q '"throttled": 0, "rc": 0'; }
gated() {   # gated <name> <cmd...>: skip if recorded clean; otherwise up to three attempts
  local name=$1; shift
  done_ "$name" && return 0
  for _ in 1 2 3; do timeit "$name" "$@"; last_ok && return 0; done
  echo "!! $name throttled or failed three times" >&2; return 1
}
json_ok() { $PY -c "import json,sys; d=json.load(open(sys.argv[1])); assert d and 'recall' in d[0]" "$1" 2>/dev/null; }

for part in $PARTS; do case $part in
b)  for r in 1 2 3 4 5; do
      # shellcheck disable=SC2086  # PIN is a command prefix
      gated "gt_${DS}_r$r" env OMP_NUM_THREADS="$GT_THREADS" OPENBLAS_NUM_THREADS="$GT_THREADS" $PIN "$BIN" gt --base "$RAW/base.fbin" --limit "$N" \
        --query "$RAW/query.fbin" --nq 500 --qoff 1000 --k 300 --normalize "$NORM" --out "$TMP/gt.ivec" || exit 1
    done ;;
c|e) for S in $SS; do KI=$(k_i "$S"); for E in $E_LIMITS; do
      EG=$(( 4*E < 1000 ? 4*E : 1000 ))
      if [ "$part" = c ]; then for r in 1 2 3 4 5; do
        f=$O/json/approx_${DS}_S${S}_E${E}_r$r.json; json_ok "$f" && continue
        # shellcheck disable=SC2086
        timeit "approx_${DS}_S${S}_E${E}_r$r" $PIN "$BIN" search --shards "$IDX/S$S" --S "$S" --query "$RAW/query.fbin" --nq 500 --qoff 1000 \
          --normalize "$NORM" --gt "$IDX/gt300_proxy.ivec" --K 300 --ki 300 --rule_k 0 --ef "$EG" --gamma 0 --repeat 1 --warmup 0 ${EXPECT[@]+"${EXPECT[@]}"} \
          --gt_out "$TMP/approx.ivec" > "$f" 2> "$f.err"
        json_ok "$f" || die "$f failed"
      done
      else for r in 1 2 3; do
        f=$O/json/trace_${DS}_S${S}_E${E}_r$r.json; json_ok "$f" && continue
        rm -f "$TMP/trace.bin"
        # shellcheck disable=SC2086
        timeit "trace_${DS}_S${S}_E${E}_r$r" $PIN "$BIN" search --shards "$IDX/S$S" --S "$S" --query "$RAW/query.fbin" --nq 500 --qoff 1000 \
          --normalize "$NORM" --gt "$IDX/gt300_proxy.ivec" --K "$K" --ki "$KI" --rule_k "$KI" --ef "$E" --gamma 0 --repeat 1 --warmup 0 ${EXPECT[@]+"${EXPECT[@]}"} \
          --trace "$TMP/trace.bin" > "$f" 2> "$f.err"
        json_ok "$f" && [ -s "$TMP/trace.bin" ] || die "$f: record failed"
        # The replay does not use BLAS. One thread keeps an idle OpenBLAS pool (started when numpy loads) out of the CPU time.
        # shellcheck disable=SC2086
        timeit "replay_${DS}_S${S}_E${E}_r$r" env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $PIN $PY "$REPRO/lib/pred_calib.py" cell \
          --ds "$DS" --N "$N" --trace "$TMP/trace.bin" --gt "$IDX/gt300_proxy.ivec" --key "S${S}_E${E}" --out "$O/json/cells_${DS}_r$r.json" \
          --gammas "$GAMMAS" > /dev/null
      done; fi
    done; done ;;
a)  SPLIT_NORM=(); [ "$NORM" = 1 ] && SPLIT_NORM=(--normalize)
    for r in 1 2; do for S in $SS; do
      D=$BUILD_DIR/${DS}_S$S
      done_ "build_${DS}_S${S}_r${r}_s$((S-1))" && continue
      rm -rf "$D"; mkdir -p "$D"
      # a split that was recorded for an interrupted build is measured again
      grep -v "\"name\": \"split_${DS}_S${S}_r$r\"" "$T" > "$T.tmp"; mv "$T.tmp" "$T"
      # shellcheck disable=SC2086
      gated "split_${DS}_S${S}_r$r" $PIN $PY "$REPRO/lib/shard_split.py" --base "$RAW/base.fbin" --N "$N" --S "$S" --out "$D" ${SPLIT_NORM[@]+"${SPLIT_NORM[@]}"} || exit 1
      for s in $(seq 0 $((S-1))); do
        # shellcheck disable=SC2086
        gated "build_${DS}_S${S}_r${r}_s$s" $PIN "$BIN" build --base "$D/shard$s.fbin" --out "$D/shard$s.index" --M 40 --efc 200 \
          > "$O/json/build_${DS}_S${S}_r${r}_s$s.json" || exit 1
        rm -f "$D/shard$s.fbin" "$D/shard$s.index"
      done
      rm -rf "$D"
    done; done ;;
*) die "unknown part $part (a | b | c | e)" ;;
esac; done
echo "[buildcost $DS] parts '$PARTS' done $(date '+%m-%d %H:%M:%S')"
