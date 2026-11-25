#!/usr/bin/env bash
# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#
# CPU rows of fig-shard-tail: per-query x per-shard CPU time, one search thread, prefixed with $PIN.
#
# Cells: S 5,15 x ef 100,200,300 x {fixed, adaptive with the calibrated per-shard gammas of adaptive termination}
#        x blocks a (#0-999) and b (#1500-), 24 per dataset, one at a time. --repeat 3 --warmup 50 --vt_threshold 999999999.
# Every repetition is dumped with its rep column (--dump_shard ... --dump_shard_time 1); the bench JSON says which
# repetitions passed the noise/throttle gate ("accept"; the throttle gate reads cgroup v2 cpu.stat and is a no-op on a
# machine without a CPU quota). The analysis (figures/) takes, per query, the accepted repetition with the smallest CPU sum.
# Check: the per-query x per-shard ndis of repetition 0 must equal the ndis-only dump of the same cell
# (${DUMPS_DIR:-$RESULTS}/shard/ or the published export per_shard_ndis/, experiments/shard_cell.sh), since the search is deterministic; the script stops if it differs.
# If that dump does not exist yet (or is a Git LFS pointer) the check is skipped with a warning.
#
# CELLS_ROOT (default $RESULTS/cells) holds the calibrated gammas.
# Output: $RESULTS/cpu_shard/<ds>_N<N>/S{S}_ef{E}_{fixed|adaptive}_{a|b}.{tsv,json}; existing cells are skipped.
# Usage:  experiments/cpu_tail.sh <ds>
set -uo pipefail
. "$(dirname "$0")/config.sh"
[ $# -eq 1 ] || die "usage: $0 <ds>"
DS=$1
N=$(ds_n "$DS") || exit 1
DSN=${DS}_N$N; OUT=$RESULTS/cpu_shard/$DSN; DR=${DUMPS_DIR:-$RESULTS}; REF=$DR/shard/$DSN; [ -d "$REF" ] || [ ! -d "$DR/per_shard_ndis/$DSN" ] || REF=$DR/per_shard_ndis/$DSN; mkdir -p "$OUT"
CELLS_ROOT=${CELLS_ROOT:-$RESULTS/cells}
need_bin
NORM=$(ds_norm "$DS"); IDX=$IDX_DIR/$DSN
EXPECT=(); [ -n "${EXPECT_CPU:-}" ] && EXPECT=(--expect_cpu "$EXPECT_CPU")
for S in 5 15; do
  KI=$(k_i "$S")
  for E in 100 200 300; do for ARM in fixed adaptive; do for BLK in a b; do
    f=$OUT/S${S}_ef${E}_${ARM}_${BLK}
    [ -s "$f.tsv" ] && [ -s "$f.json" ] && continue
    if [ $ARM = fixed ]; then G=0
    else G=$($PY -c "
import json, sys
c = json.load(open(sys.argv[1]))[sys.argv[2]]
print(','.join('0' if g is None else f'{g:g}' for g in c['gamma_rank_shard']))" "$CELLS_ROOT/$DSN/calib_cells.json" "S${S}_E${E}") \
           || die "no calibrated gammas for S${S}_E${E} in $CELLS_ROOT/$DSN/calib_cells.json"
    fi
    case $BLK in
      a) Q=(--nq 1000 --qoff 0 --gt "$IDX/gt300.ivec") ;;
      b) Q=(--nq "$(ds_nb "$DS")" --qoff 1500 --gt "$IDX/gt300_q1500.ivec") ;;
    esac
    t0=$(date +%s)
    # shellcheck disable=SC2086  # PIN is a command prefix and must be word-split
    $PIN "$BIN" search --shards "$IDX/S$S" --S "$S" --query "$RAW_DIR/$DS/query.fbin" --normalize "$NORM" "${Q[@]}" \
      --K "$K" --ki "$KI" --rule_k "$KI" --ef "$E" --gamma "$G" --repeat 3 --warmup 50 --vt_threshold 999999999 ${EXPECT[@]+"${EXPECT[@]}"} \
      --dump_shard "$f.tsv.part" --dump_shard_time 1 > "$f.json.part" 2> "$f.err" \
      && $PY -c "import json,sys; json.load(open(sys.argv[1]))" "$f.json.part" 2>/dev/null \
      || { echo "!! run failed $f: $(tail -n 2 "$f.err")" >&2; exit 1; }
    if [ -s "$REF/S${S}_ef${E}_${ARM}_${BLK}.tsv" ] && ! head -c 40 "$REF/S${S}_ef${E}_${ARM}_${BLK}.tsv" | grep -q '^version https://git-lfs'; then
      $PY - "$f.tsv.part" "$REF/S${S}_ef${E}_${ARM}_${BLK}.tsv" <<'PY' || { echo "!! ndis differs from the ndis-only dump: $f" >&2; mv "$f.tsv.part" "$f.tsv.mismatch"; exit 1; }
import sys
import numpy as np
a = np.genfromtxt(sys.argv[1], delimiter="\t", names=True)
b = np.genfromtxt(sys.argv[2], delimiter="\t", names=True)
a0 = a[a["rep"] == 0]
key = lambda x: np.lexsort((x["shard"], x["q"]))
a0, b = a0[key(a0)], b[key(b)]
sys.exit(0 if len(a0) == len(b) and np.array_equal(a0["ndis"], b["ndis"]) and np.array_equal(a0["q"], b["q"]) else 1)
PY
    else
      echo "  (no ndis-only dump $REF/S${S}_ef${E}_${ARM}_${BLK}.tsv; determinism check skipped)" >&2
    fi
    mv "$f.tsv.part" "$f.tsv" && mv "$f.json.part" "$f.json"
    $PY -c "
import json, sys
d = json.load(open(sys.argv[1])); ok = [r for r in d if r.get('accept')]
print('%s tail %-8s S=%-2s ef=%-3s %-8s %s recall=%.5f ndis=%.1f cpu=%s accepted=%d/%d (%ss)' % (sys.argv[2], sys.argv[3], d[0]['S'], d[0]['ef'],
      sys.argv[4], sys.argv[5], d[0]['recall'], d[0]['ndis_per_query'], [round(r['cpu_sum_ms']['mean'], 3) for r in d], len(ok), len(d), sys.argv[6]))" \
      "$f.json" "$(date +%H:%M:%S)" "$DS" "$ARM" "$BLK" "$(( $(date +%s) - t0 ))"
  done; done; done
done
echo "[cpu_tail $DS] done $(date '+%m-%d %H:%M:%S')"
