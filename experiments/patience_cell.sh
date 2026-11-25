#!/usr/bin/env bash
# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#
# One cell (dataset, S, E) of the comparison with queue-saturation early termination (Patience): one gamma = 0 run on
# test block a (#0-999) with --trace --trace_ins 1 (insertion counts of both queues, record magic V31Q), replayed by
# lib/patience_replay.py cell: the gamma = 0 end state against the recording run, the gammas that the same-recall
# matching of adaptive termination chose at this E (experiments/patience.py gammas), the smallest s per queue, p and
# R in {0.95, 0.90}, a p x s grid, and at E in PATIENCE_E (default E_LIMITS) Lucene's defaults with per-query hits.
# Output $RESULTS/patience/<dsn>/cell/S{S}_E{E}.json; record, recording-run JSON and hits under $TRACE_DIR/<dsn>/patience/.
# An existing result is skipped. experiments/patience.py reads the cells (run_extras.sh patience runs both).
#
# Usage: patience_cell.sh <ds> <S> <E>
set -uo pipefail
X="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$X/config.sh"
[ $# -ge 3 ] || die "usage: patience_cell.sh <ds> <S> <E>"
DS=$1; S=$2; E=$3
N=$(ds_n "$DS") || exit 1
DSN=${DS}_N${N}
CELLS_ROOT=${CELLS_ROOT:-$RESULTS/cells}
export REPRO WORK RAW_DIR IDX_DIR TRACE_DIR RESULTS CELLS_ROOT BIN PY K
export "N_$(echo "$DS" | tr '[:lower:]' '[:upper:]')=$N"    # shardlib.ds_N for datasets other than the six (synth)
OUT=$RESULTS/patience/$DSN/cell; TD=$TRACE_DIR/$DSN/patience
IDX=$IDX_DIR/$DSN; QF=$RAW_DIR/$DS/query.fbin; NORM=$(ds_norm "$DS"); KI=$(k_i "$S")
mkdir -p "$OUT" "$TD"
f=$OUT/S${S}_E${E}.json; [ -s "$f" ] && exit 0
need_bin
[ -s "$IDX/gt300.ivec" ] || die "ground truth of test block a not found: $IDX/gt300.ivec"
t=$TD/S${S}_E${E}.bin; b=$TD/S${S}_E${E}.json
if ! { [ -s "$b" ] && [ -s "$t" ]; }; then
  "$BIN" search --shards "$IDX/S$S" --S "$S" --query "$QF" --normalize "$NORM" --check_norm "$NORM" \
    --K "$K" --ki "$KI" --rule_k "$KI" --ef "$E" --nq 1000 --qoff 0 --gt "$IDX/gt300.ivec" --gamma 0 --repeat 1 --warmup 100 \
    --trace "$t.part" --trace_ins 1 > "$b.part" 2> "$b.err" && mv "$t.part" "$t" && mv "$b.part" "$b" \
    || { echo "!! record failed: patience $DSN S=$S E=$E" >&2; tail -n 3 "$b.err" >&2; rm -f "$t.part" "$b.part"; exit 1; }
  rm -f "$b.err"
fi
GAM=$($PY "$X/patience.py" gammas "$DS" "$S" "$E") || exit 1
DEF=(); case " ${PATIENCE_E:-$E_LIMITS} " in *" $E "*) DEF=(--default --hits_out "$TD/hits_S${S}_E${E}.npz");; esac
$PY "$REPRO/lib/patience_replay.py" cell --trace "$t" --gt "$IDX/gt300.ivec" --K "$K" --bench "$b" --gammas "$GAM" \
  ${DEF[@]+"${DEF[@]}"} > "$f.part" 2> "$f.err" && mv "$f.part" "$f" && rm -f "$f.err" \
  || { echo "!! replay failed: patience $DSN S=$S E=$E" >&2; tail -n 3 "$f.err" >&2; rm -f "$f.part"; exit 1; }
$PY - "$f" "$DSN" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))[0]
best = {}
for x in d["match"]:
    if x.get("feasible") and x.get("s") is not None:
        k = (x["queue"], x["R"])
        if k not in best or x["ndis"] < best[k]["ndis"]:
            best[k] = x
s = "  ".join(f"{q}@{R}: p={x['p']} s={x['s']} {x['ndis']:.0f}" for (q, R), x in sorted(best.items()))
print(f"  patience {sys.argv[2]} S={d['S']} E={d['ef']} k_i={d['ki']}  gamma=0 {d['fixed_end']['recall']:.5f}/"
      f"{d['fixed_end']['ndis']:.1f}  {s}")
EOF
