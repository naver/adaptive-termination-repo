#!/usr/bin/env bash
# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#
# Calibration with an approximate ground truth for one configuration (how many shard calibrations select the same gamma_i).
#
# Approximate ground truth: search the 500 calibration queries (#1000-1499) in every shard at ef_gt = min(1000, 4E) and merge
# the per-shard results into a global top-300 (--gt_out). The traversal record of the same calibration queries under the
# adaptive termination (gamma 0, rank k_i) is then calibrated again against that ground truth (pred_calib.py cell, per-shard gammas),
# and the result is compared with the gammas calibrated against the exact ground truth.
#
# The calibration record is shared with the main calibration (experiments/cell.sh): $PROXY_TRACE_DIR/proxy_S{S}_E{E}.bin,
# default $TRACE_DIR/<dsn>/rule/. It is recorded here if it is missing (the record is deterministic).
# CELLS_ROOT (default $RESULTS/cells) holds the exact-ground-truth gammas to compare with.
#
# Output: $RESULTS/approx/<ds>_N<N>/calib_cells_approx.json (key S{S}_E{E}) and gt_S{S}_E{E}.ivec.json (bench output of the
#         ground-truth search). The approximate ground truth itself goes to $TRACE_DIR/<dsn>/approx/gt_S{S}_E{E}.ivec.
#         Skipped if the key is already in calib_cells_approx.json.
# Usage:  experiments/approx_cell.sh <ds> <S> <E>
set -uo pipefail
. "$(dirname "$0")/config.sh"
[ $# -eq 3 ] || die "usage: $0 <ds> <S> <E>"
DS=$1; S=$2; E=$3
N=$(ds_n "$DS") || exit 1
DSN=${DS}_N$N; OUT=$RESULTS/approx/$DSN; mkdir -p "$OUT"
CELLS_ROOT=${CELLS_ROOT:-$RESULTS/cells}
PROXY_TRACE_DIR=${PROXY_TRACE_DIR:-$TRACE_DIR/$DSN/rule}
CELLS=$OUT/calib_cells_approx.json; key=S${S}_E${E}
$PY -c "import json,sys; sys.exit(0 if sys.argv[2] in json.load(open(sys.argv[1])) else 1)" "$CELLS" "$key" 2>/dev/null && exit 0
need_bin
NORM=$(ds_norm "$DS"); KI=$(k_i "$S"); IDX=$IDX_DIR/$DSN; EG=$(( 4*E < 1000 ? 4*E : 1000 ))
LOCKS=$WORK/locks; mkdir -p "$LOCKS" "$PROXY_TRACE_DIR" "$TRACE_DIR/$DSN/approx"
locked() { local l=$1; shift; if command -v flock > /dev/null 2>&1; then flock "$l" "$@"; else "$@"; fi; }   # locked <lockfile> <cmd...>

# 1. calibration record (adaptive termination, gamma 0)
tb=$PROXY_TRACE_DIR/proxy_S${S}_E${E}.bin
if [ ! -s "$tb" ]; then
  locked "$LOCKS/${DSN}_proxy_S${S}_E${E}.lock" bash -c '[ -s "$1" ] && exit 0
    "$2" search --shards "$3/S$4" --S "$4" --query "$5" --normalize "$6" --check_norm "$6" --nq 500 --qoff 1000 --gt "$3/gt300_proxy.ivec" \
      --K "$7" --ki "$8" --rule_k "$8" --ef "$9" --gamma 0 --repeat 1 --warmup 100 --trace "$1.part" > /dev/null 2> "$1.err" && mv "$1.part" "$1"' \
    _ "$tb" "$BIN" "$IDX" "$S" "$RAW_DIR/$DS/query.fbin" "$NORM" "$K" "$KI" "$E" \
    || { echo "!! calibration record failed $DS $key: $(tail -n 2 "$tb.err")" >&2; exit 1; }
fi

# 2. approximate ground truth of the calibration queries
g=$TRACE_DIR/$DSN/approx/gt_S${S}_E${E}.ivec
if [ ! -s "$g" ]; then
  "$BIN" search --shards "$IDX/S$S" --S "$S" --query "$RAW_DIR/$DS/query.fbin" --nq 500 --qoff 1000 --normalize "$NORM" --check_norm "$NORM" \
    --gt "$IDX/gt300_proxy.ivec" --K 300 --ki 300 --rule_k 0 --ef "$EG" --gamma 0 --repeat 1 --warmup 0 --gt_out "$g.part" \
    > "$OUT/gt_S${S}_E${E}.ivec.json" 2> "$OUT/gt_S${S}_E${E}.err" && mv "$g.part" "$g" \
    || { echo "!! approximate ground truth failed $DS $key: $(tail -n 2 "$OUT/gt_S${S}_E${E}.err")" >&2; exit 1; }
fi

# 3. calibrate against it (several cells may write the same JSON, hence the lock)
locked "$LOCKS/${DSN}_approx.lock" $PY "$REPRO/lib/pred_calib.py" cell --ds "$DS" --N "$N" --trace "$tb" --gt "$g" --key "$key" \
  --out "$CELLS" --gammas "$CALIB_GAMMAS" > /dev/null || die "pred_calib failed $DS $key"

# 4. compare with the exact-ground-truth gammas
$PY - "$CELLS" "$CELLS_ROOT/$DSN/calib_cells.json" "$key" "$DS" "$EG" <<'EOF'
import json, os, sys
cells, exact, key, ds, eg = sys.argv[1:]
a = json.load(open(cells))[key]["gamma_rank_shard"]
if not os.path.exists(exact) or key not in json.load(open(exact)):
    print(f"  approx {ds} {key} ef_gt={eg}  gammas {a}  (no exact-ground-truth calibration to compare with)")
    sys.exit(0)
x = json.load(open(exact))[key]["gamma_rank_shard"]
same = sum(p == q for p, q in zip(a, x))
print(f"  approx {ds} {key} ef_gt={eg}  same shard gamma {same}/{len(a)}  max diff {max(abs((p or 0) - (q or 0)) for p, q in zip(a, x)):.3f}")
EOF
