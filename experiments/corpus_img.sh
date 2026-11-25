#!/usr/bin/env bash
# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#
# Calibration before serving on T2I with image queries. T2I's benchmark queries are text embeddings, so
# held-out corpus rows (images) are drawn unlike them. This driver asks whether the held-out calibration works when
# the queries are drawn like the indexed vectors: it evaluates on image queries instead.
#
# Image queries: 20,500 rows of the T2I base.1B.fbin starting at row 10,000,000 (outside the 10M graphs, same
# distribution), split like the text queries: #0-999 test block a, #1000-1499 calibration, #1500-20499 test block b.
# On the final graphs of run_corpus.sh t2i, four arms:
#   nat       fixed ef (gamma off)
#   rule      gammas calibrated on the text calibration queries   ($HOLDOUT_RESULTS/cells/t2i_N10000000/calib_cells.json)
#   corpus    gammas calibrated on held-out rows                  ($HOLDOUT_RESULTS/cells/t2i_N10000000/corpus/calib_cells.json)
#   rule_img  gammas recalibrated on the 500 image calibration queries (written here, $IMG_OUT/cells_img.json)
# Verdicts are two runs of lib/bootstrap.py: $IMG_OUT/A (nat against rule and corpus) and $IMG_OUT/B (nat against
# rule_img, stored under the arm name "rule"). Dumps are named with the dataset "t2iimg".
#
# Needs: run_corpus.sh t2i (final graphs in $HOLDOUT_IDX_DIR, calibrations in $HOLDOUT_RESULTS) and the T2I base vectors
# in $RAW_DIR/t2i/base.fbin (data/prepare_all.sh t2i). Downloads 16.4 MB (byte range). Steps skip outputs that exist.
#
# Usage: experiments/corpus_img.sh
set -uo pipefail
EXP=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$EXP/config.sh"
need_bin
IDX=${HOLDOUT_IDX_DIR:-$WORK/idx_holdout}/t2i_N10000000
CB=${HOLDOUT_RESULTS:-$WORK/results_holdout}/cells/t2i_N10000000
OUT=${IMG_OUT:-$WORK/results_img}
URL=https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1B.fbin
NQ=20500; ROW0=10000000; D=200
log() { echo "[img $(date +%H:%M:%S)] $*"; }
[ -d "$IDX/S1" ] || die "final graphs not found: $IDX (run experiments/run_corpus.sh t2i first)"
[ -s "$CB/corpus/calib_cells.json" ] || die "corpus calibration not found: $CB/corpus/calib_cells.json"
[ -s "$RAW_DIR/t2i/base.fbin" ] || die "T2I base vectors not found: $RAW_DIR/t2i/base.fbin"
mkdir -p "$OUT/A/dumps" "$OUT/B/dumps" "$OUT/trace"

# 1. image queries (raw; the benchmark normalizes them)
Q=$OUT/img_query.fbin
if [ ! -s "$Q" ]; then
  s=$((8 + ROW0 * D * 4)); e=$((s + NQ * D * 4 - 1))
  curl -sS --retry 5 -r "$s-$e" -o "$OUT/rows.bin" "$URL" || die "download failed"
  [ "$(wc -c < "$OUT/rows.bin" | tr -d ' ')" = $((NQ * D * 4)) ] || die "short download: $OUT/rows.bin"
  $PY -c "import numpy as np,sys; open(sys.argv[1],'wb').write(np.array([$NQ,$D],np.int32).tobytes()+open(sys.argv[2],'rb').read())" \
    "$Q.part" "$OUT/rows.bin" && mv "$Q.part" "$Q" && rm -f "$OUT/rows.bin"
  log "queries: $NQ image rows from #$ROW0"
fi

# 2. exact ground truth over the 10M rows (normalized, squared L2 = cosine order), cut into the three blocks
if [ ! -s "$OUT/gt_b.ivec" ]; then
  OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS "$BIN" gt --base "$RAW_DIR/t2i/base.fbin" --query "$Q" --nq $NQ \
    --k 300 --normalize 1 --out "$OUT/gt_all.ivec" 2> "$OUT/gt.log" || die "ground truth failed (see $OUT/gt.log)"
  $PY "$REPRO/data/gt_slice.py" --src "$OUT/gt_all.ivec" --rows 0:1000 --out "$OUT/gt_a.ivec" --expect-rows 1000
  $PY "$REPRO/data/gt_slice.py" --src "$OUT/gt_all.ivec" --rows 1000:1500 --out "$OUT/gt_proxy.ivec" --expect-rows 500
  $PY "$REPRO/data/gt_slice.py" --src "$OUT/gt_all.ivec" --rows 1500: --out "$OUT/gt_b.ivec" --expect-rows 19000
  log "ground truth done"
fi

gam() {   # <calib_cells.json> <key> <S> -> the --gamma argument of bench search (0 = off)
  $PY -c "import json,sys;c=json.load(open(sys.argv[1]))[sys.argv[2]]['gamma_rank_shard'];g=['0' if x is None else f'{x:g}' for x in c];print(g[0] if sys.argv[3]=='1' else ','.join(g))" "$1" "$2" "$3"
}

# 3. rule_img calibration (500 image calibration queries, final graphs)
for S in 1 5 15; do for E in $E_LIMITS; do
  k=$(k_i "$S"); key=S${S}_E${E}
  $PY -c "import json,sys;sys.exit(0 if sys.argv[2] in json.load(open(sys.argv[1])) else 1)" "$OUT/cells_img.json" $key 2>/dev/null && continue
  t=$OUT/trace/proxy_$key.bin
  "$BIN" search --shards "$IDX/S$S" --S $S --query "$Q" --normalize 1 --check_norm 1 --K $K --ki $k --rule_k $k --ef $E \
    --nq 500 --qoff 1000 --gt "$OUT/gt_proxy.ivec" --gamma 0 --repeat 1 --warmup 100 --trace "$t" > /dev/null 2> "$t.err" \
    || die "rule_img record failed $key"
  $PY "$REPRO/lib/pred_calib.py" cell --ds t2i --N 10000000 --trace "$t" --gt "$OUT/gt_proxy.ivec" --key $key \
    --out "$OUT/cells_img.json" | sed 's/^/  rule_img /'
done; done
log "rule_img calibration done"

# 4. dumps: four arms x two blocks (A = nat, rule, corpus; B = nat, rule_img)
dump() {   # <S> <E> <arm> <block>
  local S=$1 E=$2 arm=$3 blk=$4 k key G dir name f
  k=$(k_i "$S"); key=S${S}_E${E}
  case $arm in
    nat) G=0; dir=A; name=nat ;;
    rule) G=$(gam "$CB/calib_cells.json" $key $S); dir=A; name=rule ;;
    corpus) G=$(gam "$CB/corpus/calib_cells.json" $key $S); dir=A; name=corpus ;;
    rule_img) G=$(gam "$OUT/cells_img.json" $key $S); dir=B; name=rule ;;
  esac
  f=$OUT/$dir/dumps/t2iimg_S${S}_ef${E}_${name}_$blk
  [ -s "$f.tsv" ] && [ -s "$f.json" ] && return 0
  case $blk in a) B=(--nq 1000 --qoff 0 --gt "$OUT/gt_a.ivec");; b) B=(--nq 19000 --qoff 1500 --gt "$OUT/gt_b.ivec");; esac
  "$BIN" search --shards "$IDX/S$S" --S $S --query "$Q" --normalize 1 --check_norm 1 "${B[@]}" --K $K --ki $k --rule_k $k \
    --ef $E --gamma "$G" --repeat 1 --warmup 100 --dump "$f.tsv.part" > "$f.json.part" 2> "$f.err" \
    && mv "$f.tsv.part" "$f.tsv" && mv "$f.json.part" "$f.json" && rm -f "$f.err" \
    || { echo "!! dump failed $S $E $arm $blk" >&2; tail -n 3 "$f.err" >&2; return 1; }
  echo "  $arm S=$S ef=$E $blk $($PY -c "import json,sys;d=json.load(open(sys.argv[1]))[0];print('recall=%.5f ndis=%.1f'%(d['recall'],d['ndis_per_query']))" "$f.json")"
}
for S in 1 5 15; do for E in $E_LIMITS; do for arm in nat rule corpus rule_img; do for blk in a b; do
  dump $S $E $arm $blk || die "a dump failed"
done; done; done; done
for f in "$OUT"/A/dumps/t2iimg_*_nat_*; do ln -sf "$f" "$OUT/B/dumps/$(basename "$f")"; done
log "dumps done"

# 5. verdicts
$PY "$REPRO/lib/bootstrap.py" --results "$OUT/A" > "$OUT/A/bootstrap.log" || die "bootstrap A failed"
$PY "$REPRO/lib/bootstrap.py" --results "$OUT/B" > "$OUT/B/bootstrap.log" || die "bootstrap B failed"
log "done: $OUT/A/verdicts/bootstrap.json (nat vs rule, corpus), $OUT/B/verdicts/bootstrap.json (nat vs rule_img)"
