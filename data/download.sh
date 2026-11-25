#!/usr/bin/env bash
# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#
# Download the public source files of the datasets into $DL_DIR/<ds>/ (default $WORK/download/<ds>/).
# Resumable and idempotent: rerun the same command after an interruption. Skips a dataset whose raw files
# ($RAW_DIR/<ds>/base.fbin and query.fbin) already exist. Sources, licenses and sizes: data/DATASETS.md.
#
# Usage: data/download.sh [t2i] [msmarco] [glove] [sift] [deep] [t2i100m]   (default: the four 10M/1M datasets)
#   T2I_PARTS=8         parallel byte ranges for the T2I base file (also T2I 100M and DEEP 100M)
#   CHECK_REMOTE_SIZE=0 skip the Content-Length check against the sizes below
set -euo pipefail
. "$(dirname "$0")/common.sh"

# ---- T2I: Yandex Text-to-Image 1B (Big-ANN benchmark, NeurIPS'21). We need only the first 10M base vectors.
T2I_URL=https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I
T2I_N=10000000; T2I_D=200
T2I_BASE_BYTES=$((8 + T2I_N * T2I_D * 4))           # header + 10M x 200 float32 = 8,000,000,008
T2I_QUERY_BYTES=80000008                             # 100,000 x 200 float32
T2I100M_N=100000000
T2I100M_BASE_BYTES=$((8 + T2I100M_N * T2I_D * 4))   # first 100M rows of the same file = 80,000,000,008

# ---- DEEP: Yandex DEEP1B (Big-ANN benchmark, NeurIPS'21). We need only the first 100M base vectors.
DEEP_URL=https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP
DEEP_N=100000000; DEEP_D=96
DEEP_BASE_BYTES=$((8 + DEEP_N * DEEP_D * 4))        # header + 100M x 96 float32 = 38,400,000,008
DEEP_QUERY_BYTES=3840008                             # 10,000 x 96 float32

# ---- MS MARCO v1 passage, Cohere embed-english-v3.0 vectors distributed with Anserini / Pyserini
MSM_CORPUS_URL=https://rgw.cs.uwaterloo.ca/pyserini/data/msmarco-passage-cohere-embed-english-v3.0.parquet.tar
MSM_CORPUS_BYTES=17040240640; MSM_CORPUS_MD5=760dfb5ba9e2b0cc6f7e527e518fef03
MSM_QUERY_URL=https://github.com/castorini/pyserini-data/raw/main/encoded-queries/query-embedding-cohere-embed-english-v3.0-msmarco-passage-dev-subset-20240216-2154e79.tar.gz
MSM_QUERY_BYTES=22377230; MSM_QUERY_MD5=7dd0026490117e9e6f6acfc110d6ce83
MSM_QRELS_URL=https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.small.tsv   # optional check only
MSM_QRELS_BYTES=143300; MSM_QRELS_MD5=38a80559a561707ac2ec0f150ecd1e8a

# ---- ANN-Benchmarks HDF5 files (base = train, queries = test)
AB_URL=http://ann-benchmarks.com
GLOVE_BYTES=962819488
SIFT_BYTES=525128288

have_raw() { [ -s "$RAW_DIR/$1/base.fbin" ] && [ -s "$RAW_DIR/$1/query.fbin" ]; }

dl_t2i() {
  local d="$DL_DIR/t2i"
  fetch_range "$T2I_URL/base.1B.fbin" "$d/base.1B.first10M.fbin" "$T2I_BASE_BYTES" "${T2I_PARTS:-8}"
  fetch "$T2I_URL/query.public.100K.fbin" "$d/query.public.100K.fbin" "$T2I_QUERY_BYTES"
}

dl_t2i100m() {
  local d="$DL_DIR/t2i100m"
  fetch_range "$T2I_URL/base.1B.fbin" "$d/base.1B.first100M.fbin" "$T2I100M_BASE_BYTES" "${T2I_PARTS:-8}"
  fetch "$T2I_URL/query.public.100K.fbin" "$d/query.public.100K.fbin" "$T2I_QUERY_BYTES"
}

dl_deep() {
  local d="$DL_DIR/deep"
  fetch_range "$DEEP_URL/base.1B.fbin" "$d/base.1B.first100M.fbin" "$DEEP_BASE_BYTES" "${T2I_PARTS:-8}"
  fetch "$DEEP_URL/query.public.10K.fbin" "$d/query.public.10K.fbin" "$DEEP_QUERY_BYTES"
}

dl_msmarco() {
  local d="$DL_DIR/msmarco"
  fetch "$MSM_QRELS_URL" "$d/qrels.dev.small.tsv" "$MSM_QRELS_BYTES" "$MSM_QRELS_MD5"
  if [ ! -f "$d/corpus/.extracted" ]; then
    fetch "$MSM_CORPUS_URL" "$d/msmarco-passage-cohere-embed-english-v3.0.parquet.tar" "$MSM_CORPUS_BYTES" "$MSM_CORPUS_MD5"
    log "extract corpus parquet"
    rm -rf "$d/corpus"; mkdir -p "$d/corpus"
    tar -xf "$d/msmarco-passage-cohere-embed-english-v3.0.parquet.tar" -C "$d/corpus" --no-same-owner
    touch "$d/corpus/.extracted"
  fi
  if [ ! -f "$d/queries/.extracted" ]; then
    fetch "$MSM_QUERY_URL" "$d/query-embedding-cohere-embed-english-v3.0-msmarco-passage-dev-subset.tar.gz" "$MSM_QUERY_BYTES" "$MSM_QUERY_MD5"
    rm -rf "$d/queries"; mkdir -p "$d/queries"
    tar -xzf "$d/query-embedding-cohere-embed-english-v3.0-msmarco-passage-dev-subset.tar.gz" -C "$d/queries" --no-same-owner
    touch "$d/queries/.extracted"
  fi
}

dl_glove() { fetch "$AB_URL/glove-200-angular.hdf5" "$DL_DIR/glove/glove-200-angular.hdf5" "$GLOVE_BYTES"; }
dl_sift()  { fetch "$AB_URL/sift-128-euclidean.hdf5" "$DL_DIR/sift/sift-128-euclidean.hdf5" "$SIFT_BYTES"; }

for ds in ${*:-$DATASETS_ALL}; do
  case $ds in t2i|msmarco|glove|sift|deep|t2i100m) ;; *) die "unknown dataset '$ds' (t2i msmarco glove sift deep t2i100m)";; esac
  if have_raw "$ds"; then
    log "$ds: raw files exist, skip download"
    # the qrels file (143 KB) is only for the optional check in verify_msmarco_qrels.py
    [ "$ds" != msmarco ] || fetch "$MSM_QRELS_URL" "$DL_DIR/msmarco/qrels.dev.small.tsv" "$MSM_QRELS_BYTES" "$MSM_QRELS_MD5"
    continue
  fi
  log "=== download $ds"
  "dl_$ds"
  # No reference checksum is published for the T2I, DEEP and ANN-Benchmarks files: record ours, so that two installations
  # can compare their inputs (MS MARCO files were checked against the published MD5s above).
  if [ "$ds" != msmarco ]; then
    ( cd "$DL_DIR/$ds" && for f in *.fbin *.hdf5; do [ -f "$f" ] && echo "$(md5_of "$f")  $f"; done ) > "$DL_DIR/$ds/MD5SUMS.local" || true
  fi
  log "$ds: done ($(du -sh "$DL_DIR/$ds" | cut -f1))"
done
