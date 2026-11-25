# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#
# Shared settings for every experiment script. Source it:  . "$(dirname "$0")/config.sh"
# The locations, tools and run settings below (WORK ... K) can be overridden from the environment. The experiment
# parameters RANK_GAMMAS, M_HNSW, EFC_HNSW, E_LIMITS and CALIB_GAMMAS are those of the original runs and are set here
# (run_core.sh takes other E values from CORE_E).

REPRO="${REPRO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Where large inputs live: raw vectors, shard indexes, ground truth, traversal records (hundreds of GB for the 10M sets)
WORK="${WORK:-$REPRO/work}"
RAW_DIR="${RAW_DIR:-$WORK/raw}"          # $RAW_DIR/<ds>/{base.fbin,query.fbin}
IDX_DIR="${IDX_DIR:-$WORK/idx}"          # $IDX_DIR/<ds>_N<N>/{gt300*.ivec, S<S>/shard<s>.{index,sel}}
TRACE_DIR="${TRACE_DIR:-$WORK/trace}"    # $TRACE_DIR/<ds>_N<N>/<variant>/*.bin   (gamma-replay records, regenerable)

# Where new results are written. The results of the original runs ship in $REPRO/results; we never write there by default.
RESULTS="${RESULTS:-$WORK/results}"

# Patched Faiss + benchmark driver (faiss/build_faiss.sh, bench/build_bench.sh)
PREFIX="${PREFIX:-$REPRO/_build/faiss}"
BIN="${BIN:-$REPRO/_build/bin/bench}"

PY="${PY:-python3}"
export PYTHONPATH="$REPRO/lib${PYTHONPATH:+:$PYTHONPATH}"

JOBS="${JOBS:-1}"                        # independent cells run in parallel (xargs -P). Use 1 for CPU-time measurements.
THREADS="${THREADS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"   # OpenMP threads for index build and exact ground truth
# Prefix for CPU-time measurements, e.g. PIN="numactl --cpunodebind=1 --membind=1" (the original runs used one socket of an AMD EPYC 9354)
PIN="${PIN:-}"

K="${K:-20}"                             # top-K. The experiments use 20; run_extras.sh rank100 sets K=100 for the rank comparison
# Replay grid for K != 20: a rank-K rule needs gamma near 0 at K=100, so this grid reaches down to 1e-4 (24 values).
# At K = 20 the replay uses the default grid of lib/trace_replay.py (GRID_GAMMAS).
RANK_GAMMAS="0.0001 0.00025 0.0005 0.001 0.0025 0.005 0.0075 0.01 0.0125 0.015 0.0175 0.02 0.025 0.03 0.035 0.04 0.045 0.05 0.06 0.07 0.08 0.1 0.12 0.15"
M_HNSW=40; EFC_HNSW=200                  # HNSW build parameters (M, efConstruction)
E_LIMITS="100 200 300"
CALIB_GAMMAS="$(awk 'BEGIN { for (i = 1; i <= 20; i++) printf "%s%.3f", (i > 1 ? " " : ""), i * 0.005 }')"   # calibration grid, 20 values

# Dataset helpers. "synth" is the tiny synthetic set of data/make_synthetic.py (smoke test only, not part of the experiments):
# its N and query count come from the fbin headers.
_fbin_rows() { $PY -c "import numpy as np,sys; print(int(np.fromfile(sys.argv[1], np.int32, 1)[0]))" "$1"; }
ds_n()    { case $1 in t2i) echo 10000000;; msmarco) echo 8841823;; glove) echo 1183514;; sift) echo 1000000;;
                       deep|t2i100m) echo 100000000;;
                       synth) _fbin_rows "$RAW_DIR/synth/base.fbin";; *) echo "unknown dataset $1" >&2; return 1;; esac; }
ds_norm() { case $1 in sift|deep|synth) echo 0;; *) echo 1;; esac; }   # SIFT and DEEP stay in L2; the others are normalized (cosine order)
ds_ss()   { case $1 in t2i|msmarco) echo "1 5 15 30 50";; deep|t2i100m) echo "5 15";; synth) echo "${SYNTH_SS:-1 5}";; *) echo "1 5 15";; esac; }
ds_nb()   { case $1 in t2i|t2i100m) echo 19000;; msmarco) echo 5480;; glove|sift|deep) echo 8500;;
                       synth) echo $(( $(_fbin_rows "$RAW_DIR/synth/query.fbin") - 1500 ));; esac; }   # size of test block b (#1500-)
# 100M-scale extension: only adaptive termination against fixed ef (no rank-K termination, no same-recall matching)
ds_rule_only() { case $1 in deep|t2i100m) return 0;; *) return 1;; esac; }
k_i()     { $PY -c "import shardlib as L; print(L.k_i_of(${2:-$K}, $1))"; }        # k_i <S> [K]
rank_of() { $PY -c "import shardlib as L; print(L.rule_rank('$1', ${3:-$K}, $2))"; }  # rank_of <rule|dabs> <S> [K]

die() { echo "!! $*" >&2; exit 1; }
need_bin() { [ -x "$BIN" ] || die "benchmark binary not found: $BIN (run faiss/build_faiss.sh and bench/build_bench.sh)"; }
