#!/usr/bin/env bash
# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#
# Build the patched Faiss used by every measurement in the paper and install it into $PREFIX.
#
#   Faiss v1.15.0 (commit 20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed)
#   + patches/0001-adaptive-termination.patch   (adaptive termination, oracle instrumentation)
#   + patches/0002-traversal-trace.patch        (base-layer traversal record for gamma replay)
#   + patches/0003-patience.patch               (queue-saturation early termination, the Patience baseline)
#   static library, Release, no GPU / Python / tests
#
# OPT_LEVEL selects FAISS_OPT_LEVEL: avx512 (default; the paper's binary, built and run on an
# AMD EPYC 9354 / Zen 4), avx2, or generic (machines without AVX-512, or arm64). Absolute latencies
# depend on it; ndis and recall should agree closely but not necessarily bit-for-bit, because the SIMD
# variants sum distances in different orders.
#
# Usage:  faiss/build_faiss.sh                 (env: OPT_LEVEL, PREFIX, JOBS = compile jobs, FAISS_SRC, INSTALL_DEPS=1)
# Ubuntu packages needed:
#   build-essential gfortran libopenblas-dev libomp-dev cmake ninja-build git ca-certificates
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBS_ENV="${JOBS:-}"                      # config.sh defaults JOBS to 1 (experiment parallelism)
. "$HERE/../experiments/config.sh"

OPT_LEVEL="${OPT_LEVEL:-avx512}"
BUILD_JOBS="${BUILD_JOBS:-${JOBS_ENV:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}}"   # compile jobs: $JOBS if set, else all CPUs
FAISS_TAG=v1.15.0
FAISS_COMMIT=20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed
FAISS_SRC="${FAISS_SRC:-$REPRO/_build/faiss-src}"

case "$OPT_LEVEL" in generic|avx2|avx512) ;; *) die "OPT_LEVEL must be generic|avx2|avx512 (got $OPT_LEVEL)";; esac

if [ "${INSTALL_DEPS:-0}" = 1 ]; then
  ${SUDO-sudo} apt-get update -qq
  ${SUDO-sudo} apt-get install -y -qq --no-install-recommends \
    build-essential gfortran libopenblas-dev libomp-dev cmake ninja-build git ca-certificates
fi
for t in git cmake ninja g++; do
  command -v "$t" >/dev/null || die "missing '$t'. On Ubuntu: sudo apt-get install build-essential gfortran libopenblas-dev libomp-dev cmake ninja-build git  (or rerun with INSTALL_DEPS=1)"
done

# ---- source: clone (or reuse) the tag and check the commit
if [ ! -d "$FAISS_SRC/.git" ]; then
  mkdir -p "$(dirname "$FAISS_SRC")"
  git -c advice.detachedHead=false clone -q --depth 1 --branch "$FAISS_TAG" https://github.com/facebookresearch/faiss "$FAISS_SRC"
fi
got=$(git -C "$FAISS_SRC" rev-parse HEAD)
[ "$got" = "$FAISS_COMMIT" ] || die "faiss at $FAISS_SRC is $got, expected $FAISS_COMMIT ($FAISS_TAG)"

# ---- patches, in order. A reused tree that already carries all three is accepted as is.
P0001="$HERE/patches/0001-adaptive-termination.patch"
P0002="$HERE/patches/0002-traversal-trace.patch"
P0003="$HERE/patches/0003-patience.patch"
if git -C "$FAISS_SRC" apply --check "$P0001" 2>/dev/null; then
  git -C "$FAISS_SRC" apply "$P0001"
  git -C "$FAISS_SRC" apply --check "$P0002" || die "0002 does not apply after 0001"
  git -C "$FAISS_SRC" apply "$P0002"
  git -C "$FAISS_SRC" apply --check "$P0003" || die "0003 does not apply after 0002"
  git -C "$FAISS_SRC" apply "$P0003"
  echo "[faiss] applied 0001 + 0002 + 0003"
elif git -C "$FAISS_SRC" apply --check -R "$P0003" 2>/dev/null; then
  echo "[faiss] patches already applied in $FAISS_SRC"
elif git -C "$FAISS_SRC" apply --check -R "$P0002" 2>/dev/null; then
  git -C "$FAISS_SRC" apply --check "$P0003" || die "0003 does not apply to the 0001 + 0002 tree in $FAISS_SRC"
  git -C "$FAISS_SRC" apply "$P0003"
  echo "[faiss] applied 0003 on a tree that already carried 0001 + 0002"
else
  die "patches neither apply nor are already applied in $FAISS_SRC (delete it to start clean)"
fi

# ---- configure, build, install
B="$FAISS_SRC/build-$OPT_LEVEL"
cmake -S "$FAISS_SRC" -B "$B" -G Ninja \
  -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF \
  -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release -DFAISS_OPT_LEVEL="$OPT_LEVEL" \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" > "$B.configure.log" 2>&1 || { tail -30 "$B.configure.log"; die "cmake configure failed"; }
echo "[faiss] building ($OPT_LEVEL, -j $BUILD_JOBS) ... log: $B.build.log"
cmake --build "$B" -j "$BUILD_JOBS" > "$B.build.log" 2>&1 || { tail -30 "$B.build.log"; die "faiss build failed"; }
cmake --install "$B" > /dev/null
echo "$OPT_LEVEL" > "$PREFIX/OPT_LEVEL"
ls -l "$PREFIX"/lib/libfaiss*.a
echo "[faiss] installed into $PREFIX"
