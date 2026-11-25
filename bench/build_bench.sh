#!/usr/bin/env bash
# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#
# Build the benchmark driver (bench.cpp) against the patched Faiss installed by faiss/build_faiss.sh.
#
# **Link exactly one Faiss library.** With FAISS_OPT_LEVEL=avx512, Faiss builds libfaiss.a (generic),
# libfaiss_avx2.a and libfaiss_avx512.a. They define the same symbols, so linking several makes the
# linker pick whichever comes first -- in a test that silently linked the generic one (42 zmm
# instructions in the binary, against 54,105 in libfaiss_avx512.a itself), which would have made every
# absolute latency wrong. The zmm/ymm counts below catch that.
#
# Usage:  bench/build_bench.sh        (env: OPT_LEVEL=generic|avx2|avx512, default = the level faiss was
#                                      built with, else avx512; PREFIX, BIN from experiments/config.sh)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/../experiments/config.sh"

LEVEL="${OPT_LEVEL:-$(cat "$PREFIX/OPT_LEVEL" 2>/dev/null || echo avx512)}"
OUT="$BIN"            # overwriting a running binary fails with "Text file busy": set BIN to build elsewhere
mkdir -p "$(dirname "$OUT")"

case "$LEVEL" in
  generic) LIB="$PREFIX/lib/libfaiss.a" ;;
  avx2)    LIB="$PREFIX/lib/libfaiss_avx2.a" ;;
  avx512)  LIB="$PREFIX/lib/libfaiss_avx512.a" ;;
  *) die "OPT_LEVEL must be generic|avx2|avx512 (got $LEVEL)" ;;
esac
[ -f "$LIB" ] || die "not found: $LIB (run faiss/build_faiss.sh with OPT_LEVEL=$LEVEL)"

# The driver itself uses -march=native. What is measured is Faiss; the driver is thin.
# -std=c++17 as for the paper's binary (the Faiss headers' C++20 lambda syntax is accepted as an extension).
ARCH_FLAGS="${ARCH_FLAGS:--march=native}"
g++ -O3 $ARCH_FLAGS -std=c++17 -fopenmp \
    -I"$PREFIX/include" \
    "$HERE/bench.cpp" \
    -o "$OUT" \
    "$LIB" -lopenblas -lgomp -lpthread

# Check that the intended variant was linked. Silently getting generic is the worst outcome.
if command -v objdump >/dev/null; then
  Z=$(objdump -d "$OUT" | grep -c zmm || true)
  Y=$(objdump -d "$OUT" | grep -c ymm || true)
  echo "[build] $OUT  level=$LEVEL zmm=$Z ymm=$Y"
  case "$LEVEL" in
    avx512) [ "$Z" -gt 1000 ] || die "avx512 requested but only $Z zmm instructions" ;;
    avx2)   [ "$Y" -gt 10000 ] || die "avx2 requested but only $Y ymm instructions"
            [ "$Z" -lt 1000 ] || die "avx2 requested but $Z zmm instructions" ;;
  esac
else
  echo "[build] $OUT  level=$LEVEL (objdump not found: SIMD variant check skipped)"
fi
ls -l "$OUT"
