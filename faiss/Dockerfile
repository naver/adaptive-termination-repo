# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#
# Patched Faiss + benchmark driver in a container.
#
# Build from the package root (repro/):
#   docker build -f faiss/Dockerfile -t pef-bench .                                   # avx512 (paper)
#   docker build -f faiss/Dockerfile --build-arg OPT_LEVEL=generic --build-arg JOBS=2 -t pef-bench .
# Run with the package and the data mounted; BIN already points at the driver in the image:
#   docker run --rm -it -v "$PWD":/repro -v /path/to/work:/repro/work pef-bench bash
#
# The driver is compiled with -march=native, i.e. for the build machine's CPU. The paper used
# OPT_LEVEL=avx512 on an AMD EPYC 9354 (Zen 4). Other levels change absolute latencies; ndis and recall
# should agree closely but not necessarily bit-for-bit (SIMD variants sum distances in different orders).
#
# Tested: OPT_LEVEL=generic, JOBS=2 on arm64 (image ~480 MB with the Python venv, about 3 min to build). The avx512 and avx2
# variants were not built in this container (they need an x86-64 build machine).

FROM ubuntu:24.04 AS build
ARG OPT_LEVEL=avx512
ARG JOBS=8
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
      build-essential gfortran libopenblas-dev libomp-dev cmake ninja-build git ca-certificates \
      binutils python3 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src/repro
COPY experiments/config.sh experiments/config.sh
COPY faiss/ faiss/
COPY bench/ bench/
RUN OPT_LEVEL=$OPT_LEVEL JOBS=$JOBS PREFIX=/opt/faiss FAISS_SRC=/src/faiss faiss/build_faiss.sh \
 && OPT_LEVEL=$OPT_LEVEL PREFIX=/opt/faiss BIN=/opt/bench/bin/bench bench/build_bench.sh \
 && rm -rf /src/faiss

FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
      libopenblas0 libgomp1 libgfortran5 python3 python3-venv numactl \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir "numpy==2.0.2" "matplotlib==3.9.4"
# Python packages live in a venv: figures/ needs matplotlib >= 3.7 (Ubuntu 24.04 ships 3.6). The data/ converters also
# need h5py, pyarrow and pandas:  /opt/venv/bin/pip install h5py pyarrow pandas
ENV PATH=/opt/venv/bin:$PATH
COPY --from=build /opt/bench/bin/bench /opt/bench/bin/bench
ENV BIN=/opt/bench/bin/bench
WORKDIR /repro
CMD ["bash"]
