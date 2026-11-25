# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)

"""Split base.fbin into S shards (equal-size hash partitioning) for `bench build` and `bench search`.

Assignment: shardlib.shard_assignment -- perm = numpy default_rng(0).permutation(N); shard_of[perm] = arange(N) % S.
It uses numpy's PCG64 permutation, which is why this step stays in Python (not reproducible in C++).

Outputs in --out:
    shard{s}.fbin   int32 n, int32 d, float32 rows        -> input of `bench build`
    shard{s}.sel    int32 n, int32 1, int32 global ids     -> `bench search` maps shard results back to global ids

Usage: python3 shard_split.py --base BASE.fbin --out DIR --S 5 [--N 10000000] [--normalize] [--chunk 2000000]
"""
import argparse
import os
import sys

import numpy as np

import shardlib as L


def read_fbin_header(path):
    with open(path, "rb") as f:
        n, d = np.frombuffer(f.read(8), dtype=np.int32)
    return int(n), int(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--S", type=int, required=True); ap.add_argument("--N", type=int, default=0,
                                                                    help="use the first N rows (default: all)")
    ap.add_argument("--normalize", action="store_true",
                    help="normalize rows to unit length (angular datasets; searched with squared L2)")
    ap.add_argument("--chunk", type=int, default=2_000_000)
    a = ap.parse_args()

    if not os.path.isfile(a.base):
        sys.exit(f"!! base vectors not found: {a.base}")
    n_file, d = read_fbin_header(a.base)
    if os.path.getsize(a.base) < 8 + n_file * d * 4:
        sys.exit(f"!! {a.base} is shorter than its header says ({n_file} x {d})")
    N = a.N if a.N > 0 else n_file
    if N > n_file:
        sys.exit(f"!! N={N:,} > {n_file:,} rows in {a.base}")
    os.makedirs(a.out, exist_ok=True)

    shard_of = L.shard_assignment(N, a.S)
    counts = np.bincount(shard_of, minlength=a.S)
    print(f"[shard] N={N:,} d={d} S={a.S} shard sizes {counts.min():,}..{counts.max():,}", flush=True)

    # global id lists first: the search maps shard-local results back to global ids with them
    for s in range(a.S):
        ids = np.nonzero(shard_of == s)[0].astype(np.int32)
        with open(os.path.join(a.out, f"shard{s}.sel"), "wb") as f:
            f.write(np.array([len(ids), 1], np.int32).tobytes()); f.write(ids.tobytes())

    # vectors are streamed in chunks and appended to the shard files (a 10M x 200 base does not fit at once)
    fh = [open(os.path.join(a.out, f"shard{s}.fbin"), "wb") for s in range(a.S)]
    for s in range(a.S):
        fh[s].write(np.array([counts[s], d], np.int32).tobytes())
    with open(a.base, "rb") as f:
        f.seek(8)
        done = 0
        while done < N:
            m = min(a.chunk, N - done)
            X = np.frombuffer(f.read(m * d * 4), dtype=np.float32).reshape(m, d)
            if a.normalize:
                X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
            part = shard_of[done:done + m]
            for s in range(a.S):
                sub = X[part == s]
                if len(sub):
                    fh[s].write(np.ascontiguousarray(sub, np.float32).tobytes())
            done += m
            print(f"  {done:,}/{N:,}", flush=True)
    for f_ in fh:
        f_.close()
    print(f"[shard] -> {a.out}")


if __name__ == "__main__":
    main()
