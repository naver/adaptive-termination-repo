# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)

"""Tiny synthetic dataset for the smoke test. NOT part of the experiments: no shipped result uses it.

Gaussian clusters with uneven sizes and spreads (so that searches stop at different depths), queries drawn from the same
mixture. Written in the raw layout of the other datasets:
    <out>/base.fbin    n x d float32 (not normalized; the pipeline normalizes unless NORM=0)
    <out>/query.fbin   nq x d float32, in the order the query split uses (#0-999 test block a, #1000-1499 calibration,
                       #1500- test block b)

Usage: python3 data/make_synthetic.py [--out $RAW_DIR/synth] [--n 20000] [--nq 2000] [--d 32] [--clusters 50] [--seed 0]
Then e.g.:  data/ground_truth.sh synth && data/build_indexes.sh synth 1 5 15
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import shardlib as L  # noqa: E402


def make(n, nq, d, clusters, seed):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((clusters, d)) * 3.0
    weights = rng.dirichlet(np.full(clusters, 2.0))          # uneven cluster sizes
    spread = rng.uniform(0.5, 1.5, clusters)                  # uneven cluster widths

    def draw(m):
        c = rng.choice(clusters, size=m, p=weights)
        return (centers[c] + rng.standard_normal((m, d)) * spread[c, None]).astype(np.float32)

    return draw(n), draw(nq)


def main():
    default_raw = os.environ.get("RAW_DIR") or os.path.join(os.environ.get("WORK") or os.path.join(L.repro_root(), "work"), "raw")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(default_raw, "synth"))
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--nq", type=int, default=2_000)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--clusters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    a = ap.parse_args()
    if a.nq <= L.CALIB[1]:
        raise SystemExit(f"--nq must exceed {L.CALIB[1]} so that test block b is not empty")
    bp, qp = os.path.join(a.out, "base.fbin"), os.path.join(a.out, "query.fbin")
    if os.path.exists(bp) and os.path.exists(qp) and not a.force:
        print(f"{a.out}: exists, skip (--force to overwrite)")
        return
    os.makedirs(a.out, exist_ok=True)
    X, Q = make(a.n, a.nq, a.d, a.clusters, a.seed)
    L.write_fbin(bp, X)
    L.write_fbin(qp, Q)
    print(f"{bp}: {a.n:,} x {a.d}\n{qp}: {a.nq:,} x {a.d}")


if __name__ == "__main__":
    main()
