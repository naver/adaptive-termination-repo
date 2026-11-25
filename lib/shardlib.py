# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)

"""Shared constants and helpers for the reproducibility package.

Everything that more than one script needs lives here: dataset table, query split, per-shard result count k_i,
binomial rank weights w_j, and fbin/ivec I/O.

File formats
    fbin : int32 n, int32 d, then n*d float32 (row major)
    ivec : int32 n, int32 k, then n*k int32 (row major) -- ground-truth neighbor ids, nearest first
    sel  : int32 n, int32 1, then n int32 global ids     -- rows of a shard, in shard order
"""
import os
from math import ceil, comb, sqrt
from statistics import NormalDist

import numpy as np

# ------------------------------------------------------------------ datasets
# name -> (N used in the experiments, normalize to unit length?, number of queries in query.fbin we use)
DATASETS = {
    "t2i": dict(N=10_000_000, normalize=True, nq=100_000, d=200),
    "msmarco": dict(N=8_841_823, normalize=True, nq=6_980, d=1024),
    "glove": dict(N=1_183_514, normalize=True, nq=10_000, d=200),
    "sift": dict(N=1_000_000, normalize=False, nq=10_000, d=128),
    # 100M-scale extension (fixed ef vs adaptive termination only; not in DATASETS_ALL of data/common.sh)
    "deep": dict(N=100_000_000, normalize=False, nq=10_000, d=96),
    "t2i100m": dict(N=100_000_000, normalize=True, nq=100_000, d=200),
}
# S values per dataset
S_VALUES = {"t2i": (1, 5, 15, 30, 50), "msmarco": (1, 5, 15, 30, 50), "glove": (1, 5, 15), "sift": (1, 5, 15),
            "deep": (5, 15), "t2i100m": (5, 15)}

K = 20                    # results per query (fixed except in the K-axis experiment)
M, EFC = 40, 200          # HNSW parameters
SEED = 0                  # shard assignment seed
EPS_PP = 1.0              # tolerance epsilon, percentage points
E_LIMITS = (100, 200, 300)
CALIB_GAMMAS = [round(0.005 * i, 3) for i in range(1, 21)]      # 0.005 .. 0.100, calibration grid (20 values)

# query split (absolute query numbers in query.fbin)
EVAL_A = (0, 1000)        # test block a: #0-999
CALIB = (1000, 1500)      # calibration queries: #1000-1499 (500)
BLOCK_B_START = 1500      # test block b: #1500 - (end of block)
BLOCK_B_N = {"t2i": 19_000, "msmarco": 5_480, "glove": 8_500, "sift": 8_500, "deep": 8_500, "t2i100m": 19_000}
GT_K = 300                # ground-truth depth stored on disk


def ds_N(ds):
    """N of a dataset. Datasets other than the six (e.g. the smoke test's "synth") take N from the environment, N_<DS>."""
    if ds in DATASETS:
        return DATASETS[ds]["N"]
    v = os.environ.get(f"N_{ds.upper()}")
    if v is None:
        raise KeyError(f"unknown dataset {ds!r}: set N_{ds.upper()} for datasets other than the six")
    return int(v)


def dsn(ds):
    """'t2i' -> 't2i_N10000000' (directory names carry N)"""
    return ds if "_N" in ds else f"{ds}_N{ds_N(ds)}"


def ds_short(name):
    return name.split("_N")[0]


def norm_flag(ds):
    d = ds_short(ds)
    return (1 if DATASETS[d]["normalize"] else 0) if d in DATASETS else int(os.environ.get(f"NORM_{d.upper()}", "0"))


# ------------------------------------------------------------------ per-shard result count and rank weights
def shard_result_count(K, S, p=0.9999):
    """k_i: the number of results a shard returns.

    Closed form of the Clarke-Terra criterion (CIKM 2004): the smallest count for which every one of the global
    top-K lands within the per-shard limit with probability >= p. Normal approximation to Bin(K, 1/S) plus a union
    bound over the S shards: k_i = min(K, ceil(mu + z_S * sigma)), z_S = Phi^-1(1 - (1 - p)/S).
    """
    if S == 1:
        return K
    mu = K / S
    sig = sqrt(mu * (1 - 1 / S))
    z = NormalDist().inv_cdf(1 - (1 - p) / S)
    return min(K, max(1, ceil(mu + z * sig)))


def k_i_of(K, S):
    return shard_result_count(K, S)


def rule_rank(mode, K, S):
    """Rank the stopping test compares against. mode: rule (adaptive termination, k_i) | dabs (rank-K termination, K)."""
    if mode == "rule":
        return k_i_of(K, S)
    if mode == "dabs":
        return K
    raise ValueError(mode)


def w_j(K, S, k_i=None):
    """Binomial rank weights w_j = (S/K) * P[Bin(K, 1/S) >= j], j = 1..k_i. S=1 -> 1/K each."""
    k_i = k_i or k_i_of(K, S)
    if S == 1:
        return [1.0 / K] * min(k_i, K)
    p = 1.0 / S
    return [(S / K) * sum(comb(K, i) * p ** i * (1 - p) ** (K - i) for i in range(j, K + 1)) for j in range(1, k_i + 1)]


# ------------------------------------------------------------------ I/O
def read_fbin(path, start=0, count=None):
    n, d = np.fromfile(path, dtype=np.int32, count=2)
    n, d = int(n), int(d)
    count = n - start if count is None else min(count, n - start)
    return np.fromfile(path, dtype=np.float32, count=count * d, offset=8 + start * d * 4).reshape(count, d)


def write_fbin(path, a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    with open(path, "wb") as f:
        f.write(np.array(a.shape, np.int32).tobytes())
        f.write(a.tobytes())


def read_ivec(path):
    n, k = np.fromfile(path, dtype=np.int32, count=2)
    return np.fromfile(path, dtype=np.int32, offset=8).reshape(int(n), int(k))


def write_ivec(path, a):
    a = np.ascontiguousarray(a, dtype=np.int32)
    with open(path, "wb") as f:
        f.write(np.array(a.shape, np.int32).tobytes())
        f.write(a.tobytes())


def normalize_rows(X):
    X = np.asarray(X, dtype=np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def shard_assignment(N, S, seed=SEED):
    """shard_of[i] for global row i: a seeded random permutation dealt round-robin (equal-size hash partitioning)."""
    shard_of = np.empty(N, np.int32)
    shard_of[np.random.default_rng(seed).permutation(N)] = np.arange(N) % S
    return shard_of


def repro_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ per-query dumps (DUMPS_DIR)
# Two layouts are accepted:
#   package run  (written by experiments/*.sh under RESULTS):
#       dumps/<ds>_S{S}_ef{E}_{arm}_{blk}.*   arm in nat | rule | dabs | corpus | p125 | p250 | approx
#       shard/<dsn>/S{S}_ef{E}_{fixed|adaptive}_{blk}.*      cpu_shard/<dsn>/...
#   export of the original study's dumps (not shipped with this package):
#       per_query/<ds>_S{S}_ef{E}_{arm}_{blk}.*  with the original arm names: nat, dabs, ki = rank k_i where k_i < 8
#       (S >= 15 at K = 20), ours = rank k_i where k_i >= 8 (S <= 5; at S >= 15 the original "ours" used another rank and
#       is not read)
#       per_shard_ndis/<dsn>/...                  per_shard_cpu/<dsn>/...
# At S = 1 adaptive termination and rank-K termination are one configuration (rank K); for T2I only the rank-K termination dump exists there.
LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"


def dumps_dir(results=None):
    return os.environ.get("DUMPS_DIR") or results or os.environ.get("RESULTS") or os.path.join(repro_root(), "results")


def is_lfs_pointer(path):
    try:
        with open(path, "rb") as f:
            return f.read(len(LFS_POINTER)) == LFS_POINTER
    except OSError:
        return False


def _usable(p):
    return os.path.isfile(p) and not is_lfs_pointer(p)


def _export_arms(arm, S, K=K):
    if arm == "rule":
        return ["ours"] if k_i_of(K, S) >= 8 else ["ki"]
    return [arm]


def dump_path(root, kind, ds, S, E, arm, blk, ext="tsv", K=K):
    """Path of one dump file in either layout, or None if absent (or a Git LFS pointer stub).
    kind: per_query (arm = nat|rule|dabs|...), shard or cpu_shard (arm = fixed|adaptive)."""
    ds = ds_short(ds)
    if kind == "per_query":
        arms = [arm] + (["dabs"] if S == 1 and arm == "rule" else [])
        for a in arms:
            cands = [os.path.join(root, "dumps", f"{ds}_S{S}_ef{E}_{a}_{blk}.{ext}")]
            cands += [os.path.join(root, "per_query", f"{ds}_S{S}_ef{E}_{x}_{blk}.{ext}") for x in _export_arms(a, S, K)]
            for p in cands:
                if _usable(p):
                    return p
        return None
    sub = {"shard": ("shard", "per_shard_ndis"), "cpu_shard": ("cpu_shard", "per_shard_cpu")}[kind]
    for s in sub:
        p = os.path.join(root, s, dsn(ds), f"S{S}_ef{E}_{arm}_{blk}.{ext}")
        if _usable(p):
            return p
    return None
