# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)

"""Per-shard gamma calibration from one calibration-query record: one cell (S, E) -> calib_cells.json entry.

Input: a gamma = 0 traversal record of the calibration queries (#1000-1499, bench search --trace) and their exact
ground truth (gt300_proxy.ivec). Output: one key "S{S}_E{E}" in a JSON table (merged into --out; the file is locked
so parallel cells can share it).

The experiments use `gamma_rank_shard`: each shard i chooses its own gamma_i from its own counts only.
  - The shard's true top-j list is the global exact ground truth (top 300) filtered to the shard's ids, in order.
  - found_rank[i, g, j] = number of calibration queries for which shard i, stopped by gamma g, holds its true j-th
    nearest point (by id).
  - Predicted merged recall R_hat_i(g) = sum_{j<=j_max} w_j * found_rank[i, g, j] / nq, with the binomial rank weights
    w_j = (S/K) * P[Bin(K, 1/S) >= j], j_max = min(k_i, rank, K).
  - Choice: the reference is R_hat_i(0) (rule off); gamma_i = the smallest gamma > 0 on the grid with
    R_hat_i(gamma) >= R_hat_i(0) - eps (eps = 1pp); if none, the rule is switched off for that shard (None -> 0).
  - The merged result of all shards stopping at their gamma_i is replayed on the calibration queries
    (R_rank_shard_gammas, ratio_rank_shard_gammas).
Diagnostics kept in the entry (not the method): the direct-measurement choice gamma_star (every shard replayed at a
common gamma, merged recall within eps of gamma = 0), a prefix-count prediction (found[] = shard holds its whole true
top-j; Rhat_pool / gamma_pool / gamma_shard) and the pooled rank prediction (Rhat_rank / gamma_rank).

The shard assignment is shardlib.shard_assignment (the same as shard_split.py).

`shard` is the calibration before serving (the `corpus` arm, experiments/corpus_cell.sh): one record per shard of its
held-out rows against its partial graph, the same per-shard choice, written in the same form (gamma_rank_shard).

Usage: python3 pred_calib.py cell --ds t2i [--N 10000000] --trace T.bin --gt gt300_proxy.ivec --key S15_E200 --out cells.json
                              [--gammas "0.005 0.010 ..."] [--K 20]
       python3 pred_calib.py shard --trace T0.bin T1.bin ... --gt G0.ivec G1.ivec ... --key S5_E200 --out cells.json
"""
import argparse
import fcntl
import hashlib
import json
import os
import tempfile

import numpy as np

import shardlib as L
from trace_replay import as_calibrate, choose, read_gt, read_trace, replay, stop_row

EPS = 0.01          # tolerance of the prediction-based choice (1pp)
DEFAULT_GAMMAS = " ".join(f"{g:.3f}" for g in L.CALIB_GAMMAS)      # "0.005 0.010 ... 0.100"

_shard_cache = {}


def shard_of(N, S):
    if S == 1:
        return np.zeros(N, np.int16)
    if (N, S) not in _shard_cache:
        _shard_cache[(N, S)] = L.shard_assignment(N, S).astype(np.int16)
    return _shard_cache[(N, S)]


def binomial_tail(n, p, j_max):
    """P(Bin(n, p) >= j), j = 0..j_max, by the pmf recurrence ([0] is unused)."""
    tail = [0.0] * (j_max + 1)
    if p >= 1.0:
        return [0.0] + [1.0] * j_max
    pmf = (1.0 - p) ** n
    cdf = pmf
    for j in range(1, j_max + 1):
        tail[j] = max(0.0, 1.0 - cdf)
        pmf *= ((n - j + 1) / j) * (p / (1.0 - p))
        cdf += pmf
    return tail


def merge_weights(k_user, S, j_max):
    """w_j = (S / K) * P[Bin(K, 1/S) >= j]"""
    return [v * S / k_user for v in binomial_tail(k_user, 1.0 / S, j_max)]


def pred_choose(rhat, gammas, eps=EPS):
    """rhat[0] is the rule switched off. Smallest gamma > 0 with R_hat >= R_hat(0) - eps, else None (rule off)."""
    floor = rhat[0] - eps
    for g, r in zip(gammas[1:], rhat[1:]):
        if r >= floor:
            return g
    return None


def lock_path(out):
    """Lock file for updates of table `out`, kept outside the results tree (several cells share one table)."""
    d = os.path.join(tempfile.gettempdir(), "repro-locks")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, hashlib.sha1(os.path.abspath(out).encode()).hexdigest()[:16] + ".lock")


def update_table(out, key, entry):
    """tab[key] = entry in JSON file `out`, under an exclusive lock, written atomically."""
    with open(lock_path(out), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        tab = json.load(open(out)) if os.path.exists(out) else {}
        tab[key] = entry
        json.dump(tab, open(out + ".part", "w"), indent=1, sort_keys=True)
        os.replace(out + ".part", out)


def cmd_cell(a):
    K = a.K
    N = a.N or L.ds_N(L.ds_short(a.ds))
    meta, recs = read_trace(a.trace)
    if a.nq_limit:                     # calibrate on the first nq_limit calibration queries of the record
        recs = [r for r in recs if r[1] < a.nq_limit]
        meta = dict(meta, nq=min(meta["nq"], a.nq_limit))
    S, nq, ki, rule_k = meta["S"], meta["nq"], meta["ki"], meta["rule_k"]
    gt = read_gt(a.gt, nq)
    if gt.max() >= N:
        raise SystemExit(f"!! ground truth ids exceed N={N} (wrong --N or --ds?)")
    gammas = ["0"] + a.gammas.split()
    G = len(gammas)
    j_max = min(ki, rule_k, K)
    w = merge_weights(K, S, j_max)
    so = shard_of(N, S)

    # the shard's true top list: gt300 filtered to the shard's ids, first j_max
    short = 0
    inner = {}
    for q in range(nq):
        sq = so[gt[q]]
        for s in range(S):
            lst = gt[q][sq == s][:j_max]
            if len(lst) < j_max:
                short += 1
            inner[(s, q)] = lst

    # found[s, g, j]: queries for which shard s, stopped at gamma_g, holds all of its true top j (prefix condition)
    found = np.zeros((S, G, j_max + 1), np.int64)
    # found_rank[s, g, j]: queries for which shard s holds its true j-th point (the expectation of merged recall
    # decomposes over these, not over the prefix condition)
    found_rank = np.zeros((S, G, j_max + 1), np.int64)
    stop_ids = {}                                   # (s, q) -> [result id set per gamma]
    for s, q, upper, d0, dk, nd, gid in recs:
        lst = inner[(s, q)]
        per_g = []
        for gi, gs in enumerate(gammas):
            ids = set(int(x) for x in gid[stop_row(d0, dk, gs)] if x >= 0)
            per_g.append(ids)
            each = [int(x) in ids for x in lst]
            hit = np.cumprod(each) if len(lst) else np.zeros(0)
            for j in range(1, len(lst) + 1):
                found[s, gi, j] += int(hit[j - 1])
                found_rank[s, gi, j] += int(each[j - 1])
        stop_ids[(s, q)] = per_g

    def rhat(cnt, n):
        return [sum(w[j] * cnt[gi, j] / n for j in range(1, j_max + 1)) for gi in range(G)]

    rh_pool = rhat(found.sum(0), S * nq)
    rh_shard = [rhat(found[s], nq) for s in range(S)]
    rh_rank = rhat(found_rank.sum(0), S * nq)
    g_pool = pred_choose(rh_pool, gammas)
    g_shard = [pred_choose(r, gammas) for r in rh_shard]

    # direct measurement (diagnostic): replay every shard at a common gamma, same rounding and choice as a sweep
    pairs, R, ND = [], [], []
    for gs in gammas:
        r, nd = replay(meta, recs, gt, K, gs)
        rr, nn = as_calibrate(r, nd)
        pairs.append((float(gs), rr, nn)); R.append(r); ND.append(nd)
    g_star, r_star, ratio_star = choose(pairs, "1.0")

    def merged_recall(gidx):
        """calibration-query merged recall when shard s stops at gammas[gidx[s]]"""
        tot = 0.0
        for q in range(nq):
            u = set().union(*(stop_ids[(s, q)][gidx[s]] for s in range(S)))
            tot += len(set(int(x) for x in gt[q, :K]) & u) / K
        return tot / nq

    def gi_of(g):
        return None if g is None else gammas.index(g)

    res = {
        "S": S, "k_i": ki, "rule_k": rule_k, "ef": meta["ef"], "nq": nq, "j_max": j_max, "short_lists": short,
        "gammas": gammas, "R": R, "ndis": ND, "Rhat_pool": rh_pool, "Rhat_shard": rh_shard, "Rhat_rank": rh_rank,
        "gamma_rank": pred_choose(rh_rank, gammas),
        "gamma_star": g_star, "recall_star": r_star, "ratio_star": ratio_star,
        "gamma_pool": None if g_pool is None else float(g_pool),
        "gamma_shard": [None if g is None else float(g) for g in g_shard],
    }
    gs_i = [float(x) for x in gammas].index(float(g_star))
    res["Rhat_minus_R_at_star"] = rh_pool[gs_i] - R[gs_i]
    res["Rhat_minus_R_natural"] = rh_pool[0] - R[0]
    res["Rhat_rank_minus_R_at_star"] = rh_rank[gs_i] - R[gs_i]
    if g_pool is not None:
        i = gi_of(g_pool)
        res["R_at_pool"] = R[i]; res["ratio_at_pool"] = ND[i] / ND[0]
    # every shard at its own prefix-count gamma (None = rule off)
    gi_shard = [0 if g is None else gi_of(g) for g in g_shard]
    res["R_shard_gammas"] = merged_recall(gi_shard)
    nd_sh = sum(upper + int(nd[stop_row(d0, dk, gammas[gi_shard[s]])]) for s, q, upper, d0, dk, nd, gid in recs) / nq
    res["ratio_shard_gammas"] = nd_sh / ND[0]
    # the method: per-rank counts, per-shard gamma_i, and the merged result of stopping every shard at its gamma_i
    rh_rank_shard = [rhat(found_rank[s], nq) for s in range(S)]
    g_rank_shard = [pred_choose(r, gammas) for r in rh_rank_shard]
    gi_rs = [0 if g is None else gi_of(g) for g in g_rank_shard]
    res["Rhat_rank_shard"] = rh_rank_shard
    res["gamma_rank_shard"] = [None if g is None else float(g) for g in g_rank_shard]
    res["R_rank_shard_gammas"] = merged_recall(gi_rs)
    res["ratio_rank_shard_gammas"] = sum(upper + int(nd[stop_row(d0, dk, gammas[gi_rs[s]])])
                                         for s, q, upper, d0, dk, nd, gid in recs) / nq / ND[0]

    update_table(a.out, a.key, res)
    gsh = res["gamma_rank_shard"]
    real = [g for g in gsh if g is not None]
    print(f"[calib] {a.key}  gamma_i {min(real) if real else '-'}..{max(real) if real else '-'} "
          f"(off {gsh.count(None)}/{S})  calib recall {res['R_rank_shard_gammas']:.4f}  "
          f"ndis ratio {res['ratio_rank_shard_gammas']:.4f}  short lists {short}")


def cmd_shard(a):
    """Calibration before serving (the `corpus` arm): every shard chooses gamma_i from its own held-out rows, searched
    alone against its partial graph (lib/holdout.py). Same choice as `gamma_rank_shard` of `cell`, with the shard's true
    list given directly (exact top list of the held-out row among the shard's other rows) and the weights w_j of the
    real S. There is no merged replay: the shards' calibration queries differ."""
    if len(a.trace) != len(a.gt):
        raise SystemExit("!! one --gt per --trace")
    S, K = len(a.trace), a.K
    gammas = ["0"] + a.gammas.split()
    G = len(gammas)
    res = {"S": S, "gammas": gammas, "source": "held-out shard rows, partial graph",
           "nq": [], "short_lists": [], "Rhat_rank_shard": [], "gamma_rank_shard": [], "ndis": [], "ratio_rank_shard": []}
    for s, (tf, gf) in enumerate(zip(a.trace, a.gt)):
        meta, recs = read_trace(tf)
        if meta["S"] != 1:
            raise SystemExit(f"!! {tf}: expected a single-shard record (S=1), got S={meta['S']}")
        nq, ki, rule_k = meta["nq"], meta["ki"], meta["rule_k"]
        if s == 0:
            j_max = min(ki, rule_k, K)
            w = merge_weights(K, S, j_max)
            res.update(k_i=ki, rule_k=rule_k, ef=meta["ef"], j_max=j_max)
        elif (ki, rule_k, meta["ef"]) != (res["k_i"], res["rule_k"], res["ef"]):
            raise SystemExit(f"!! {tf}: k_i/rank/ef differ from shard 0")
        gt = read_gt(gf, nq)
        found_rank = np.zeros((G, j_max + 1), np.int64)
        nd_sum = np.zeros(G)
        short = 0
        for _, q, upper, d0, dk, nd, gid in recs:
            lst = [int(x) for x in gt[q][:j_max] if x >= 0]
            short += len(lst) < j_max
            for gi, gs in enumerate(gammas):
                row = stop_row(d0, dk, gs)
                ids = set(int(x) for x in gid[row] if x >= 0)
                for j, x in enumerate(lst, 1):
                    found_rank[gi, j] += x in ids
                nd_sum[gi] += upper + int(nd[row])
        rh = [sum(w[j] * found_rank[gi, j] / nq for j in range(1, j_max + 1)) for gi in range(G)]
        g = pred_choose(rh, gammas)
        ndq = (nd_sum / nq).tolist()
        res["nq"].append(nq); res["short_lists"].append(short)
        res["Rhat_rank_shard"].append(rh); res["ndis"].append(ndq)
        res["gamma_rank_shard"].append(None if g is None else float(g))
        res["ratio_rank_shard"].append(ndq[0 if g is None else gammas.index(g)] / ndq[0])
    update_table(a.out, a.key, res)
    gsh = res["gamma_rank_shard"]
    real = [g for g in gsh if g is not None]
    print(f"[calib corpus] {a.key}  gamma_i {min(real) if real else '-'}..{max(real) if real else '-'} "
          f"(off {gsh.count(None)}/{S})  ndis ratio {np.mean(res['ratio_rank_shard']):.4f}  "
          f"short lists {sum(res['short_lists'])}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("shard", help="calibrate one cell (S, E) from per-shard held-out records (corpus arm)")
    c.add_argument("--trace", nargs="+", required=True, help="one single-shard record per shard, in shard order")
    c.add_argument("--gt", nargs="+", required=True, help="the held-out rows' global ground truth, one per shard")
    c.add_argument("--key", required=True); c.add_argument("--out", required=True)
    c.add_argument("--gammas", default=DEFAULT_GAMMAS)
    c.add_argument("--K", type=int, default=20)
    c = sub.add_parser("cell", help="calibrate one cell (S, E)")
    c.add_argument("--ds", required=True, help="dataset id (t2i, msmarco, glove, sift) or <ds>_N<N>")
    c.add_argument("--N", type=int, default=0, help="number of base vectors (default: the dataset's N)")
    c.add_argument("--trace", required=True); c.add_argument("--gt", required=True)
    c.add_argument("--key", required=True); c.add_argument("--out", required=True)
    c.add_argument("--gammas", default=DEFAULT_GAMMAS)
    c.add_argument("--K", type=int, default=20, help="results per query (the K-axis experiment varies it)")
    c.add_argument("--nq_limit", type=int, default=0,
                   help="use only the first n calibration queries of the record (125, 250; 0 = all 500)")
    a = ap.parse_args()
    {"cell": cmd_cell, "shard": cmd_shard}[a.cmd](a)


if __name__ == "__main__":
    main()
