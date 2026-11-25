#!/usr/bin/env python3
# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)

"""When does the stopping test become able to fire, and where does it stop? One traversal record (gamma = 0), streamed.

The record format is documented in lib/trace_replay.py. Per shard search (query x shard):
  pops_to_have   first base-layer pop (1-based) at which d_k is defined, i.e. the search has found as many points as the rank
                 it compares against (dk is NaN before). None if it never has them (counted as never_have)
  base_at_have   cumulative base-layer distance computations at that pop
  stop           the row where the rule with gamma g stops (test `have && d0 > (1+g)^2 * dk` in float32, right after the pop,
                 as in lib/trace_replay.py); the last row if it never fires
  stop_at_have   the stop is the first pop at which the rank was filled
  base_at_stop   cumulative base-layer distance computations at the stop row
A shard search is "empty" (non-contributing) when the final result of the gamma = 0 search holds none of the global top-K,
the definition the oracle uses for its 'empty' share.

Usage: python3 rank_mech.py --trace T.bin --gt gt300.ivec --K 100 --gamma 0.0123 [--gamma ...] > summary.json
"""
import argparse
import json
import sys

import numpy as np

MAGIC = b"V31T"


def records(path):
    """Stream (meta, s, q, upper, d0, dk, nd, gid) per shard search."""
    with open(path, "rb") as f:
        if f.read(4) != MAGIC:
            sys.exit(f"!! not a traversal record: {path}")
        S, nq, ki, rule_k, ef = np.frombuffer(f.read(20), np.int32)
        meta = dict(S=int(S), nq=int(nq), ki=int(ki), rule_k=int(rule_k), ef=int(ef))
        while True:
            h = f.read(16)
            if not h:
                break
            s, q, npop, upper = np.frombuffer(h, np.int32)
            d0 = np.frombuffer(f.read(4 * npop), np.float32)
            dk = np.frombuffer(f.read(4 * npop), np.float32)
            nd = np.frombuffer(f.read(4 * npop), np.int32)
            gid = np.frombuffer(f.read(4 * npop * ki), np.int32).reshape(npop, ki)
            yield meta, int(s), int(q), int(upper), d0, dk, nd, gid


def stop_row(d0, dk, gamma_str):
    """Row where the test with this gamma fires (the last row is the end state, not a pop)."""
    g = np.float32(float(gamma_str))
    if g <= 0:
        return len(d0) - 1
    factor = (np.float32(1.0) + g) * (np.float32(1.0) + g)
    body = slice(0, len(d0) - 1)
    with np.errstate(invalid="ignore"):
        fire = ~np.isnan(dk[body]) & (d0[body] > (factor * dk[body]).astype(np.float32))
    hit = np.flatnonzero(fire)
    return int(hit[0]) if hit.size else len(d0) - 1


def pct(xs, p):
    return float(np.percentile(xs, p)) if len(xs) else None


def summarize(rows, gammas):
    n = len(rows)
    have = [r for r in rows if r["pops_to_have"] is not None]
    ph = [r["pops_to_have"] for r in have]
    bh = [r["base_at_have"] for r in have]
    out = {"n": n, "never_have": n - len(have),
           "pops_to_have": {"min": min(ph, default=None), "p1": pct(ph, 1), "median": pct(ph, 50), "p99": pct(ph, 99),
                            "max": max(ph, default=None)},
           "base_at_have": {"min": min(bh, default=None), "p1": pct(bh, 1), "median": pct(bh, 50), "p99": pct(bh, 99)},
           "upper_ndis": {"median": pct([r["upper"] for r in rows], 50)},
           "npop_total": {"median": pct([r["npop"] for r in rows], 50)},
           "base_end": {"median": pct([r["base_end"] for r in rows], 50)},
           "at_gamma": {}}
    for g in gammas:
        st = [r["stop"][g] for r in rows]
        fired = [x for x in st if x["fired"]]
        base = [x["base"] for x in st]
        out["at_gamma"][g] = {
            "fired_frac": len(fired) / n if n else None,
            "stop_at_have_frac": sum(x["stop_at_have"] for x in st) / n if n else None,
            "stop_at_have_frac_of_fired": (sum(x["stop_at_have"] for x in fired) / len(fired)) if fired else None,
            "stop_within_2pops_of_have_frac": sum(x["within2"] for x in st) / n if n else None,
            "base_at_stop": {"min": min(base, default=None), "p1": pct(base, 1), "p10": pct(base, 10),
                             "median": pct(base, 50), "mean": float(np.mean(base)) if base else None},
            "pops_at_stop": {"min": min((x["pop"] for x in st), default=None),
                             "median": pct([x["pop"] for x in st], 50)},
        }
    return out


def analyze(trace, gt_path, K, gammas):
    n, k = np.fromfile(gt_path, np.int32, 2)
    gt = np.fromfile(gt_path, np.int32, offset=8).reshape(int(n), int(k))
    truth = [set(int(x) for x in gt[q, :K]) for q in range(int(n))]
    rows, meta = [], None
    for meta, s, q, upper, d0, dk, nd, gid in records(trace):
        npop = len(d0) - 1
        hv = np.flatnonzero(~np.isnan(dk[:npop]))
        first = int(hv[0]) if hv.size else None
        final = set(int(x) for x in gid[-1] if x >= 0)
        r = {"s": s, "q": q, "upper": upper, "npop": npop, "base_end": int(nd[-1]),
             "empty": len(final & truth[q]) == 0,
             "pops_to_have": None if first is None else first + 1,
             "base_at_have": None if first is None else int(nd[first]), "stop": {}}
        for g in gammas:
            st = stop_row(d0, dk, g)
            hit = st < npop
            r["stop"][g] = {"fired": hit, "pop": st + 1 if hit else npop, "base": int(nd[st]),
                            "stop_at_have": hit and first is not None and st == first,
                            "within2": hit and first is not None and st <= first + 2}
        rows.append(r)
    return dict(meta or {}, K=K, gammas=list(gammas), nrec=len(rows),
                all=summarize(rows, gammas),
                empty=summarize([r for r in rows if r["empty"]], gammas),
                useful=summarize([r for r in rows if not r["empty"]], gammas))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--gamma", action="append", required=True, help="gamma to replay (kept as given); repeatable")
    a = ap.parse_args()
    json.dump(analyze(a.trace, a.gt, a.K, a.gamma), sys.stdout, indent=1)


if __name__ == "__main__":
    main()
