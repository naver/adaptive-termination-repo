# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)

"""Comparison rank at K = 20 and K = 100: the rule (rank k_i, arm "rule") against rank-K termination (rank K, arm "dabs").

Cost: on each arm's gamma-grid curve (test block a), the distance computations at the recall 1pp (and 2pp) below that of
fixed ef (the gamma = 0 point, shared by both arms), linear in recall; ratio = dabs / rule. The same definition as the
K = 20 rank comparison. Floor: the dabs curve's point at the smallest gamma of the grid. A rank-K test fires only when the
candidate is farther than the K-th point found, so its recall does not fall below this point as gamma goes to 0; a target
below it has no dabs value ("below floor"), and the ratio at the floor recall is given instead.

Mechanism (subcommand mech): lib/rank_mech.py on each arm's gamma = 0 record, replayed at the gamma with which that arm
reaches the target (the floor gamma where it cannot reach it).

Inputs: K = 100 grids $RESULTS/cells_K100/<dsn>/{,dabs/}grid/S{S}_E{E}.json (experiments/run_extras.sh rank100),
K = 20 grids $RESULTS/cells/<dsn>/{,dabs/}grid/ (experiments/run_core.sh); records under $TRACE_DIR/K100/<dsn>/<arm>/ and
$TRACE_DIR/<dsn>/<arm>/ (mech only). Outputs: $RESULTS/rank_k100/summary.json, $RESULTS/rank_k100/mech/<dsn>/K{K}_{arm}_S{S}_E{E}.json.

Usage: python3 rank_k100.py summary [--results DIR]
       python3 rank_k100.py mech [--results DIR] [--trace-dir DIR] [--idx-dir DIR]
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import shardlib as L  # noqa: E402

DS = ("t2i", "msmarco")
SS = (5, 15, 30, 50)
ES = {100: (200, 300), 20: (200, 300)}      # ef < K is not run at K = 100; K = 20 uses the same two limits here
DROPS = (1.0, 2.0)


def ndis_at(curve, t):
    """ndis at recall t, linear in recall along (gamma, recall, ndis); None outside the curve's recall range."""
    pts = sorted((r, n) for _, r, n in curve)
    if t <= pts[0][0] or t >= pts[-1][0]:
        return None
    for (r0, n0), (r1, n1) in zip(pts, pts[1:]):
        if r0 <= t <= r1:
            return n0 if r1 == r0 else n0 + (n1 - n0) * (t - r0) / (r1 - r0)
    return None


def gamma_at(curve, t):
    """gamma that reaches recall t (gamma > 0 points only), linear in recall; None outside the range."""
    pts = sorted((r, g) for g, r, _ in curve if g > 0)
    if t <= pts[0][0] or t >= pts[-1][0]:
        return None
    for (r0, g0), (r1, g1) in zip(pts, pts[1:]):
        if r0 <= t <= r1:
            return g0 if r1 == r0 else g0 + (g1 - g0) * (t - r0) / (r1 - r0)
    return None


def grid(results, K, ds, S, E, arm):
    root = "cells" if K == 20 else f"cells_K{K}"
    f = os.path.join(results, root, L.dsn(ds), "" if arm == "rule" else arm, "grid", f"S{S}_E{E}.json")
    return json.load(open(f)) if os.path.exists(f) else None


def cell(gr, gd):
    tup = lambda g: [(p["gamma"], p["recall"], p["ndis"]) for p in g["curve"]]
    cr, cd = tup(gr), tup(gd)
    r0 = gr["curve"][0]["recall"]
    fl = min((p for p in cd if p[0] > 0), key=lambda p: p[0])
    nr_fl = ndis_at(cr, fl[1])
    out = {"recall_fixed": r0, "ndis_fixed": gr["curve"][0]["ndis"],
           "fixed_same": gr["curve"][0] == gd["curve"][0], "ki": gr["ki"], "rank": [gr["rule_k"], gd["rule_k"]],
           "floor_dabs": {"gamma": fl[0], "recall": fl[1], "ndis": fl[2], "drop_pp": (r0 - fl[1]) * 100},
           "at_floor": {"ndis_rule": nr_fl, "ratio": (fl[2] / nr_fl) if nr_fl else None}}
    for d in DROPS:
        t = r0 - d / 100.0
        a, b = ndis_at(cr, t), ndis_at(cd, t)
        out[f"drop{d:g}"] = {"target": t, "ndis_rule": a, "ndis_dabs": b, "ratio": (b / a) if (a and b) else None,
                             "gamma_rule": gamma_at(cr, t), "gamma_dabs": gamma_at(cd, t), "below_floor": t < fl[1]}
    return out


def summary(results):
    out = {}
    for K in (100, 20):
        for ds in DS:
            for S in SS:
                for E in ES[K]:
                    gr, gd = grid(results, K, ds, S, E, "rule"), grid(results, K, ds, S, E, "dabs")
                    if gr and gd:
                        out[f"K{K}_{ds}_S{S}_E{E}"] = dict(K=K, ds=ds, S=S, E=E, **cell(gr, gd))
    means = {}
    for K in (100, 20):
        for ds in DS:
            for S in SS:
                cs = [v for v in out.values() if (v["K"], v["ds"], v["S"]) == (K, ds, S)]
                r1 = [v["drop1"]["ratio"] for v in cs if v["drop1"]["ratio"]]
                rf = [v["at_floor"]["ratio"] for v in cs if v["at_floor"]["ratio"]]
                if cs:
                    means[f"K{K}_{ds}_S{S}"] = {"ratio_drop1": statistics.mean(r1) if r1 else None, "n_drop1": len(r1),
                                                "ratio_at_floor": statistics.mean(rf) if rf else None}
    res = {"cells": out, "means": means}
    os.makedirs(os.path.join(results, "rank_k100"), exist_ok=True)
    with open(os.path.join(results, "rank_k100", "summary.json"), "w") as f:
        json.dump(res, f, indent=1)
    for k, v in means.items():
        print(f"{k:22s} ratio(-1pp) {v['ratio_drop1'] or float('nan'):.3f} ({v['n_drop1']})  at floor {v['ratio_at_floor'] or float('nan'):.3f}")
    return res


def mech(results, trace_dir, idx_dir):
    import rank_mech  # noqa: E402
    for K in (100, 20):
        for ds in DS:
            for S in SS:
                for E in ES[K]:
                    gr, gd = grid(results, K, ds, S, E, "rule"), grid(results, K, ds, S, E, "dabs")
                    if not (gr and gd):
                        continue
                    c = cell(gr, gd)
                    for arm in ("rule", "dabs"):
                        f = os.path.join(results, "rank_k100", "mech", L.dsn(ds), f"K{K}_{arm}_S{S}_E{E}.json")
                        if os.path.exists(f):
                            continue
                        gs = []
                        for d in DROPS:
                            x = c[f"drop{d:g}"]
                            g = x[f"gamma_{arm}"]
                            if g is None and arm == "dabs" and x["below_floor"]:
                                g = c["floor_dabs"]["gamma"]           # cannot reach the target: replay at the floor
                            gs.append(g)
                        if any(g is None for g in gs):
                            print(f"!! mech K={K} {ds} S={S} E={E} {arm}: target gamma outside the grid", file=sys.stderr)
                            continue
                        tr = os.path.join(trace_dir, "" if K == 20 else f"K{K}", L.dsn(ds), arm, f"eval_S{S}_E{E}.bin")
                        if not os.path.exists(tr):
                            print(f"   mech K={K} {ds} S={S} E={E} {arm}: no record {tr}", file=sys.stderr)
                            continue
                        r = rank_mech.analyze(tr, os.path.join(idx_dir, L.dsn(ds), "gt300.ivec"), K, [f"{g:.6g}" for g in gs])
                        os.makedirs(os.path.dirname(f), exist_ok=True)
                        with open(f, "w") as fh:
                            json.dump(r, fh, indent=1)
                        a = r["all"]
                        print(f"   mech K={K} {arm:4s} {ds} S={S} E={E} pops_to_have median {a['pops_to_have']['median']} "
                              f"stop_at_have {a['at_gamma'][r['gammas'][0]]['stop_at_have_frac']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("summary", "mech"))
    ap.add_argument("--results", default=os.environ.get("RESULTS", os.path.join(L.repro_root(), "results")))
    ap.add_argument("--trace-dir", default=os.environ.get("TRACE_DIR"))
    ap.add_argument("--idx-dir", default=os.environ.get("IDX_DIR"))
    a = ap.parse_args()
    if a.what == "summary":
        summary(a.results)
    else:
        if not (a.trace_dir and a.idx_dir):
            sys.exit("!! mech needs --trace-dir and --idx-dir (or TRACE_DIR, IDX_DIR)")
        mech(a.results, a.trace_dir, a.idx_dir)


if __name__ == "__main__":
    main()
