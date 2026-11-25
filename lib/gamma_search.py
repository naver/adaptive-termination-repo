# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)

"""Smallest shard-common gamma whose replayed recall reaches a target R, for one traversal record (cell S, E).

A larger gamma stops later, so recall and ndis grow (almost) monotonically in gamma. gamma = 0 is the rule switched
off (fixed ef end state); if its recall is below R, this E cannot reach R (infeasible). The bench reads gamma as a
string parsed to float32, so the search runs on a string grid (%.5f): integer steps i * step, bisected.

Output (stdout, JSON): {S, nq, ki, rule_k, ef, R, fixed_end, feasible, [gamma, recall, ndis], probes}

Usage: python3 gamma_search.py --trace T.bin --gt gt300.ivec --R 0.95 [--K 20]  > out.json
"""
import argparse
import json
import sys

from trace_replay import read_gt, read_trace, replay


def search(meta, recs, gt, R, K=20, gmax=0.5, step=1e-5):
    cache = {}

    def at(g):
        s = "0" if g == 0 else f"{g:.5f}"
        if s not in cache:
            cache[s] = replay(meta, recs, gt, K, s)
        return s, cache[s]

    out = dict(meta, R=R)
    s0, (r0, n0) = at(0)
    out["fixed_end"] = {"recall": r0, "ndis": n0}
    if r0 < R:
        out["feasible"] = False
    else:
        lo, hi = 0, int(round(gmax / step))            # search on integer steps (gamma = i * step)
        if at(hi * step)[1][0] < R:
            hi = 0                                     # no gamma reaches R: the answer is the rule switched off
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if at(mid * step)[1][0] >= R:
                hi = mid
            else:
                lo = mid
        g = hi * step
        s, (r, nd) = at(g)
        out.update(feasible=True, gamma=s, recall=r, ndis=nd)
    out["probes"] = [{"gamma": s, "recall": r, "ndis": nd} for s, (r, nd) in sorted(cache.items(), key=lambda x: float(x[0]))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True); ap.add_argument("--gt", required=True)
    ap.add_argument("--R", type=float, required=True); ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--gmax", type=float, default=0.5); ap.add_argument("--step", type=float, default=1e-5)
    a = ap.parse_args()
    meta, recs = read_trace(a.trace)
    gt = read_gt(a.gt, meta["nq"])
    json.dump(search(meta, recs, gt, a.R, a.K, a.gmax, a.step), sys.stdout, indent=1)


if __name__ == "__main__":
    main()
