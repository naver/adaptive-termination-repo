# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)

"""Replay one gamma = 0 traversal record of the test queries over a gamma grid -> (recall, ndis) curve.

The replay is trace_replay.replay (exact against direct runs). The gamma = 0 point must equal the bench result of the
recording run itself; pass that bench JSON with --bench and the script fails if they differ.

Output (stdout, JSON): {S, nq, ki, rule_k, ef, curve: [{gamma, recall, ndis}, ...], bench: <bench JSON row 0>}

Usage: python3 grid_replay.py --trace T.bin --gt gt300.ivec [--bench B.json] [--K 20] [--gammas "..."]  > curve.json
"""
import argparse
import json
import sys

from trace_replay import GRID_GAMMAS, read_gt, read_trace, replay


def curve(meta, recs, gt, K, gammas):
    out = []
    for gs in ["0"] + gammas.split():
        r, nd = replay(meta, recs, gt, K, gs)
        out.append({"gamma": float(gs), "recall": r, "ndis": nd})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True); ap.add_argument("--gt", required=True)
    ap.add_argument("--bench"); ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--gammas", default=GRID_GAMMAS)
    a = ap.parse_args()
    meta, recs = read_trace(a.trace)
    gt = read_gt(a.gt, meta["nq"])
    c = curve(meta, recs, gt, a.K, a.gammas)
    out = dict(meta, curve=c)
    if a.bench:
        b = json.load(open(a.bench))[0]
        r0, n0 = c[0]["recall"], c[0]["ndis"]
        if f"{r0:.5f}" != f"{b['recall']:.5f}" or f"{n0:.1f}" != f"{b['ndis_per_query']:.1f}":
            sys.exit(f"!! gamma=0 replay {r0:.5f}/{n0:.1f} != bench {b['recall']:.5f}/{b['ndis_per_query']:.1f}")
        out["bench"] = b
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
