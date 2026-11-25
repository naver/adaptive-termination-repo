# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)

"""Invariant checks on the synthetic run of calibration before serving (smoke/smoke.sh --full, experiments/run_corpus.sh).

  1. held-out rows: 500 per shard, the same rows as holdout.hold_rows (deterministic), and the final graph's id list is
     the split's id list reordered (rest, then held out): no row lost or duplicated
  2. corpus calibration: an entry per (S, E) with one gamma per shard and 500 held-out queries per shard
  3. dumps: per query, corpus ndis <= fixed-ef ndis, hits in [0, K], recall = mean(hits) / K; where the corpus and
     rule gammas are equal the two dumps are identical (same graph, same configuration)
  4. verdicts: a rule and a corpus row per cell
Exit 1 on any violation.

Usage: python3 check_corpus.py HOLDOUT_IDX HOLDOUT_RESULTS DSN "S..." "E..."
"""
import csv
import hashlib
import json
import os
import sys

import numpy as np

import holdout as H
import shardlib as L

idx, res, dsn = sys.argv[1], sys.argv[2], sys.argv[3]
SS, ES = [int(x) for x in sys.argv[4].split()], [int(x) for x in sys.argv[5].split()]
ds, K = dsn.split("_N")[0], L.K
bad = []

for S in SS:
    for s in range(S):
        D = os.path.join(idx, dsn, f"S{S}")
        info = json.load(open(os.path.join(D, f"part{s}", "hold.json")))
        rows = H.hold_rows(info["n_s"], S, s)
        if info["n_hold"] != H.N_HOLD or info["rows_md5"] != hashlib.md5(rows.astype(np.int64).tobytes()).hexdigest():
            bad.append(f"S={S} shard {s}: held-out rows differ from holdout.hold_rows")
        orig, fin = H.read_sel(os.path.join(D, f"shard{s}.orig.sel")), H.read_sel(os.path.join(D, f"shard{s}.sel"))
        keep = np.ones(len(orig), bool); keep[rows] = False
        if not np.array_equal(fin, np.concatenate([orig[keep], orig[rows]])):
            bad.append(f"S={S} shard {s}: final id list is not (rest, held out) of the split")

cc = json.load(open(os.path.join(res, "cells", dsn, "corpus", "calib_cells.json")))
rc = json.load(open(os.path.join(res, "cells", dsn, "calib_cells.json")))
for S in SS:
    for E in ES:
        e = cc.get(f"S{S}_E{E}")
        if e is None or len(e["gamma_rank_shard"]) != S or e["nq"] != [H.N_HOLD] * S:
            bad.append(f"S={S} E={E}: corpus calibration entry missing or malformed")


def dump(S, E, arm, blk):
    p = os.path.join(res, "dumps", f"{ds}_S{S}_ef{E}_{arm}_{blk}")
    with open(p + ".tsv") as f:
        rows = {int(r["q"]): (int(r["hits"]), float(r["ndis"])) for r in csv.DictReader(f, delimiter="\t")}
    return rows, json.load(open(p + ".json"))[0]


n = 0
for S in SS:
    for E in ES:
        same = cc[f"S{S}_E{E}"]["gamma_rank_shard"] == rc[f"S{S}_E{E}"]["gamma_rank_shard"]
        for blk in ("a", "b"):
            nat, _ = dump(S, E, "nat", blk)
            co, jc = dump(S, E, "corpus", blk)
            ru, _ = dump(S, E, "rule", blk)
            for q, (h, nd) in co.items():
                if nd > nat[q][1] + 1e-9:
                    bad.append(f"S={S} E={E} {blk} q={q}: corpus ndis {nd} > fixed {nat[q][1]}")
                if not 0 <= h <= K:
                    bad.append(f"S={S} E={E} {blk} q={q}: hits {h}")
            if abs(sum(h for h, _ in co.values()) / (K * len(co)) - jc["recall"]) > 1e-6:
                bad.append(f"S={S} E={E} {blk}: corpus recall != mean(hits)/K")
            if same and co != ru:
                bad.append(f"S={S} E={E} {blk}: equal gammas but corpus and rule dumps differ")
            n += len(co)
        print(f"  S={S} E={E}: gammas rule {rc[f'S{S}_E{E}']['gamma_rank_shard']} corpus "
              f"{cc[f'S{S}_E{E}']['gamma_rank_shard']}{' (equal, dumps identical)' if same else ''}")

V = {(r["arm"], r["S"], r["ef"]) for r in json.load(open(os.path.join(res, "verdicts", "bootstrap.json"))) if r["ds"] == ds}
for S in SS:
    for E in ES:
        for arm in ("rule", "corpus"):
            if (arm, S, E) not in V:
                bad.append(f"S={S} E={E}: no {arm} verdict")

print(f"checked {n} per-query corpus rows; {len(bad)} violations")
for b in bad[:20]:
    print("  !!", b)
sys.exit(1 if bad else 0)
