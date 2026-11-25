# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)

"""Optional check that the MS MARCO vectors are the published ones and that queries and passages are paired correctly.

Scores the exact top-300 neighbors (gt300.ivec over all 6,980 dev queries, written by ground_truth.sh) against the official
dev relevance judgments (qrels.dev.small.tsv, 7,437 judgments). Exhaustive search over these vectors is the same as the
flat-index regression published by Anserini 1.0.0 for this corpus, so the scores must match it:

                                         RR@10    R@100
    Anserini 1.0.0 regression (flat)     0.3658   0.8935
    our vectors + our ground truth       0.3659   0.8941   (the original run; differences in the 4th digit come from ties)

RR@10 = mean over queries of 1/rank of the first relevant passage within the top 10 (0 if none).
R@100 = mean over queries of |relevant in top 100| / |relevant|. Row i of base.fbin is passage id i; row j of query.fbin is
the query id on line j of query_ids.txt.

Usage: python3 data/verify_msmarco_qrels.py [--gt $IDX_DIR/msmarco_N8841823/gt300.ivec] [--query-ids $RAW_DIR/msmarco/query_ids.txt]
                                            [--qrels $WORK/download/msmarco/qrels.dev.small.tsv] [--tol 0.002]
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import shardlib as L  # noqa: E402

EXPECTED = {"rr10": 0.3659, "r100": 0.8941}      # original run of this pipeline
ANSERINI = {"rr10": 0.3658, "r100": 0.8935}      # Anserini 1.0.0 regression, same vectors, flat index


def score(gt, qids, qrels):
    rr, rec, n = 0.0, 0.0, 0
    for row, qid in enumerate(qids):
        rel = qrels.get(qid)
        if not rel:
            continue
        n += 1
        top = gt[row]
        for r, pid in enumerate(top[:10]):
            if int(pid) in rel:
                rr += 1.0 / (r + 1)
                break
        rec += len(rel.intersection(int(p) for p in top[:100])) / len(rel)
    return {"queries_scored": n, "rr10": rr / n, "r100": rec / n}


def main():
    work = os.environ.get("WORK") or os.path.join(L.repro_root(), "work")
    raw = os.environ.get("RAW_DIR") or os.path.join(work, "raw")
    idx = os.environ.get("IDX_DIR") or os.path.join(work, "idx")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", default=os.path.join(idx, L.dsn("msmarco"), "gt300.ivec"))
    ap.add_argument("--query-ids", default=os.path.join(raw, "msmarco", "query_ids.txt"))
    ap.add_argument("--qrels", default=os.path.join(os.environ.get("DL_DIR") or os.path.join(work, "download"),
                                                     "msmarco", "qrels.dev.small.tsv"))
    ap.add_argument("--tol", type=float, default=0.002, help="allowed absolute difference from the original run")
    a = ap.parse_args()

    gt = L.read_ivec(a.gt)
    qids = [int(x) for x in open(a.query_ids).read().split()]
    if gt.shape[0] != len(qids):
        raise SystemExit(f"{a.gt} has {gt.shape[0]} rows, {a.query_ids} {len(qids)} ids: need ground truth for every query")
    if gt.shape[1] < 100:
        raise SystemExit("need at least the top-100 neighbors")
    qrels = defaultdict(set)
    for line in open(a.qrels):
        qid, _, pid, lab = line.split()
        if int(lab) > 0:
            qrels[int(qid)].add(int(pid))
    res = score(gt, qids, qrels)
    res.update(judgments=sum(len(v) for v in qrels.values()), expected=EXPECTED, anserini=ANSERINI)
    res["ok"] = all(abs(res[k] - EXPECTED[k]) <= a.tol for k in EXPECTED)
    print(json.dumps(res, indent=1))
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
