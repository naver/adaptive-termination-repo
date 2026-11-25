# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)

"""Summary of the calibration-before-serving experiment (experiments/run_corpus.sh): one row per cell (dataset, S, E).

For every cell: nat recall and distance computations per query (blocks a + b), the rule and corpus arms' recall
difference and verdict against nat (lib/bootstrap.py), their distance computations, the retained share of the saving
r = (nat - corpus) / (nat - rule) (empty where the rule saves nothing), and the per-shard gammas of both calibrations
(shards with the same gamma; shards where the rule is switched off, None).

The case of the plan is read off the in-distribution datasets (MS MARCO, GloVe, SIFT; T2I is reported, not counted):
  A  no corpus `exceeds`, median r >= R_MIN       B  no corpus `exceeds`, median r < R_MIN
  C  a corpus `exceeds` in a cell where the rule does not exceed
It is printed only when all 27 in-distribution cells are present.

Output: $RESULTS/corpus_summary.json and corpus_summary.md.
Usage: python3 corpus_report.py [--results DIR] [--r_min 0.8]
"""
import argparse
import glob
import json
import os
import statistics

import shardlib as L
from bootstrap import Dumps, default_results

ID_SETS = ("msmarco", "glove", "sift")


def cells_json(results, ds, sub):
    f = sorted(glob.glob(os.path.join(results, "cells", f"{ds}_N*", *([sub] if sub else []), "calib_cells.json")))
    return json.load(open(f[0])) if f else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=default_results())
    ap.add_argument("--r_min", type=float, default=0.8)
    a = ap.parse_args()
    rows = json.load(open(os.path.join(a.results, "verdicts", "bootstrap.json")))
    V = {(r["ds"], r["S"], r["ef"], r["arm"]): r for r in rows}
    D = Dumps(a.results, a.results, L.K)
    out = []
    for ds, S, ef in sorted({(r["ds"], r["S"], r["ef"]) for r in rows if r["arm"] == "corpus"}):
        ru, co = V.get((ds, S, ef, "rule")), V[(ds, S, ef, "corpus")]
        nat = D.load(ds, S, ef, "nat", ("a", "b"))
        nat_r, nat_n = float(nat[1].mean() / L.K), float(nat[2].mean())
        key = f"S{S}_E{ef}"
        gr = cells_json(a.results, ds, "").get(key, {}).get("gamma_rank_shard")
        gc = cells_json(a.results, ds, "corpus").get(key, {}).get("gamma_rank_shard")
        n_rule = nat_n * ru["ndis_ratio"] if ru else None
        n_corp = nat_n * co["ndis_ratio"]
        r = (nat_n - n_corp) / (nat_n - n_rule) if n_rule is not None and nat_n - n_rule > 0 else None
        out.append(dict(ds=ds, S=S, E=ef, nq=co["nq"], nat_recall=nat_r, nat_ndis=nat_n,
                        rule_delta_pp=ru and ru["delta_pp"], rule_verdict=ru and ru["verdict"], rule_ndis=n_rule,
                        corpus_delta_pp=co["delta_pp"], corpus_verdict=co["verdict"], corpus_ndis=n_corp, retained=r,
                        gamma_rule=gr, gamma_corpus=gc,
                        same_gamma=None if gr is None or gc is None else sum(x == y for x, y in zip(gr, gc)),
                        rule_off=None if gr is None else gr.count(None), corpus_off=None if gc is None else gc.count(None)))

    ids = [c for c in out if c["ds"] in ID_SETS]
    rs = [c["retained"] for c in ids if c["retained"] is not None]
    case = None
    if len(ids) == 27:
        bad = [c for c in ids if c["corpus_verdict"] == "exceeds" and c["rule_verdict"] != "exceeds"]
        if bad:
            case = "C"
        elif not any(c["corpus_verdict"] == "exceeds" for c in ids):
            case = "A" if rs and statistics.median(rs) >= a.r_min else "B"
    summary = dict(cells=out, id_cells=len(ids), id_median_retained=statistics.median(rs) if rs else None,
                   id_corpus_exceeds=sum(c["corpus_verdict"] == "exceeds" for c in ids), case=case, r_min=a.r_min)
    json.dump(summary, open(os.path.join(a.results, "corpus_summary.json"), "w"), indent=1)

    f = lambda x, p: "-" if x is None else f"{x:{p}}"  # noqa: E731
    lines = ["| dataset | S | E | nat recall | rule Dpp, verdict | corpus Dpp, verdict | nat ndis | rule | corpus "
             "| retained r | same gamma |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in out:
        lines.append(f"| {c['ds']} | {c['S']} | {c['E']} | {c['nat_recall']:.4f} | {f(c['rule_delta_pp'], '+.2f')} "
                     f"{c['rule_verdict'] or ''} | {c['corpus_delta_pp']:+.2f} {c['corpus_verdict']} | {c['nat_ndis']:.0f} "
                     f"| {f(c['rule_ndis'], '.0f')} | {c['corpus_ndis']:.0f} | {f(c['retained'], '.2f')} "
                     f"| {'-' if c['same_gamma'] is None else str(c['same_gamma']) + '/' + str(c['S'])} |")
    lines += ["", f"in-distribution cells: {len(ids)}/27, corpus exceeds: {summary['id_corpus_exceeds']}, "
                  f"median r: {f(summary['id_median_retained'], '.3f')}, case: {case or '- (incomplete)'}"]
    open(os.path.join(a.results, "corpus_summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
