# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
"""Bonferroni-corrected tolerance verdicts: one joint statement over all configurations of one arm.

lib/bootstrap.py gives each configuration its own 95% interval. Read together, m such intervals hold jointly with less
than 95%. A joint level of 1 - alpha over the m configurations of an arm (arm rule: m = 60) needs, per
configuration, a two-sided 1 - alpha/m interval. The verdict rule is that of lib/bootstrap.py, applied to this wider
interval: `undecided` if it contains -eps, else `exceeds` if Delta < -eps, else `within`.

Each configuration takes its interval from one of two sources:
  bootstrap  paired bootstrap at the corrected level on the per-query dumps, with the pairing and the per-configuration
             seed of lib/bootstrap.py and B resamples (default 200,000, drawn in chunks), when blocks a and b of both
             nat and the arm resolve under DUMPS_DIR (either layout of shardlib.dump_path);
  normal     otherwise, the 95% interval of verdicts/bootstrap.json widened about its point estimate, each side by
             z(1 - alpha/2m) / z(0.975) (normal approximation). The 100M configurations use this source, since
             their dumps are not available; --normal-only uses it for every configuration.
From the shipped results (no dumps), --normal-only reproduces the reference counts; figures/text_numbers.py
checks them through normal_rows().

Output: <results>/verdicts/bonferroni.json with the level, m, the counts and one entry per configuration.
Usage: python3 bonferroni.py [--results DIR] [--dumps DIR] [--arm rule] [--alpha 0.05] [--B 200000] [--seed 20290917]
                             [--normal-only] [--out FILE]
"""
import argparse
import collections
import json
import os
import sys
from statistics import NormalDist

import numpy as np

import shardlib as L
import bootstrap as BS


def verdict(point, lo, hi, eps=L.EPS_PP):
    """the verdict rule of lib/bootstrap.py"""
    return "undecided" if lo < -eps < hi else ("exceeds" if point < -eps else "within")


def widen_factor(alpha, m):
    return NormalDist().inv_cdf(1 - alpha / (2 * m)) / NormalDist().inv_cdf(0.975)


def normal_cell(row, f, eps=L.EPS_PP):
    """one verdicts/bootstrap.json row -> its interval widened by f about the point estimate"""
    p, (lo, hi) = row["delta_pp"], row["delta_ci"]
    lo, hi = p - (p - lo) * f, p + (hi - p) * f
    return dict(ds=row["ds"], S=row["S"], ef=row["ef"], nq=row["nq"], delta_pp=p,
                ci=[round(lo, 3), round(hi, 3)], verdict=verdict(p, lo, hi, eps), source="normal")


def normal_rows(rows, alpha=0.05, m=None, eps=L.EPS_PP):
    """Bonferroni verdicts of the given rows (one arm) by the normal approximation alone; m defaults to len(rows)."""
    f = widen_factor(alpha, m or len(rows))
    return [normal_cell(r, f, eps) for r in rows]


def counts(cells):
    c = collections.Counter(x["verdict"] for x in cells)
    return {v: c.get(v, 0) for v in ("within", "undecided", "exceeds")}


def boot_cell(ds, S, ef, nat, adp, rng, B, K, q_lo, eps, chunk=2000):
    """paired bootstrap of Delta (pp) at the corrected level; resamples are drawn chunk rows at a time"""
    _, h0, _ = nat
    _, h1, _ = adp
    nq = len(h0)
    d = np.empty(B)
    for at in range(0, B, chunk):
        idx = rng.integers(0, nq, size=(min(chunk, B - at), nq))
        d[at:at + len(idx)] = (h1[idx].mean(1) - h0[idx].mean(1)) / K * 100.0
    point = (h1.mean() - h0.mean()) / K * 100.0
    lo, hi = float(np.quantile(d, q_lo)), float(np.quantile(d, 1 - q_lo))
    return dict(ds=ds, S=S, ef=ef, nq=nq, delta_pp=round(point, 3), ci=[round(lo, 3), round(hi, 3)],
                verdict=verdict(point, lo, hi, eps), source="bootstrap")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=BS.default_results(),
                    help="holds verdicts/bootstrap.json (and cells/); default $RESULTS")
    ap.add_argument("--dumps", default=None, help="root holding dumps/ or per_query/ (default: $DUMPS_DIR, else --results)")
    ap.add_argument("--arm", default="rule")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--B", type=int, default=200000)
    ap.add_argument("--K", type=int, default=L.K)
    ap.add_argument("--eps", type=float, default=L.EPS_PP)
    ap.add_argument("--seed", type=int, default=20290917)
    ap.add_argument("--normal-only", action="store_true", help="widen the 95%% intervals for every configuration")
    ap.add_argument("--out", default=None, help="default: <results>/verdicts/bonferroni.json")
    a = ap.parse_args()

    vf = os.path.join(a.results, "verdicts", "bootstrap.json")
    if not os.path.isfile(vf):
        sys.exit(f"!! {vf} not found (run lib/bootstrap.py first)")
    rows = sorted((r for r in json.load(open(vf)) if r["arm"] == a.arm), key=lambda r: (r["ds"], r["S"], r["ef"]))
    if not rows:
        sys.exit(f"!! no arm {a.arm!r} in {vf}")
    m = len(rows)
    f = widen_factor(a.alpha, m)
    q_lo = a.alpha / (2 * m)
    D = BS.Dumps(a.results, a.dumps or L.dumps_dir(a.results), a.K)
    cells = []
    for r in rows:
        ds, S, ef = r["ds"], r["S"], r["ef"]
        nat = adp = None
        if not a.normal_only:
            bl = [b for b in (D.blocks(ds, S, ef, a.arm) or []) if b in (D.blocks(ds, S, ef, "nat") or [])]
            if {"a", "b"} <= set(bl):
                nat, adp = D.load(ds, S, ef, "nat", bl), D.load(ds, S, ef, a.arm, bl)
        if nat is not None and adp is not None:
            if not (nat[0] == adp[0]).all():
                sys.exit(f"!! query numbers of nat and {a.arm} differ: {ds} S={S} ef={ef}")
            cells.append(boot_cell(ds, S, ef, nat, adp, BS.cell_rng(a.seed, ds, S, ef), a.B, a.K, q_lo, a.eps))
        else:
            cells.append(normal_cell(r, f, a.eps))
    c = counts(cells)
    src = collections.Counter(x["source"] for x in cells)
    res = dict(arm=a.arm, m=m, alpha=a.alpha, level=round(1 - a.alpha / m, 6), B=a.B, widen_factor=round(f, 4),
               sources=dict(src), **c, cells=cells)
    out = a.out or os.path.join(a.results, "verdicts", "bonferroni.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(f"arm {a.arm}: {m} configurations, joint level {1 - a.alpha:.0%} -> {1 - a.alpha / m:.4%} per configuration "
          f"({', '.join(f'{k} {v}' for k, v in sorted(src.items()))})")
    for x in cells:
        if x["verdict"] != "within":
            print(f"   {x['ds']:<9} S={x['S']:<3} ef={x['ef']:<4} {x['delta_pp']:+.3f}  "
                  f"[{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]  {x['verdict']}  ({x['source']})")
    print(f"within {c['within']}, undecided {c['undecided']}, exceeds {c['exceeds']}\n-> {out}")


if __name__ == "__main__":
    main()
