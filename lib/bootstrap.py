# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)

"""Tolerance verdicts with paired-bootstrap confidence intervals over the enlarged test queries.

Input: per-query dumps under DUMPS_DIR (default $RESULTS), found through shardlib.dump_path in either layout:
  dumps/<ds>_S{S}_ef{E}_{arm}_{blk}.tsv (+ .json, the bench summary of the same run), written by experiments/dump_cell.sh,
  or the published per_query/<ds>_S{S}_ef{E}_{nat|ours|ki|dabs|s1g}_{blk}.{tsv,json} (original arm names: ours = the
  rule at S <= 5, ki = the rule at S >= 15; "ours" at S >= 15 used another rank and is not read).
Only block-a .json files are read (reproduction check). Output: $RESULTS/verdicts/bootstrap.json, one row per
(ds, arm, S, ef).

  - A configuration is (dataset, arm, S, ef). Every adaptive arm is paired with `nat` (gamma = 0, fixed ef at ef = E,
    the same queries). Arms: rule (adaptive termination, rank k_i), dabs (rank K),
    corpus (rank k_i, calibrated before serving on held-out shard rows; experiments/run_corpus.sh), p125 and p250
    (rank k_i, calibrated on the first 125 or 250 calibration queries; experiments/ncalib_cell.sh), s1g (rank k_i, the
    gamma calibrated on a single index (S = 1) applied unchanged to every shard; experiments/run_extras.sh transfer).
  - Blocks are concatenated: a = #0-999, b = #1500-...
    q in a dump is the index inside its block, so block offsets are added to get absolute query numbers.
  - Paired bootstrap: B = 10,000 resamples of queries; a resample takes both rows (nat, arm) of a query. Every
    configuration has its own seed, seed + crc32("ds|S|ef"), shared by all arms of that (ds, S, ef): arms see the same
    resamples, and a verdict does not depend on the order in which configurations are processed.
  - Verdict with tolerance eps = 1pp on the recall difference Delta (pp): `undecided` if the 95% CI contains -eps,
    else `exceeds` if Delta < -eps, else `within`.
  - Queries needed to decide: n = ceil((1.96 * sd(per-query Delta) / |Delta + eps|)^2).
  - The same verdict is also computed on block a alone (`at1000`).
  - Reproduction check: the block-a bench summary of a dump must equal the result of the same configuration in
    cells/ (nat: the gamma = 0 point of grid/; rule: calib/). `repro_*` = {same, dump, cell}; None when the
    arm has no counterpart in cells/ (dabs, corpus, p125, p250) or the cell file is missing.

At S = 1 the rule and rank-K termination coincide (rank k_i = K = 20, same calibration). A missing rule dump at S = 1 falls
back to the rank-K termination dump of the same configuration (Dumps.path).

Usage: python3 bootstrap.py [--results DIR] [--dumps DIR] [--out DIR/verdicts/bootstrap.json] [--B 10000] [--seed 20290917]
"""
import argparse
import glob
import json
import os
import re
import sys
import zlib

import numpy as np

import shardlib as L

QOFF = {"a": 0, "b": L.BLOCK_B_START}
# configurations are discovered from block-a dumps in either layout (see shardlib.dump_path):
#   package run   dumps/<ds>_S{S}_ef{E}_{nat|rule|dabs|corpus|p125|p250|s1g}_{blk}.tsv
#                 (approx dumps, rank k_i with approximate-ground-truth gammas, are judged by experiments/approx_verdicts.py)
#   published     per_query/<ds>_S{S}_ef{E}_{nat|ours|ki|dabs|s1g}_{blk}.tsv   (original arm names)
RUN_RE = re.compile(r"([a-z0-9]+)_S(\d+)_ef(\d+)_(rule|dabs|corpus|p125|p250|s1g)_a\.tsv$")
PUB_RE = re.compile(r"([a-z0-9]+)_S(\d+)_ef(\d+)_(ours|ki|dabs|s1g)_a\.tsv$")


def default_results():
    return os.environ.get("RESULTS") or os.path.join(L.repro_root(), "work", "results")


def pub_arm(arm, S, K):
    """original arm name of the published dumps -> arm name of this package (None: not an arm of this package)"""
    if arm == "ours":                  # adaptive termination where k_i >= 8; where k_i < 8 the original arm used another rank
        return "rule" if L.k_i_of(K, S) >= 8 else None
    return "rule" if arm == "ki" else arm


class Dumps:
    def __init__(self, results, dumps_root, K):
        self.results = results                    # cells/ (reproduction check) live here
        self.root = dumps_root                    # dumps/ or per_query/ live here
        self.K = K

    def path(self, ds, S, ef, arm, blk, ext="tsv"):
        """The dump file, or None if absent or a Git LFS pointer stub. At S = 1 the rule and rank-K termination are the same
        configuration (rank k_i = K, same calibration): without a rule dump at S = 1, the rank-K termination dump is used."""
        return L.dump_path(self.root, "per_query", ds, S, ef, arm, blk, ext=ext, K=self.K)

    def configurations(self):
        cells, stubs = set(), []
        for sub, rx, conv in (("dumps", RUN_RE, lambda a, S: a), ("per_query", PUB_RE, lambda a, S: pub_arm(a, S, self.K))):
            for f in glob.glob(os.path.join(self.root, sub, "*_a.tsv")):
                m = rx.match(os.path.basename(f))
                if not m:
                    continue
                if L.is_lfs_pointer(f):
                    stubs.append(f); continue
                ds, S, ef = m.group(1), int(m.group(2)), int(m.group(3))
                arm = conv(m.group(4), S)
                if arm is None:
                    continue
                cells.add((ds, S, ef, arm))
                if m.group(4) == "dabs" and S == 1:
                    cells.add((ds, 1, ef, "rule"))       # see path()
        if stubs:
            print(f"!! {len(stubs)} dumps are Git LFS pointers, not data (fetch them or rerun experiments/dump_cell.sh), "
                  f"e.g. {stubs[0]}", file=sys.stderr)
        return cells

    def blocks(self, ds, S, ef, arm):
        """Blocks present for this arm. a and b are required."""
        have = [b for b in "ab" if self.path(ds, S, ef, arm, b)]
        return have if {"a", "b"} <= set(have) else None

    def load(self, ds, S, ef, arm, blks=("a", "b")):
        """Concatenated (absolute q, hits, ndis) over blocks blks; None if one is missing."""
        parts = []
        for blk in "ab":
            if blk not in blks:
                continue
            f = self.path(ds, S, ef, arm, blk)
            if f is None:
                return None
            a = np.genfromtxt(f, delimiter="\t", names=True)
            parts.append((np.atleast_1d(a["q"]).astype(int) + QOFF[blk], np.atleast_1d(a["hits"]).astype(float),
                          np.atleast_1d(a["ndis"]).astype(float)))
        return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))

    def repro(self, ds, S, ef, arm):
        """Does the block-a bench summary equal the same configuration in cells/? -> {same, dump, cell} or None"""
        fj = self.path(ds, S, ef, arm, "a", "json")
        if fj is None:
            print(f"   (no block-a bench summary for {ds} S={S} ef={ef} {arm}; reproduction not checked)", file=sys.stderr)
            return None
        b = json.load(open(fj))[0]
        mine = (round(b["recall"], 5), round(b["ndis_per_query"], 1))
        try:
            cells = os.path.join(self.results, "cells", L.dsn(ds))
        except KeyError:                       # dataset other than the six, without N_<DS>: find its cells directory
            cand = sorted(glob.glob(os.path.join(self.results, "cells", f"{ds}_N*")))
            cells = cand[0] if cand else os.path.join(self.results, "cells", ds)
        if arm == "nat":           # gamma = 0: the fixed-ef point of the grid cell
            f = os.path.join(cells, "grid", f"S{S}_E{ef}.json")
            c = json.load(open(f))["bench"] if os.path.exists(f) else None
        elif arm == "rule":
            f = os.path.join(cells, "calib", f"S{S}_E{ef}.json")
            c = json.load(open(f))[0] if os.path.exists(f) else None
        else:
            return None
        if c is None:
            print(f"   (no cells/ counterpart for {ds} S={S} ef={ef} {arm}; reproduction not checked)", file=sys.stderr)
            return None
        ref = (round(c["recall"], 5), round(c["ndis_per_query"], 1))
        return dict(same=mine == ref, dump=mine, cell=ref)


def cell_rng(seed, *key):
    """One seed per configuration: seed + crc32("|".join(key))."""
    return np.random.default_rng(seed + zlib.crc32("|".join(map(str, key)).encode()))


def ci(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def judge(h0, h1, n0, n1, rng, B, K, eps):
    nq = len(h0)
    idx = rng.integers(0, nq, size=(B, nq))
    d_pp = (h1[idx].mean(1) - h0[idx].mean(1)) / K * 100.0
    ratio = n1[idx].mean(1) / n0[idx].mean(1)
    point = (h1.mean() - h0.mean()) / K * 100.0
    lo, hi = ci(d_pp)
    sd = np.std((h1 - h0) / K * 100.0, ddof=1)
    gap = abs(point + eps)
    verdict = "undecided" if lo < -eps < hi else ("exceeds" if point < -eps else "within")
    return dict(nq=nq, delta_pp=round(point, 3), delta_ci=[round(lo, 3), round(hi, 3)],
                p_violate=round(float((d_pp < -eps).mean()), 4), verdict=verdict,
                n_needed=int(np.ceil((1.96 * sd / gap) ** 2)) if gap > 0 else None,
                ndis_ratio=round(float(n1.mean() / n0.mean()), 4), ndis_ci=[round(x, 4) for x in ci(ratio)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=default_results())
    ap.add_argument("--dumps", default=None,
                    help="root holding dumps/ or per_query/ (default: $DUMPS_DIR, else --results)")
    ap.add_argument("--out", default=None, help="default: <results>/verdicts/bootstrap.json")
    ap.add_argument("--K", type=int, default=L.K); ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--eps", type=float, default=L.EPS_PP); ap.add_argument("--seed", type=int, default=20290917)
    a = ap.parse_args()
    D = Dumps(a.results, a.dumps or L.dumps_dir(a.results), a.K)
    out = a.out or os.path.join(a.results, "verdicts", "bootstrap.json")
    cells = D.configurations()
    if not cells:
        sys.exit(f"!! no dumps under {D.root}/dumps or {D.root}/per_query (run experiments/dump_cell.sh first)")
    rows = []
    for ds, S, ef, arm in sorted(cells, key=lambda c: (c[0], c[3], c[1], c[2])):
        nat_bl = D.blocks(ds, S, ef, "nat") or []
        bl = [b for b in (D.blocks(ds, S, ef, arm) or []) if b in nat_bl]      # pair on blocks both arms have
        nat, adp = (D.load(ds, S, ef, "nat", bl), D.load(ds, S, ef, arm, bl)) if {"a", "b"} <= set(bl) else (None, None)
        if nat is None or adp is None:
            print(f"   skip {ds} S={S} ef={ef} {arm}: blocks a and b are needed for both nat and {arm}", file=sys.stderr)
            continue
        if not (nat[0] == adp[0]).all():
            sys.exit(f"!! query numbers of nat and {arm} differ: {ds} S={S} ef={ef}")
        q, h0, n0 = nat; _, h1, n1 = adp
        full = judge(h0, h1, n0, n1, cell_rng(a.seed, ds, S, ef), a.B, a.K, a.eps)
        first = q < 1000
        k1000 = judge(h0[first], h1[first], n0[first], n1[first], cell_rng(a.seed, ds, S, ef, 1000), a.B, a.K, a.eps)
        rows.append(dict(ds=ds, arm=arm, S=S, ef=ef, **full, at1000=k1000,
                         repro_nat=D.repro(ds, S, ef, "nat"), repro_arm=D.repro(ds, S, ef, arm)))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(rows, open(out, "w"), indent=1)
    print(f"{'dataset':<10}{'arm':<8}{'S':>3}{'ef':>5}{'n':>7}{'Dpp':>7}{'95% CI':>17}{'P(exc)':>8}  {'verdict':<10}"
          f"{'need n':>9}  {'n=1000':<10}{'ndis':>8}  repro")
    for r in rows:
        band = "[%+.2f, %+.2f]" % tuple(r["delta_ci"])
        rp = all(x is None or x["same"] for x in (r["repro_nat"], r["repro_arm"]))
        print(f"{r['ds']:<10}{r['arm']:<8}{r['S']:>3}{r['ef']:>5}{r['nq']:>7}{r['delta_pp']:>7.2f}{band:>17}"
              f"{r['p_violate']:>8.3f}  {r['verdict']:<10}{r['n_needed'] or 0:>9,}  {r['at1000']['verdict']:<10}"
              f"{r['ndis_ratio']:>8.4f}  {'ok' if rp else '!! differs'}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
