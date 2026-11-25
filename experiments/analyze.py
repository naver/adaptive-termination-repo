# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)

"""Collect the verdicts into one summary: $RESULTS/verdicts/summary.json.

  tolerance  rows of $RESULTS/verdicts/bootstrap.json (lib/bootstrap.py: paired bootstrap on the extended test queries) for the
             arms rule (adaptive termination, rank k_i) and dabs (rank-K termination).
  kaxis      K axis: $RESULTS/kaxis/<dsn>/K{K}_{nat|dep|ref}_{a|b}.tsv, judged per K with the same paired bootstrap
             (B=10,000, per-cell seed, tolerance 1pp of recall@K). dep = K=20 gammas used unchanged, ref = recalibrated at that K.
  approx     approximate ground truth: $RESULTS/approx/<dsn>/calib_cells_approx.json against the exact-ground-truth
             gammas of adaptive termination ($CELLS_ROOT/<dsn>/calib_cells.json, default $RESULTS/cells), one row per
             configuration (S, E): same = shard calibrations with the same gamma_i (of shards = S), larger = shards
             where the approximate ground truth selects the larger gamma_i. The configurations with a differing gamma_i
             are judged again by experiments/approx_verdicts.py.

Verdicts: "within" (the 95% interval of the recall change lies above -epsilon), "exceeds" (the interval lies below -epsilon),
"undecided" (the interval contains -epsilon).

Usage: python3 experiments/analyze.py [--results DIR] [--out FILE]      (defaults: $RESULTS and $RESULTS/verdicts/summary.json)
"""
import argparse, collections, json, os, sys, zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "lib"))
import shardlib as L  # noqa: E402

SEED, B_RESAMPLES, EPS = 20290917, 10000, 1.0
QOFF_B = L.BLOCK_B_START
# verdict strings of bootstrap.json -> the strings of this summary ("OK" and "stays" are older spellings of "within")
VERDICT = {"OK": "within", "stays": "within", "within": "within", "exceeds": "exceeds", "undecided": "undecided"}
ARMS = ("rule", "dabs")


def default_results():
    return os.environ.get("RESULTS") or os.path.join(os.environ.get("WORK") or os.path.join(L.repro_root(), "work"), "results")


# ------------------------------------------------------------------ paired bootstrap (same computation as lib/bootstrap.py)
try:
    import bootstrap as _B                              # lib/bootstrap.py
    cell_rng, judge = _B.cell_rng, _B.judge
except (ImportError, AttributeError):
    def cell_rng(seed, *key):
        """One seed per cell (seed + crc32 of the key): arms of a cell share resamples, results do not depend on order."""
        return np.random.default_rng(seed + zlib.crc32("|".join(map(str, key)).encode()))

    def judge(h0, h1, n0, n1, rng, B, K, eps):
        nq = len(h0)
        idx = rng.integers(0, nq, size=(B, nq))
        d_pp = (h1[idx].mean(1) - h0[idx].mean(1)) / K * 100.0
        ratio = n1[idx].mean(1) / n0[idx].mean(1)
        point = (h1.mean() - h0.mean()) / K * 100.0
        lo, hi = float(np.percentile(d_pp, 2.5)), float(np.percentile(d_pp, 97.5))
        sd = np.std((h1 - h0) / K * 100.0, ddof=1)
        gap = abs(point + eps)
        verdict = "undecided" if lo < -eps < hi else ("exceeds" if point < -eps else "within")
        return dict(nq=nq, delta_pp=round(point, 3), delta_ci=[round(lo, 3), round(hi, 3)],
                    p_violate=round(float((d_pp < -eps).mean()), 4), verdict=verdict,
                    n_needed=int(np.ceil((1.96 * sd / gap) ** 2)) if gap > 0 else None,
                    ndis_ratio=round(float(n1.mean() / n0.mean()), 4),
                    ndis_ci=[round(float(np.percentile(ratio, q)), 4) for q in (2.5, 97.5)])


def is_lfs_stub(p):
    """A Git LFS pointer file (a checkout without the LFS objects) stands in for the data; treat it as missing."""
    try:
        with open(p, "rb") as f:
            return f.read(40).startswith(b"version https://git-lfs.github.com/spec/")
    except OSError:
        return False


def have(p):
    if os.path.exists(p) and is_lfs_stub(p):
        print(f"  !! {p} is a Git LFS pointer (data not fetched); treated as missing")
        return False
    return os.path.exists(p)


def tsv(f, off=0):
    a = np.genfromtxt(f, delimiter="\t", names=True)
    return (np.atleast_1d(a["q"]).astype(int) + off, np.atleast_1d(a["hits"]).astype(float),
            np.atleast_1d(a["ndis"]).astype(float))


def cat(*parts):
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))


def is_arm(arm):
    return arm in ARMS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=default_results())
    ap.add_argument("--out")
    a = ap.parse_args()
    res = a.results
    out = a.out or f"{res}/verdicts/summary.json"
    cells_root = os.environ.get("CELLS_ROOT", f"{res}/cells")
    S_ = {}

    # 1. tolerance verdicts
    rows = []
    f = f"{res}/verdicts/bootstrap.json"
    for r in (json.load(open(f)) if have(f) else []):
        if not is_arm(r["arm"]):
            continue
        rows.append(dict(ds=r["ds"], arm=r["arm"], S=r["S"], E=r["ef"], n=r["nq"], d=r["delta_pp"], ci=r["delta_ci"],
                         v=VERDICT[r["verdict"]], v1000=VERDICT[r["at1000"]["verdict"]], ndis=r["ndis_ratio"], need=r["n_needed"]))
    S_["tolerance"] = rows
    cnt = collections.Counter((r["ds"], r["arm"], r["v"]) for r in rows)
    print("== tolerance verdicts (extended test queries), dataset x arm")
    arms = sorted({r["arm"] for r in rows}, key=lambda x: (x not in ARMS, ARMS.index(x) if x in ARMS else int(x[1:])))
    for ds in L.DATASETS:
        for arm in arms:
            c = {v: cnt[(ds, arm, v)] for v in ("within", "exceeds", "undecided")}
            if sum(c.values()):
                w = min((r for r in rows if r["ds"] == ds and r["arm"] == arm), key=lambda r: r["d"])
                print(f"  {ds:8} {arm:5} within {c['within']:>2} . exceeds {c['exceeds']:>2} . undecided {c['undecided']:>2}"
                      f"   worst {w['d']:+.2f}pp (S={w['S']} E={w['E']})")
    if not rows:
        print(f"  (no {f})")

    # 2. K axis
    ka = []
    for ds in L.DATASETS:
        d = f"{res}/kaxis/{L.dsn(ds)}"
        for K in (5, 10, 20, 50, 100):
            fn = [(f"{d}/K{K}_nat_{b}.tsv", off) for b, off in (("a", 0), ("b", QOFF_B))]
            if not all(have(x) for x, _ in fn):
                continue
            nat = cat(*(tsv(x, off) for x, off in fn))
            for arm in ("dep", "ref"):
                fa = [(f"{d}/K{K}_{arm}_{b}.tsv", off) for b, off in (("a", 0), ("b", QOFF_B))]
                if all(have(x) for x, _ in fa):
                    adp = cat(*(tsv(x, off) for x, off in fa))
                    assert (nat[0] == adp[0]).all(), (ds, K, arm)
                    j = judge(nat[1], adp[1], nat[2], adp[2], cell_rng(SEED, ds, K), B_RESAMPLES, K, EPS)
                    ka.append(dict(ds=ds, K=K, arm=arm, n=j["nq"], d=j["delta_pp"], ci=j["delta_ci"], v=VERDICT[j["verdict"]],
                                   ndis=j["ndis_ratio"]))
    if not ka and have(out):
        # the K-axis dumps are not shipped: keep the K-axis rows already in the output file (the shipped summary.json)
        ka = json.load(open(out)).get("kaxis", [])
        if ka:
            print(f"  (no K-axis dumps; K-axis rows kept from {out})")
    S_["kaxis"] = ka
    print("== K axis (S=15, ef=200): dep = K=20 gammas unchanged, ref = recalibrated at that K")
    for r in ka:
        print(f"  {r['ds']:8} K={r['K']:<3} {r['arm']}  delta {r['d']:+.2f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}] {r['v']:9} ndis ratio {r['ndis']:.3f}")
    if not ka:
        print("  (no K-axis dumps)")

    # 3. approximate ground truth
    apx = []
    for ds in L.DATASETS:
        f = f"{res}/approx/{L.dsn(ds)}/calib_cells_approx.json"
        fx = f"{cells_root}/{L.dsn(ds)}/calib_cells.json"
        if not (have(f) and have(fx)):
            continue
        A, X = json.load(open(f)), json.load(open(fx))
        for key, v in sorted(A.items()):
            if key not in X:
                continue
            a_, x_ = v["gamma_rank_shard"], X[key]["gamma_rank_shard"]
            apx.append(dict(ds=ds, key=key, S=int(key.split("_")[0][1:]), same=sum(p == q for p, q in zip(a_, x_)), shards=len(a_),
                            all_same=a_ == x_, maxdiff=max(abs((p or 0) - (q or 0)) for p, q in zip(a_, x_)),
                            larger=sum((p or 0) > (q or 0) for p, q in zip(a_, x_))))
    S_["approx"] = apx
    print("== approximate ground truth: shard calibrations with the same gamma_i as the exact ground truth")
    for ds in L.DATASETS:
        xs = [r for r in apx if r["ds"] == ds]
        if xs:
            per_s = ", ".join(f"S={S} {sum(r['same'] for r in xs if r['S'] == S)}/{sum(r['shards'] for r in xs if r['S'] == S)}"
                              for S in sorted({r["S"] for r in xs}))
            print(f"  {ds:8} shards {sum(r['same'] for r in xs)}/{sum(r['shards'] for r in xs)} ({per_s}); configurations with "
                  f"every gamma_i the same {sum(r['all_same'] for r in xs)}/{len(xs)}; max diff {max(r['maxdiff'] for r in xs):.3f}")
    if apx:
        diff = sum(r["shards"] - r["same"] for r in apx)
        print(f"  all datasets: shards {sum(r['same'] for r in apx)}/{sum(r['shards'] for r in apx)} over {len(apx)} configurations;"
              f" configurations with every gamma_i the same {sum(r['all_same'] for r in apx)}/{len(apx)};"
              f" mismatches with the larger gamma_i {sum(r['larger'] for r in apx)}/{diff}")
    else:
        print("  (no approximate-ground-truth calibration)")

    # The K-axis dumps are not shipped: keep the K-axis verdicts of an existing summary rather than dropping them.
    prev = json.load(open(out)) if os.path.exists(out) else {}
    if not ka and prev.get("kaxis"):
        S_["kaxis"] = prev["kaxis"]
        print(f"  (K-axis verdicts kept from the existing {out})")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(S_, open(out, "w"), indent=1)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
