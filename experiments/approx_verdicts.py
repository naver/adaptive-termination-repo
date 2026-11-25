# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)

"""Tolerance verdicts of the per-shard gammas chosen with the approximate ground truth.

experiments/approx_cell.sh calibrates every configuration a second time against an approximate ground truth (one search
at ef_gt = min(1000, 4E) in every shard). In most shard calibrations it selects the same gamma_i as the exact ground
truth; a configuration whose gammas all agree is deployed exactly as before, so its verdict in verdicts/bootstrap.json
(arm rule) is already the verdict of the approximate ground truth. The configurations in which at least one shard's
gamma_i differs are judged again here, on the same test queries and with the same method as the other verdicts.

  list   the configurations whose gammas differ: $RESULTS/approx/<dsn>/calib_cells_approx.json against the exact
         calibration $CELLS_ROOT/<dsn>/calib_cells.json (default $RESULTS/cells), key S{S}_E{E}, per-shard
         gamma_rank_shard. One "<ds> <S> <E>" per line on stdout (experiments/run_extras.sh approxjudge dumps the arm
         approx of each with experiments/dump_cell.sh, blocks a and b).
  judge  for each listed configuration, from the per-query dumps of the arms nat (fixed ef at ef = E), rule (rank k_i,
         exact-ground-truth gammas) and approx (rank k_i, approximate-ground-truth gammas), blocks a + b:
           approx          approx against nat: the verdict of the approximate-ground-truth gammas
           exact           rule against nat: the existing verdict, recomputed (it must equal the arm-rule row of
                           verdicts/bootstrap.json when that file is present; a difference aborts)
           approx_vs_exact approx against rule: the recall the approximate-ground-truth gammas give up (or keep) relative
                           to the exact ones, and the ratio of their distance computations
         All three use lib/bootstrap.py (judge, B = 10,000, tolerance 1pp) with the seed of the configuration in
         verdicts/bootstrap.json, cell_rng(seed, ds, S, ef), and cell_rng(seed, ds, S, ef, 1000) for block a alone
         (`at1000`), so they see the same resamples as the existing verdicts.
         -> $RESULTS/approx_verdicts/verdicts.json, one row per configuration, with the shards whose gamma_i differs.

The dumps are found as lib/bootstrap.py finds them (DUMPS_DIR or --dumps, else the results tree; dumps/ of a package
run, or per_query/ with the original arm names for nat and rule). The per-query dumps are not shipped; the shipped
results/approx_verdicts/verdicts.json is the output of this script.

Usage: python3 experiments/approx_verdicts.py list  [--results DIR]
       python3 experiments/approx_verdicts.py judge [--results DIR] [--dumps DIR] [--out FILE]
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "lib"))
import bootstrap as BS  # noqa: E402
import shardlib as L  # noqa: E402

FIELDS = ("nq", "delta_pp", "delta_ci", "p_violate", "verdict", "n_needed", "ndis_ratio", "ndis_ci")
PAIR_FIELDS = ("nq", "delta_pp", "delta_ci", "ndis_ratio", "ndis_ci")


def default_results():
    return os.environ.get("RESULTS") or os.path.join(os.environ.get("WORK") or os.path.join(L.repro_root(), "work"), "results")


def key_order(k):
    S, E = k[1:].split("_E")
    return int(S), int(E)


def differing(results, cells_root):
    """[(ds, S, E, [(shard, exact gamma, approx gamma), ...])] for every configuration whose per-shard gammas differ"""
    out = []
    for f in sorted(glob.glob(os.path.join(results, "approx", "*", "calib_cells_approx.json"))):
        dsn = os.path.basename(os.path.dirname(f))
        fx = os.path.join(cells_root, dsn, "calib_cells.json")
        if not os.path.exists(fx):
            print(f"!! {fx} not found: the approximate calibration of {dsn} has nothing to compare with", file=sys.stderr)
            continue
        A, X = json.load(open(f)), json.load(open(fx))
        for key in sorted(A, key=key_order):
            if key not in X:
                print(f"!! {dsn} {key}: no exact calibration in {fx}", file=sys.stderr)
                continue
            a, x = A[key]["gamma_rank_shard"], X[key]["gamma_rank_shard"]
            diff = [(i, p, q) for i, (p, q) in enumerate(zip(x, a)) if p != q]
            if diff:
                S, E = key_order(key)
                out.append((L.ds_short(dsn), S, E, diff))
    return out


def judged(D, base, arm, ds, S, ef, a):
    """bootstrap.py main(): blocks a + b, and block a alone (at1000)"""
    q, h0, n0 = base
    q1, h1, n1 = arm
    if not (q == q1).all():
        sys.exit(f"!! query numbers differ: {ds} S={S} ef={ef}")
    full = BS.judge(h0, h1, n0, n1, BS.cell_rng(a.seed, ds, S, ef), a.B, a.K, a.eps)
    f = q < 1000
    full["at1000"] = BS.judge(h0[f], h1[f], n0[f], n1[f], BS.cell_rng(a.seed, ds, S, ef, 1000), a.B, a.K, a.eps)
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("list", "judge"))
    ap.add_argument("--results", default=default_results())
    ap.add_argument("--dumps", default=None, help="root holding dumps/ or per_query/ (default: $DUMPS_DIR, else --results)")
    ap.add_argument("--out", default=None, help="default: <results>/approx_verdicts/verdicts.json")
    ap.add_argument("--K", type=int, default=L.K); ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--eps", type=float, default=L.EPS_PP); ap.add_argument("--seed", type=int, default=20290917)
    a = ap.parse_args()
    cells_root = os.environ.get("CELLS_ROOT") or os.path.join(a.results, "cells")
    confs = differing(a.results, cells_root)
    if a.cmd == "list":
        for ds, S, E, _ in confs:
            print(ds, S, E)
        return
    if not confs:
        sys.exit("!! no configuration whose approximate-ground-truth gammas differ (run experiments/approx_cell.sh first)")
    D = BS.Dumps(a.results, a.dumps or L.dumps_dir(a.results), a.K)
    bf = os.path.join(a.results, "verdicts", "bootstrap.json")
    ref = {(r["ds"], r["S"], r["ef"]): r for r in json.load(open(bf)) if r["arm"] == "rule"} if os.path.exists(bf) else {}
    rows = []
    for ds, S, ef, diff in confs:
        nat, ex, apx = (D.load(ds, S, ef, arm) for arm in ("nat", "rule", "approx"))
        miss = [arm for arm, x in (("nat", nat), ("rule", ex), ("approx", apx)) if x is None]
        if miss:
            sys.exit(f"!! {ds} S={S} ef={ef}: no dumps of {', '.join(miss)} for blocks a and b (experiments/dump_cell.sh)")
        r = dict(ds=ds, S=S, ef=ef, shards=[dict(shard=i, gamma_exact=p, gamma_approx=q) for i, p, q in diff],
                 approx=judged(D, nat, apx, ds, S, ef, a), exact=judged(D, nat, ex, ds, S, ef, a),
                 approx_vs_exact={k: v for k, v in judged(D, ex, apx, ds, S, ef, a).items() if k in PAIR_FIELDS})
        b = ref.get((ds, S, ef))
        if b is not None:
            bad = [k for k in FIELDS if r["exact"][k] != b[k]] + \
                  [f"at1000.{k}" for k in FIELDS if r["exact"]["at1000"][k] != b["at1000"][k]]
            if bad:
                sys.exit(f"!! {ds} S={S} ef={ef}: the recomputed rule verdict differs from {bf} in {bad}")
        rows.append(r)
    out = a.out or os.path.join(a.results, "approx_verdicts", "verdicts.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(rows, open(out, "w"), indent=1)
    band = lambda x: f"{x['delta_pp']:+.3f} [{x['delta_ci'][0]:+.3f}, {x['delta_ci'][1]:+.3f}]"
    print(f"{'dataset':<9}{'S':>3}{'ef':>5}  {'approx - fixed ef':<26}{'verdict':<10}{'ndis':>7}  "
          f"{'exact - fixed ef':<26}{'verdict':<10}{'approx - exact':<26}{'ndis':>7}")
    for r in rows:
        A, X, P = r["approx"], r["exact"], r["approx_vs_exact"]
        print(f"{r['ds']:<9}{r['S']:>3}{r['ef']:>5}  {band(A):<26}{A['verdict']:<10}{A['ndis_ratio']:>7.4f}  "
              f"{band(X):<26}{X['verdict']:<10}{band(P):<26}{P['ndis_ratio']:>7.4f}")
    print(f"\n{len(rows)} configurations; exact verdicts {'checked against ' + bf if ref else 'not checked (no bootstrap.json)'}"
          f"\n-> {out}")


if __name__ == "__main__":
    main()
