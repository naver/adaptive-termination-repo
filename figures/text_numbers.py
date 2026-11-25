#!/usr/bin/env python3
# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)

"""Recompute the main numbers of the study from RESULTS and compare each with its reference value.

Each line: quantity, the reference value (computed from the shipped results, rounded), the value recomputed here
(rounded the same way), and OK / DIFF. Only numbers are compared. A few quantities are
checked against a numeric condition instead of one value (for example "8--12" or "<= 1%"); the recomputed column then
repeats the condition when the value meets it and prints the value itself otherwise. "yes" stands for a property of
the data (for example "in every configuration").
Among them are the deployed saving at every E and S, the per-S cost of the calibration table, the per-S saving of the
comparison rank and the verdict counts of the calibration-size and held-out-calibration runs.
Numbers that need the per-query dumps (not shipped) are recomputed from results/derived/ where possible; the rest are
listed as unverified.
Calibration before serving reads RESULTS/holdout/ (experiments/run_corpus.sh, corpus_img.sh); the 100M datasets are
cells/deep_N100000000 and cells/t2i100m_N100000000.
Usage: python3 text_numbers.py [--strict]      (--strict: exit 1 if any value differs)
"""
import json, statistics, sys
from fractions import Fraction
from math import comb, factorial

import numpy as np

import data as D

L = D.L
rows = []
UNVERIFIED = []


def check(what, ref, mine):
    rows.append((what, ref, mine, "OK" if ref == mine else "DIFF"))


def rng(vals, nd=0, sep="--"):
    return f"{min(vals):.{nd}f}{sep}{max(vals):.{nd}f}"


def meets(ok, cond, value):
    """the numeric condition when the recomputed value meets it, the value otherwise"""
    return cond if ok else value


T2I, MSM = "t2i", "msmarco"
MAIN4 = ("t2i", "msmarco", "glove", "sift")
BIG = ("deep", "t2i100m")
NUM = {n: str(n) for n in range(13)}                    # counts are printed as digits
boot = D._load("verdicts", "bootstrap.json")
rule = [r for r in boot if r["arm"] == "rule"]

# ---------------------------------------------------------------- overhead of forced sharding
ov = {ds: D.overhead(ds, 0.95, "fixed") for ds in (T2I, MSM)}
check("fifteen shards multiply ndis at the same recall by (T2I to MS MARCO)", "4--12x",
      f"{ov[T2I]['R'][15][0]:.0f}--{ov[MSM]['R'][15][0]:.0f}x")
check("ndis x one index at the same recall 0.95, S=15, T2I / MS MARCO", "4.0x / 12x",
      f"{ov[T2I]['R'][15][0]:.1f}x / {ov[MSM]['R'][15][0]:.0f}x")
check("ndis x one index at the same recall 0.95, S=50, T2I / MS MARCO", "8.1x / 31x",
      f"{ov[T2I]['R'][50][0]:.1f}x / {ov[MSM]['R'][50][0]:.0f}x")
check("ef that reaches recall 0.95, T2I S=15 vs S=1", "88 vs 387", f"{D.matched_ef(T2I, 15)} vs {D.matched_ef(T2I, 1)}")
check("MS MARCO: ef that reaches recall 0.95 at S=1; at S=5; five shards cost (x one index)", "63; 50; 4.9x",
      f"{D.matched_ef(MSM, 1)}; {D.matched_ef(MSM, 5)}; {ov[MSM]['R'][5][0]:.1f}x")
check("same ef=200, S=15: ndis x one index, T2I / MS MARCO", "12.7x / 18x",
      f"{ov[T2I]['ef'][15][0]:.1f}x / {ov[MSM]['ef'][15][0]:.0f}x")
check("T2I merged recall of fixed ef at ef=200, S=1 / S=15 / S=50", "0.916 / 0.975 / 0.987",
      " / ".join(f"{ov[T2I]['ef'][S][2]:.3f}" for S in (1, 15, 50)))

# ---------------------------------------------------------------- arrival-time oracle, fixed ef matched to recall 0.95
w = {(ds, S): D.waste(S, ds) for ds in (T2I, MSM) for S in D.SS}
orc = {k: v["oracle"] / v["total"] * 100 for k, v in w.items()}
empty = {k: v["empty"] / v["total"] * 100 for k, v in w.items()}
check("oracle needs (% of fixed-ef ndis), T2I S=1 / 15 / 50; MS MARCO S=1 / 15 / 50", "47 / 22 / 16; 54 / 26 / 18",
      "; ".join(" / ".join(f"{orc[ds, S]:.0f}" for S in (1, 15, 50)) for ds in (T2I, MSM)))
check("oracle would skip at S=15, recall 0.95 (%), T2I / MS MARCO", "78 / 74",
      f"{100 - orc[T2I, 15]:.0f} / {100 - orc[MSM, 15]:.0f}")
check("avoidable share of the fixed-ef ndis at fifteen shards", "70--80%",
      meets(all(abs(1 - orc[ds, 15] / 100 - 0.75) < 0.05 for ds in (T2I, MSM)), "70--80%",
            f"{100 - orc[T2I, 15]:.0f}, {100 - orc[MSM, 15]:.0f}%"))
check("non-contributing shards' bottom-layer share of all ndis (%), S=15 / S=50, T2I; MS MARCO", "26 / 62; 25 / 61",
      "; ".join(f"{empty[ds, 15]:.0f} / {empty[ds, 50]:.0f}" for ds in (T2I, MSM)))
check("(1-1/S)^K at S=15, K=20 (%)", "25", f"{(1 - 1 / 15) ** 20 * 100:.0f}")
third = [empty[ds, 15] / (100 - orc[ds, 15]) for ds in (T2I, MSM)]
check("share of the avoidable work in non-contributing shards, S=15, T2I and MS MARCO", "0.33 +- 0.05",
      meets(all(abs(x - 1 / 3) < 0.05 for x in third), "0.33 +- 0.05", ", ".join(f"{x:.2f}" for x in third)))

# ---------------------------------------------------------------- result count k_i
check("k_i at K=20, S=1, 5, 15, 30, 50", "20, 12, 7, 5, 4", ", ".join(str(D.k_i(S)) for S in D.SS))


def p_lost(K, S, k):
    """E[true neighbors cut by the limit k] / K with c_i ~ Bin(K, 1/S), summed over the S shards"""
    p = Fraction(1, S)
    return float(S * sum(comb(K, c) * p ** c * (1 - p) ** (K - c) * (c - k) for c in range(k + 1, K + 1)) / K)


check("recall lost when a shard holds more than k_i true neighbors, S=15 / S=50 (pp)", "0.002 / 0.010",
      " / ".join(f"{p_lost(20, S, D.k_i(S)) * 100:.3f}" for S in (15, 50)))

# ---------------------------------------------------------------- prediction error and per-shard choice (calibration queries)
# level: measured minus predicted merged recall with the test off (gamma index 0), no choice of gamma involved.
# drop: each shard picks its own gamma_i from its prediction (gamma_rank_shard, as deployed); measured drop of
# merged recall vs predicted drop, the mean over shards of Rhat_i(off) - Rhat_i(gamma_i).
pe, md, de, spread, share_low = [], [], [], [], []
for ds in MAIN4:
    cc = D._load(D.cells(ds, "calib_cells.json"))
    for S in L.S_VALUES[ds]:
        if S < 5:
            continue
        for E in L.E_LIMITS:
            c = cc[f"S{S}_E{E}"]
            g = [float(x) for x in c["gammas"]]
            sh = c["Rhat_rank_shard"]
            gi = [g.index(0.0 if x is None else x) for x in c["gamma_rank_shard"]]
            pred = statistics.mean(r[0] - r[i] for r, i in zip(sh, gi))
            meas = c["R"][0] - c["R_rank_shard_gammas"]
            pe.append((c["R"][0] - statistics.mean(r[0] for r in sh)) * 100)
            md.append(meas * 100)
            de.append((meas - pred) * 100)
            on = [x for x in c["gamma_rank_shard"] if x is not None]
            spread.append(max(on) - min(on) if on else 0.0)
            share_low.append((c["R"][0] - c["R"][g.index(min(on))]) * 100 if on else 0.0)
check("configurations with S >= 5; test off: measured above the prediction in; by up to (pp); the last below by (pp)",
      "36; 35; 1.2; 0.05",
      f"{len(pe)}; {sum(x > 0 for x in pe)}; {max(pe):.1f}; {max(-x for x in pe):.2f}")
check("gamma_i within a configuration differ by at most", "0.03", f"{max(spread):.2f}")
check("smallest gamma_i applied to every shard: drop beyond the tolerance in", "23 of the 36",
      f"{sum(x > L.EPS_PP for x in share_low)} of the {len(share_low)}")
check("deployed gamma_i on the calibration queries: merged recall falls by at most (pp); below the tolerance",
      "0.84; yes", f"{max(md):.2f}; {'yes' if max(md) < L.EPS_PP else 'no'}")
check("measured drop smaller than predicted in; larger by at most (pp) in the other", "32; 0.12 in the other 4",
      f"{sum(x < 0 for x in de)}; {max(de):.2f} in the other {NUM.get(sum(x >= 0 for x in de), sum(x >= 0 for x in de))}")
sl = [c["short_lists"] / (c["nq"] * c["S"]) * 100 for ds in (T2I, MSM)
      for c in D._load(D.cells(ds, "calib_cells.json")).values() if c["S"] == 50]
check("ground-truth lists from the global top-300 shorter than the table needs at S=50 (%)", "15", rng(sl, 0, "--")
      if round(min(sl)) != round(max(sl)) else f"{min(sl):.0f}")

# ---------------------------------------------------------------- setup, metrics and matching
nq = {ds: sorted({x["nq"] for x in rule if x["ds"] == ds}) for ds in MAIN4 + BIG}
check("test queries T2I 10M, MS MARCO, GloVe, SIFT, DEEP 100M, T2I 100M",
      "20,000, 6,480, 9,500, 9,500, 9,500, 20,000",
      ", ".join("/".join(f"{n:,}" for n in nq[ds]) for ds in ("t2i", "msmarco", "glove", "sift", "deep", "t2i100m")))
cpu1, cpu50 = D.matched(T2I, 1, 0.95, "fixed")[2], D.matched(T2I, 50, 0.95, "fixed")[2]
check("T2I CPU of fixed ef per query at recall 0.95, summed over shards, S=1 / S=50 (ms)", "1.47 / 7.42",
      f"{cpu1:.2f} / {cpu50:.2f}")
cn = D.cpu_ndis(T2I, L.E_LIMITS)
div = max(abs(rn - rc) for rows_ in cn.values() for _, rn, rc, _, _ in rows_) * 100
ncn = sum(len(rows_) for rows_ in cn.values())
check("CPU ratio vs ndis ratio at the same S and ef, T2I: configurations; differ by at most (points)", "9; 1.9",
      f"{NUM.get(ncn, ncn)}; {div:.1f}")
sv95 = {ds: D.savings(ds, 0.95) for ds in (T2I, MSM)}


def saved(ds, key, i, Ss=D.SS):
    return [(1 - sv95[ds][key][S][i]) * 100 for S in Ss]


check("T2I S=15, same recall: adaptive saves CPU / ndis (%)", "8 / 16",
      f"{saved(T2I, 'R', 1, (15,))[0]:.0f} / {saved(T2I, 'R', 0, (15,))[0]:.0f}")
check("T2I S=5 R=0.95: fixed ef; adaptive E; adaptive needs this share of fixed-ef ndis", "160; 300; 0.82",
      f"{D.matched(T2I, 5, 0.95, 'fixed')[3]}; {D.matched(T2I, 5, 0.95, 'adaptive')[3][0]}; "
      f"{sv95[T2I]['R'][5][0]:.2f}")
dep5 = (1 - D.dataset_ratio(T2I, 5)) * 100
rec5 = D._load(D.cells(T2I, "calib", "S5_E200.json"))[0]["recall"]
check("T2I S=5: deployed configuration saves (%) at recall; the same-recall example saves (%)", "15 at 0.951; 18",
      f"{dep5:.0f} at {rec5:.3f}; {(1 - sv95[T2I]['R'][5][0]) * 100:.0f}")
shared = []
for ds in MAIN4:
    cc = D._load(D.cells(ds, "calib_cells.json"))
    for v in cc.values():
        if v["S"] == 1 or v.get("rule_k") != v.get("k_i"):
            continue
        Rg = np.array(v["R"][1:] + v["R"][:1]); Ng = np.array(v["ndis"][1:] + v["ndis"][:1]) / v["ndis"][0]
        o = np.argsort(Rg)
        shared.append(v["ratio_rank_shard_gammas"] / float(np.interp(v["R_rank_shard_gammas"], Rg[o], Ng[o])))
check("one shared gamma vs per-shard gamma_i, S > 1: configurations; more ndis at most (%); median (%)",
      "36; 4; 1.2", f"{len(shared)}; {(1 / min(shared) - 1) * 100:.0f}; {(1 / statistics.median(shared) - 1) * 100:.1f}")

# ---------------------------------------------------------------- deployed saving
REF_DEPLOYED = {     # (E, ds): (saved % at S=1, 5, 15; recall at S=1, 5, 15)
    (100, "t2i"): ("4, 12, 11", ".860, .918, .946"), (100, "msmarco"): ("23, 29, 32", ".953, .959, .966"),
    (100, "glove"): ("10, 16, 19", ".799, .882, .933"), (100, "sift"): ("11, 6, 3", ".981, .991, .993"),
    (200, "t2i"): ("6, 15, 21", ".911, .951, .968"), (200, "msmarco"): ("36, 39, 34", ".968, .969, .973"),
    (200, "glove"): ("15, 24, 29", ".852, .925, .967"), (200, "sift"): ("12, 5, 3", ".987, .992, .994"),
    (300, "t2i"): ("13, 19, 28", ".930, .963, .975"), (300, "msmarco"): ("46, 42, 37", ".973, .972, .977"),
    (300, "glove"): ("20, 27, 29", ".880, .949, .978"), (300, "sift"): ("14, 5, 3", ".990, .992, .994"),
}
dsr = {(ds, S, E): (1 - D.dataset_ratio(ds, S, E)) * 100 for ds in MAIN4 for S in (1, 5, 15) for E in L.E_LIMITS}
for (E, ds), (p_saved, p_rec) in REF_DEPLOYED.items():
    check(f"deployed, E={E}, {ds}: saved (%) at S=1, 5, 15", p_saved,
          ", ".join(f"{dsr[ds, S, E]:.0f}" for S in (1, 5, 15)))
    check(f"deployed, E={E}, {ds}: recall at S=1, 5, 15", p_rec,
          ", ".join(f"{D._load(D.cells(ds, 'calib', f'S{S}_E{E}.json'))[0]['recall']:.3f}"[1:] for S in (1, 5, 15)))
check("deployed: configurations that save ndis", "36", str(sum(v > 0 for v in dsr.values())))
check("deployed at E=200, S=5 and 15: saved, T2I / MS MARCO (%)", "15--21 / 34--39",
      f"{rng([dsr[T2I, 5, 200], dsr[T2I, 15, 200]])} / {rng([dsr[MSM, 5, 200], dsr[MSM, 15, 200]])}")
sift_ef = [D._load(D.cells("sift", "match", f"tab_S{S}_E200_fixed.json"))["ef"] for S in (1, 5, 15)]
check("SIFT: matched fixed ef at E=200, S=1, 5, 15", "93, 50, 33", ", ".join(map(str, sift_ef)))
big = {(ds, S): (1 - D.dataset_ratio(ds, S)) * 100 for ds in BIG for S in (5, 15)}
check("100M, E=200: saved at S=5 and 15 (%), DEEP; T2I 100M", "19 and 15; 10 and 14",
      f"{big['deep', 5]:.0f} and {big['deep', 15]:.0f}; {big['t2i100m', 5]:.0f} and {big['t2i100m', 15]:.0f}")

# ---------------------------------------------------------------- fixed recall 0.95 and same ef 200
check("saved at recall 0.95, CPU, S=1..50, T2I / MS MARCO (%)", "8--14 / 16--25",
      f"{rng(saved(T2I, 'R', 1))} / {rng(saved(MSM, 'R', 1))}")
s15 = {ds: saved(ds, "R", 0, (15,))[0] for ds in (T2I, MSM)}
check("saved at recall 0.95, S=15, ndis, T2I / MS MARCO (%)", "16 / 23", f"{s15[T2I]:.0f} / {s15[MSM]:.0f}")
frac = [s15[ds] / (100 - orc[ds, 15]) for ds in (T2I, MSM)]
check("that saving as a share of the avoidable work, T2I and MS MARCO", "0.20 and 0.33, each +- 0.05",
      meets(abs(frac[0] - 1 / 5) < 0.05 and abs(frac[1] - 1 / 3) < 0.05, "0.20 and 0.33, each +- 0.05",
            ", ".join(f"{x:.2f}" for x in frac)))
check("MS MARCO S=15, saved at recall 0.95: CPU; ndis (%)", "23; 23",
      f"{saved(MSM, 'R', 1, (15,))[0]:.0f}; {s15[MSM]:.0f}")
ova = {ds: D.overhead(ds, 0.95, "adaptive") for ds in (T2I, MSM)}
check("overhead at S=15, recall 0.95, ndis: fixed -> adaptive, T2I / MS MARCO", "4.0x -> 3.4x / 12x -> 9.4x",
      f"{ov[T2I]['R'][15][0]:.1f}x -> {ova[T2I]['R'][15][0]:.1f}x / {ov[MSM]['R'][15][0]:.0f}x -> {ova[MSM]['R'][15][0]:.1f}x")
check("same ef=200, S=15: CPU saved, MS MARCO / T2I (%)", "61 / 36",
      f"{saved(MSM, 'ef', 1, (15,))[0]:.0f} / {saved(T2I, 'ef', 1, (15,))[0]:.0f}")
check("S=50: ndis saved at the same ef=200, T2I / MS MARCO; at recall 0.95 (%)", "49 / 63; 15 / 19",
      f"{saved(T2I, 'ef', 0, (50,))[0]:.0f} / {saved(MSM, 'ef', 0, (50,))[0]:.0f}; "
      f"{saved(T2I, 'R', 0, (50,))[0]:.0f} / {saved(MSM, 'R', 0, (50,))[0]:.0f}")
loss = max((D.same_ef(ds, S, "fixed")[0] - D.same_ef(ds, S, "adaptive")[0]) * 100 for ds in (T2I, MSM) for S in D.SS)
check("recall adaptive termination gives up at ef=200, up to (pp)", "1", f"{loss:.0f}")
keep, krec = {}, []
for ds in (T2I, MSM):
    keep[ds] = []
    for S in (15, 30, 50):
        f, a, m = D.same_ef(ds, S, "fixed"), D.same_ef(ds, S, "adaptive"), D.matched(ds, S, 0.95, "fixed")
        keep[ds].append((f[1] - a[1]) / (f[1] - m[1]) * 100)
        krec.append(a[0])
check("E=200, S=15..50: share of the work retuning fixed ef to 0.95 removes, T2I / MS MARCO (%); recall",
      "75--79 / 80--85; 0.968--0.981", f"{rng(keep[T2I])} / {rng(keep[MSM])}; {rng(krec, 3)}")
check("E=200 from S=15 to 50: saves of what retuning ef would save", ">= 74.5%",
      meets(min(keep[T2I] + keep[MSM]) >= 74.5, ">= 74.5%", f"{min(keep[T2I] + keep[MSM]):.0f}%"))

# ---------------------------------------------------------------- queue saturation (experiments/patience.py; results/patience/)
pr = {(ds, S): D.patience_same_recall(ds, S, 0.95) for ds in (T2I, MSM) for S in D.SS}
check("Patience at recall 0.95 counting the E nearest points: saved vs fixed ef, S=1..50 (%)", "2--7",
      rng([(1 - v["ef"] / v["fixed"]) * 100 for v in pr.values()]))
check("Patience at recall 0.95 counting the k_i results: saved at most (%)", "13",
      f"{max((1 - v['ki'] / v['fixed']) * 100 for v in pr.values()):.0f}")
check("adaptive termination vs the better Patience variant at recall 0.95: fewer ndis (%); in every configuration",
      "9--19; yes", f"{rng([(1 - v['adaptive'] / min(v['ef'], v['ki'])) * 100 for v in pr.values()])}; "
      f"{'yes' if all(v['adaptive'] < min(v['ef'], v['ki']) for v in pr.values()) else 'no'}")
pd_ = [D.patience_default(ds, S, E) for ds in (T2I, MSM) for S in D.SS for E in L.E_LIMITS]
exc = [d for d in pd_ if d["bootstrap"]["verdict"] == "exceeds"]
check("Patience with Lucene's defaults: test queries; exceeds the tolerance in; all on T2I at S <= 5",
      "1,000; 4 of 30; yes", f"{pd_[0]['bootstrap']['nq']:,}; {len(exc)} of {len(pd_)}; "
      f"{'yes' if all(d['ds'] == T2I and d['S'] <= 5 for d in exc) else 'no'}")
check("Patience with Lucene's defaults: saved at the recall it reaches, at most (%)", "7",
      f"{max((1 - d['ratio_same_recall']) * 100 for d in pd_):.0f}")
p15 = {ds: (1 - D.patience_default(ds, 15, 200)["ratio_same_recall"]) * 100 for ds in (T2I, MSM)}
check("S=15, E=200, saved: Patience with defaults T2I / MS MARCO vs adaptive termination T2I / MS MARCO (%)",
      "3 / 1 vs 21 / 34", f"{p15[T2I]:.0f} / {p15[MSM]:.0f} vs {dsr[T2I, 15, 200]:.0f} / {dsr[MSM, 15, 200]:.0f}")

# ---------------------------------------------------------------- slowest shard and ef limit (derived or raw)
tail = D.shard_tail_reader()
t = tail("t2i", 15, 300, "cpu")
check("T2I S=15 ef=300 CPU saved: summed, slowest shard; costliest 1% summed, slowest (%)", "44.9, 29.2, 7.8, 3.9",
      ", ".join(f"{t[k][0]:.1f}" for k in (("total", "mean"), ("slowest", "mean"), ("total", "worst1"), ("slowest", "worst1"))))
gap = max(abs(tail(ds, S, ef, "cpu")[k][0] - tail(ds, S, ef, "ndis")[k][0]) for ds in (T2I, MSM) for S, ef in D.TAIL_CELLS
          for k in (("total", "mean"), ("slowest", "mean"), ("total", "worst1"), ("slowest", "worst1")))
order = all(tail(ds, S, ef, met)[("total", st)][0] >= tail(ds, S, ef, met)[("slowest", st)][0]
            for ds in (T2I, MSM) for S, ef in D.TAIL_CELLS for met in ("cpu", "ndis") for st in ("mean", "worst1"))
# "within 2.6 points" is a bound: from the dumps the largest gap is 2.54, from the 0.1-rounded derived/ values 2.6
check("distance computations vs CPU: same order; within (points)", "yes; <= 2.6",
      f"{'yes' if order else 'no'}; " + meets(gap <= 2.6 + 1e-9, "<= 2.6", f"{gap:.2f}"))
cap = D.efcap_reader()
grp = {(ds, g): [cap(ds, S, ef)[g] for S, ef in D.EFCAP_CELLS] for ds in (T2I, MSM) for g in ("low", "perfect")}
check("shard searches that reach the ef limit, hard vs easy queries, T2I; MS MARCO (%)", "77--95 vs 15--42; 42--62 vs 1--3",
      f"{rng(grp[T2I, 'low'])} vs {rng(grp[T2I, 'perfect'])}; {rng(grp[MSM, 'low'])} vs {rng(grp[MSM, 'perfect'])}")

# ---------------------------------------------------------------- comparison rank
dab = {(ds, S): D.dabs_cost(ds, S) for ds in MAIN4 for S in (5, 15, 30, 50) if S in L.S_VALUES[ds]}
sv = lambda r: (1 - 1 / r) * 100            # distance computations k_i saves against rank K, %
worst20 = max(-sv(v) for v in dab.values())
check("K=20, mean over E: k_i costs more than rank-K termination by", "<= 1%",
      meets(worst20 <= 1, "<= 1%", f"{worst20:.2f}%"))
within1 = max(abs(sv(v)) for (d, _), v in dab.items() if d in ("t2i", "sift"))
ms20 = [sv(dab["msmarco", S]) for S in (5, 15, 30, 50)]
check("K=20: T2I and SIFT differ by less than 1%; GloVe saved (%); MS MARCO saved",
      "yes; 3.5--3.6; 8--12",
      f"{'yes' if within1 < 1 else f'no ({within1:.2f})'}; {rng([sv(dab['glove', S]) for S in (5, 15)], 1)}; "
      + meets(all(8 <= x <= 12 for x in ms20), "8--12", rng(ms20, 1)))
rk = D._load("rank_k100", "summary.json")
rkc, rkm = rk["cells"], rk["means"]
k100 = {(ds, S): sv(rkm[f"K100_{ds}_S{S}"]["ratio_drop1"]) for ds in (T2I, MSM) for S in (5, 15, 30, 50)}
check("K=100, mean over E 200, 300: k_i saves in every configuration; T2I S=5 to S=50; MS MARCO S=5; S >= 15 (%)",
      "yes; 4 to 20; 17; 25--29",
      f"{'yes' if min(k100.values()) > 0 else 'no'}; {k100[T2I, 5]:.0f} to {k100[T2I, 50]:.0f}; {k100[MSM, 5]:.0f}; "
      f"{rng([k100[MSM, S] for S in (15, 30, 50)])}")
check("K=100: k_i saves up to, T2I (%); MS MARCO", "20; > 25%",
      f"{max(k100[T2I, S] for S in (5, 15, 30, 50)):.0f}; "
      + meets(max(k100[MSM, S] for S in (5, 15, 30, 50)) > 25, "> 25%",
              f"{max(k100[MSM, S] for S in (5, 15, 30, 50)):.0f}%"))
# per S: k_i at K = 20 and 100, then the distance computations k_i saves against rank K (%), T2I and MS MARCO at
# K = 20 (mean over E 100, 200, 300) and K = 100 (mean over E 200, 300; E = 300 alone for MS MARCO at S >= 30)
REF_RANK = {5: "12, 37; 1, 4; 9, 17", 15: "7, 18; -1, 12; 11, 25", 30: "5, 12; -1, 18; 9, 29", 50: "4, 9; 0, 20; 10, 26"}
for S, want in REF_RANK.items():
    check(f"comparison rank, S={S}: k_i at K=20, 100; saved (%) T2I K=20, 100; MS MARCO K=20, 100", want,
          f"{L.k_i_of(20, S)}, {L.k_i_of(100, S)}; " + "; ".join(
              f"{round(sv(dab[ds, S]))}, {round(sv(rkm[f'K100_{ds}_S{S}']['ratio_drop1']))}" for ds in (T2I, MSM)))
# deployed at K=20 with per-shard gamma_i, same E: verdicts/bootstrap.json, arms rule (k_i) and dabs (rank K)
bv = {(r["arm"], r["ds"], r["S"], r["ef"]): r for r in boot if r["arm"] in ("rule", "dabs")}
more, less_loss = {}, []
for ds in MAIN4:
    more[ds] = []
    for S in L.S_VALUES[ds]:
        if S < 5:
            continue
        more[ds].append(statistics.mean(bv["dabs", ds, S, E]["ndis_ratio"] / bv["rule", ds, S, E]["ndis_ratio"] - 1
                                        for E in L.E_LIMITS) * 100)
        less_loss += [bv["dabs", ds, S, E]["delta_pp"] - bv["rule", ds, S, E]["delta_pp"] for E in L.E_LIMITS]
check("deployed at K=20, S >= 5: rank-K termination needs more ndis, T2I / SIFT / GloVe / MS MARCO (%)",
      "0--5 / 5--6 / 7--11 / 8--23", " / ".join(rng(more[ds]) for ds in ("t2i", "sift", "glove", "msmarco")))
check("deployed at K=20: rank-K recall loss smaller by at most (pp)", "0.45", f"{max(less_loss):.2f}")
fl20 = min(v["floor_dabs"]["drop_pp"] for v in rkc.values() if v["K"] == 20)
check("rank-K recall floor below fixed ef: K=20; K=100 (pp)", "> 4; 0.9--5.8",
      meets(fl20 > 4, "> 4", f"{fl20:.1f}") + "; "
      + rng([v["floor_dabs"]["drop_pp"] for v in rkc.values() if v["K"] == 100], 1))
gk = [v["drop1"]["gamma_dabs"] for v in rkc.values() if v["K"] == 100 and v["drop1"]["gamma_dabs"]]
check("K=100: gamma of rank-K termination at the target; cells where it cannot reach it",
      "0.0009--0.012; msmarco S=30 E=200, msmarco S=50 E=200",
      f"{min(gk):.4f}--{max(gk):.3f}; " + ", ".join(f"{v['ds']} S={v['S']} E={v['E']}" for v in rkc.values()
                                                    if v["K"] == 100 and v["drop1"]["below_floor"]))
check("K=20 T2I S=50 E=200: gamma of rank-K termination; of adaptive termination", "0.024; 0.050",
      f"{rkc['K20_t2i_S50_E200']['drop1']['gamma_dabs']:.3f}; {rkc['K20_t2i_S50_E200']['drop1']['gamma_rule']:.3f}")
mk = [D._load("rank_k100", "mech", D.DSN(ds), f"K100_dabs_S{S}_E{E}.json")
      for ds in (T2I, MSM) for S in (5, 15, 30, 50) for E in (200, 300)]
have = statistics.median(m["all"]["base_at_have"]["median"] for m in mk)
least = min(m["all"]["at_gamma"][m["gammas"][0]]["base_at_stop"]["min"] for m in mk)
check("K=100, rank-K termination: ndis to find 100 points (median); ndis of every shard search", "115 +- 5; 600--699",
      meets(abs(have - 115) <= 5, "115 +- 5", f"{have:.0f}") + "; "
      + meets(600 <= least < 700, "600--699", f"{least:.0f}"))

# ---------------------------------------------------------------- tolerance verdicts (verdicts/bootstrap.json)
main = [r for r in rule if r["ds"] in MAIN4]
bigv = [r for r in rule if r["ds"] in BIG]
check("configurations: up to 10M + 100M = all; exceed the tolerance; 100M all sharded",
      "48 + 12 = 60; 0; yes", f"{len(main)} + {len(bigv)} = {len(main) + len(bigv)}; "
      f"{sum(r['verdict'] == 'exceeds' for r in rule)}; {'yes' if all(r['S'] > 1 for r in bigv) else 'no'}")
top = max((r for r in rule if r["S"] > 1), key=lambda r: -r["delta_pp"])
check("largest loss on a sharded index (pp)", "0.91, msmarco ef=300", f"{-top['delta_pp']:.2f}, {top['ds']} ef={top['ef']}")
und = sorted((r["ds"], r["S"], r["ef"]) for r in rule if r["verdict"] != "within")
check("configurations not within (all undecided, all on a single index)", "sift S=1 ef=200, t2i S=1 ef=300",
      ", ".join(f"{d} S={S} ef={e}" for d, S, e in und) + ("" if all(r["verdict"] == "undecided" for r in rule
                                                                   if r["verdict"] != "within") else " (some exceed)"))
for (ds, S, ef), ref in ((("t2i", 1, 300), "-1.05 [-1.12, -0.99]"), (("sift", 1, 200), "-1.05 [-1.10, -1.00]")):
    r = next(r for r in rule if (r["ds"], r["S"], r["ef"]) == (ds, S, ef))
    check(f"{ds} S={S} ef={ef}: delta [95% interval] (pp)", ref,
          f"{r['delta_pp']:.2f} [{r['delta_ci'][0]:.2f}, {r['delta_ci'][1]:.2f}]")
beyond = max(-r["delta_pp"] - 1 for r in rule if r["verdict"] == "undecided")
check("undecided point estimates beyond the tolerance, at most (pp)", "0.05", f"{beyond:.2f}")
excess = max(-r["delta_ci"][0] - 1 for r in rule)
check("largest excess beyond the tolerance by interval, below the 0.2 pp margin", "yes",
      "yes" if max(excess, beyond) < 0.2 else f"no ({excess:.2f})")

# ---------------------------------------------------------------- calibration cost
bc = {ds: D.buildcost(ds) for ds in (T2I, MSM)}
# per S: build CPU s (all shard graphs); exact ground truth % of it, CPU (wall); approximate ground truth % CPU;
# shard calibrations that select the same gamma_i with the approximate ground truth
REF_BUILDCOST = {
    T2I: {1: "10104; 1.6 (5.4); 0.1; 3/3", 5: "7367; 2.4 (10.5); 0.4; 14/15", 15: "5918; 3.2 (20.4); 1.2; 45/45",
          30: "5064; 4.2 (35.9); 2.6; 90/90", 50: "4456; 5.5 (57.8); 4.6; 149/150"},
    MSM: {1: "22497; 2.4 (5.9); 0.0; 2/3", 5: "23599; 2.3 (7.1); 0.2; 15/15", 15: "22236; 2.6 (11.0); 0.7; 44/45",
          30: "20271; 3.1 (17.3); 1.4; 89/90", 50: "18384; 3.7 (26.0); 2.4; 146/150"},
}
for ds, want_s in REF_BUILDCOST.items():
    for S, want in want_s.items():
        v = bc[ds][S]
        same = "---" if v["same"] is None else f"{v['same']}/{v['n_cal']}"
        check(f"calibration cost, {ds} S={S}: build CPU s; exact GT % CPU (% wall); approx. GT % CPU; same gamma_i",
              want, f"{v['build_cpu']:.0f}; {v['exact_pct']:.1f} ({v['exact_wall_pct']:.1f}); {v['approx_pct']:.1f}; {same}")
check("exact ground truth CPU s, T2I / MS MARCO", "161 / 533",
      f"{bc[T2I][1]['gt_cpu']:.0f} / {bc[MSM][1]['gt_cpu']:.0f}")
rr = {ds: [v["record_replay_cpu"] for v in bc[ds].values()] for ds in (T2I, MSM)}
check("record + replay CPU s over three ef limits, T2I / MS MARCO", "3.7--82 / 3.9--144",
      f"{min(rr[T2I]):.1f}--{max(rr[T2I]):.0f} / {min(rr[MSM]):.1f}--{max(rr[MSM]):.0f}")
busy = [bc[ds][1]["gt_cpu"] / bc[ds][1]["gt_wall"] for ds in (T2I, MSM)]
check("exact ground truth: threads busy of its 16", "10--11", f"{min(busy):.0f}--{max(busy):.0f}")
apx = [v["approx_pct"] for ds in (T2I, MSM) for S, v in bc[ds].items() if S <= 15]
check("approximate ground truth at S <= 15, at most (% of build)", "1.2", f"{max(apx):.1f}")
check("S=50: approx vs exact, T2I; MS MARCO (%)", "4.6 vs 5.5; 2.4 vs 3.7",
      f"{bc[T2I][50]['approx_pct']:.1f} vs {bc[T2I][50]['exact_pct']:.1f}; {bc[MSM][50]['approx_pct']:.1f} vs {bc[MSM][50]['exact_pct']:.1f}")
sh, nsh, cfg, ncfg, bigger, steps = 0, 0, 0, 0, 0, []
for ds in MAIN4:
    ap = D._load("approx", D.DSN(ds), "calib_cells_approx.json")
    exc = D._load(D.cells(ds, "calib_cells.json"))
    for k, v in ap.items():
        a, x = v["gamma_rank_shard"], exc[k]["gamma_rank_shard"]
        cfg += a == x
        ncfg += 1
        sh += sum(p == q for p, q in zip(a, x))
        nsh += len(a)
        steps += [(round(abs((p or 0) - (q or 0)) / 0.005), ds, v["S"]) for p, q in zip(a, x) if p != q]
        bigger += sum((p or 0) > (q or 0) for p, q in zip(a, x))
check("approx GT selects the same gamma_i: shard calibrations; configurations", "723 of 732; 48",
      f"{sh} of {nsh}; {ncfg}")
check("mismatches; all pick a larger gamma_i", "9; yes",
      f"{NUM.get(nsh - sh, nsh - sh)}; {'yes' if bigger == nsh - sh else 'no'}")
two = [(d, S) for n, d, S in steps if n == 2]
check("mismatches larger by one grid step of 0.005; by two steps (where)", "8; 1, msmarco S=50",
      f"{NUM.get(sum(n == 1 for n, _, _ in steps), sum(n == 1 for n, _, _ in steps))}; "
      f"{len(two)}, " + ", ".join(f"{d} S={S}" for d, S in two))
# the configurations with a mismatch, judged again with the approximate-GT gamma_i (experiments/approx_verdicts.py)
av = D._load("approx_verdicts", "verdicts.json")
check("approx-GT gamma_i judged again: configurations with a mismatch; within the tolerance", "8; 8",
      f"{NUM.get(len(av), len(av))}; {NUM.get(sum(r['approx']['verdict'] == 'within' for r in av), '?')}")
gain = [r["approx_vs_exact"]["delta_pp"] for r in av]
cost = [(r["approx_vs_exact"]["ndis_ratio"] - 1) * 100 for r in av]
check("approx-GT gamma_i: less recall lost (pp); more distance computations (%)", "0.005--0.24; 0.2--7.0",
      f"{min(gain):.3f}--{max(gain):.2f}; {rng(cost, 1)}")
check("configurations whose verdict is unchanged (same gamma_i)", "40", str(cfg))

# ---------------------------------------------------------------- what the calibration needs


def needs_row(rs):
    return (f"{len(rs)}, {sum(r['verdict'] == 'within' for r in rs)}, {sum(r['verdict'] == 'undecided' for r in rs)}, "
            f"{sum(r['verdict'] == 'exceeds' for r in rs)}, {min(r['delta_pp'] for r in rs):.2f}")


nc = {arm: [r for r in boot if r["arm"] == arm and r["ds"] in MAIN4 and r["S"] in (1, 5, 15)]
      for arm in ("rule", "p250", "p125")}
for arm, n, ref in (("rule", 500, "36, 34, 2, 0, -1.05"), ("p250", 250, "36, 31, 4, 1, -1.12"),
                      ("p125", 125, "36, 22, 7, 7, -1.37")):
    check(f"{n} calibration queries: configurations, within, undecided, exceeds, worst (pp)", ref,
          needs_row(nc[arm]))
e250 = [r for r in nc["p250"] if r["verdict"] == "exceeds"]
check("250 queries: the configuration that exceeds; 125 queries: how many exceed", "t2i S=5 ef=300; 7",
      ", ".join(f"{r['ds']} S={r['S']} ef={r['ef']}" for r in e250) + "; "
      + str(NUM.get(sum(r["verdict"] == "exceeds" for r in nc["p125"]), "?")))
hb = D._load("holdout", "verdicts", "bootstrap.json")
co = {ds: [r for r in hb if r["arm"] == "corpus" and r["ds"] == ds] for ds in MAIN4}
ia = [r for r in D._load("holdout", "img", "A", "verdicts", "bootstrap.json") if r["arm"] == "corpus"]
for name, rs, ref in (("GloVe, SIFT", co["glove"] + co["sift"], "18, 16, 1, 1, -1.14"),
                        ("T2I, image queries", ia, "9, 8, 1, 0, -1.00"),
                        ("T2I, text queries", co["t2i"], "9, 9, 0, 0, -0.20"),
                        ("MS MARCO", co["msmarco"], "9, 0, 2, 7, -1.61")):
    check(f"held-out vectors, {name}: configurations, within, undecided, exceeds, worst (pp)", ref,
          needs_row(rs))
gs = co["glove"] + co["sift"]
check("held-out vectors, GloVe and SIFT: exceed of all; by (pp)", "1 of 18; 0.14",
      f"{sum(r['verdict'] == 'exceeds' for r in gs)} of "
      f"{len(gs)}; {max(-r['delta_pp'] - 1 for r in gs if r['verdict'] == 'exceeds'):.2f}")
m_ = co["msmarco"]
check("held-out passages, MS MARCO: exceed of all; by at most (pp)", "7 of 9; 0.61",
      f"{sum(r['verdict'] == 'exceeds' for r in m_)} of {len(m_)}; "
      f"{max(-r['delta_pp'] - 1 for r in m_ if r['verdict'] == 'exceeds'):.2f}")
hd = D._load("holdout", "holdout.json")["msmarco"]["S1"]
check("MS MARCO median squared distance to the nearest neighbor, passages; questions", "0.35; 0.52",
      f"{hd['hold_d1_median']:.2f}; {hd['bench_d1_median']:.2f}")
summ = D._load("holdout", "corpus_summary.json")["cells"]
t2c = [c for c in summ if c["ds"] == "t2i"]
check("T2I held-out images: within the tolerance; share of the saving kept (%)", "yes; 8--61",
      f"{'yes' if all(c['corpus_verdict'] == 'within' for c in t2c) else 'no'}; "
      f"{rng([c['retained'] * 100 for c in t2c if c['retained'] is not None])}")
ci = D._load("holdout", "img", "cells_img.json")
cc_ = D._load("holdout", "cells", D.DSN("t2i"), "corpus", "calib_cells.json")
gd = [abs(statistics.mean(x or 0 for x in v["gamma_rank_shard"]) - statistics.mean(x or 0 for x in cc_[k]["gamma_rank_shard"]))
      for k, v in ci.items()]
check("T2I: mean gamma_i from held-out vectors vs from image queries, within", "0.005", f"{max(gd):.3f}")
kx = D._load("verdicts", "summary.json")["kaxis"]
dep_small = [r for r in kx if r["arm"] == "dep" and r["K"] in (5, 10)]
check("gamma_i of K=20 used at K=5 and 10: exceed in; losing up to (pp)", "all 4; 2.72",
      f"{'all ' + NUM.get(len(dep_small), str(len(dep_small))) if all(r['v'] == 'exceeds' for r in dep_small) else 'not all'}; "
      f"{max(-r['d'] for r in dep_small):.2f}")
ref = [r for r in kx if r["arm"] == "ref"]
check("calibrated at the K of the query, K in {5, 10, 20, 50, 100}: within in", "all 10",
      f"{'all ' + NUM.get(len(ref), str(len(ref))) if all(r['v'] == 'within' for r in ref) else 'not all'}")
dep_big = {(r["ds"], r["K"]): r for r in kx if r["arm"] == "dep" and r["K"] in (50, 100)}
ref_big = {(r["ds"], r["K"]): r for r in kx if r["arm"] == "ref" and r["K"] in (50, 100)}
check("gamma_i of K=20 at K=50 and 100: within; saves less than one calibrated for that K", "yes; yes",
      f"{'yes' if all(r['v'] == 'within' for r in dep_big.values()) else 'no'}; "
      f"{'yes' if all(dep_big[k]['ndis'] > ref_big[k]['ndis'] for k in dep_big) else 'no'}")
check("smallest rank tested: k_i at K=5, S=15", "3", str(L.k_i_of(5, 15)))

# ---------------------------------------------------------------- per-query dumps (not shipped) and numbers outside the results
T, M = T2I, MSM
if all(L.dump_path(D.DUMPS_DIR, "shard", ds, 15, 300, arm, blk) for ds in (T, M) for arm in ("fixed", "adaptive")
       for blk in "ab"):
    imb = {}
    for ds in (T, M):
        for arm in ("fixed", "adaptive"):
            p = D._shard_ndis(ds, 15, 300, arm)
            imb[ds, arm] = statistics.mean(max(v.values()) for v in p.values()) / \
                (statistics.mean(sum(v.values()) for v in p.values()) / 15)
    check("slowest shard / mean shard ndis, fixed -> adaptive, S=15 ef=300, T2I; MS MARCO",
          "1.10 -> 1.41; 1.06 -> 1.62",
          f"{imb[T, 'fixed']:.2f} -> {imb[T, 'adaptive']:.2f}; {imb[M, 'fixed']:.2f} -> {imb[M, 'adaptive']:.2f}")
else:
    UNVERIFIED.append(("slowest shard / mean shard ndis 1.10 -> 1.41 (T2I), 1.06 -> 1.62 (MS MARCO)",
                       f"needs the per-shard ndis dumps under DUMPS_DIR={D.DUMPS_DIR}"))
if all(L.dump_path(D.DUMPS_DIR, "cpu_shard", T, 15, 300, arm, blk, ext) for arm in ("fixed", "adaptive")
       for blk in "ab" for ext in ("tsv", "json")):
    q = {arm: [v[1] for v in D._per_query(T, 15, 300, arm, "cpu").values()] for arm in ("fixed", "adaptive")}
    check("T2I S=15 ef=300, slowest-shard CPU per query: median fixed / adaptive; 99th percentile (ms)",
          "0.87 / 0.60; 1.40 / 1.33", f"{np.median(q['fixed']):.2f} / {np.median(q['adaptive']):.2f}; "
          f"{np.percentile(q['fixed'], 99):.2f} / {np.percentile(q['adaptive'], 99):.2f}")
else:
    UNVERIFIED.append(("slowest-shard CPU 0.87 -> 0.60 ms (median), 1.40 -> 1.33 ms (99th percentile)",
                       f"needs the per-shard CPU dumps under DUMPS_DIR={D.DUMPS_DIR}"))
if all(L.dump_path(D.DUMPS_DIR, *s) for ds in (T, M) for arm in ("fixed", "adaptive") for s in D._hits_specs(ds, 15, 300, arm)):
    fewer = {}
    for ds in (T, M):
        F, A = D._hits(ds, 15, 300, "fixed"), D._hits(ds, 15, 300, "adaptive")
        dq = [F[k] - A[k] for k in F]
        fewer[ds] = (sum(x >= 1 for x in dq) / len(dq) * 100, sum(x >= 4 for x in dq) / len(dq) * 100)
    check("S=15 ef=300: queries with at least one fewer true neighbor, T2I / MS MARCO (%); four or more fewer",
          "11 / 11; < 1", f"{fewer[T][0]:.0f} / {fewer[M][0]:.0f}; "
          + meets(max(fewer[T][1], fewer[M][1]) < 1, "< 1", f"{max(fewer[T][1], fewer[M][1]):.1f}"))
else:
    UNVERIFIED.append(("11% of queries lose at least one true neighbor, < 1% four or more (S=15, ef=300)",
                       f"needs the per-query hit dumps under DUMPS_DIR={D.DUMPS_DIR}"))
# Bonferroni-corrected intervals, joint 95% over the configurations of the rule (lib/bonferroni.py). From the shipped
# results: the 95% intervals widened by the normal approximation; with dumps, lib/bonferroni.py bootstraps each
# configuration at the corrected level (both give these counts on the shipped results).
import bonferroni as BF                                   # noqa: E402  (lib/, on the path through data)
bc = BF.counts(BF.normal_rows(rule))
check("Bonferroni-corrected intervals over the configurations of the rule: configurations; undecided; exceed",
      "60; 4; 0", f"{len(rule)}; {bc['undecided']}; {bc['exceeds']}")
# the gamma calibrated on the unsharded index applied to every shard (arm s1g, experiments/run_extras.sh transfer)
s1 = {ds: [r for r in boot if r["arm"] == "s1g" and r["ds"] == ds and r["S"] >= 5] for ds in (T, M)}
check("gamma_i calibrated on a single index, used at S >= 5: configurations that exceed, T2I; MS MARCO",
      "12 of 12; 10 of 12", "; ".join(f"{sum(r['verdict'] == 'exceeds' for r in s1[ds])} of {len(s1[ds])}" for ds in (T, M)))

if __name__ == "__main__":
    w0 = max(len(r[0]) for r in rows)
    for what, ref, mine, ok in rows:
        print(f"{ok:4}  {what:<{w0}}  reference: {ref:<34} recomputed: {mine}")
    for what, why in UNVERIFIED:
        print(f"--    {what:<{w0}}  unverified: {why}")
    bad = [r for r in rows if r[3] != "OK"]
    print(f"\n{len(rows) - len(bad)} of {len(rows)} reference values reproduced, {len(UNVERIFIED)} not recomputable here")
    if bad and "--strict" in sys.argv:
        sys.exit(1)
