# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)

"""Same-recall matching: fixed ef and adaptive termination at a target recall R. This script finds the smallest
integer ef with recall >= R; figures/data.py interpolates the fixed-ef cost between ef - 1 and ef when that ef
overshoots R by more than 0.001.

  fixed     smallest integer ef with recall >= R on test block a (direct gamma = 0 runs). The gamma = 0 points of
            the grid cells bracket it; the interval is narrowed by direct runs.
  adaptive  for every grid E whose gamma = 0 end state reaches R, the smallest shard-common gamma with recall >= R
            (lib/gamma_search.py on that cell's traversal record); the (E, gamma) with the fewest ndis is kept and
            confirmed by a direct run, which must equal the replay to the printed digit.
  tab       fixed ef matched to the recall of the calibrated adaptive cell calib/S{S}_E{E}.json (S = 1, 5, 15): the
            deployed comparison, adaptive termination with each shard's gamma_i against fixed ef at the recall it reaches
            (run_core.sh: E in E_LIMITS on the four datasets up to 10M vectors, E = 200 on the 100M sets).

The matching at R = 0.90 is not read by any figure or check; its adaptive candidates set some of the E at which the
queue-saturation comparison (experiments/patience.py elist) records, so it is kept.

Outputs (adaptive termination, rank k_i):
  cells/<dsn>/match/R{R:.3f}_S{S}_{fixed,adaptive}.json, cells/<dsn>/match/tab_S{S}_E{E}_fixed.json
Direct runs are kept as cells/<dsn>/direct/S{S}_E{E}_g{gamma}.json (same names as cell.sh direct) and reused.

Paths come from the environment set by experiments/config.sh (run_core.sh exports them).
Usage: python3 match.py <ds> "<S values>" [--tab | --tab-only] [--tab-E "200"]
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
import shardlib as L                                      # noqa: E402
from gamma_search import search as gamma_search           # noqa: E402
from trace_replay import read_gt, read_trace              # noqa: E402

R_LIST = (0.90, 0.95)
TOL = 0.001

ap = argparse.ArgumentParser()
ap.add_argument("ds"); ap.add_argument("SS")
ap.add_argument("--tab", action="store_true"); ap.add_argument("--tab-only", action="store_true")
ap.add_argument("--tab-E", default="200", help='E values of the tab cells (default "200")')
args = ap.parse_args()
ds, SS = args.ds, [int(s) for s in args.SS.split()]
TAB = args.tab or args.tab_only
TAB_ONLY = args.tab_only                  # no R matching: only fixed ef matched to the calibrated cells' recall
TAB_E = [int(e) for e in args.tab_E.split()]
RANK_MODE = os.environ.get("RANK_MODE", "rule")
if RANK_MODE != "rule":
    sys.exit(f"!! matching is for adaptive termination only (RANK_MODE=rule, got {RANK_MODE})")


def env(name, default):
    return os.environ.get(name) or default


REPRO = env("REPRO", os.path.dirname(HERE))
WORK = env("WORK", f"{REPRO}/work")
RAW_DIR, IDX_DIR = env("RAW_DIR", f"{WORK}/raw"), env("IDX_DIR", f"{WORK}/idx")
TRACE_DIR, RESULTS = env("TRACE_DIR", f"{WORK}/trace"), env("RESULTS", f"{WORK}/results")
BIN = env("BIN", f"{REPRO}/_build/bin/bench")
K = int(env("K", str(L.K)))
DSN = L.dsn(ds)
NORM = L.norm_flag(ds)
BASE = f"{RESULTS}/cells/{DSN}"
OUT = BASE
TD = f"{TRACE_DIR}/{DSN}/{RANK_MODE}"
IDX = f"{IDX_DIR}/{DSN}"
GT = f"{IDX}/gt300.ivec"
os.makedirs(f"{BASE}/match", exist_ok=True); os.makedirs(f"{OUT}/match", exist_ok=True)
os.makedirs(f"{OUT}/direct", exist_ok=True)


def log(*a):
    print(*a, flush=True)


def ki(S):
    return L.k_i_of(K, S)


def direct(S, E, g="0"):
    """One direct run on test block a -> first row of the bench JSON. Reads the file if it exists.
    For gamma = 0 on a grid E, the recording run of the grid cell (same binary, same queries, gamma = 0) is used."""
    if g == "0":
        gf = f"{OUT}/grid/S{S}_E{E}.json"
        if os.path.exists(gf):
            return json.load(open(gf))["bench"]
    f = f"{OUT}/direct/S{S}_E{E}_g{g}.json"
    if not os.path.exists(f):
        if not os.access(BIN, os.X_OK):
            sys.exit(f"!! benchmark binary not found: {BIN}")
        cmd = [BIN, "search", "--shards", f"{IDX}/S{S}", "--S", str(S), "--query", f"{RAW_DIR}/{ds}/query.fbin",
               "--normalize", str(NORM), "--check_norm", str(NORM), "--K", str(K), "--ki", str(ki(S)),
               "--rule_k", str(L.rule_rank(RANK_MODE, K, S)), "--ef", str(E), "--nq", "1000", "--qoff", "0",
               "--gt", GT, "--gamma", g, "--repeat", "1", "--warmup", "100"]
        p = subprocess.run(cmd, capture_output=True, text=True)
        try:
            json.loads(p.stdout)
        except Exception:
            sys.exit(f"!! direct run failed S={S} E={E} gamma={g}: {p.stderr[-300:]}")
        open(f + ".part", "w").write(p.stdout); os.replace(f + ".part", f)
    return json.load(open(f))[0]


def grid(S):
    """grid cells {E: record meta + curve}"""
    d = {}
    for f in glob.glob(f"{OUT}/grid/S{S}_E*.json"):
        g = json.load(open(f))
        d[g["ef"]] = g
    return dict(sorted(d.items()))


def match_fixed(S, R, key):
    """Smallest integer ef with recall >= R. The grid's gamma = 0 points bracket it, direct runs split the interval."""
    f = f"{BASE}/match/{key}_fixed.json"
    if os.path.exists(f) or RANK_MODE != "rule":          # fixed ef is rank-independent: computed in rule mode only
        return json.load(open(f)) if os.path.exists(f) else None
    pts = {E: g["curve"][0]["recall"] for E, g in grid(S).items()}
    hi = min((E for E, r in pts.items() if r >= R), default=None)
    if hi is None:
        log(f"  fixed S={S} R={R:.4f}: not reached at the largest grid E - skipped"); return None
    lo = max((E for E, r in pts.items() if E < hi), default=ki(S) - 1)
    lo = max(lo, ki(S) - 1)                              # ef < k_i is never run
    # Two points per round, run in parallel: the ef estimated by linear interpolation of the interval ends' recall
    # and the one just below. Recall is smooth in ef, so one or two rounds usually suffice. The answer is the same as
    # bisection's (smallest integer ef with recall >= R); runs are deterministic, so the order does not matter.
    probes = {}
    r_lo = direct(S, lo)["recall"] if lo >= ki(S) else 0.0
    r_hi = direct(S, hi)["recall"]
    while hi - lo > 1:
        est = lo + (R - r_lo) / (r_hi - r_lo) * (hi - lo) if r_hi > r_lo else (lo + hi) / 2
        e1 = min(max(int(np.ceil(est)), lo + 1), hi - 1)
        cand = sorted({e1, max(e1 - 1, lo + 1)} if e1 - 1 > lo else {e1, min(e1 + 1, hi - 1)})
        with ThreadPoolExecutor(len(cand)) as ex:
            got = dict(zip(cand, ex.map(lambda e: direct(S, e)["recall"], cand)))
        probes.update(got)
        for e in sorted(got):
            if got[e] >= R:
                if e < hi:
                    hi, r_hi = e, got[e]
            elif lo < e < hi:
                lo, r_lo = e, got[e]
    b = direct(S, hi)
    row = {"S": S, "R": R, "ef": hi, "recall": b["recall"], "ndis": b["ndis_per_query"],
           "over": round(b["recall"] - R, 6), "probes": probes}
    if b["recall"] - R > TOL and lo >= ki(S):           # no integer ef within TOL: keep both neighbours
        bl = direct(S, lo)
        row["below"] = {"ef": lo, "recall": bl["recall"], "ndis": bl["ndis_per_query"]}
    json.dump(row, open(f, "w"), indent=1)
    return row


def match_adaptive(S, R, key):
    f = f"{OUT}/match/{key}_adaptive.json"
    if os.path.exists(f):
        return json.load(open(f))
    cands = []
    for E, g in grid(S).items():
        if g["curve"][0]["recall"] < R:
            continue
        t = f"{TD}/eval_S{S}_E{E}.bin"
        if not os.path.exists(t):
            sys.exit(f"!! traversal record {t} not found: remove {OUT}/grid/S{S}_E{E}.json and rerun "
                     f"RANK_MODE={RANK_MODE} cell.sh grid {ds} {S} {E}")
        meta, recs = read_trace(t)
        cands.append(gamma_search(meta, recs, read_gt(GT, meta["nq"]), R, K))
    ok = [c for c in cands if c.get("feasible")]
    if not ok:
        log(f"  adaptive S={S} R={R:.4f}: no grid E reaches R - skipped"); return None
    best = min(ok, key=lambda c: c["ndis"])
    b = direct(S, best["ef"], best["gamma"])
    # replay and direct run must agree to the printed digits of the bench JSON
    same = f"{best['recall']:.5f}" == f"{b['recall']:.5f}" and f"{best['ndis']:.1f}" == f"{b['ndis_per_query']:.1f}"
    if not same:
        sys.exit(f"!! replay != direct S={S} E={best['ef']} gamma={best['gamma']}: "
                 f"{best['recall']:.5f}/{best['ndis']:.1f} vs {b['recall']:.5f}/{b['ndis_per_query']:.1f}")
    row = {"S": S, "R": R, "E": best["ef"], "gamma": best["gamma"], "recall": b["recall"],
           "ndis": b["ndis_per_query"], "over": round(b["recall"] - R, 6),
           "candidates": [{k: c.get(k) for k in ("ef", "feasible", "gamma", "recall", "ndis", "fixed_end")} for c in cands]}
    json.dump(row, open(f, "w"), indent=1)
    return row


# S is the outer loop so that one S's index stays in the page cache across its cells
for S in SS:
    for R in (() if TAB_ONLY else R_LIST):
        key = f"R{R:.3f}_S{S}"
        fx = match_fixed(S, R, key); ad = match_adaptive(S, R, key)
        if fx and ad:
            log(f"  {RANK_MODE} {ds} R={R} S={S}: fixed ef={fx['ef']} {fx['recall']:.4f}/{fx['ndis']:.0f}  "
                f"adaptive E={ad['E']} gamma={ad['gamma']} {ad['recall']:.4f}/{ad['ndis']:.0f}  ratio {ad['ndis']/fx['ndis']:.3f}")
    if TAB and RANK_MODE == "rule" and S in (1, 5, 15):
        for E in TAB_E:
            cf = f"{OUT}/calib/S{S}_E{E}.json"
            if not os.path.exists(cf):
                sys.exit(f"!! {cf} not found (run cell.sh calib {ds} {S} {E} first)")
            c = json.load(open(cf))[0]
            fx = match_fixed(S, c["recall"], f"tab_S{S}_E{E}")
            if fx:
                log(f"  tab {ds} S={S} E={E}: adaptive {c['recall']:.4f}/{c['ndis_per_query']:.0f}  "
                    f"fixed ef={fx['ef']} {fx['recall']:.4f}/{fx['ndis']:.0f}  saving {1 - c['ndis_per_query']/fx['ndis']:.1%}")
log(f"[match {RANK_MODE} {ds} S={args.SS}] done")
