# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)

"""Comparison with queue-saturation early termination (Patience; Teofili and Lin, ECIR 2025), test block a (#0-999).

Patience runs per shard in the bounded-queue search (faiss/patches/0003-patience.patch), counting insertions into one
of two queues: ef = the E nearest points computed so far (the queue is the beam, k = E), ki = the shard's result heap
(k = k_i). Every number comes from one gamma = 0 record with insertion counts per (dataset, S, E)
(experiments/patience_cell.sh, replayed by lib/patience_replay.py), and every chosen setting is confirmed by a direct
run that must equal the replay to the printed digits.

  elist    the E of a (dataset, S): PATIENCE_E (default E_LIMITS) and every E that the same-recall matching of adaptive
           termination tried (cells/<dsn>/match/R{0.950,0.900}_S{S}_adaptive.json, candidates)
  gammas   the gammas that matching chose at one E (replayed by the cell to check the new record against it)
  check    the cells: gamma = 0 replay = the recording run = the fixed-ef end state of the adaptive matching's candidate
           at the same E (and the gamma = 0 point of cells/<dsn>/grid/ when present); gamma replays = the candidates' recall
           and ndis; on the p x s grid recall and ndis do not decrease with p or s and never exceed fixed ef
  verify   direct runs of one cell: six settings (both queues x defaults, a small p, a small s) equal their replay, and
           p = 1e9 equals fixed ef for both queues -> patience/<dsn>/verify/S{S}_E{E}.json
  match    same recall, R in {0.95, 0.90}, per queue: over the E of elist and p in {1, 2, ..., 120}, the smallest s on a
           %.6f grid with recall >= R; the (E, p, s) with the fewest distance computations, confirmed by a direct run.
           s = None: not even s = 1 reaches R, Patience has to be off (the matched fixed ef) -> match/R{R}_S{S}_patience_{q}.json
           (figures and checks read R = 0.95 only)
  default  Lucene's defaults s = 0.995, p = max(7, floor(0.3 k)) at every E of PATIENCE_E: recall and ndis (direct run);
           difference to fixed ef at E with the paired bootstrap of lib/bootstrap.py (judge, B = 10,000, the
           resamples cell_rng(seed, ds, S, E, 1000) of the block-a verdicts, `at1000`), tolerance 1pp; ndis against
           fixed ef at E and against fixed ef matched to the recall Patience reaches (smallest integer ef with recall >=
           it, direct runs; interpolated with the ef below when it overshoots by more than 0.001, as figures/data.py
           dataset_ratio does) -> default/S{S}_E{E}_{q}.json
  mech     where each rule stops at the same-recall settings (lib/patience_replay.py mech) -> mech/R{R}_S{S}_E{E}.json
  summary  the same-recall table (with fixed ef and adaptive termination matched in cells/<dsn>/match/) and the defaults
           table -> patience/summary.json, printed
Results go to $RESULTS/patience/<dsn>/; records to $TRACE_DIR/<dsn>/patience/. The comparison cells come from
$CELLS_ROOT (default $RESULTS/cells, experiments/run_core.sh).

Usage: python3 patience.py elist <ds> <S> | gammas <ds> <S> <E> | check <ds...> | verify <ds> <S> <E>
       python3 patience.py match|default|mech <ds> ["<S values>"] | summary ["<ds values>"]
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
import shardlib as L                                       # noqa: E402
from bootstrap import cell_rng, judge                      # noqa: E402
from patience_replay import QUEUES, default_p              # noqa: E402


def env(name, default):
    return os.environ.get(name) or default


REPRO = env("REPRO", os.path.dirname(HERE))
WORK = env("WORK", f"{REPRO}/work")
RAW_DIR, IDX_DIR = env("RAW_DIR", f"{WORK}/raw"), env("IDX_DIR", f"{WORK}/idx")
TRACE_DIR, RESULTS = env("TRACE_DIR", f"{WORK}/trace"), env("RESULTS", f"{WORK}/results")
CELLS_ROOT = env("CELLS_ROOT", f"{RESULTS}/cells")
BIN = env("BIN", f"{REPRO}/_build/bin/bench")
PY = env("PY", sys.executable)
K = L.K
E_DEFAULT = [int(x) for x in env("PATIENCE_E", env("E_LIMITS", " ".join(map(str, L.E_LIMITS)))).split()]
RS = (0.95, 0.90)
SEED, B, TOL = 20290917, 10000, 0.001
P_HUGE = "1000000000"


def log(*a):
    print(*a, flush=True)


def ki(S):
    return L.k_i_of(K, S)


def out_dir(ds, *parts):
    d = os.path.join(RESULTS, "patience", L.dsn(ds), *parts)
    os.makedirs(d, exist_ok=True)
    return d


def trace_dir(ds):
    return os.path.join(TRACE_DIR, L.dsn(ds), "patience")


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def adaptive_match(ds, S, R):
    return load(f"{CELLS_ROOT}/{L.dsn(ds)}/match/R{R:.3f}_S{S}_adaptive.json")


def fixed_match(ds, S, R):
    return load(f"{CELLS_ROOT}/{L.dsn(ds)}/match/R{R:.3f}_S{S}_fixed.json")


def s_values(ds):
    """S of a dataset; datasets other than the six (the smoke test's synth) take SYNTH_SS"""
    return list(L.S_VALUES.get(ds, ())) or [int(x) for x in env("SYNTH_SS", "1 5").split()]


def e_list(ds, S):
    es = set(E_DEFAULT)
    for R in RS:
        a = adaptive_match(ds, S, R)
        if a:
            es |= {c["ef"] for c in a["candidates"]}
    return sorted(es)


def gammas_at(ds, S, E):
    """{R: candidate} of the adaptive same-recall matching at this E"""
    out = {}
    for R in RS:
        a = adaptive_match(ds, S, R)
        for c in (a or {}).get("candidates", []):
            if c["ef"] == E and c.get("feasible"):
                out[R] = c
    return out


def cells(ds, S):
    d = {}
    for f in glob.glob(os.path.join(RESULTS, "patience", L.dsn(ds), "cell", f"S{S}_E*.json")):
        c = json.load(open(f))[0]
        d[c["ef"]] = c
    return dict(sorted(d.items()))


def direct(ds, S, E, tag, args):
    """One direct run on test block a (bench JSON, first repetition), kept as patience/<dsn>/direct/S{S}_E{E}_{tag}.json"""
    f = os.path.join(out_dir(ds, "direct"), f"S{S}_E{E}_{tag}.json")
    if not os.path.exists(f):
        if not os.access(BIN, os.X_OK):
            sys.exit(f"!! benchmark binary not found: {BIN}")
        idx = f"{IDX_DIR}/{L.dsn(ds)}"
        norm = str(L.norm_flag(ds))
        cmd = [BIN, "search", "--shards", f"{idx}/S{S}", "--S", str(S), "--query", f"{RAW_DIR}/{ds}/query.fbin",
               "--normalize", norm, "--check_norm", norm, "--K", str(K), "--ki", str(ki(S)), "--rule_k", str(ki(S)),
               "--ef", str(E), "--nq", "1000", "--qoff", "0", "--gt", f"{idx}/gt300.ivec", "--repeat", "1",
               "--warmup", "100"] + args
        p = subprocess.run(cmd, capture_output=True, text=True)
        try:
            json.loads(p.stdout)
        except Exception:
            sys.exit(f"!! direct run failed {ds} S={S} E={E} {tag}: {p.stderr[-300:]}")
        open(f + ".part", "w").write(p.stdout); os.replace(f + ".part", f)
    return json.load(open(f))[0]


def pat_args(q, p, s):
    return ["--gamma", "0", "--patience_p", str(p), "--patience_s", s, "--patience_queue", q]


def digits(recall, ndis):
    return f"{recall:.5f}", f"{ndis:.1f}"


def replay(ds, S, E, configs):
    t = os.path.join(trace_dir(ds), f"S{S}_E{E}.bin")
    p = subprocess.run([PY, os.path.join(REPRO, "lib", "patience_replay.py"), "eval", "--trace", t,
                        "--gt", f"{IDX_DIR}/{L.dsn(ds)}/gt300.ivec", "--K", str(K), "--configs", " ".join(configs)],
                       capture_output=True, text=True)
    if p.returncode:
        sys.exit(f"!! replay failed {ds} S={S} E={E}: {p.stderr[-300:]}")
    return json.loads(p.stdout)[0]["results"]


# ---------------------------------------------------------------- check (cells only, no search)
def cmd_check(dss):
    ok, bad = 0, []
    close = lambda a, b, tol=1e-9: abs(a - b) <= tol                       # noqa: E731
    for ds in dss:
        for S in s_values(ds):
            for E, c in cells(ds, S).items():
                tag = f"{ds} S={S} E={E}"
                fe = c["fixed_end"]
                if not c.get("bench", {}).get("same"):
                    bad.append(f"{tag}: gamma = 0 replay != recording run")
                gf = load(f"{CELLS_ROOT}/{L.dsn(ds)}/grid/S{S}_E{E}.json")
                if gf and not (close(fe["recall"], gf["curve"][0]["recall"]) and close(fe["ndis"], gf["curve"][0]["ndis"], 1e-6)):
                    bad.append(f"{tag}: fixed ef {fe['recall']:.5f}/{fe['ndis']:.1f} != grid {gf['curve'][0]['recall']:.5f}")
                g = {x["gamma"]: x for x in c["gamma"]}
                for R, a in gammas_at(ds, S, E).items():
                    x = g.get(a["gamma"])
                    if not (close(fe["recall"], a["fixed_end"]["recall"]) and close(fe["ndis"], a["fixed_end"]["ndis"], 1e-6)):
                        bad.append(f"{tag}: fixed ef differs from the adaptive matching's candidate (R={R})")
                    elif x is None or not (close(x["recall"], a["recall"]) and close(x["ndis"], a["ndis"], 1e-6)):
                        bad.append(f"{tag}: gamma {a['gamma']} replay differs from the adaptive matching (R={R})")
                    else:
                        ok += 1
                mono = {(x["queue"], x["p"], x["s"]): (x["recall"], x["ndis"]) for x in c["mono"]}
                for q in QUEUES:
                    ps = sorted({k[1] for k in mono if k[0] == q})
                    ss = sorted({k[2] for k in mono if k[0] == q}, key=float)
                    for i, p in enumerate(ps):
                        for j, s in enumerate(ss):
                            r, n = mono[q, p, s]
                            bad += [f"{tag} {q} p={p} s={s}: above fixed ef"] if r > fe["recall"] + 1e-12 or n > fe["ndis"] + 1e-6 else []
                            if i and (r + 1e-12 < mono[q, ps[i - 1], s][0] or n + 1e-6 < mono[q, ps[i - 1], s][1]):
                                bad.append(f"{tag} {q} s={s}: decreases from p={ps[i - 1]} to {p}")
                            if j and (r + 1e-12 < mono[q, p, ss[j - 1]][0] or n + 1e-6 < mono[q, p, ss[j - 1]][1]):
                                bad.append(f"{tag} {q} p={p}: decreases from s={ss[j - 1]} to {s}")
                ok += 1
    log(f"[patience check] {ok} items passed, {len(bad)} problems")
    for b in bad[:20]:
        log("  !!", b)
    return 1 if bad else 0


# ---------------------------------------------------------------- verify (direct runs of one cell)
def cmd_verify(ds, S, E):
    c = cells(ds, S).get(E)
    if c is None:
        sys.exit(f"!! no cell {ds} S={S} E={E} (run patience_cell.sh first)")
    fx = digits(c["bench"]["recall"], c["bench"]["ndis_per_query"])
    combos = []
    for q in QUEUES:
        p0 = default_p(E if q == "ef" else ki(S))
        combos += [(q, p0, "0.995"), (q, max(1, p0 // 10), "0.995"), (q, p0, "0.9")]
    rep = replay(ds, S, E, [f"{q}:{p}:{s}" for q, p, s in combos])
    rows, good = [], True
    for (q, p, s), x in zip(combos, rep):
        b = direct(ds, S, E, f"pat_{q}_p{p}_s{s}", pat_args(q, p, s))
        same = digits(x["recall"], x["ndis"]) == digits(b["recall"], b["ndis_per_query"])
        good &= same
        rows.append(dict(queue=q, p=p, s=s, replay=digits(x["recall"], x["ndis"]),
                         direct=digits(b["recall"], b["ndis_per_query"]), same=same))
    for q in QUEUES:
        b = direct(ds, S, E, f"pat_{q}_p{P_HUGE}", pat_args(q, P_HUGE, "0.995"))
        same = digits(b["recall"], b["ndis_per_query"]) == fx
        good &= same
        rows.append(dict(queue=q, p=int(P_HUGE), s="0.995", direct=digits(b["recall"], b["ndis_per_query"]),
                         fixed=fx, same=same))
    json.dump(dict(ds=ds, S=S, E=E, ok=good, rows=rows), open(os.path.join(out_dir(ds, "verify"), f"S{S}_E{E}.json"), "w"), indent=1)
    for r in rows:
        log(f"  verify {ds} S={S} E={E} {r['queue']} p={r['p']} s={r['s']}: "
            + (f"replay {'/'.join(r['replay'])} direct {'/'.join(r['direct'])}" if "replay" in r
               else f"direct {'/'.join(r['direct'])} fixed ef {'/'.join(r['fixed'])}") + ("" if r["same"] else "  !! differs"))
    return 0 if good else 1


# ---------------------------------------------------------------- stage 1: same recall
def cmd_match(ds, SS):
    for S in SS:
        cs = cells(ds, S)
        missing = [E for E in e_list(ds, S) if E not in cs]
        if missing:
            log(f"  {ds} S={S}: cells missing for E={missing} - skipped"); continue
        for R in RS:
            for q in QUEUES:
                f = os.path.join(out_dir(ds, "match"), f"R{R:.3f}_S{S}_patience_{q}.json")
                if os.path.exists(f):
                    continue
                cand = [dict(ef=E, p=x["p"], s=x["s"], recall=x["recall"], ndis=x["ndis"])
                        for E, c in cs.items() for x in c["match"]
                        if x["queue"] == q and abs(x["R"] - R) < 1e-12 and x.get("feasible")]
                if not cand:
                    log(f"  {ds} S={S} R={R} {q}: no E reaches R - skipped"); continue
                best = min(cand, key=lambda c: (c["ndis"], c["ef"], c["p"]))     # ties: smaller E, then smaller p
                if best["s"] is None:
                    tag, b = "g0", direct(ds, S, best["ef"], "g0", ["--gamma", "0"])
                else:
                    tag = f"pat_{q}_p{best['p']}_s{best['s']}"
                    b = direct(ds, S, best["ef"], tag, pat_args(q, best["p"], best["s"]))
                same = digits(best["recall"], best["ndis"]) == digits(b["recall"], b["ndis_per_query"])
                if not same:
                    sys.exit(f"!! replay != direct {ds} S={S} R={R} {q} E={best['ef']} p={best['p']} s={best['s']}: "
                             f"{best['recall']:.5f}/{best['ndis']:.1f} vs {b['recall']:.5f}/{b['ndis_per_query']:.1f}")
                row = dict(S=S, R=R, queue=q, E=best["ef"], p=best["p"], s=best["s"], recall=b["recall"],
                           ndis=b["ndis_per_query"], over=round(b["recall"] - R, 6),
                           direct=dict(recall=b["recall"], ndis_per_query=b["ndis_per_query"], same=same, tag=tag),
                           candidates=cand)
                json.dump(row, open(f, "w"), indent=1)
                log(f"  {ds} S={S} R={R} {q}: E={best['ef']} p={best['p']} s={best['s']} recall {b['recall']:.5f} "
                    f"ndis {b['ndis_per_query']:.1f} (direct = replay)")


# ---------------------------------------------------------------- stage 2: Lucene's defaults
def fixed_at(ds, S, ef, cs):
    """fixed ef (gamma = 0) on block a -> (recall, ndis, source). Reuses earlier runs of the same search."""
    f = os.path.join(RESULTS, "patience", L.dsn(ds), "direct", f"S{S}_E{ef}_g0.json")
    if os.path.exists(f):
        b = json.load(open(f))[0]; return b["recall"], b["ndis_per_query"], "direct"
    if ef in cs:
        b = cs[ef]["bench"]; return b["recall"], b["ndis_per_query"], "record"
    g = load(f"{CELLS_ROOT}/{L.dsn(ds)}/grid/S{S}_E{ef}.json")
    if g:
        b = g["bench"]; b = b[0] if isinstance(b, list) else b
        return b["recall"], b["ndis_per_query"], "record"
    f = f"{CELLS_ROOT}/{L.dsn(ds)}/direct/S{S}_E{ef}_g0.json"
    if os.path.exists(f):
        b = json.load(open(f))[0]; return b["recall"], b["ndis_per_query"], "direct"
    b = direct(ds, S, ef, "g0", ["--gamma", "0"])
    return b["recall"], b["ndis_per_query"], "direct"


def match_fixed(ds, S, target, cs):
    """smallest integer ef with recall >= target (bracketed by the cells, narrowed by direct runs)"""
    pts = {E: c["bench"]["recall"] for E, c in cs.items()}
    hi = min((E for E, r in pts.items() if r >= target), default=None)
    if hi is None:
        return None
    lo = max(max((E for E in pts if E < hi), default=ki(S) - 1), ki(S) - 1)
    probes = {}
    r_lo = fixed_at(ds, S, lo, cs)[0] if lo >= ki(S) else 0.0
    r_hi = fixed_at(ds, S, hi, cs)[0]
    while hi - lo > 1:
        est = lo + (target - r_lo) / (r_hi - r_lo) * (hi - lo) if r_hi > r_lo else (lo + hi) / 2
        e = min(max(int(np.ceil(est)), lo + 1), hi - 1)
        r = fixed_at(ds, S, e, cs)[0]; probes[e] = r
        if r >= target:
            hi, r_hi = e, r
        else:
            lo, r_lo = e, r
    rh, nh, src = fixed_at(ds, S, hi, cs)
    out = dict(ef=hi, recall=rh, ndis=nh, src=src, probes=probes)
    nd = nh
    if rh - target > TOL and lo >= ki(S):
        rl, nl, srl = fixed_at(ds, S, lo, cs)
        out["below"] = dict(ef=lo, recall=rl, ndis=nl, src=srl)
        nd = nl + (target - rl) / (rh - rl) * (nh - nl)
    out["ndis_at_target"] = nd
    return out


def cmd_default(ds, SS):
    nq = 1000
    for S in SS:
        cs = cells(ds, S)
        for E in E_DEFAULT:
            if E not in cs:
                log(f"  {ds} S={S} E={E}: no cell - skipped"); continue
            c = cs[E]
            hp = os.path.join(trace_dir(ds), f"hits_S{S}_E{E}.npz")
            if not os.path.exists(hp):
                sys.exit(f"!! {hp} not found (rerun patience_cell.sh {ds} {S} {E})")
            h = np.load(hp)
            for d in c["default"]:
                q = d["queue"]
                f = os.path.join(out_dir(ds, "default"), f"S{S}_E{E}_{q}.json")
                if os.path.exists(f):
                    continue
                b = direct(ds, S, E, f"pat_{q}_p{d['p']}_s{d['s']}", pat_args(q, d["p"], d["s"]))
                same = digits(d["recall"], d["ndis"]) == digits(b["recall"], b["ndis_per_query"])
                if not same:
                    sys.exit(f"!! replay != direct {ds} S={S} E={E} {q}: {d['recall']:.5f}/{d['ndis']:.1f} vs "
                             f"{b['recall']:.5f}/{b['ndis_per_query']:.1f}")
                h0, n0 = h["fixed_h"].astype(float), h["fixed_n"].astype(float)
                h1, n1 = h[f"{q}_h"].astype(float), h[f"{q}_n"].astype(float)
                if abs(h1.sum() / K / nq - d["recall"]) > 1e-9 or abs(n1.mean() - d["ndis"]) > 1e-6:
                    sys.exit(f"!! per-query hits disagree with the summary {ds} S={S} E={E} {q}")
                v = judge(h0, h1, n0, n1, cell_rng(SEED, ds, S, E, 1000), B, K, L.EPS_PP)
                fe = c["fixed_end"]
                mf = match_fixed(ds, S, d["recall"], cs)
                row = dict(ds=ds, S=S, E=E, queue=q, k=d["k"], p=d["p"], s=d["s"], recall=b["recall"],
                           ndis=b["ndis_per_query"], direct_same=same,
                           fixed_at_E=dict(recall=fe["recall"], ndis=fe["ndis"]),
                           ratio_same_ef=b["ndis_per_query"] / fe["ndis"], bootstrap=v, fixed_matched=mf,
                           ratio_same_recall=(b["ndis_per_query"] / mf["ndis_at_target"]) if mf else None)
                json.dump(row, open(f, "w"), indent=1)
                log(f"  {ds} S={S} E={E} {q}: p={d['p']} recall {b['recall']:.5f} (delta {v['delta_pp']:+.2f}pp "
                    f"[{v['delta_ci'][0]:+.2f}, {v['delta_ci'][1]:+.2f}] {v['verdict']}) same ef {row['ratio_same_ef']:.3f} "
                    f"same recall {'-' if mf is None else round(row['ratio_same_recall'], 3)}")


# ---------------------------------------------------------------- mechanism
def cmd_mech(ds, SS):
    for S in SS:
        for R in RS:
            cfg = {}
            a = adaptive_match(ds, S, R)
            if a:
                cfg.setdefault(a["E"], []).append(f"g:{a['gamma']}")
            for q in QUEUES:
                m = load(os.path.join(RESULTS, "patience", L.dsn(ds), "match", f"R{R:.3f}_S{S}_patience_{q}.json"))
                if m and m["s"] is not None:
                    cfg.setdefault(m["E"], []).append(f"{q}:{m['p']}:{m['s']}")
            for E, cl in sorted(cfg.items()):
                f = os.path.join(out_dir(ds, "mech"), f"R{R:.3f}_S{S}_E{E}.json")
                if os.path.exists(f):
                    continue
                t = os.path.join(trace_dir(ds), f"S{S}_E{E}.bin")
                p = subprocess.run([PY, os.path.join(REPRO, "lib", "patience_replay.py"), "mech", "--trace", t,
                                    "--gt", f"{IDX_DIR}/{L.dsn(ds)}/gt300.ivec", "--K", str(K), "--configs", " ".join(cl)],
                                   capture_output=True, text=True)
                if p.returncode:
                    sys.exit(f"!! mech replay failed {ds} S={S} E={E}: {p.stderr[-300:]}")
                open(f + ".part", "w").write(p.stdout); os.replace(f + ".part", f)
                log(f"  mech {ds} S={S} R={R} E={E}: {' '.join(cl)}")


# ---------------------------------------------------------------- summary
def same_recall_rows(ds):
    rows = []
    for S in s_values(ds):
        for R in RS:
            fx, ad = fixed_match(ds, S, R), adaptive_match(ds, S, R)
            pm = {q: load(os.path.join(RESULTS, "patience", L.dsn(ds), "match", f"R{R:.3f}_S{S}_patience_{q}.json"))
                  for q in QUEUES}
            if not fx or not ad or not all(pm.values()):
                continue
            eff = {q: fx["ndis"] if m["s"] is None else m["ndis"] for q, m in pm.items()}   # off = matched fixed ef
            row = dict(ds=ds, S=S, R=R, fixed=dict(ef=fx["ef"], ndis=fx["ndis"], recall=fx["recall"]),
                       adaptive=dict(E=ad["E"], gamma=ad["gamma"], ndis=ad["ndis"], recall=ad["recall"]),
                       patience={q: dict(E=m["E"], p=m["p"], s=m["s"], ndis=m["ndis"], recall=m["recall"],
                                         off=m["s"] is None) for q, m in pm.items()},
                       ratio={**{f"{q}_fixed": eff[q] / fx["ndis"] for q in QUEUES},
                              "adaptive_fixed": ad["ndis"] / fx["ndis"],
                              **{f"adaptive_{q}": ad["ndis"] / eff[q] for q in QUEUES},
                              "adaptive_better": ad["ndis"] / min(eff.values())})
            rows.append(row)
    return rows


def cmd_summary(dss):
    A = [r for ds in dss for r in same_recall_rows(ds)]
    Bt = []
    for ds in dss:
        for f in sorted(glob.glob(os.path.join(RESULTS, "patience", L.dsn(ds), "default", "S*_E*_*.json"))):
            d = json.load(open(f))
            Bt.append({k: d[k] for k in ("ds", "S", "E", "queue", "k", "p", "s", "recall", "ndis", "ratio_same_ef",
                                         "ratio_same_recall")} | dict(delta_pp=d["bootstrap"]["delta_pp"],
                                                                      delta_ci=d["bootstrap"]["delta_ci"],
                                                                      verdict=d["bootstrap"]["verdict"]))
    Bt.sort(key=lambda r: (r["ds"], r["S"], r["E"], r["queue"]))
    out = os.path.join(RESULTS, "patience", "summary.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(dict(same_recall=A, defaults=Bt), open(out, "w"), indent=1)
    log(f"{'dataset':<10}{'S':>3}{'R':>6}{'fixed':>9}{'adaptive':>10}{'P(ef)':>9}{'P(ki)':>9}"
        f"{'P(ef)/fix':>11}{'P(ki)/fix':>11}{'ad/better':>11}")
    for r in A:
        x = r["ratio"]
        pc = lambda q: "off" if r["patience"][q]["off"] else f"{r['patience'][q]['ndis']:.0f}"     # noqa: E731
        log(f"{r['ds']:<10}{r['S']:>3}{r['R']:>6.2f}{r['fixed']['ndis']:>9.0f}{r['adaptive']['ndis']:>10.0f}{pc('ef'):>9}"
            f"{pc('ki'):>9}{x['ef_fixed']:>11.3f}{x['ki_fixed']:>11.3f}{x['adaptive_better']:>11.3f}")
    log(f"\n{'dataset':<10}{'S':>3}{'E':>5}{'queue':>6}{'p':>4}{'recall':>8}{'Dpp':>7}{'95% CI':>17}  {'verdict':<10}"
        f"{'same ef':>8}{'same R':>8}")
    for r in Bt:
        band = "[%+.2f, %+.2f]" % tuple(r["delta_ci"])
        sr = "-" if r["ratio_same_recall"] is None else f"{r['ratio_same_recall']:.3f}"
        log(f"{r['ds']:<10}{r['S']:>3}{r['E']:>5}{r['queue']:>6}{r['p']:>4}{r['recall']:>8.4f}{r['delta_pp']:>7.2f}"
            f"{band:>17}  {r['verdict']:<10}{r['ratio_same_ef']:>8.3f}{sr:>8}")
    log(f"\n-> {out}")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd, args = sys.argv[1], sys.argv[2:]
    ss = lambda ds, i: [int(x) for x in args[i].split()] if len(args) > i else s_values(ds)   # noqa: E731
    if cmd == "elist":
        print(" ".join(map(str, e_list(args[0], int(args[1])))))
    elif cmd == "gammas":
        print(" ".join(sorted({c["gamma"] for c in gammas_at(args[0], int(args[1]), int(args[2])).values()})))
    elif cmd == "check":
        sys.exit(cmd_check(args))
    elif cmd == "verify":
        sys.exit(cmd_verify(args[0], int(args[1]), int(args[2])))
    elif cmd in ("match", "default", "mech"):
        dict(match=cmd_match, default=cmd_default, mech=cmd_mech)[cmd](args[0], ss(args[0], 1))
    elif cmd == "summary":
        cmd_summary(args[0].split() if args else ["t2i", "msmarco"])
    else:
        sys.exit(f"!! unknown command {cmd}\n{__doc__}")


if __name__ == "__main__":
    main()
