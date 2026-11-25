# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)

"""Read the result files under a RESULTS root for the figures and the checked numbers. No number is typed
in by hand.

RESULTS defaults to the shipped results ($REPRO/results); point it at $WORK/results to plot a re-run.
Every reader raises (FileNotFoundError / ValueError with the path) when an input is missing -- nothing is skipped silently.

Layout of RESULTS:
  cells/<dsn>/{grid,calib,match,cpu}/ + calib_cells.json   adaptive termination (rank k_i)
  cells/<dsn>/dabs/grid/                                   rank-K termination
  oracle/<dsn>/S{S}_ef200.json                             arrival-time oracle
  verdicts/bootstrap.json                                  paired-bootstrap tolerance verdicts (fig-verdicts)
  cells_K100/<dsn>/{,dabs/}grid/                           K=100 gamma grids, rule and rank K (run_extras.sh rank100)
  rank_k100/{summary.json,mech/<dsn>/}                     comparison rank at K=20 and 100 (experiments/rank_k100.py)
  patience/<dsn>/{match,default,mech}/ + summary.json      queue saturation (Patience) on test block a (experiments/patience.py)
  buildcost/<ds>_run*/{times.jsonl,json/}                  calibration-table cost
  approx/<dsn>/calib_cells_approx.json                     approximate-ground-truth calibration
  dumps/, shard/, cpu_shard/                               per-query dumps (not shipped; under DUMPS_DIR, default RESULTS)
                                                           -- fig-shard-tail and the share of shard searches
                                                           that ran to the ef limit
  derived/{shard_tail,efcap}.json                          those values, used when the dumps are absent
"""
import collections, csv, glob, json, os, re, statistics, sys

REPRO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPRO, "lib"))
import shardlib as L                                  # noqa: E402

RESULTS = os.path.abspath(os.environ.get("RESULTS", os.path.join(REPRO, "results")))
# Per-query dumps (dumps/, shard/, cpu_shard/) may live elsewhere (hundreds of MB, not shipped). Default: next to RESULTS.
DUMPS_DIR = os.path.abspath(L.dumps_dir(RESULTS))
TAIL_SOURCE = os.environ.get("TAIL_SOURCE", "auto")  # auto | raw | derived  (inputs of fig-shard-tail and the ef-limit share)
DSN = L.dsn
SS = (1, 5, 15, 30, 50)
QOFF_B = L.BLOCK_B_START
MATCH_TOL = 0.001       # a matched fixed-ef cell that overshoots R by more than this is interpolated with the ef below it


class MissingInput(FileNotFoundError):
    pass


def _path(*parts, root=None):
    p = os.path.join(root or RESULTS, *parts)
    if not os.path.exists(p):
        raise MissingInput(f"missing input: {p}")
    if _is_lfs_pointer(p):
        raise MissingInput(f"input is a Git LFS pointer stub, not data (fetch it first): {p}")
    return p


def _load(*parts, root=None):
    return json.load(open(_path(*parts, root=root)))


def cells(ds, *parts):
    return os.path.join("cells", DSN(ds), *parts)


def k_i(S, K=L.K):
    return L.k_i_of(K, S)


# ---------------------------------------------------------------- bench records
def _pick(rel):
    """A bench JSON is a list of repetitions; take the accepted one with the smallest CPU (interference only adds time)."""
    d = _load(rel)
    if isinstance(d, dict):
        return d
    ok = [r for r in d if r.get("accept")]
    if not ok:
        raise ValueError(f"no accepted repetition in {os.path.join(RESULTS, rel)}")
    return min(ok, key=lambda r: r["cpu_sum_ms"]["mean"])


def cpu_cell(ds, name):
    """cells/<dsn>/cpu/<name>.json -> (recall, ndis per query, CPU ms per query). name = S{S}_E{E}_g0 | _gcal | _g{gamma}"""
    r = _pick(cells(ds, "cpu", f"{name}.json"))
    return r["recall"], r["ndis_per_query"], r["cpu_sum_ms"]["mean"]


def _fixed_matched(ds, m):
    """Matched fixed-ef cell -> (recall, ndis, cpu, ef). If the integer ef overshoots R by more than MATCH_TOL,
    interpolate at R between it and the ef below."""
    hi = cpu_cell(ds, f"S{m['S']}_E{m['ef']}_g0")
    if m["recall"] - m["R"] <= MATCH_TOL or "below" not in m:
        return hi[0], hi[1], hi[2], m["ef"]
    lo = cpu_cell(ds, f"S{m['S']}_E{m['below']['ef']}_g0")
    t = (m["R"] - lo[0]) / (hi[0] - lo[0])
    return (m["R"], lo[1] + t * (hi[1] - lo[1]), lo[2] + t * (hi[2] - lo[2]),
            m["below"]["ef"] + t * (m["ef"] - m["below"]["ef"]))


def matched(ds, S, R, arm):
    """Cell matched to recall R -> (recall, ndis, cpu_ms, knob). knob = fixed ef, or (E, gamma) for adaptive."""
    m = _load(cells(ds, "match", f"R{R:.3f}_S{S}_{arm}.json"))
    if arm == "fixed":
        return _fixed_matched(ds, m)
    c = cpu_cell(ds, f"S{S}_E{m['E']}_g{m['gamma']}")
    return c[0], c[1], c[2], (m["E"], float(m["gamma"]))


def same_ef(ds, S, arm, ef=200):
    """Cell at ef -> (recall, ndis, cpu_ms). Fixed = gamma 0, adaptive = the calibrated per-shard gammas."""
    return cpu_cell(ds, f"S{S}_E{ef}_g{'0' if arm == 'fixed' else 'cal'}")


# ---------------------------------------------------------------- fig-overhead-same-recall, fig-adaptive-savings-both
def overhead(ds, R, arm, ef=200):
    """Sharding overhead relative to fixed ef at S=1 (same ef, or matched to the same R).
    {"ef": {S: (ndis ratio, CPU ratio, recall)}, "R": {S: (ndis ratio, CPU ratio, knob)}}"""
    b_ef, b_r = same_ef(ds, 1, "fixed", ef), matched(ds, 1, R, "fixed")
    out = {"ef": {}, "R": {}}
    for S in SS:
        v = same_ef(ds, S, arm, ef)
        out["ef"][S] = (v[1] / b_ef[1], v[2] / b_ef[2], v[0])
        m = matched(ds, S, R, arm)
        out["R"][S] = (m[1] / b_r[1], m[2] / b_r[2], m[3])
    return out


def savings(ds, R, ef=200):
    """Adaptive / fixed at the same S: {"ef": {S: (ndis, CPU)}, "R": {S: (ndis, CPU)}}"""
    out = {"ef": {}, "R": {}}
    for S in SS:
        a, f = same_ef(ds, S, "adaptive", ef), same_ef(ds, S, "fixed", ef)
        out["ef"][S] = (a[1] / f[1], a[2] / f[2])
        a, f = matched(ds, S, R, "adaptive"), matched(ds, S, R, "fixed")
        out["R"][S] = (a[1] / f[1], a[2] / f[2])
    return out


# ---------------------------------------------------------------- fig-waste
def matched_ef(ds, S, R=0.95):
    """the integer ef at which fixed ef reaches recall R (cells/<dsn>/match/R{R}_S{S}_fixed.json, no interpolation)"""
    return _load(cells(ds, "match", f"R{R:.3f}_S{S}_fixed.json"))["ef"]


def waste(S, ds="t2i", ef="matched"):
    """Arrival-time oracle cell. Default: fixed ef matched to a recall of 0.95 (fig-waste);
    ef=200: the oracle at the ef limit 200."""
    e = matched_ef(ds, S) if ef == "matched" else ef
    d = _load("oracle", DSN(ds), f"S{S}_ef{e}.json")
    ws = d["waste_split"]
    return {"total": d["ref_ndis_per_query"], "oracle": d["oracle_ndis_per_query"],
            "empty": ws["empty"], "confirm": ws["confirm"], "recall": d["recall"], "ef": e}


# ---------------------------------------------------------------- deployed saving
def dataset_ratio(ds, S, ef=200):
    """Adaptive (calibrated gammas, ef limit ef) ndis / ndis of fixed ef matched to that recall (match/tab_*).
    If the integer ef overshoots the recall by more than MATCH_TOL, interpolate with the ef below."""
    m = _load(cells(ds, "match", f"tab_S{S}_E{ef}_fixed.json"))
    c = _load(cells(ds, "calib", f"S{S}_E{ef}.json"))[0]
    nd = m["ndis"]
    if m["recall"] - m["R"] > MATCH_TOL and "below" in m:
        b = m["below"]
        t = (m["R"] - b["recall"]) / (m["recall"] - b["recall"])
        nd = b["ndis"] + t * (m["ndis"] - b["ndis"])
    return c["ndis_per_query"] / nd


# ---------------------------------------------------------------- per-query dumps (fig-shard-tail, ef-limit share)
# A dump is named by a spec (kind, ds, S, ef, arm, blk, ext); shardlib.dump_path resolves it under DUMPS_DIR in either
# layout -- the package's own runs (dumps/, shard/, cpu_shard/ with arm name rule) or the published export
# (per_query/, per_shard_ndis/, per_shard_cpu/ with the original arm names) -- and returns None when the file is absent
# or a Git LFS pointer stub. kind per_query: arm nat (fixed ef) | rule (adaptive; at S=1 a dabs dump stands in, the
# rule and rank-K termination being the same search there, rank K = k_i). kind shard | cpu_shard: arm fixed | adaptive.
def _spec_name(spec):
    kind, ds, S, ef, arm, blk, ext = spec
    sub = {"per_query": "dumps", "shard": f"shard/{DSN(ds)}", "cpu_shard": f"cpu_shard/{DSN(ds)}"}[kind]
    stem = f"{L.ds_short(DSN(ds))}_S{S}_ef{ef}_{arm}" if kind == "per_query" else f"S{S}_ef{ef}_{arm}"
    return f"{sub}/{stem}_{blk}.{ext}"


def _dump(spec):
    p = L.dump_path(DUMPS_DIR, *spec)
    if p is None:
        raise MissingInput(f"per-query dump not found (or a Git LFS pointer stub) under DUMPS_DIR={DUMPS_DIR}: "
                           f"{_spec_name(spec)}")
    return p


def _hits_specs(ds, S, ef, arm):
    return [("per_query", ds, S, ef, "nat" if arm == "fixed" else "rule", blk, "tsv") for blk in "ab"]


def _shard_specs(ds, S, ef, arm, kind):
    return [(kind, ds, S, ef, arm, blk, ext) for blk in "ab" for ext in (("tsv",) if kind == "shard" else ("tsv", "json"))]


def _hits(ds, S, ef, arm):
    h = {}
    for spec, off in zip(_hits_specs(ds, S, ef, arm), (0, QOFF_B)):
        with open(_dump(spec)) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                h[int(r["q"]) + off] = int(r["hits"])
    return h


def _shard_ndis(ds, S, ef, arm):
    """query (absolute number) -> {shard: ndis}, blocks a and b"""
    p = collections.defaultdict(dict)
    for blk, off in (("a", 0), ("b", QOFF_B)):
        with open(_dump(("shard", ds, S, ef, arm, blk, "tsv"))) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                p[int(r["q"]) + off][int(r["shard"])] = int(r["ndis"])
    return p


def _per_query(ds, S, ef, arm, metric):
    """query -> (sum over shards, slowest shard). ndis from the shard dumps, cpu from the cpu_shard dumps: per query,
    the accepted repetition with the smallest CPU sum (picking one whole repetition lets per-query noise into the tail)."""
    if metric == "ndis":
        return {q: (sum(d.values()), max(d.values())) for q, d in _shard_ndis(ds, S, ef, arm).items()}
    per = {}
    for blk, off in (("a", 0), ("b", QOFF_B)):
        ok = {r["rep"] for r in json.load(open(_dump(("cpu_shard", ds, S, ef, arm, blk, "json")))) if r.get("accept")}
        reps = collections.defaultdict(lambda: collections.defaultdict(dict))
        with open(_dump(("cpu_shard", ds, S, ef, arm, blk, "tsv"))) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if int(r["rep"]) in ok:
                    reps[int(r["q"]) + off][int(r["rep"])][int(r["shard"])] = float(r["cpu_ms"])
        for q, rr in reps.items():
            d = min(rr.values(), key=lambda d: sum(d.values()))
            per[q] = (sum(d.values()), max(d.values()))
    return per


def _is_lfs_pointer(p):
    return L.is_lfs_pointer(p)


def _source(name, specs):
    """raw if every per-query dump resolves under DUMPS_DIR, else RESULTS/derived/<name>.json.
    TAIL_SOURCE=raw|derived forces one."""
    miss = [s for s in specs if L.dump_path(DUMPS_DIR, *s) is None]
    if TAIL_SOURCE == "raw" or (TAIL_SOURCE == "auto" and not miss):
        if miss:
            raise MissingInput(f"TAIL_SOURCE=raw but {len(miss)} per-query dump(s) are missing under "
                               f"DUMPS_DIR={DUMPS_DIR}, e.g. {_spec_name(miss[0])}")
        print(f"[{name}] source: raw per-query dumps under DUMPS_DIR={DUMPS_DIR}")
        return "raw", None
    if TAIL_SOURCE not in ("auto", "derived"):
        raise ValueError(f"TAIL_SOURCE must be auto, raw or derived, not {TAIL_SOURCE!r}")
    d = _load("derived", f"{name}.json")
    why = "forced by TAIL_SOURCE=derived" if TAIL_SOURCE == "derived" else \
        f"{len(miss)} of {len(specs)} per-query dumps absent or Git LFS pointer stubs under DUMPS_DIR={DUMPS_DIR}"
    print(f"[{name}] source: {os.path.join(RESULTS, 'derived', name + '.json')} ({why}; values rounded to "
          f"{d['rounding_pct'] * 2:g} pp)")
    return "derived", d


EFCAP_CELLS = [(S, ef) for S in (1, 5, 15) for ef in (200, 300)]
TAIL_CELLS = [(S, ef) for S in (5, 15) for ef in (100, 200, 300)]
MAIN_DS = ("t2i", "msmarco")


def efcap_reader():
    """-> function (ds, S, ef) -> {group: % of shard searches that ran to the ef limit}"""
    specs = [s for ds in MAIN_DS for S, ef in EFCAP_CELLS
             for s in _hits_specs(ds, S, ef, "fixed")
             + _shard_specs(ds, S, ef, "fixed", "shard") + _shard_specs(ds, S, ef, "adaptive", "shard")]
    kind, d = _source("efcap", specs)
    if kind == "raw":
        return cap_share
    tab = {(c["dataset"], c["S"], c["ef"]): c["ef_limit_share_pct"] for c in d["cells"]}

    def get(ds, S, ef):
        k = (DSN(ds), S, ef)
        if k not in tab:
            raise MissingInput(f"derived/efcap.json has no cell {k}")
        return tab[k]
    return get


def cap_share(ds, S, ef, K=L.K, low_max=14):
    """Per query group (by the query's recall under fixed ef: perfect = 1.0, mid, low <= 0.7), the share of shard
    searches that ran to the ef limit. A shard where the rule did not fire follows the fixed-ef trajectory exactly,
    so its ndis is identical."""
    F = _hits(ds, S, ef, "fixed")
    NF, NA = _shard_ndis(ds, S, ef, "fixed"), _shard_ndis(ds, S, ef, "adaptive")
    grp = {q: "perfect" if F[q] == K else ("low" if F[q] <= low_max else "mid") for q in F}
    out = {}
    for g in ("perfect", "mid", "low"):
        qs = [q for q in F if grp[q] == g]
        tot = sum(len(NF[q]) for q in qs)
        out[g] = sum(1 for q in qs for s in NF[q] if NA[q][s] == NF[q][s]) / tot * 100
    return out


def shard_tail_reader():
    """-> function (ds, S, ef, metric) -> {(agg, stat): (saving %, 95% CI low, high[, bar label])}. The derived file
    carries the integer bar label of the published figure: a 1-decimal value ending in .5 does not determine it."""
    specs = [s for ds in MAIN_DS for S, ef in TAIL_CELLS for arm in ("fixed", "adaptive")
             for s in _shard_specs(ds, S, ef, arm, "shard") + _shard_specs(ds, S, ef, arm, "cpu_shard")]
    kind, d = _source("shard_tail", specs)
    if kind == "raw":
        return shard_tail_ci
    tab = {(c["dataset"], c["metric"], c["S"], c["ef"]): (c["saved_pct"], c["label"]) for c in d["cells"]}

    def get(ds, S, ef, metric):
        k = (DSN(ds), metric, S, ef)
        if k not in tab:
            raise MissingInput(f"derived/shard_tail.json has no cell {k}")
        v, lab = tab[k]
        return {(agg, stat): tuple(v[f"{agg}_{stat}"]) + (lab[f"{agg}_{stat}"],)
                for agg in ("total", "slowest") for stat in ("mean", "worst1")}
    return get


def shard_tail_ci(ds, S, ef, metric, B=2000, seed=20290917):
    """One fig-shard-tail cell -- {(agg, stat): (saving %, 95% CI low, high)}. agg: total (sum over shards) or slowest
    (slowest shard); stat: mean over queries or worst1 (mean of the heaviest 1% of queries of each search).
    Paired bootstrap: a resampled query brings both searches' values."""
    import numpy as np
    F, A = _per_query(ds, S, ef, "fixed", metric), _per_query(ds, S, ef, "adaptive", metric)
    qs = sorted(set(F) & set(A))
    f = np.array([F[q] for q in qs])
    a = np.array([A[q] for q in qs])
    n = len(qs)
    k = max(1, n // 100)
    rng = np.random.default_rng(seed + S * 1000 + ef + (0 if metric == "ndis" else 7))
    idx = rng.integers(0, n, size=(B, n))
    out = {}
    for j, agg in enumerate(("total", "slowest")):
        for stat in ("mean", "worst1"):
            def val(x):
                return x.mean(-1) if stat == "mean" else np.sort(x, -1)[..., -k:].mean(-1)
            pt = (1 - val(a[:, j]) / val(f[:, j])) * 100
            bs = (1 - val(a[idx, j]) / val(f[idx, j])) * 100
            out[agg, stat] = (float(pt), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    return out


# ---------------------------------------------------------------- fig-verdicts
def verdict_rows(arm="rule"):
    """Rows of verdicts/bootstrap.json for one arm: ds, S, ef, delta_pp (merged recall@20 of the arm minus fixed ef, pp),
    delta_ci (paired-bootstrap 95% interval), verdict (within | undecided | exceeds)"""
    rows = [r for r in _load("verdicts", "bootstrap.json") if r["arm"] == arm]
    if not rows:
        raise ValueError(f"no arm {arm!r} in {os.path.join(RESULTS, 'verdicts', 'bootstrap.json')}")
    return rows


# ---------------------------------------------------------------- comparison rank
def _ndis_at(curve, t):
    """ndis at recall t on a (gamma, recall, ndis) curve, linear in recall; None outside the curve."""
    pts = sorted((r, n) for _, r, n in curve)
    if t <= pts[0][0] or t >= pts[-1][0]:
        return None
    for (r0, n0), (r1, n1) in zip(pts, pts[1:]):
        if r0 <= t <= r1:
            return n0 if r1 == r0 else n0 + (n1 - n0) * (t - r0) / (r1 - r0)
    return None


def grid_curve(ds, S, E, mode="rule"):
    sub = ("grid",) if mode == "rule" else ("dabs", "grid")
    return _load(cells(ds, *sub, f"S{S}_E{E}.json"))["curve"]


def dabs_cost(ds, S, drop=1.0, ES=L.E_LIMITS):
    """ndis of rank-K termination / ndis of adaptive termination at the recall `drop` pp below fixed ef (gamma 0) along each rule's gamma
    grid, averaged over the ef limits. An ef limit where the target lies outside a curve is left out (printed)."""
    rs = []
    for E in ES:
        o, d = grid_curve(ds, S, E, "rule"), grid_curve(ds, S, E, "dabs")
        t = o[0]["recall"] - drop / 100.0
        a = _ndis_at([(p["gamma"], p["recall"], p["ndis"]) for p in o], t)
        b = _ndis_at([(p["gamma"], p["recall"], p["ndis"]) for p in d], t)
        if a and b:
            rs.append(b / a)
        else:
            print(f"  note: {DSN(ds)} S={S} E={E}: target recall outside a gamma curve, left out of the mean")
    if not rs:
        raise ValueError(f"no usable ef limit for rank-K termination / rule at {DSN(ds)} S={S}")
    return statistics.mean(rs)


# ---------------------------------------------------------------- CPU vs distance computations, same S and ef
def cpu_ndis(ds, efs):
    """S -> [(ef, ndis ratio, CPU ratio, recall fixed, recall adaptive)] -- adaptive (calibrated gammas) / fixed, same ef"""
    out = {}
    for S in (1, 5, 15):
        rows = []
        for E in efs:
            f, a = cpu_cell(ds, f"S{S}_E{E}_g0"), cpu_cell(ds, f"S{S}_E{E}_gcal")
            rows.append((E, a[1] / f[1], a[2] / f[2], f[0], a[0]))
        out[S] = rows
    return out


# ---------------------------------------------------------------- cost of the calibration table
def buildcost(ds, SS_=SS, ES=L.E_LIMITS, NQ=500):
    """Cost of building the calibration table relative to the index build (CPU of all shard graphs).
      (a) build: sum over shards of build CPU s (smallest over repetitions)   (b) exact ground truth of the calibration
      queries (once per dataset)   (c) approximate ground truth: one search at ef_gt = min(1000, 4E) per E
      (e) one recorded traversal + gamma replay per E.   exact = (b + sum e) / a,  approx = (sum c + sum e) / a.
    Wall-clock (exact only): build 28 threads, ground truth 16 threads (OpenMP and OpenBLAS each set to 16),
    search/record/replay 1 thread.
    same = number of shard calibrations (S shards x len(ES) ef limits, n_cal) in which the approximate ground truth
    selects the same gamma as the exact one (None where the approximate calibration was not run). Several measurement runs of one dataset are pooled."""
    short = L.ds_short(DSN(ds))
    dirs = sorted(glob.glob(os.path.join(RESULTS, "buildcost", f"{short}_*")))
    if not dirs:
        raise MissingInput(f"missing input: {os.path.join(RESULTS, 'buildcost', short + '_*')}")
    rows, J = [], {}
    for x in dirs:
        rows += [json.loads(l) for l in open(_path(os.path.relpath(os.path.join(x, "times.jsonl"), RESULTS)))]
        for f in glob.glob(os.path.join(x, "json", "*.json")):
            J[os.path.basename(f)[:-5]] = f
    T = {r["name"]: r for r in rows if r["rc"] == 0 and r["throttled"] == 0}

    def bench(name):
        return json.load(open(J[name]))[0] if name in J else None

    def pick(vals, what):
        if not vals:
            raise MissingInput(f"buildcost {short}: no record for {what}")
        good = [v for v, a in vals if a]
        return min(good) if good else min(v for v, _ in vals)

    gts = [T[n] for n in T if re.fullmatch(rf"gt_{short}_r\d+", n)]
    if not gts:
        raise MissingInput(f"buildcost {short}: no exact ground-truth timing (gt_{short}_r*)")
    b_cpu, b_wall = min(g["cpu_s"] for g in gts), min(g["wall_s"] for g in gts)
    ap = _load("approx", DSN(ds), "calib_cells_approx.json")
    ex = _load(cells(ds, "calib_cells.json"))
    out = {}
    for S in SS_:
        reps = collections.defaultdict(list)
        for nm, r in T.items():
            m = re.fullmatch(rf"build_{short}_S{S}_r(\d+)_s(\d+)", nm)
            if m:
                reps[int(m.group(1))].append(r)
        full = [v for v in reps.values() if len(v) == S]
        if not full:
            raise MissingInput(f"buildcost {short}: no complete build repetition at S={S}")
        a_cpu = min(sum(x["cpu_s"] for x in v) for v in full)
        a_wall = min(sum(x["wall_s"] for x in v) for v in full)
        c_sum = e_sum = e_wall = 0.0
        for E in ES:
            c_sum += pick([(j["cgroup_cpu_ms_total"] / 1000, j["accept"]) for r in range(1, 6)
                           if (j := bench(f"approx_{short}_S{S}_E{E}_r{r}"))], f"approx S={S} E={E}")
            ev, ew = [], []
            for r in range(1, 4):
                j, rp = bench(f"trace_{short}_S{S}_E{E}_r{r}"), T.get(f"replay_{short}_S{S}_E{E}_r{r}")
                if j and rp:
                    ev.append((j["cgroup_cpu_ms_total"] / 1000 + rp["cpu_s"], j["accept"]))
                    ew.append(j["wall_sum_ms"]["mean"] * NQ / 1000 + rp["wall_s"])
            e_sum += pick(ev, f"trace+replay S={S} E={E}")
            e_wall += min(ew)
        keys = [f"S{S}_E{E}" for E in ES if f"S{S}_E{E}" in ap]
        pairs = [(p, q) for k in keys for p, q in zip(ap[k]["gamma_rank_shard"], ex[k]["gamma_rank_shard"])]
        same = sum(p == q for p, q in pairs) if len(keys) == len(ES) else None
        out[S] = dict(build_cpu=a_cpu, build_wall=a_wall, exact_pct=100 * (b_cpu + e_sum) / a_cpu,
                      exact_wall_pct=100 * (b_wall + e_wall) / a_wall, approx_pct=100 * (c_sum + e_sum) / a_cpu,
                      exact_share=100 * b_cpu / (b_cpu + e_sum), same=same, n_cal=S * len(ES),
                      gt_cpu=b_cpu, gt_wall=b_wall, record_replay_cpu=e_sum)
    return out


# ---------------------------------------------------------------- queue saturation (Patience), test block a
def patience_same_recall(ds, S, R):
    """Same recall R on test block a -> {"fixed", "adaptive", "ef", "ki"}: ndis per query of fixed ef and adaptive termination
    matched in cells/<dsn>/match/, and of Patience counting the E nearest points (ef) or the k_i results (ki), matched in
    patience/<dsn>/match/. A Patience setting that cannot reach R (s None) is off and costs the matched fixed ef."""
    fx = _load(cells(ds, "match", f"R{R:.3f}_S{S}_fixed.json"))["ndis"]
    out = {"fixed": fx, "adaptive": _load(cells(ds, "match", f"R{R:.3f}_S{S}_adaptive.json"))["ndis"]}
    for q in ("ef", "ki"):
        m = _load("patience", DSN(ds), "match", f"R{R:.3f}_S{S}_patience_{q}.json")
        out[q] = fx if m["s"] is None else m["ndis"]
    return out


def patience_default(ds, S, E, q="ef"):
    """Lucene's defaults at ef limit E (patience/<dsn>/default/S{S}_E{E}_{q}.json): recall, bootstrap verdict against fixed ef
    at E (test block a), ndis ratios at the same ef and at the same recall"""
    return _load("patience", DSN(ds), "default", f"S{S}_E{E}_{q}.json")
