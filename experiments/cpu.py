# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)

"""CPU-time cells of one dataset, measured in one session: fig-overhead-same-recall, fig-adaptive-savings-both,
and the comparison of the CPU ratio with the distance-computation ratio at the same S and ef (T2I).

Measurement: one search thread, prefixed with $PIN (e.g. PIN="numactl --cpunodebind=1 --membind=1" to keep
the run on one socket; the original runs used one socket of an AMD EPYC 9354), --nq 1000 --repeat 3 --vt_threshold 999999999 (the
visited set is always the dense array, so S=1 and S>1 run the same search code). Each repetition carries the bench's own
gate verdict ("accept"): it is rejected if the noise-floor kernel drifted by 10% or more across the repetition, or if the
cgroup v2 throttle counter (/sys/fs/cgroup/cpu.stat nr_throttled) grew. On a machine without a CPU quota the throttle gate
is a no-op (the counter is absent or never grows). The analysis takes the smallest CPU time among the accepted repetitions.
Cells run one at a time; both searches of every ratio come from this one session.

Cells (inputs from $CELLS_ROOT/<dsn>/, the outputs of experiments/run_core.sh; default CELLS_ROOT=$RESULTS/cells):
  match    for each target recall R and each S: fixed ef matched to R (match/R*_S*_fixed.json, plus the ef just below it when
           no integer ef lands on R) and adaptive (E, gamma) matched to R (match/R*_S*_adaptive.json)
  ef200    every S: fixed ef=200 and adaptive E=200 with the calibrated per-shard gammas
  cpundis  T2I S 1,5,15 x E 100,300: fixed and calibrated adaptive (CPU ratio against ndis ratio; E=200 comes from ef200)
  tab      MS MARCO S 1,5,15 x E 100,200,300: calibrated adaptive and the fixed ef matched to its recall (match/tab_*)
  recheck  S=1 fixed ef=200 once more at the end of the session (stability check; compare with S1_E200_g0)
Every cell must reproduce the recall and ndis of the earlier non-timed run of the same configuration; otherwise the
script stops (the search is deterministic, so a difference means a different index, binary or input).

Output: $RESULTS/cells/<dsn>/cpu/<name>.json (+ .err), names S{S}_E{E}_g0 | S{S}_E{E}_gcal | S{S}_E{E}_g{gamma} |
recheck_S1_E200_g0. Existing files are skipped.

Usage: python3 experiments/cpu.py <t2i|msmarco> [--list]
"""
import glob, json, os, shlex, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))


def load_config():
    """The settings of experiments/config.sh (environment overrides included)."""
    names = "REPRO RAW_DIR IDX_DIR RESULTS BIN PIN K".split()
    script = f'. "{HERE}/config.sh"; ' + "; ".join(f'printf "%s\\0" "${{{n}}}"' for n in names)
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout.split("\0")
    return dict(zip(names, out))


CFG = load_config()
sys.path.insert(0, os.path.join(CFG["REPRO"], "lib"))
import shardlib as L  # noqa: E402

K = int(CFG["K"])
SS = (1, 5, 15, 30, 50)
EXPECT_CPU = os.environ.get("EXPECT_CPU", "")          # e.g. "EPYC 9354": the bench refuses to run on another CPU model


def gamma_hat(cells_dir, S, E):
    c = json.load(open(f"{cells_dir}/calib_cells.json"))[f"S{S}_E{E}"]
    gl = ",".join("0" if g is None else f"{g:g}" for g in c["gamma_rank_shard"])
    return gl.split(",")[0] if S == 1 else gl


def cells(ds):
    """[(S, E, gamma string, name, expected (recall, ndis) or None)] in measurement order."""
    D = f"{CELLS_ROOT}/{L.dsn(ds)}"
    out, seen = [], set()

    def add(S, E, g, name, expect=None):
        if (S, E, g) in seen:
            return
        seen.add((S, E, g))
        out.append((S, E, g, name, expect))

    def gcal(S, E):
        c = json.load(open(f"{D}/calib/S{S}_E{E}.json"))[0]
        add(S, E, gamma_hat(D, S, E), f"S{S}_E{E}_gcal", (c["recall"], c["ndis_per_query"]))

    def fixed(m):
        add(m["S"], m["ef"], "0", f"S{m['S']}_E{m['ef']}_g0", (m["recall"], m["ndis"]))
        if "below" in m:
            b = m["below"]
            add(m["S"], b["ef"], "0", f"S{m['S']}_E{b['ef']}_g0", (b["recall"], b["ndis"]))

    for f in sorted(glob.glob(f"{D}/match/R*_fixed.json")):
        fixed(json.load(open(f)))
    for S in SS:
        for f in sorted(glob.glob(f"{D}/match/R*_S{S}_adaptive.json")):
            m = json.load(open(f))
            add(S, m["E"], m["gamma"], f"S{S}_E{m['E']}_g{m['gamma']}", (m["recall"], m["ndis"]))
        g200 = json.load(open(f"{D}/grid/S{S}_E200.json"))["curve"][0]
        add(S, 200, "0", f"S{S}_E200_g0", (g200["recall"], g200["ndis"]))
        gcal(S, 200)
    for S in (1, 5, 15):
        if ds == "t2i":                                   # CPU ratio against ndis ratio
            for E in (100, 300):
                g = json.load(open(f"{D}/grid/S{S}_E{E}.json"))["curve"][0]
                add(S, E, "0", f"S{S}_E{E}_g0", (g["recall"], g["ndis"]))
                gcal(S, E)
        else:                                             # calibrated cells and the fixed ef matched to their recall
            for E in (100, 200, 300):
                gcal(S, E)
                f = f"{D}/match/tab_S{S}_E{E}_fixed.json"
                if os.path.exists(f):
                    fixed(json.load(open(f)))
    # Cells with the same S run back to back (fewer index loads, steadier cache state).
    out.sort(key=lambda c: (c[0], c[1], c[2]))
    return out + [(1, 200, "0", "recheck_S1_E200_g0", None)]


def run(ds, S, E, g, name, expect):
    dsn = L.dsn(ds)
    f = f"{RESULTS}/cells/{dsn}/cpu/{name}.json"
    if os.path.exists(f):
        return
    os.makedirs(os.path.dirname(f), exist_ok=True)
    idx = f"{CFG['IDX_DIR']}/{dsn}"
    ki = L.k_i_of(K, S)
    norm = L.norm_flag(ds)
    cmd = shlex.split(CFG["PIN"]) + [
        CFG["BIN"], "search", "--shards", f"{idx}/S{S}", "--S", str(S), "--query", f"{CFG['RAW_DIR']}/{ds}/query.fbin",
        "--nq", "1000", "--normalize", str(norm), "--gt", f"{idx}/gt300.ivec", "--K", str(K), "--ki", str(ki),
        "--rule_k", str(ki), "--ef", str(E), "--gamma", g, "--repeat", "3", "--vt_threshold", "999999999"]
    if EXPECT_CPU:
        cmd += ["--expect_cpu", EXPECT_CPU]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    open(f + ".err", "w").write(p.stderr)
    try:
        d = json.loads(p.stdout)
    except Exception:
        sys.exit(f"!! {ds} {name} failed: {p.stderr[-300:]}")
    ok = [r for r in d if r.get("accept")]
    best = min(ok, key=lambda r: r["cpu_sum_ms"]["mean"]) if ok else None
    msg = (f"{time.strftime('%H:%M:%S')} {ds:8} {name:28} recall={d[0]['recall']:.5f} ndis={d[0]['ndis_per_query']:>10.1f} "
           f"cpu={[round(r['cpu_sum_ms']['mean'], 3) for r in d]} accepted={len(ok)}/{len(d)} "
           f"throttled={[r.get('nr_throttled_delta') for r in d]} ({time.time() - t0:.0f}s)")
    print(msg + (f" -> {best['cpu_sum_ms']['mean']:.4f}" if best else "  ** no accepted repetition **"), flush=True)
    if expect and (f"{d[0]['recall']:.5f}" != f"{expect[0]:.5f}" or f"{d[0]['ndis_per_query']:.1f}" != f"{expect[1]:.1f}"):
        open(f + ".mismatch", "w").write(p.stdout)
        sys.exit(f"!! {ds} {name}: recall/ndis differ from the earlier run {expect[0]:.5f}/{expect[1]:.1f}")
    open(f + ".part", "w").write(p.stdout)
    os.replace(f + ".part", f)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("t2i", "msmarco"):
        sys.exit(__doc__)
    ds = sys.argv[1]
    RESULTS = CFG["RESULTS"]
    CELLS_ROOT = os.environ.get("CELLS_ROOT", f"{RESULTS}/cells")
    cs = cells(ds)
    if "--list" in sys.argv:
        for c in cs:
            print(ds, c[3])
        print(len(cs), "cells")
        sys.exit()
    if not os.access(CFG["BIN"], os.X_OK):
        sys.exit(f"!! benchmark binary not found: {CFG['BIN']}")
    for c in cs:
        run(ds, *c)
    print(f"[cpu {ds}] done", time.strftime("%m-%d %H:%M:%S"), flush=True)
