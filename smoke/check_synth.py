# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)

"""Invariant checks on the synthetic smoke-test results (smoke/smoke.sh --full).

For every (S, E) of the run:
  1. the gamma-grid replay at gamma = 0 equals the fixed-ef bench run of the same record (recall and ndis)
  2. the rule only truncates a traversal: per query, adaptive ndis <= fixed-ef ndis, for adaptive termination and rank-K termination
  3. rank-K termination never stops earlier than adaptive termination (rank k_i <= K) at the same gamma -- checked on the replay grid
  4. per-query hits lie in [0, K] and the bench's recall equals mean(hits) / K
  5. replayed recall/ndis at a grid gamma equal a direct run at that gamma (cell.sh direct), when present
When RESULTS/patience/<dsn>/ exists (run_extras.sh patience), also:
  6. the record with insertion counts (V31Q) follows the same search: its gamma = 0 end state equals the grid's gamma = 0
     point, and lib/trace_replay.py on it gives the grid's whole gamma curve (needs TRACE_DIR and IDX_DIR)
  7. Patience replay equals direct runs (patience.py verify: six settings, and p = 1e9 equals fixed ef for both queues;
     every matched or default setting was confirmed by a direct run), and Patience never costs more than fixed ef
Exit 1 on any violation.

Usage: python3 check_synth.py RESULTS DSN "S..." "E..."
"""
import csv, glob, json, os, sys

res, dsn, SS, ES = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split()], [int(x) for x in sys.argv[4].split()]
ds, K = dsn.split("_N")[0], 20
bad = []
NDIS_TOL = 0.051          # the bench prints ndis_per_query with one decimal; the replay keeps full precision


def first(x):
    return x[0] if isinstance(x, list) else x


def dump(S, E, arm, blk):
    rows = {}
    with open(f"{res}/dumps/{ds}_S{S}_ef{E}_{arm}_{blk}.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows[int(r["q"])] = (int(r["hits"]), float(r["ndis"]))
    return rows, first(json.load(open(f"{res}/dumps/{ds}_S{S}_ef{E}_{arm}_{blk}.json")))


n_checked = 0
for S in SS:
    for E in ES:
        g = json.load(open(f"{res}/cells/{dsn}/grid/S{S}_E{E}.json"))
        c0, b = g["curve"][0], first(g["bench"])
        if abs(c0["recall"] - b["recall"]) > 1e-9 or abs(c0["ndis"] - b["ndis_per_query"]) > NDIS_TOL:
            bad.append(f"S={S} E={E}: replay(gamma=0) {c0['recall']:.6f}/{c0['ndis']:.2f} != bench {b['recall']:.6f}/{b['ndis_per_query']:.2f}")
        if S > 1 and os.path.exists(f"{res}/cells/{dsn}/dabs/grid/S{S}_E{E}.json"):
            gd = json.load(open(f"{res}/cells/{dsn}/dabs/grid/S{S}_E{E}.json"))
            for pr, pd in zip(g["curve"], gd["curve"]):
                if pd["ndis"] + 1e-9 < pr["ndis"]:
                    bad.append(f"S={S} E={E} gamma={pr['gamma']}: rank-K termination ndis {pd['ndis']:.1f} < rule {pr['ndis']:.1f}")
        for f in glob.glob(f"{res}/cells/{dsn}/direct/S{S}_E{E}_g*.json"):
            gam = float(f.rsplit("_g", 1)[1][:-5]); d = first(json.load(open(f)))
            pt = min(g["curve"], key=lambda p: abs(p["gamma"] - gam))
            if abs(pt["gamma"] - gam) < 1e-12 and (abs(pt["recall"] - d["recall"]) > 1e-9 or abs(pt["ndis"] - d["ndis_per_query"]) > NDIS_TOL):
                bad.append(f"S={S} E={E} gamma={gam}: replay {pt['recall']:.6f}/{pt['ndis']:.2f} != direct {d['recall']:.6f}/{d['ndis_per_query']:.2f}")
        for blk in ("a", "b"):
            nat, jn = dump(S, E, "nat", blk)
            for arm in ("rule", "dabs"):
                p = f"{res}/dumps/{ds}_S{S}_ef{E}_{arm}_{blk}.tsv"
                if not os.path.exists(p):
                    continue
                ad, ja = dump(S, E, arm, blk)
                for q, (h, nd) in ad.items():
                    if nd > nat[q][1] + 1e-9:
                        bad.append(f"S={S} E={E} {arm} {blk} q={q}: ndis {nd} > fixed {nat[q][1]}")
                    if not 0 <= h <= K:
                        bad.append(f"S={S} E={E} {arm} {blk} q={q}: hits {h}")
                for rows, j, name in ((ad, ja, arm), (nat, jn, "nat")):
                    r = sum(h for h, _ in rows.values()) / (K * len(rows))
                    if abs(r - j["recall"]) > 1e-6:
                        bad.append(f"S={S} E={E} {name} {blk}: recall {j['recall']} != mean(hits)/K {r}")
                n_checked += len(ad)
                print(f"  S={S} E={E} blk={blk} {arm:4s}: recall {ja['recall']:.4f} vs fixed {jn['recall']:.4f}, "
                      f"ndis {ja['ndis_per_query']:.0f} vs {jn['ndis_per_query']:.0f}")

n_pat = 0
PAT = f"{res}/patience/{dsn}"
if os.path.isdir(PAT):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
    from trace_replay import read_gt, read_trace, replay
    tdir, idir = os.environ.get("TRACE_DIR"), os.environ.get("IDX_DIR")
    for f in sorted(glob.glob(f"{PAT}/cell/S*_E*.json")):
        c = first(json.load(open(f)))
        S, E, fe = c["S"], c["ef"], c["fixed_end"]
        gp = f"{res}/cells/{dsn}/grid/S{S}_E{E}.json"
        if not os.path.exists(gp):
            continue
        cv = json.load(open(gp))["curve"]
        if abs(fe["recall"] - cv[0]["recall"]) > 1e-9 or abs(fe["ndis"] - cv[0]["ndis"]) > 1e-6:
            bad.append(f"patience S={S} E={E}: gamma = 0 end state {fe['recall']:.6f}/{fe['ndis']:.2f} != grid {cv[0]['recall']:.6f}")
        if tdir and idir:
            meta, recs = read_trace(f"{tdir}/{dsn}/patience/S{S}_E{E}.bin")
            gt = read_gt(f"{idir}/{dsn}/gt300.ivec", meta["nq"])
            for pt in cv:
                r, nd = replay(meta, recs, gt, K, repr(pt["gamma"]) if pt["gamma"] else "0")
                if abs(r - pt["recall"]) > 1e-9 or abs(nd - pt["ndis"]) > 1e-6:
                    bad.append(f"patience S={S} E={E} gamma={pt['gamma']}: V31Q replay {r:.6f}/{nd:.2f} != grid {pt['recall']:.6f}/{pt['ndis']:.2f}")
        for x in c["mono"]:
            if x["recall"] > fe["recall"] + 1e-12 or x["ndis"] > fe["ndis"] + 1e-6:
                bad.append(f"patience S={S} E={E} {x['queue']} p={x['p']} s={x['s']}: above fixed ef")
        n_pat += 1
    vf = glob.glob(f"{PAT}/verify/*.json")
    if not vf:
        bad.append("patience: no verify/ result")
    for f in vf:
        v = json.load(open(f))
        for r in v["rows"]:
            if not r["same"]:
                bad.append(f"patience verify S={v['S']} E={v['E']} {r['queue']} p={r['p']} s={r['s']}: {r}")
        n_pat += len(v["rows"])
    for f in glob.glob(f"{PAT}/match/*.json") + glob.glob(f"{PAT}/default/*.json"):
        d = json.load(open(f))
        if not (d.get("direct", {}).get("same") if "direct" in d else d.get("direct_same")):
            bad.append(f"patience {os.path.basename(f)}: direct run differs from the replay")
        n_pat += 1
    print(f"  patience: {n_pat} checks on {len(glob.glob(f'{PAT}/cell/*.json'))} cells")

print(f"checked {n_checked} per-query rows; {len(bad)} violations")
for b in bad[:20]:
    print("  !!", b)
sys.exit(1 if bad else 0)
