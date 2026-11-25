# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)

"""Replay a traversal record (bench `search --trace`) at any gamma, without searching again.

Why replay is exact: the rule only decides *when* a shard search stops; until it fires, the traversal is the same as
with the rule off. A search with gamma g is therefore the gamma = 0 search cut at the first base-layer pop where the
rule fires. The bench, run once with --gamma 0, records for every base-layer pop the frontier distance d0, the
distance d_k of the rank-th result found so far (rank = --rule_k), the cumulative base-layer distance count and the
ids in the result heap. Replaying a gamma is then a scan over those rows.

To agree with a direct run to the last printed digit, the replay mirrors the patched Faiss test exactly:
  - test `have && d0 > factor * dk` in float32, factor = (1 + g) * (1 + g) with g = float32(atof(string));
    have = the auxiliary heap already holds rank results (dk is not NaN). The test runs right after the pop,
    before Faiss' own stopping test, which is also where the record row is written.
  - stopped state: the row of the pop where the rule fires (result heap before expansion); if it never fires,
    the last row (state after the loop).
  - recall: union of the shards' result ids (-1 dropped), intersected with the true top-K, summed in query order
    as double and divided by nq.
  - ndis: per shard, upper-layer count + base-layer cumulative count at the stopped row; summed and divided by nq.

Trace file format (little endian), written by bench.cpp:
    header            4 bytes magic "V31T", int32 S, nq, ki, rule_k, ef
    per (shard, query) int32 s, q, npop, upper_ndis | float32 d0[npop] | float32 dk[npop] (NaN = aux heap not full)
                      | int32 base_ndis[npop] | int32 gid[npop * ki] (global ids, -1 = empty slot)
    The last row of a record (d0 = NaN) is the end state after the search loop, not a pop.
With --trace_ins 1 the bench writes magic "V31Q" and two more columns after base_ndis, int32 ins_ef[npop] and
int32 ins_ki[npop] (queue insertion counts for lib/patience_replay.py). read_trace accepts both and skips those columns.

Usage (diagnostic): python3 trace_replay.py --trace T.bin --gt GT.ivec [--K 20] [--gammas "0.005 0.01 ..."]
"""
import argparse
import os
import sys

import numpy as np

TRACE_MAGIC = b"V31T"   # format tag written by bench.cpp --trace (part of the file format, keep in sync)
TRACE_MAGIC_INS = b"V31Q"   # the same with --trace_ins 1: two insertion-count columns after base_ndis

# gamma grid of the replay curves (grid_replay.py); the calibration grid is shardlib.CALIB_GAMMAS (20 values)
GRID_GAMMAS = "0.005 0.01 0.015 0.02 0.025 0.03 0.035 0.04 0.045 0.05 0.06 0.08"


def need(path, what):
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        sys.exit(f"!! {what} not found or empty: {path}")


def read_trace(path):
    """-> (meta {S, nq, ki, rule_k, ef}, [(s, q, upper, d0, dk, nd, gid), ...])"""
    need(path, "traversal record")
    buf = open(path, "rb").read()
    if buf[:4] not in (TRACE_MAGIC, TRACE_MAGIC_INS):
        sys.exit(f"!! not a traversal record (bad magic {buf[:4]!r}): {path}")
    extra = 2 if buf[:4] == TRACE_MAGIC_INS else 0     # insertion-count columns, not used by the gamma replay
    S, nq, ki, rule_k, ef = np.frombuffer(buf, np.int32, 5, 4)
    off, recs = 24, []
    while off < len(buf):
        s, q, npop, upper = np.frombuffer(buf, np.int32, 4, off); off += 16
        d0 = np.frombuffer(buf, np.float32, npop, off); off += 4 * npop
        dk = np.frombuffer(buf, np.float32, npop, off); off += 4 * npop
        nd = np.frombuffer(buf, np.int32, npop, off); off += 4 * npop
        off += 4 * npop * extra
        gid = np.frombuffer(buf, np.int32, npop * ki, off).reshape(npop, ki); off += 4 * npop * ki
        recs.append((int(s), int(q), int(upper), d0, dk, nd, gid))
    if off != len(buf):
        sys.exit(f"!! truncated traversal record: {path}")
    return dict(S=int(S), nq=int(nq), ki=int(ki), rule_k=int(rule_k), ef=int(ef)), recs


def read_gt(path, nq):
    """First nq rows of a ground-truth ivec (row i = query qoff + i)."""
    need(path, "ground truth")
    n, k = np.fromfile(path, np.int32, 2)
    if int(n) < nq:
        sys.exit(f"!! ground truth {path} has {int(n)} rows, need {nq}")
    return np.fromfile(path, np.int32, offset=8).reshape(int(n), int(k))[:nq]


def stop_row(d0, dk, gamma_str):
    """Row at which a shard search with gamma (given as the string passed to the bench) stops."""
    g = np.float32(float(gamma_str))
    if g <= 0:
        return len(d0) - 1                             # rule off: end state
    factor = (np.float32(1.0) + g) * (np.float32(1.0) + g)
    body = slice(0, len(d0) - 1)                       # the last row (d0 = NaN) is not a pop
    with np.errstate(invalid="ignore"):
        fire = ~np.isnan(dk[body]) & (d0[body] > (factor * dk[body]).astype(np.float32))
    hit = np.flatnonzero(fire)
    return int(hit[0]) if hit.size else len(d0) - 1


def replay(meta, recs, gt, K, gamma_str):
    """One gamma (same for every shard) -> (recall, ndis per query), the same computation as bench search."""
    nq = meta["nq"]
    merged = [set() for _ in range(nq)]
    ndis_total = 0
    for s, q, upper, d0, dk, nd, gid in recs:
        stop = stop_row(d0, dk, gamma_str)
        merged[q].update(int(x) for x in gid[stop] if x >= 0)
        ndis_total += upper + int(nd[stop])
    recall_sum = 0.0
    for q in range(nq):                                # summed in query order, as the bench does (double)
        truth = set(int(x) for x in gt[q, :K])
        recall_sum += float(len(truth & merged[q])) / K
    return recall_sum / nq, ndis_total / nq


def as_calibrate(recall, ndis):
    """Round as a bench JSON (%.5f / %.1f) read back by a calibration script (%.6f / %.1f)."""
    r5 = float(f"{recall:.5f}")
    n1 = float(f"{ndis:.1f}")
    return float(f"{r5:.6f}"), float(f"{n1:.1f}")


def choose(pairs, eps_pp):
    """Direct-measurement choice gamma*: the smallest gamma > 0 whose recall is within eps_pp of gamma = 0.
    pairs = [(gamma, recall, ndis), ...] with pairs[0] the gamma = 0 point. -> (gamma, recall, ndis ratio)"""
    eps = float(eps_pp) / 100.0
    base_r, base_n = pairs[0][1], pairs[0][2]
    ok = [p for p in pairs if p[1] >= base_r - eps]
    cand = [p for p in ok if p[0] > 0] or ok
    g, r, n = min(cand, key=lambda p: p[0])
    return g, r, n / base_n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trace", required=True); ap.add_argument("--gt", required=True)
    ap.add_argument("--K", type=int, default=20); ap.add_argument("--gammas", default=GRID_GAMMAS)
    a = ap.parse_args()
    meta, recs = read_trace(a.trace)
    gt = read_gt(a.gt, meta["nq"])
    print(f"S={meta['S']} nq={meta['nq']} k_i={meta['ki']} rank={meta['rule_k']} ef={meta['ef']} records={len(recs)}")
    for gs in ["0"] + a.gammas.split():
        r, nd = replay(meta, recs, gt, a.K, gs)
        print(f"gamma={gs:<7} recall={r:.5f} ndis={nd:.1f}")


if __name__ == "__main__":
    main()
