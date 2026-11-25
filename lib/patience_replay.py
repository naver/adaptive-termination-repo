# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)

"""Replay a traversal record with insertion counts (bench `search --trace ... --trace_ins 1`) under queue-saturation
early termination (Patience; Teofili and Lin, ECIR 2025) and under adaptive termination, without searching again.

The rule (faiss/patches/0003-patience.patch). The search counts, cumulatively, how many points have entered one queue:
  ef  the ef nearest points whose distance has been computed so far (k = E, the queue is the beam), or
  ki  the shard's result heap (k = k_i).
After each expanded candidate it divides the count after the previous expansion by the current count (double; 0/0 is
NaN and not saturated). A ratio >= s is a saturated expansion; when more than p expansions in a row are saturated, the
search stops before the next pop. Lucene's PatienceKnnVectorQuery defaults are s = 0.995 and p = max(7, floor(0.3 k)).

Why replay is exact: Patience, like adaptive termination, only decides when a shard search stops. A run with
(s, p, queue) is the gamma = 0 search cut at the first expansion where it fires. Rows and updates correspond as follows:
  - row t (the pop of candidate t) holds the counts when candidate t is popped, i.e. after candidate t-1 was expanded;
    row 0 holds the counts after the entry point only.
  - the update after expanding candidate j (j = 0 .. n-2) uses cur = ins[j+1] and prev = ins[j] (prev = 0 for j = 0).
    If it fires, the search stops in the state of row j+1 (result heap and base-layer ndis before the next pop);
    if it never fires, the state is the last row (end of the loop).
  - if the loop left at the last pop without expanding it, the update read at the last row compares two equal counts;
    firing there still yields the end state, so the result is the same.
  - adaptive termination: as lib/trace_replay.py (float32 test right after the pop, stopped state = that pop's row).
Recall and ndis follow lib/trace_replay.py: shards are disjoint, so the union of the shards' results intersected with
the true top-K is the sum of per-shard hits; recall is summed in query order as double; ndis = upper + base at the stop.

Record format: lib/trace_replay.py, magic "V31Q" (columns ins_ef[npop], ins_ki[npop] after base_ndis).

Subcommands (JSON on stdout, one entry per record):
  eval   --configs "ef:p:s ki:p:s g:gamma ..."     recall and ndis of each setting
  cell   for one record: gamma = 0 end state (checked against the recording run's bench JSON with --bench), gamma
         replays (--gammas), for every queue and p in --ps and R in --R the smallest s on a %.6f grid with recall >= R
         (bisection; None = not even s = 1 reaches R, i.e. Patience has to be off), a p x s grid for the monotonicity
         check, and with --default the Lucene defaults (with --hits_out, per-query hits and ndis as .npz)
  mech   where each setting stops, split into shard searches that contribute to the global top-K and those that do not
Usage: python3 patience_replay.py <eval|cell|mech> --trace T.bin [T2.bin ...] --gt GT.ivec [--K 20] [options]
"""
import argparse
import json
import os
import sys

import numpy as np

from trace_replay import TRACE_MAGIC_INS, need, read_gt

QUEUES = ("ef", "ki")
P_GRID = "1 2 3 5 7 10 15 20 30 45 60 90 120"
MONO_S = ("0.9", "0.95", "0.99", "0.995", "0.999", "1.0")
DEFAULT_S = "0.995"


def default_p(k):
    """Lucene's default patience for a queue of size k: max(7, floor(0.3 k))"""
    return max(7, int(k * 0.3))


class Trace:
    """One record as flat arrays over all rows: d0, dk, base ndis, the two insertion counts and the hits (number of the
    global top-K among the row's result ids). Per shard search: start row, row count, upper-layer ndis, query."""

    def __init__(self, path, gt_path, K):
        need(path, "traversal record")
        buf = np.memmap(path, np.uint8, "r")
        if bytes(buf[:4]) != TRACE_MAGIC_INS:
            sys.exit(f"!! not a record with insertion counts (bench --trace_ins 1): {path}")
        S, nq, ki, rule_k, ef = (int(x) for x in np.frombuffer(buf[4:24], np.int32))
        self.meta = dict(S=S, nq=nq, ki=ki, rule_k=rule_k, ef=ef)
        self.K = K
        gt = read_gt(gt_path, nq)[:, :K]
        truth = [np.sort(gt[q]) for q in range(nq)]
        heads, off, total, L = [], 24, 0, len(buf)
        while off < L:                                   # first pass: record headers only
            s, q, npop, up = (int(x) for x in np.frombuffer(buf[off:off + 16], np.int32))
            heads.append((s, q, npop, up, off + 16))
            total += npop
            off += 16 + 4 * npop * (5 + ki)
        if off != L:
            sys.exit(f"!! truncated traversal record: {path}")
        R = len(heads)
        self.rec_q = np.empty(R, np.int32); self.rec_up = np.empty(R, np.int64)
        self.start = np.empty(R, np.int64); self.n = np.empty(R, np.int64)
        self.d0 = np.empty(total, np.float32); self.dk = np.empty(total, np.float32)
        self.nd = np.empty(total, np.int64); self.ins_ef = np.empty(total, np.int64)
        self.ins_ki = np.empty(total, np.int64); self.hits = np.empty(total, np.int32)
        at = 0
        for i, (s, q, npop, up, o) in enumerate(heads):
            cols = np.frombuffer(buf[o:o + 4 * npop * (5 + ki)], np.int32)
            self.rec_q[i], self.rec_up[i], self.start[i], self.n[i] = q, up, at, npop
            sl = slice(at, at + npop)
            self.d0[sl] = cols[0:npop].view(np.float32)
            self.dk[sl] = cols[npop:2 * npop].view(np.float32)
            self.nd[sl] = cols[2 * npop:3 * npop]
            self.ins_ef[sl] = cols[3 * npop:4 * npop]
            self.ins_ki[sl] = cols[4 * npop:5 * npop]
            gid = cols[5 * npop:].reshape(npop, ki)
            t = truth[q]
            pos = np.searchsorted(t, gid).clip(0, K - 1)
            self.hits[sl] = ((t[pos] == gid) & (gid >= 0)).sum(1)
            at += npop
        self.end = self.start + self.n - 1                # end-state row of each shard search
        self.local = np.arange(total) - np.repeat(self.start, self.n)    # row index inside its shard search
        self._sat, self._win = {}, None

    def evaluate(self, stop):
        """stopped row per shard search -> (recall, ndis per query, per-query hits, per-query ndis)"""
        nq = self.meta["nq"]
        hq = np.bincount(self.rec_q, weights=self.hits[stop], minlength=nq).astype(np.int64)
        nqd = np.bincount(self.rec_q, weights=(self.rec_up + self.nd[stop]), minlength=nq).astype(np.int64)
        rs = 0.0
        for q in range(nq):                               # in query order, as the bench sums (double)
            rs += float(hq[q]) / self.K
        return rs / nq, float(nqd.sum()) / nq, hq, nqd

    def _first(self, fire):
        """first row with fire = True in each shard search, else its end row"""
        idx = np.flatnonzero(fire)
        if idx.size == 0:
            return self.end.copy()
        p = np.searchsorted(idx, self.start)
        ok = p < len(idx)
        cand = np.where(ok, idx[np.minimum(p, len(idx) - 1)], -1)
        return np.where(ok & (cand <= self.end), cand, self.end)

    def stop_gamma(self, gamma_str):
        """adaptive termination with one gamma for every shard (lib/trace_replay.stop_row, vectorized)"""
        g = np.float32(float(gamma_str))
        if g <= 0:
            return self.end.copy()
        f = (np.float32(1.0) + g) * (np.float32(1.0) + g)
        with np.errstate(invalid="ignore"):
            fire = ~np.isnan(self.dk) & (self.d0 > (f * self.dk).astype(np.float32))
        fire[self.end] = False                            # the end row is not a pop
        return self._first(fire)

    def sat(self, queue):
        """Per row r >= 1 the saturation ratio of the update whose firing stops the search at row r (the update after
        expanding candidate r-1): cur = ins[r], prev = ins[r-1] (0 for r = 1). -1 where there is none (row 0) or the
        ratio is NaN, so that no threshold s accepts it."""
        if queue not in self._sat:
            ins = self.ins_ef if queue == "ef" else self.ins_ki
            cur = ins.astype(np.float64)
            prev = np.empty_like(cur); prev[0] = 0.0; prev[1:] = cur[:-1]
            prev[(self.start + 1)[self.n > 1]] = 0.0
            with np.errstate(invalid="ignore", divide="ignore"):
                s = np.minimum(cur, prev) / cur
            s[np.isnan(s)] = -1.0
            s[self.start] = -1.0
            self._sat[queue] = s
        return self._sat[queue]

    def window(self, queue, p):
        """Per row r the smallest ratio of the p + 1 updates ending at row r (rows r-p .. r); the search stops at the
        first row whose window value is >= s. -1 where the window would reach before the first update. Calls with the
        same queue and growing p extend the previous window instead of starting over."""
        s = self.sat(queue)
        if self._win and self._win[0] == queue and self._win[1] <= p:
            W, k0 = self._win[2], self._win[1] + 1
        else:
            W, k0 = s.copy(), 1
        for k in range(k0, p + 1):
            np.minimum(W[k:], s[:-k], out=W[k:])
        self._win = (queue, p, W)
        out = W.copy()
        out[self.local < p + 1] = -1.0
        return out

    def stop_patience(self, W, s_str):
        return self._first(W >= float(s_str))

    def run(self, queue, p, s_str):
        return self.evaluate(self.stop_patience(self.window(queue, p), s_str))


def search_s(T, W, R, step=1e-6):
    """Smallest s on the %.6f grid with recall >= R (recall does not decrease with s). -> ((s, (recall, ndis)) or
    None when even s = 1 does not reach R, number of evaluations)"""
    cache = {}

    def at(i):
        st = f"{i * step:.6f}"
        if st not in cache:
            r, nd, _, _ = T.evaluate(T.stop_patience(W, st))
            cache[st] = (r, nd)
        return st, cache[st]

    hi = int(round(1 / step))
    if at(hi)[1][0] < R:
        return None, len(cache)
    lo = 0
    if at(lo)[1][0] >= R:
        return at(lo), len(cache)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if at(mid)[1][0] >= R:
            hi = mid
        else:
            lo = mid
    return at(hi), len(cache)


def match_rows(T, fixed_end, Rs, ps):
    """for every queue, p and R: the smallest s reaching R (s None = Patience off, the fixed-ef end state)"""
    out = []
    r0, n0 = fixed_end
    for q in QUEUES:
        for p in ps:
            W = T.window(q, p)
            for R in Rs:
                if r0 < R:
                    out.append(dict(queue=q, p=p, R=R, feasible=False)); continue
                best, probes = search_s(T, W, R)
                x = dict(queue=q, p=p, R=R, feasible=True, probes=probes)
                if best is None:
                    x.update(s=None, recall=r0, ndis=n0)
                else:
                    x.update(s=best[0], recall=best[1][0], ndis=best[1][1])
                out.append(x)
    return out


def mech_rows(T, configs):
    """Where each setting stops. A shard search is empty (non-contributing) when the end state of the gamma = 0 search
    holds none of the global top-K, the oracle's definition of its 'empty' share."""
    empty = T.hits[T.end] == 0
    nq = T.meta["nq"]
    out = []
    for c in ["fixed"] + configs:
        if c == "fixed":
            st, desc = T.end.copy(), dict(kind="fixed")
        elif c.startswith("g:"):
            st, desc = T.stop_gamma(c[2:]), dict(kind="gamma", gamma=c[2:])
        else:
            q, p, sv = c.split(":")
            st, desc = T.stop_patience(T.window(q, int(p)), sv), dict(kind="patience", queue=q, p=int(p), s=sv)
        base = T.nd[st]
        tot = T.rec_up + base
        r, nd, _, _ = T.evaluate(st)
        row = dict(desc, recall=r, ndis=nd, stopped=float((st < T.end).mean()), empty_frac=float(empty.mean()),
                   empty_share=float(tot[empty].sum() / tot.sum()))
        for name, m in (("all", np.ones_like(empty)), ("empty", empty), ("useful", ~empty)):
            x = base[m]
            row[name] = None if x.size == 0 else dict(
                n=int(x.size), ndis_per_query=float(tot[m].sum() / nq), base_mean=float(x.mean()),
                base_median=float(np.median(x)), base_p1=float(np.percentile(x, 1)), base_min=int(x.min()),
                stopped=float((st[m] < T.end[m]).mean()))
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["eval", "cell", "mech"])
    ap.add_argument("--trace", required=True, nargs="+"); ap.add_argument("--gt", required=True)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--configs", default="", help='eval, mech: "queue:p:s ..." and "g:gamma ..."')
    ap.add_argument("--R", default="0.95 0.90", help="cell: target recalls of the s search")
    ap.add_argument("--ps", default=P_GRID, help="cell: p grid")
    ap.add_argument("--gammas", default="", help="cell: gammas to replay (adaptive termination)")
    ap.add_argument("--bench", default="", help="cell: bench JSON of the recording run (gamma = 0 must equal it)")
    ap.add_argument("--default", action="store_true", help="cell: also the Lucene defaults")
    ap.add_argument("--hits_out", default="", help="cell --default: per-query hits and ndis (.npz)")
    a = ap.parse_args()
    out = []
    for tp in a.trace:
        T = Trace(tp, a.gt, a.K)
        m = T.meta
        r0, n0, h0, d0 = T.evaluate(T.end.copy())
        base = dict(m, trace=os.path.basename(tp), fixed_end={"recall": r0, "ndis": n0})
        if a.cmd == "eval":
            res = []
            for c in a.configs.split():
                parts = c.split(":")
                if parts[0] == "g":
                    r, nd, _, _ = T.evaluate(T.stop_gamma(parts[1]))
                    res.append(dict(kind="gamma", gamma=parts[1], recall=r, ndis=nd))
                else:
                    r, nd, _, _ = T.run(parts[0], int(parts[1]), parts[2])
                    res.append(dict(kind="patience", queue=parts[0], p=int(parts[1]), s=parts[2], recall=r, ndis=nd))
            out.append(dict(base, results=res))
        elif a.cmd == "mech":
            out.append(dict(base, results=mech_rows(T, a.configs.split())))
        else:
            row = dict(base)
            if a.bench:
                b = json.load(open(a.bench))[0]
                same = f"{r0:.5f}" == f"{b['recall']:.5f}" and f"{n0:.1f}" == f"{b['ndis_per_query']:.1f}"
                row["bench"] = {"recall": b["recall"], "ndis_per_query": b["ndis_per_query"], "same": same}
                if not same:
                    sys.exit(f"!! gamma = 0 replay {r0:.5f}/{n0:.1f} != bench {b['recall']:.5f}/{b['ndis_per_query']:.1f}")
            row["gamma"] = []
            for g in a.gammas.split():
                r, nd, _, _ = T.evaluate(T.stop_gamma(g))
                row["gamma"].append(dict(gamma=g, recall=r, ndis=nd))
            ps = [int(x) for x in a.ps.split()]
            row["match"] = match_rows(T, (r0, n0), [float(x) for x in a.R.split()], ps)
            row["mono"] = []                              # recall and ndis must not decrease with p or s
            for q in QUEUES:
                for p in ps:
                    W = T.window(q, p)
                    for st in MONO_S:
                        r, nd, _, _ = T.evaluate(T.stop_patience(W, st))
                        row["mono"].append(dict(queue=q, p=p, s=st, recall=r, ndis=nd))
            if a.default:
                hits = {"fixed_h": h0, "fixed_n": d0}
                row["default"] = []
                for q in QUEUES:
                    k = m["ef"] if q == "ef" else m["ki"]
                    p = default_p(k)
                    r, nd, hq, nqd = T.run(q, p, DEFAULT_S)
                    row["default"].append(dict(queue=q, k=k, p=p, s=DEFAULT_S, recall=r, ndis=nd))
                    hits[f"{q}_h"], hits[f"{q}_n"] = hq, nqd
                if a.hits_out:
                    np.savez(a.hits_out, **hits)
            out.append(row)
        print(f"  {os.path.basename(tp)} done", file=sys.stderr, flush=True)
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
