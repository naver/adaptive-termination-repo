# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)

"""Held-out calibration rows of a shard (calibration before serving, the `corpus` arm).

A shard that has not served any query yet can only calibrate with its own vectors. For every shard s of a split, n_hold
rows (500) are set aside, the graph is built from the rest (partial graph), the held-out rows are searched against it
as calibration queries, and then they are inserted (final graph, the one that is served and evaluated).

Held-out rows of shard s at a given S: np.random.default_rng([20261024, S, s]).choice(n_s, n_hold, replace=False),
sorted, as positions in shard order (the order of shard<s>.sel written by shard_split.py). They do not depend on E.
The final graph holds the rest in shard order followed by the held-out rows, which is the order of its .sel file.

Commands
  split    shard<s>.fbin + its .sel -> <part>/shard0.fbin, shard0.sel (rest, the partial graph's input and id map),
           <part>/hold.fbin, hold.sel (held-out vectors and global ids), <part>/hold.json
  gtmap    ground truth of the held-out rows against the rest, computed by `bench gt` with local ids -> global ids
  finalsel the final graph's id map: rest ids, then held-out ids
  diag     per shard: distances to the 1st and k_i-th nearest in-shard point (squared L2, the search metric), for the
           held-out rows (against the rest) and for the benchmark calibration queries #1000-1499 (against the shard);
           held-out rows at distance 0 from a row of the rest (duplicate vectors) are counted, not removed

Usage: python3 holdout.py split --fbin D/shard3.fbin --sel D/shard3.orig.sel --S 5 --s 3 --out D/part3 [--n_hold 500]
       python3 holdout.py gtmap --gt local.ivec --sel D/part3/shard0.sel --out D/part3/hold_gt.ivec
       python3 holdout.py finalsel --part D/part3 --out D/shard3.sel
       python3 holdout.py diag --ds glove --idx IDX/<dsn> --base base.fbin --query query.fbin --S 5 --out holdout.json
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

import shardlib as L

SEED = 20261024
N_HOLD = 500
DUP_EPS = 1e-9          # squared distance below which a held-out row counts as a duplicate of a row of the rest


def hold_rows(n_s, S, s, n_hold=N_HOLD):
    return np.sort(np.random.default_rng([SEED, S, s]).choice(n_s, n_hold, replace=False))


def read_sel(path):
    return L.read_ivec(path)[:, 0]


def write_sel(path, ids):
    L.write_ivec(path, np.asarray(ids, np.int32).reshape(-1, 1))


def cmd_split(a):
    n_s, d = (int(x) for x in np.fromfile(a.fbin, np.int32, 2))
    sel = read_sel(a.sel)
    if len(sel) != n_s:
        sys.exit(f"!! {a.sel} has {len(sel)} ids, {a.fbin} has {n_s} rows")
    rows = hold_rows(n_s, a.S, a.s, a.n_hold)
    is_hold = np.zeros(n_s, bool)
    is_hold[rows] = True
    os.makedirs(a.out, exist_ok=True)
    write_sel(os.path.join(a.out, "shard0.sel"), sel[~is_hold])
    write_sel(os.path.join(a.out, "hold.sel"), sel[rows])
    rest = open(os.path.join(a.out, "shard0.fbin.part"), "wb")
    rest.write(np.array([n_s - len(rows), d], np.int32).tobytes())
    hold = np.empty((len(rows), d), np.float32)
    nh = 0
    with open(a.fbin, "rb") as f:                   # streamed: a shard of MS MARCO at S=1 is 36 GB
        f.seek(8)
        for off in range(0, n_s, a.chunk):
            m = min(a.chunk, n_s - off)
            X = np.frombuffer(f.read(m * d * 4), np.float32).reshape(m, d)
            h = is_hold[off:off + m]
            rest.write(np.ascontiguousarray(X[~h]).tobytes())
            hold[nh:nh + h.sum()] = X[h]
            nh += int(h.sum())
    rest.close()
    os.replace(os.path.join(a.out, "shard0.fbin.part"), os.path.join(a.out, "shard0.fbin"))
    L.write_fbin(os.path.join(a.out, "hold.fbin"), hold)
    info = dict(S=a.S, s=a.s, n_s=n_s, n_hold=len(rows), seed=[SEED, a.S, a.s],
                rows_md5=hashlib.md5(rows.astype(np.int64).tobytes()).hexdigest())
    json.dump(info, open(os.path.join(a.out, "hold.json"), "w"), indent=1)
    print(f"[holdout] S={a.S} shard {a.s}: {n_s:,} rows -> rest {n_s - len(rows):,} + held out {len(rows)} "
          f"(rows md5 {info['rows_md5'][:8]})")


def cmd_gtmap(a):
    gt = L.read_ivec(a.gt)
    sel = read_sel(a.sel)
    if gt.max() >= len(sel):
        sys.exit(f"!! local ids in {a.gt} exceed the {len(sel)} rows of {a.sel}")
    L.write_ivec(a.out, np.where(gt >= 0, sel[np.maximum(gt, 0)], -1))


def cmd_finalsel(a):
    ids = np.concatenate([read_sel(os.path.join(a.part, "shard0.sel")), read_sel(os.path.join(a.part, "hold.sel"))])
    write_sel(a.out, ids)


def sqdist(base, norm, q, ids):
    """squared L2 between query rows q (nq x d) and base rows ids (nq x k, global), as the search computes it"""
    X = np.asarray(base[ids.reshape(-1)], np.float32)
    if norm:
        X = L.normalize_rows(X)
    X = X.reshape(ids.shape[0], ids.shape[1], -1)
    return ((X - q[:, None, :].astype(np.float32)) ** 2).sum(-1, dtype=np.float64)


def cmd_diag(a):
    ds = L.ds_short(a.ds)
    N = a.N or L.ds_N(ds)
    norm = L.norm_flag(ds)
    n_file, d = (int(x) for x in np.fromfile(a.base, np.int32, 2))
    base = np.memmap(a.base, np.float32, "r", offset=8, shape=(n_file, d))
    ki = L.k_i_of(a.K, a.S)
    Q = L.read_fbin(a.query, L.CALIB[0], L.CALIB[1] - L.CALIB[0])
    if norm:
        Q = L.normalize_rows(Q)
    gtp = L.read_ivec(a.gt_proxy)[: len(Q)]
    so = L.shard_assignment(N, a.S) if a.S > 1 else np.zeros(N, np.int32)
    D = os.path.join(a.idx, f"S{a.S}")
    shards, h1, hk, b1, bk = [], [], [], [], []
    for s in range(a.S):
        p = os.path.join(D, f"part{s}")
        info = json.load(open(os.path.join(p, "hold.json")))
        H = L.read_fbin(os.path.join(p, "hold.fbin"))
        hg = L.read_ivec(os.path.join(p, "hold_gt.ivec"))[:, :ki]
        dh = sqdist(base, norm, H, hg)
        # benchmark calibration queries: the shard's true list is the global ground truth filtered to the shard
        lists = [r[so[r] == s][:ki] for r in gtp]
        ok = np.array([len(x) == ki for x in lists])
        db = sqdist(base, norm, Q[ok], np.stack([x for x, o in zip(lists, ok) if o]))
        info.update(k_i=ki, duplicates=int((dh[:, 0] < DUP_EPS).sum()),
                    hold_d1_median=float(np.median(dh[:, 0])), hold_dki_median=float(np.median(dh[:, -1])),
                    bench_d1_median=float(np.median(db[:, 0])), bench_dki_median=float(np.median(db[:, -1])),
                    bench_short_lists=int((~ok).sum()))
        shards.append(info)
        h1.append(dh[:, 0]); hk.append(dh[:, -1]); b1.append(db[:, 0]); bk.append(db[:, -1])
    key = f"S{a.S}"
    entry = dict(ds=ds, S=a.S, k_i=ki, duplicates=sum(x["duplicates"] for x in shards),
                 hold_d1_median=float(np.median(np.concatenate(h1))), hold_dki_median=float(np.median(np.concatenate(hk))),
                 bench_d1_median=float(np.median(np.concatenate(b1))), bench_dki_median=float(np.median(np.concatenate(bk))),
                 shards=shards)
    tab = json.load(open(a.out)) if os.path.exists(a.out) else {}
    tab.setdefault(ds, {})[key] = entry
    json.dump(tab, open(a.out + ".part", "w"), indent=1, sort_keys=True)
    os.replace(a.out + ".part", a.out)
    print(f"[diag] {ds} S={a.S}: median d1 / d_k{ki}  held-out {entry['hold_d1_median']:.4g} / {entry['hold_dki_median']:.4g}"
          f"  benchmark {entry['bench_d1_median']:.4g} / {entry['bench_dki_median']:.4g}  duplicates {entry['duplicates']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("split")
    c.add_argument("--fbin", required=True); c.add_argument("--sel", required=True)
    c.add_argument("--S", type=int, required=True); c.add_argument("--s", type=int, required=True)
    c.add_argument("--out", required=True); c.add_argument("--n_hold", type=int, default=N_HOLD)
    c.add_argument("--chunk", type=int, default=1_000_000)
    c = sub.add_parser("gtmap")
    c.add_argument("--gt", required=True); c.add_argument("--sel", required=True); c.add_argument("--out", required=True)
    c = sub.add_parser("finalsel")
    c.add_argument("--part", required=True); c.add_argument("--out", required=True)
    c = sub.add_parser("diag")
    c.add_argument("--ds", required=True); c.add_argument("--N", type=int, default=0)
    c.add_argument("--idx", required=True, help="the holdout index directory of the dataset, IDX_DIR/<dsn>")
    c.add_argument("--base", required=True); c.add_argument("--query", required=True)
    c.add_argument("--gt_proxy", required=True, help="exact ground truth of the calibration queries (gt300_proxy.ivec)")
    c.add_argument("--S", type=int, required=True); c.add_argument("--K", type=int, default=L.K)
    c.add_argument("--out", required=True)
    a = ap.parse_args()
    {"split": cmd_split, "gtmap": cmd_gtmap, "finalsel": cmd_finalsel, "diag": cmd_diag}[a.cmd](a)


if __name__ == "__main__":
    main()
