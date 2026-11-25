# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)

"""Convert the downloaded source files into the raw layout every other script reads:

    $RAW_DIR/<ds>/base.fbin    int32 n, int32 d, float32 rows (original order, NOT normalized)
    $RAW_DIR/<ds>/query.fbin   same format, all queries in their original order
    $RAW_DIR/<ds>/prep_report.json   shapes, norms and the source files used

Vectors are written as published. Normalization to unit length happens later and in exactly one way: `shard_split.py
--normalize` for the indexed rows and `bench ... --normalize 1` for queries and exact ground truth (all datasets but SIFT and DEEP).

Subcommands
    t2i      --src CROPPED_BASE --query QUERY_FBIN --out DIR [--n 10000000] [--d 200] [--dataset t2i]
             (also the 100M sets: T2I 100M with --n 100000000 --dataset t2i100m, and DEEP 100M, whose base.1B.fbin has the
             same layout, with --n 100000000 --d 96 --dataset deep)
             CROPPED_BASE holds the first 8 + n*d*4 bytes of base.1B.fbin, so its header still says 1e9 rows; the header
             is rewritten to n and the file is moved (not copied) into DIR.
    msmarco  --parquet-dir DIR --query-pkl FILE --out DIR [--expect-n 8841823]
             Rows of base.fbin are ordered by passage id (row i = MS MARCO passage i); query.fbin keeps the order of the
             query pickle, and query_ids.txt lists the MS MARCO query id of each row.
    hdf5     --src FILE.hdf5 --out DIR [--expect-distance angular|euclidean]
             ANN-Benchmarks layout: base = 'train', queries = all of 'test'.
    selftest Run the converters on tiny fabricated inputs (needs pyarrow + pandas, and h5py; skips what is missing).

Usage: python3 data/convert.py <subcommand> ...
"""
import argparse
import glob
import json
import os
import pickle
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import shardlib as L  # noqa: E402

ID_COLS = ("docid", "id", "_id")
VEC_COLS = ("vector", "embedding", "emb")


# ------------------------------------------------------------------ small helpers
def fbin_header(path):
    n, d = np.fromfile(path, np.int32, 2)
    return int(n), int(d)


def write_header(f, n, d):
    f.write(np.array([n, d], np.int32).tobytes())


def write_fbin_blocks(path, blocks, n, d):
    """Write an fbin from an iterator of row blocks, via a temporary name (an interrupted run leaves no valid file)."""
    tmp = path + ".tmp"
    rows = 0
    with open(tmp, "wb") as f:
        write_header(f, n, d)
        for b in blocks:
            b = np.ascontiguousarray(b, np.float32)
            assert b.ndim == 2 and b.shape[1] == d, b.shape
            f.write(b.tobytes())
            rows += b.shape[0]
    if rows != n:
        os.remove(tmp)
        raise SystemExit(f"{path}: wrote {rows:,} rows, header says {n:,}")
    os.replace(tmp, path)


def norm_stats(path, rows=200_000):
    """Norm range of the first rows (to document whether the published vectors are unit length)."""
    n, d = fbin_header(path)
    m = min(n, rows)
    x = np.fromfile(path, np.float32, m * d, offset=8).reshape(m, d)
    nr = np.linalg.norm(x.astype(np.float64), axis=1)
    return {"rows_checked": m, "norm_min": float(nr.min()), "norm_max": float(nr.max()), "norm_mean": float(nr.mean())}


def finish(out, report, check=True):
    for k in ("base", "query"):
        p = os.path.join(out, f"{k}.fbin")
        n, d = fbin_header(p)
        report[k] = dict(report.get(k, {}), rows=n, dim=d, bytes=os.path.getsize(p), **norm_stats(p))
    spec = L.DATASETS.get(report.get("dataset"))
    if check and spec:                                 # the shapes the original runs used
        got = (report["base"]["rows"], report["base"]["dim"], report["query"]["rows"], report["query"]["dim"])
        want = (spec["N"], spec["d"], spec["nq"], spec["d"])
        if got != want:
            raise SystemExit(f"{report['dataset']}: (base rows, dim, query rows, dim) = {got}, expected {want}")
    with open(os.path.join(out, "prep_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


# ------------------------------------------------------------------ T2I
def convert_t2i(src, query, out, n=10_000_000, d=200, check=True, dataset="t2i"):
    os.makedirs(out, exist_ok=True)
    base_p, q_p = os.path.join(out, "base.fbin"), os.path.join(out, "query.fbin")
    if not os.path.exists(base_p):
        want = 8 + n * d * 4
        if os.path.getsize(src) != want:
            raise SystemExit(f"{src}: {os.path.getsize(src):,} bytes, expected {want:,} (first {n:,} rows)")
        hn, hd = fbin_header(src)
        if hd != d or hn not in (1_000_000_000, n):
            raise SystemExit(f"{src}: header ({hn}, {hd}), expected (1000000000, {d})")
        with open(src, "r+b") as f:          # the cropped file still carries the 1B header
            write_header(f, n, d)
        shutil.move(src, base_p)
    if not os.path.exists(q_p):
        nq, dq = fbin_header(query)
        if dq != d or os.path.getsize(query) != 8 + nq * dq * 4:
            raise SystemExit(f"{query}: bad header ({nq}, {dq}) or size")
        shutil.copyfile(query, q_p + ".tmp")
        os.replace(q_p + ".tmp", q_p)
    finish(out, {"dataset": dataset, "source": {"base": "base.1B.fbin (first %d rows)" % n,
                                               "query": os.path.basename(query)}}, check)


# ------------------------------------------------------------------ MS MARCO
def pick(cols, cands, what):
    for c in cands:
        if c in cols:
            return c
    raise SystemExit(f"no {what} column in {cols}")


def parquet_files(root):
    """The distribution keeps its parquet files inside a *directory* whose name ends in .parquet -- skip directories."""
    return sorted(p for p in glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True) if os.path.isfile(p))


def convert_msmarco_base(files, out_path, expect_n=None):
    """Write the parquet rows as an fbin ordered by passage id. Ids must cover 0..N-1 exactly once."""
    import pyarrow.parquet as pq
    meta = [pq.ParquetFile(p) for p in files]
    cols = meta[0].schema_arrow.names
    idc, vc = pick(cols, ID_COLS, "id"), pick(cols, VEC_COLS, "vector")
    n = sum(m.metadata.num_rows for m in meta)
    if expect_n is not None and n != expect_n:
        raise SystemExit(f"{n:,} rows, expected {expect_n:,}")
    d = len(meta[0].read_row_group(0, columns=[vc]).column(0)[0].as_py())
    info = {"files": len(files), "rows": n, "dim": d, "id_col": idc, "vec_col": vc,
            "vec_type": str(meta[0].schema_arrow.field(vc).type)}
    print(json.dumps(info), flush=True)

    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        write_header(f, n, d)
    mm = np.memmap(tmp, np.float32, "r+", offset=8, shape=(n, d))
    seen = np.zeros(n, bool)
    for m in meta:
        for rg in range(m.num_row_groups):
            t = m.read_row_group(rg, columns=[idc, vc])
            ids = np.array([int(str(x)) for x in t.column(idc).to_pylist()], np.int64)
            arr = t.column(vc).combine_chunks()
            v = np.asarray(arr.flatten() if hasattr(arr, "flatten") else arr.values, np.float32).reshape(-1, d)
            if ids.min() < 0 or ids.max() >= n:
                raise SystemExit(f"id out of range: {ids.min()}..{ids.max()}")
            if seen[ids].any():
                raise SystemExit("duplicate passage id")
            seen[ids] = True
            mm[ids] = v
    if not seen.all():
        raise SystemExit(f"{int((~seen).sum()):,} passage ids missing")
    mm.flush()
    del mm
    os.replace(tmp, out_path)
    return info


class _FrameUnpickler(pickle.Unpickler):
    """Load the query DataFrame without running arbitrary code: only numpy / pandas classes and plain builtins."""
    SAFE_BUILTINS = {"slice", "range", "complex", "set", "frozenset", "bytearray", "list", "dict", "tuple", "object"}

    def find_class(self, module, name):
        root = module.split(".")[0]
        if root in ("numpy", "pandas", "copyreg", "_codecs", "collections", "datetime") or \
                (module == "builtins" and name in self.SAFE_BUILTINS):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing to load {module}.{name} from the query pickle")


def convert_msmarco_queries(pkl_path, out_path, ids_path):
    """The Pyserini query file is a pickled pandas DataFrame with columns id, text, embedding (float64 holding fp16 values)."""
    import pandas  # noqa: F401  (needed by the unpickler)
    with open(pkl_path, "rb") as f:
        df = _FrameUnpickler(f).load()
    q = np.stack(df["embedding"].values).astype(np.float32)
    write_fbin_blocks(out_path, [q], q.shape[0], q.shape[1])
    with open(ids_path + ".tmp", "w") as f:
        f.writelines(f"{i}\n" for i in df["id"].tolist())
    os.replace(ids_path + ".tmp", ids_path)
    return {"nq": int(q.shape[0]), "dim": int(q.shape[1])}


def convert_msmarco(parquet_dir, query_pkl, out, expect_n=8_841_823, check=True):
    os.makedirs(out, exist_ok=True)
    report = {"dataset": "msmarco"}
    base_p, q_p = os.path.join(out, "base.fbin"), os.path.join(out, "query.fbin")
    if not os.path.exists(base_p):
        files = parquet_files(parquet_dir)
        if not files:
            raise SystemExit(f"no parquet files under {parquet_dir}")
        report["base"] = convert_msmarco_base(files, base_p, expect_n)
    if not os.path.exists(q_p):
        report["query"] = convert_msmarco_queries(query_pkl, q_p, os.path.join(out, "query_ids.txt"))
    finish(out, report, check)


# ------------------------------------------------------------------ ANN-Benchmarks HDF5 (GloVe, SIFT)
def convert_hdf5(src, out, expect_distance=None, block=200_000, check=True):
    import h5py
    os.makedirs(out, exist_ok=True)
    with h5py.File(src, "r") as z:
        dist = z.attrs.get("distance")
        dist = dist.decode() if isinstance(dist, bytes) else dist
        if expect_distance and dist != expect_distance:
            raise SystemExit(f"{src}: distance attribute {dist!r}, expected {expect_distance!r}")
        tr, te = z["train"], z["test"]
        n, d = tr.shape
        base_p, q_p = os.path.join(out, "base.fbin"), os.path.join(out, "query.fbin")
        if not os.path.exists(base_p):
            write_fbin_blocks(base_p, (tr[a:a + block] for a in range(0, n, block)), n, d)
        if not os.path.exists(q_p):
            write_fbin_blocks(q_p, [te[:]], te.shape[0], te.shape[1])
        src_info = {"file": os.path.basename(src), "distance": dist, "train": list(tr.shape), "test": list(te.shape),
                    "train_dtype": str(tr.dtype)}
    finish(out, {"dataset": os.path.basename(os.path.normpath(out)), "source": src_info}, check)


# ------------------------------------------------------------------ self test
def selftest():
    rng = np.random.default_rng(1)
    ran = []
    with tempfile.TemporaryDirectory() as t:
        # T2I: a fake 1B-header file cropped to n rows
        n, d = 300, 200
        X = rng.standard_normal((n, d)).astype(np.float32)
        Q = rng.standard_normal((40, d)).astype(np.float32)
        with open(f"{t}/crop.fbin", "wb") as f:
            write_header(f, 1_000_000_000, d)
            f.write(X.tobytes())
        L.write_fbin(f"{t}/q.fbin", Q)
        convert_t2i(f"{t}/crop.fbin", f"{t}/q.fbin", f"{t}/t2i", n=n, d=d, check=False)
        assert fbin_header(f"{t}/t2i/base.fbin") == (n, d)
        assert np.array_equal(L.read_fbin(f"{t}/t2i/base.fbin"), X)
        assert np.array_equal(L.read_fbin(f"{t}/t2i/query.fbin"), Q)
        ran.append("t2i")

        try:
            import pandas as pd
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as e:
            print(f"skip msmarco selftest ({e})")
        else:
            N, d, nq = 5000, 32, 50
            X = rng.standard_normal((N, d)).astype(np.float32)
            Qv = rng.standard_normal((nq, d)).astype(np.float16).astype(np.float64)
            perm = rng.permutation(N)                       # shuffled ids inside the files test the id placement
            pdir = f"{t}/src/corpus.parquet"                # the distribution's directory name also ends in .parquet
            os.makedirs(pdir)
            for part, idx in enumerate(np.array_split(perm, 3)):
                tbl = pa.table({"docid": [str(i) for i in idx],
                                "vector": pa.array(list(X[idx]), pa.list_(pa.float32()))})
                pq.write_table(tbl, f"{pdir}/chunk_{part}_float.parquet", row_group_size=700)
            qids = rng.choice(10 ** 6, nq, replace=False)
            pd.DataFrame({"id": qids, "text": ["q"] * nq, "embedding": list(Qv)}).to_pickle(f"{t}/src/embedding.pkl")
            convert_msmarco(f"{t}/src", f"{t}/src/embedding.pkl", f"{t}/msmarco", expect_n=N, check=False)
            assert np.array_equal(L.read_fbin(f"{t}/msmarco/base.fbin"), X), "base rows not in passage-id order"
            assert np.array_equal(L.read_fbin(f"{t}/msmarco/query.fbin"), Qv.astype(np.float32))
            assert [int(x) for x in open(f"{t}/msmarco/query_ids.txt")] == qids.tolist()
            # the restricted unpickler must refuse anything that is not numpy / pandas data
            with open(f"{t}/evil.pkl", "wb") as f:
                pickle.dump(os.system, f)
            try:
                with open(f"{t}/evil.pkl", "rb") as f:
                    _FrameUnpickler(f).load()
                raise AssertionError("unsafe pickle was loaded")
            except pickle.UnpicklingError:
                pass
            ran.append("msmarco")

        try:
            import h5py
        except ImportError as e:
            print(f"skip hdf5 selftest ({e})")
        else:
            X = rng.standard_normal((1234, 16))            # float64 on purpose: the converter must write float32
            Q = rng.standard_normal((77, 16)).astype(np.float32)
            with h5py.File(f"{t}/x.hdf5", "w") as z:
                z.attrs["distance"] = "euclidean"
                z.create_dataset("train", data=X)
                z.create_dataset("test", data=Q)
                z.create_dataset("neighbors", data=np.zeros((77, 100), np.int32))
            convert_hdf5(f"{t}/x.hdf5", f"{t}/sift", expect_distance="euclidean", block=500, check=False)
            assert np.array_equal(L.read_fbin(f"{t}/sift/base.fbin"), X.astype(np.float32))
            assert np.array_equal(L.read_fbin(f"{t}/sift/query.fbin"), Q)
            ran.append("hdf5")
    print("selftest ok:", ", ".join(ran))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("t2i")
    a.add_argument("--src", required=True)
    a.add_argument("--query", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--n", type=int, default=L.DATASETS["t2i"]["N"])
    a.add_argument("--d", type=int, default=L.DATASETS["t2i"]["d"])
    a.add_argument("--dataset", default="t2i", choices=("t2i", "t2i100m", "deep"))
    a = sub.add_parser("msmarco")
    a.add_argument("--parquet-dir", required=True)
    a.add_argument("--query-pkl", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--expect-n", type=int, default=L.DATASETS["msmarco"]["N"])
    a = sub.add_parser("hdf5")
    a.add_argument("--src", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--expect-distance")
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "t2i":
        convert_t2i(a.src, a.query, a.out, a.n, a.d, dataset=a.dataset)
    elif a.cmd == "msmarco":
        convert_msmarco(a.parquet_dir, a.query_pkl, a.out, a.expect_n)
    elif a.cmd == "hdf5":
        convert_hdf5(a.src, a.out, a.expect_distance)
    else:
        selftest()


if __name__ == "__main__":
    main()
