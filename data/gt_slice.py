# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)

"""Cut row ranges out of a ground-truth ivec (int32 n, int32 k, n*k int32 ids).

Usage: python3 data/gt_slice.py --src gt300.ivec --rows A:B --out gt300_proxy.ivec [--expect-rows R]
    --rows A:B   rows A..B-1 of --src (B may be empty = to the end). Row i of --src is query qoff+i of the run that wrote it.
"""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-rows", type=int)
    a = ap.parse_args()
    n, k = (int(x) for x in np.fromfile(a.src, np.int32, 2))
    if os.path.getsize(a.src) != 8 + n * k * 4:
        raise SystemExit(f"{a.src}: size does not match its header ({n} x {k})")
    lo, hi = a.rows.split(":")
    lo = int(lo)
    hi = n if hi == "" else int(hi)
    if not 0 <= lo < hi <= n:
        raise SystemExit(f"rows {lo}:{hi} outside 0:{n}")
    if a.expect_rows is not None and hi - lo != a.expect_rows:
        raise SystemExit(f"{hi - lo} rows, expected {a.expect_rows}")
    rows = np.fromfile(a.src, np.int32, (hi - lo) * k, offset=8 + lo * k * 4).reshape(hi - lo, k)
    tmp = a.out + ".tmp"
    with open(tmp, "wb") as f:
        np.array(rows.shape, np.int32).tofile(f)
        rows.tofile(f)
    os.replace(tmp, a.out)
    print(f"{a.out}: rows {lo}..{hi - 1} of {a.src} ({hi - lo} x {k})")


if __name__ == "__main__":
    main()
