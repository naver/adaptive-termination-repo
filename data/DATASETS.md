# Datasets

The experiments use six datasets from five public sources (T2I 10M and T2I 100M are two crops of one collection). This
directory downloads them from their original distributors, converts them to
one flat format, computes exact ground truth and builds the shard indexes. **No data is redistributed with this
package**; each dataset stays under its own license and terms ([Licenses and terms](#licenses-and-terms)).

| id | Name | N (indexed) | d | Queries | Distance used | Source |
|---|---|---|---|---|---|---|
| `t2i` | T2I 10M | 10,000,000 (first 10M of 1B) | 200 | 100,000 (public set) | unit-normalized, squared L2 (= cosine order) | Big-ANN benchmark (Yandex Text-to-Image-1B) |
| `msmarco` | MS MARCO | 8,841,823 | 1024 | 6,980 (dev small) | unit-normalized, squared L2 (= cosine order) | Anserini / Pyserini (Cohere embed-english-v3.0) |
| `glove` | GloVe | 1,183,514 | 200 | 10,000 | unit-normalized, squared L2 (= cosine order) | ANN-Benchmarks `glove-200-angular` |
| `sift` | SIFT | 1,000,000 | 128 | 10,000 | raw L2 | ANN-Benchmarks `sift-128-euclidean` |
| `deep` | DEEP 100M | 100,000,000 (first 100M of 1B) | 96 | 10,000 (public set) | raw L2 | Big-ANN benchmark (Yandex DEEP1B) |
| `t2i100m` | T2I 100M | 100,000,000 (first 100M of 1B) | 200 | 100,000 (public set, same file as `t2i`) | unit-normalized, squared L2 (= cosine order) | Big-ANN benchmark (Yandex Text-to-Image-1B) |

The two 100M sets are a scale extension: only adaptive termination against fixed ef is run on them (S = 5, 15, at the
ef limit 200 for the savings; no rank-K termination, no same-recall matching), and the data scripts process them only
when named (`data/prepare_all.sh deep t2i100m`).

`make_synthetic.py` also writes a tiny synthetic set (`synth`, 20,000 x 32, 2,000 queries) for the smoke test. It is
**not part of the experiments**.

## Licenses and terms

This repository does not distribute any dataset. Each one is downloaded from the URL below and stays under the license
or terms of its original source; users are responsible for complying with them. The per-dataset sections give the
exact files, the wording of each license and the date it was checked.

| Dataset | Creator | Original source | Downloaded from | License / terms |
|---|---|---|---|---|
| T2I 10M, T2I 100M | Yandex (Text-to-Image-1B) | https://big-ann-benchmarks.com/neurips21.html | `https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/` | CC BY 4.0 |
| DEEP 100M | Yandex (DEEP1B; Babenko and Lempitsky, CVPR 2016) | https://big-ann-benchmarks.com/neurips21.html | `https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/` | CC BY 4.0 |
| MS MARCO | Microsoft (MS MARCO v1 passage text); vectors encoded with Cohere embed-english-v3.0 and published by Anserini / Pyserini (University of Waterloo) | https://microsoft.github.io/msmarco/ | `https://rgw.cs.uwaterloo.ca/pyserini/data/`, `https://github.com/castorini/pyserini-data` | **Non-commercial research purposes only** (Microsoft MS MARCO terms; the vectors are treated under the same terms) |
| GloVe | Pennington, Socher and Manning (Stanford NLP Group) | https://nlp.stanford.edu/projects/glove/ | `http://ann-benchmarks.com/glove-200-angular.hdf5` | Open Data Commons PDDL 1.0 |
| SIFT | Amsaleg and Jegou (INRIA, TEXMEX corpus ANN_SIFT1M) | http://corpus-texmex.irisa.fr/ | `http://ann-benchmarks.com/sift-128-euclidean.hdf5` | CC0 1.0 (the authors waived all copyright and related rights); subject to the conditions of the original source (INRIA TEXMEX) |

## Quick start

```bash
data/prepare_all.sh glove sift              # the two 1M datasets: about 1.5 GB download, ~11 GB disk
data/prepare_all.sh t2i                     # ~8 GB download, ~75 GB disk
data/prepare_all.sh msmarco                 # ~17 GB download, ~270-300 GB disk, 128 GB RAM
data/prepare_all.sh deep                    # ~38 GB download, ~200 GB disk (see "100M sets" below)
data/prepare_all.sh t2i100m                 # ~80 GB download, ~330 GB disk
data/prepare_all.sh --steps gt,index synth  # after make_synthetic.py (or just: data/prepare_all.sh synth)
```

The steps can also be run one by one; each one skips outputs that already exist, so rerunning a command resumes it.

| Step | Script | Writes |
|---|---|---|
| download | `download.sh DS...` | `$WORK/download/<ds>/` (resumable `curl`; sizes and published MD5s are checked) |
| convert | `convert.sh DS...` (`convert.py`) | `$RAW_DIR/<ds>/base.fbin`, `query.fbin`, `prep_report.json` (+ `query_ids.txt` for MS MARCO) |
| ground truth | `ground_truth.sh DS...` (`gt_slice.py`) | `$IDX_DIR/<ds>_N<N>/gt300.ivec`, `gt300_proxy.ivec`, `gt300_q1500.ivec` |
| index | `build_indexes.sh DS [S...]` | `$IDX_DIR/<ds>_N<N>/S<S>/shard<s>.{index,sel}`, `build.json`, `.done` |
| optional check | `verify_msmarco_qrels.py` | prints RR@10 / R@100 of the exact neighbors against the MS MARCO dev qrels |

Locations come from `experiments/config.sh` (`WORK`, `RAW_DIR`, `IDX_DIR`; downloads go to `DL_DIR`, default
`$WORK/download`). Ground truth and index builds need the benchmark binary (`bench/build_bench.sh`); `THREADS` sets
their OpenMP/OpenBLAS threads. Python needs numpy, plus pyarrow and pandas for MS MARCO and h5py for GloVe and SIFT.
`python3 data/convert.py selftest` runs the converters on small fabricated inputs.

## File formats and conventions

- `fbin`: int32 n, int32 d, then n x d float32, row major. `ivec`: int32 n, int32 k, then n x k int32.
- Raw files keep the vectors **as published and in their published order** (not normalized). Normalization is applied
  in exactly two places, both dividing each row by its L2 norm: `lib/shard_split.py --normalize` for the indexed rows,
  and `bench gt|search --normalize 1` for queries and the exact ground truth. Index builds do not normalize again.
  This is how the indexes of the original runs were built. SIFT is never normalized.
- Ground truth: exact (brute-force) top-300 over the first N base rows, global row ids, nearest first. Row i of a
  ground-truth file belongs to query `qoff + i` of the run that wrote it.
- Query split (absolute query numbers in `query.fbin`): test block a #0-999, calibration #1000-1499 (500), test
  block b #1500 to the end of the block.

| Ground-truth file | Query rows | T2I, T2I 100M | MS MARCO | GloVe, SIFT, DEEP 100M |
|---|---|---|---|---|
| `gt300.ivec` | #0- | #0-999 (one run, `--nq 1000`) | #0-6979 (all) | #0-9999 (all) |
| `gt300_proxy.ivec` | #1000-1499 | own run, `--qoff 1000 --nq 500` | cut from `gt300.ivec` | cut from `gt300.ivec` |
| `gt300_q1500.ivec` | #1500- | own run, `--qoff 1500 --nq 19000` | 5,480 rows, cut | 8,500 rows, cut |

The benches read the first `nq` rows of `gt300.ivec`, so the longer files of MS MARCO, GloVe and SIFT serve block a
as well. The T2I and 1M splits mirror the commands of the original runs.

## T2I 10M (Yandex Text-to-Image-1B, Big-ANN benchmark)

- Base: first 10,000,000 rows of `https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1B.fbin`
  (800,000,000,008 bytes; header 1,000,000,000 x 200). `download.sh` fetches bytes 0 .. 8 + 10M x 200 x 4 - 1
  (8,000,000,008 bytes) in 8 parallel byte ranges (`T2I_PARTS`); `convert.py t2i` rewrites the header to 10,000,000 rows
  and moves the file to `$RAW_DIR/t2i/base.fbin`. This is the same crop the Big-ANN benchmark code makes for its 10M
  subset (`benchmark/datasets.py`: download the first `8 + d * nb * 4` bytes, then overwrite `header[0] = nb`).
- Queries: `https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin`
  (80,000,008 bytes; 100,000 x 200), used in file order.
- Both URLs are the ones in the Big-ANN benchmark source (`Text2Image1B`: `base_url` + `ds_fn` / `qs_fn`) and answered
  HTTP 200 with the sizes above and `Accept-Ranges: bytes` on 2026-10-18.
- Checksums: none published. Server ETags (multipart, not MD5): base `273267a0a489f31e116ec3a287caa11b-5961`, queries
  `eee1ed2f5db70c05355c742368d65521-10`, both last modified 2021-05-01. `download.sh` records the MD5s of what it
  fetched in `$WORK/download/t2i/MD5SUMS.local` so two installations can compare.
- Distance: the benchmark uses inner product. Corpus norms are not uniform (about 0.81-0.99 on the first rows; queries
  1.0), so the experiments normalize both sides and uses cosine order; its ground truth therefore differs slightly from the
  official inner-product ground truth, which is not used.
- License: CC BY 4.0 (the "Release terms" column of the NeurIPS'21 Big-ANN dataset table,
  https://big-ann-benchmarks.com/neurips21.html, checked 2026-10-18). Cite the Big-ANN benchmark.

## T2I 100M and DEEP 100M (100M sets)

- T2I 100M (`t2i100m`): the first 100,000,000 rows of the same T2I `base.1B.fbin` (bytes 0 .. 8 + 100M x 200 x 4 - 1 =
  80,000,000,008 bytes) and the same public query file; distance, query split and license as T2I 10M above.
- DEEP 100M (`deep`): the first 100,000,000 rows of
  `https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin` (384,000,000,008 bytes; header
  1,000,000,000 x 96), i.e. bytes 0 .. 8 + 100M x 96 x 4 - 1 = 38,400,000,008 bytes, with the header rewritten to
  100,000,000 rows (`convert.py t2i --n 100000000 --d 96 --dataset deep`; the file layout is the same as T2I).
  Queries: `.../DEEP/query.public.10K.fbin` (3,840,008 bytes; 10,000 x 96), in file order, split like GloVe and SIFT
  (#0-999 test block a, #1000-1499 calibration, #1500-9999 test block b). Raw L2, not normalized (the benchmark's
  distance for DEEP is L2).
- Both DEEP URLs answered HTTP 200 with the sizes above and `Accept-Ranges: bytes` on 2026-10-24. Server ETags
  (multipart, not MD5): base `ed22edcea08640670973dde650b852b9-5723`, queries `369d06140c8db325bdb62d621dfa2a67`, both
  last modified 2021-05-01. No checksum is published; `download.sh` records the MD5s it fetched in `MD5SUMS.local`.
- License: DEEP1B and Text-to-Image-1B are both CC BY 4.0 (the "Release terms" column of
  https://big-ann-benchmarks.com/neurips21.html, checked 2026-10-24). Cite the Big-ANN benchmark.
- Size: `base.fbin` 38.4 GB (DEEP) / 80.0 GB (T2I 100M). One S of shard indexes holds the vectors plus about 332 bytes
  per row of graph, about 72 GB (DEEP) / 113 GB (T2I 100M) per S (estimate), and a cell of the experiment loads all
  shards of its S into memory, so plan on 96 GB / 160 GB of RAM per concurrently running cell.

## MS MARCO v1 passage + Cohere embed-english-v3.0

| File | URL | Bytes | MD5 |
|---|---|---|---|
| corpus vectors (tar of 9 parquet files) | `https://rgw.cs.uwaterloo.ca/pyserini/data/msmarco-passage-cohere-embed-english-v3.0.parquet.tar` | 17,040,240,640 | `760dfb5ba9e2b0cc6f7e527e518fef03` (published by Anserini) |
| dev-query vectors (tar.gz with `embedding.pkl`) | `https://github.com/castorini/pyserini-data/raw/main/encoded-queries/query-embedding-cohere-embed-english-v3.0-msmarco-passage-dev-subset-20240216-2154e79.tar.gz` | 22,377,230 | `7dd0026490117e9e6f6acfc110d6ce83` (re-checked 2026-10-18) |
| dev qrels (optional check only) | `https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.small.tsv` | 143,300 | `38a80559a561707ac2ec0f150ecd1e8a` (computed 2026-10-18; not published) |

- Official description of the corpus vectors: the Anserini 1.0.0 regression page
  https://github.com/castorini/anserini/blob/anserini-1.0.0/docs/regressions/regressions-msmarco-v1-passage.cohere-embed-english-v3.0.parquet.flat.cached.md
  (use the tag: the page was removed from the main branch in 2026 when Anserini dropped all "cached query" regressions;
  the files remained downloadable, all three URLs answered HTTP 200 on 2026-10-18). Query vectors are listed in
  Pyserini `pyserini/encoded_query_info.py` (`cohere-embed-english-v3.0-msmarco-passage-dev-subset`).
- Corpus: parquet columns `docid` (string) and `vector` (list<float>, 1024), 8,841,823 rows, ids 0..N-1, norms
  0.9999-1.0001. `convert.py msmarco` writes `base.fbin` in passage-id order (row i = passage i) and checks that every id
  appears exactly once.
- Queries: a pickled pandas DataFrame with columns `id`, `text`, `embedding` (float64 holding float16-exact values),
  6,980 rows. `query.fbin` keeps the DataFrame order; `query_ids.txt` gives the MS MARCO query id of each row. The
  pickle is loaded with a restricted unpickler that accepts only numpy / pandas classes. Checked on the real file
  (2026-10-18): 6,980 x 1024, norms 0.9994-1.0005, all 6,980 ids present in the dev qrels.
- The published vectors have float16 precision, so exact distance ties (near-duplicate passages) occur; the original
  run counted ties between ranks 10/11 and 100/101 in 1 of 6,980 queries each.
- Optional check: `verify_msmarco_qrels.py` scores `gt300.ivec` (all 6,980 rows) against the dev qrels. The original
  run gave RR@10 0.3659 and R@100 0.8941, against 0.3658 / 0.8935 in the Anserini 1.0.0 flat regression on the same
  vectors -- evidence that the vectors are the published ones and that query and passage encodings
  (`search_query` / `search_document`) are paired correctly.
- Terms: "The MS MARCO datasets are intended for non-commercial research purposes only ... and is made available free
  of charge without extending any license or other intellectual property rights" (https://microsoft.github.io/msmarco/,
  checked 2026-10-18). The vectors are derived from MS MARCO text, so treat them under the same terms. Whether Cohere attaches
  separate terms to these embedding outputs was not verified by the authors; check before any use beyond research. Anserini and Pyserini code is Apache-2.0 (we use only their files).
- Cite MS MARCO, Anserini and Pyserini; the embedding model has no paper (name it and the corpus URL in a footnote).

## GloVe 1.18M (ANN-Benchmarks `glove-200-angular`)

- `http://ann-benchmarks.com/glove-200-angular.hdf5` (962,819,488 bytes; HTTP 200, last modified 2024-12-09, checked
  2026-10-18). Base = `train` (1,183,514 x 200), queries = all of `test` (10,000 x 200), both cast to float32.
  The `distance` attribute must be `angular`.
- ANN-Benchmarks builds it from `glove.twitter.27B.200d.txt` (http://nlp.stanford.edu/data/glove.twitter.27B.zip) with
  a fixed train/test split (`sklearn.model_selection.train_test_split(test_size=10000, random_state=1)`).
- Checksum: none published (ETag `c4aecb89b5369bc649bacb0b3ec3be4b-115`, multipart). Recorded locally in
  `MD5SUMS.local`.
- License: GloVe pre-trained vectors are under the Open Data Commons Public Domain Dedication and License (PDDL) v1.0
  (https://nlp.stanford.edu/projects/glove/, checked 2026-10-18). ANN-Benchmarks code is MIT; the HDF5 file states no
  terms of its own beyond its source.

## SIFT 1M (ANN-Benchmarks `sift-128-euclidean`)

- `http://ann-benchmarks.com/sift-128-euclidean.hdf5` (525,128,288 bytes; HTTP 200, last modified 2024-12-09, checked
  2026-10-18). Base = `train` (1,000,000 x 128), queries = all of `test` (10,000 x 128). The `distance` attribute must
  be `euclidean`. Not normalized.
- ANN-Benchmarks builds it from TEXMEX ANN_SIFT1M (`ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz`:
  `sift_base.fvecs` -> train, `sift_query.fvecs` -> test), so the base and queries are the TEXMEX ones in TEXMEX order.
- Checksum: none published (ETag `2bf1ef4c0c3e17031bd64980774eea09-63`, multipart). Recorded locally.
- License: the TEXMEX authors (Amsaleg, Jegou) waived all copyright and related rights under CC0 1.0 ("To the extent
  possible under law ... have waived all copyright", with the CC0 1.0 mark, http://corpus-texmex.irisa.fr/, checked
  2026-11-12); they ask users to cite the
  paper that introduced ANN_SIFT1M (Jegou, Douze, Schmid, TPAMI 2011). ANN-Benchmarks code is MIT.

## Disk, memory and time

Sizes of files are exact; index sizes are estimates (IndexHNSWFlat, M=40: the vectors plus about 332 bytes per row of
graph); times marked "measured" are from the original runs on one socket of an AMD EPYC 9354 (32 cores) with 28 build
threads and 16 ground-truth threads (the build-cost record shipped in `results/buildcost/`; where it holds several
repetitions, the smallest or their range); everything else is an **estimate**.

| | T2I 10M | MS MARCO 8.8M | GloVe 1.18M | SIFT 1M |
|---|---|---|---|---|
| download | 8.0 GB + 80 MB | 17.0 GB tar + 22 MB (+ 17 GB extracted) | 963 MB | 525 MB |
| `base.fbin` / `query.fbin` | 8.0 GB / 80 MB | 36.2 GB / 29 MB | 947 MB / 8 MB | 512 MB / 5 MB |
| indexes, per S (est.) | ~11.3 GB | ~39 GB | ~1.3 GB | ~0.85 GB |
| indexes, all S (est.) | ~57 GB (S = 1, 5, 15, 30, 50) | ~196 GB (5 values) | ~4 GB (S = 1, 5, 15) | ~2.5 GB (3 values) |
| transient during a split (est.) | +8 GB (one normalized copy) | +36 GB | +1 GB | +0.5 GB |
| transient during download | +8 GB (range parts are concatenated) | - | - | - |
| peak disk (est.) | ~75 GB | ~300 GB with downloads kept, ~265 GB with `CLEAN_DOWNLOADS=1` | ~7 GB | ~4 GB |
| peak RAM (est.) | ~20 GB (S=1 build: input + index) | ~80 GB (S=1 build: 36 GB input + 39 GB index); 128 GB recommended | ~3 GB | ~2 GB |
| index build, all S | measured: 21 min wall, 32,900 CPU s (S=1: 376 s wall / 10,104 CPU s ... S=50: 169 s / 4,456) | measured: 69 min wall, 107,000 CPU s (S=1: 861 s / 22,497 ... S=50: 726 s / 18,384) | a few minutes | a few minutes |
| exact ground truth | measured: 500 queries 17-38 s wall / 161-172 CPU s at 16 threads; 19,000 queries ~500 s wall at 28 threads; all 40,500 est. ~20 min | measured: 500 queries 47-132 s wall / 533-553 CPU s at 16 threads; all 6,980 est. ~10-30 min | est. minutes | est. minutes |
| convert | seconds (header rewrite + move) | not recorded; est. 10-30 min (parquet decode) | est. ~1 min | est. ~1 min |

Download times depend on the network and were not recorded systematically (an earlier run fetched 40 GB of the T2I
base in about 6 minutes with 16 parallel ranges). The traversal records and results written by the experiments
(`$TRACE_DIR`, `$RESULTS`) are extra and are described with the experiments.

## Provenance notes and open points

- **T2I base and query order.** The original T2I files were the first 10M rows of `base.1B.fbin` (the original run kept
  the 1B header and passed `--limit 10000000`; rewriting the header, as here, selects the same rows) and the public
  query file in file order. The original notes record that the T2I query file and the first 10M base rows were
  byte-identical to independently obtained copies. Not re-checked here (no multi-GB download).
- **GloVe / SIFT.** The original GloVe `base.fbin` was written from the HDF5 `train` set exactly as here. The script that
  wrote the original SIFT `base.fbin` is not available; the original pipeline checked that its first and last 1,000
  rows equal `train` of `sift-128-euclidean.hdf5` (the middle rows were not compared) before it rewrote the query file
  from all 10,000 `test` rows. Both HDF5 files on the server were last modified 2024-12-09; whether they are
  byte-identical to the copies the original runs used was not re-checked for this package (compare `MD5SUMS.local` with the
  values a fresh download records).
- **MS MARCO ground truth.** The original MS MARCO ground truth was computed by an equivalent Python exhaustive search
  (normalize, then squared L2) and checked against `bench gt`; this package uses `bench gt` for all datasets. With
  float16-precision vectors, exactly tied neighbors may be ordered differently, which can change a few ids at the
  rank boundary of a few queries.
- **Synthetic set.** `synth` is not one of the config.sh datasets: the data scripts read its N from the `base.fbin`
  header, normalize it (override with `NORM=0`), use S = 1, 5, 15 and treat every query after #1499 as block b
  (500 rows by default).
