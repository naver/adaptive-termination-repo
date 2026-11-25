# Each Shard Knows When to Stop: reproducibility package

Code, scripts and results for the paper *"Each Shard Knows When to Stop: Measuring and Reducing the Cost of
Hash-Sharded HNSW Search"* (arXiv link to be added). The paper measures what an HNSW index loses when a collection is
split into shards it did not choose, and how much a per-shard stopping rule with a calibrated distance threshold
recovers.

| Directory | Contents |
|---|---|
| `faiss/` | patches to Faiss and the script that builds it, plus a Dockerfile |
| `bench/` | C++ driver for sharded HNSW search |
| `data/` | download and conversion of the public datasets, exact ground truth, shard indexes; no dataset is redistributed |
| `experiments/` | experiment drivers; settings in `config.sh` |
| `lib/` | shared Python code |
| `figures/` | the data figures and a check of the main numbers against reference values |
| `results/` | the aggregated measurements of the original runs (the per-query dumps are not shipped) |
| `smoke/` | smoke test |

Each script describes its inputs, outputs and options in its header.

## Quick check

**From the shipped results** (no data, no compiler; seconds):

```bash
pip install -r requirements.txt
smoke/smoke.sh
```

This draws the five data figures, recomputes the main numbers from the shipped results and compares each with a
reference value (`figures/text_numbers.py` lists them, and the few it cannot recompute), and fails if one of them
differs.

**The whole pipeline on a small synthetic dataset**:

```bash
docker build -f faiss/Dockerfile --build-arg OPT_LEVEL=generic -t pef-bench .     # from this directory
docker run --rm -v "$PWD":/repro pef-bench smoke/smoke.sh --full
```

This runs the main stages on the synthetic set (ground truth to the bootstrap verdicts, the approximate ground truth,
the Patience comparison and calibration before serving) and checks the pipeline's invariants (listed in `smoke/check_synth.py` and `smoke/check_corpus.py`).
Without Docker, [build](#build) the driver first.

**The real experiments**: [Data](#data) and [Running the experiments](#running-the-experiments). Plan for days of
machine time and several hundred GB of disk.

## Requirements

- Linux x86-64 (the smoke test also runs on arm64)
- GCC >= 11, CMake >= 3.24, OpenMP, OpenBLAS
- Python >= 3.9 with `requirements.txt`
- Memory and disk per dataset: [`data/DATASETS.md`](data/DATASETS.md)

CPU times depend on the machine. Distance computations, the main metric, do not, but another SIMD level or a
rebuilt index (builds are multithreaded) can change them slightly.

## Build

```bash
faiss/build_faiss.sh        # clones the pinned Faiss release, applies faiss/patches/, builds
bench/build_bench.sh        # compiles bench/bench.cpp against it -> _build/bin/bench
```

`OPT_LEVEL=avx512` (default, the build of the original runs), `avx2` or `generic` selects the Faiss SIMD level. Every search runs on
Faiss 1.15.0 with three patches: `0001` adds the stopping test, the rank it compares against (k_i for adaptive
termination, K for rank-K termination) and the bookkeeping of the arrival-time oracle; `0002` records the traversal
that calibration replays; `0003` adds queue saturation (Patience), the baseline. The patches are switched off by
default, so an unmodified unfiltered call behaves like stock Faiss; `0001` also changes filtered search on the
unbounded-queue path, which no experiment uses. What each patch adds is in its header.

## Data

```bash
data/prepare_all.sh glove sift
data/prepare_all.sh t2i msmarco
data/prepare_all.sh deep t2i100m
```

Each run downloads, converts, computes exact ground truth and builds every shard index. Sources, terms, sizes,
checksums and the query split are in [`data/DATASETS.md`](data/DATASETS.md). No dataset is included in this
repository; each is subject to its own license and terms, and MS MARCO may be used for non-commercial research
purposes only (see [License](#license)).

## Running the experiments

```bash
run_all.sh                                     # build, data, all stages, figures (days)
run_all.sh --steps core,extras glove sift      # e.g. the main experiments on the smaller datasets only
```

or stage by stage (each skips outputs that already exist, so a rerun resumes):

| Stage | Command |
|---|---|
| core | `experiments/run_core.sh <ds>` |
| extras | `experiments/run_extras.sh [oracle] [shard] [approx] [approxjudge] [kaxis] [ncalib] [transfer] [rank100] [patience]` |
| cpu | `experiments/run_cpu.sh [t2i] [msmarco]` |
| before serving | `experiments/run_corpus.sh <ds>`, `experiments/corpus_img.sh` |
| figures | `RESULTS=work/results figures/make_all.sh` |

Settings and their defaults are in `experiments/config.sh`. Its locations, tools and run settings (the variables from
`WORK` to `K`, such as `RESULTS`, `BIN`, `JOBS`, `THREADS` and `PIN`) can be overridden from the environment. The experiment
parameters are fixed in that file: the HNSW build parameters (`M_HNSW`, `EFC_HNSW`), the ef limits (`E_LIMITS`), the
calibration grid (`CALIB_GAMMAS`) and the replay grid for the rank comparison at K = 100 (`RANK_GAMMAS`);
`run_core.sh` takes other ef limits from `CORE_E`. New results go to `work/results`, and the shipped `results/` is
never written. Run CPU-time sessions with `JOBS=1` on an otherwise idle machine.

The approximate ground truth takes the true neighbors of the calibration queries from one search at
ef_gt = min(1000, 4E) in every shard. `run_extras.sh approx` calibrates every configuration again against it;
`run_extras.sh approxjudge` then judges again, on the same test queries, the configurations in which it selects another
gamma_i in at least one shard (`experiments/approx_verdicts.py`; the shipped result is
`results/approx_verdicts/verdicts.json`). In the other configurations both ground truths give the same gamma_i, so their
verdicts in `results/verdicts/bootstrap.json` stand.

## Figures and numbers

`figures/make_all.sh` works from `RESULTS` (default: the shipped `results/`):

- `figures/paper_figs.py` draws the five data figures as PDF: `fig-waste`, `fig-overhead-same-recall`,
  `fig-adaptive-savings-both`, `fig-shard-tail` and `fig-verdicts`.
- `figures/text_numbers.py` recomputes the main numbers and compares each with a reference value (a number, or a
  numeric condition such as `8--12`): among them the deployed saving at every E and S, the per-S cost of the
  calibration table and the per-S saving of the comparison rank.
- `lib/bonferroni.py` turns the per-configuration verdicts into one statement at a joint 95% over all configurations of
  an arm (Bonferroni). With the per-query dumps under `DUMPS_DIR` it bootstraps each configuration at the corrected
  level; without them (`--normal-only`, and for configurations whose dumps are absent) it widens the 95% intervals of
  `verdicts/bootstrap.json` by the normal approximation. `text_numbers.py` checks the resulting counts.

`fig-shard-tail` and the share of shard searches that ran to the ef limit read the per-query dumps of a run when
`DUMPS_DIR` is set and the rounded values in `results/derived/` otherwise. A few numbers need dumps that
`results/derived/` does not summarize; `text_numbers.py` lists them as unverified. The layout of `results/` is described in
`figures/data.py`.

Names used in the files:

| Name | Meaning |
|---|---|
| `nat`, `fixed`, gamma = 0 | fixed ef |
| `rule`, `adaptive`, `gcal` | adaptive termination |
| `dabs` | rank-K termination |
| `approx` | adaptive termination with the gamma_i chosen on the approximate ground truth |
| `ndis` | distance computations |
| `corpus`, `p125`, `p250` | calibrated on held-out shard rows, or on the first 125 / 250 calibration queries |
| `s1g` | the gamma calibrated on a single index (S = 1), applied unchanged to every shard |
| `patience` | queue saturation (Patience); queue `ef` = the E nearest points, `ki` = the shard's k_i results |

## What was verified

- From the shipped `results/`, `smoke/smoke.sh` draws the five data figures and reproduces every number `figures/text_numbers.py` can recompute. `fig-shard-tail` and the ef-limit
  share come from the rounded values in `results/derived/`. With the per-query dumps of the original runs under `DUMPS_DIR`,
  `text_numbers.py` also reproduces the numbers that need them.
- `experiments/approx_verdicts.py judge`, run on the per-query dumps of the original runs, reproduces
  `results/approx_verdicts/verdicts.json` in every field.
- `lib/bootstrap.py`, run on the per-query dumps of the arm `s1g`, reproduces its 24 rows in
  `results/verdicts/bootstrap.json` in every field. `lib/bonferroni.py` with B = 200,000 on the per-query dumps
  (48 configurations; the 12 at 100M from the normal approximation) gives the same counts and the same four undecided
  configurations as `--normal-only` on the shipped results, with interval ends within 0.01pp.
- The patches apply, Faiss and the driver build, and `smoke/smoke.sh --full` passes its invariants in the Docker image
  (`OPT_LEVEL=generic`, arm64).
- On the synthetic dataset the approximate ground truth selected the same gamma_i in every shard in our runs, so
  `smoke/smoke.sh --full` has nothing to judge again. With one shard's gamma_i changed by hand, `dump_cell.sh` arm
  `approx` and `experiments/approx_verdicts.py judge` ran on the synthetic results, and the rows it recomputes for the
  exact ground truth equal those of `verdicts/bootstrap.json`.
- Patience (`0003-patience.patch`): its replay equals direct runs to the printed digits (`smoke/smoke.sh --full`). With
  it off, recall, ndis, dumps and traversal records equal those of a build without the patch (checked once on the
  synthetic dataset). `results/patience/` are the original runs.
- **Not rebuilt for this package**: `avx512` (the build of the original runs) and `avx2`.
- **Not rerun for this package**: the experiments on the real datasets; `results/` are the original runs.

## License

```
adaptive-termination-repo
Copyright (c) 2026-present NAVER Corp.
CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
```

- This repository implements methods covered by pending patent applications owned by NAVER Corp.
- The files in `faiss/patches/` modify [Faiss](https://github.com/facebookresearch/faiss) and contain excerpts of it.
  The Faiss excerpts stay under the MIT License (Copyright (c) Meta Platforms, Inc. and affiliates, see
  [`NOTICE`](NOTICE)); NAVER's modifications in them are under CC BY-NC-SA 4.0 like the rest of the package.
  Faiss itself is not included; `faiss/build_faiss.sh` clones it and applies the patches.
- This repository does not distribute any dataset. Datasets are downloaded from their original sources and are
  subject to their respective licenses and terms; [`data/DATASETS.md`](data/DATASETS.md#licenses-and-terms) lists the
  source, creator and license of each. **MS MARCO is provided by Microsoft for non-commercial research purposes only;
  users are responsible for complying with its terms.**
- The Python dependencies are installed with pip and not included.

## Citation

Please cite the paper (the arXiv identifier will be added once available), and also Faiss and the datasets you use.
The Patience baseline (`faiss/patches/0003-patience.patch`) implements the rule of Teofili and Lin, "Patience in
Proximity: A Simple Early Termination Strategy for HNSW Graph Traversal in Approximate k-Nearest Neighbor Search"
(ECIR 2025), with the defaults of Lucene's `PatienceKnnVectorQuery`.
Machine-readable metadata: [`CITATION.cff`](CITATION.cff).

```bibtex
@misc{shin2026eachshard,
  title  = {Each Shard Knows When to Stop: Measuring and Reducing the Cost of Hash-Sharded HNSW Search},
  author = {Shin, Woojin and Choe, Gyusik},
  year   = {2026},
  note   = {Preprint. Code: https://github.com/naver/adaptive-termination-repo}
}
```

## Issues and contributions

This package is kept in the state that produced the shipped results. Bug reports and questions are welcome as GitHub
issues (please include the command, `OPT_LEVEL`, the CPU and the output of `smoke/smoke.sh`). Pull requests that fix a
reproducibility problem are welcome; changes to the method belong in a separate project.
