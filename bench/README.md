# bench — paper-recipe dataset generator and benchmark runner

Imported verbatim from the scratch workspace `/home/pwwang/p0a/` on 2026-10-06.
These scripts generate the synthetic datasets used for the memory/performance
comparison against the Python implementation, and run one liana method on one
dataset.

## Provenance

| File | Origin | Notes |
|---|---|---|
| `gen_data.py` | `/home/pwwang/p0a/gen_data.py` | dataset generator |
| `run_bench.py` | `/home/pwwang/p0a/run_bench.py` | single method × single dataset runner |
| `run_one.sh` | `/home/pwwang/p0a/run_one.sh` | shell wrapper: `timeout` + `/usr/bin/time -v` |
| `benchmark.py` | `/home/pwwang/p0a/benchmark.py` | **added** — see below |

All four files are byte-identical to their sources (sha256-verified at import).

The sampling recipe itself (`_sample_anndata`, `_sample_resource`) comes from
liana's own manuscript benchmark script (`benchmark.py`): Poisson counts at
sparsity 0.90, 2000 genes, `default_rng(seed=1337)`, CP10K + `log1p`
normalisation, 10 cell types, and an LR resource sampled from the gene-name
product with `random_state=1337`.

### Deviation from the requested copy list

`benchmark.py` was **not** in the requested file list but is imported by
`gen_data.py` (`from benchmark import _sample_anndata, _sample_resource`).
Without it the generator does not run, so it is included. No other files were
added and no source file was modified.

## Hardcoded paths

These scripts were written for the scratch workspace and hardcode absolute
paths. Copying them in as-is means they are **not** relocatable:

- `gen_data.py` writes to `outdir = "/home/pwwang/p0a/data"` (line 11).
- `run_bench.py` reads `/home/pwwang/p0a/data/sc_{n_obs}.h5ad` and
  `resource_{n_obs}.csv`.
- `run_one.sh` does `cd /home/pwwang/p0a` and `source venv/bin/activate`.

So `gen_data.py` overwrites the reference datasets in place. To generate
elsewhere without touching that directory, copy `gen_data.py` + `benchmark.py`
to a scratch dir and edit the `outdir` constant there.

## Usage

```bash
# 1. generate a dataset (n_obs, then optional n_lrs, default 2000)
/home/pwwang/p0a/venv/bin/python bench/gen_data.py 10000

# 2. run one method on it
/home/pwwang/p0a/venv/bin/python bench/run_bench.py \
    --method cellphonedb --n_obs 10000 --n_jobs 4 --n_perms 100

# or via the wrapper, which adds timing/rlimits and logs to p0a/logs/
bench/run_one.sh <label> <method> <n_obs> <n_jobs> <timeout_s> [extra args...]
```

`run_bench.py` prints `WALL_S=<seconds>` (plus `RES_SHAPE`/`RES_INPLACE`); wrap it
in `/usr/bin/time -v` for peak RSS.

## The Rust harness (W4-T3)

`bench/engine_bench/` is a workspace member (`cargo build --release -p
engine-bench`) that runs one liana method end to end on one `.h5ad` + resource
and prints a single line:

```
method=cellphonedb n_obs=50000 n_genes=2000 n_lrs=2000 n_perms=1000 seed=1337 \
threads=4 rows=200000 wall_s=6.68 rss_kb=287660
```

`rows` mirrors `run_bench.py`'s `RES_INPLACE rows=` (both are the reassembled
result's row count — 200,000 on the 50k dataset). `expr_prop`/`min_cells` are
liana 2.0.0's `V.expr_prop = 0.05` / `V.min_cells = 5`, the same defaults the
Python harness leaves in place. `threads` is the rayon pool's effective size
(`RAYON_NUM_THREADS`). `rss_kb` is `VmHWM` from `/proc/self/status` — the
process's peak RSS, the counter `/usr/bin/time -v` prints as "Maximum resident
set size".

`bench/run_engine_bench.sh [outdir] [data_dir]` is the sweep: the 50k dataset
at 1/4/8/32 threads (with a 1-perm run per thread count as the read/prep
reference), the `n_perms` 100-vs-1000 flatness pair, three 4-thread repeats,
the 10k point, and the trimean path (`cellchat`) at 100/1000 perms. Raw output
lands in `target/bench/`.

## The paper suite (W7)

`bench/bench_suite.sh [stage ...]` (stages `t1`–`t5`, default all) is the frozen
benchmark matrix over `rank_aggregate` through three arms:

| arm | what runs |
|---|---|
| `rust` | `bench/engine_bench` — `liana_core::run::Method`, the `liana-rs run` CLI's own dispatch; `RAYON_NUM_THREADS` sets the pool |
| `release` | liana 2.0.0 in the pinned oracle venv, through `bench/run_arm.py` |
| `patched` | the same venv with `PYTHONPATH=/home/pwwang/p0a/patched` (W4/W5 memory patch) |

Every run is serial (one measurement at a time) and wrapped in `/usr/bin/time
-v` for peak RSS + whole-process elapsed; the in-process wall (`wall_s=` /
`WALL_S=`) rides along. Raw stdout/stderr land in `target/bench7/<label>.{out,err}`,
one TSV row per run in `target/bench7/results_<stage>.tsv` (rewritten per stage,
so re-running a stage is idempotent), and the exact command of every row in
`target/bench7/cmds_<stage>.tsv`.

`bench/run_arm.py` is `run_bench.py`'s sibling for the Python arms: same
measurement contract, but it takes `--h5ad`/`--resource` paths instead of
hardcoding the 2000-LR pair — which is what the memory-law sweep (truncated
resources), the law's 10k anchors and the 4,620-LR consensus run need.

`bench/collect_results.py` reads the TSVs plus the box's versions and checksums
and writes the committed manifest **`bench/results.json`**; `bench/RESULTS.md`
is the human table generated from it (the paper's source).

## Verification (W1-A)

The 10k dataset was regenerated from these scripts into a temp directory and
compared against the reference:

| Check | Result |
|---|---|
| `sc_10000.h5ad` sha256 vs `/home/pwwang/p0a/data/sc_10000.h5ad` | **byte-identical** (`f5a41869…5757d`) |
| `resource_10000.csv` sha256 vs reference | **byte-identical** (`dbb4bba4…28e7d`) |
| `manifest.json` fields (shape, nnz, nnz_frac, n_celltypes, size_mb) | **all match** |

The reference datasets in `/home/pwwang/p0a/data/` were not modified; the
regeneration wrote to a temp dir. See `ops/logs/w1a-report.md` (T4) for the
exact commands.

`manifest.json` itself carries these structural fingerprints (not checksums)
per dataset — `sc_1000`, `sc_10000`, `sc_50000`, `sc_100000` — and lives at
`<phase0-workspace>/data_manifest/manifest.json`.
