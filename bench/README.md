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
`/mnt/d/Programs/hermes/profiles/work/cache/scratch/p0a/data_manifest/manifest.json`.
