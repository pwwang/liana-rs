# Parity fixtures — provenance and classification

Vendored verbatim from [`scverse/liana`](https://github.com/scverse/liana) `tests/data/`
(BSD-3-Clause, same license as this project).

## Provenance

| Field | Value |
|---|---|
| Source repo | `https://github.com/scverse/liana` |
| Source path | `tests/data/` |
| Fetched from ref | `main` |
| `main` HEAD at fetch time | `08590dbd45e90576fb0a8dcb9d1dbb139900bb34` |
| Project parity target | tag `V2.0.0` = `c59472ccc9de8360dbbf5016db75f8abde08dd3e` |
| Fetch date | 2026-10-06 |

**Drift check:** all four files were re-fetched from the pinned `V2.0.0` commit
(`raw.githubusercontent.com/scverse/liana/c59472c.../tests/data/<name>`) and their
sha256 digests match the `main`-fetched copies **byte-for-byte**. The fixture set has
not drifted between `main` and `V2.0.0`, so vendoring from `main` is safe.

`testdata/checksums.txt` holds the digests in `sha256sum -c` format.

## Classification

Determined by reading each file's content **and** the test code that consumes it
(`tests/method/sc/`, `tests/method/sp/`, `tests/conftest.py` at the pinned commit).

| File | Kind | Consumed by | Content |
|---|---|---|---|
| `all_defaults.csv` | **golden output** | `tests/method/sc/test_liana_pipe.py::test_liana_pipe_defaults` | 1658 × 23 |
| `not_defaults.csv` | **golden output** | `tests/method/sc/test_liana_pipe.py::test_liana_pipe_not_defaults` | 4200 × 26 |
| `aggregate_rank_rest.csv` | **golden output** | `tests/method/sc/test_rank_aggregate.py::test_aggregate_res` | 1658 × 19 |
| `synthetic.h5ad` | **input** | `tests/method/sp/test_misty.py::adata` (spatial) | AnnData 4205 × 11 |

`conftest.py` describes `tests/data` as holding "the tests' inputs and expected outputs".

### Golden outputs — inputs are NOT vendored

None of the three CSVs' inputs live in `tests/data/`. Each is an *expected output*
produced by liana's own test suite from an input that is constructed at test time.
Regeneration recipes, verbatim from the pinned commit:

**`all_defaults.csv`** — output of the internal `liana.method.sc._liana_pipe.liana_pipe`:
```python
from scanpy.datasets import pbmc68k_reduced
adata = pbmc68k_reduced()
adata.X = adata.raw.to_adata().X.copy()   # log-norm into .X (use_raw=False path)
liana_pipe(adata, groupby="bulk_labels", resource_name="consensus",
           expr_prop=0.05, min_cells=5, de_method="t-test", base=np.e,
           n_perms=1000, seed=1337, use_raw=False, n_jobs=1, supp_columns=[])
```
Contains `liana_pipe` intermediates (`prop_min`, `ligand_cdf`, `receptor_cdf`,
`ligand_means_sums`, …), not a public `li.mt.*` result table.

**`not_defaults.csv`** — same entry point, non-default parameters:
```python
liana_pipe(adata, groupby="bulk_labels", resource_name="consensus",
           expr_prop=0.2, min_cells=5, de_method="wilcoxon", supp_columns=["ligand_pvals", "receptor_pvals"],
           return_all_lrs=True, n_perms=1000, seed=1337, use_raw=False, n_jobs=1)
```
Adds `ligand_pvals` / `receptor_pvals` / `lrs_to_keep` columns.

**`aggregate_rank_rest.csv`** — expected output of the public `li.mt.rank_aggregate`:
```python
from liana.datasets import generate_toy_adata
toy_adata = generate_toy_adata()          # pbmc68k_reduced + `sample`/`case` obs columns
rank_aggregate(toy_adata, groupby="bulk_labels", n_perms=2, seed=1337, inplace=False, n_jobs=1)
```

Both input constructors ship inside liana (`liana.datasets.generate_toy_adata`) or scanpy
(`scanpy.datasets.pbmc68k_reduced`, which downloads/caches on first use), so all three
golden outputs are regenerable without the liana git checkout.

**Verified (W1-A):** all three were regenerated from the pinned oracle with the recipes
above and matched the vendored CSVs under liana's own test tolerances
(`assert_frame_equal`, `rtol=1e-3` for the two `liana_pipe` tables, `rtol=1e-4`/`atol=1e-6`
for `aggregate_rank_rest`). The inputs are therefore available offline, and these three
CSVs are usable as true input→output golden pairs. See `ops/logs/w1a-report.md` (T3) for
the exact commands.

They are **not** wired into `scripts/oracle.sh`: that script's contract covers fixtures
vendored as *input files* under `testdata/fixtures/`. Adopting the reconstructed
`pbmc68k`/`toy_adata` inputs is a decision for the workstream that owns the Rust-side
parity tests.

### Input — `synthetic.h5ad`

A genuine **input** AnnData: 4205 cells × 11 genes (`ECM`, `ligA`, `ligB`, …),
`obs["cell_type"]` ∈ {`A`, `B`}, `obsm["spatial"]` coordinates, no `.var` columns,
dense `float64` `.X`, single (empty-named) layer.

Note it is consumed **only** by the *spatial* misty test, which subsamples it to 100
cells (`sc.pp.subsample(adata, n_obs=100)`). No single-cell test uses it. It is still a
valid generic AnnData for exercising the single-cell methods, and is the only vendored
file that can be fed to `li.mt.<method>` as-is.

## License

liana is BSD-3-Clause; these fixtures are redistributed under the same terms. See the
repository `LICENSE`. Attribution: <https://github.com/scverse/liana>.
