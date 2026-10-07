# Real-data fixtures (G2)

Not committed (size); `scripts/real_g2.sh` regenerates everything here.

## `anndata/kang.h5ad` — the source dataset

Kang et al., 2018 (GSE96583): ~25k IFN-β-stimulated PBMCs, 8 cell types, as
distributed by `liana.datasets.kang_2018` (LIANA+ `datasets/registry.yaml`).

| Field | Value |
|---|---|
| URL | `https://exampledata.scverse.org/liana/kang.h5ad` (fallback: figshare `34464122`) |
| sha256 | `e6a5adac64dcdeb36eaba27db49b63e0c64bb0ed4a64c6705971506b41c39830` |
| Shape | 24 673 cells × 15 706 genes, `X` = raw counts (float32, CSC) |

```python
import scanpy as sc
sc.settings.datasetdir = "testdata/real"      # keeps the cache inside the repo
import liana as li
adata = li.ds.kang_2018()                     # -> testdata/real/anndata/kang.h5ad
```

## `kang_lognorm.h5ad` — the G2 input

`li.ds.kang_2018()` restores `X` from `layers["counts"]` and derives
`obs["cell_abbr"]`; the G2 input adds the standard log-normalisation:

```python
adata.X = adata.layers["counts"].copy()
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
adata.write_h5ad("testdata/real/kang_lognorm.h5ad", compression="gzip")
```

| Field | Value |
|---|---|
| sha256 | `eacdf3a22459691dfab97f0b16f97d9f5792997835cb3fc8c4310b25dc234754` |
| Labels | `obs["cell_abbr"]` — CD4T, CD14, B, NK, CD8T, FGR3, DCs, Mega |
| Written by | liana 2.0.0's own loader + scanpy 1.12.4 |

## `expected/`

The pinned oracle's result CSVs for this input — `kang__<method>__p<N>.csv`,
nine methods, `n_perms` ∈ {100, 1000} (the script's `G2_N_PERMS`), seed 1337,
consensus resource, `n_jobs=1` — as written by `scripts/real_g2.sh`, the
reference side of the G2 differential.
