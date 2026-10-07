#!/usr/bin/env python3
"""Dump reference values for the h5ad inputs of `liana-core::io::read_h5ad`.

Every hash is over UTF-8 text with an explicitly documented format (also stored
in the JSON as `hash_format`), so the Rust reader reproduces it by writing the
same stream:

- `*_names_sha256`      — one name per line, `\\n`-terminated
- `var_sums_sha256`     — per-var sums as `"{sum:.6f}"` lines, in `var_names` order
- `labels_sha256`       — per-cell label *name*, one per line

The per-var sums are accumulated in `float32` in CSR order, which is
bit-identical to scipy's `csr_matrix.sum(axis=0)` for a float32 matrix (verified
here, see `sums_match_scipy`). `x_f32_sum_scipy` records scipy's own `X.sum()`,
which uses numpy's pairwise summation and is *not* reproduced by the naive
order — kept only as a documented reference value.

Run with the oracle venv interpreter (liana 2.0.0 / anndata / scipy):

    /home/pwwang/p0a/venv/bin/python scripts/dump_io_ref.py
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib

import anndata as ad
import h5py
import numpy as np
from scipy.sparse import csr_matrix

REPO_ROOT = pathlib.Path(os.environ.get("REPO_ROOT", pathlib.Path(__file__).resolve().parents[1]))
OUT_DIR = REPO_ROOT / "testdata" / "io_ref"

HASH_FORMAT = {
    "names": 'one name per line, "\\n"-terminated, UTF-8',
    "var_sums": 'float32 sums formatted "{sum:.6f}", one per line in var_names order',
    "labels": 'the label *name* of each cell, one per line, cell order',
}

FILES = [
    {
        "name": "synthetic",
        "path": REPO_ROOT / "testdata" / "fixtures" / "synthetic.h5ad",
        "label_key": "cell_type",
    },
    {
        "name": "sc_10000",
        "path": pathlib.Path("/home/pwwang/p0a/data/sc_10000.h5ad"),
        "label_key": "cell_type",
    },
]


def sha256_text(lines: list[str]) -> str:
    return hashlib.sha256("".join(line + "\n" for line in lines).encode()).hexdigest()


def x_encoding(path: pathlib.Path) -> str:
    """`dense` when `X` is a dataset, else the group's `encoding-type`."""
    with h5py.File(path, "r") as f:
        obj = f["X"]
        if isinstance(obj, h5py.Dataset):
            return "dense"
        return str(obj.attrs["encoding-type"])


def var_sums_f32(x: csr_matrix) -> np.ndarray:
    """scipy's `sum(axis=0)`, reproduced as an explicit row-order f32 accumulation."""
    out = np.zeros(x.shape[1], dtype=np.float32)
    for i in range(x.shape[0]):
        lo, hi = x.indptr[i], x.indptr[i + 1]
        np.add.at(out, x.indices[lo:hi], x.data[lo:hi])
    return out


def sequential_f32_sum(x: csr_matrix) -> float:
    """Plain left-to-right f32 accumulation over `x.data` — what a naive port does."""
    acc = np.float32(0.0)
    for v in x.data:
        acc += v
    return float(acc)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for spec in FILES:
        path: pathlib.Path = spec["path"]
        adata = ad.read_h5ad(path)
        x = csr_matrix(adata.X).astype(np.float32)
        x.sum_duplicates()

        sums = var_sums_f32(x)
        scipy_sums = np.asarray(x.sum(axis=0)).ravel().astype(np.float32)
        assert np.array_equal(sums, scipy_sums), "f32 row order != scipy sum(axis=0)"

        labels = adata.obs[spec["label_key"]]
        assert labels.dtype.name == "category", f"{spec['label_key']} is not categorical"
        label_names = [str(c) for c in labels.cat.categories]
        label_codes = labels.cat.codes.to_numpy()
        assert (label_codes >= 0).all(), "null label code"
        counts = {str(k): int(v) for k, v in labels.value_counts().sort_index().items()}

        obs_spatial = adata.obsm["spatial"] if "spatial" in adata.obsm else None
        payload = {
            "name": spec["name"],
            "file": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
            "sha256_file": hashlib.sha256(path.read_bytes()).hexdigest(),
            "hash_format": HASH_FORMAT,
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
            "x_encoding": x_encoding(path),
            "x_raw_dtype": str(adata.X.dtype),
            "x_f32_nnz": int(x.nnz),
            "x_f32_sum_sequential": sequential_f32_sum(x),
            "x_f32_sum_scipy": float(np.asarray(x.sum()).ravel()[0]),
            "var_sums_sha256": sha256_text([f"{v:.6f}" for v in sums]),
            "label_key": spec["label_key"],
            "label_names": label_names,
            "label_counts": counts,
            "labels_sha256": sha256_text([label_names[c] for c in label_codes]),
            "var_names_sha256": sha256_text([str(v) for v in adata.var_names]),
            "obs_names_sha256": sha256_text([str(v) for v in adata.obs_names]),
            "obsm_spatial": None
            if obs_spatial is None
            else {
                "n_cols": int(obs_spatial.shape[1]),
                "dtype": str(obs_spatial.dtype),
                "first": [int(v) for v in obs_spatial[0]],
            },
        }
        out = OUT_DIR / f"{spec['name']}.json"
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"{spec['name']}: {payload['n_obs']}x{payload['n_vars']} nnz={payload['x_f32_nnz']} "
              f"enc={payload['x_encoding']}/{payload['x_raw_dtype']} "
              f"labels={payload['label_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
