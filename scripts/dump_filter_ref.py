#!/usr/bin/env python3
"""Dump liana's `expr_prop`/`min_cells` filtering as parity references.

Runs liana's own single-cell pipeline — `prep_check_adata` (cluster drop),
`filter_resource`, `_get_lr` (per-cluster proportions), then
`_filter_reassemble_complexes` — on `synthetic.h5ad` and records, per case, the
surviving labels and the ligand–receptor keys that pass `expr_prop`.

A key is `(source, target, ligand_complex, receptor_complex)`; `prop_min` is the
minimum subunit proportion of that key, over every exploded subunit on both
sides. The kept stream is one line per key, in the order liana yields it:

    "{source}\\t{target}\\t{ligand_complex}\\t{receptor_complex}\\t{prop_min:016x}"

`prop_min` is written as the big-endian IEEE-754 bit pattern of the float64
liana compares against `expr_prop`, so a Rust port pins the arithmetic, not a
rounded decimal.

Writes `testdata/filter_ref/synthetic.json`.

Run with the oracle venv interpreter (liana 2.0.0):

    /home/pwwang/p0a/venv/bin/python scripts/dump_filter_ref.py
"""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import os
import pathlib
import struct

import anndata as ad
import numpy as np
import pandas as pd
from liana._core._constants import CommonColumns as C
from liana._core._constants import PrimaryColumns as P
from liana._core._pipe_utils._pre import filter_resource, prep_check_adata
from liana.method.sc._liana_pipe import _get_lr
from liana.resource._reassemble_complexes import _explode_complexes, _filter_reassemble_complexes

PINNED_VERSION = "2.0.0"
PINNED_COMMIT = "c59472ccc9de8360dbbf5016db75f8abde08dd3e"

REPO_ROOT = pathlib.Path(os.environ.get("REPO_ROOT", pathlib.Path(__file__).resolve().parents[1]))
OUT_DIR = REPO_ROOT / "testdata" / "filter_ref"
ADATA_PATH = REPO_ROOT / "testdata" / "fixtures" / "synthetic.h5ad"
TOY_PATH = REPO_ROOT / "testdata" / "expected" / "synthetic__resource.csv"
GROUPBY = "cell_type"

# A resource with protein complexes, built from synthetic.h5ad's own genes so
# `filter_resource` has something to reject: `ligD`/`protX` are absent, and the
# complexes containing `protZ`/`protW` have a subunit the matrix does not carry.
COMPLEX_PAIRS = [
    ("ECM_ligA", "protE_protF"),
    ("ligB_ligC", "protE"),
    ("ECM", "protF"),
    ("ligD", "protX"),
    ("ligD_protZ", "protF"),
    ("protZ_ligD", "protF_protW"),
]

# `relevant_cols` of the real pipeline, minus the method-specific score columns:
# the filter needs the key columns plus the two proportion columns only.
RELEVANT_COLS = [
    P.source,
    P.target,
    P.ligand,
    P.receptor,
    P.ligand_complex,
    P.receptor_complex,
    C.ligand_props,
    C.receptor_props,
]

# One row per key; `prop_min` is 0 for the keys `return_all_lrs=True` re-appends.
KEY_COLS = [P.source, P.target, P.ligand_complex, P.receptor_complex]


def run_pipeline(adata: ad.AnnData, resource: pd.DataFrame, min_cells: int) -> tuple:
    """liana's `_liana_pipe` up to the (unfiltered) long-format `lr_res`."""
    exploded = _explode_complexes(resource.copy())
    prepped = prep_check_adata(
        adata=adata,
        groupby=GROUPBY,
        min_cells=min_cells,
        block_negatives=True,
        verbose=False,
    )
    exploded = filter_resource(exploded, prepped.var_names)
    entities = np.union1d(np.unique(exploded[P.ligand]), np.unique(exploded[P.receptor]))
    matrix = prepped[:, np.intersect1d(entities, prepped.var.index)]
    lr_res = _get_lr(
        adata=matrix,
        resource=exploded,
        groupby_pairs=None,
        relevant_cols=RELEVANT_COLS,
        mat_mean=None,
        mat_max=None,
        de_method="t-test",
        base=float(np.e),
        verbose=False,
    )
    labels = [str(label) for label in prepped.obs["@label"].cat.categories]
    return prepped, exploded, lr_res, labels


def prop_min_bits(prop_min: float) -> str:
    """The float64 liana compared against `expr_prop`, as big-endian hex bits."""
    return struct.pack(">d", float(prop_min)).hex()


def kept_sha256(frame: pd.DataFrame) -> str:
    payload = "".join(
        f"{source}\t{target}\t{lig_c}\t{rec_c}\t{prop_min_bits(prop_min)}\n"
        for source, target, lig_c, rec_c, prop_min in zip(
            frame[P.source],
            frame[P.target],
            frame[P.ligand_complex],
            frame[P.receptor_complex],
            frame["prop_min"],
            strict=True,
        )
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def case(adata: ad.AnnData, pairs: pd.DataFrame, min_cells: int, expr_prop: float, **extra) -> dict:
    prepped, exploded, lr_res, labels = run_pipeline(adata, pairs, min_cells)
    n_keys = int(len(lr_res.drop_duplicates(subset=KEY_COLS)))
    entry = {
        "min_cells": min_cells,
        "expr_prop": expr_prop,
        "labels": labels,
        "n_cells": int(prepped.n_obs),
        "n_exploded": int(len(exploded)),
        "n_lr_rows": int(len(lr_res)),
        **extra,
    }
    try:
        kept = _filter_reassemble_complexes(
            lr_res=lr_res,
            _key_cols=KEY_COLS,
            complex_cols=[],
            expr_prop=expr_prop,
            return_all_lrs=False,
        )
    except ValueError as error:
        entry["raises"] = "ValueError"
        entry["message"] = str(error)
        return entry

    all_lrs = _filter_reassemble_complexes(
        lr_res=lr_res,
        _key_cols=KEY_COLS,
        complex_cols=[],
        expr_prop=expr_prop,
        return_all_lrs=True,
    )
    assert len(all_lrs) == n_keys, "return_all_lrs should keep one row per key"
    assert int(all_lrs["lrs_to_keep"].sum()) == len(kept), "kept counts disagree"
    entry["n_kept"] = int(len(kept))
    entry["sha256_kept"] = kept_sha256(kept)
    return entry


def main() -> int:
    version = md.version("liana")
    direct_url = md.distribution("liana").read_text("direct_url.json")
    commit = json.loads(direct_url)["vcs_info"]["commit_id"] if direct_url else None
    if version != PINNED_VERSION or commit != PINNED_COMMIT:
        raise SystemExit(f"liana {version} @ {commit} is not the pinned {PINNED_VERSION} @ {PINNED_COMMIT}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(ADATA_PATH)
    toy = pd.read_csv(TOY_PATH)
    complex_resource = pd.DataFrame(COMPLEX_PAIRS, columns=[P.ligand, P.receptor])

    cases = {
        "toy": case(adata.copy(), toy, 0, 0.05, resource="toy"),
        # above the highest subunit proportion of the low clusters, so rows drop
        "toy_cut": case(adata.copy(), toy, 0, 0.35, resource="toy"),
        # 2088 of the 4205 cells sit in `A`, below this cut: `A` disappears
        "toy_min_cells": case(adata.copy(), toy, 2100, 0.05, resource="toy"),
        "complex": case(adata.copy(), complex_resource, 0, 0.05, resource_pairs=COMPLEX_PAIRS),
        # nothing can pass: liana raises instead of returning an empty frame
        "empty": case(adata.copy(), toy, 0, 1.5, resource="toy"),
    }

    payload = {
        "liana_version": version,
        "liana_commit": commit,
        "adata": str(ADATA_PATH.relative_to(REPO_ROOT)),
        "adata_sha256": hashlib.sha256(ADATA_PATH.read_bytes()).hexdigest(),
        "groupby": GROUPBY,
        "toy_resource": str(TOY_PATH.relative_to(REPO_ROOT)),
        "toy_resource_sha256": hashlib.sha256(TOY_PATH.read_bytes()).hexdigest(),
        "cases": cases,
    }
    (OUT_DIR / "synthetic.json").write_text(json.dumps(payload, indent=2) + "\n")

    print(f"liana {version} @ {commit[:8]}")
    for name, entry in cases.items():
        if "raises" in entry:
            print(f"  {name:14s} min_cells={entry['min_cells']:<5} expr_prop={entry['expr_prop']} -> raises {entry['raises']}")
        else:
            print(
                f"  {name:14s} min_cells={entry['min_cells']:<5} expr_prop={entry['expr_prop']}"
                f"  labels={entry['labels']}  rows={entry['n_lr_rows']:3d}  kept={entry['n_kept']:3d}"
                f"  {entry['sha256_kept'][:16]}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
