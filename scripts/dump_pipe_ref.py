#!/usr/bin/env python3
"""Dump liana's single-cell pipeline intermediates as parity references.

Runs liana's own `_prepare_lr_stats` and `_run_method` (the two halves of
`liana_pipe`) on `synthetic.h5ad` against the toy resource, for the
`cellphonedb` (mean aggregation) and `cellchat` (trimean aggregation) methods,
then records the stages a Rust port has to reproduce *between* the fixture and
the oracle CSV:

    prep      per-cluster per-gene `means` (f32) / `props` (f64) / `trimean`
              (f64, cellchat), the prepared var order and the label counts
    lr_rows   the pre-reassembly `lr_res` keys, in liana's row order
    p<n>      the permutation cube (first perms literal, whole cube hashed),
              the ligand/receptor permutation selection and combined statistics
              for three sample rows, and the final score / p-value columns

Floats are written as Python's shortest round-trip repr of the float64 they
widen to, which is exact for the f32 columns too; every vector also carries a
sha256 over the big-endian IEEE-754 bit patterns
(`sha256_{means,lr_means,pvals,cube}_bits`) so the test pins the arithmetic,
not a rounded decimal.

Writes `testdata/pipe_ref/synthetic__cellphonedb.json` and
`testdata/pipe_ref/synthetic__cellchat.json`.

Run with the oracle venv interpreter (liana 2.0.0):

    /home/pwwang/p0a/venv/bin/python scripts/dump_pipe_ref.py
"""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import os
import pathlib
import struct

import anndata as ad
import liana as li
import numpy as np
import pandas as pd
from liana._core._constants import CommonColumns as C
from liana._core._constants import DefaultValues as V
from liana._core._constants import MethodColumns as M
from liana._core._constants import PrimaryColumns as P
from liana._core._types import get_obs, get_x
from liana._core._pipe_utils._pre import _choose_mtx_rep
from liana._core._pipe_utils._get_mean_perms import (
    _get_mat_idx,
    _get_means_perms,
    _trimean,
)
from liana.method.sc._cellchat import _cellchat, _lr_probability
from liana.method.sc._cellphonedb import _cellphonedb
from liana.method.sc._liana_pipe import _prepare_lr_stats, _run_method, _sort_by_score

PINNED_VERSION = "2.0.0"
PINNED_COMMIT = "c59472ccc9de8360dbbf5016db75f8abde08dd3e"

REPO_ROOT = pathlib.Path(os.environ.get("REPO_ROOT", pathlib.Path(__file__).resolve().parents[1]))
OUT_DIR = REPO_ROOT / "testdata" / "pipe_ref"
ADATA_PATH = REPO_ROOT / "testdata" / "fixtures" / "synthetic.h5ad"
TOY_PATH = REPO_ROOT / "testdata" / "expected" / "synthetic__resource.csv"
GROUPBY = "cell_type"
SEED = 1337
N_PERMS = [100, 1000]
# the copy of `liana_pipe`'s column selection for a `cellphonedb` score
COMPLEX_COLS = _cellphonedb.complex_cols
ADD_COLS = _cellphonedb.add_cols + ["ligand", "receptor", "ligand_props", "receptor_props"]
KEY_COLS = [P.source, P.target, P.ligand_complex, P.receptor_complex]

# ...and for a `cellchat` score: the trimean columns reassemble instead of the
# means, and `mat_max` joins the frame.
CHAT_COMPLEX_COLS = _cellchat.complex_cols
CHAT_ADD_COLS = _cellchat.add_cols + ["ligand", "receptor", "ligand_props", "receptor_props"]


def bits(value: float, fmt: str) -> str:
    """`value` as big-endian IEEE-754 hex bits, for hashing."""
    return struct.pack(fmt, float(value)).hex()


def sha256_stream(lines) -> str:
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def f32_bits_stream(values) -> str:
    return sha256_stream(bits(v, ">f") + "\n" for v in values)


def f64_bits_stream(values) -> str:
    return sha256_stream(bits(v, ">d") + "\n" for v in values)


def num(value) -> float:
    """Shortest round-trip repr of the float64 a f32 or f64 widens to."""
    return float(value)


def main() -> int:
    version = md.version("liana")
    direct_url = md.distribution("liana").read_text("direct_url.json")
    commit = json.loads(direct_url)["vcs_info"]["commit_id"] if direct_url else None
    if version != PINNED_VERSION or commit != PINNED_COMMIT:
        raise SystemExit(f"liana {version} @ {commit} is not the pinned {PINNED_VERSION} @ {PINNED_COMMIT}")

    adata = ad.read_h5ad(ADATA_PATH)
    toy = pd.read_csv(TOY_PATH)

    adata, lr_res = _prepare_lr_stats(
        adata=adata,
        groupby=GROUPBY,
        resource_name="consensus",
        resource=toy,
        interactions=None,
        groupby_pairs=None,
        min_cells=V.min_cells,
        base=V.logbase,
        de_method=V.de_method,
        verbose=False,
        use_raw=False,
        layer=None,
        complex_cols=COMPLEX_COLS,
        add_cols=ADD_COLS,
        spatial_key=None,
        spatial_kwargs=None,
        mdata_kwargs={},
    )
    obs = get_obs(adata)
    labels = [str(label) for label in obs["@label"].cat.categories]
    var_names = [str(name) for name in adata.var_names]
    counts = {label: int(np.sum(obs["@label"] == label)) for label in labels}

    # per-cluster means (f32) and props (f64), as `_get_lr` computes them
    means: dict[str, dict[str, float]] = {}
    props: dict[str, dict[str, float]] = {}
    for label in labels:
        temp = adata[obs["@label"] == label, :]
        dense = get_x(temp).mean(axis=0)
        props_label = get_x(temp).getnnz(axis=0) / temp.shape[0]
        means[label] = {name: num(value) for name, value in zip(var_names, np.asarray(dense).ravel())}
        props[label] = {name: num(value) for name, value in zip(var_names, props_label)}

    prep = {
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "var_names": var_names,
        "labels": labels,
        "counts": counts,
        "means": means,
        "props": props,
        "sha256_means_bits": f32_bits_stream(v for label in labels for v in means[label].values()),
        "sha256_props_bits": f64_bits_stream(v for label in labels for v in props[label].values()),
    }

    # the pre-reassembly rows, in `_get_lr`'s order
    lr_rows = {
        "n_rows": int(len(lr_res)),
        "columns": [str(col) for col in lr_res.columns],
        "sha256_keys": sha256_stream(
            f"{s}\t{t}\t{lc}\t{rc}\n"
            for s, t, lc, rc in zip(lr_res[P.source], lr_res[P.target], lr_res[P.ligand_complex], lr_res[P.receptor_complex], strict=True)
        ),
        "sha256_ligand_means_bits": f32_bits_stream(lr_res[C.ligand_means]),
        "sha256_receptor_means_bits": f32_bits_stream(lr_res[C.receptor_means]),
        "sha256_ligand_props_bits": f64_bits_stream(lr_res[C.ligand_props]),
        "sha256_receptor_props_bits": f64_bits_stream(lr_res[C.receptor_props]),
    }

    def as_key(row) -> dict:
        return {
            "source": str(row[P.source]),
            "target": str(row[P.target]),
            "ligand_complex": str(row[P.ligand_complex]),
            "receptor_complex": str(row[P.receptor_complex]),
        }

    per_n_perms: dict[str, dict] = {}
    for n_perms in N_PERMS:
        scored = _run_method(
            lr_res=lr_res.copy(),
            adata=adata,
            groupby=GROUPBY,
            expr_prop=V.expr_prop,
            _score=_cellphonedb,
            _key_cols=KEY_COLS,
            _complex_cols=COMPLEX_COLS,
            _add_cols=ADD_COLS,
            n_perms=n_perms,
            seed=SEED,
            return_all_lrs=False,
            n_jobs=1,
            verbose=False,
        )
        cube = _get_means_perms(
            adata=adata,
            n_perms=n_perms,
            seed=SEED,
            aggregation="mean",
            norm_factor=None,
            n_jobs=1,
            verbose=False,
        )
        ligand_idx, receptor_idx, source_idx, target_idx = _get_mat_idx(adata, scored)
        ligand_perms = cube[:, source_idx, ligand_idx]
        receptor_perms = cube[:, target_idx, receptor_idx]
        perm_means = np.mean(np.stack((ligand_perms, receptor_perms), axis=0), axis=0)

        # three sample rows: the first, the middle and the last of the liana order
        sample_rows = [0, len(scored) // 2, len(scored) - 1]
        samples = []
        for row in sample_rows:
            samples.append(
                {
                    **as_key(scored.iloc[row]),
                    "ligand": str(scored[P.ligand].iloc[row]),
                    "receptor": str(scored[P.receptor].iloc[row]),
                    "ligand_means": num(scored[C.ligand_means].iloc[row]),
                    "receptor_means": num(scored[C.receptor_means].iloc[row]),
                    "lr_means": num(scored["lr_means"].iloc[row]),
                    "cellphone_pvals": num(scored["cellphone_pvals"].iloc[row]),
                    "ligand_perm_means": [num(v) for v in ligand_perms[:3, row]],
                    "receptor_perm_means": [num(v) for v in receptor_perms[:3, row]],
                    "perm_means": [num(v) for v in perm_means[:3, row]],
                }
            )

        per_n_perms[str(n_perms)] = {
            "seed": SEED,
            "n_perms": n_perms,
            "n_chunks": max(1, min(n_perms, -(-n_perms * adata.shape[0] // (1 << 24)))),
            "first_perms": [[num(v) for v in row] for row in cube[:3].reshape(3, -1)],
            "sha256_cube_bits": f64_bits_stream(cube.reshape(-1)),
            "sha256_lr_means_bits": f32_bits_stream(scored["lr_means"]),
            "sha256_pvals_bits": f64_bits_stream(scored["cellphone_pvals"]),
            "samples": samples,
        }

    payload = {
        "liana_version": version,
        "liana_commit": commit,
        "numpy_version": np.__version__,
        "adata": str(ADATA_PATH.relative_to(REPO_ROOT)),
        "adata_sha256": hashlib.sha256(ADATA_PATH.read_bytes()).hexdigest(),
        "resource": str(TOY_PATH.relative_to(REPO_ROOT)),
        "resource_sha256": hashlib.sha256(TOY_PATH.read_bytes()).hexdigest(),
        "groupby": GROUPBY,
        "expr_prop": V.expr_prop,
        "min_cells": V.min_cells,
        "method": "cellphonedb",
        "prep": prep,
        "lr_rows": lr_rows,
        "n_perms": per_n_perms,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "synthetic__cellphonedb.json").write_text(json.dumps(payload, indent=2) + "\n")

    print(f"liana {version} @ {commit[:8]}  numpy {np.__version__}")
    print(f"  cellphonedb")
    print(f"  prep   labels={labels} counts={counts} vars={len(var_names)}")
    print(f"  lr_rows n={lr_rows['n_rows']} keys={lr_rows['sha256_keys'][:16]}")
    for n_perms in N_PERMS:
        entry = per_n_perms[str(n_perms)]
        print(
            f"  p{n_perms:<5} n_chunks={entry['n_chunks']} cube={entry['sha256_cube_bits'][:16]}"
            f" lr_means={entry['sha256_lr_means_bits'][:16]} pvals={entry['sha256_pvals_bits'][:16]}"
        )

    dump_cellchat(version, commit)
    return 0


def dump_cellchat(version: str, commit: str) -> None:
    """The `cellchat` half: the trimean aggregation and its probability null.

    Same two halves as the cellphonedb dump, with cellchat's column selection
    (`ligand_trimean`/`receptor_trimean` reassemble in place of the means) and
    `aggregation="trimean"` + `norm_factor=mat_max` for the null cube. The
    reassembled run is asserted equal to `li.mt.cellchat`'s own output, so a
    drifted reconstruction of the call path fails here rather than silently
    pinning the wrong oracle.
    """
    assert list(P.primary) == KEY_COLS, "`_run_method`'s key columns"
    adata_in = ad.read_h5ad(ADATA_PATH)
    toy = pd.read_csv(TOY_PATH)

    adata, lr_res = _prepare_lr_stats(
        adata=adata_in,
        groupby=GROUPBY,
        resource_name="consensus",
        resource=toy,
        interactions=None,
        groupby_pairs=None,
        min_cells=V.min_cells,
        base=V.logbase,
        de_method=V.de_method,
        verbose=False,
        use_raw=False,
        layer=None,
        complex_cols=CHAT_COMPLEX_COLS,
        add_cols=CHAT_ADD_COLS,
        spatial_key=None,
        spatial_kwargs=None,
        mdata_kwargs={},
    )
    obs = get_obs(adata)
    labels = [str(label) for label in obs["@label"].cat.categories]
    var_names = [str(name) for name in adata.var_names]
    counts = {label: int(np.sum(obs["@label"] == label)) for label in labels}
    mat_max = np.unique(lr_res[M.mat_max].to_numpy())[0]

    # the observed side of the null: per-label trimeans of `X / mat_max`,
    # exactly as `_get_lr` computes them (`_liana_pipe.py:542`)
    trimeans: dict[str, dict[str, float]] = {}
    for label in labels:
        temp = adata[obs["@label"] == label, :]
        values = np.asarray(_trimean(_choose_mtx_rep(temp) / mat_max))
        trimeans[label] = {name: num(value) for name, value in zip(var_names, values)}

    def as_key(row) -> dict:
        return {
            "source": str(row[P.source]),
            "target": str(row[P.target]),
            "ligand_complex": str(row[P.ligand_complex]),
            "receptor_complex": str(row[P.receptor_complex]),
        }

    per_n_perms: dict[str, dict] = {}
    for n_perms in N_PERMS:
        scored = _sort_by_score(
            _run_method(
                lr_res=lr_res.copy(),
                adata=adata,
                groupby=GROUPBY,
                expr_prop=V.expr_prop,
                _score=_cellchat,
                _key_cols=P.primary,
                _complex_cols=CHAT_COMPLEX_COLS,
                _add_cols=CHAT_ADD_COLS,
                n_perms=n_perms,
                seed=SEED,
                return_all_lrs=False,
                n_jobs=1,
                verbose=False,
            ),
            _cellchat,
        )
        direct = li.mt.cellchat(
            adata_in,
            groupby=GROUPBY,
            resource=toy,
            n_perms=n_perms,
            seed=SEED,
            n_jobs=1,
            inplace=False,
            verbose=False,
        )
        pd.testing.assert_frame_equal(scored, direct)

        cube = _get_means_perms(
            adata=adata,
            n_perms=n_perms,
            seed=SEED,
            aggregation="trimean",
            norm_factor=mat_max,
            n_jobs=1,
            verbose=False,
        )
        ligand_idx, receptor_idx, source_idx, target_idx = _get_mat_idx(adata, scored)
        ligand_perms = cube[:, source_idx, ligand_idx]
        receptor_perms = cube[:, target_idx, receptor_idx]
        perm_probs = np.asarray(
            _lr_probability(np.stack((ligand_perms, receptor_perms), axis=0))
        )

        sample_rows = [0, len(scored) // 2, len(scored) - 1]
        samples = []
        for row in sample_rows:
            samples.append(
                {
                    **as_key(scored.iloc[row]),
                    "ligand": str(scored[P.ligand].iloc[row]),
                    "receptor": str(scored[P.receptor].iloc[row]),
                    "ligand_trimean": num(scored[M.ligand_trimean].iloc[row]),
                    "receptor_trimean": num(scored[M.receptor_trimean].iloc[row]),
                    "lr_probs": num(scored["lr_probs"].iloc[row]),
                    "cellchat_pvals": num(scored["cellchat_pvals"].iloc[row]),
                    "mat_max": num(scored[M.mat_max].iloc[row]),
                    "ligand_perm_trimeans": [num(v) for v in ligand_perms[:3, row]],
                    "receptor_perm_trimeans": [num(v) for v in receptor_perms[:3, row]],
                    "perm_probs": [num(v) for v in perm_probs[:3, row]],
                }
            )

        per_n_perms[str(n_perms)] = {
            "seed": SEED,
            "n_perms": n_perms,
            "n_chunks": max(1, min(n_perms, -(-n_perms * adata.shape[0] // (1 << 24)))),
            "first_perms": [[num(v) for v in row] for row in cube[:3].reshape(3, -1)],
            "sha256_cube_bits": f64_bits_stream(cube.reshape(-1)),
            "sha256_lr_probs_bits": f64_bits_stream(scored["lr_probs"]),
            "sha256_pvals_bits": f64_bits_stream(scored["cellchat_pvals"]),
            "samples": samples,
        }

    payload = {
        "liana_version": version,
        "liana_commit": commit,
        "numpy_version": np.__version__,
        "adata": str(ADATA_PATH.relative_to(REPO_ROOT)),
        "adata_sha256": hashlib.sha256(ADATA_PATH.read_bytes()).hexdigest(),
        "resource": str(TOY_PATH.relative_to(REPO_ROOT)),
        "resource_sha256": hashlib.sha256(TOY_PATH.read_bytes()).hexdigest(),
        "groupby": GROUPBY,
        "expr_prop": V.expr_prop,
        "min_cells": V.min_cells,
        "method": "cellchat",
        "mat_max": num(mat_max),
        "prep": {
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
            "var_names": var_names,
            "labels": labels,
            "counts": counts,
            "trimeans": trimeans,
            "sha256_trimean_bits": f64_bits_stream(
                v for label in labels for v in trimeans[label].values()
            ),
        },
        "lr_rows": {
            "n_rows": int(len(lr_res)),
            "columns": [str(col) for col in lr_res.columns],
            "sha256_keys": sha256_stream(
                f"{s}\t{t}\t{lc}\t{rc}\n"
                for s, t, lc, rc in zip(lr_res[P.source], lr_res[P.target], lr_res[P.ligand_complex], lr_res[P.receptor_complex], strict=True)
            ),
            "sha256_ligand_trimean_bits": f64_bits_stream(lr_res[M.ligand_trimean]),
            "sha256_receptor_trimean_bits": f64_bits_stream(lr_res[M.receptor_trimean]),
            "sha256_ligand_props_bits": f64_bits_stream(lr_res[C.ligand_props]),
            "sha256_receptor_props_bits": f64_bits_stream(lr_res[C.receptor_props]),
        },
        "n_perms": per_n_perms,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "synthetic__cellchat.json").write_text(json.dumps(payload, indent=2) + "\n")

    print(f"  cellchat mat_max={mat_max!r} rows={len(lr_res)} labels={labels}")
    for n_perms in N_PERMS:
        entry = per_n_perms[str(n_perms)]
        print(
            f"  p{n_perms:<5} cube={entry['sha256_cube_bits'][:16]}"
            f" lr_probs={entry['sha256_lr_probs_bits'][:16]} pvals={entry['sha256_pvals_bits'][:16]}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
