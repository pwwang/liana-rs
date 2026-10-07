#!/usr/bin/env bash
# Pinned reference oracle for liana-rs parity testing.
#
# Regenerates expected outputs from the pinned Python oracle — liana 2.0.0
# (V2.0.0, commit c59472ccc9de8360dbbf5016db75f8abde08dd3e) — for every *input*
# fixture in testdata/fixtures/, for the single-cell methods
# {cellphonedb, geometric_mean, rank_aggregate} x n_perms {100, 1000}, seed 1337.
#
# Writes testdata/expected/<fixture>__<method>__p<N>.csv
#    and testdata/expected/<fixture>__<method>__p<N>.meta.json
#
# Usage:
#   scripts/oracle.sh                     # regenerate everything
#   LIANA_ORACLE_PYTHON=/path/to/python scripts/oracle.sh
#
# Determinism: the run is single-threaded (n_jobs=1) and fully seeded; rerunning
# must reproduce byte-identical CSVs. Verify with: scripts/oracle.sh && git -C
# testdata/expected diff --stat   (see docs/DEVELOPMENT.md).
#
# Fixture classification lives in testdata/FIXTURES.md. Golden-output CSVs
# (all_defaults, not_defaults, aggregate_rank_rest) are NOT inputs and are
# deliberately absent from the table below — see FIXTURES.md for why.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT
LIANA_ORACLE_PYTHON="${LIANA_ORACLE_PYTHON:-/home/pwwang/p0a/venv/bin/python}"
export LIANA_ORACLE_PYTHON

exec "$LIANA_ORACLE_PYTHON" - <<'PY'
import datetime
import hashlib
import itertools
import json
import os
import pathlib
import platform
import sys

PINNED_VERSION = "2.0.0"
PINNED_COMMIT = "c59472ccc9de8360dbbf5016db75f8abde08dd3e"

# fixture -> call adaptation. Only genuine *inputs* belong here.
#   groupby  : obs column holding the cluster labels
#   resource : "consensus" (liana's default) or "toy_all_pairs"
#
# "toy_all_pairs" is needed for fixtures whose var_names are not real gene
# symbols (e.g. synthetic.h5ad: ECM, ligA, ligB, ...), where the consensus
# resource matches nothing and liana refuses to run. It builds every ordered
# non-self (ligand, receptor) pair from the fixture's own var_names — no RNG,
# so the resource is reproducible from the fixture alone. The resource is
# frozen to testdata/expected/<fixture>__resource.csv for the Rust side.
FIXTURES = {
    "synthetic.h5ad": {"groupby": "cell_type", "resource": "toy_all_pairs"},
}
METHODS = ["cellphonedb", "geometric_mean", "cellchat", "rank_aggregate"]
N_PERMS = [100, 1000]
SEED = 1337
N_JOBS = 1  # single-threaded: keeps permutation results reproducible

repo = pathlib.Path(os.environ["REPO_ROOT"])
fixtures_dir = repo / "testdata" / "fixtures"
out_dir = repo / "testdata" / "expected"

import anndata as ad
import liana as li
import numba
import numpy as np
import pandas as pd


def die(msg):
    print(f"oracle: ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


# --- pin check: refuse to generate expected outputs from the wrong oracle ------
if li.__version__ != PINNED_VERSION:
    die(f"liana {li.__version__} installed, expected {PINNED_VERSION}")

import importlib.metadata as md

dist = md.distribution("liana")
direct_url = dist.read_text("direct_url.json")
commit = None
if direct_url:
    info = json.loads(direct_url)
    commit = info.get("vcs_info", {}).get("commit_id")
if commit != PINNED_COMMIT:
    die(f"liana commit {commit!r}, expected {PINNED_COMMIT}")
print(f"oracle: liana {li.__version__} @ {commit}")

if not fixtures_dir.is_dir():
    die(f"fixtures dir not found: {fixtures_dir}")

out_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")

written = []
for fixture, spec in FIXTURES.items():
    path = fixtures_dir / fixture
    if not path.is_file():
        die(f"input fixture not found: {path}")
    adata = ad.read_h5ad(path)
    if spec["groupby"] not in adata.obs.columns:
        die(f"{fixture}: obs[{spec['groupby']!r}] missing (has {list(adata.obs.columns)})")

    # Build the resource this fixture is scored against.
    resource = None
    if spec["resource"] == "toy_all_pairs":
        resource = pd.DataFrame(
            itertools.product(adata.var_names, adata.var_names), columns=["ligand", "receptor"]
        )
        resource = resource[resource["ligand"] != resource["receptor"]].reset_index(drop=True)
        res_path = out_dir / f"{path.stem}__resource.csv"
        resource.to_csv(res_path, index=False)
        print(f"oracle: wrote {res_path.name}  pairs={resource.shape[0]}")
    elif spec["resource"] != "consensus":
        die(f"{fixture}: unknown resource spec {spec['resource']!r}")

    for method in METHODS:
        fn = getattr(li.mt, method)
        for n_perms in N_PERMS:
            res = fn(
                adata,
                groupby=spec["groupby"],
                resource_name="consensus",  # ignored when `resource` is given
                resource=resource,
                n_perms=n_perms,
                seed=SEED,
                n_jobs=N_JOBS,
                inplace=False,
                verbose=False,
            )
            stem = f"{path.stem}__{method}__p{n_perms}"
            csv_path = out_dir / f"{stem}.csv"
            res.to_csv(csv_path, index=False)

            meta = {
                "fixture": fixture,
                "fixture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "method": method,
                "groupby": spec["groupby"],
                "resource_name": "consensus",
                "resource": spec["resource"],
                "resource_sha256": (
                    hashlib.sha256((out_dir / f"{path.stem}__resource.csv").read_bytes()).hexdigest()
                    if resource is not None
                    else None
                ),
                "n_perms": n_perms,
                "seed": SEED,
                "n_jobs": N_JOBS,
                "inplace": False,
                "return_all_lrs": False,
                "n_obs": int(adata.n_obs),
                "n_vars": int(adata.n_vars),
                "result_rows": int(res.shape[0]),
                "result_cols": list(res.columns),
                "liana_version": li.__version__,
                "liana_commit": commit,
                "numpy_version": np.__version__,
                "numba_version": numba.__version__,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "generated_at": timestamp,
            }
            meta_path = out_dir / f"{stem}.meta.json"
            meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
            written.append(stem)
            print(f"oracle: wrote {csv_path.name}  rows={res.shape[0]} cols={res.shape[1]}")

print(f"oracle: done — {len(written)} outputs in {out_dir}")
PY
