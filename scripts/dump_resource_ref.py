#!/usr/bin/env python3
"""Dump liana's `omni_resource.csv` selections as parity references.

For every resource name `select_resource` accepts, records the pair count and a
sha256 of the `(ligand, receptor)` stream — in the order the CSV/`select_resource`
yields them, and, for `consensus` only, sorted. `consensus` also records its
protein-complex inventory (the entries whose symbol contains `_`).

Hashes are defined over UTF-8 text: one `"{ligand}\\t{receptor}"` line per pair,
`\\n`-terminated, so a Rust port reproduces them by writing the same stream.

Writes `testdata/resource_ref/resources.json` + `testdata/resource_ref/consensus.json`.

Run with the oracle venv interpreter (liana 2.0.0):

    /home/pwwang/p0a/venv/bin/python scripts/dump_resource_ref.py
"""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import os
import pathlib

import liana.resource
import pandas as pd
from liana.resource.select_resource import select_resource, show_resources

PINNED_VERSION = "2.0.0"
PINNED_COMMIT = "c59472ccc9de8360dbbf5016db75f8abde08dd3e"

REPO_ROOT = pathlib.Path(os.environ.get("REPO_ROOT", pathlib.Path(__file__).resolve().parents[1]))
OUT_DIR = REPO_ROOT / "testdata" / "resource_ref"
CSV_PATH = pathlib.Path(liana.resource.__file__).parent / "omni_resource.csv"


def stream_sha256(pairs: list[tuple[str, str]]) -> str:
    payload = "".join(f"{lig}\t{rec}\n" for lig, rec in pairs).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    version = md.version("liana")
    direct_url = md.distribution("liana").read_text("direct_url.json")
    commit = json.loads(direct_url)["vcs_info"]["commit_id"] if direct_url else None
    if version != PINNED_VERSION or commit != PINNED_COMMIT:
        raise SystemExit(f"liana {version} @ {commit} is not the pinned {PINNED_VERSION} @ {PINNED_COMMIT}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_sha = hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()

    resources: dict[str, dict] = {}
    consensus: dict = {}
    for name in show_resources():
        # `select_resource` lower-cases internally; keep the CSV's own casing on output.
        resource = select_resource(name)
        pairs = list(zip(resource["ligand"], resource["receptor"], strict=True))
        entry = {
            "n_pairs": len(pairs),
            "n_unique_ligands": int(resource["ligand"].nunique()),
            "n_unique_receptors": int(resource["receptor"].nunique()),
            "sha256_pairs_ordered": stream_sha256(pairs),
        }
        resources[str(name)] = entry

        if name == "consensus":
            lig_cplx = sorted({lig for lig, _ in pairs if "_" in lig})
            rec_cplx = sorted({rec for _, rec in pairs if "_" in rec})
            consensus = {
                **entry,
                "sha256_pairs_sorted": stream_sha256(sorted(pairs)),
                "n_ligand_complexes": len(lig_cplx),
                "n_receptor_complexes": len(rec_cplx),
                "ligand_complexes": lig_cplx,
                "receptor_complexes": rec_cplx,
            }

    common = {
        "liana_version": version,
        "liana_commit": commit,
        "csv": "liana/resource/omni_resource.csv",
        "csv_sha256": csv_sha,
        "n_csv_rows": len(pd.read_csv(CSV_PATH, index_col=False)),
        "n_resources": len(resources),
    }
    (OUT_DIR / "resources.json").write_text(json.dumps({**common, "resources": resources}, indent=2) + "\n")
    (OUT_DIR / "consensus.json").write_text(json.dumps({**common, "consensus": consensus}, indent=2) + "\n")

    print(f"liana {version} @ {commit[:8]} — {len(resources)} resources")
    print(f"csv sha256 {csv_sha}")
    print(f"consensus: {consensus['n_pairs']} pairs,"
          f" {consensus['n_ligand_complexes']} ligand complexes,"
          f" {consensus['n_receptor_complexes']} receptor complexes")
    for name, entry in resources.items():
        print(f"  {name:20s} {entry['n_pairs']:5d} pairs  {entry['sha256_pairs_ordered'][:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
