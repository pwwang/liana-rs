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
from liana.resource._reassemble_complexes import _explode_complexes
from liana.resource.select_resource import select_resource, show_resources

PINNED_VERSION = "2.0.0"
PINNED_COMMIT = "c59472ccc9de8360dbbf5016db75f8abde08dd3e"

REPO_ROOT = pathlib.Path(os.environ.get("REPO_ROOT", pathlib.Path(__file__).resolve().parents[1]))
OUT_DIR = REPO_ROOT / "testdata" / "resource_ref"
CSV_PATH = pathlib.Path(liana.resource.__file__).parent / "omni_resource.csv"
TOY_PATH = REPO_ROOT / "testdata" / "expected" / "synthetic__resource.csv"


def stream_sha256(pairs: list[tuple[str, str]]) -> str:
    payload = "".join(f"{lig}\t{rec}\n" for lig, rec in pairs).encode()
    return hashlib.sha256(payload).hexdigest()


def exploded_sha256(exploded: pd.DataFrame) -> str:
    """sha256 of the exploded subunits: `lig\\trec\\tlig_complex\\trec_complex` lines."""
    payload = "".join(
        f"{lig}\t{rec}\t{lig_c}\t{rec_c}\n"
        for lig, rec, lig_c, rec_c in zip(
            exploded["ligand"],
            exploded["receptor"],
            exploded["ligand_complex"],
            exploded["receptor_complex"],
            strict=True,
        )
    ).encode()
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
            exploded = _explode_complexes(resource.copy())
            consensus = {
                **entry,
                "sha256_pairs_sorted": stream_sha256(sorted(pairs)),
                "n_ligand_complexes": len(lig_cplx),
                "n_receptor_complexes": len(rec_cplx),
                "ligand_complexes": lig_cplx,
                "receptor_complexes": rec_cplx,
                "n_exploded_pairs": len(exploded),
                "sha256_exploded": exploded_sha256(exploded),
            }

    common = {
        "liana_version": version,
        "liana_commit": commit,
        "csv": "liana/resource/omni_resource.csv",
        "csv_sha256": csv_sha,
        "n_csv_rows": len(pd.read_csv(CSV_PATH, index_col=False)),
        "n_resources": len(resources),
    }
    toy = pd.read_csv(TOY_PATH)
    toy_pairs = list(zip(toy["ligand"], toy["receptor"], strict=True))
    toy_payload = {
        **common,
        "name": "synthetic__resource",
        "file": str(TOY_PATH.relative_to(REPO_ROOT)),
        "sha256_file": hashlib.sha256(TOY_PATH.read_bytes()).hexdigest(),
        "n_pairs": len(toy_pairs),
        "n_complex_pairs": sum(1 for lig, rec in toy_pairs if "_" in lig or "_" in rec),
        "sha256_pairs_ordered": stream_sha256(toy_pairs),
    }
    (OUT_DIR / "resources.json").write_text(json.dumps({**common, "resources": resources}, indent=2) + "\n")
    (OUT_DIR / "consensus.json").write_text(json.dumps({**common, "consensus": consensus}, indent=2) + "\n")
    (OUT_DIR / "toy_synthetic.json").write_text(json.dumps(toy_payload, indent=2) + "\n")

    print(f"liana {version} @ {commit[:8]} — {len(resources)} resources")
    print(f"csv sha256 {csv_sha}")
    print(f"consensus: {consensus['n_pairs']} pairs,"
          f" {consensus['n_ligand_complexes']} ligand complexes,"
          f" {consensus['n_receptor_complexes']} receptor complexes,"
          f" {consensus['n_exploded_pairs']} exploded rows")
    print(f"toy: {toy_payload['n_pairs']} pairs ({toy_payload['n_complex_pairs']} complex)")
    for name, entry in resources.items():
        print(f"  {name:20s} {entry['n_pairs']:5d} pairs  {entry['sha256_pairs_ordered'][:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
