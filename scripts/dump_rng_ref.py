#!/usr/bin/env python3
"""Dump liana's permutation stream as .npy parity references + a manifest.

The stream is the one `_generate_perms_cube` consumes: a single
`np.random.default_rng(seed)` feeding `_chunk_permutations`, reshaped to the
`(n_perms, n_obs)` matrix each chunk yields. Dumps go to
`testdata/rng_ref/<name>.npy`; `manifest.json` records, per config, the file's
sha256, the sha256 of the raw C-order little-endian data buffer (what a port
has to reproduce), dtype and shape.

Run with the oracle venv interpreter (liana 2.0.0 / numpy 2.5.3):

    /home/pwwang/p0a/venv/bin/python scripts/dump_rng_ref.py
"""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import os
import pathlib
import platform

import liana
import numpy as np
from liana._core._pipe_utils._get_mean_perms import _MAX_PERM_INDEX_ELEMENTS, _chunk_permutations

SEEDS = (0, 1, 1337)
N_OBS = (1000, 10000, 50000)
N_PERMS = (100,)
EXTRA = ((1337, 50000, 1000),)

REPO_ROOT = pathlib.Path(os.environ.get("REPO_ROOT", pathlib.Path(__file__).resolve().parents[1]))
OUT_DIR = REPO_ROOT / "testdata" / "rng_ref"


def permutations(seed: int, n_obs: int, n_perms: int) -> np.ndarray:
    """The exact call `_generate_perms_cube` makes, with its own chunk count."""
    rng = np.random.default_rng(seed=seed)
    n_chunks = max(1, min(n_perms, -(-n_perms * n_obs // _MAX_PERM_INDEX_ELEMENTS)))
    return np.concatenate(list(_chunk_permutations(rng, n_obs, n_perms, n_chunks)))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    commit = None
    direct_url = md.distribution("liana").read_text("direct_url.json")
    if direct_url:
        commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")

    entries = []
    for seed, n_obs, n_perms in [(s, o, p) for s in SEEDS for o in N_OBS for p in N_PERMS] + list(EXTRA):
        matrix = permutations(seed, n_obs, n_perms)
        name = f"rng_seed{seed}_nobs{n_obs}_nperms{n_perms}.npy"
        path = OUT_DIR / name
        np.save(path, matrix)
        entries.append(
            {
                "seed": seed,
                "n_obs": n_obs,
                "n_perms": n_perms,
                "npy": name,
                "npy_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "data_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
                "dtype": matrix.dtype.str,  # '<u2'
                "shape": list(matrix.shape),
            }
        )
        print(f"wrote {name}  shape={matrix.shape} dtype={matrix.dtype}")

    manifest = {
        "generated_by": "scripts/dump_rng_ref.py",
        "generator": "np.random.default_rng(seed) through liana._core._pipe_utils._chunk_permutations",
        "liana_version": liana.__version__,
        "liana_commit": commit,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "entries": sorted(entries, key=lambda e: (e["seed"], e["n_obs"], e["n_perms"])),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote manifest.json  entries={len(entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
