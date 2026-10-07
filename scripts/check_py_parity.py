#!/usr/bin/env python3
"""Python-module parity gate: `liana_rs.run` on the synthetic fixture + the toy
resource must reproduce every oracle CSV value-exactly (`parity_diff.py --rtol 0`)
for all nine methods x n_perms {100, 1000} — the same contract
`scripts/check_pipe_parity.sh` (library) and `scripts/check_cli_parity.sh` (CLI)
pin. Also asserts the shim's no-pandas fallback returns the columnar dict.

Run with the venv the README's "Python module" section builds:

    target/py-venv/bin/python scripts/check_py_parity.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import liana_rs

METHODS = [
    "cellphonedb",
    "geometric_mean",
    "cellchat",
    "connectome",
    "logfc",
    "natmi",
    "scseqcomm",
    "singlecellsignalr",
    "rank_aggregate",
]
N_PERMS = (100, 1000)

ROOT = Path(__file__).resolve().parent.parent
H5AD = ROOT / "testdata/fixtures/synthetic.h5ad"
RESOURCE = ROOT / "testdata/expected/synthetic__resource.csv"


def check_pandas_fallback() -> bool:
    """With pandas unimportable, `liana_rs.run` returns the columnar dict."""
    saved = sys.modules.get("pandas", "absent")
    sys.modules["pandas"] = None  # `import pandas` then raises ImportError
    try:
        result = liana_rs.run(str(H5AD), "cell_type", str(RESOURCE), "cellphonedb", n_perms=100)
    finally:
        if saved == "absent":
            del sys.modules["pandas"]
        else:
            sys.modules["pandas"] = saved
    ok = isinstance(result, dict) and isinstance(result.get("cellphone_pvals"), list)
    print(f"GATE: {'PASS' if ok else 'FAIL'} — no-pandas fallback returns the dict")
    return ok


def main() -> int:
    out_dir = ROOT / "target/py_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    failures = not check_pandas_fallback()
    for method in METHODS:
        for n_perms in N_PERMS:
            frame = liana_rs.run(str(H5AD), "cell_type", str(RESOURCE), method, n_perms=n_perms)
            actual = out_dir / f"synthetic__{method}__p{n_perms}.csv"
            frame.to_csv(actual, index=False)
            expected = ROOT / f"testdata/expected/synthetic__{method}__p{n_perms}.csv"
            gate = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/parity_diff.py"),
                    "--expected",
                    str(expected),
                    "--actual",
                    str(actual),
                    "--rtol",
                    "0",
                ],
                check=False,
            )
            failures += gate.returncode != 0

    total = 1 + len(METHODS) * len(N_PERMS)
    print(f"py gate: {total - failures}/{total} PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
