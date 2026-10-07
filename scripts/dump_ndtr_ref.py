#!/usr/bin/env python3
"""Dump scipy 1.18.1's `ndtr` — the standard normal CDF — as a bit-exact
parity reference.

**Which kernel.** `scipy.stats.norm.cdf(x)` (what `liana`'s `_gene_cdf` calls,
`method/sc/_liana_pipe.py:755-767`) reduces to `scipy.special.ndtr`, scipy's
cephes transcription (`scipy/special/xsf/cephes/ndtr.h`): the `scipy.special`
ufuncs are compiled from the vendored C++ (`xsf`) sources, so the result is
the cephes rational-polynomial algorithm — not Boost, not the platform libm
(the only libm call left inside it is `exp`, for `erfc`'s `exp(-a*a)`).
The dump asserts `norm.cdf(x) == special.ndtr(x)` on every dumped input, so
the recorded outputs pin the ufunc the pipeline actually runs.

**What is dumped.** One f64 input family of ~45k values, as raw IEEE-754 bit
patterns, plus scipy's own outputs: the whole domain, dense around each of
`ndtr`'s branch switches (|x| = 1, the erf/erfc split; |a| = 1 and |a| = 8
inside erfc; the `-a*a < -MAXLOG` underflow guard at |a| ~ 26.6), ±ulp churn
around them, subnormals, and NaN/±inf/±0.

Every vector carries a sha256 over the big-endian u64 words, so the Rust test
pins the bit patterns, not any decimal spelling. The real `scseqcomm` inputs
are *not* a separate family here: they are the `(ligand_means - cluster_mean)
/ (std / sqrt(n))` values of the parity fixture, which the pipe gate
(`crates/liana-core/tests/pipe_parity.rs`, `scseqcomm_pipeline_matches_the_
oracle_csv`) checks end to end at zero tolerance against the oracle CSV's
`*_cdf` columns.

Writes `testdata/math_ref/scipy_ndtr_ref.json`.

Run with the oracle venv interpreter (scipy 1.18.1):

    /home/pwwang/p0a/venv/bin/python scripts/dump_ndtr_ref.py
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib

import numpy as np
import scipy
from scipy.special import ndtr
from scipy.stats import norm

REPO_ROOT = pathlib.Path(os.environ.get("REPO_ROOT", pathlib.Path(__file__).resolve().parents[1]))
OUT_DIR = REPO_ROOT / "testdata" / "math_ref"


def f64(bits: np.ndarray) -> np.ndarray:
    """u64 (or int) array -> same-shape f64 by bit pattern."""
    return np.asarray(bits, dtype=np.uint64).view(np.float64)


def ulp_sweep(center: float, k: int) -> np.ndarray:
    """center ± k ulp, as f64 bit arithmetic (Python ints, kept in u64 range)."""
    bits = int(np.float64(center).view(np.uint64))
    return f64(np.array([(bits + step) % 2**64 for step in range(-k, k + 1)], dtype=np.uint64))


def sweep(rng: np.random.Generator) -> np.ndarray:
    parts = [
        # the whole live domain, where ndtr is (0, 1)
        rng.normal(0.0, 1.0, 8000),
        rng.normal(0.0, 8.0, 6000),
        rng.uniform(-40.0, 40.0, 6000),
        rng.uniform(-1.0, 1.0, 4000),
        # the |x| < 1 branch, ±ulp churn around the switch at ±1
        rng.uniform(0.99999, 1.00001, 2000) * rng.choice([-1.0, 1.0], 2000),
        ulp_sweep(1.0, 2000),
        ulp_sweep(-1.0, 2000),
        # erfc's own switch at |a| = 1 (reached from ndtr only via |x| >= 1)
        ulp_sweep(np.sqrt(2.0), 500),
        ulp_sweep(-np.sqrt(2.0), 500),
        # the P/Q -> R/S switch at |a| = 8, in a-space and x-space
        ulp_sweep(8.0, 2000),
        ulp_sweep(-8.0, 2000),
        ulp_sweep(8.0 * np.sqrt(2.0), 500),
        # the exp(-a*a) underflow guard, -a*a < -MAXLOG (~ -709.78)
        ulp_sweep(np.sqrt(709.782712893384), 500),
        ulp_sweep(-np.sqrt(709.782712893384), 500),
        # large magnitudes either side, including the 2 - y rounding edge
        np.exp2(rng.uniform(-8.0, 6.0, 2000)) * rng.choice([-1.0, 1.0], 2000),
        # subnormals and near-zero, plus ±0
        f64(rng.integers(1, 0x0010000000000000, 2000, dtype=np.uint64)),
        f64(rng.integers(1, 0x0010000000000000, 2000, dtype=np.uint64)) * -1.0,
        f64([0x0000000000000000, 0x8000000000000000]),
        # ulp churn around the real pipeline's own input region (|x| ~ 0.5)
        ulp_sweep(0.5, 500),
        ulp_sweep(-0.5, 500),
        # specials
        f64([0x7FF0000000000000, 0xFFF0000000000000, 0x7FF8000000000000,
             0xFFF8000000000000, 0x7FF0000000000001, 0x7FEFFFFFFFFFFFFF,
             0xFFEFFFFFFFFFFFFF, 0x3FF0000000000000, 0xBFF0000000000000,
             0x0000000000000001, 0x8000000000000001]),
    ]
    return np.concatenate(parts)


def vector(values: np.ndarray) -> dict:
    """One bit-exact vector: hex blob of the BE u64 words + sha256."""
    words = np.asarray(values, dtype=np.float64).view(np.uint64)
    be = words.astype(">u8").tobytes()
    return {
        "count": int(words.size),
        "sha256": hashlib.sha256(be).hexdigest(),
        "hex": be.hex(),
    }


def main() -> int:
    rng = np.random.default_rng(1337)
    inputs = sweep(rng)
    assert inputs.size >= 10_000, "sweep size contract"

    with np.errstate(all="ignore"):
        outputs = norm.cdf(inputs)
        special = ndtr(inputs)

    checks = {
        "norm_cdf_vs_special_ndtr_diffs": int(
            (np.asarray(outputs).view(np.uint64) != np.asarray(special).view(np.uint64)).sum()
        ),
        "nonfinite_outputs": int((~np.isfinite(outputs)).sum()),
    }
    assert checks["norm_cdf_vs_special_ndtr_diffs"] == 0, \
        "norm.cdf and special.ndtr disagree: this dump no longer pins one ufunc"

    dump = {
        "generated_by": "scripts/dump_ndtr_ref.py",
        "scipy_version": scipy.__version__,
        "kernel": {
            "source": "scipy 1.18.1 scipy/special/xsf/cephes/ndtr.h",
            "called_by": "liana/method/sc/_liana_pipe.py:755-767 (_gene_cdf)",
            "implementation": "cephes ndtr -> erf/erfc rational polynomials "
                               "(polevl/p1evl over T/U, P/Q, R/S); exp is the "
                               "platform libm's, the rest is scipy's own",
            "note": "norm.cdf(x) == special.ndtr(x) on every dumped input (checked), "
                    "so this pins the ufunc the pipeline runs",
        },
        "checks": checks,
        "sweep": {"input": vector(inputs), "output": vector(outputs)},
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "scipy_ndtr_ref.json"
    path.write_text(json.dumps(dump, indent=1, sort_keys=True) + "\n")
    print(f"wrote {path.relative_to(REPO_ROOT)}  sweep={dump['sweep']['input']['count']}")
    for key, value in checks.items():
        print(f"  check {key} = {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
