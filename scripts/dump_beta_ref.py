#!/usr/bin/env python3
"""Dump scipy 1.18.1's `betainc` as a bit-exact parity reference for the RRA
kernel behind `rank_aggregate`.

**Which kernel.** `liana`'s robust rank aggregation reduces each interaction
to `scipy.stats.beta.cdf(x, a, b)` (`liana/_core/_pipe_utils/_aggregate.py:212`,
inside `_rho_scores`), which is `scipy.special.betainc`. In scipy 1.18.1 that
ufunc is Boost, not cephes — `scipy/special/functions.json` maps
`betainc -> boost_special_functions.h++: ibeta_double` — so the algorithm is
`boost::math::detail::ibeta_imp` (`subprojects/boost_math/.../beta.hpp:1198`)
with scipy's own argument wrapper (`ibeta_wrap`, `boost_special_functions.h:72`).

The RRA shapes are always `a = j + 1`, `b = k - j + 1` for a column index `j`
of a rank matrix with `k` columns, i.e. small positive integers with
`a + b = k + 1` (`k` is 3 or 4 for `rank_aggregate`, whose magnitude and
specificity specs each collapse to 3–4 distinct score columns). For integers
`ibeta_imp` takes the finite-sum branch: `a - 1` and `b + a - 1` are the
binomial ccdf's `k`/`n`, evaluated by `binomial_ccdf` (`beta.hpp:1078`).
`binomial_ccdf`'s only libm calls are `pow`, and the `b == 1` sub-branch adds
`expm1`/`log1p`; the rest is plain `f64` arithmetic in the header's order.

**What is dumped.** Four bit-exact input/output vectors per family, all as raw
IEEE-754 bit patterns (`a`, `b` and `x` in, `y` out):

* `rra_p100`, `rra_p1000` — the *actual* `(a, b, x)` triples the pipeline
  evaluates for the two parity fixtures. Captured by wrapping
  `_aggregate.beta`, so these are the real values, not a reconstruction.
* `sweep` — every integer pair with `1 <= a, b <= 4` plus the wider pairs up to
  `a + b = 8`, each against a boundary-rich `x` pool: the `lambda` switch at
  `x = a / (a + b)` and the `y < 0.5` switch at `x = 0.5` ± 64 ulp, dense
  linspace/logspace coverage, `x -> 0` and `x -> 1` tails, and `{0, 1}`.
* `limits` — the wrapper's guard clauses (`a`/`b` zero, ±inf, NaN, `x` outside
  `[0, 1]`) and both `x` endpoints, which return before `ibeta_imp`.

Every vector carries a sha256 over its big-endian u64 words, so the Rust test
pins bit patterns, not any decimal spelling.

Writes `testdata/math_ref/scipy_betainc_ref.json`.

Run with the oracle venv interpreter (scipy 1.18.1):

    /home/pwwang/p0a/venv/bin/python scripts/dump_beta_ref.py
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import pathlib
import sys

import numpy as np
import scipy
from scipy.special import betainc

REPO_ROOT = pathlib.Path(os.environ.get("REPO_ROOT", pathlib.Path(__file__).resolve().parents[1]))
OUT_DIR = REPO_ROOT / "testdata" / "math_ref"


def f64(bits) -> np.ndarray:
    """u64 (or float) array -> same-shape f64 by bit pattern."""
    return np.asarray(bits, dtype=np.uint64).view(np.float64)


def ulp_sweep(center: float, k: int) -> np.ndarray:
    """center ± k ulp, as f64 bit arithmetic (Python ints, kept in u64 range)."""
    bits = int(np.float64(center).view(np.uint64))
    return f64(np.array([(bits + step) % 2**64 for step in range(-k, k + 1)], dtype=np.uint64))


def pair_xs(a: int, b: int, rng: np.random.Generator) -> np.ndarray:
    """Boundary-rich x pool for one (a, b) pair."""
    parts = [
        np.linspace(0.0, 1.0, 129),  # includes both endpoints
        np.linspace(1e-9, 0.5, 129),
        1.0 - np.linspace(1e-9, 0.5, 129),
        ulp_sweep(a / (a + b), 64),  # the lambda = 0 switch
        ulp_sweep(0.5, 64),  # the b == 1 y < 0.5 switch
        rng.random(256) ** 3,  # skewed toward 0
        1.0 - rng.random(256) ** 3,  # skewed toward 1
        np.exp2(rng.uniform(-30.0, 0.0, 128)),  # deep into the underflow tail
    ]
    return np.unique(np.clip(np.concatenate(parts), 0.0, 1.0))


def sweep(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pairs = [(a, b) for a in range(1, 5) for b in range(1, 5)]
    pairs += [(a, 8 - a) for a in range(1, 8)]  # a + b = 8 margin, unused by RRA
    pairs += [(8 - a, a) for a in range(1, 8)]
    aa, bb, xx = [], [], []
    for a, b in sorted(set(pairs)):
        xs = pair_xs(a, b, rng)
        aa.append(np.full(xs.size, float(a)))
        bb.append(np.full(xs.size, float(b)))
        xx.append(xs)
    return np.concatenate(aa), np.concatenate(bb), np.concatenate(xx)


def limits() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cases = [
        # x endpoints (return before ibeta_imp)
        (2.0, 3.0, 0.0), (2.0, 3.0, 1.0), (1.0, 1.0, 0.0), (1.0, 1.0, 1.0),
        (1.0, 4.0, 0.0), (1.0, 4.0, 1.0), (4.0, 1.0, 0.0), (4.0, 1.0, 1.0),
        # a == 0 / b == 0 and the infinities: scipy's ibeta_wrap limits
        (0.0, 0.0, 0.5), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
        (0.0, 1.0, 0.5), (0.0, 1.0, 0.0), (1.0, 0.0, 0.5), (1.0, 0.0, 1.0),
        (np.inf, 1.0, 0.5), (1.0, np.inf, 0.5), (np.inf, np.inf, 0.5),
        # out of domain
        (-1.0, 1.0, 0.5), (1.0, -1.0, 0.5), (1.0, 1.0, -0.5), (1.0, 1.0, 1.5),
        # NaN in each slot
        (np.nan, 1.0, 0.5), (1.0, np.nan, 0.5), (1.0, 1.0, np.nan),
    ]
    arr = np.array(cases, dtype=np.float64)
    return arr[:, 0].copy(), arr[:, 1].copy(), arr[:, 2].copy()


def rra_triples(n_perms: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The real (a, b, x) triples `rank_aggregate` feeds to `beta.cdf`."""
    import anndata as ad
    import pandas as pd

    import liana as li
    from liana._core._pipe_utils import _aggregate as ag

    adata = ad.read_h5ad(REPO_ROOT / "testdata" / "fixtures" / "synthetic.h5ad")
    resource = pd.DataFrame(
        itertools.product(adata.var_names, adata.var_names), columns=["ligand", "receptor"]
    )
    resource = resource[resource["ligand"] != resource["receptor"]].reset_index(drop=True)

    real_beta = ag.beta
    triples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    class Recorder:
        """`beta`, with `.cdf` recording what `_rho_scores` asks for."""

        def __getattr__(self, name):
            return getattr(real_beta, name)

        def cdf(self, x, a, b):
            triples.append(
                (
                    np.asarray(a, dtype=np.float64).ravel(),
                    np.asarray(b, dtype=np.float64).ravel(),
                    np.asarray(x, dtype=np.float64).ravel(),
                )
            )
            return real_beta.cdf(x, a, b)

    ag.beta = Recorder()
    try:
        li.mt.rank_aggregate(
            adata,
            groupby="cell_type",
            resource_name="consensus",
            resource=resource,
            n_perms=n_perms,
            seed=1337,
            n_jobs=1,
            inplace=False,
            verbose=False,
        )
    finally:
        ag.beta = real_beta

    assert len(triples) == 2, f"expected one beta.cdf call per consensus option, got {len(triples)}"
    return (
        np.concatenate([t[0] for t in triples]),
        np.concatenate([t[1] for t in triples]),
        np.concatenate([t[2] for t in triples]),
    )


def vector(values: np.ndarray) -> dict:
    """One bit-exact vector: hex blob of the BE u64 words + sha256."""
    words = np.asarray(values, dtype=np.float64).view(np.uint64)
    be = words.astype(">u8").tobytes()
    return {
        "count": int(words.size),
        "sha256": hashlib.sha256(be).hexdigest(),
        "hex": be.hex(),
    }


def family(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> dict:
    with np.errstate(all="ignore"):
        y = betainc(a, b, x)
    return {"a": vector(a), "b": vector(b), "x": vector(x), "y": vector(y)}


def main() -> int:
    rng = np.random.default_rng(1337)
    dump = {
        "generated_by": "scripts/dump_beta_ref.py",
        "scipy_version": scipy.__version__,
        "kernel": {
            "source": "scipy 1.18.1 scipy/special/functions.json -> "
                      "boost_special_functions.h++: ibeta_double -> "
                      "boost/math/special_functions/beta.hpp detail::ibeta_imp",
            "called_by": "liana/_core/_pipe_utils/_aggregate.py:212 (_rho_scores)",
            "implementation": "For the integer (a, b) with a + b <= 8 the RRA "
                              "always hits ibeta_imp's binomial branch: "
                              "binomial_ccdf(b + a - 1, a - 1, x, 1 - x) with "
                              "pow, and expm1/log1p on the b == 1 sub-branch",
        },
        "families": {
            "rra_p100": family(*rra_triples(100)),
            "rra_p1000": family(*rra_triples(1000)),
            "sweep": family(*sweep(rng)),
            "limits": family(*limits()),
        },
    }
    for name, fam in dump["families"].items():
        assert fam["a"]["count"] == fam["x"]["count"], name
        print(f"  {name}: {fam['a']['count']} triples")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "scipy_betainc_ref.json"
    path.write_text(json.dumps(dump, indent=1, sort_keys=True) + "\n")
    print(f"wrote {path.relative_to(REPO_ROOT)} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
