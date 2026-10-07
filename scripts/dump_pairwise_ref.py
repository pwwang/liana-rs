#!/usr/bin/env python3
"""Dump numpy 2.5.3's `f32` pairwise reductions as bit-exact parity references.

**Which kernel.** `np.sum` on a contiguous `f32` array does not fold the
elements sequentially: numpy's pairwise reduction (`FLOAT_pairwise_sum`,
`numpy/_core/src/umath/loops_utils.h.src`) sums eight lane accumulators in
blocks (block size 128) and recurses on halves above that. Its SIMD dispatch
(`loops_arithm_fp.dispatch.c.src`) keeps the same block/halves shape with
zero-initialised lanes, so the model dumped here — zero-init lanes, `+0.0`
scalar start, `n < 8` sequential, `n <= 128` blocked, recursion above — is
what `np.add.reduce` runs. `np.std` (`numpy/_core/_methods.py::_var`) is that
sum twice: the `f32` mean, then the `f32` sum of the squared deviations,
divided by `n` in `f32` and rooted.

**Callers.** `scseqcomm`'s cluster statistics (`_cluster_stats`,
`liana/method/sc/_liana_pipe.py:742-751`): `temp.mean()` for a cluster's
sparse block and `np.std(temp.toarray())` for its dense one; the dense
flattening is row-major, which is the order dumped here.

**What is dumped.** One case per (size, value profile): `n = 0..=200` — the
whole range up to the block bound, so the `n < 8` / block / tail branches are
all pinned — plus block-and-recursion sizes (256, 4096, 22968 ≈ the fixture's
own cluster block); value profiles: uniform, zeros-heavy (the fixture's
shape), signed, tiny/subnormal, huge, all-`-0.0`, mixed `±0.0`, an `inf`/NaN
mix, and a `-0.0` block with one interior value. Every case carries numpy's
`np.sum(dtype=f32)` and `np.std` output as raw bit patterns, and the script's
own model of the kernel is asserted against numpy on every dumped input, so
the dump pins one reduction path.

The real `scseqcomm` inputs are not a separate family here: they are the
fixture's own cluster blocks, checked end to end by the pipe gate
(`crates/liana-core/tests/pipe_parity.rs`,
`scseqcomm_pipeline_matches_the_oracle_csv`) against the oracle CSV's `*_cdf`
and `inter_score` columns at zero tolerance.

Writes `testdata/math_ref/numpy_pairwise_ref.json`.

Run with the oracle venv interpreter (numpy 2.5.3):

    /home/pwwang/p0a/venv/bin/python scripts/dump_pairwise_ref.py
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib

import numpy as np

REPO_ROOT = pathlib.Path(os.environ.get("REPO_ROOT", pathlib.Path(__file__).resolve().parents[1]))
OUT_DIR = REPO_ROOT / "testdata" / "math_ref"

F32 = np.float32

# The sizes every profile runs on, and the profile list, kept in one place so
# the Rust test can restate the contract without re-deriving it.
SMALL_SIZES = list(range(0, 201))
LARGE_SIZES = [256, 4096, 22968]
PROFILES = [
    "uniform",
    "zeros",
    "signed",
    "tiny",
    "huge",
    "negzero",
    "zero_signs",
    "mostly_negzero",
    "mixed",
]
# Above the block bound the branch structure is size-determined, so the large
# sizes keep only the profiles that differ in outcome class.
LARGE_PROFILES = ["uniform", "zeros", "negzero", "mixed"]


# --- the kernel model (mirrors crates/liana-core/src/math/pairwise.rs) -----


def sum_f32(values: list) -> np.float32:
    a = [F32(v) for v in values]
    n = len(a)
    if n == 0:
        return F32(0.0)  # the reduce's identity, before the loop is entered
    if n < 8:
        res = F32(0.0)
        for v in a:
            res = F32(res + v)
        return res
    if n <= 128:
        r = [F32(0.0)] * 8
        i = 0
        while i < n - (n % 8):
            for j in range(8):
                r[j] = F32(r[j] + a[i + j])
            i += 8
        res = F32(F32(F32(r[0] + r[1]) + F32(r[2] + r[3])) + F32(F32(r[4] + r[5]) + F32(r[6] + r[7])))
        while i < n:
            res = F32(res + a[i])
            i += 1
        return res
    n2 = n // 2
    n2 -= n2 % 8
    return F32(sum_f32(a[:n2]) + sum_f32(a[n2:]))


def std_f32(values: list) -> np.float32:
    a = [F32(v) for v in values]
    n = F32(len(a))
    m = F32(sum_f32(a) / n)
    squares = [F32(F32(v - m) * F32(v - m)) for v in a]
    return F32(np.sqrt(F32(sum_f32(squares) / n)))


def profile(rng: np.random.Generator, n: int, kind: str) -> np.ndarray:
    if kind == "zeros":
        v = rng.uniform(0.0, 5.0, n).astype(F32)
        v[rng.random(n) < 0.9] = F32(0.0)
        return v
    if kind == "uniform":
        return rng.uniform(0.0, 5.0, n).astype(F32)
    if kind == "signed":
        return rng.uniform(-5.0, 5.0, n).astype(F32)
    if kind == "tiny":
        return (rng.uniform(0.0, 1.0, n) * 10.0 ** rng.integers(-38, -20, n)).astype(F32)
    if kind == "huge":
        return (rng.uniform(1.0, 2.0, n) * 10.0 ** rng.integers(30, 38, n)).astype(F32)
    if kind == "negzero":
        return np.full(n, F32(-0.0), dtype=F32)
    if kind == "zero_signs":
        return F32(rng.choice([0.0, -0.0], n))
    if kind == "mostly_negzero":
        v = np.full(n, F32(-0.0), dtype=F32)
        if n >= 3:
            v[n // 2] = F32(2.5)
        return v
    if kind == "mixed":
        v = rng.uniform(-5.0, 5.0, n).astype(F32)
        if n:
            v[0] = F32(np.inf)
            v[-1] = F32(-np.inf)
        return v
    raise AssertionError(kind)


def vector(values: np.ndarray) -> dict:
    """One bit-exact vector: hex blob of the BE u32 words + sha256."""
    words = np.asarray(values, dtype=np.float32).view(np.uint32)
    be = words.astype(">u4").tobytes()
    return {
        "count": int(words.size),
        "sha256": hashlib.sha256(be).hexdigest(),
        "hex": be.hex(),
    }


def scalar(value: np.float32) -> str:
    return f"0x{int(np.asarray(value, dtype=np.float32).view(np.uint32)):08X}"


def main() -> int:
    rng = np.random.default_rng(1337)
    cases = []
    sum_diffs = std_diffs = 0
    with np.errstate(all="ignore"):  # inf/NaN profiles overflow by design
        for n in SMALL_SIZES + LARGE_SIZES:
            for kind in PROFILES if n <= 200 else LARGE_PROFILES:
                values = profile(rng, n, kind)
                total = np.sum(values, dtype=F32)
                deviation = None if n == 0 else np.std(values)
                if np.asarray(sum_f32(list(values))).view(np.uint32) != np.asarray(total).view(np.uint32):
                    sum_diffs += 1
                if deviation is not None and (
                    np.asarray(std_f32(list(values))).view(np.uint32)
                    != np.asarray(deviation).view(np.uint32)
                ):
                    std_diffs += 1
                cases.append(
                    {
                        "n": n,
                        "kind": kind,
                        "input": vector(values),
                        "sum": scalar(total),
                        "std": None if n == 0 else scalar(deviation),
                    }
                )

    values_total = sum(case["input"]["count"] for case in cases)
    assert sum_diffs == 0 and std_diffs == 0, (
        "the dumped kernel model disagrees with numpy: "
        f"sum {sum_diffs}, std {std_diffs}"
    )
    assert len(cases) >= 500 and values_total >= 10_000, "sweep size contract"

    dump = {
        "generated_by": "scripts/dump_pairwise_ref.py",
        "numpy_version": np.__version__,
        "kernel": {
            "source": "numpy 2.5.3 numpy/_core/src/umath/loops_utils.h.src "
                      "(FLOAT_pairwise_sum), SIMD block shape per "
                      "loops_arithm_fp.dispatch.c.src",
            "called_by": "liana/method/sc/_liana_pipe.py:742-751 (_cluster_stats)",
            "implementation": "np.sum: zero-init 8-lane blocks (block 128), "
                              "sequential below 8, halves recursion above; "
                              "np.std: that sum for mean and squares, /n and "
                              "sqrt in f32",
        },
        "checks": {
            "cases": len(cases),
            "values": values_total,
            "sum_model_vs_numpy_diffs": sum_diffs,
            "std_model_vs_numpy_diffs": std_diffs,
        },
        "sizes": {"small": SMALL_SIZES, "large": LARGE_SIZES},
        "profiles": PROFILES,
        "cases": cases,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "numpy_pairwise_ref.json"
    path.write_text(json.dumps(dump, indent=1, sort_keys=True) + "\n")
    print(f"wrote {path.relative_to(REPO_ROOT)}  cases={len(cases)} values={values_total}")
    for key, value in dump["checks"].items():
        print(f"  check {key} = {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
