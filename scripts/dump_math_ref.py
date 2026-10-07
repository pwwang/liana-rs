#!/usr/bin/env python3
"""Dump numpy 2.5.3's f32 `log`/`exp` kernels as bit-exact parity references.

**Which kernel.** `numpy/_core/src/umath/loops_exponent_log.dispatch.c.src`
(checked out at tag v2.5.3) defines `FLOAT_log`/`FLOAT_exp` for exactly two
dispatch targets — `X86_V4` and `X86_V3` — and the `X86_V3` compile defines
`NPY_HAVE_AVX2` + `NPY_HAVE_FMA3`, selecting the `SIMD_AVX2_FMA3` bodies
`simd_log_FLOAT` / `simd_exp_FLOAT` (numpy's own Cody-Waite + rational-
polynomial kernels, constants in `npy_simd_data.h`). Google Highway is *not*
involved: numpy vendors it (`numpy/_core/src/highway`) only for qsort,
trigonometric, hyperbolic and logical loops. This machine reports
`found: X86_V3` (`np.show_runtime()`), no AVX-512, so the AVX2+FMA3 bodies are
what a f32 `np.log`/`np.exp` call runs.

**What is dumped.** Two input families per kernel, as raw IEEE-754 bit
patterns, plus numpy's own outputs:

    pipeline  the geometric_mean column path, exactly: the oracle CSV's
              `ligand_means`/`receptor_means` f32 columns (440 rows) stacked
              to a (2, 440) array -> `np.log` (880 inputs) ->
              `np.mean(axis=0)` (440 f32 means = the exp inputs) -> `np.exp`.
              The script asserts this chain reproduces the oracle `lr_gmeans`
              column bit for bit and equals `scipy.stats.gmean` on the stack.
    sweep     >= 10k mixed inputs per kernel: full-space bit patterns,
              log-uniform magnitudes, ±ulp churn around 1/ln2 boundaries,
              subnormals on both sides of FLT_MIN, the NEP mask boundaries
              (xmax/xmin of exp, sqrt(1/2) of log), and NaN/inf/±0.

Every vector carries a sha256 over the big-endian u32 words, so the Rust test
pins the bit patterns, not any decimal spelling.

Writes `testdata/math_ref/numpy_math_ref.json`.

Run with the oracle venv interpreter (numpy 2.5.3):

    /home/pwwang/p0a/venv/bin/python scripts/dump_math_ref.py
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib

import numpy as np

REPO_ROOT = pathlib.Path(os.environ.get("REPO_ROOT", pathlib.Path(__file__).resolve().parents[1]))
OUT_DIR = REPO_ROOT / "testdata" / "math_ref"
CSV = REPO_ROOT / "testdata" / "expected" / "synthetic__geometric_mean__p100.csv"

# The exp kernel's NEP saturation bounds and the log kernel's denormal switch,
# as literal f32 (npy_simd_data.h / the kernel bodies).
XMAX = np.float32(88.72283935546875)
XMIN = np.float32(-103.97208404541015625)
FLT_MIN = np.float32(np.finfo(np.float32).tiny)
LN2 = np.float32(0.693147180559945309417232121458176568)


def f32(bits: np.ndarray) -> np.ndarray:
    """u32 (or int) array -> same-shape f32 by bit pattern."""
    return np.asarray(bits, dtype=np.uint32).view(np.float32)


def ulp_sweep(center: np.float32, k: int) -> np.ndarray:
    """center ± k ulp, as f32 bit arithmetic (needs center > 0)."""
    bits = np.uint32(center.view(np.uint32))
    steps = np.arange(-k, k + 1, dtype=np.int64)
    return f32((bits.astype(np.int64) + steps) % 2**32)


def log_sweep(rng: np.random.Generator) -> np.ndarray:
    parts = [
        # full-space random bit patterns: negatives, ±0, ±inf, NaNs, subnormals
        f32(rng.integers(0, 2**32, 4000, dtype=np.uint32)),
        # log-uniform positive normals across the full exponent range
        np.exp2(rng.uniform(-126.0, 127.0, 2500).astype(np.float32)).astype(np.float32),
        rng.uniform(0.0, 1.0, 1000).astype(np.float32),
        rng.uniform(1.0, 2.0, 1000).astype(np.float32),
        # ±ulp churn around 1.0 (the poly's expansion point)
        ulp_sweep(np.float32(1.0), 500),
        # subnormals up to FLT_MIN and its neighbours
        f32(rng.integers(1, 0x800000, 1500, dtype=np.uint32)),
        f32([0x00000001, 0x00000002, 0x00400000, 0x007FFFFF, 0x00800000, 0x00800001]),
        # every power of two and its ±1 ulp neighbours
        f32(np.concatenate([
            ((np.arange(-126, 128, dtype=np.int64) + 127) << 23) + d for d in (-1, 0, 1)
        ])),
        # the sqrt(1/2) normalisation switch, ±1 ulp
        ulp_sweep(np.float32(0.70710678118654752440), 4),
        # magnitudes drawn across every decimal scale
        (rng.random(1500) * 10.0 ** rng.integers(-38, 39, 1500)).astype(np.float32),
        # specials
        f32([0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000,
             0x7F800001, 0x7FBFFFFF, 0x7F7FFFFF, 0xFF7FFFFF, 0x3F800000, 0xBF800000,
             0x40000000, 0x3DCCCCCD, 0x41200000, 0x42C80000]),
    ]
    return np.concatenate(parts)


def exp_sweep(rng: np.random.Generator) -> np.ndarray:
    parts = [
        # the whole non-saturating range
        rng.uniform(-104.0, 89.0, 4000).astype(np.float32),
        # dense in the poly's main reduction cell and [0, ln2]
        rng.uniform(-1.0, 1.0, 1500).astype(np.float32),
        rng.uniform(0.0, 0.6932, 1000).astype(np.float32),
        # ±500 ulp around the overflow / saturation masks
        ulp_sweep(XMAX, 500),
        ulp_sweep(XMIN, 500),
        # the denormal-result region (quadrant <= -125) and its surroundings
        rng.uniform(-103.97, -87.3, 2000).astype(np.float32),
        # k*ln2 (range-reduction cell edges) and just above them
        (np.arange(-150, 129, dtype=np.float32) * LN2),
        (np.arange(-150, 129, dtype=np.float32) * LN2 + np.float32(1e-3)),
        # the scalef denormal cutoff at quadrant = -125, -124, -126
        ulp_sweep(np.float32(-125.0) * LN2, 50),
        ulp_sweep(np.float32(-124.0) * LN2, 20),
        ulp_sweep(np.float32(-126.0) * LN2, 20),
        # full-space random bit patterns (mostly saturate; pins the masks)
        f32(rng.integers(0, 2**32, 2000, dtype=np.uint32)),
        # specials
        f32([0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000,
             0x7F800001, 0x3F800000, 0xBF800000, 0x3F317218, 0xBF317218,
             0x42B17217, 0x42B17218, 0xC2CF0000, 0xC2CF0A51, 0xC2CF0A52,
             0x00000001, 0x80000001, 0x4B000000, 0xCB000000]),
    ]
    return np.concatenate(parts)


def vector(values: np.ndarray) -> dict:
    """One bit-exact vector: hex blob of the BE u32 words + sha256.

    The hex is the array's own big-endian buffer — one source for both the
    blob and the hash. (Per-scalar `.tobytes()` would silently emit native
    order: numpy scalars ignore the dtype's byte order.)
    """
    words = np.asarray(values, dtype=np.float32).view(np.uint32)
    be = words.astype(">u4").tobytes()
    return {
        "count": int(words.size),
        "sha256": hashlib.sha256(be).hexdigest(),
        "hex": be.hex(),
    }


def main() -> int:
    rng = np.random.default_rng(1337)

    # --- the geometric_mean column path, end to end ------------------------
    text = CSV.read_text().splitlines()
    header = text[0].split(",")
    lig_col, rec_col, gm_col = (header.index(c) for c in
                                ("ligand_means", "receptor_means", "lr_gmeans"))
    lig = np.array([line.split(",")[lig_col] for line in text[1:]], dtype=np.float32)
    rec = np.array([line.split(",")[rec_col] for line in text[1:]], dtype=np.float32)
    oracle = np.array([line.split(",")[gm_col] for line in text[1:]], dtype=np.float32)

    stack = np.stack([lig, rec])          # (2, n) C-contiguous f32
    log_in = stack.ravel()                # 880: ligand row then receptor row
    log_out = np.log(stack).ravel()
    exp_in = np.mean(np.log(stack), axis=0)   # f32 two-element mean per row
    exp_out = np.exp(exp_in)

    # the chain must reproduce the oracle column exactly
    assert np.array_equal(exp_out.view(np.uint32), oracle.view(np.uint32)), \
        "np.log/np.mean/np.exp over the CSV's own mean columns != oracle lr_gmeans"
    from scipy.stats import gmean
    assert np.array_equal(gmean(stack, axis=0).view(np.uint32), oracle.view(np.uint32)), \
        "scipy gmean over the CSV's own mean columns != oracle lr_gmeans"

    # --- kernel-identity probes (recorded, and asserted where exact) -------
    def scalar_vs_array(x: np.ndarray, f) -> int:
        return sum(int(f(np.float32(v)).view(np.uint32) != o.view(np.uint32))
                   for v, o in zip(x, f(x)))

    f64 = np.log(stack.astype(np.float64)).astype(np.float32).ravel()
    checks = {
        "pipeline_rows": len(oracle),
        "pipeline_chain_matches_oracle": True,
        "pipeline_chain_matches_scipy_gmean": True,
        "array_vs_scalar_log_diffs": scalar_vs_array(log_in, np.log),
        "array_vs_scalar_exp_diffs": scalar_vs_array(exp_in, np.exp),
        "log_vs_rounded_f64_diffs": int((log_out.view(np.uint32)
                                        != f64.view(np.uint32)).sum()),
        "exp_vs_rounded_f64_diffs": int((exp_out.view(np.uint32)
                                        != np.exp(exp_in.astype(np.float64))
                                        .astype(np.float32).view(np.uint32)).sum()),
    }

    # --- sweeps ------------------------------------------------------------
    log_sweep_in = log_sweep(rng)
    exp_sweep_in = exp_sweep(rng)
    assert log_sweep_in.size >= 10_000 and exp_sweep_in.size >= 10_000, "sweep size contract"
    with np.errstate(all="ignore"):  # the specials log/exp to inf/nan by design
        log_sweep_out, exp_sweep_out = np.log(log_sweep_in), np.exp(exp_sweep_in)
        sweep_checks = {
            "log_sweep_vs_rounded_f64_diffs": int(
                (log_sweep_out.view(np.uint32)
                 != np.log(log_sweep_in.astype(np.float64)).astype(np.float32)
                 .view(np.uint32)).sum()),
            "exp_sweep_vs_rounded_f64_diffs": int(
                (exp_sweep_out.view(np.uint32)
                 != np.exp(exp_sweep_in.astype(np.float64)).astype(np.float32)
                 .view(np.uint32)).sum()),
        }

    dump = {
        "generated_by": "scripts/dump_math_ref.py",
        "numpy_version": np.__version__,
        "kernel": {
            "source": "numpy v2.5.3 numpy/_core/src/umath/loops_exponent_log.dispatch.c.src",
            "implementation": "simd_log_FLOAT / simd_exp_FLOAT, SIMD_AVX2_FMA3 body",
            "dispatch_target": "X86_V3 (AVX2+FMA3; runtime reports found X86_V3, no AVX-512)",
            "constants": "numpy/_core/src/umath/npy_simd_data.h",
            "note": "Google Highway is not used for exp/log (qsort/trig/hyperbolic only)",
        },
        "checks": checks | sweep_checks,
        "log": {
            "pipeline": {"input": vector(log_in), "output": vector(log_out)},
            "sweep": {"input": vector(log_sweep_in), "output": vector(log_sweep_out)},
        },
        "exp": {
            "pipeline": {"input": vector(exp_in), "output": vector(exp_out)},
            "sweep": {"input": vector(exp_sweep_in), "output": vector(exp_sweep_out)},
        },
        "gmean_pipeline": {
            "ligand_means": vector(lig),
            "receptor_means": vector(rec),
            "oracle_lr_gmeans": vector(oracle),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "numpy_math_ref.json"
    path.write_text(json.dumps(dump, indent=1, sort_keys=True) + "\n")
    print(f"wrote {path.relative_to(REPO_ROOT)}  "
          f"log pipeline={dump['log']['pipeline']['input']['count']} "
          f"sweep={dump['log']['sweep']['input']['count']}  "
          f"exp pipeline={dump['exp']['pipeline']['input']['count']} "
          f"sweep={dump['exp']['sweep']['input']['count']}")
    for key, value in (checks | sweep_checks).items():
        print(f"  check {key} = {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
