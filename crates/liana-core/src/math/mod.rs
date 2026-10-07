//! Bit-exact scalar ports of numpy 2.5.3's `f32` `log`/`exp` kernels and
//! `f32` pairwise reductions, and scipy 1.18.1's f64 `ndtr`.
//!
//! numpy's `np.log`/`np.exp` on `f32` arrays do not call the platform libm:
//! `loops_exponent_log.dispatch.c.src` (tag v2.5.3) defines `FLOAT_log` /
//! `FLOAT_exp` for exactly two dispatch targets, `X86_V4` and `X86_V3`, and
//! the `X86_V3` compile (`NPY_HAVE_AVX2 && NPY_HAVE_FMA3`, reported as
//! `found: X86_V3` here) selects numpy's own `simd_log_FLOAT` /
//! `simd_exp_FLOAT` bodies — Cody-Waite range reduction plus a rational
//! polynomial, constants in `npy_simd_data.h`. Google Highway is not
//! involved: numpy vendors it only for qsort, trigonometric, hyperbolic and
//! logical loops.
//!
//! The `ndtr` port (`ndtr.rs`) is scipy's cephes normal CDF, the kernel
//! behind `scseqcomm`'s `*_cdf` columns (`scipy.stats.norm.cdf` through
//! `_gene_cdf`, `method/sc/_liana_pipe.py:755-767`).
//!
//! The `pairwise` port (`pairwise.rs`) is numpy's `f32` pairwise reduction,
//! `np.sum`/`np.std`, which back `scseqcomm`'s cluster statistics
//! (`_cluster_stats`, `method/sc/_liana_pipe.py:742-751`).
//!
//! These ports are 1:1 transcriptions: same constants (as raw bit patterns),
//! same operation order, `f32::mul_add` for every `_mm256_fmadd_ps` and plain
//! `f32` division for `_mm256_div_ps` (both correctly rounded, so neither
//! introduces a divergence), the same mask/blend fallbacks. Bit-exactness is
//! gated by `tests/math_parity.rs` against `testdata/math_ref/` — 100% match
//! required, no tolerance.
//!
//! This exists because the `geometric_mean` score column is
//! `exp((log(l) + log(r)) / 2)` evaluated in `f32` by the oracle
//! (`method/sc/_geometric_mean.py:28` through scipy's `gmean`); Rust's libm
//! disagrees with numpy's kernels by up to 4 ulp on 182 of the 440 rows of
//! the parity fixture (W3 D2).
//!
//! Scope: exactness holds for the X86_V3 dispatch target the oracle runs on
//! (and any target where numpy resolves the same kernels — a numpy build
//! dispatching to AVX-512/SVML would call different ones).

pub mod expf;
pub mod logf;
pub mod ndtr;
pub mod pairwise;

pub use expf::expf;
pub use logf::logf;
pub use ndtr::ndtr;
pub use pairwise::{std_f32, sum_f32};
