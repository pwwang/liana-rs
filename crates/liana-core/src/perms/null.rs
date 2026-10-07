//! Permutation nulls — `_get_means_perms` (the `aggregation="mean"` path of
//! `liana/_core/_pipe_utils/_get_mean_perms.py`) and `_calculate_pvals`.
//!
//! Ground truth: `testdata/pipe_ref/synthetic__cellphonedb.json` (`n_perms`),
//! dumped from the pinned oracle by `scripts/dump_pipe_ref.py`.

use crate::prep::Prep;

/// `_TIE_RTOL` (`_get_mean_perms.py:38`): how close a permuted mean has to be
/// to the observed one to count as tied with it.
pub(crate) const TIE_RTOL: f64 = 1e-6;

/// The `(n_perms, n_labels, n_vars)` cube of per-cluster permuted means,
/// row-major — `_generate_perms_cube` with `aggregation="mean"`
/// (`_get_mean_perms.py:274-325`).
///
/// `perms` is the `(n_perms, n_obs)` shuffle matrix from
/// [`crate::perms::rng::permutation_matrix`]: position `j` of permutation `p`
/// contributes the row `perms[p, j]` to the label that position `j` itself
/// carries (`_perm_group_sums`, `:198-203`). Sums accumulate in `f64` over the
/// `f32` stored values in stored-entry order, and each label's row is divided
/// by that label's cell count in `f64` (`:303-305`).
///
/// Liana's chunking of the draw (`_chunk_permutations`, `:258-271`) and the
/// kernel's `prange` over permutations only bound memory: every permutation is
/// an independent accumulation over one RNG stream, so the cube is a function
/// of `seed` and `n_perms` alone.
pub fn means_cube(prep: &Prep, perms: &[u32], n_perms: usize) -> Vec<f64> {
    let n_obs = prep.x.n_rows;
    assert_eq!(perms.len(), n_perms * n_obs, "permutation matrix shape");
    let n_labels = prep.n_labels();
    let n_vars = prep.n_vars();
    let mut cube = vec![0f64; n_perms * n_labels * n_vars];

    for p in 0..n_perms {
        for (position, &row) in perms[p * n_obs..(p + 1) * n_obs].iter().enumerate() {
            let label = prep.cell_cluster[position] as usize;
            let row = row as usize;
            let entries = prep.x.indptr[row]..prep.x.indptr[row + 1];
            for k in entries {
                let gene = prep.x.indices[k] as usize;
                cube[(p * n_labels + label) * n_vars + gene] += f64::from(prep.x.data[k]);
            }
        }
    }

    for p in 0..n_perms {
        for label in 0..n_labels {
            let count = prep.counts[label] as f64;
            for value in
                &mut cube[(p * n_labels + label) * n_vars..(p * n_labels + label + 1) * n_vars]
            {
                *value /= count;
            }
        }
    }
    cube
}

/// `_calculate_pvals` (`_get_mean_perms.py:367-397`) for the stacked
/// `(2, n_perms, n_rows)` permutation statistic: the fraction of permutations
/// whose combined ligand/receptor statistic is at least the observed `truth`.
///
/// `ligand` and `receptor` are the two stacked statistics — what the caller
/// selected as `perms[:, source, ligand]` and `perms[:, target, receptor]` —
/// laid out `(n_rows, n_perms)` row-major. `combine` is the method's scorer,
/// what `_calculate_pvals` takes as `_score_fn` and applies to the stack's
/// leading axis. A permutation counts when its statistic is `>= truth` or
/// within `rtol = 1e-6` of it — `np.isclose(..., atol=0.0)`, the tie tolerance
/// of the `f32` scores — and the count is divided by `n_perms` in `f64`.
fn exceed_fraction(
    ligand: &[f64],
    receptor: &[f64],
    truth: &[f32],
    n_perms: usize,
    combine: impl Fn(f64, f64) -> f64,
) -> Vec<f64> {
    assert_eq!(ligand.len(), receptor.len(), "statistic shapes");
    assert_eq!(
        ligand.len() % n_perms,
        0,
        "n_perms does not divide the stats"
    );
    let n_rows = ligand.len() / n_perms;
    assert_eq!(truth.len(), n_rows, "truth length");

    (0..n_rows)
        .map(|row| {
            let observed = f64::from(truth[row]);
            let nulls = &ligand[row * n_perms..(row + 1) * n_perms];
            let other = &receptor[row * n_perms..(row + 1) * n_perms];
            let exceeds = nulls
                .iter()
                .zip(other)
                .filter(|&(&ligand, &receptor)| {
                    let stat = combine(ligand, receptor);
                    stat >= observed || (stat - observed).abs() <= TIE_RTOL * observed.abs()
                })
                .count();
            exceeds as f64 / n_perms as f64
        })
        .collect()
}

/// `_calculate_pvals` with cellphonedb's `_score_fn`: the arithmetic mean of
/// the two nulls, `np.mean(stack, axis=0)` (`method/sc/_cellphonedb.py:31-33`).
pub fn pvals(ligand: &[f64], receptor: &[f64], truth: &[f32], n_perms: usize) -> Vec<f64> {
    exceed_fraction(ligand, receptor, truth, n_perms, |ligand, receptor| {
        (ligand + receptor) / 2.0
    })
}

/// `_calculate_pvals` with geometric mean's `_score_fn`: scipy's `gmean` over
/// the two nulls (`method/sc/_geometric_mean.py:31` → `_get_mean_perms.py:392`),
/// i.e. `exp(mean(log(x)))` in `f64`, the nulls' dtype.
///
/// The `f64` route stays on the platform libm: numpy's `DOUBLE_log`/`DOUBLE_exp`
/// only go vectorized through SVML under AVX-512 (absent here), otherwise they
/// call `npy_log`/`npy_exp` — the same glibc Rust's `f64::ln`/`exp` reach, so
/// this needs no port (unlike the `f32` kernels in [`crate::math`]).
pub fn gmean_pvals(ligand: &[f64], receptor: &[f64], truth: &[f32], n_perms: usize) -> Vec<f64> {
    exceed_fraction(ligand, receptor, truth, n_perms, |ligand, receptor| {
        ((ligand.ln() + receptor.ln()) / 2.0).exp()
    })
}

/// scipy's `gmean` of one observed ligand/receptor mean pair, in `f32` — the
/// `lr_gmeans` of `_gmean_score` (`method/sc/_geometric_mean.py:28`).
///
/// `gmean((ligand_means, receptor_means), axis=0)` runs on the two `f32`
/// columns, and scipy keeps that dtype (`xp_result_type(a, weights,
/// force_floating=True)`, `scipy/stats/_stats_py.py`), so the logs, the
/// two-element mean and the exponential all evaluate in `f32` through numpy's
/// own kernels — Rust's libm disagrees with them by up to 4 ulp, so the log
/// and exp come from [`crate::math`], the bit-exact ports (W3 D2 closed). The
/// `f64` route cast back at the end differs from the oracle on 248 of the 440
/// rows. A zero mean logs to `-inf` and exponentiates back to `0.0`, which is
/// what the oracle records for one.
pub fn gmean32(ligand: f32, receptor: f32) -> f32 {
    crate::math::expf((crate::math::logf(ligand) + crate::math::logf(receptor)) / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::Csr;

    /// Two cells in `A`, one in `B`, one gene: values `[1, 3]` and `[5]`.
    fn prep() -> Prep {
        Prep {
            var_names: vec!["g".into()],
            labels: vec!["A".into(), "B".into()],
            counts: vec![2, 1],
            cell_cluster: vec![0, 0, 1],
            x: Csr {
                n_rows: 3,
                n_cols: 1,
                indptr: vec![0, 1, 2, 3],
                indices: vec![0, 0, 0],
                data: vec![1.0, 3.0, 5.0],
            },
            means: vec![2.0, 5.0],
            props: vec![1.0, 1.0],
        }
    }

    /// The shuffle moves the `B` cell to position 0 and the `A` cells to
    /// positions 1 and 2 — which carry label `A` and `B` respectively, so the
    /// sums follow the *position's* label, not the row's.
    #[test]
    fn cube_gathers_rows_by_position_label() {
        let cube = means_cube(&prep(), &[2, 1, 0], 1);
        assert_eq!(cube, [4.0, 1.0], "A = (5 + 3) / 2, B = 1");
    }

    /// A zero mean logs to `-inf`, so the geometric mean of a pair with one
    /// zero side is exactly zero (no `NaN`, no mask needed).
    #[test]
    fn gmean32_zero_side_is_zero() {
        assert_eq!(gmean32(0.0, 4.0), 0.0);
        assert_eq!(gmean32(0.0, 0.0), 0.0);
        assert!((gmean32(2.0, 8.0) - 4.0).abs() < 1e-6, "gmean(2, 8) = 4");
    }

    #[test]
    fn pvals_count_exact_and_close_exceeds() {
        let truth = [1.0f32];
        //              p0     p1     p2                p3
        let means = [1.0, 0.5, 0.999_999_5, 0.999_998_8];
        let (ligand, receptor): (Vec<f64>, Vec<f64>) = means.iter().map(|&m| (m, m)).unzip();
        // truth 1.0: p0 ties exactly, p2 is 5e-7 below (within rtol), p3 is
        // 1.2e-6 below (outside it); p1 is far below.
        assert_eq!(pvals(&ligand, &receptor, &truth, 4), [0.5]);
    }
}
