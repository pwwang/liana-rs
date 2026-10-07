//! Permutation nulls — `_get_means_perms` (the `aggregation="mean"` path of
//! `liana/_core/_pipe_utils/_get_mean_perms.py`) and `_calculate_pvals`.
//!
//! Ground truth: `testdata/pipe_ref/synthetic__cellphonedb.json` (`n_perms`),
//! dumped from the pinned oracle by `scripts/dump_pipe_ref.py`.

use crate::prep::Prep;

/// `_TIE_RTOL` (`_get_mean_perms.py:38`): how close a permuted mean has to be
/// to the observed one to count as tied with it.
const TIE_RTOL: f64 = 1e-6;

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
pub fn means_cube(prep: &Prep, perms: &[u16], n_perms: usize) -> Vec<f64> {
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
/// whose combined ligand/receptor mean is at least the observed `truth`.
///
/// `ligand` and `receptor` are the two stacked statistics — what the caller
/// selected as `perms[:, source, ligand]` and `perms[:, target, receptor]` —
/// laid out `(n_rows, n_perms)` row-major. Their mean is `(ligand + receptor)
/// / 2` in `f64`, matching `np.mean(stack, axis=0)`. A permutation counts when
/// its mean is `>= truth` or within `rtol = 1e-6` of it — `np.isclose(...,
/// atol=0.0)`, the tie tolerance of the `f32` scores — and the count is
/// divided by `n_perms` in `f64`.
pub fn pvals(ligand: &[f64], receptor: &[f64], truth: &[f32], n_perms: usize) -> Vec<f64> {
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
                    let mean = (ligand + receptor) / 2.0;
                    mean >= observed || (mean - observed).abs() <= TIE_RTOL * observed.abs()
                })
                .count();
            exceeds as f64 / n_perms as f64
        })
        .collect()
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
