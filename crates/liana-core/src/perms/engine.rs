//! The streaming permutation engine: the W3 [`crate::perms::null`] cube and
//! `_calculate_pvals` computed a block of permutations at a time.
//!
//! The engine consumes the same inputs as the materialised path — a prepared
//! [`Prep`], a seed and `n_perms` — and produces the same per-row p-values,
//! bit for bit, but never holds `(n_perms, n_obs)` or `(n_perms, n_rows)`:
//! each block's cube slab is consumed into per-row exceedance counters
//! before the next block is drawn, so peak memory is set by the block size
//! (the thread count) rather than by `n_perms`.

use rayon::prelude::*;

use crate::perms::null::TIE_RTOL;
use crate::perms::rng::PermsStream;
use crate::perms::trimean::{self, LabelIndex};
use crate::prep::Prep;

/// The null aggregation `_get_means_perms` runs: the per-label means every
/// permutation-scored method scores against, or the trimeans of `X / norm`
/// cellchat does (`_liana_pipe.py:661-665` sets `aggregation="trimean"` and
/// `norm_factor=mat_max` from the frame's `mat_max` column).
#[derive(Debug, Clone, Copy)]
pub enum Aggregation {
    Mean,
    Trimean { norm: f32 },
}

/// One output row's four cube coordinates per permutation: the ligand
/// statistic is `cube[p, source, ligand]`, the receptor one
/// `cube[p, target, receptor]` — `_run_method`'s
/// `perms[:, source_idx, ligand_idx]` selection (`_liana_pipe.py:682-686`).
#[derive(Debug, Clone, Copy)]
pub struct RowSel {
    pub source: u32,
    pub target: u32,
    pub ligand: u32,
    pub receptor: u32,
}

/// One block of `nb` permutations of per-cluster means, row-major
/// `(nb, n_labels, n_vars)` — [`crate::perms::null::means_cube`] restricted to
/// a permutation range.
///
/// Each permutation is an independent accumulation in the same order
/// `means_cube` runs (positions in order, each row's entries in stored-entry
/// order, `f64` sums divided by the label's cell count), so concatenating the
/// blocks reproduces the cube exactly — `consecutive_blocks_are_the_cube`
/// pins that against the materialised kernel.
pub fn block_sums(prep: &Prep, perms: &[u32], nb: usize) -> Vec<f64> {
    let (n_obs, n_labels, n_vars) = (prep.x.n_rows, prep.n_labels(), prep.n_vars());
    assert_eq!(perms.len(), nb * n_obs, "permutation block shape");
    let mut cube = vec![0f64; nb * n_labels * n_vars];

    cube.par_chunks_mut(n_labels * n_vars)
        .enumerate()
        .for_each(|(p, slab)| {
            let base = p * n_obs;
            for (position, &row) in perms[base..base + n_obs].iter().enumerate() {
                let label = prep.cell_cluster[position] as usize;
                let row = row as usize;
                let entries = prep.x.indptr[row]..prep.x.indptr[row + 1];
                for k in entries {
                    let gene = prep.x.indices[k] as usize;
                    slab[label * n_vars + gene] += f64::from(prep.x.data[k]);
                }
            }
            for label in 0..n_labels {
                let count = prep.counts[label] as f64;
                for value in &mut slab[label * n_vars..(label + 1) * n_vars] {
                    *value /= count;
                }
            }
        });
    cube
}

/// One block of `nb` permutations of per-cluster trimeans, row-major
/// `(nb, n_labels, n_vars)` — `_perm_group_trimeans` (`_get_mean_perms.py:194`)
/// restricted to a permutation range: per (permutation, label) the permuted
/// rows' stored values are bucketed by gene (counting sort) and reduced with
/// `_sparse_trimean` over the label's cell count. `scaled` is the `X / norm`
/// the trimeans run on, `labels` the position grouping.
pub(crate) fn block_trimeans(
    prep: &Prep,
    scaled: &[f32],
    labels: &LabelIndex,
    perms: &[u32],
    nb: usize,
) -> Vec<f64> {
    let (n_obs, n_labels, n_vars) = (prep.x.n_rows, prep.n_labels(), prep.n_vars());
    assert_eq!(perms.len(), nb * n_obs, "permutation block shape");
    let mut cube = vec![0f64; nb * n_labels * n_vars];

    cube.par_chunks_mut(n_vars)
        .enumerate()
        .for_each(|(task, slab)| {
            let (p, label) = (task / n_labels, task % n_labels);
            let rows = &perms[p * n_obs..(p + 1) * n_obs];
            trimean::label_trimeans(
                prep,
                scaled,
                labels.cells(label),
                prep.counts[label],
                |position| rows[position] as usize,
                |value| value,
                slab,
            );
        });
    cube
}

/// The per-row p-values of `_calculate_pvals` (`_get_mean_perms.py:367-397`),
/// streamed: the fraction of permutations whose combined ligand/receptor
/// statistic — `combine`, the method's score function — is at least the
/// observed `truth`, or within liana's tie tolerance of it.
///
/// `rows` selects each output row's cube coordinates and `truth` holds its
/// observed statistic (`f64`, as the caller widens it). `threads` is the
/// worker count (`0` = the process default); it is also the block size, so the
/// result does not depend on it: the permutation stream and every per-row
/// count are what they would be at any other count. The p-values are returned
/// in `rows`' order.
// Every argument is a distinct input (the prepared matrix, the row selection,
// the observed statistics, the run's seed/permutation/worker counts, the
// aggregation, the scorer); an argument struct would only move the same list
// to the call sites.
#[allow(clippy::too_many_arguments)]
pub fn pvals_streaming(
    prep: &Prep,
    rows: &[RowSel],
    truth: &[f64],
    seed: u64,
    n_perms: usize,
    threads: usize,
    aggregation: Aggregation,
    combine: impl Fn(f64, f64) -> f64 + Sync,
) -> Vec<f64> {
    assert_eq!(rows.len(), truth.len(), "row selections vs truth");
    assert!(n_perms > 0, "n_perms must be positive");
    // The block size is the worker count: the fusion keeps one block's slab in
    // flight, so `threads` bounds peak memory as well as the parallelism.
    let workers = if threads == 0 {
        rayon::current_num_threads()
    } else {
        threads
    };
    let (n_labels, n_vars) = (prep.n_labels(), prep.n_vars());
    let slab = n_labels * n_vars;

    // The trimean aggregation's inputs, built once per run: the `X / norm` the
    // null's rows gather from and the label position groups.
    let trimean_inputs = match aggregation {
        Aggregation::Mean => None,
        Aggregation::Trimean { norm } => Some((trimean::scaled(prep, norm), LabelIndex::new(prep))),
    };

    let counts = in_pool(threads, || {
        let mut counts = vec![0u32; rows.len()];
        let mut stream = PermsStream::new(seed, prep.x.n_rows);
        let mut done = 0;
        while done < n_perms {
            let nb = workers.min(n_perms - done);
            let perms = stream.next_block(nb);
            let cube = match &trimean_inputs {
                None => block_sums(prep, &perms, nb),
                Some((scaled, labels)) => block_trimeans(prep, scaled, labels, &perms, nb),
            };

            let chunk = counts.len().div_ceil(workers).max(1);
            counts
                .par_chunks_mut(chunk)
                .enumerate()
                .for_each(|(c, counters)| {
                    for (j, count) in counters.iter_mut().enumerate() {
                        let row = rows[c * chunk + j];
                        let observed = truth[c * chunk + j];
                        for p in 0..nb {
                            let base = p * slab;
                            let ligand =
                                cube[base + row.source as usize * n_vars + row.ligand as usize];
                            let receptor =
                                cube[base + row.target as usize * n_vars + row.receptor as usize];
                            let stat = combine(ligand, receptor);
                            if stat >= observed
                                || (stat - observed).abs() <= TIE_RTOL * observed.abs()
                            {
                                *count += 1;
                            }
                        }
                    }
                });
            done += nb;
        }
        counts
    });

    counts
        .iter()
        .map(|&count| count as f64 / n_perms as f64)
        .collect()
}

/// Run `f` on a pool of `threads` workers (`0` = the process default).
fn in_pool<T: Send>(threads: usize, f: impl FnOnce() -> T + Send) -> T {
    match threads {
        0 => f(),
        n => rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .expect("rayon pool")
            .install(f),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::Csr;
    use crate::perms::null::{means_cube, pvals};
    use crate::perms::rng::permutation_matrix;

    /// Two cells in `A`, one in `B`, two genes; `g1` stored by the `B` cell
    /// only, so the labels' means differ per permutation.
    fn prep() -> Prep {
        Prep {
            var_names: vec!["g0".into(), "g1".into()],
            labels: vec!["A".into(), "B".into()],
            counts: vec![2, 1],
            cell_cluster: vec![0, 0, 1],
            x: Csr {
                n_rows: 3,
                n_cols: 2,
                indptr: vec![0, 1, 3, 4],
                indices: vec![0, 0, 1, 0],
                data: vec![1.0, 3.0, 5.0, 7.0],
            },
            means: vec![2.0, 2.5, 7.0, 0.0],
            props: vec![1.0, 0.5, 1.0, 0.0],
        }
    }

    /// Consecutive blocks concatenate to the materialised cube, bit for bit —
    /// block boundaries do not move a single `f64` sum.
    #[test]
    fn consecutive_blocks_are_the_cube() {
        let prep = prep();
        let perms = permutation_matrix(1337, prep.x.n_rows, 5);
        let cube = means_cube(&prep, &perms, 5);
        let slab = prep.n_labels() * prep.n_vars();

        let mut streamed = Vec::new();
        for (start, nb) in [(0, 2), (2, 2), (4, 1)] {
            let block = block_sums(
                &prep,
                &perms[start * prep.x.n_rows..(start + nb) * prep.x.n_rows],
                nb,
            );
            streamed.extend_from_slice(&block);
        }
        assert_eq!(streamed.len(), cube.len());
        for (index, (s, c)) in streamed.iter().zip(&cube).enumerate() {
            assert_eq!(s.to_bits(), c.to_bits(), "cube value {index}");
        }

        // ...and the cube is the hand-checkable one: seed 1337's first
        // permutation is `[2, 0, 1]`, so position 0 of `A` takes cell 2
        // (`g0` = 7), position 1 takes cell 0 (`g0` = 1) and `B`'s single
        // position takes cell 1 (`g0` = 3, `g1` = 5)
        assert_eq!(perms[..3], [2, 0, 1], "first permutation");
        assert_eq!(
            &cube[0..slab],
            &[4.0, 0.0, 3.0, 5.0],
            "A = (8/2, 0), B = (3, 5)"
        );
    }

    /// The streamed p-values equal `_calculate_pvals` over the materialised
    /// cube, including across a block tail (`n_perms = 5` over `threads = 2`).
    #[test]
    fn streamed_pvalues_match_the_materialised_path() {
        let prep = prep();
        let n_perms = 5;
        let perms = permutation_matrix(1337, prep.x.n_rows, n_perms);
        let cube = means_cube(&prep, &perms, n_perms);
        let (n_labels, n_vars) = (prep.n_labels(), prep.n_vars());

        // rows: (A -> B, g0 -> g0), (B -> A, g1 -> g0), (A -> A, g0 -> g1)
        let rows = [
            RowSel {
                source: 0,
                target: 1,
                ligand: 0,
                receptor: 0,
            },
            RowSel {
                source: 1,
                target: 0,
                ligand: 1,
                receptor: 0,
            },
            RowSel {
                source: 0,
                target: 0,
                ligand: 0,
                receptor: 1,
            },
        ];
        let truth = [3.0f32, 1.0, 0.0];

        let ligand_nulls: Vec<f64> = rows
            .iter()
            .flat_map(|row| {
                (0..n_perms).map(|p| {
                    cube[(p * n_labels + row.source as usize) * n_vars + row.ligand as usize]
                })
            })
            .collect();
        let receptor_nulls: Vec<f64> = rows
            .iter()
            .flat_map(|row| {
                (0..n_perms).map(|p| {
                    cube[(p * n_labels + row.target as usize) * n_vars + row.receptor as usize]
                })
            })
            .collect();
        let expected = pvals(&ligand_nulls, &receptor_nulls, &truth, n_perms);

        let widened: Vec<f64> = truth.iter().map(|&t| f64::from(t)).collect();
        for threads in [0, 1, 2, 32] {
            assert_eq!(
                pvals_streaming(
                    &prep,
                    &rows,
                    &widened,
                    1337,
                    n_perms,
                    threads,
                    Aggregation::Mean,
                    |a, b| (a + b) / 2.0,
                ),
                expected,
                "threads = {threads}"
            );
        }
    }

    /// The trimean blocks against the oracle's own `_sparse_trimean` over the
    /// five permutations of seed 1337 (`[2,0,1] [0,2,1] [1,2,0] [1,0,2]
    /// [1,2,0]`), as `(perm, A(g0,g1), B(g0,g1))` slabs — e.g. in permutation
    /// 0, A's positions take cells {2, 0} so `g0` is the trimean of `{1, 7}`
    /// (`q25 2.5`, median `4`, `q75 5.5`) and `g1` is empty.
    #[test]
    fn trimean_blocks_match_the_oracle_kernel() {
        let prep = prep();
        let labels = LabelIndex::new(&prep);
        let scaled = trimean::scaled(&prep, 1.0);
        let perms = permutation_matrix(1337, prep.x.n_rows, 5);
        let cube = block_trimeans(&prep, &scaled, &labels, &perms, 5);

        let expected = [
            [4.0, 0.0, 3.0, 5.0],
            [4.0, 0.0, 3.0, 5.0],
            [5.0, 2.5, 1.0, 0.0],
            [2.0, 2.5, 7.0, 0.0],
            [5.0, 2.5, 1.0, 0.0],
        ];
        for (p, slab) in expected.iter().enumerate() {
            assert_eq!(
                &cube[p * 4..(p + 1) * 4],
                slab,
                "permutation {p} ({:?})",
                &perms[p * 3..(p + 1) * 3]
            );
        }

        // the (A -> B, g0 -> g0) row against `truth = 4`: the permuted
        // statistics `(4+3)/2, (4+3)/2, (5+1)/2, (2+7)/2, (5+1)/2` clear it
        // once (permutation 3), so the p-value is 1/5
        let rows = [RowSel {
            source: 0,
            target: 1,
            ligand: 0,
            receptor: 0,
        }];
        let streamed = pvals_streaming(
            &prep,
            &rows,
            &[4.0],
            1337,
            5,
            1,
            Aggregation::Trimean { norm: 1.0 },
            |a, b| (a + b) / 2.0,
        );
        assert_eq!(streamed, vec![0.2]);
    }
}
