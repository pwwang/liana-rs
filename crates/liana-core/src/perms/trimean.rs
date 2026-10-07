//! Tukey's trimean as `_get_mean_perms.py` computes it, on both sides of the
//! permutation null.
//!
//! The observed row is dense `f64` (`_trimean`, `:48-52`, over the f64
//! quotient `X / mat_max`), the null's rows stay `f32` (`_sparse_trimean`,
//! `:159-175`, gathered straight out of the `X.dtype` storage). Both are the
//! same order-statistics skeleton — numpy's `linear` quantiles over the row
//! the zeros splice into (`_at`, `:56-66`) and an arithmetic middle — so one
//! generic runs both, the element type owning the single rounding they differ
//! by.

use crate::prep::Prep;

/// The two element types the skeleton runs over.
pub(crate) trait Elem: Copy + Default {
    const ZERO: Self;
    fn widen(self) -> f64;
    /// The even-count middle: `mean` of the two central values in the element's
    /// own precision — a `f32` add and a `f32` halving for the null
    /// (`np.float32((lower + upper) / np.float32(2.0))`, `:168`), plain `f64`
    /// for the observed row.
    fn middle(self, other: Self) -> f64;
}

impl Elem for f32 {
    const ZERO: Self = 0.0;

    fn widen(self) -> f64 {
        f64::from(self)
    }

    fn middle(self, other: Self) -> f64 {
        f64::from((self + other) / 2.0)
    }
}

impl Elem for f64 {
    const ZERO: Self = 0.0;

    fn widen(self) -> f64 {
        self
    }

    fn middle(self, other: Self) -> f64 {
        (self + other) / 2.0
    }
}

/// `_lerp` (`:73-78`): interpolate from whichever end is nearer.
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    let diff = b - a;
    if t >= 0.5 {
        b - diff * (1.0 - t)
    } else {
        a + diff * t
    }
}

/// `_sparse_quantile` (`:139-154`): the linear-interpolation quantile of the
/// spliced row, both endpoints in `f64`.
fn quantile<T: Elem>(at: impl Fn(usize) -> T, n_total: usize, q: f64) -> f64 {
    let pos = q * (n_total as f64 - 1.0);
    let lo = pos.floor();
    let hi = (lo + 1.0).min(n_total as f64 - 1.0);
    lerp(at(lo as usize).widen(), at(hi as usize).widen(), pos - lo)
}

/// `_sparse_trimean` (`:159-175`) / `_trimean` (`:48-52`): the trimean of a row
/// whose implicit zeros are not stored — `values` holds the sorted stored
/// entries, `n_zeros` the count of zeros the storage elided, and `at` is the
/// row's `i`-th order statistic under `_at`'s splice (zeros sit after the
/// negatives).
fn trimean<T: Elem>(values: &[T], n_zeros: usize, n_total: usize) -> f64 {
    assert!(n_total > 0, "trimean of an empty row");
    assert_eq!(values.len() + n_zeros, n_total, "spliced row shape");
    let n_below = values.partition_point(|value| value.widen() < 0.0);
    let at = |i: usize| -> T {
        if i < n_below {
            values[i]
        } else if i < n_below + n_zeros {
            T::ZERO
        } else {
            values[i - n_zeros]
        }
    };
    let half = n_total / 2;
    let median = if n_total % 2 == 1 {
        at(half).widen()
    } else {
        T::middle(at(half - 1), at(half))
    };
    (quantile(at, n_total, 0.25) + 2.0 * median + quantile(at, n_total, 0.75)) / 4.0
}

/// The cells of each label, in ascending position order — the position
/// grouping both trimean paths bucket by (liana's per-label row masks).
pub(crate) struct LabelIndex {
    ptr: Vec<usize>,
    cells: Vec<u32>,
}

impl LabelIndex {
    pub(crate) fn new(prep: &Prep) -> Self {
        let mut ptr = vec![0usize; prep.n_labels() + 1];
        for &label in &prep.cell_cluster {
            ptr[label as usize + 1] += 1;
        }
        for label in 0..prep.n_labels() {
            ptr[label + 1] += ptr[label];
        }
        let mut cursor = ptr[..prep.n_labels()].to_vec();
        let mut cells = vec![0u32; prep.x.n_rows];
        for (position, &label) in prep.cell_cluster.iter().enumerate() {
            cells[cursor[label as usize]] = position as u32;
            cursor[label as usize] += 1;
        }
        Self { ptr, cells }
    }

    /// The positions carrying `label`.
    pub(crate) fn cells(&self, label: usize) -> &[u32] {
        &self.cells[self.ptr[label]..self.ptr[label + 1]]
    }
}

/// One label's trimeans into `out` (one slot per gene): bucket the cells'
/// stored values per gene — the counting sort `_perm_group_trimeans`
/// (`:194-215`) runs — sort each bucket, and reduce it with the trimean over
/// the label's cell count.
///
/// `data` is the matrix the aggregation reads (`X` or its scaled copy) and
/// `row_of` maps a position to the row sitting there (`X / norm`; the
/// identity for the observed side, the permutation for the null).
pub(crate) fn label_trimeans<T: Elem>(
    prep: &Prep,
    data: &[f32],
    cells: &[u32],
    n_total: usize,
    row_of: impl Fn(usize) -> usize,
    value: impl Fn(f32) -> T,
    out: &mut [f64],
) {
    let n_vars = prep.n_vars();
    let mut offsets = vec![0usize; n_vars + 1];
    for &position in cells {
        let row = row_of(position as usize);
        for &gene in &prep.x.indices[prep.x.indptr[row]..prep.x.indptr[row + 1]] {
            offsets[gene as usize + 1] += 1;
        }
    }
    for gene in 0..n_vars {
        offsets[gene + 1] += offsets[gene];
    }
    let mut values = vec![T::ZERO; offsets[n_vars]];
    let mut cursor = offsets[..n_vars].to_vec();
    for &position in cells {
        let row = row_of(position as usize);
        let entries = prep.x.indptr[row]..prep.x.indptr[row + 1];
        for (&gene, &stored) in prep.x.indices[entries.clone()].iter().zip(&data[entries]) {
            let gene = gene as usize;
            values[cursor[gene]] = value(stored);
            cursor[gene] += 1;
        }
    }

    for (gene, slot) in out.iter_mut().enumerate() {
        let column = &mut values[offsets[gene]..offsets[gene + 1]];
        column.sort_unstable_by(|a, b| a.widen().total_cmp(&b.widen()));
        *slot = trimean(column, n_total - column.len(), n_total);
    }
}

/// `norm`'s reciprocal as scipy forms it: `np.float32(1) / norm` in `f32`
/// (NEP 50 keeps the weak scalar `f32`), widened exactly.
fn recip(norm: f32) -> f64 {
    f64::from(1.0f32 / norm)
}

/// `(X / norm).astype(X.dtype)` over the whole matrix, as the null's rows see
/// it: scipy casts the CSR to `f64` and multiplies by the reciprocal, so the
/// product is exact in `f64` and `.astype` rounds it back once.
pub(crate) fn scaled(prep: &Prep, norm: f32) -> Vec<f32> {
    let recip = recip(norm);
    prep.x
        .data
        .iter()
        .map(|&x| (f64::from(x) * recip) as f32)
        .collect()
}

/// The observed side of the cellchat null: the per-label trimeans of the f64
/// quotient `X / norm`, `(n_labels, n_vars)` row-major — `_get_lr`'s
/// `_trimean(_choose_mtx_rep(temp) / mat_max)` (`_liana_pipe.py:542`). The
/// quotient stays `f64` here, so no second rounding.
pub(crate) fn observed(prep: &Prep, norm: f32) -> Vec<f64> {
    let n_vars = prep.n_vars();
    let recip = recip(norm);
    let labels = LabelIndex::new(prep);
    let mut out = vec![0f64; prep.n_labels() * n_vars];
    for (label, row) in out.chunks_mut(n_vars).enumerate() {
        label_trimeans(
            prep,
            &prep.x.data,
            labels.cells(label),
            prep.counts[label],
            |position| position,
            |x| f64::from(x) * recip,
            row,
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::Csr;

    /// One label of four cells over one gene, so every order statistic is
    /// hand-checkable; the second label's cell stores nothing.
    fn prep() -> Prep {
        Prep {
            var_names: vec!["g".into()],
            labels: vec!["A".into(), "B".into()],
            counts: vec![4, 1],
            cell_cluster: vec![0, 0, 0, 0, 1],
            x: Csr {
                n_rows: 5,
                n_cols: 1,
                indptr: vec![0, 1, 2, 3, 4, 4],
                indices: vec![0, 0, 0, 0],
                data: vec![0.0, 1.0, 2.0, 4.0],
            },
            means: vec![1.75, 0.0],
            props: vec![1.0, 0.0],
        }
    }

    /// The observed and null sides agree on a row that stores its zeros, and
    /// both splice an elided zero the same way: `_at` is the row's order
    /// statistics either way.
    #[test]
    fn sparse_and_dense_agree() {
        let fixture = prep();
        // observed: norm 1, so the dense f64 row is [0, 1, 2, 4]
        let out = observed(&fixture, 1.0);
        assert_eq!(out, vec![1.5625, 0.0], "dense [0,1,2,4] and empty");

        // null: the same row as `f32`, gathered with the one stored zero
        let scaled = scaled(&fixture, 1.0);
        assert_eq!(scaled, vec![0.0, 1.0, 2.0, 4.0]);
        let labels = LabelIndex::new(&fixture);
        let mut row = [0f64];
        // label A: all four cells, in position order
        label_trimeans(
            &fixture,
            &scaled,
            labels.cells(0),
            fixture.counts[0],
            |position| position,
            |x| x,
            &mut row,
        );
        assert_eq!(row, [1.5625], "sparse [0,1,2,4]");
        // ...and an elided zero splices to the same statistic: [1, 2, 4]
        let mut row = [0f64];
        label_trimeans(
            &fixture,
            &scaled,
            &[1, 2, 3],
            4,
            |position| position,
            |x| x,
            &mut row,
        );
        assert_eq!(row, [1.5625], "sparse [1,2,4] + 1 implicit zero");

        // a negative first: [-2, 3, 0, 0] and its sparse form [-2, 3]
        let mut neg = prep();
        neg.x.data = vec![-2.0, 3.0, 0.0, 0.0];
        let out = observed(&neg, 1.0);
        assert_eq!(out, vec![0.0625, 0.0]);
        let mut row = [0f64];
        label_trimeans(
            &neg,
            &neg.x.data,
            &[0, 1],
            4,
            |position| position,
            |x| x,
            &mut row,
        );
        assert_eq!(row, [0.0625], "sparse [-2,3] + 2 implicit zeros");
    }

    /// The scaling multiplies by `np.float32(1) / norm` — the `f32`
    /// reciprocal, not the f64 one — and rounds once back to `f32`. Pinned
    /// against numpy's own bits for `norm = 7`, where the two reciprocals
    /// differ in the last place (`0x3e124925` vs the f64-rounded
    /// `0x3e124924`).
    #[test]
    fn scaling_uses_the_f32_reciprocal() {
        let fixture = prep();
        let scaled = scaled(&fixture, 7.0);
        let expected = [0x00000000u32, 0x3e124925, 0x3e924925, 0x3f124925];
        let bits: Vec<u32> = scaled.iter().map(|value| value.to_bits()).collect();
        assert_eq!(bits, expected);
    }
}
