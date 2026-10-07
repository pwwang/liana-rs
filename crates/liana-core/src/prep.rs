//! Cluster-level expression statistics — the stretch of liana's `_liana_pipe`
//! between the raw matrix and the per-interaction score columns:
//! `prep_check_adata` plus the per-label `means`/`props` `_get_lr` builds.
//!
//! Ground truth: `testdata/pipe_ref/synthetic__cellphonedb.json` (`prep`),
//! dumped from the pinned oracle by `scripts/dump_pipe_ref.py`.

use std::collections::HashMap;

use anyhow::Result;

use crate::io::{Adata, Csr};

/// The prepared matrix and the per-cluster statistics liana joins onto the
/// resource rows.
///
/// `x` and its per-cluster `means`/`props` are what `_get_lr` sees; `labels`
/// and `counts` are the clusters surviving `min_cells`.
#[derive(Debug, Clone, PartialEq)]
pub struct Prep {
    /// Feature names in the prepared (alphabetical) order — the columns of `x`
    /// and of the `means`/`props` rows.
    pub var_names: Vec<String>,
    /// Clusters surviving `min_cells`, in the input's category order.
    pub labels: Vec<String>,
    /// Cells per surviving cluster.
    pub counts: Vec<usize>,
    /// Cluster of each row of `x` (cells whose cluster was dropped are gone).
    pub cell_cluster: Vec<u32>,
    /// The expression matrix liana's statistics run over: rows are the cells
    /// of the surviving clusters in input order, columns the prepared vars,
    /// entries stored values (`f32`, as `prep_check_adata` casts).
    pub x: Csr,
    /// Per-cluster feature means, `labels.len() * var_names.len()` row-major.
    pub means: Vec<f32>,
    /// Per-cluster feature proportions, same shape as [`Prep::means`].
    pub props: Vec<f64>,
}

impl Prep {
    pub fn n_labels(&self) -> usize {
        self.labels.len()
    }

    pub fn n_vars(&self) -> usize {
        self.var_names.len()
    }

    /// `_get_lr`'s per-cluster mean of one feature.
    pub fn mean(&self, cluster: usize, gene: usize) -> f32 {
        self.means[cluster * self.n_vars() + gene]
    }

    /// `_get_lr`'s per-cluster proportion of one feature — the fraction of the
    /// cluster's cells with a *stored* entry for it (`_get_props`).
    pub fn prop(&self, cluster: usize, gene: usize) -> f64 {
        self.props[cluster * self.n_vars() + gene]
    }

    /// Position of a feature in the prepared matrix.
    pub fn gene_index(&self, name: &str) -> Option<usize> {
        self.var_names.iter().position(|var| var == name)
    }

    /// Position of a cluster in `labels`.
    pub fn cluster_index(&self, name: &str) -> Option<usize> {
        self.labels.iter().position(|label| label == name)
    }
}

/// liana's `prep_check_adata` + the `means`/`props` of `_get_lr`.
///
/// The order of the steps is `prep_check_adata`'s (`_pre.py:161-269`):
/// features whose `f32` column sum is zero are dropped *before* `min_cells`
/// takes cells away, and the surviving features are then re-ordered
/// alphabetically. `min_cells` counts cells per cluster; a cluster below it
/// loses every cell and drops out of the pair enumeration (0 keeps all).
pub fn prepare(adata: &Adata, min_cells: usize) -> Result<Prep> {
    let n_genes = adata.x.n_cols;

    // `prep_check_adata` (`_pre.py:176`): `X.sum(axis=0) == 0` drops a
    // feature. scipy sums a `csr_matrix` along axis 0 as `ones(1, n_obs) @ X`
    // (`scipy/sparse/_base.py:1516`), whose kernel adds one column's entries in
    // ascending row order, in the matrix's `f32` dtype.
    let mut sums = vec![0f32; n_genes];
    for row in 0..adata.x.n_rows {
        let (columns, values) = row_entries(&adata.x, row);
        for (gene, &value) in columns.iter().zip(values) {
            sums[*gene as usize] += value;
        }
    }

    // `prep_check_adata` (`_pre.py:268`): `adata[:, np.sort(adata.var_names)]`.
    let mut columns: Vec<usize> = (0..n_genes).filter(|&gene| sums[gene] != 0.0).collect();
    columns.sort_by(|&left, &right| adata.var_names[left].cmp(&adata.var_names[right]));
    let mut prepared_of = vec![usize::MAX; n_genes];
    for (prepared, &column) in columns.iter().enumerate() {
        prepared_of[column] = prepared;
    }
    let var_names: Vec<String> = columns
        .iter()
        .map(|&c| adata.var_names[c].clone())
        .collect();

    // `prep_check_adata` (`_pre.py:252-259`): clusters below `min_cells` lose
    // their cells, and only the surviving categories remain (`anndata._core`
    // re-derives them on subset, in category order).
    let mut cells_per_cluster = vec![0usize; adata.label_names.len()];
    for &label in &adata.labels {
        cells_per_cluster[label as usize] += 1;
    }
    let surviving: Vec<u32> = (0..cells_per_cluster.len() as u32)
        .filter(|&label| cells_per_cluster[label as usize] >= min_cells)
        .collect();
    let labels: Vec<String> = surviving
        .iter()
        .map(|&label| adata.label_names[label as usize].clone())
        .collect();
    let counts: Vec<usize> = surviving
        .iter()
        .map(|&label| cells_per_cluster[label as usize])
        .collect();
    let mut cluster_of = vec![u32::MAX; cells_per_cluster.len()];
    for (cluster, &label) in surviving.iter().enumerate() {
        cluster_of[label as usize] = cluster as u32;
    }

    // The surviving rows, with each entry re-indexed into the prepared column
    // order. scipy's column indexing (`X[:, cols]`, what anndata's subset runs)
    // yields CSR rows with ascending indices, so the entries are sorted here
    // too — per-column statistics do not care, but the layout stays canonical.
    let mut x = Csr {
        n_rows: 0,
        n_cols: var_names.len(),
        indptr: vec![0],
        indices: Vec::new(),
        data: Vec::new(),
    };
    let mut cell_cluster: Vec<u32> = Vec::new();
    for row in 0..adata.x.n_rows {
        let cluster = cluster_of[adata.labels[row] as usize];
        if cluster == u32::MAX {
            continue;
        }
        let (columns, values) = row_entries(&adata.x, row);
        let mut entries: Vec<(usize, f32)> = columns
            .iter()
            .zip(values)
            .filter(|(gene, _)| prepared_of[**gene as usize] != usize::MAX)
            .map(|(&gene, &value)| (prepared_of[gene as usize], value))
            .collect();
        entries.sort_unstable_by_key(|&(column, _)| column);
        for (column, value) in entries {
            x.indices.push(column as u32);
            x.data.push(value);
        }
        x.indptr.push(x.data.len());
        x.n_rows += 1;
        cell_cluster.push(cluster);
    }

    // `_get_lr` (`_liana_pipe.py:522`): `props` are
    // `getnnz(axis=0) / n_cells` (`_common.py:56`); `_liana_pipe.py:536`:
    // `means` are scipy's `mean(axis=0)` =
    // `(X * (1.0 / n_cells)).sum(axis=0, dtype=float32)`
    // (`scipy/sparse/_base.py:1574`) — each stored value is scaled to `f32`
    // *before* the `f32` accumulation (numpy casts the Python-float scalar to
    // the array's dtype), so the mean is not `sum / n_cells`.
    let n_vars = var_names.len();
    let mut means = vec![0f32; labels.len() * n_vars];
    let mut props = vec![0f64; labels.len() * n_vars];
    for cluster in 0..labels.len() {
        let scale = (1.0f64 / counts[cluster] as f64) as f32;
        let mut nnz = vec![0usize; n_vars];
        for (row, &of_cluster) in cell_cluster.iter().enumerate() {
            if of_cluster as usize != cluster {
                continue;
            }
            let (columns, values) = row_entries(&x, row);
            for (gene, &value) in columns.iter().zip(values) {
                means[cluster * n_vars + *gene as usize] += value * scale;
                nnz[*gene as usize] += 1;
            }
        }
        for gene in 0..n_vars {
            props[cluster * n_vars + gene] = nnz[gene] as f64 / counts[cluster] as f64;
        }
    }

    Ok(Prep {
        var_names,
        labels,
        counts,
        cell_cluster,
        x,
        means,
        props,
    })
}

fn row_entries(x: &Csr, row: usize) -> (&[u32], &[f32]) {
    let range = x.indptr[row]..x.indptr[row + 1];
    (&x.indices[range.clone()], &x.data[range])
}

/// Name → position lookups for the prepared frame, used by the pipe's joins.
pub fn index_by_name(names: &[String]) -> HashMap<&str, usize> {
    names
        .iter()
        .enumerate()
        .map(|(index, name)| (name.as_str(), index))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::Csr;

    /// Three cells over genes `rec`, `keep`, `dead` (never stored) and `empty`
    /// (never stored): `B` has one cell (below `min_cells=2`) and stores a zero
    /// for `keep`. Columns are deliberately unsorted to pin the re-ordering.
    fn adata() -> Adata {
        Adata {
            x: Csr {
                n_rows: 3,
                n_cols: 4,
                indptr: vec![0, 2, 3, 5],
                indices: vec![1, 0, 1, 1, 0],
                data: vec![1.0, 2.0, 1.0, 0.0, 3.0],
            },
            obs_names: vec!["c0".into(), "c1".into(), "c2".into()],
            // `rec` < `keep` < `dead` < `empty` alphabetically is not the
            // column order, so a port that forgets to sort cannot pass
            var_names: vec!["rec".into(), "keep".into(), "dead".into(), "empty".into()],
            labels: vec![0, 0, 1],
            label_names: vec!["A".into(), "B".into()],
            obsm_spatial: None,
        }
    }

    #[test]
    fn drops_empty_features_sorts_vars_and_min_cells_clusters() {
        let prep = prepare(&adata(), 2).unwrap();
        assert_eq!(prep.var_names, ["keep", "rec"], "both dropped empty");
        // `B` has a single cell, so it is dropped whole
        assert_eq!(prep.labels, ["A"]);
        assert_eq!(prep.counts, [2]);
        assert_eq!(prep.cell_cluster, [0, 0]);
        assert_eq!(prep.x.n_rows, 2);
        assert_eq!(prep.x.indptr, [0, 2, 3]);
        assert_eq!(prep.x.indices, [0, 1, 0], "rows re-indexed and sorted");
        assert_eq!(prep.x.data, [1.0, 2.0, 1.0]);
        // `A`'s means and props, hand-computed in f32/f64
        assert_eq!(prep.mean(0, 0), 1.0);
        assert_eq!(prep.mean(0, 1), 1.0);
        assert_eq!(prep.prop(0, 0), 1.0);
        assert_eq!(prep.prop(0, 1), 0.5);
    }

    /// The `f32` arithmetic the oracle pins: `means` scale each value *before*
    /// summing, which differs from `sum / n_cells` at the last bit.
    /// `scipy/sparse/_base.py:1574`, `_liana_pipe.py:536`.
    #[test]
    fn mean_scales_before_summing() {
        let adata = Adata {
            x: Csr {
                n_rows: 3,
                n_cols: 1,
                indptr: vec![0, 1, 2, 3],
                indices: vec![0, 0, 0],
                data: vec![1.0, 1.0, 3.0],
            },
            obs_names: vec!["c0".into(), "c1".into(), "c2".into()],
            var_names: vec!["g".into()],
            labels: vec![0, 0, 0],
            label_names: vec!["A".into()],
            obsm_spatial: None,
        };
        let prep = prepare(&adata, 0).unwrap();
        let scaled = {
            let scale = (1.0f64 / 3.0) as f32;
            (1.0f32 * scale) + (1.0f32 * scale) + (3.0f32 * scale)
        };
        assert_eq!(prep.mean(0, 0), scaled);
        assert_ne!(prep.mean(0, 0), (5.0f32 / 3.0), "sum-then-divide is wrong");
        assert_eq!(prep.prop(0, 0), 1.0);
    }

    #[test]
    fn props_count_stored_entries_not_nonzeros() {
        let prep = prepare(&adata(), 0).unwrap();
        assert_eq!(prep.labels, ["A", "B"]);
        assert_eq!(prep.counts, [2, 1]);
        // `B` stores a zero for `keep`: it is counted as expressed even though
        // its mean is zero — that is what `getnnz` measures
        let b = prep.cluster_index("B").unwrap();
        let keep = prep.gene_index("keep").unwrap();
        assert_eq!(prep.mean(b, keep), 0.0);
        assert_eq!(prep.prop(b, keep), 1.0);
        // ...and a gene stored in one of `A`'s two cells is a proportion of 0.5
        let a = prep.cluster_index("A").unwrap();
        let rec = prep.gene_index("rec").unwrap();
        assert_eq!(prep.prop(a, rec), 0.5);
        assert_eq!(prep.prop(b, rec), 1.0);
    }
}
