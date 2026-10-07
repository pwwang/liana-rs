//! `expr_prop`/`min_cells` filtering — the statistical core of liana's
//! `_liana_pipe` (`prep_check_adata`, `filter_resource`, `_get_lr`,
//! `_filter_reassemble_complexes`). See `docs/resource-semantics.md`.

use std::collections::{HashMap, HashSet};

use anyhow::{Result, bail};

use crate::io::Adata;

use super::{LrPair, LrSubunit, explode_complexes};

/// One ligand–receptor complex pair that passed `expr_prop` in one cluster pair.
#[derive(Debug, Clone, PartialEq)]
pub struct KeptPair {
    /// Cluster codes, indexing [`FilterResult::labels`].
    pub source: u32,
    pub target: u32,
    pub ligand_complex: String,
    pub receptor_complex: String,
    /// The minimum subunit proportion of the pair — over every exploded subunit,
    /// on both the source and the target side. `>= expr_prop` is what kept it.
    pub prop_min: f64,
}

/// What [`filter_lrs`] kept.
#[derive(Debug, Clone, PartialEq)]
pub struct FilterResult {
    /// The labels surviving `min_cells`, in `Adata::label_names` order.
    pub labels: Vec<String>,
    /// One entry per kept complex pair and cluster pair, in liana's row order.
    pub kept: Vec<KeptPair>,
}

/// Drop the resource rows whose subunits the matrix cannot provide, mirroring
/// liana's `filter_resource`: a row needs both its subunit genes among
/// `var_names`, and a complex pair is dropped as a whole when any subunit of
/// either of its complexes is missing.
pub fn filter_resource(subunits: &[LrSubunit], var_names: &[String]) -> Vec<LrSubunit> {
    let vars: HashSet<&str> = var_names.iter().map(String::as_str).collect();
    subunits
        .iter()
        .filter(|subunit| {
            vars.contains(subunit.ligand.as_str())
                && vars.contains(subunit.receptor.as_str())
                // `all_units = ligand_complex + "_" + receptor_complex`, split on `_`:
                // the subunits of both complexes, and liana skips the check for a
                // plain pair, where the two symbols are the only parts.
                && format!("{}_{}", subunit.ligand_complex, subunit.receptor_complex)
                    .split('_')
                    .all(|unit| vars.contains(unit))
        })
        .cloned()
        .collect()
}

/// liana's `expr_prop`/`min_cells` filter, from the expression matrix to the
/// ligand–receptor pairs that survive it.
///
/// `min_cells` is the minimum number of cells per cluster: cells of a smaller
/// cluster are dropped, and the cluster drops out of the pair enumeration
/// (liana's `min_cells=0` keeps everything).
///
/// A pair survives when the *minimum* subunit proportion over its exploded
/// subunits — ligand side in the source cluster, receptor side in the target
/// cluster — is `>= expr_prop`. The proportion of a gene in a cluster is the
/// fraction of that cluster's cells with a stored entry for it
/// (`X.getnnz(axis=0) / n_cells`), so an explicitly stored zero counts as
/// expressed, as it does in the CSR liana builds.
///
/// Errors when there are pairs but none pass, which liana raises rather than
/// returning an empty frame.
pub fn filter_lrs(
    adata: &Adata,
    resource: &[LrPair],
    expr_prop: f64,
    min_cells: usize,
) -> Result<FilterResult> {
    let n_genes = adata.var_names.len();
    let var_index: HashMap<&str, usize> = {
        let mut index = HashMap::with_capacity(n_genes);
        for (i, name) in adata.var_names.iter().enumerate() {
            index.entry(name.as_str()).or_insert(i);
        }
        index
    };

    // `prep_check_adata`: clusters with fewer than `min_cells` cells are dropped
    // whole, and the surviving labels become the categories `_get_lr` pairs up.
    let mut n_cells = vec![0usize; adata.label_names.len()];
    for &label in &adata.labels {
        n_cells[label as usize] += 1;
    }
    let surviving: Vec<u32> = (0..n_cells.len() as u32)
        .filter(|&label| n_cells[label as usize] >= min_cells)
        .collect();
    let mut cluster_of = vec![u32::MAX; n_cells.len()];
    for (cluster, &label) in surviving.iter().enumerate() {
        cluster_of[label as usize] = cluster as u32;
    }
    let labels: Vec<String> = surviving
        .iter()
        .map(|&label| adata.label_names[label as usize].clone())
        .collect();

    // `prep_check_adata` removes features whose entries sum to zero, so the
    // resource lookup below cannot see them.
    let mut sums = vec![0f32; n_genes];
    for cell in 0..adata.x.n_rows {
        let (columns, values) = (
            &adata.x.indices[adata.x.indptr[cell]..adata.x.indptr[cell + 1]],
            &adata.x.data[adata.x.indptr[cell]..adata.x.indptr[cell + 1]],
        );
        for (gene, &value) in columns.iter().zip(values) {
            sums[*gene as usize] += value;
        }
    }
    let present: Vec<String> = adata
        .var_names
        .iter()
        .zip(&sums)
        .filter(|&(_, &sum)| sum != 0.0)
        .map(|(name, _)| name.clone())
        .collect();

    // `_get_props`, per cluster: stored entries per gene over the cluster's cells.
    let mut nnz = vec![0usize; surviving.len() * n_genes];
    for (cell, &label) in adata.labels.iter().enumerate() {
        let cluster = cluster_of[label as usize];
        if cluster == u32::MAX {
            continue;
        }
        for &gene in &adata.x.indices[adata.x.indptr[cell]..adata.x.indptr[cell + 1]] {
            nnz[cluster as usize * n_genes + gene as usize] += 1;
        }
    }
    let prop = |cluster: usize, gene: &str| -> f64 {
        let column = var_index[gene];
        nnz[cluster * n_genes + column] as f64 / n_cells[surviving[cluster] as usize] as f64
    };

    let subunits = filter_resource(&explode_complexes(resource), &present);

    // `_get_lr`'s cluster pairs: `np.meshgrid(labels, labels)` — the source
    // varies fastest, the target slowest — crossed with the resource's rows.
    let mut kept: Vec<KeptPair> = Vec::new();
    let mut seen: HashMap<(u32, u32, &str, &str), usize> = HashMap::new();
    for target in 0..surviving.len() {
        for source in 0..surviving.len() {
            for subunit in &subunits {
                let key = (
                    source as u32,
                    target as u32,
                    subunit.ligand_complex.as_str(),
                    subunit.receptor_complex.as_str(),
                );
                let prop_min = prop(source, &subunit.ligand).min(prop(target, &subunit.receptor));
                match seen.get(&key) {
                    Some(&row) => kept[row].prop_min = kept[row].prop_min.min(prop_min),
                    None => {
                        seen.insert(key, kept.len());
                        kept.push(KeptPair {
                            source: source as u32,
                            target: target as u32,
                            ligand_complex: subunit.ligand_complex.clone(),
                            receptor_complex: subunit.receptor_complex.clone(),
                            prop_min,
                        });
                    }
                }
            }
        }
    }

    let pass = |pair: &KeptPair| pair.prop_min >= expr_prop;
    if !kept.is_empty() && !kept.iter().any(pass) {
        let highest = kept
            .iter()
            .map(|pair| pair.prop_min)
            .fold(f64::NEG_INFINITY, f64::max);
        bail!(
            "no ligand-receptor pair passed expr_prop={expr_prop}: the highest minimum subunit \
             proportion is {highest}"
        );
    }
    kept.retain(pass);
    Ok(FilterResult { labels, kept })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::Csr;

    /// Two clusters — `A` with two cells, `B` with one — over genes `lig`, `rec`
    /// (expressed in `A`) and `dead` (a stored zero in every cell).
    fn adata() -> Adata {
        Adata {
            x: Csr {
                n_rows: 3,
                n_cols: 3,
                indptr: vec![0, 2, 3, 4],
                indices: vec![0, 1, 0, 2],
                data: vec![1.0, 2.0, 3.0, 0.0],
            },
            obs_names: vec!["c0".into(), "c1".into(), "c2".into()],
            var_names: vec!["lig".into(), "rec".into(), "dead".into()],
            labels: vec![0, 0, 1],
            label_names: vec!["A".into(), "B".into()],
            obsm_spatial: None,
        }
    }

    fn pair(ligand: &str, receptor: &str) -> LrPair {
        LrPair {
            ligand: ligand.to_string(),
            receptor: receptor.to_string(),
        }
    }

    /// Every cluster pair, as liana's `meshgrid` orders them: source fastest.
    fn cluster_pairs(result: &FilterResult) -> Vec<(u32, u32)> {
        result
            .kept
            .iter()
            .map(|pair| (pair.source, pair.target))
            .collect()
    }

    #[test]
    fn a_feature_that_sums_to_zero_is_absent() {
        let adata = adata();
        // `dead` is stored in every cell but holds only zeros: liana removes the
        // feature, so the pair that needs it never reaches the filter.
        let pairs = vec![pair("lig", "rec"), pair("dead", "rec")];
        let result = filter_lrs(&adata, &pairs, 0.5, 0).unwrap();
        assert_eq!(cluster_pairs(&result), [(0, 0)], "`lig&rec` in `A` only");
        assert_eq!(
            result.kept[0].prop_min, 0.5,
            "min(2/2 ligands, 1/2 receptors)"
        );
    }

    #[test]
    fn min_cells_drops_whole_clusters_from_the_pairs() {
        let adata = adata();
        let pairs = vec![pair("lig", "rec")];
        // `B` has a single cell, `A` two: at a threshold of two it is gone, and
        // with it every pair that would have had `B` on either side.
        let result = filter_lrs(&adata, &pairs, 0.5, 2).unwrap();
        assert_eq!(result.labels, ["A"]);
        assert_eq!(cluster_pairs(&result), [(0, 0)]);

        // ...and, as liana raises there, a threshold nothing can pass is an error
        // rather than an empty result.
        assert!(filter_lrs(&adata, &pairs, 1.5, 0).is_err());
    }
}
