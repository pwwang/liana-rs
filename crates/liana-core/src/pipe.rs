//! The permutation-scored single-cell pipelines end to end — `_prepare_lr_stats`,
//! `_run_method` and `_sort_by_score` for `liana.method.sc._cellphonedb` and
//! `liana.method.sc._geometric_mean`.
//!
//! Reproduces `testdata/expected/synthetic__<method>__p{100,1000}.csv` from
//! `testdata/fixtures/synthetic.h5ad`, the toy resource, `seed=1337` and the
//! `expr_prop`/`min_cells` the oracle ran with.

use std::collections::{HashMap, HashSet};

use anyhow::{Result, bail};

use crate::io::Adata;
use crate::perms::engine::{self, RowSel};
use crate::perms::null::gmean32;
use crate::prep::{self, Prep};
use crate::resource::{self, LrPair, LrSubunit};

/// The oracle CSV header of [`run_cellphonedb`]'s rows.
pub const CPDB_CSV_HEADER: &str = "ligand,ligand_complex,ligand_means,ligand_props,\
                                   receptor,receptor_complex,receptor_means,receptor_props,\
                                   source,target,lr_means,cellphone_pvals";

/// The oracle CSV header of [`run_geometric_mean`]'s rows.
pub const GMEAN_CSV_HEADER: &str = "ligand,ligand_complex,ligand_means,ligand_props,\
                                    receptor,receptor_complex,receptor_means,receptor_props,\
                                    source,target,lr_gmeans,gmean_pvals";

/// One row of a method's output, in the oracle CSV's column order.
///
/// The first ten columns are the shared `lr_res` frame; `magnitude` and
/// `specificity` are the method's score and p-value — liana's own
/// `magnitude`/`specificity` terms (`method/sc/_Method.py`): `lr_means` and
/// `cellphone_pvals` for cellphonedb, `lr_gmeans` and `gmean_pvals` for
/// geometric mean.
#[derive(Debug, Clone, PartialEq)]
pub struct LrRow {
    pub ligand: String,
    pub ligand_complex: String,
    pub ligand_means: f32,
    pub ligand_props: f64,
    pub receptor: String,
    pub receptor_complex: String,
    pub receptor_means: f32,
    pub receptor_props: f64,
    pub source: String,
    pub target: String,
    pub magnitude: f32,
    pub specificity: f64,
}

impl LrRow {
    /// The row as the oracle CSVs write it — plain comma-separated, no quoting
    /// (every string is a gene or cluster symbol), shortest-round-trip floats.
    pub fn to_csv_line(&self) -> String {
        format!(
            "{},{},{},{},{},{},{},{},{},{},{},{}",
            self.ligand,
            self.ligand_complex,
            self.ligand_means,
            self.ligand_props,
            self.receptor,
            self.receptor_complex,
            self.receptor_means,
            self.receptor_props,
            self.source,
            self.target,
            self.magnitude,
            self.specificity,
        )
    }
}

/// One row of `lr_res` — an exploded resource subunit under one cluster pair —
/// as `_get_lr`'s `_join_stats` merges build it.
#[derive(Debug, Clone)]
struct StatsRow {
    /// Cluster indices into [`Prep::labels`].
    source: usize,
    target: usize,
    /// The exploded subunit this row belongs to.
    subunit: usize,
    /// Identity of the row's (`ligand_complex`, `receptor_complex`) pair,
    /// deduplicated across subunits — the last two parts of liana's
    /// `[source, target, ligand_complex, receptor_complex]` key.
    pair: usize,
    ligand_means: f32,
    ligand_props: f64,
    receptor_means: f32,
    receptor_props: f64,
}

type Key = (usize, usize, usize);

fn key_of(row: &StatsRow) -> Key {
    (row.target, row.source, row.pair)
}

/// The shared front half of a permutation-scored method: `_prepare_lr_stats`'
/// reassembled row frame plus each row's `_run_method` permutation selection,
/// ready for a scorer and the streaming null engine.
struct Frame {
    prep: Prep,
    /// The surviving exploded subunits; `rows[i].subunit` names row `i`'s genes.
    subunits: Vec<LrSubunit>,
    rows: Vec<StatsRow>,
    /// The `perms[:, source, ligand]` / `perms[:, target, receptor]`
    /// coordinates `_run_method` selects.
    sel: Vec<RowSel>,
}

impl Frame {
    /// Row `index` as an output row, with the method's score columns.
    fn row(&self, index: usize, magnitude: f32, specificity: f64) -> LrRow {
        let row = &self.rows[index];
        let exploded = &self.subunits[row.subunit];
        LrRow {
            ligand: exploded.ligand.clone(),
            ligand_complex: exploded.ligand_complex.clone(),
            ligand_means: row.ligand_means,
            ligand_props: row.ligand_props,
            receptor: exploded.receptor.clone(),
            receptor_complex: exploded.receptor_complex.clone(),
            receptor_means: row.receptor_means,
            receptor_props: row.receptor_props,
            source: self.prep.labels[row.source].clone(),
            target: self.prep.labels[row.target].clone(),
            magnitude,
            specificity,
        }
    }

    /// The output rows with their p-values, sorted by `magnitude` descending —
    /// `_sort_by_score` (`_liana_pipe.py:465-476`).
    ///
    /// Deviation, documented in `ops/logs/w3-report.md`: pandas sorts through
    /// numpy's `nargsort`, whose tie order comes from an introsort/SIMD argsort
    /// with no cross-build contract — 220 of the 440 oracle rows sit in two-row
    /// tie groups. This port sorts stably instead: the values are unaffected and
    /// the parity gate is keyed on each row's four key columns.
    fn finish(&self, magnitudes: &[f32], specificities: Vec<f64>) -> Vec<LrRow> {
        let mut out: Vec<LrRow> = magnitudes
            .iter()
            .zip(specificities)
            .enumerate()
            .map(|(index, (&magnitude, specificity))| self.row(index, magnitude, specificity))
            .collect();
        out.sort_by(|a, b| {
            b.magnitude
                .partial_cmp(&a.magnitude)
                .expect("magnitude is never NaN")
        });
        out
    }
}

/// `_prepare_lr_stats` + reassembly, shared by the permutation-scored methods.
///
/// `expr_prop` and `min_cells` are liana's `expr_prop` (as in
/// `_filter_reassemble_complexes`) and the `min_cells` of `prep_check_adata`.
fn frame(adata: &Adata, resource: &[LrPair], expr_prop: f64, min_cells: usize) -> Result<Frame> {
    let prep = prep::prepare(adata, min_cells)?;

    // A passed resource is deduplicated on its pair columns, first occurrence
    // winning (`resource/select_resource.py:107`); the `dropna` half of that
    // line is unreachable for `LrPair`, which cannot carry NaN.
    let mut seen = HashSet::new();
    let resource: Vec<LrPair> = resource
        .iter()
        .filter(|pair| seen.insert((pair.ligand.clone(), pair.receptor.clone())))
        .cloned()
        .collect();

    let subunits =
        resource::filter_resource(&resource::explode_complexes(&resource), &prep.var_names);
    // Identity of each subunit's complex pair, so a key can group subunits of
    // the same pair without owning the symbols per row.
    let pair_ids: Vec<usize> = {
        let mut ids = Vec::with_capacity(subunits.len());
        let mut index: HashMap<(&str, &str), usize> = HashMap::new();
        for subunit in &subunits {
            let next = index.len();
            ids.push(
                *index
                    .entry((
                        subunit.ligand_complex.as_str(),
                        subunit.receptor_complex.as_str(),
                    ))
                    .or_insert(next),
            );
        }
        ids
    };

    // `_get_lr`'s rows: the cluster pairs of `np.meshgrid(labels, labels)` —
    // source fastest, target slowest — each crossed with the resource's
    // subunits in resource order (`_liana_pipe.py:545-555`), with
    // `_join_stats`' per-label `means`/`props` attached.
    let mut rows: Vec<StatsRow> =
        Vec::with_capacity(prep.n_labels() * prep.n_labels() * subunits.len());
    for target in 0..prep.n_labels() {
        for source in 0..prep.n_labels() {
            for (subunit, exploded) in subunits.iter().enumerate() {
                let ligand = gene_index(&prep, &exploded.ligand, "ligand")?;
                let receptor = gene_index(&prep, &exploded.receptor, "receptor")?;
                rows.push(StatsRow {
                    source,
                    target,
                    subunit,
                    pair: pair_ids[subunit],
                    ligand_means: prep.mean(source, ligand),
                    ligand_props: prep.prop(source, ligand),
                    receptor_means: prep.mean(target, receptor),
                    receptor_props: prep.prop(target, receptor),
                });
            }
        }
    }

    reassemble(&mut rows, expr_prop)?;

    // `_run_method`'s selection (`_liana_pipe.py:682-686`): the cube
    // coordinates each row scores against, resolved once.
    let mut sel = Vec::with_capacity(rows.len());
    for row in &rows {
        let exploded = &subunits[row.subunit];
        sel.push(RowSel {
            source: row.source as u32,
            target: row.target as u32,
            ligand: gene_index(&prep, &exploded.ligand, "ligand")? as u32,
            receptor: gene_index(&prep, &exploded.receptor, "receptor")? as u32,
        });
    }

    Ok(Frame {
        prep,
        subunits,
        rows,
        sel,
    })
}

/// The cellphonedb run for one `n_perms`: `lr_means` = the `f32` mean of the
/// two subunit means, zeroed when either side is zero
/// (`method/sc/_cellphonedb.py:17-41`), and `cellphone_pvals` over the nulls.
pub fn run_cellphonedb(
    adata: &Adata,
    resource: &[LrPair],
    expr_prop: f64,
    min_cells: usize,
    seed: u64,
    n_perms: usize,
) -> Result<Vec<LrRow>> {
    let frame = frame(adata, resource, expr_prop, min_cells)?;

    // `_cpdb_score`: zero_msk = either subunit mean zero (`_cellphonedb.py:22-26`)
    let magnitudes: Vec<f32> = frame
        .rows
        .iter()
        .map(|row| {
            if row.ligand_means == 0.0 || row.receptor_means == 0.0 {
                0.0
            } else {
                (row.ligand_means + row.receptor_means) / 2.0
            }
        })
        .collect();
    let truth: Vec<f64> = magnitudes.iter().map(|&m| f64::from(m)).collect();
    let specificities = engine::pvals_streaming(
        &frame.prep,
        &frame.sel,
        &truth,
        seed,
        n_perms,
        0,
        |ligand, receptor| (ligand + receptor) / 2.0,
    );
    Ok(frame.finish(&magnitudes, specificities))
}

/// The geometric_mean run for one `n_perms`: `lr_gmeans` = scipy's `gmean` of
/// the two subunit means in `f32` (`method/sc/_geometric_mean.py:28`) and
/// `gmean_pvals` over the nulls.
pub fn run_geometric_mean(
    adata: &Adata,
    resource: &[LrPair],
    expr_prop: f64,
    min_cells: usize,
    seed: u64,
    n_perms: usize,
) -> Result<Vec<LrRow>> {
    let frame = frame(adata, resource, expr_prop, min_cells)?;
    let magnitudes: Vec<f32> = frame
        .rows
        .iter()
        .map(|row| gmean32(row.ligand_means, row.receptor_means))
        .collect();
    let truth: Vec<f64> = magnitudes.iter().map(|&m| f64::from(m)).collect();
    let specificities = engine::pvals_streaming(
        &frame.prep,
        &frame.sel,
        &truth,
        seed,
        n_perms,
        0,
        |ligand, receptor| ((ligand.ln() + receptor.ln()) / 2.0).exp(),
    );
    Ok(frame.finish(&magnitudes, specificities))
}

/// `_filter_reassemble_complexes` (`liana/resource/_reassemble_complexes.py:11-87`):
/// drop the keys whose `prop_min` is below `expr_prop`, reduce each complex's
/// statistics to its subunits' minimum, and leave one row per key.
fn reassemble(rows: &mut Vec<StatsRow>, expr_prop: f64) -> Result<()> {
    // `prop_min` (`:46-54`): the minimum of the key's stacked
    // `ligand_props`/`receptor_props` — over every exploded subunit row, both
    // sides.
    let mut prop_min: HashMap<Key, f64> = HashMap::new();
    for row in rows.iter() {
        let value = row.ligand_props.min(row.receptor_props);
        prop_min
            .entry(key_of(row))
            .and_modify(|min| *min = min.min(value))
            .or_insert(value);
    }

    let kept: HashSet<Key> = prop_min
        .iter()
        .filter(|&(_, &min)| min >= expr_prop)
        .map(|(&key, _)| key)
        .collect();
    if kept.is_empty() && !prop_min.is_empty() {
        let highest = prop_min.values().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        bail!(
            "no ligand-receptor pair passed expr_prop={expr_prop}: the highest minimum subunit \
             proportion is {highest}"
        );
    }
    rows.retain(|row| kept.contains(&key_of(row)));

    // `_reduce_complexes` (`:79-85`), `ligand_means` then `receptor_means`:
    // each pass keeps only the rows tied at that column's per-key minimum.
    reduce_by_min(rows, |row| row.ligand_means);
    reduce_by_min(rows, |row| row.receptor_means);

    // `drop_duplicates(subset=_key_cols, keep="first")` (`:85`): one row per
    // key, the first in the surviving order.
    let mut seen = HashSet::new();
    rows.retain(|row| seen.insert(key_of(row)));
    Ok(())
}

fn reduce_by_min(rows: &mut Vec<StatsRow>, value: impl Fn(&StatsRow) -> f32) {
    let mut mins: HashMap<Key, f32> = HashMap::new();
    for row in rows.iter() {
        mins.entry(key_of(row))
            .and_modify(|min| *min = min.min(value(row)))
            .or_insert_with(|| value(row));
    }
    rows.retain(|row| value(row) == mins[&key_of(row)]);
}

fn gene_index(prep: &Prep, name: &str, what: &str) -> Result<usize> {
    prep.gene_index(name)
        .ok_or_else(|| anyhow::anyhow!("{what} {name:?} is absent from the prepared var_names"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::Csr;

    fn prep() -> Prep {
        Prep {
            var_names: vec!["lig".into(), "rec".into()],
            labels: vec!["A".into()],
            counts: vec![1],
            cell_cluster: vec![0],
            x: Csr {
                n_rows: 1,
                n_cols: 2,
                indptr: vec![0, 2],
                indices: vec![0, 1],
                data: vec![2.0, 4.0],
            },
            means: vec![2.0, 4.0],
            props: vec![1.0, 1.0],
        }
    }

    /// The pipeline's row frame and score columns on a one-cell, one-cluster,
    /// one-pair input, where every intermediate is hand-checkable.
    #[test]
    fn a_plain_pair_scores_through() {
        let adata = Adata {
            x: prep().x,
            obs_names: vec!["c".into()],
            var_names: prep().var_names,
            labels: vec![0],
            label_names: vec!["A".into()],
            obsm_spatial: None,
        };
        let resource = vec![LrPair {
            ligand: "lig".into(),
            receptor: "rec".into(),
        }];
        let rows = run_cellphonedb(&adata, &resource, 0.05, 0, 1337, 4).unwrap();
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.ligand, "lig");
        assert_eq!(row.receptor, "rec");
        assert_eq!((row.source.as_str(), row.target.as_str()), ("A", "A"));
        assert_eq!(row.ligand_means, 2.0);
        assert_eq!(row.receptor_means, 4.0);
        assert_eq!(row.magnitude, 3.0);
        // with a single cell every permutation leaves the mean unchanged, so
        // every permutation ties the truth and the p-value saturates
        assert_eq!(row.specificity, 1.0);

        // the same frame through the geometric-mean scorer: gmean(2, 4)
        let rows = run_geometric_mean(&adata, &resource, 0.05, 0, 1337, 4).unwrap();
        let row = &rows[0];
        assert!((row.magnitude - (2.0f32 * 4.0).sqrt()).abs() < 1e-6);
        assert_eq!(row.specificity, 1.0);
    }

    fn row(subunit: usize, ligand_means: f32, receptor_means: f32) -> StatsRow {
        StatsRow {
            source: 0,
            target: 0,
            subunit,
            pair: 0,
            ligand_means,
            ligand_props: 1.0,
            receptor_means,
            receptor_props: 1.0,
        }
    }

    /// `_reduce_complexes` (`:90-114`) runs the two columns' minimum reductions
    /// in sequence, so the receptor statistic is the one the ligand pass left
    /// behind — not the key's receptor minimum.
    #[test]
    fn complexes_reduce_column_by_column() {
        let mut rows = vec![row(0, 5.0, 1.0), row(1, 1.0, 9.0)];
        reassemble(&mut rows, 0.05).unwrap();
        assert_eq!(rows.len(), 1, "one row per key survives");
        assert_eq!(rows[0].subunit, 1, "the ligand pass dropped subunit 0");
        assert_eq!(rows[0].ligand_means, 1.0);
        assert_eq!(rows[0].receptor_means, 9.0, "subunit 1's own stat");
    }

    /// A tie on the first column narrows the second pass, and the whole
    /// surviving row — subunit symbol included — is the first one left
    /// (`drop_duplicates(keep="first")`, `:85`).
    #[test]
    fn complex_ties_keep_the_first_row_whole() {
        let mut rows = vec![row(0, 1.0, 5.0), row(1, 1.0, 9.0)];
        reassemble(&mut rows, 0.05).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].subunit, 0);
        assert_eq!(rows[0].receptor_means, 5.0);
    }
}
