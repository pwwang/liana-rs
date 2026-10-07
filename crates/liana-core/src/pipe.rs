//! The single-cell pipelines end to end — `_prepare_lr_stats`, `_run_method`
//! and `_sort_by_score` for the permutation-scored
//! `liana.method.sc._cellphonedb`, `_geometric_mean` and `_cellchat`, and the
//! non-permutation `liana.method.sc._connectome`, `_logfc`, `_natmi`,
//! `_scseqcomm` and `_singlecellsignalr`.
//!
//! Reproduces `testdata/expected/synthetic__<method>__p{100,1000}.csv` from
//! `testdata/fixtures/synthetic.h5ad`, the toy resource, `seed=1337` and the
//! `expr_prop`/`min_cells` the oracle ran with. The non-permutation methods
//! ignore `seed`/`n_perms` — their two `p<N>` runs are identical by
//! construction.

use std::collections::{HashMap, HashSet};

use anyhow::{Result, bail};

use crate::io::Adata;
use crate::math::{betainc, ndtr, std_f32, sum_f32};
use crate::perms::engine::{self, Aggregation, RowSel};
use crate::perms::null::gmean32;
use crate::perms::trimean;
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

/// The oracle CSV header of [`run_cellchat`]'s rows (`np.union1d` orders the
/// frame's columns alphabetically, and cellchat's `_complex_cols` reassemble
/// in place of the means).
pub const CELLCHAT_CSV_HEADER: &str = "ligand,ligand_complex,ligand_props,ligand_trimean,\
                                       mat_max,receptor,receptor_complex,receptor_props,\
                                       receptor_trimean,source,target,lr_probs,cellchat_pvals";

/// The oracle CSV header of [`run_connectome`]'s rows.
pub const CONNECTOME_CSV_HEADER: &str = "ligand,ligand_complex,ligand_means,ligand_props,ligand_zscores,\
                                        receptor,receptor_complex,receptor_means,receptor_props,\
                                        receptor_zscores,source,target,expr_prod,scaled_weight";

/// The oracle CSV header of [`run_logfc`]'s rows.
pub const LOGFC_CSV_HEADER: &str = "ligand,ligand_complex,ligand_logfc,ligand_means,ligand_props,\
                                   receptor,receptor_complex,receptor_logfc,receptor_means,\
                                   receptor_props,source,target,lr_logfc";

/// The oracle CSV header of [`run_natmi`]'s rows.
pub const NATMI_CSV_HEADER: &str = "ligand,ligand_complex,ligand_means,ligand_means_sums,ligand_props,\
                                   receptor,receptor_complex,receptor_means,receptor_means_sums,\
                                   receptor_props,source,target,expr_prod,spec_weight";

/// The oracle CSV header of [`run_scseqcomm`]'s rows.
pub const SCSEQCOMM_CSV_HEADER: &str = "ligand,ligand_cdf,ligand_complex,ligand_means,ligand_props,\
                                        receptor,receptor_cdf,receptor_complex,receptor_means,\
                                        receptor_props,source,target,inter_score";

/// The oracle CSV header of [`run_singlecellsignalr`]'s rows.
pub const SINGLECELLSIGNALR_CSV_HEADER: &str = "ligand,ligand_complex,ligand_means,ligand_props,\
                                               mat_mean,receptor,receptor_complex,\
                                               receptor_means,receptor_props,source,target,lrscore";

/// The oracle CSV header of [`run_rank_aggregate`]'s rows — `_aggregate`'s
/// joined frame (`_core/_pipe_utils/_aggregate.py:75-79`): the four keys, the
/// payload columns `liana_pipe_consensus`' first method's frame carries, then
/// one column per distinct score in `methods=` order, then the two consensus
/// ranks, assigned by `_aggregate`'s two `lr_res[consensus.<option>] = ...`.
pub const RANK_AGGREGATE_CSV_HEADER: &str = "source,target,ligand_complex,receptor_complex,\
                                             ligand,receptor,ligand_means,receptor_means,\
                                             ligand_props,receptor_props,lr_means,cellphone_pvals,\
                                             expr_prod,scaled_weight,lr_logfc,spec_weight,lrscore,\
                                             specificity_rank,magnitude_rank";

/// One row of a non-permutation method's output: every cell as the oracle CSV
/// writes it, in the oracle's column order — the frame's columns as
/// `np.union1d` alphabetizes them, then the method's score columns, appended by
/// `_run_method` (`_liana_pipe.py:721-724`); [`run_rank_aggregate`] follows
/// `_aggregate`'s join order instead.
///
/// Each cell carries Rust's shortest-round-trip `Display`, which parses back —
/// Python's `float()`, or `str::parse` in the parity test — to the same bits;
/// the reader never depends on the decimal spelling matching pandas'.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub cells: Vec<String>,
}

impl Row {
    /// The row as the oracle CSV writes it, like [`LrRow::to_csv_line`].
    pub fn to_csv_line(&self) -> String {
        self.cells.join(",")
    }
}

/// One row of cellchat's output, in the oracle CSV's column order: `lr_probs`
/// and `cellchat_pvals` are `_cellchat`'s magnitude and specificity, and the
/// trimean columns replace the means`.
#[derive(Debug, Clone, PartialEq)]
pub struct CellchatRow {
    pub ligand: String,
    pub ligand_complex: String,
    pub ligand_props: f64,
    pub ligand_trimean: f64,
    pub mat_max: f32,
    pub receptor: String,
    pub receptor_complex: String,
    pub receptor_props: f64,
    pub receptor_trimean: f64,
    pub source: String,
    pub target: String,
    pub lr_probs: f64,
    pub cellchat_pvals: f64,
}

impl CellchatRow {
    /// The row as the oracle CSV writes it, like [`LrRow::to_csv_line`].
    pub fn to_csv_line(&self) -> String {
        format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{}",
            self.ligand,
            self.ligand_complex,
            self.ligand_props,
            self.ligand_trimean,
            self.mat_max,
            self.receptor,
            self.receptor_complex,
            self.receptor_props,
            self.receptor_trimean,
            self.source,
            self.target,
            self.lr_probs,
            self.cellchat_pvals,
        )
    }
}

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
    /// NATMI's `_sum_means` columns, tagged over the *pre*-reassemble frame
    /// before the complex reduction drops subunit rows.
    ligand_means_sums: f32,
    receptor_means_sums: f32,
    /// The trimean columns cellchat's `_complex_cols` reassemble in place of
    /// the means; `None` on the mean-aggregation frames, which never read
    /// them.
    ligand_trimean: Option<f64>,
    receptor_trimean: Option<f64>,
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
    /// cellchat's `mat_max` — `np.float32(get_x(adata).max())` over the
    /// prepared matrix (`_liana_pipe.py:141`), which is also the null's
    /// `norm_factor`; `None` on the mean-aggregation frames.
    mat_max: Option<f32>,
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

    /// The cellchat output rows with their p-values, sorted by `lr_probs`
    /// descending — `_sort_by_score` again, on `_cellchat`'s magnitude
    /// (`magnitude_ascending=False`).
    fn finish_cellchat(&self, lr_probs: &[f64], pvals: Vec<f64>) -> Vec<CellchatRow> {
        let mat_max = self.mat_max.expect("trimean frame");
        let mut out: Vec<CellchatRow> = lr_probs
            .iter()
            .zip(pvals)
            .enumerate()
            .map(|(index, (&lr_probs, cellchat_pvals))| {
                let row = &self.rows[index];
                let exploded = &self.subunits[row.subunit];
                CellchatRow {
                    ligand: exploded.ligand.clone(),
                    ligand_complex: exploded.ligand_complex.clone(),
                    ligand_props: row.ligand_props,
                    ligand_trimean: row.ligand_trimean.expect("trimean frame"),
                    mat_max,
                    receptor: exploded.receptor.clone(),
                    receptor_complex: exploded.receptor_complex.clone(),
                    receptor_props: row.receptor_props,
                    receptor_trimean: row.receptor_trimean.expect("trimean frame"),
                    source: self.prep.labels[row.source].clone(),
                    target: self.prep.labels[row.target].clone(),
                    lr_probs,
                    cellchat_pvals,
                }
            })
            .collect();
        out.sort_by(|a, b| {
            b.lr_probs
                .partial_cmp(&a.lr_probs)
                .expect("lr_probs is never NaN")
        });
        out
    }
}

/// `_prepare_lr_stats` + reassembly, shared by the permutation-scored methods.
///
/// `expr_prop` and `min_cells` are liana's `expr_prop` (as in
/// `_filter_reassemble_complexes`) and the `min_cells` of `prep_check_adata`.
/// `trimean` selects cellchat's frame: the per-label trimeans of `X / mat_max`
/// join the rows and reassemble in place of the means.
fn frame(
    adata: &Adata,
    resource: &[LrPair],
    expr_prop: f64,
    min_cells: usize,
    trimean: bool,
) -> Result<Frame> {
    let prep = prep::prepare(adata, min_cells)?;
    // `_liana_pipe.py:144`'s overlap guard, before anything is filtered: a
    // resource that does not belong to this data is an error, not 0 rows.
    resource::assert_covered(resource, &prep.var_names)?;

    // cellchat's `mat_max`: the prepared matrix' maximum, the implicit zeros
    // as its floor (`np.max` of a sparse matrix), with `1.0f32 / mat_max` the
    // multiplier `(X / mat_max)` reduces to.
    let mat_max = trimean.then(|| prep.x.data.iter().copied().fold(0.0f32, f32::max));
    let trimeans = mat_max.map(|norm| trimean::observed(&prep, norm));

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
                let (ligand_trimean, receptor_trimean) = match &trimeans {
                    Some(table) => (
                        Some(table[source * prep.n_vars() + ligand]),
                        Some(table[target * prep.n_vars() + receptor]),
                    ),
                    None => (None, None),
                };
                rows.push(StatsRow {
                    source,
                    target,
                    subunit,
                    pair: pair_ids[subunit],
                    ligand_means: prep.mean(source, ligand),
                    ligand_props: prep.prop(source, ligand),
                    receptor_means: prep.mean(target, receptor),
                    receptor_props: prep.prop(target, receptor),
                    ligand_means_sums: 0.0,
                    receptor_means_sums: 0.0,
                    ligand_trimean,
                    receptor_trimean,
                });
            }
        }
    }

    // `_sum_means` runs on this unfiltered frame (`_liana_pipe.py:180-185`),
    // so tag the totals before `_filter_reassemble_complexes` narrows it.
    tag_means_sums(&mut rows);

    if trimean {
        reassemble(
            &mut rows,
            expr_prop,
            |row| row.ligand_trimean.expect("trimean frame"),
            |row| row.receptor_trimean.expect("trimean frame"),
        )?;
    } else {
        reassemble(
            &mut rows,
            expr_prop,
            |row| row.ligand_means,
            |row| row.receptor_means,
        )?;
    }

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
        mat_max,
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
    threads: usize,
) -> Result<Vec<LrRow>> {
    let frame = frame(adata, resource, expr_prop, min_cells, false)?;

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
        threads,
        Aggregation::Mean,
        |ligand, receptor| (ligand + receptor) / 2.0,
    );
    Ok(frame.finish(&magnitudes, specificities))
}

/// The cellchat run for one `n_perms`: `lr_probs` = the two subunit trimeans'
/// product through the Hill function `p / (0.5 + p)` (`_cellchat.py:16-25`),
/// and `cellchat_pvals` over the trimean nulls of the same statistic.
pub fn run_cellchat(
    adata: &Adata,
    resource: &[LrPair],
    expr_prop: f64,
    min_cells: usize,
    seed: u64,
    n_perms: usize,
    threads: usize,
) -> Result<Vec<CellchatRow>> {
    let frame = frame(adata, resource, expr_prop, min_cells, true)?;
    let norm = frame.mat_max.expect("trimean frame");

    // `_lr_probability` with `_KH = 0.5`; no zero mask here (`_cellchat_score`
    // has none), and `_apply_proximity_weights` is the identity without a
    // `proximity` column.
    let probability = |ligand: f64, receptor: f64| {
        let product = ligand * receptor;
        product / (0.5 + product)
    };
    let probs: Vec<f64> = frame
        .rows
        .iter()
        .map(|row| {
            probability(
                row.ligand_trimean.expect("trimean frame"),
                row.receptor_trimean.expect("trimean frame"),
            )
        })
        .collect();
    let pvals = engine::pvals_streaming(
        &frame.prep,
        &frame.sel,
        &probs,
        seed,
        n_perms,
        threads,
        Aggregation::Trimean { norm },
        probability,
    );
    Ok(frame.finish_cellchat(&probs, pvals))
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
    threads: usize,
) -> Result<Vec<LrRow>> {
    let frame = frame(adata, resource, expr_prop, min_cells, false)?;
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
        threads,
        Aggregation::Mean,
        |ligand, receptor| ((ligand.ln() + receptor.ln()) / 2.0).exp(),
    );
    Ok(frame.finish(&magnitudes, specificities))
}

/// The connectome run: `expr_prod` = the two subunit means' `f32` product and
/// `scaled_weight` = the two `*_zscores`' `f64` mean
/// (`method/sc/_connectome.py:17-38`), sorted by `expr_prod` descending
/// (`magnitude_ascending=False`).
///
/// `seed`/`n_perms` do not exist for connectome — `permute=False`, so the run
/// has no nulls at all (the p-value columns the oracle CSV lacks).
pub fn run_connectome(
    adata: &Adata,
    resource: &[LrPair],
    expr_prop: f64,
    min_cells: usize,
    _seed: u64,
    _n_perms: usize,
    _threads: usize,
) -> Result<Vec<Row>> {
    let frame = frame(adata, resource, expr_prop, min_cells, false)?;
    let zscores = scale_zscores(&frame.prep);
    let n_vars = frame.prep.n_vars();

    let mut scored: Vec<(f32, Row)> = Vec::with_capacity(frame.rows.len());
    for row in &frame.rows {
        let exploded = &frame.subunits[row.subunit];
        let ligand = gene_index(&frame.prep, &exploded.ligand, "ligand")?;
        let receptor = gene_index(&frame.prep, &exploded.receptor, "receptor")?;
        let ligand_zscores = zscores[row.source * n_vars + ligand];
        let receptor_zscores = zscores[row.target * n_vars + receptor];
        let magnitude = row.ligand_means * row.receptor_means;
        scored.push((
            magnitude,
            Row {
                cells: vec![
                    exploded.ligand.clone(),
                    exploded.ligand_complex.clone(),
                    row.ligand_means.to_string(),
                    row.ligand_props.to_string(),
                    ligand_zscores.to_string(),
                    exploded.receptor.clone(),
                    exploded.receptor_complex.clone(),
                    row.receptor_means.to_string(),
                    row.receptor_props.to_string(),
                    receptor_zscores.to_string(),
                    frame.prep.labels[row.source].clone(),
                    frame.prep.labels[row.target].clone(),
                    magnitude.to_string(),
                    ((ligand_zscores + receptor_zscores) / 2.0).to_string(),
                ],
            },
        ));
    }
    scored
        .sort_by(|(left, _), (right, _)| right.partial_cmp(left).expect("expr_prod is never NaN"));
    Ok(scored.into_iter().map(|(_, row)| row).collect())
}

/// The logfc run: `lr_logfc` = the two `*_logfc` columns' `f64` mean
/// (`method/sc/_logfc.py:11-13`), sorted by `lr_logfc` descending — logfc has
/// no magnitude, so `_sort_by_score` falls back to its specificity
/// (`_liana_pipe.py:465-475`). `seed`/`n_perms` do not exist for it
/// (`permute=False`).
pub fn run_logfc(
    adata: &Adata,
    resource: &[LrPair],
    expr_prop: f64,
    min_cells: usize,
    _seed: u64,
    _n_perms: usize,
    _threads: usize,
) -> Result<Vec<Row>> {
    let frame = frame(adata, resource, expr_prop, min_cells, false)?;
    let logfc = log2fc(&frame.prep);
    let n_vars = frame.prep.n_vars();

    let mut scored: Vec<(f64, Row)> = Vec::with_capacity(frame.rows.len());
    for row in &frame.rows {
        let exploded = &frame.subunits[row.subunit];
        let ligand = gene_index(&frame.prep, &exploded.ligand, "ligand")?;
        let receptor = gene_index(&frame.prep, &exploded.receptor, "receptor")?;
        let ligand_logfc = logfc[row.source * n_vars + ligand];
        let receptor_logfc = logfc[row.target * n_vars + receptor];
        let specificity = (ligand_logfc + receptor_logfc) / 2.0;
        scored.push((
            specificity,
            Row {
                cells: vec![
                    exploded.ligand.clone(),
                    exploded.ligand_complex.clone(),
                    ligand_logfc.to_string(),
                    row.ligand_means.to_string(),
                    row.ligand_props.to_string(),
                    exploded.receptor.clone(),
                    exploded.receptor_complex.clone(),
                    receptor_logfc.to_string(),
                    row.receptor_means.to_string(),
                    row.receptor_props.to_string(),
                    frame.prep.labels[row.source].clone(),
                    frame.prep.labels[row.target].clone(),
                    specificity.to_string(),
                ],
            },
        ));
    }
    scored.sort_by(|(left, _), (right, _)| right.partial_cmp(left).expect("lr_logfc is never NaN"));
    Ok(scored.into_iter().map(|(_, row)| row).collect())
}

/// The natmi run: `expr_prod` = the two subunit means' `f32` product and
/// `spec_weight` = those means over their `*_means_sums` totals, in `f32`
/// (`method/sc/_natmi.py:6-27`), sorted by `expr_prod` descending.
/// `seed`/`n_perms` do not exist for it (`permute=False`).
pub fn run_natmi(
    adata: &Adata,
    resource: &[LrPair],
    expr_prop: f64,
    min_cells: usize,
    _seed: u64,
    _n_perms: usize,
    _threads: usize,
) -> Result<Vec<Row>> {
    let frame = frame(adata, resource, expr_prop, min_cells, false)?;

    let mut scored: Vec<(f32, Row)> = Vec::with_capacity(frame.rows.len());
    for row in frame.rows.iter() {
        let exploded = &frame.subunits[row.subunit];
        let magnitude = row.ligand_means * row.receptor_means;
        let specificity = (row.ligand_means / row.ligand_means_sums)
            * (row.receptor_means / row.receptor_means_sums);
        scored.push((
            magnitude,
            Row {
                cells: vec![
                    exploded.ligand.clone(),
                    exploded.ligand_complex.clone(),
                    row.ligand_means.to_string(),
                    row.ligand_means_sums.to_string(),
                    row.ligand_props.to_string(),
                    exploded.receptor.clone(),
                    exploded.receptor_complex.clone(),
                    row.receptor_means.to_string(),
                    row.receptor_means_sums.to_string(),
                    row.receptor_props.to_string(),
                    frame.prep.labels[row.source].clone(),
                    frame.prep.labels[row.target].clone(),
                    magnitude.to_string(),
                    specificity.to_string(),
                ],
            },
        ));
    }
    scored
        .sort_by(|(left, _), (right, _)| right.partial_cmp(left).expect("expr_prod is never NaN"));
    Ok(scored.into_iter().map(|(_, row)| row).collect())
}

/// The scseqcomm run: `*_cdf` = the cluster's standard normal CDF at each
/// subunit's mean — `_gene_cdf` (`_liana_pipe.py:755-767`), zeroed where the
/// mean is zero — and `inter_score` = the pair-wise minimum of the two
/// (`method/sc/_scseqcomm.py:6-24`, its magnitude, descending).
/// `seed`/`n_perms` do not exist for it (`permute=False`).
///
/// The `z` is `(gene_mean − cluster_mean) / (cluster_std / sqrt(counts))`:
/// both means are the frame's `f32` columns, so the subtraction is `f32`;
/// the `std`/`counts` promotion to `f64` and the division are `f64`, as is
/// scipy's `norm.cdf` behind it (`_cluster_stats` and `_gene_cdf`,
/// `_liana_pipe.py:742-767`).
pub fn run_scseqcomm(
    adata: &Adata,
    resource: &[LrPair],
    expr_prop: f64,
    min_cells: usize,
    _seed: u64,
    _n_perms: usize,
    _threads: usize,
) -> Result<Vec<Row>> {
    let frame = frame(adata, resource, expr_prop, min_cells, false)?;
    let (cluster_means, cluster_stds) = cluster_stats(&frame.prep);

    let mut scored: Vec<(f64, Row)> = Vec::with_capacity(frame.rows.len());
    for row in &frame.rows {
        let exploded = &frame.subunits[row.subunit];
        let ligand_cdf = gene_cdf(
            row.ligand_means,
            cluster_means[row.source],
            cluster_stds[row.source],
            frame.prep.counts[row.source],
        );
        let receptor_cdf = gene_cdf(
            row.receptor_means,
            cluster_means[row.target],
            cluster_stds[row.target],
            frame.prep.counts[row.target],
        );
        let inter_score = ligand_cdf.min(receptor_cdf);
        scored.push((
            inter_score,
            Row {
                cells: vec![
                    exploded.ligand.clone(),
                    ligand_cdf.to_string(),
                    exploded.ligand_complex.clone(),
                    row.ligand_means.to_string(),
                    row.ligand_props.to_string(),
                    exploded.receptor.clone(),
                    receptor_cdf.to_string(),
                    exploded.receptor_complex.clone(),
                    row.receptor_means.to_string(),
                    row.receptor_props.to_string(),
                    frame.prep.labels[row.source].clone(),
                    frame.prep.labels[row.target].clone(),
                    inter_score.to_string(),
                ],
            },
        ));
    }
    scored.sort_by(|(left, _), (right, _)| {
        right.partial_cmp(left).expect("inter_score is never NaN")
    });
    Ok(scored.into_iter().map(|(_, row)| row).collect())
}

/// The singlecellsignalr run: `lrscore` = `sqrt(l) * sqrt(r)` over that plus
/// `mat_mean` (`method/sc/_singlecellsignalr.py:6-24`, its magnitude,
/// descending) — all `f32`, so the two roots and the division are `f32` too.
/// `seed`/`n_perms` do not exist for it (`permute=False`).
pub fn run_singlecellsignalr(
    adata: &Adata,
    resource: &[LrPair],
    expr_prop: f64,
    min_cells: usize,
    _seed: u64,
    _n_perms: usize,
    _threads: usize,
) -> Result<Vec<Row>> {
    let frame = frame(adata, resource, expr_prop, min_cells, false)?;
    let mat_mean = mat_mean(&frame.prep);

    let mut scored: Vec<(f32, Row)> = Vec::with_capacity(frame.rows.len());
    for row in &frame.rows {
        let exploded = &frame.subunits[row.subunit];
        let lr_sqrt = row.ligand_means.sqrt() * row.receptor_means.sqrt();
        let lrscore = lr_sqrt / (lr_sqrt + mat_mean);
        scored.push((
            lrscore,
            Row {
                cells: vec![
                    exploded.ligand.clone(),
                    exploded.ligand_complex.clone(),
                    row.ligand_means.to_string(),
                    row.ligand_props.to_string(),
                    mat_mean.to_string(),
                    exploded.receptor.clone(),
                    exploded.receptor_complex.clone(),
                    row.receptor_means.to_string(),
                    row.receptor_props.to_string(),
                    frame.prep.labels[row.source].clone(),
                    frame.prep.labels[row.target].clone(),
                    lrscore.to_string(),
                ],
            },
        ));
    }
    scored.sort_by(|(left, _), (right, _)| right.partial_cmp(left).expect("lrscore is never NaN"));
    Ok(scored.into_iter().map(|(_, row)| row).collect())
}

/// The `rank_aggregate` run for one `n_perms`: every consensus method over one
/// shared frame (`liana_pipe_consensus`, `method/sc/_liana_pipe.py:411-462`),
/// joined on the primary keys (`_aggregate`, `_core/_pipe_utils/_aggregate.py:35-99`),
/// then each consensus option's Robust Rank Aggregation (`_rank_aggregate` +
/// `_robust_rank_aggregate`, `:102-241`), sorted by `magnitude_rank` ascending
/// (`sort_values(order_col)`, then `_sort_by_score` with
/// `magnitude_ascending=True`).
///
/// The join is a pure payload merge: every method's reassembled frame carries
/// the same 440 keys in the same order and `_run_method(..., _aggregate_flag=True)`
/// narrows each to that key set plus its own scores, so `frames[0].merge(
/// frame.drop(columns=shared), how="outer")` only appends score columns. The
/// frame is therefore built once and scored seven ways — `lr_means` /
/// `cellphone_pvals` (CellPhoneDB, `frames[0]`), `expr_prod` (Connectome and
/// NATMI's shared product), `scaled_weight`, `lr_logfc`, `spec_weight`,
/// `lrscore` — matching the oracle CSV's column order.
///
/// Deviation, as documented in `ops/logs/w3-report.md` for the other runners:
/// the oracle's order inside `magnitude_rank` ties comes from pandas' unstable
/// sort/argsort; this port sorts stably, and the parity gate is keyed on the
/// four key columns.
pub fn run_rank_aggregate(
    adata: &Adata,
    resource: &[LrPair],
    expr_prop: f64,
    min_cells: usize,
    seed: u64,
    n_perms: usize,
    threads: usize,
) -> Result<Vec<Row>> {
    let frame = frame(adata, resource, expr_prop, min_cells, false)?;
    let n_vars = frame.prep.n_vars();

    // CellPhoneDB's `lr_means`: `_cpdb_score`'s zero mask and `f32` mean
    // (`method/sc/_cellphonedb.py:22-26`), then the permutation p-values over
    // the same statistic.
    let lr_means: Vec<f32> = frame
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
    let truth: Vec<f64> = lr_means.iter().map(|&m| f64::from(m)).collect();
    let cellphone_pvals = engine::pvals_streaming(
        &frame.prep,
        &frame.sel,
        &truth,
        seed,
        n_perms,
        threads,
        Aggregation::Mean,
        |ligand, receptor| (ligand + receptor) / 2.0,
    );

    // Connectome's and NATMI's `expr_prod`, the two subunit means' `f32`
    // product — the one score both methods rank, so it joins once.
    let expr_prod: Vec<f32> = frame
        .rows
        .iter()
        .map(|row| row.ligand_means * row.receptor_means)
        .collect();

    // Connectome's `scaled_weight` and logfc's `lr_logfc`: the `f64` means of
    // the two per-label tables `_get_lr` builds.
    let zscores = scale_zscores(&frame.prep);
    let logfc = log2fc(&frame.prep);
    let mut scaled_weight = vec![0.0f64; frame.rows.len()];
    let mut lr_logfc = vec![0.0f64; frame.rows.len()];
    for (index, row) in frame.rows.iter().enumerate() {
        let exploded = &frame.subunits[row.subunit];
        let ligand = gene_index(&frame.prep, &exploded.ligand, "ligand")?;
        let receptor = gene_index(&frame.prep, &exploded.receptor, "receptor")?;
        scaled_weight[index] =
            (zscores[row.source * n_vars + ligand] + zscores[row.target * n_vars + receptor]) / 2.0;
        lr_logfc[index] =
            (logfc[row.source * n_vars + ligand] + logfc[row.target * n_vars + receptor]) / 2.0;
    }

    // NATMI's `spec_weight` (`method/sc/_natmi.py:22-26`) and
    // SingleCellSignalR's `lrscore` (`method/sc/_singlecellsignalr.py:20-24`),
    // both all-`f32`.
    let mat_mean = mat_mean(&frame.prep);
    let mut spec_weight = vec![0.0f32; frame.rows.len()];
    let mut lrscore = vec![0.0f32; frame.rows.len()];
    for (index, row) in frame.rows.iter().enumerate() {
        spec_weight[index] = (row.ligand_means / row.ligand_means_sums)
            * (row.receptor_means / row.receptor_means_sums);
        let lr_sqrt = row.ligand_means.sqrt() * row.receptor_means.sqrt();
        lrscore[index] = lr_sqrt / (lr_sqrt + mat_mean);
    }

    // `AggregateClass`'s specs (`method/sc/_rank_aggregate.py:66-79`): one
    // `(score, ascending)` per method, deduplicated on the score name in
    // `_methods` order — Connectome's and NATMI's `expr_prod` are one column,
    // ranked once, descending. `lr_res` carries no missing score here, so the
    // `fillna(_assign_min_or_max(...))` arm of `_rank_aggregate:158` is
    // unreachable (it ranks with `nan_policy="propagate"`, which the frame's
    // no-NaN contract makes moot too).
    let specificity_rank = robust_rank_aggregate(&[
        RankSpec::F64(&cellphone_pvals, true),
        RankSpec::F64(&scaled_weight, false),
        RankSpec::F64(&lr_logfc, false),
        RankSpec::F32(&spec_weight, false),
    ]);
    let magnitude_rank = robust_rank_aggregate(&[
        RankSpec::F32(&lr_means, false),
        RankSpec::F32(&expr_prod, false),
        RankSpec::F32(&lrscore, false),
    ]);

    let mut scored: Vec<(f64, Row)> = Vec::with_capacity(frame.rows.len());
    for (index, row) in frame.rows.iter().enumerate() {
        let exploded = &frame.subunits[row.subunit];
        scored.push((
            magnitude_rank[index],
            Row {
                cells: vec![
                    frame.prep.labels[row.source].clone(),
                    frame.prep.labels[row.target].clone(),
                    exploded.ligand_complex.clone(),
                    exploded.receptor_complex.clone(),
                    exploded.ligand.clone(),
                    exploded.receptor.clone(),
                    row.ligand_means.to_string(),
                    row.receptor_means.to_string(),
                    row.ligand_props.to_string(),
                    row.receptor_props.to_string(),
                    lr_means[index].to_string(),
                    cellphone_pvals[index].to_string(),
                    expr_prod[index].to_string(),
                    scaled_weight[index].to_string(),
                    lr_logfc[index].to_string(),
                    spec_weight[index].to_string(),
                    lrscore[index].to_string(),
                    specificity_rank[index].to_string(),
                    magnitude_rank[index].to_string(),
                ],
            },
        ));
    }
    scored.sort_by(|(left, _), (right, _)| {
        left.partial_cmp(right)
            .expect("magnitude_rank is never NaN")
    });
    Ok(scored.into_iter().map(|(_, row)| row).collect())
}

/// One `(score column, ascending)` pair of a consensus option's specs, over
/// the columns of the joined frame.
enum RankSpec<'a> {
    /// A `f64` score column — `cellphone_pvals`, `scaled_weight` or `lr_logfc`.
    F64(&'a [f64], bool),
    /// An `f32` one. `rankdata` is dtype-preserving in scipy 1.18.1, but the
    /// average ranks are half-integers in `1..=440`, exact in both dtypes, so
    /// ranking in `f64` and casting the matrix back to `f32` is bit-identical.
    F32(&'a [f32], bool),
}

impl RankSpec<'_> {
    /// The column's row count.
    fn len(&self) -> usize {
        match self {
            RankSpec::F64(values, _) => values.len(),
            RankSpec::F32(values, _) => values.len(),
        }
    }
}

/// `_rank_aggregate`'s `rra` branch, `_robust_rank_aggregate` and `_rho_scores`
/// (`_core/_pipe_utils/_aggregate.py:156-241`), returning the consensus rank.
///
/// `np.column_stack` promotes the ranked columns to `f64` as soon as one of
/// them is `f64` — the specificity option mixes three `f64` scores with `f32`
/// `spec_weight` and lands in `f64`, the magnitude option is `f32` throughout.
/// The normalisation `rmat / np.max(rmat, axis=0)` and the row sort stay in
/// that dtype (`np.sort` on the `f32` matrix orders by the same values), and
/// the `f64` round-to-half-integer behaviour of the `f32` division is
/// observable in the CDF output, so it is reproduced. `beta.cdf(x, a, b)`
/// promotes to `f64` (int64 shape args) — so the CDF, the row minimum and the
/// `p * k` clip are all `f64`, through the [`betainc`] kernel.
///
/// `a = j + 1` and `b = k - j` for the column index `j` of a `k`-column matrix,
/// over the row's *sorted* normalised ranks (`dist_a`/`dist_b` are assigned
/// before `np.sort` moves the values, but they are functions of the column
/// index, not of the values).
fn robust_rank_aggregate(specs: &[RankSpec<'_>]) -> Vec<f64> {
    let k = specs.len();
    let n = specs[0].len();
    let double = specs.iter().any(|spec| matches!(spec, RankSpec::F64(..)));

    // rankdata(col * (1 if asc else -1), method="average"), per column, in f64.
    let mut columns: Vec<Vec<f64>> = Vec::with_capacity(k);
    for spec in specs {
        let (values, ascending) = match *spec {
            RankSpec::F64(values, ascending) => (values.to_vec(), ascending),
            RankSpec::F32(values, ascending) => {
                (values.iter().map(|&v| f64::from(v)).collect(), ascending)
            }
        };
        let signed: Vec<f64> = if ascending {
            values
        } else {
            values.into_iter().map(|v| -v).collect()
        };
        columns.push(average_ranks(&signed));
    }

    // `np.column_stack`'s dtype promotion, then `rmat / np.max(rmat, axis=0)`.
    let maxima: Vec<f64> = (0..k)
        .map(|j| columns[j].iter().copied().fold(f64::NEG_INFINITY, f64::max))
        .collect();
    let mut rmat: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..k).map(|j| columns[j][i]).collect())
        .collect();
    for row in rmat.iter_mut() {
        for (j, value) in row.iter_mut().enumerate() {
            *value = if double {
                *value / maxima[j]
            } else {
                f64::from((*value as f32) / (maxima[j] as f32))
            };
        }
        row.sort_by(|a, b| a.partial_cmp(b).expect("normalised ranks are never NaN"));
    }

    // `_rho_scores`: the beta CDF per (row, column), the row minimum, and
    // `_corr_beta_pvals`' `np.clip(p * k, 0, 1)`.
    rmat.iter()
        .map(|row| {
            let p = row
                .iter()
                .enumerate()
                .map(|(j, &x)| betainc((j + 1) as f64, (k - j) as f64, x))
                .fold(f64::INFINITY, f64::min);
            (p * k as f64).clamp(0.0, 1.0)
        })
        .collect()
}

/// `scipy.stats.rankdata(method="average")` over one score column: each tie
/// group's mean of the 1-based ranks `start + 1 ..= end` (0-based `start`,
/// exclusive `end`), i.e. the half-integer `(start + end + 1) / 2`
/// (`_rankdata.py:143-153`). The sort is only a grouping device — every member
/// of a group takes the group's rank — so the port's stability is immaterial.
fn average_ranks(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .expect("scores are never NaN")
    });

    let mut ranks = vec![0.0f64; values.len()];
    let mut start = 0usize;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && values[order[end]] == values[order[start]] {
            end += 1;
        }
        let rank = (start + end + 1) as f64 / 2.0;
        for &index in &order[start..end] {
            ranks[index] = rank;
        }
        start = end;
    }
    ranks
}

/// `_cluster_stats` (`_liana_pipe.py:742-751`): each cluster's scalar mean
/// and standard deviation over the prepared matrix, both `f32`.
///
/// The mean is scipy's sparse `mean(axis=None)` ([`sparse_mean`]). The
/// deviation is `np.std` of the cluster's *dense* block — the implicit zeros
/// included, flattened row-major — which is numpy's `f32` pairwise mean and
/// squared-deviation sum (`math::pairwise`).
fn cluster_stats(prep: &Prep) -> (Vec<f32>, Vec<f32>) {
    let n_vars = prep.n_vars();
    let mut means = vec![0.0f32; prep.n_labels()];
    let mut stds = vec![0.0f32; prep.n_labels()];
    for cluster in 0..prep.n_labels() {
        let scale = (1.0f64 / (prep.counts[cluster] * n_vars) as f64) as f32;
        let is_cluster = |row: usize| prep.cell_cluster[row] as usize == cluster;
        means[cluster] = sparse_mean(
            prep,
            scale,
            (0..prep.x.n_rows).filter(|&row| is_cluster(row)),
        );

        let mut dense = Vec::with_capacity(prep.counts[cluster] * n_vars);
        for row in (0..prep.x.n_rows).filter(|&row| is_cluster(row)) {
            let range = prep.x.indptr[row]..prep.x.indptr[row + 1];
            let (columns, values) = (&prep.x.indices[range.clone()], &prep.x.data[range]);
            let mut next = 0;
            for gene in 0..n_vars {
                let value = if next < columns.len() && columns[next] as usize == gene {
                    let value = values[next];
                    next += 1;
                    value
                } else {
                    0.0
                };
                dense.push(value);
            }
        }
        stds[cluster] = std_f32(&dense);
    }
    (means, stds)
}

/// `mat_mean` (`_liana_pipe.py:141`): `np.float32(get_x(adata).mean(dtype="float32"))`
/// — scipy's sparse `f32` mean over the whole prepared matrix.
fn mat_mean(prep: &Prep) -> f32 {
    let scale = (1.0f64 / (prep.x.n_rows * prep.n_vars()) as f64) as f32;
    sparse_mean(prep, scale, 0..prep.x.n_rows)
}

/// scipy's sparse scalar `mean(dtype=f32)`: every stored value of the given
/// rows scaled to `f32` by `scale` — the cached `1 / (rows * cols)` of the
/// block's shape — then numpy's pairwise sum, in row-major canonical order
/// (`scipy/sparse/_base.py:1574`; `prep`'s rows are column-sorted).
fn sparse_mean(prep: &Prep, scale: f32, rows: impl Iterator<Item = usize>) -> f32 {
    let mut scaled = Vec::new();
    for row in rows {
        let range = prep.x.indptr[row]..prep.x.indptr[row + 1];
        scaled.extend(prep.x.data[range].iter().map(|&value| value * scale));
    }
    sum_f32(&scaled)
}

/// `_gene_cdf` (`_liana_pipe.py:755-767`): `norm.cdf(gene_mean, loc=cluster_mean,
/// scale=cluster_std / sqrt(cluster_counts))`, with `probability[gene_mean == 0] = 0`.
///
/// The two means are `f32`, so their difference is `f32`; the scale promotes
/// to `f64` (`f32 / int` in numpy is `f64`) and the division and CDF are
/// `f64` — scipy's `ndtr` (`math::ndtr`).
fn gene_cdf(gene_mean: f32, cluster_mean: f32, cluster_std: f32, counts: usize) -> f64 {
    if gene_mean == 0.0 {
        return 0.0;
    }
    let scale = f64::from(cluster_std) / (counts as f64).sqrt();
    ndtr(f64::from(gene_mean - cluster_mean) / scale)
}

/// `_sum_means` (`_liana_pipe.py:570-571`): `lr_res.groupby(on)[what].sum()`
/// joined back on `on`, so every row of a group carries the group's `f32`
/// total. liana runs both passes on the *unfiltered* exploded frame
/// (`:180-185`), before `_filter_reassemble_complexes` narrows it, so the
/// totals cover the subunit rows the reduction later drops.
///
/// A group is one exploded subunit (`subunit` already names liana's key: the
/// `ligand_complex`/`receptor_complex` pair and both exploded symbols) under a
/// fixed target — the ligand pass, `P.complete` minus `source` — or a fixed
/// source (the receptor pass); all of the other side's labels sit in it.
fn tag_means_sums(rows: &mut [StatsRow]) {
    let mut ligand: HashMap<(usize, usize), (f32, f32)> = HashMap::new();
    let mut receptor: HashMap<(usize, usize), (f32, f32)> = HashMap::new();
    for row in rows.iter() {
        kahan(&mut ligand, (row.target, row.subunit), row.ligand_means);
        kahan(&mut receptor, (row.source, row.subunit), row.receptor_means);
    }
    for row in rows.iter_mut() {
        row.ligand_means_sums = ligand[&(row.target, row.subunit)].0;
        row.receptor_means_sums = receptor[&(row.source, row.subunit)].0;
    }
}

/// pandas' `groupby.sum()` over a `float32` column is Kahan-compensated, in
/// the frame's row order (`(total, compensation)`).
fn kahan(map: &mut HashMap<(usize, usize), (f32, f32)>, key: (usize, usize), value: f32) {
    let (total, compensation) = map.entry(key).or_insert((0.0, 0.0));
    let y = value - *compensation;
    let next = *total + y;
    *compensation = (next - *total) - y;
    *total = next;
}

/// `_calc_log2fc` (`_liana_pipe.py:583-596`) per label, `labels.len() *
/// var_names.len()` row-major: each subunit gene's subject-versus-rest log2
/// fold change of the normcounts layer's column means.
///
/// The layer inverts the `log1p(base=)` transform of the prepared matrix's
/// stored entries — `_expm1_base` (`:599-616`) with `base = V.logbase =
/// np.exp(1)`, `np.power` in `f64`. The means are scipy's
/// `(X * (1 / n)).sum(axis=0)`: each stored value scaled in `f64`, then summed
/// per column in ascending row order; `+ 1` and `np.log2` are `f64` libm —
/// bit-identical to numpy's loops over this data (checked against the oracle
/// venv, `scripts/dump_math_ref.py`'s precedent for f32).
fn log2fc(prep: &Prep) -> Vec<f64> {
    let base = std::f64::consts::E;
    let normcounts: Vec<f64> = prep
        .x
        .data
        .iter()
        .map(|&value| base.powf(f64::from(value)) - 1.0)
        .collect();

    let n_vars = prep.n_vars();
    let mut out = vec![0.0f64; prep.n_labels() * n_vars];
    for cluster in 0..prep.n_labels() {
        let subject_scale = 1.0 / prep.counts[cluster] as f64;
        let rest_scale = 1.0 / (prep.x.n_rows - prep.counts[cluster]) as f64;
        let mut subject = vec![0.0f64; n_vars];
        let mut rest = vec![0.0f64; n_vars];
        for row in 0..prep.x.n_rows {
            let range = prep.x.indptr[row]..prep.x.indptr[row + 1];
            let is_subject = prep.cell_cluster[row] as usize == cluster;
            let scale = if is_subject {
                subject_scale
            } else {
                rest_scale
            };
            let means = if is_subject { &mut subject } else { &mut rest };
            for (&gene, &value) in prep.x.indices[range.clone()].iter().zip(&normcounts[range]) {
                means[gene as usize] += value * scale;
            }
        }
        let base_index = cluster * n_vars;
        for gene in 0..n_vars {
            out[base_index + gene] = (subject[gene] + 1.0).log2() - (rest[gene] + 1.0).log2();
        }
    }
    out
}

/// The `*_zscores` table `_get_lr` builds for connectome: `sc.pp.scale` over
/// the prepared matrix, then each label's dense column means
/// (`_liana_pipe.py:494, 538`), `labels.len() * var_names.len()` row-major.
///
/// `sc.pp.scale` standardizes each column with `fast_array_utils`'
/// `mean_var` (`stats/_mean_var.py`, `correction=1`): `f64` accumulators, the
/// variance as `E[x²] − mean²` with each square taken in `f32` first, then
/// `(x − mean) / std` in `f64` (`zero_center` densifies) and zero `std`s
/// replaced by `1` (`scanpy/preprocessing/_scale.py`). The per-label means are
/// `np.mean(axis=0)` of that dense layer — sequential `f64` accumulation,
/// then one division.
fn scale_zscores(prep: &Prep) -> Vec<f64> {
    let n_vars = prep.n_vars();
    let n_rows = prep.x.n_rows;

    let mut sums = vec![0.0f64; n_vars];
    let mut squares = vec![0.0f64; n_vars];
    for row in 0..n_rows {
        let range = prep.x.indptr[row]..prep.x.indptr[row + 1];
        for (&gene, &value) in prep.x.indices[range.clone()]
            .iter()
            .zip(&prep.x.data[range])
        {
            sums[gene as usize] += f64::from(value);
            squares[gene as usize] += f64::from(value * value);
        }
    }
    let rows = n_rows as f64;
    let mut mean = vec![0.0f64; n_vars];
    let mut std = vec![0.0f64; n_vars];
    for gene in 0..n_vars {
        mean[gene] = sums[gene] / rows;
        let mut var = squares[gene] / rows - mean[gene] * mean[gene];
        var *= rows / (rows - 1.0);
        let deviation = var.sqrt();
        std[gene] = if deviation == 0.0 { 1.0 } else { deviation };
    }

    // The scaled layer's per-label column means, accumulated over each label's
    // cells in row order; a row without a stored entry contributes the dense
    // layer's own `(0 - mean) / std`.
    let mut out = vec![0.0f64; prep.n_labels() * n_vars];
    for row in 0..n_rows {
        let base = prep.cell_cluster[row] as usize * n_vars;
        let range = prep.x.indptr[row]..prep.x.indptr[row + 1];
        let (columns, values) = (&prep.x.indices[range.clone()], &prep.x.data[range]);
        let mut next = 0;
        for gene in 0..n_vars {
            let value = if next < columns.len() && columns[next] as usize == gene {
                let value = f64::from(values[next]);
                next += 1;
                value
            } else {
                0.0
            };
            out[base + gene] += (value - mean[gene]) / std[gene];
        }
    }
    for cluster in 0..prep.n_labels() {
        let base = cluster * n_vars;
        let cells = prep.counts[cluster] as f64;
        for gene in 0..n_vars {
            out[base + gene] /= cells;
        }
    }
    out
}

/// `_filter_reassemble_complexes` (`liana/resource/_reassemble_complexes.py:11-87`):
/// drop the keys whose `prop_min` is below `expr_prop`, reduce each complex's
/// statistics — the method's `_complex_cols`, ligand column first, then the
/// receptor's — to its subunits' minimum, and leave one row per key.
fn reassemble<T: PartialOrd + Copy>(
    rows: &mut Vec<StatsRow>,
    expr_prop: f64,
    ligand: impl Fn(&StatsRow) -> T,
    receptor: impl Fn(&StatsRow) -> T,
) -> Result<()> {
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

    // `_reduce_complexes` (`:79-85`), `_complex_cols` in order: each pass
    // keeps only the rows tied at that column's per-key minimum.
    reduce_by_min(rows, &ligand);
    reduce_by_min(rows, &receptor);

    // `drop_duplicates(subset=_key_cols, keep="first")` (`:85`): one row per
    // key, the first in the surviving order.
    let mut seen = HashSet::new();
    rows.retain(|row| seen.insert(key_of(row)));
    Ok(())
}

fn reduce_by_min<T: PartialOrd + Copy>(rows: &mut Vec<StatsRow>, value: impl Fn(&StatsRow) -> T) {
    let mut mins: HashMap<Key, T> = HashMap::new();
    for row in rows.iter() {
        let current = value(row);
        mins.entry(key_of(row))
            .and_modify(|min| {
                if current < *min {
                    *min = current;
                }
            })
            .or_insert(current);
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
        let rows = run_cellphonedb(&adata, &resource, 0.05, 0, 1337, 4, 0).unwrap();
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
        let rows = run_geometric_mean(&adata, &resource, 0.05, 0, 1337, 4, 0).unwrap();
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
            ligand_means_sums: 0.0,
            receptor_means_sums: 0.0,
            ligand_trimean: None,
            receptor_trimean: None,
        }
    }

    fn by_means(rows: &mut Vec<StatsRow>, expr_prop: f64) -> Result<()> {
        reassemble(
            rows,
            expr_prop,
            |row| row.ligand_means,
            |row| row.receptor_means,
        )
    }

    /// `_reduce_complexes` (`:90-114`) runs the two columns' minimum reductions
    /// in sequence, so the receptor statistic is the one the ligand pass left
    /// behind — not the key's receptor minimum.
    #[test]
    fn complexes_reduce_column_by_column() {
        let mut rows = vec![row(0, 5.0, 1.0), row(1, 1.0, 9.0)];
        by_means(&mut rows, 0.05).unwrap();
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
        by_means(&mut rows, 0.05).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].subunit, 0);
        assert_eq!(rows[0].receptor_means, 5.0);
    }

    /// `_sum_means` sums each group's `f32` values the way pandas does —
    /// Kahan-compensated, in frame order — so this group's total is
    /// `0.2015939`, not the naive summation's `0.20159392`.
    #[test]
    fn means_sums_are_kahan_compensated() {
        let values = [
            0.0151153505f32,
            0.017532349,
            0.013437928,
            0.026385676,
            0.016282007,
            0.039996352,
            0.0,
            0.07284425,
        ];
        let mut rows: Vec<StatsRow> = values
            .iter()
            .enumerate()
            .map(|(source, &value)| {
                let mut row = row(0, value, value);
                row.source = source;
                row
            })
            .collect();
        tag_means_sums(&mut rows);
        assert_eq!(rows[0].ligand_means_sums, 0.2015939);
        assert_eq!(
            values.iter().fold(0.0f32, |sum, &value| sum + value),
            0.20159392
        );
    }
}
