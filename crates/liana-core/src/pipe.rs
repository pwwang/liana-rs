//! The single-cell pipelines end to end — `_prepare_lr_stats`, `_run_method`
//! and `_sort_by_score` for `liana.method.sc._cellphonedb`,
//! `liana.method.sc._geometric_mean`, `liana.method.sc._cellchat` and the
//! non-permutation `liana.method.sc._connectome`.
//!
//! Reproduces `testdata/expected/synthetic__<method>__p{100,1000}.csv` from
//! `testdata/fixtures/synthetic.h5ad`, the toy resource, `seed=1337` and the
//! `expr_prop`/`min_cells` the oracle ran with. The non-permutation methods
//! ignore `seed`/`n_perms` — their two `p<N>` runs are identical by
//! construction.

use std::collections::{HashMap, HashSet};

use anyhow::{Result, bail};

use crate::io::Adata;
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

/// One row of a non-permutation method's output: every cell as the oracle CSV
/// writes it, in the oracle's column order — the frame's columns as
/// `np.union1d` alphabetizes them, then the method's score columns, appended by
/// `_run_method` (`_liana_pipe.py:721-724`).
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
                    ligand_trimean,
                    receptor_trimean,
                });
            }
        }
    }

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
        0,
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
        0,
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
        0,
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
}
