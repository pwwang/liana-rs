//! The named-method dispatch the `liana-rs` CLI and the `liana_rs` Python
//! module share: [`Method`] enumerates liana's nine single-cell methods with
//! their CSV contracts, [`Settings`] carries liana's defaults
//! (`_core/_constants.py`'s `DefaultValues`), and [`Method::run`] returns the
//! oracle CSV's header and raw fields.
//!
//! The field lists are the ones `tests/{pipe,cellchat}_parity.rs` pin — the
//! port's score columns in the oracle CSV's order.

use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::io::{Adata, read_h5ad};
use crate::pipe::{
    CELLCHAT_CSV_HEADER, CONNECTOME_CSV_HEADER, CPDB_CSV_HEADER, CellchatRow, GMEAN_CSV_HEADER,
    LOGFC_CSV_HEADER, LrRow, NATMI_CSV_HEADER, RANK_AGGREGATE_CSV_HEADER, SCSEQCOMM_CSV_HEADER,
    SINGLECELLSIGNALR_CSV_HEADER, run_cellchat, run_cellphonedb, run_connectome,
    run_geometric_mean, run_logfc, run_natmi, run_rank_aggregate, run_scseqcomm,
    run_singlecellsignalr,
};
use crate::resource::{self, LrPair};

/// liana's `li.mt`[`Method`] names, as the CLI and the Python module spell them
/// (`logfc`, not the class's `log2FC`).
pub const METHOD_NAMES: &[&str] = &[
    "cellphonedb",
    "geometric_mean",
    "cellchat",
    "connectome",
    "logfc",
    "natmi",
    "scseqcomm",
    "singlecellsignalr",
    "rank_aggregate",
];

/// One of liana's nine single-cell methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Cellphonedb,
    GeometricMean,
    Cellchat,
    Connectome,
    Logfc,
    Natmi,
    Scseqcomm,
    Singlecellsignalr,
    RankAggregate,
}

impl Method {
    /// The method a name selects; the error lists every valid name.
    pub fn parse(name: &str) -> Result<Self> {
        Ok(match name {
            "cellphonedb" => Method::Cellphonedb,
            "geometric_mean" => Method::GeometricMean,
            "cellchat" => Method::Cellchat,
            "connectome" => Method::Connectome,
            "logfc" => Method::Logfc,
            "natmi" => Method::Natmi,
            "scseqcomm" => Method::Scseqcomm,
            "singlecellsignalr" => Method::Singlecellsignalr,
            "rank_aggregate" => Method::RankAggregate,
            other => bail!("unknown method {other:?}; choose from {METHOD_NAMES:?}"),
        })
    }

    /// The name [`Self::parse`] accepts.
    pub fn name(self) -> &'static str {
        match self {
            Method::Cellphonedb => "cellphonedb",
            Method::GeometricMean => "geometric_mean",
            Method::Cellchat => "cellchat",
            Method::Connectome => "connectome",
            Method::Logfc => "logfc",
            Method::Natmi => "natmi",
            Method::Scseqcomm => "scseqcomm",
            Method::Singlecellsignalr => "singlecellsignalr",
            Method::RankAggregate => "rank_aggregate",
        }
    }

    /// The oracle CSV header the output rows match, column for column.
    pub fn csv_header(self) -> &'static str {
        match self {
            Method::Cellphonedb => CPDB_CSV_HEADER,
            Method::GeometricMean => GMEAN_CSV_HEADER,
            Method::Cellchat => CELLCHAT_CSV_HEADER,
            Method::Connectome => CONNECTOME_CSV_HEADER,
            Method::Logfc => LOGFC_CSV_HEADER,
            Method::Natmi => NATMI_CSV_HEADER,
            Method::Scseqcomm => SCSEQCOMM_CSV_HEADER,
            Method::Singlecellsignalr => SINGLECELLSIGNALR_CSV_HEADER,
            Method::RankAggregate => RANK_AGGREGATE_CSV_HEADER,
        }
    }

    /// The method's run, as the CSV's raw fields per row.
    ///
    /// `seed`/`n_perms` drive the permutation-scored methods and are ignored
    /// by the non-permutation ones; `threads` reaches the permutation engine
    /// for the first four and is ignored by the rest.
    pub fn run(self, adata: &Adata, resource: &[LrPair], settings: &Settings) -> Result<Output> {
        let Settings {
            expr_prop,
            min_cells,
            seed,
            n_perms,
            threads,
        } = *settings;
        let rows = match self {
            Method::Cellphonedb => lr_fields(&run_cellphonedb(
                adata, resource, expr_prop, min_cells, seed, n_perms, threads,
            )?),
            Method::GeometricMean => lr_fields(&run_geometric_mean(
                adata, resource, expr_prop, min_cells, seed, n_perms, threads,
            )?),
            Method::Cellchat => cellchat_fields(&run_cellchat(
                adata, resource, expr_prop, min_cells, seed, n_perms, threads,
            )?),
            Method::Connectome => cells(run_connectome(
                adata, resource, expr_prop, min_cells, seed, n_perms, threads,
            )?),
            Method::Logfc => cells(run_logfc(
                adata, resource, expr_prop, min_cells, seed, n_perms, threads,
            )?),
            Method::Natmi => cells(run_natmi(
                adata, resource, expr_prop, min_cells, seed, n_perms, threads,
            )?),
            Method::Scseqcomm => cells(run_scseqcomm(
                adata, resource, expr_prop, min_cells, seed, n_perms, threads,
            )?),
            Method::Singlecellsignalr => cells(run_singlecellsignalr(
                adata, resource, expr_prop, min_cells, seed, n_perms, threads,
            )?),
            Method::RankAggregate => cells(run_rank_aggregate(
                adata, resource, expr_prop, min_cells, seed, n_perms, threads,
            )?),
        };
        Ok(Output {
            header: self.csv_header(),
            rows,
        })
    }
}

/// liana's run defaults (`_core/_constants.py:DefaultValues`): `expr_prop=0.05`,
/// `min_cells=5`, `n_perms=1000`, `seed=1337`. `threads = 0` is the rayon
/// default pool — liana's `n_jobs=1` is a liana-side reproducibility choice,
/// not part of the result contract: the engine's p-values are bit-identical
/// for every worker count.
#[derive(Debug, Clone, Copy)]
pub struct Settings {
    pub expr_prop: f64,
    pub min_cells: usize,
    pub n_perms: usize,
    pub seed: u64,
    pub threads: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            expr_prop: 0.05,
            min_cells: 5,
            n_perms: 1000,
            seed: 1337,
            threads: 0,
        }
    }
}

/// One method's output: the CSV contract's header and its rows as raw fields,
/// each cell spelled so that parsing it back recovers the same value (Rust's
/// shortest-round-trip `Display`).
#[derive(Debug, Clone, PartialEq)]
pub struct Output {
    pub header: &'static str,
    pub rows: Vec<Vec<String>>,
}

impl Output {
    /// The header and rows as an oracle-style CSV (comma-separated, no
    /// quoting — every field is a gene, cluster or number).
    pub fn to_csv(&self) -> String {
        let mut text = String::from(self.header);
        text.push('\n');
        for row in &self.rows {
            text.push_str(&row.join(","));
            text.push('\n');
        }
        text
    }
}

/// Read one `.h5ad`, resolve one resource and run one named method — the whole
/// `liana-rs run` / `liana_rs.run` call.
pub fn run_file(
    h5ad: &Path,
    label_key: &str,
    resource: &str,
    method: &str,
    settings: &Settings,
) -> Result<Output> {
    let adata = read_h5ad(h5ad, label_key).with_context(|| format!("read {}", h5ad.display()))?;
    let pairs = resolve_resource(resource)?;
    Method::parse(method)?.run(&adata, &pairs, settings)
}

/// The pairs a single `resource` string names: the `ligand,receptor` CSV at
/// that path when the path exists, otherwise a resource name in the vendored
/// omni resource (`consensus`, …) — `resource::select`'s error when neither.
pub fn resolve_resource(resource: &str) -> Result<Vec<LrPair>> {
    let path = Path::new(resource);
    if path.exists() {
        resource::read_pairs(path)
    } else {
        resource::select(resource).with_context(|| {
            format!("{resource:?} is neither an existing resource file nor a resource name")
        })
    }
}

fn lr_fields(rows: &[LrRow]) -> Vec<Vec<String>> {
    rows.iter()
        .map(|row| {
            vec![
                row.ligand.clone(),
                row.ligand_complex.clone(),
                row.ligand_means.to_string(),
                row.ligand_props.to_string(),
                row.receptor.clone(),
                row.receptor_complex.clone(),
                row.receptor_means.to_string(),
                row.receptor_props.to_string(),
                row.source.clone(),
                row.target.clone(),
                row.magnitude.to_string(),
                row.specificity.to_string(),
            ]
        })
        .collect()
}

fn cellchat_fields(rows: &[CellchatRow]) -> Vec<Vec<String>> {
    rows.iter()
        .map(|row| {
            vec![
                row.ligand.clone(),
                row.ligand_complex.clone(),
                row.ligand_props.to_string(),
                row.ligand_trimean.to_string(),
                row.mat_max.to_string(),
                row.receptor.clone(),
                row.receptor_complex.clone(),
                row.receptor_props.to_string(),
                row.receptor_trimean.to_string(),
                row.source.clone(),
                row.target.clone(),
                row.lr_probs.to_string(),
                row.cellchat_pvals.to_string(),
            ]
        })
        .collect()
}

fn cells(rows: Vec<crate::pipe::Row>) -> Vec<Vec<String>> {
    rows.into_iter().map(|row| row.cells).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_name_round_trips_through_parse() {
        for name in METHOD_NAMES {
            let method = Method::parse(name).unwrap();
            assert_eq!(method.name(), *name);
        }
        let error = Method::parse("nope").unwrap_err().to_string();
        assert!(METHOD_NAMES.iter().all(|name| error.contains(name)));
    }

    /// A resource string is a name unless a file of that name exists — so the
    /// same call serves `"consensus"` and a toy CSV path.
    #[test]
    fn resource_strings_resolve_by_path_then_name() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/expected/synthetic__resource.csv"
        );
        assert_eq!(resolve_resource("consensus").unwrap().len(), 4620);
        assert_eq!(resolve_resource(path).unwrap().len(), 110);
        assert!(resolve_resource("nope").is_err());
    }

    /// The defaults are liana's, and the CSV round-trips through the header.
    #[test]
    fn defaults_and_csv_shape() {
        let settings = Settings::default();
        assert_eq!(settings.expr_prop, 0.05);
        assert_eq!(settings.min_cells, 5);
        assert_eq!(settings.n_perms, 1000);
        assert_eq!(settings.seed, 1337);
        let output = Output {
            header: Method::Cellphonedb.csv_header(),
            rows: vec![vec!["a".into(), "b".into()]],
        };
        assert_eq!(output.to_csv().lines().count(), 2);
        assert_eq!(output.to_csv().lines().next().unwrap(), output.header);
    }
}
