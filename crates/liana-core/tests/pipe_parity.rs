//! End-to-end parity of the `pipe` runners against the oracle CSVs
//! (`testdata/expected/synthetic__<method>__p{100,1000}.csv`), value-exactly.
//!
//! The produced CSVs land in `target/pipe_out/` for `scripts/parity_diff.py`
//! to cross-check; the assertions here compare the parsed values by bit, keyed
//! on each row's four key columns, so neither side depends on float formatting.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use liana_core::io::{Adata, read_h5ad};
use liana_core::pipe::{
    CPDB_CSV_HEADER, GMEAN_CSV_HEADER, LrRow, run_cellphonedb, run_geometric_mean,
};
use liana_core::resource::{self, LrPair};
use serde::Deserialize;

#[derive(Deserialize)]
struct Ref {
    adata: String,
    resource: String,
    groupby: String,
    expr_prop: f64,
    min_cells: usize,
    lr_rows: LrRows,
    n_perms: BTreeMap<String, Entry>,
}

#[derive(Deserialize)]
struct LrRows {
    n_rows: usize,
}

#[derive(Deserialize)]
struct Entry {
    seed: u64,
    n_perms: usize,
}

/// The key columns of liana's `_key_cols` — `(source, target, ligand_complex,
/// receptor_complex)`; the oracle CSVs have no duplicate keys.
type Key = (String, String, String, String);

fn key_of(row: &LrRow) -> Key {
    (
        row.source.clone(),
        row.target.clone(),
        row.ligand_complex.clone(),
        row.receptor_complex.clone(),
    )
}

/// The method under test: its runner and its CSV contract.
#[derive(Clone, Copy)]
enum Method {
    Cellphonedb,
    GeometricMean,
}

impl Method {
    fn name(self) -> &'static str {
        match self {
            Method::Cellphonedb => "cellphonedb",
            Method::GeometricMean => "geometric_mean",
        }
    }

    fn header(self) -> &'static str {
        match self {
            Method::Cellphonedb => CPDB_CSV_HEADER,
            Method::GeometricMean => GMEAN_CSV_HEADER,
        }
    }

    fn run(
        self,
        adata: &Adata,
        resource: &[LrPair],
        expr_prop: f64,
        min_cells: usize,
        seed: u64,
        n_perms: usize,
    ) -> anyhow::Result<Vec<LrRow>> {
        match self {
            Method::Cellphonedb => {
                run_cellphonedb(adata, resource, expr_prop, min_cells, seed, n_perms)
            }
            Method::GeometricMean => {
                run_geometric_mean(adata, resource, expr_prop, min_cells, seed, n_perms)
            }
        }
    }
}

/// The CSV's header and rows, as raw fields.
fn read_csv(path: &Path) -> (Vec<String>, Vec<Vec<String>>) {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut lines = text.lines();
    let header: Vec<String> = lines
        .next()
        .expect("header")
        .split(',')
        .map(str::to_owned)
        .collect();
    let rows = lines
        .map(|line| line.split(',').map(str::to_owned).collect())
        .collect();
    (header, rows)
}

fn row_fields(row: &LrRow) -> Vec<String> {
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
}

#[test]
fn cellphonedb_pipeline_matches_the_oracle_csv() {
    run_and_compare(Method::Cellphonedb);
}

#[test]
fn geometric_mean_pipeline_matches_the_oracle_csv() {
    run_and_compare(Method::GeometricMean);
}

fn run_and_compare(method: Method) {
    let name = method.name();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let reference: Ref = serde_json::from_str(
        &fs::read_to_string(root.join("testdata/pipe_ref/synthetic__cellphonedb.json"))
            .expect("dump"),
    )
    .unwrap();
    let adata = read_h5ad(&root.join(&reference.adata), &reference.groupby).unwrap();
    let resource = resource::read_pairs(&root.join(&reference.resource)).unwrap();

    let out_dir = root.join("target/pipe_out");
    fs::create_dir_all(&out_dir).unwrap();

    for (n, entry) in &reference.n_perms {
        let rows = method
            .run(
                &adata,
                &resource,
                reference.expr_prop,
                reference.min_cells,
                entry.seed,
                entry.n_perms,
            )
            .unwrap();

        let mut text = String::from(method.header());
        text.push('\n');
        for row in &rows {
            text.push_str(&row.to_csv_line());
            text.push('\n');
        }
        fs::write(out_dir.join(format!("synthetic__{name}__p{n}.csv")), text).unwrap();

        let expected_path = root.join(format!("testdata/expected/synthetic__{name}__p{n}.csv"));
        compare(&expected_path, method, &rows, reference.lr_rows.n_rows, n);
    }
}

/// Exact keyed comparison of the oracle CSV and the produced rows, reporting
/// the first diverging row/column with both values. Every column of every
/// method is compared by bit.
fn compare(expected_path: &Path, method: Method, rows: &[LrRow], oracle_rows: usize, n: &str) {
    let name = method.name();
    let (header, expected_rows) = read_csv(expected_path);
    assert_eq!(
        header.join(","),
        method.header(),
        "{name} p{n}: the oracle CSV's column contract"
    );
    // an empty oracle CSV must not pass the comparison vacuously
    assert_eq!(
        (expected_rows.len(), rows.len()),
        (oracle_rows, oracle_rows),
        "{name} p{n}: row count"
    );
    let column = |name: &str| header.iter().position(|c| c == name).unwrap();

    // The expected rows, re-keyed; a duplicate key would make lookup ambiguous.
    let mut expected: BTreeMap<Key, &Vec<String>> = BTreeMap::new();
    for expected_row in &expected_rows {
        let key = (
            expected_row[column("source")].clone(),
            expected_row[column("target")].clone(),
            expected_row[column("ligand_complex")].clone(),
            expected_row[column("receptor_complex")].clone(),
        );
        assert!(
            expected.insert(key, expected_row).is_none(),
            "{name} p{n}: duplicate oracle key"
        );
    }
    let actual: BTreeMap<Key, &LrRow> = rows.iter().map(|row| (key_of(row), row)).collect();
    assert_eq!(
        actual.len(),
        rows.len(),
        "{name} p{n}: duplicate produced key"
    );

    let missing: Vec<&Key> = expected
        .keys()
        .filter(|key| !actual.contains_key(key))
        .collect();
    let extra: Vec<&Key> = actual
        .keys()
        .filter(|key| !expected.contains_key(key))
        .collect();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "{name} p{n}: key mismatch — {} oracle rows absent (first: {:?}), {} produced rows \
         unexpected (first: {:?})",
        missing.len(),
        missing.first(),
        extra.len(),
        extra.first(),
    );

    for (key, expected_row) in &expected {
        let actual_fields = row_fields(actual[key]);
        for (index, column_name) in header.iter().enumerate() {
            let expected_field = &expected_row[index];
            let actual_field = &actual_fields[index];
            if !equal_field(column_name, expected_field, actual_field) {
                panic!(
                    "{name} p{n}: first divergence at source={} target={} \
                     ligand_complex={} receptor_complex={} column={column_name}: \
                     expected {expected_field:?} got {actual_field:?}",
                    key.0, key.1, key.2, key.3,
                );
            }
        }
    }
}

/// Whether two fields of one column carry the same value — the float columns
/// by their parsed bits (each side's shortest round-trip parse is exact), the
/// rest as strings.
fn equal_field(column: &str, expected: &str, actual: &str) -> bool {
    match column {
        "ligand_means" | "receptor_means" | "lr_means" | "lr_gmeans" => {
            expected.parse::<f32>().unwrap().to_bits() == actual.parse::<f32>().unwrap().to_bits()
        }
        "ligand_props" | "receptor_props" | "cellphone_pvals" | "gmean_pvals" => {
            expected.parse::<f64>().unwrap().to_bits() == actual.parse::<f64>().unwrap().to_bits()
        }
        _ => expected == actual,
    }
}
