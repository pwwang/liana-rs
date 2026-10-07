//! End-to-end parity of the `cellchat` runner against the oracle CSVs
//! (`testdata/expected/synthetic__cellchat__p{100,1000}.csv`), value-exactly:
//! every column is bit-compared — `mat_max` as the `f32` both sides print,
//! the rest of the floats as `f64` — keyed on the four key columns, the same
//! contract `scripts/check_pipe_parity.sh` re-checks from outside.
//!
//! The produced CSVs land in `target/pipe_out/` for that cross-check.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use liana_core::io::read_h5ad;
use liana_core::pipe::{CELLCHAT_CSV_HEADER, CellchatRow, run_cellchat};
use liana_core::resource;
use serde::Deserialize;

#[derive(Deserialize)]
struct Ref {
    adata: String,
    resource: String,
    groupby: String,
    expr_prop: f64,
    min_cells: usize,
    /// The frame's `mat_max`, as the f64 the oracle's `f32` widens to.
    mat_max: f64,
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

/// The key columns of liana's `_key_cols`.
type Key = (String, String, String, String);

fn key_of(row: &CellchatRow) -> Key {
    (
        row.source.clone(),
        row.target.clone(),
        row.ligand_complex.clone(),
        row.receptor_complex.clone(),
    )
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

fn row_fields(row: &CellchatRow) -> Vec<String> {
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
}

/// Whether two fields of one column carry the same value — `mat_max` by the
/// `f32` both sides print (`4.1955333` is the `f32`'s shortest form, not the
/// f64's), the other floats by their parsed bits, the rest as strings.
fn equal_field(column: &str, expected: &str, actual: &str) -> bool {
    if column == "mat_max" {
        return expected.parse::<f32>().unwrap().to_bits()
            == actual.parse::<f32>().unwrap().to_bits();
    }
    match column {
        "ligand_props" | "ligand_trimean" | "receptor_props" | "receptor_trimean" | "lr_probs"
        | "cellchat_pvals" => {
            expected.parse::<f64>().unwrap().to_bits() == actual.parse::<f64>().unwrap().to_bits()
        }
        _ => expected == actual,
    }
}

#[test]
fn cellchat_pipeline_matches_the_oracle_csv() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let reference: Ref = serde_json::from_str(
        &fs::read_to_string(root.join("testdata/pipe_ref/synthetic__cellchat.json")).expect("dump"),
    )
    .unwrap();
    let adata = read_h5ad(&root.join(&reference.adata), &reference.groupby).unwrap();
    let pairs = resource::read_pairs(&root.join(&reference.resource)).unwrap();

    let out_dir = root.join("target/pipe_out");
    fs::create_dir_all(&out_dir).unwrap();

    let mut probs: Vec<Vec<f64>> = Vec::new();
    for (n, entry) in &reference.n_perms {
        let rows = run_cellchat(
            &adata,
            &pairs,
            reference.expr_prop,
            reference.min_cells,
            entry.seed,
            entry.n_perms,
        )
        .unwrap();

        // the frame's `mat_max` is the JSON's f32-widened one
        for row in &rows {
            assert_eq!(f64::from(row.mat_max), reference.mat_max, "mat_max");
        }
        probs.push(rows.iter().map(|row| row.lr_probs).collect());

        let mut text = String::from(CELLCHAT_CSV_HEADER);
        text.push('\n');
        for row in &rows {
            text.push_str(&row.to_csv_line());
            text.push('\n');
        }
        fs::write(out_dir.join(format!("synthetic__cellchat__p{n}.csv")), text).unwrap();

        let expected_path = root.join(format!("testdata/expected/synthetic__cellchat__p{n}.csv"));
        compare(&expected_path, &rows, reference.lr_rows.n_rows, n);
    }

    // the observed column does not depend on `n_perms`: the two runs' `lr_probs`
    // are the same vector (the oracle's own dump hashes them identically)
    assert_eq!(probs[0], probs[1], "lr_probs across n_perms");
}

/// Exact keyed comparison of the oracle CSV and the produced rows, reporting
/// the first diverging row/column with both values.
fn compare(expected_path: &Path, rows: &[CellchatRow], oracle_rows: usize, n: &str) {
    let (header, expected_rows) = read_csv(expected_path);
    assert_eq!(
        header.join(","),
        CELLCHAT_CSV_HEADER,
        "cellchat p{n}: the oracle CSV's column contract"
    );
    // an empty oracle CSV must not pass the comparison vacuously
    assert_eq!(
        (expected_rows.len(), rows.len()),
        (oracle_rows, oracle_rows),
        "cellchat p{n}: row count"
    );
    let column = |name: &str| header.iter().position(|c| c == name).unwrap();

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
            "cellchat p{n}: duplicate oracle key"
        );
    }
    let actual: BTreeMap<Key, &CellchatRow> = rows.iter().map(|row| (key_of(row), row)).collect();
    assert_eq!(
        actual.len(),
        rows.len(),
        "cellchat p{n}: duplicate produced key"
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
        "cellchat p{n}: key mismatch — {} oracle rows absent (first: {:?}), {} produced rows \
         unexpected (first: {:?})",
        missing.len(),
        missing.first(),
        extra.len(),
        extra.first(),
    );

    for (key, expected_row) in &expected {
        let actual_fields = row_fields(actual[key]);
        for (index, column_name) in header.iter().enumerate() {
            if !equal_field(column_name, &expected_row[index], &actual_fields[index]) {
                panic!(
                    "cellchat p{n}: first divergence at source={} target={} ligand_complex={} \
                     receptor_complex={} column={column_name}: expected {:?} got {:?}",
                    key.0, key.1, key.2, key.3, expected_row[index], actual_fields[index],
                );
            }
        }
    }
}
