//! Parity of `io::read_h5ad` against the oracle dumps from `scripts/dump_io_ref.py`.
//!
//! `testdata/io_ref/{synthetic,sc_10000}.json` was produced by liana 2.0.0's own
//! anndata/scipy under the pinned venv. Every hash is reproduced by writing the
//! same byte stream the dumper documents in `hash_format`; the per-var sums are
//! accumulated in `f32` in CSR order, which is bit-identical to scipy's
//! `csr_matrix.sum(axis=0)` for a float32 matrix.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use liana_core::io::{Adata, read_h5ad};
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Deserialize)]
struct Ref {
    file: String,
    sha256_file: String,
    n_obs: usize,
    n_vars: usize,
    x_f32_nnz: usize,
    /// A plain left-to-right f32 accumulation over the values, in stored order.
    x_f32_sum_sequential: f64,
    /// scipy's own `X.sum()` (numpy pairwise summation) — reference, not reproduced.
    x_f32_sum_scipy: f64,
    var_sums_sha256: String,
    label_key: String,
    label_names: Vec<String>,
    label_counts: BTreeMap<String, u64>,
    labels_sha256: String,
    var_names_sha256: String,
    obs_names_sha256: String,
    obsm_spatial: Option<SpatialRef>,
}

#[derive(Debug, Deserialize)]
struct SpatialRef {
    n_cols: usize,
    first: Vec<i64>,
}

fn refs_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/io_ref")
}

fn load(name: &str) -> Ref {
    let path = refs_dir().join(format!("{name}.json"));
    serde_json::from_str(&fs::read_to_string(&path).expect("reference dump")).unwrap()
}

fn sha256_text(lines: impl IntoIterator<Item = String>) -> String {
    let mut hasher = Sha256::new();
    for line in lines {
        hasher.update(line.as_bytes());
        hasher.update(b"\n");
    }
    format!("{:x}", hasher.finalize())
}

/// Per-var sums accumulated in `f32` in CSR order — scipy's `sum(axis=0)`.
fn var_sums(x: &liana_core::io::Csr) -> Vec<f32> {
    let mut sums = vec![0.0f32; x.n_cols];
    for row in 0..x.n_rows {
        for i in x.indptr[row]..x.indptr[row + 1] {
            sums[x.indices[i] as usize] += x.data[i];
        }
    }
    sums
}

fn check(name: &str, path: &Path, r: &Ref, a: &Adata) {
    assert_eq!(a.x.n_rows, r.n_obs, "{name}: n_obs");
    assert_eq!(a.x.n_cols, r.n_vars, "{name}: n_vars");
    assert_eq!(a.x.data.len(), r.x_f32_nnz, "{name}: f32 nnz");
    assert_eq!(a.obs_names.len(), r.n_obs, "{name}: obs_names");
    assert_eq!(a.var_names.len(), r.n_vars, "{name}: var_names");
    assert_eq!(a.labels.len(), r.n_obs, "{name}: labels");

    assert_eq!(
        sha256_text(a.obs_names.iter().cloned()),
        r.obs_names_sha256,
        "{name}: obs_names"
    );
    assert_eq!(
        sha256_text(a.var_names.iter().cloned()),
        r.var_names_sha256,
        "{name}: var_names"
    );
    assert_eq!(
        sha256_text(var_sums(&a.x).iter().map(|s| format!("{s:.6}"))),
        r.var_sums_sha256,
        "{name}: per-var f32 sums"
    );
    assert_eq!(
        sha256_text(a.labels.iter().map(|&c| a.label_names[c as usize].clone())),
        r.labels_sha256,
        "{name}: labels"
    );

    // A naive left-to-right f32 sum over the stored values: exact equality, so it
    // pins the values *and* their order (scipy's pairwise `X.sum()` is recorded in
    // the dump only as a documented reference; the per-var sums above are the
    // bit-exact claim).
    let sequential = a.x.data.iter().copied().sum::<f32>();
    assert_eq!(
        sequential, r.x_f32_sum_sequential as f32,
        "{name}: sequential f32 sum of X"
    );
    let scipy = r.x_f32_sum_scipy as f32;
    assert!(
        (sequential - scipy).abs() <= 0.01 * scipy.abs().max(1.0),
        "{name}: sequential sum {sequential} differs from scipy's {scipy} by more than rounding"
    );

    assert_eq!(a.label_names, r.label_names, "{name}: label names");
    let counts: BTreeMap<String, u64> = a.labels.iter().fold(BTreeMap::new(), |mut acc, &c| {
        *acc.entry(a.label_names[c as usize].clone()).or_insert(0) += 1;
        acc
    });
    assert_eq!(counts, r.label_counts, "{name}: label counts");

    match (&a.obsm_spatial, &r.obsm_spatial) {
        (None, None) => {}
        (Some(s), Some(want)) => {
            assert_eq!(s.n_cols, want.n_cols, "{name}: spatial n_cols");
            let first: Vec<i64> = s.data[..s.n_cols].iter().map(|&v| v as i64).collect();
            assert_eq!(first, want.first, "{name}: spatial first cell");
        }
        (got, want) => panic!("{name}: spatial presence mismatch: {got:?} vs {want:?}"),
    }

    let file_sha = format!("{:x}", Sha256::digest(fs::read(path).expect("read h5ad")));
    assert_eq!(file_sha, r.sha256_file, "{name}: file sha256");
}

#[test]
fn synthetic_parity() {
    let r = load("synthetic");
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../")
        .join(&r.file);
    let adata = read_h5ad(&path, &r.label_key).expect("read synthetic.h5ad");
    check("synthetic", &path, &r, &adata);
}

/// `sc_10000.h5ad` lives in the local `p0a` scratch data, not in the repo; the
/// dump's file sha256 still pins it when present.
#[test]
fn sc_10000_parity() {
    let r = load("sc_10000");
    let path = PathBuf::from(&r.file);
    if !path.exists() {
        eprintln!("skipping: {} is not present", path.display());
        return;
    }
    let adata = read_h5ad(&path, &r.label_key).expect("read sc_10000.h5ad");
    check("sc_10000", &path, &r, &adata);
}
