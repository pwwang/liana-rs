//! Parity of `resource::filter_lrs`/`filter_resource` against the pinned liana
//! oracle dump (`scripts/dump_filter_ref.py`).
//!
//! Each case pins the surviving labels, the exploded row count, and a sha256 of
//! the kept `(source, target, ligand_complex, receptor_complex, prop_min)` rows,
//! where `prop_min` is hashed as its float64 bit pattern — the exact value liana
//! compared against `expr_prop`.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use liana_core::io::read_h5ad;
use liana_core::resource::{self, LrPair};
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct Ref {
    groupby: String,
    adata: String,
    toy_resource: String,
    cases: BTreeMap<String, Case>,
}

#[derive(Deserialize)]
struct Case {
    min_cells: usize,
    expr_prop: f64,
    /// The toy resource for every case but `complex`, which lists its own pairs.
    #[serde(default)]
    resource: Option<String>,
    #[serde(default)]
    resource_pairs: Option<Vec<(String, String)>>,
    #[serde(default)]
    raises: Option<String>,
    labels: Vec<String>,
    n_exploded: usize,
    n_lr_rows: usize,
    #[serde(default)]
    n_kept: usize,
    #[serde(default)]
    sha256_kept: String,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn filtering_matches_the_oracle() {
    let root = repo_root();
    let reference: Ref = serde_json::from_str(
        &fs::read_to_string(root.join("testdata/filter_ref/synthetic.json")).expect("dump"),
    )
    .unwrap();
    let adata = read_h5ad(&root.join(&reference.adata), &reference.groupby).unwrap();
    let toy = resource::read_pairs(&root.join(&reference.toy_resource)).unwrap();

    for (name, case) in &reference.cases {
        let pairs: Vec<LrPair> = match &case.resource_pairs {
            Some(pairs) => pairs
                .iter()
                .map(|(ligand, receptor)| LrPair {
                    ligand: ligand.clone(),
                    receptor: receptor.clone(),
                })
                .collect(),
            None => {
                assert_eq!(case.resource.as_deref(), Some("toy"));
                toy.clone()
            }
        };

        let exploded =
            resource::filter_resource(&resource::explode_complexes(&pairs), &adata.var_names);
        assert_eq!(exploded.len(), case.n_exploded, "{name}: exploded rows");

        let result = resource::filter_lrs(&adata, &pairs, case.expr_prop, case.min_cells);
        if let Some(raises) = &case.raises {
            let error = result.expect_err("the oracle raises for this case");
            assert_eq!(raises, "ValueError", "{name}");
            assert!(
                error.to_string().contains("no ligand-receptor pair passed"),
                "{name}: {error}"
            );
            continue;
        }

        let result = result.unwrap();
        assert_eq!(result.labels, case.labels, "{name}: labels");
        assert_eq!(result.kept.len(), case.n_kept, "{name}: kept pairs");
        assert_eq!(
            exploded.len() * result.labels.len().pow(2),
            case.n_lr_rows,
            "{name}: candidate rows"
        );

        let mut hasher = Sha256::new();
        for pair in &result.kept {
            hasher.update(
                format!(
                    "{}\t{}\t{}\t{}\t{:016x}\n",
                    result.labels[pair.source as usize],
                    result.labels[pair.target as usize],
                    pair.ligand_complex,
                    pair.receptor_complex,
                    pair.prop_min.to_bits(),
                )
                .as_bytes(),
            );
        }
        assert_eq!(
            format!("{:x}", hasher.finalize()),
            case.sha256_kept,
            "{name}: kept stream"
        );
    }
}
