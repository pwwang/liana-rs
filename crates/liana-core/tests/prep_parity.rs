//! Parity of `prep::prepare` against the pinned liana oracle dump
//! (`scripts/dump_pipe_ref.py`): the prepared var order, the surviving
//! clusters, the per-cluster `means` (f32) and `props` (f64), pinned to the
//! bit by sha256 over their IEEE-754 patterns.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use liana_core::io::read_h5ad;
use liana_core::prep;
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct Ref {
    groupby: String,
    adata: String,
    min_cells: usize,
    prep: PrepRef,
}

#[derive(Deserialize)]
struct PrepRef {
    n_obs: usize,
    n_vars: usize,
    var_names: Vec<String>,
    labels: Vec<String>,
    counts: BTreeMap<String, usize>,
    means: BTreeMap<String, BTreeMap<String, f64>>,
    props: BTreeMap<String, BTreeMap<String, f64>>,
    sha256_means_bits: String,
    sha256_props_bits: String,
}

#[test]
fn prep_matches_the_oracle() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let reference: Ref = serde_json::from_str(
        &fs::read_to_string(root.join("testdata/pipe_ref/synthetic__cellphonedb.json"))
            .expect("dump"),
    )
    .unwrap();
    let adata = read_h5ad(&root.join(&reference.adata), &reference.groupby).unwrap();
    let prep = prep::prepare(&adata, reference.min_cells).unwrap();
    let expected = &reference.prep;

    assert_eq!(prep.x.n_rows, expected.n_obs, "cells");
    assert_eq!(prep.n_vars(), expected.n_vars, "features");
    assert_eq!(prep.var_names, expected.var_names, "prepared var order");
    assert_eq!(prep.labels, expected.labels, "surviving clusters");
    for (cluster, label) in prep.labels.iter().enumerate() {
        assert_eq!(
            prep.counts[cluster], expected.counts[label],
            "{label}: count"
        );
    }

    let mut means_hasher = Sha256::new();
    let mut props_hasher = Sha256::new();
    for (cluster, label) in prep.labels.iter().enumerate() {
        for (gene, name) in prep.var_names.iter().enumerate() {
            let mean = prep.mean(cluster, gene);
            assert_eq!(
                f64::from(mean),
                expected.means[label][name],
                "{label}/{name}: mean"
            );
            means_hasher.update(format!("{:08x}\n", mean.to_bits()).as_bytes());

            let prop = prep.prop(cluster, gene);
            assert_eq!(prop, expected.props[label][name], "{label}/{name}: prop");
            props_hasher.update(format!("{:016x}\n", prop.to_bits()).as_bytes());
        }
    }
    assert_eq!(
        format!("{:x}", means_hasher.finalize()),
        expected.sha256_means_bits,
        "means stream"
    );
    assert_eq!(
        format!("{:x}", props_hasher.finalize()),
        expected.sha256_props_bits,
        "props stream"
    );
}
