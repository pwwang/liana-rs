//! Parity of `perms::null` against the pinned liana oracle dump
//! (`scripts/dump_pipe_ref.py`): the permutation cube under `seed=1337`
//! (first three permutations by value, the whole cube by sha256 over its f64
//! bit patterns), the ligand/receptor/combined permutation means of three
//! sample interactions, and their `cellphone_pvals`.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use liana_core::io::read_h5ad;
use liana_core::perms::null::{means_cube, pvals};
use liana_core::prep;
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct Ref {
    groupby: String,
    adata: String,
    min_cells: usize,
    n_perms: BTreeMap<String, Entry>,
}

#[derive(Deserialize)]
struct Entry {
    seed: u64,
    n_perms: usize,
    n_chunks: usize,
    first_perms: Vec<Vec<f64>>,
    sha256_cube_bits: String,
    samples: Vec<Sample>,
}

#[derive(Deserialize)]
struct Sample {
    source: String,
    target: String,
    ligand: String,
    receptor: String,
    lr_means: f64,
    cellphone_pvals: f64,
    ligand_perm_means: Vec<f64>,
    receptor_perm_means: Vec<f64>,
    perm_means: Vec<f64>,
}

/// `_generate_perms_cube`'s `n_chunks` (`_get_mean_perms.py:297`).
fn n_chunks(n_perms: usize, n_obs: usize) -> usize {
    let max_perm_index_elements = 1usize << 24;
    1.max(n_perms.min((n_perms * n_obs).div_ceil(max_perm_index_elements)))
}

#[test]
fn permutation_nulls_match_the_oracle() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let reference: Ref = serde_json::from_str(
        &fs::read_to_string(root.join("testdata/pipe_ref/synthetic__cellphonedb.json"))
            .expect("dump"),
    )
    .unwrap();
    let adata = read_h5ad(&root.join(&reference.adata), &reference.groupby).unwrap();
    let prep = prep::prepare(&adata, reference.min_cells).unwrap();

    let n_labels = prep.n_labels();
    let n_vars = prep.n_vars();
    for (name, entry) in &reference.n_perms {
        let n_perms = entry.n_perms;
        assert_eq!(
            n_chunks(n_perms, prep.x.n_rows),
            entry.n_chunks,
            "{name}: chunking math"
        );

        let perms = liana_core::perms::rng::permutation_matrix(entry.seed, prep.x.n_rows, n_perms);
        let cube = means_cube(&prep, &perms, n_perms);

        // the first three permutations, value by value
        for (p, expected) in entry.first_perms.iter().enumerate() {
            let start = p * n_labels * n_vars;
            assert_eq!(
                &cube[start..start + n_labels * n_vars],
                expected.as_slice(),
                "{name}: perm {p}"
            );
        }

        // the whole cube, to the bit
        let mut hasher = Sha256::new();
        for value in &cube {
            hasher.update(format!("{:016x}\n", value.to_bits()).as_bytes());
        }
        assert_eq!(
            format!("{:x}", hasher.finalize()),
            entry.sha256_cube_bits,
            "{name}: cube stream"
        );

        // the ligand/receptor selection of three sample interactions, and their
        // p-values over the full cube
        for sample in &entry.samples {
            let source = prep.cluster_index(&sample.source).expect("source label");
            let target = prep.cluster_index(&sample.target).expect("target label");
            let ligand = prep.gene_index(&sample.ligand).expect("ligand gene");
            let receptor = prep.gene_index(&sample.receptor).expect("receptor gene");

            let at = |p: usize, cluster: usize, gene: usize| -> f64 {
                cube[(p * n_labels + cluster) * n_vars + gene]
            };
            let ligand_nulls: Vec<f64> = (0..n_perms).map(|p| at(p, source, ligand)).collect();
            let receptor_nulls: Vec<f64> = (0..n_perms).map(|p| at(p, target, receptor)).collect();

            let key = format!(
                "{} -> {} ({}, {})",
                sample.source, sample.target, sample.ligand, sample.receptor
            );
            assert_eq!(
                &ligand_nulls[..3],
                sample.ligand_perm_means.as_slice(),
                "{name} {key}: ligand nulls"
            );
            assert_eq!(
                &receptor_nulls[..3],
                sample.receptor_perm_means.as_slice(),
                "{name} {key}: receptor nulls"
            );
            for p in 0..3 {
                assert_eq!(
                    (ligand_nulls[p] + receptor_nulls[p]) / 2.0,
                    sample.perm_means[p],
                    "{name} {key}: combined null {p}"
                );
            }

            let truth = [sample.lr_means as f32];
            let pvalues = pvals(&ligand_nulls, &receptor_nulls, &truth, n_perms);
            assert_eq!(pvalues[0], sample.cellphone_pvals, "{name} {key}: p-value");
        }
    }
}
