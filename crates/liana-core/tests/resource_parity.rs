//! Parity of `resource::select`/`explode_complexes` against the pinned liana
//! oracle dumps (`scripts/dump_resource_ref.py`).
//!
//! Every stream hash is `"{ligand}\t{receptor}\n"` per pair (four tab-separated
//! fields for exploded subunits), in the order the reference yields them.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use liana_core::resource::{self, LrPair};
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct Resources {
    liana_version: String,
    csv_sha256: String,
    n_csv_rows: usize,
    n_resources: usize,
    resources: BTreeMap<String, Entry>,
}

#[derive(Deserialize)]
struct Entry {
    n_pairs: usize,
    n_unique_ligands: usize,
    n_unique_receptors: usize,
    sha256_pairs_ordered: String,
}

#[derive(Deserialize)]
struct Consensus {
    consensus: ConsensusEntry,
}

#[derive(Deserialize)]
struct ConsensusEntry {
    n_pairs: usize,
    n_ligand_complexes: usize,
    n_receptor_complexes: usize,
    ligand_complexes: Vec<String>,
    receptor_complexes: Vec<String>,
    sha256_pairs_sorted: String,
    n_exploded_pairs: usize,
    sha256_exploded: String,
}

#[derive(Deserialize)]
struct Toy {
    n_pairs: usize,
    n_complex_pairs: usize,
    sha256_pairs_ordered: String,
}

fn refs_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/resource_ref")
}

fn load<T: for<'de> Deserialize<'de>>(name: &str) -> T {
    let path = refs_dir().join(format!("{name}.json"));
    serde_json::from_str(&fs::read_to_string(&path).expect("reference dump")).unwrap()
}

fn sha256_lines(lines: impl IntoIterator<Item = String>) -> String {
    let mut hasher = Sha256::new();
    for line in lines {
        hasher.update(line.as_bytes());
        hasher.update(b"\n");
    }
    format!("{:x}", hasher.finalize())
}

fn pair_stream(pairs: &[LrPair]) -> String {
    sha256_lines(
        pairs
            .iter()
            .map(|p| format!("{}\t{}", p.ligand, p.receptor)),
    )
}

#[test]
fn csv_identity() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/liana-core/data/omni_resource.csv");
    let r: Resources = load("resources");
    assert_eq!(r.liana_version, "2.0.0");
    let sha = format!(
        "{:x}",
        Sha256::digest(fs::read(&path).expect("vendored omni_resource.csv"))
    );
    assert_eq!(
        sha, r.csv_sha256,
        "vendored CSV differs from the pinned oracle copy"
    );
    assert_eq!(
        fs::read_to_string(&path).unwrap().lines().count() - 1,
        r.n_csv_rows
    );
}

#[test]
fn every_resource_matches_the_oracle() {
    let r: Resources = load("resources");
    let names = resource::names().unwrap();
    assert_eq!(names.len(), r.n_resources, "resource count");
    let mut sorted = names.clone();
    sorted.sort();
    let mut expected: Vec<String> = r.resources.keys().cloned().collect();
    expected.sort();
    assert_eq!(sorted, expected, "resource names");

    for (name, entry) in &r.resources {
        let pairs = resource::select(name).unwrap();
        assert_eq!(pairs.len(), entry.n_pairs, "{name}: pairs");
        assert_eq!(
            pair_stream(&pairs),
            entry.sha256_pairs_ordered,
            "{name}: ordered stream"
        );

        let ligands: BTreeMap<&str, ()> = pairs.iter().map(|p| (p.ligand.as_str(), ())).collect();
        let receptors: BTreeMap<&str, ()> =
            pairs.iter().map(|p| (p.receptor.as_str(), ())).collect();
        assert_eq!(
            ligands.len(),
            entry.n_unique_ligands,
            "{name}: unique ligands"
        );
        assert_eq!(
            receptors.len(),
            entry.n_unique_receptors,
            "{name}: unique receptors"
        );
    }
}

#[test]
fn consensus_complexes_and_explosion_match_the_oracle() {
    let r: Consensus = load("consensus");
    let c = r.consensus;
    let pairs = resource::select("consensus").unwrap();
    assert_eq!(pairs.len(), c.n_pairs);

    let mut sorted = pairs.clone();
    sorted.sort_by(|a, b| (&a.ligand, &a.receptor).cmp(&(&b.ligand, &b.receptor)));
    assert_eq!(pair_stream(&sorted), c.sha256_pairs_sorted);

    let complexes = |pick: fn(&LrPair) -> &String| -> Vec<String> {
        let mut symbols: Vec<String> = pairs
            .iter()
            .map(pick)
            .filter(|symbol| symbol.contains('_'))
            .cloned()
            .collect();
        symbols.sort();
        symbols.dedup();
        symbols
    };
    let ligand_complexes = complexes(|p| &p.ligand);
    let receptor_complexes = complexes(|p| &p.receptor);
    assert_eq!(ligand_complexes.len(), c.n_ligand_complexes);
    assert_eq!(receptor_complexes.len(), c.n_receptor_complexes);
    assert_eq!(ligand_complexes, c.ligand_complexes);
    assert_eq!(receptor_complexes, c.receptor_complexes);

    let exploded = resource::explode_complexes(&pairs);
    assert_eq!(exploded.len(), c.n_exploded_pairs, "exploded rows");
    assert_eq!(
        sha256_lines(exploded.iter().map(|s| format!(
            "{}\t{}\t{}\t{}",
            s.ligand, s.receptor, s.ligand_complex, s.receptor_complex
        ))),
        c.sha256_exploded,
        "exploded subunit stream"
    );
}

#[test]
fn toy_resource_parses_in_file_order() {
    let r: Toy = load("toy_synthetic");
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/expected/synthetic__resource.csv");
    let pairs = resource::read_pairs(&path).unwrap();
    assert_eq!(pairs.len(), r.n_pairs);
    assert_eq!(pair_stream(&pairs), r.sha256_pairs_ordered);
    let complex = pairs
        .iter()
        .filter(|p| p.ligand.contains('_') || p.receptor.contains('_'))
        .count();
    assert_eq!(complex, r.n_complex_pairs);
    assert_eq!(
        resource::explode_complexes(&pairs).len(),
        r.n_pairs,
        "no complexes, no expansion"
    );
}
