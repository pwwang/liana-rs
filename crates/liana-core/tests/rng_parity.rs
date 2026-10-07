//! Parity of `perms::rng` against the pinned numpy reference dumps.
//!
//! `testdata/rng_ref/manifest.json` was produced by `scripts/dump_rng_ref.py`
//! running liana 2.0.0's own `_chunk_permutations` under numpy 2.5.3. Each
//! entry's `.npy` payload is compared byte for byte (sha256, then element-wise
//! for the failure message) against the matrix this crate draws.

use std::fs;
use std::path::PathBuf;

use liana_core::perms::rng::permutation_matrix;
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct Manifest {
    numpy_version: String,
    entries: Vec<Entry>,
}

#[derive(Deserialize)]
struct Entry {
    seed: u64,
    n_obs: usize,
    n_perms: usize,
    npy: String,
    npy_sha256: String,
    data_sha256: String,
    dtype: String,
    shape: [usize; 2],
}

fn refs_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/rng_ref")
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// The data payload of a `.npy` file written by `numpy.save`, header validated.
fn npy_payload(bytes: &[u8]) -> &[u8] {
    assert_eq!(&bytes[..6], b"\x93NUMPY", "not a .npy file");
    let (header_len, offset) = match (bytes[6], bytes[7]) {
        (1, _) => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        (2 | 3, _) => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
        (major, minor) => panic!("unsupported .npy version {major}.{minor}"),
    };
    let header =
        std::str::from_utf8(&bytes[offset..offset + header_len]).expect("ASCII .npy header");
    assert!(
        header.contains("'descr': '<u2'"),
        "unexpected descr: {header}"
    );
    assert!(
        header.contains("'fortran_order': False"),
        "unexpected order: {header}"
    );
    &bytes[offset + header_len..]
}

#[test]
fn permutation_stream_matches_numpy_reference() {
    let dir = refs_dir();
    let manifest: Manifest = serde_json::from_str(
        &fs::read_to_string(dir.join("manifest.json")).expect("manifest.json"),
    )
    .expect("parse manifest");
    assert_eq!(
        manifest.entries.len(),
        10,
        "expected the 10 reference configs"
    );
    println!("reference numpy {}", manifest.numpy_version);

    let mut failures = Vec::new();
    for entry in &manifest.entries {
        let label = format!(
            "seed={} n_obs={} n_perms={}",
            entry.seed, entry.n_obs, entry.n_perms
        );
        let npy = fs::read(dir.join(&entry.npy)).expect("read .npy");
        assert_eq!(entry.dtype, "<u2");
        assert_eq!(entry.shape, [entry.n_perms, entry.n_obs]);
        assert_eq!(
            sha256_hex(&npy),
            entry.npy_sha256,
            "{label}: .npy file hash"
        );
        let payload = npy_payload(&npy);
        assert_eq!(
            sha256_hex(payload),
            entry.data_sha256,
            "{label}: .npy payload hash"
        );

        let produced: Vec<u8> = permutation_matrix(entry.seed, entry.n_obs, entry.n_perms)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        if sha256_hex(&produced) != entry.data_sha256 {
            let at = produced
                .chunks_exact(2)
                .zip(payload.chunks_exact(2))
                .position(|(a, b)| a != b)
                .expect("equal length payload");
            failures.push(format!(
                "{label}: first divergence at row {} col {}: rust {} != numpy {}",
                at / entry.n_obs,
                at % entry.n_obs,
                u16::from_le_bytes([produced[2 * at], produced[2 * at + 1]]),
                u16::from_le_bytes([payload[2 * at], payload[2 * at + 1]]),
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "parity failures:\n{}",
        failures.join("\n")
    );
}
