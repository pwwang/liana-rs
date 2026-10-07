//! Bit-exact parity of `math::pairwise`'s `f32` reductions against numpy
//! 2.5.3 — the `np.sum`/`np.std` behind `scseqcomm`'s cluster statistics —
//! over the reference dump `testdata/math_ref/numpy_pairwise_ref.json`
//! (`scripts/dump_pairwise_ref.py`).
//!
//! Every dumped case (`n = 0..=200` across the nine value profiles, plus
//! block-and-recursion sizes up to the fixture's own cluster block) is
//! compared by bit pattern with **zero** tolerance: a mismatch is a porting
//! bug, never a budget to spend. The dump asserts its own model against
//! numpy on every input, so these outputs pin the reduction path the
//! pipeline actually runs.

use std::fs;
use std::path::PathBuf;

use liana_core::math::pairwise::{std_f32, sum_f32};
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct Dump {
    numpy_version: String,
    checks: Checks,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Checks {
    cases: usize,
    values: usize,
    sum_model_vs_numpy_diffs: usize,
    std_model_vs_numpy_diffs: usize,
}

#[derive(Deserialize)]
struct Case {
    n: usize,
    kind: String,
    input: Vector,
    /// The expected `np.sum(..., dtype=f32)`, as a BE u32 bit pattern.
    sum: String,
    /// The expected `np.std` — absent for the empty input.
    std: Option<String>,
}

/// A bit-exact vector: `count` u32 words, big-endian hex blob, and the sha256
/// over that blob (checked on decode, so a doctored dump fails loudly).
#[derive(Deserialize)]
struct Vector {
    count: usize,
    sha256: String,
    hex: String,
}

fn dump() -> Dump {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/math_ref/numpy_pairwise_ref.json");
    serde_json::from_str(&fs::read_to_string(&path).expect("numpy_pairwise_ref.json")).unwrap()
}

fn decode(vector: &Vector) -> Vec<f32> {
    let bytes: Vec<u8> = (0..vector.hex.len() / 2)
        .map(|i| u8::from_str_radix(&vector.hex[2 * i..2 * i + 2], 16).expect("hex"))
        .collect();
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        vector.sha256,
        "vector blob sha256"
    );
    assert_eq!(bytes.len(), vector.count * 4, "vector length");
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_bits(u32::from_be_bytes(*c)))
        .collect()
}

fn scalar(bits: &str) -> f32 {
    let word = u32::from_str_radix(bits.trim_start_matches("0x"), 16).expect("scalar hex");
    f32::from_bits(word)
}

#[test]
fn pairwise_reductions_match_numpy_over_the_cases() {
    let dump = dump();
    assert_eq!(
        dump.checks.sum_model_vs_numpy_diffs, 0,
        "the dump's own sum-kernel identity"
    );
    assert_eq!(
        dump.checks.std_model_vs_numpy_diffs, 0,
        "the dump's own std-kernel identity"
    );
    assert_eq!(dump.cases.len(), dump.checks.cases, "case count contract");
    assert!(dump.cases.len() >= 500, "case count contract");
    let values: usize = dump.cases.iter().map(|case| case.input.count).sum();
    assert_eq!(values, dump.checks.values, "value count contract");
    assert!(values >= 10_000, "value count contract");

    let mut diffs = 0usize;
    let mut first: Option<String> = None;
    for case in &dump.cases {
        let input = decode(&case.input);
        assert_eq!(input.len(), case.n, "case n");
        let expected = scalar(&case.sum);
        let got = sum_f32(&input);
        if got.to_bits() != expected.to_bits() {
            diffs += 1;
            first.get_or_insert(format!(
                "n={} kind={}: sum expected=0x{:08X} got=0x{:08X}",
                case.n,
                case.kind,
                expected.to_bits(),
                got.to_bits(),
            ));
        }
        if let Some(bits) = &case.std {
            let expected = scalar(bits);
            let got = std_f32(&input);
            if got.to_bits() != expected.to_bits() {
                diffs += 1;
                first.get_or_insert(format!(
                    "n={} kind={}: std expected=0x{:08X} got=0x{:08X}",
                    case.n,
                    case.kind,
                    expected.to_bits(),
                    got.to_bits(),
                ));
            }
        } else {
            assert_eq!(case.n, 0, "a non-empty case must carry its std");
        }
    }
    assert_eq!(
        diffs,
        0,
        "pairwise reductions: {diffs} divergences; first: {}",
        first.unwrap(),
    );
    println!(
        "pairwise: {} cases, {} values, 0 bit differences (numpy {})",
        dump.cases.len(),
        values,
        dump.numpy_version
    );
}
