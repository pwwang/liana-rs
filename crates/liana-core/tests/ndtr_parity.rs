//! Bit-exact parity of `math::ndtr` against scipy 1.18.1's `ndtr` — the
//! standard normal CDF behind `scseqcomm`'s `*_cdf` columns — over the
//! reference dump `testdata/math_ref/scipy_ndtr_ref.json`
//! (`scripts/dump_ndtr_ref.py`).
//!
//! One f64 input family of ≥10k values, compared element-by-element by bit
//! pattern with **zero** tolerance (a mismatch is a porting bug, never a
//! budget to spend): the whole live domain, ±ulp churn around every branch
//! switch (`|x| = 1`, `erfc`'s `|a| = 1` and `|a| = 8`, the `-MAXLOG`
//! underflow guard), subnormals, and NaN/±inf/±0. The dump asserts
//! `norm.cdf == special.ndtr` on every input, so these outputs pin the ufunc
//! the pipeline actually runs.

use std::fs;
use std::path::PathBuf;

use liana_core::math::ndtr;
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct Dump {
    scipy_version: String,
    checks: Checks,
    sweep: Pair,
}

#[derive(Deserialize)]
struct Checks {
    norm_cdf_vs_special_ndtr_diffs: usize,
    nonfinite_outputs: usize,
}

#[derive(Deserialize)]
struct Pair {
    input: Vector,
    output: Vector,
}

/// A bit-exact vector: `count` u64 words, big-endian hex blob, and the sha256
/// over that blob (checked on decode, so a doctored dump fails loudly).
#[derive(Deserialize)]
struct Vector {
    count: usize,
    sha256: String,
    hex: String,
}

fn dump() -> Dump {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/math_ref/scipy_ndtr_ref.json");
    serde_json::from_str(&fs::read_to_string(&path).expect("scipy_ndtr_ref.json")).unwrap()
}

fn decode(vector: &Vector) -> Vec<f64> {
    let bytes: Vec<u8> = (0..vector.hex.len() / 2)
        .map(|i| u8::from_str_radix(&vector.hex[2 * i..2 * i + 2], 16).expect("hex"))
        .collect();
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        vector.sha256,
        "vector blob sha256"
    );
    assert_eq!(bytes.len(), vector.count * 8, "vector length");
    bytes
        .chunks_exact(8)
        .map(|c| f64::from_bits(u64::from_be_bytes(c.try_into().unwrap())))
        .collect()
}

#[test]
fn ndtr_matches_scipy_over_the_sweep() {
    let dump = dump();
    assert!(
        dump.sweep.input.count >= 10_000,
        "sweep size contract, got {}",
        dump.sweep.input.count
    );
    assert_eq!(
        dump.checks.norm_cdf_vs_special_ndtr_diffs, 0,
        "the dump's own norm.cdf/ndtr identity"
    );
    assert!(
        dump.checks.nonfinite_outputs > 0,
        "NaN propagation is covered"
    );

    let inputs = decode(&dump.sweep.input);
    let expected = decode(&dump.sweep.output);
    assert_eq!(inputs.len(), expected.len(), "vector lengths");

    let mut diffs = 0usize;
    let mut first: Option<String> = None;
    for (i, (&x, &e)) in inputs.iter().zip(&expected).enumerate() {
        let got = ndtr(x);
        if got.to_bits() != e.to_bits() {
            diffs += 1;
            if first.is_none() {
                first = Some(format!(
                    "index {i}: in=0x{:016X} ({x:e}) expected=0x{:016X} ({e:e}) \
                     got=0x{:016X} ({got:e})",
                    x.to_bits(),
                    e.to_bits(),
                    got.to_bits(),
                ));
            }
        }
    }
    assert_eq!(
        diffs,
        0,
        "ndtr: {diffs}/{} inputs diverge; first divergence: {}",
        inputs.len(),
        first.unwrap(),
    );
    println!(
        "ndtr sweep: {} inputs, 0 bit differences (scipy {})",
        inputs.len(),
        dump.scipy_version
    );
}
