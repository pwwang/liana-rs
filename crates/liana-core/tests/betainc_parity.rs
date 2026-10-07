//! Bit-exact parity of `math::betainc` against scipy 1.18.1's `betainc` — the
//! Boost `ibeta` behind `rank_aggregate`'s robust rank aggregation — over the
//! reference dump `testdata/math_ref/scipy_betainc_ref.json`
//! (`scripts/dump_beta_ref.py`).
//!
//! Four `(a, b, x) -> y` families, compared element-by-element by bit pattern
//! with **zero** tolerance (a mismatch is a porting bug, never a budget to
//! spend):
//!
//! * `rra_p100` / `rra_p1000` — the exact triples the pipeline evaluates for
//!   the two parity fixtures, captured by wrapping `_aggregate.beta`;
//! * `sweep` — every integer pair `1 <= a, b <= 4` plus the `a + b = 8`
//!   margin pairs, each against a boundary-rich `x` pool (the `lambda` switch
//!   at `a / (a + b)` and `x = 0.5` ± 64 ulp, linspace/logspace coverage, the
//!   `x -> 0` / `x -> 1` tails, both endpoints);
//! * `limits` — the wrapper's guard clauses and both `x` endpoints, which
//!   return before `ibeta_imp`.

use std::fs;
use std::path::PathBuf;

use liana_core::math::betainc;
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct Dump {
    scipy_version: String,
    families: Families,
}

#[derive(Deserialize)]
struct Families {
    rra_p100: Family,
    rra_p1000: Family,
    sweep: Family,
    limits: Family,
}

#[derive(Deserialize)]
struct Family {
    a: Vector,
    b: Vector,
    x: Vector,
    y: Vector,
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
        .join("../../testdata/math_ref/scipy_betainc_ref.json");
    serde_json::from_str(&fs::read_to_string(&path).expect("scipy_betainc_ref.json")).unwrap()
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

/// Compare one family; panics with the first divergence (all three arguments,
/// in full hex, plus the outputs) when anything differs.
fn check_family(name: &str, family: &Family) -> usize {
    let (a, b, x, expected) = (
        decode(&family.a),
        decode(&family.b),
        decode(&family.x),
        decode(&family.y),
    );
    assert_eq!(a.len(), x.len(), "{name}: vector lengths");
    assert_eq!(a.len(), expected.len(), "{name}: vector lengths");

    let mut diffs = 0usize;
    let mut first: Option<String> = None;
    for i in 0..a.len() {
        let got = betainc(a[i], b[i], x[i]);
        if got.to_bits() != expected[i].to_bits() {
            diffs += 1;
            if first.is_none() {
                first = Some(format!(
                    "index {i}: a={} b={} x=0x{:016X} ({:e}) expected=0x{:016X} ({:e}) \
                     got=0x{:016X} ({:e})",
                    a[i],
                    b[i],
                    x[i].to_bits(),
                    x[i],
                    expected[i].to_bits(),
                    expected[i],
                    got.to_bits(),
                    got,
                ));
            }
        }
    }
    assert_eq!(
        diffs,
        0,
        "{name}: {diffs}/{} inputs diverge; first divergence: {}",
        a.len(),
        first.unwrap(),
    );
    println!("betainc {name}: {} inputs, 0 bit differences", a.len());
    a.len()
}

#[test]
fn betainc_matches_scipy_over_every_family() {
    let dump = dump();
    let mut total = 0usize;
    for (name, family) in [
        ("rra_p100", &dump.families.rra_p100),
        ("rra_p1000", &dump.families.rra_p1000),
        ("sweep", &dump.families.sweep),
        ("limits", &dump.families.limits),
    ] {
        total += check_family(name, family);
    }
    assert!(total >= 10_000, "sweep size contract, got {total}");
    println!(
        "betainc sweep: {total} triples, 0 bit differences (scipy {})",
        dump.scipy_version
    );
}
