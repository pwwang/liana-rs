//! Bit-exact parity of `math::{logf, expf}` against numpy 2.5.3's `f32`
//! kernels, over the reference dump `testdata/math_ref/numpy_math_ref.json`
//! (`scripts/dump_math_ref.py`).
//!
//! Two input families per kernel, both compared element-by-element by bit
//! pattern with **zero** tolerance (a mismatch is a porting bug, never a
//! budget to spend): the exact 880 log + 440 exp inputs of the
//! `geometric_mean` column path, and a ≥10k mixed/subnormal/boundary sweep.
//! The dump carries the numpy outputs; the kernel identity (which numpy
//! implementation produced them) is argued in `scripts/dump_math_ref.py` and
//! `math/mod.rs`, and *proven* here by the match.

use std::fs;
use std::path::PathBuf;

use liana_core::math::{expf, logf};
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct Dump {
    numpy_version: String,
    checks: Checks,
    log: Families,
    exp: Families,
    gmean_pipeline: GmeanPipeline,
}

#[derive(Deserialize)]
struct Checks {
    pipeline_rows: usize,
    log_vs_rounded_f64_diffs: usize,
    exp_vs_rounded_f64_diffs: usize,
}

#[derive(Deserialize)]
struct Families {
    pipeline: Pair,
    sweep: Pair,
}

#[derive(Deserialize)]
struct Pair {
    input: Vector,
    output: Vector,
}

/// A bit-exact vector: `count` u32 words, big-endian hex blob, and the sha256
/// over that blob (checked on decode, so a doctored dump fails loudly).
#[derive(Deserialize)]
struct Vector {
    count: usize,
    sha256: String,
    hex: String,
}

#[derive(Deserialize)]
struct GmeanPipeline {
    ligand_means: Vector,
    receptor_means: Vector,
    oracle_lr_gmeans: Vector,
}

fn dump() -> Dump {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/math_ref/numpy_math_ref.json");
    serde_json::from_str(&fs::read_to_string(&path).expect("numpy_math_ref.json")).unwrap()
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
        .chunks_exact(4)
        .map(|c| f32::from_bits(u32::from_be_bytes(c.try_into().unwrap())))
        .collect()
}

/// Element-wise bit comparison; the panic names the first divergence with the
/// input's own bits, so a failure is directly reproducible.
fn assert_bit_exact(name: &str, input: &Vector, expected: &Vector, kernel: fn(f32) -> f32) {
    let inputs = decode(input);
    let expected = decode(expected);
    assert_eq!(inputs.len(), expected.len(), "{name}: vector lengths");

    let mut diffs = 0usize;
    let mut first: Option<String> = None;
    for (i, (&x, &e)) in inputs.iter().zip(&expected).enumerate() {
        let got = kernel(x);
        if got.to_bits() != e.to_bits() {
            diffs += 1;
            if first.is_none() {
                first = Some(format!(
                    "index {i}: in=0x{:08X} ({x:e}) expected=0x{:08X} ({e:e}) got=0x{:08X} ({got:e})",
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
        "{name}: {diffs}/{} inputs diverge; first divergence: {}",
        inputs.len(),
        first.unwrap(),
    );
    println!("{name}: {} inputs, 0 bit differences", inputs.len());
}

#[test]
fn log_kernel_matches_numpy_on_the_gmean_path() {
    let dump = dump();
    assert_eq!(
        dump.checks.pipeline_rows, 440,
        "the oracle fixture's row count"
    );
    assert_eq!(dump.log.pipeline.input.count, 880, "log inputs: 2 per row");
    assert_bit_exact(
        "log pipeline",
        &dump.log.pipeline.input,
        &dump.log.pipeline.output,
        logf,
    );
}

#[test]
fn log_kernel_matches_numpy_sweep() {
    let dump = dump();
    assert!(dump.log.sweep.input.count >= 10_000, "sweep size contract");
    assert_bit_exact(
        "log sweep",
        &dump.log.sweep.input,
        &dump.log.sweep.output,
        logf,
    );
}

#[test]
fn exp_kernel_matches_numpy_on_the_gmean_path() {
    let dump = dump();
    assert_eq!(dump.exp.pipeline.input.count, 440, "exp inputs: 1 per row");
    assert_bit_exact(
        "exp pipeline",
        &dump.exp.pipeline.input,
        &dump.exp.pipeline.output,
        expf,
    );
}

#[test]
fn exp_kernel_matches_numpy_sweep() {
    let dump = dump();
    assert!(dump.exp.sweep.input.count >= 10_000, "sweep size contract");
    assert_bit_exact(
        "exp sweep",
        &dump.exp.sweep.input,
        &dump.exp.sweep.output,
        expf,
    );
}

/// The whole `lr_gmeans` column path in `f32`, kernel by kernel: every dumped
/// exp input is exactly the two-element mean of the dumped log outputs
/// (numpy's `np.mean` over the stacked `(2, n)` array), and
/// `exp(log(l) + log(r) / 2)` over the ligand/receptor means reproduces the
/// oracle column bit for bit — which is the point of the port (W3 D2).
#[test]
fn gmean_chain_reproduces_the_oracle_column() {
    let dump = dump();
    let g = &dump.gmean_pipeline;
    let ligand = decode(&g.ligand_means);
    let receptor = decode(&g.receptor_means);
    let oracle = decode(&g.oracle_lr_gmeans);
    let n = oracle.len();
    assert_eq!(ligand.len(), n, "ligand column");
    assert_eq!(receptor.len(), n, "receptor column");

    // the pipeline log inputs are the two columns, ligand first
    let log_input = decode(&dump.log.pipeline.input);
    assert!(
        log_input
            .iter()
            .zip(ligand.iter().chain(&receptor))
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "log pipeline inputs are the stacked ligand/receptor columns"
    );

    // exp inputs are the f32 means of the log outputs
    let exp_input = decode(&dump.exp.pipeline.input);
    let mut diffs = 0usize;
    for i in 0..n {
        let mean = (logf(ligand[i]) + logf(receptor[i])) / 2.0;
        if mean.to_bits() != exp_input[i].to_bits() {
            diffs += 1;
        }
    }
    assert_eq!(diffs, 0, "mean(log) matches the dumped exp inputs");

    // and the chain is the oracle column, bit for bit
    let mut bad: Vec<(usize, u32, u32)> = Vec::new();
    for i in 0..n {
        let got = expf((logf(ligand[i]) + logf(receptor[i])) / 2.0);
        if got.to_bits() != oracle[i].to_bits() {
            bad.push((i, got.to_bits(), oracle[i].to_bits()));
        }
    }
    assert!(
        bad.is_empty(),
        "lr_gmeans diverges on {} of {n} rows (first: {:?})",
        bad.len(),
        bad.first(),
    );

    // the dump itself records that this column is *not* a rounded f64 route
    assert!(dump.checks.log_vs_rounded_f64_diffs > 0 && dump.checks.exp_vs_rounded_f64_diffs > 0);
    println!(
        "gmean chain: {n} rows bit-exact against the oracle column (numpy {})",
        dump.numpy_version
    );
}
