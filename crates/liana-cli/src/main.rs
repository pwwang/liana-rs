//! `liana-rs` — run one liana single-cell method over one `.h5ad` and write
//! the result CSV (`liana-rs run --help`).

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use liana_core::io::read_h5ad;
use liana_core::resource;
use liana_core::run::{Method, Settings};

/// liana's single-cell ligand–receptor methods, in Rust.
#[derive(Parser)]
#[command(name = "liana-rs", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run one method and write the result CSV.
    Run(Run),
}

#[derive(Args)]
struct Run {
    /// Input `.h5ad` file.
    #[arg(long)]
    h5ad: PathBuf,
    /// `obs` column holding the cluster labels.
    #[arg(long)]
    label_key: String,
    /// Resource name in the vendored omni resource (liana's `resource_name`).
    #[arg(long, default_value = "consensus", conflicts_with = "resource_file")]
    resource: String,
    /// A `ligand,receptor` CSV to score against instead of `--resource`.
    #[arg(long)]
    resource_file: Option<PathBuf>,
    /// Method to run: cellphonedb, geometric_mean, cellchat, connectome, logfc,
    /// natmi, scseqcomm, singlecellsignalr, rank_aggregate.
    #[arg(long, value_parser = Method::parse)]
    method: Method,
    /// Permutations (permutation-scored methods).
    #[arg(long, default_value_t = 1000)]
    n_perms: usize,
    /// RNG seed.
    #[arg(long, default_value_t = 1337)]
    seed: u64,
    /// Worker threads; 0 = one per core.
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// liana's `expr_prop`.
    #[arg(long, default_value_t = 0.05)]
    expr_prop: f64,
    /// liana's `min_cells`.
    #[arg(long, default_value_t = 5)]
    min_cells: usize,
    /// Output CSV path.
    #[arg(long)]
    out: PathBuf,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Run(args) => run(args),
    }
}

fn run(args: Run) -> Result<()> {
    let adata = read_h5ad(&args.h5ad, &args.label_key)
        .with_context(|| format!("read {}", args.h5ad.display()))?;
    let pairs = match &args.resource_file {
        Some(path) => resource::read_pairs(path)?,
        None => resource::select(&args.resource)?,
    };
    let output = args.method.run(
        &adata,
        &pairs,
        &Settings {
            expr_prop: args.expr_prop,
            min_cells: args.min_cells,
            n_perms: args.n_perms,
            seed: args.seed,
            threads: args.threads,
        },
    )?;
    std::fs::write(&args.out, output.to_csv())
        .with_context(|| format!("write {}", args.out.display()))?;
    eprintln!(
        "liana-rs: {} rows x {} columns -> {}",
        output.rows.len(),
        output.header.split(',').count(),
        args.out.display()
    );
    Ok(())
}
