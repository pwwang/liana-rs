//! Run one liana method over the synthetic fixture through `liana_core`'s
//! public API and print the top rows — the library-level equivalent of
//! `liana-rs run`.
//!
//! Usage: `cargo run -p liana-core --example run_synthetic`

use std::path::Path;

use anyhow::Result;
use liana_core::io::read_h5ad;
use liana_core::resource::read_pairs;
use liana_core::run::{Method, Settings};

/// How many result rows to print.
const TOP_ROWS: usize = 5;

fn main() -> Result<()> {
    // `CARGO_MANIFEST_DIR` is `crates/liana-core`, so this works from any cwd.
    let testdata = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata");
    let h5ad = testdata.join("fixtures/synthetic.h5ad");
    let resource = testdata.join("expected/synthetic__resource.csv");

    let adata = read_h5ad(&h5ad, "cell_type")?;
    let pairs = read_pairs(&resource)?;
    let method = Method::parse("cellphonedb")?;
    let output = method.run(&adata, &pairs, &Settings::default())?;

    println!(
        "{}: {} cells x {} features, {} labels; resource of {} pairs",
        method.name(),
        adata.obs_names.len(),
        adata.var_names.len(),
        adata.label_names.len(),
        pairs.len()
    );
    println!(
        "{} rows x {} columns (top {TOP_ROWS})\n",
        output.rows.len(),
        output.header.split(',').count(),
    );

    println!("{}", output.header);
    for row in output.rows.iter().take(TOP_ROWS) {
        println!("{}", row.join(","));
    }
    Ok(())
}
