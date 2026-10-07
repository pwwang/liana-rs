//! Full-pipeline memory/speed harness for the streaming permutation engine:
//! one liana method over one `.h5ad` + resource, reporting wall time and the
//! process's peak RSS.
//!
//! Peak RSS is `VmHWM` from `/proc/self/status` — the kernel's high-water mark
//! for the process, the same counter `/usr/bin/time -v` prints as "Maximum
//! resident set size" (the driver records both and cross-checks). It covers
//! the `.h5ad` read as well, which is the point: the number is the whole
//! pipeline's, not just the permutation loop's. The wall window is the same
//! one `bench/run_bench.py` times on the Python side (read + run, not process
//! startup), so the two are comparable.
//!
//! Threads are the rayon pool's, sized by `RAYON_NUM_THREADS` at pool init;
//! the effective count is printed as `threads=`.
//!
//! The method dispatch is `liana-rs run`'s own (`run::{Method, Settings}`),
//! so the numbers are the CLI binary's, with the read kept in the timed
//! window and the metadata (`n_obs`, `n_genes`, `n_lrs`) taken from the
//! already-loaded inputs instead of a second pass.
//!
//! Usage: engine-bench --adata <sc_N.h5ad> --resource <resource_N.csv|name>
//!        [--method <any of METHOD_NAMES>] [--n-perms N] [--seed S]

use std::path::PathBuf;
use std::time::Instant;

use liana_core::run::{Method, Settings};

fn main() {
    let mut adata = None;
    let mut resource: Option<String> = None;
    let mut method = String::from("cellphonedb");
    let mut n_perms = 1000usize;
    let mut seed = 1337u64;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| panic!("{arg} needs a value"));
        match arg.as_str() {
            "--adata" => adata = Some(PathBuf::from(value())),
            "--resource" => resource = Some(value()),
            "--method" => method = value(),
            "--n-perms" => n_perms = value().parse().expect("--n-perms"),
            "--seed" => seed = value().parse().expect("--seed"),
            other => panic!("unknown argument {other}"),
        }
    }
    let adata_path = adata.expect("--adata");
    let resource = resource.expect("--resource");

    let method = Method::parse(&method).expect("method");

    let start = Instant::now();
    let adata = liana_core::io::read_h5ad(&adata_path, "cell_type").expect("read h5ad");
    let pairs = liana_core::run::resolve_resource(&resource).expect("resolve resource");
    // `Settings::default()` is liana 2.0.0's `_core/_constants.py` `DefaultValues`
    // (expr_prop 0.05, min_cells 5, seed 1337) — what `bench/run_bench.py` leaves
    // at their defaults on the Python side; `threads = 0` selects the rayon pool
    // `RAYON_NUM_THREADS` sized, the `liana-rs run` CLI default.
    let rows = method
        .run(
            &adata,
            &pairs,
            &Settings {
                seed,
                n_perms,
                ..Settings::default()
            },
        )
        .expect("run")
        .rows
        .len();
    let wall = start.elapsed().as_secs_f64();

    println!(
        "method={} n_obs={} n_genes={} n_lrs={} n_perms={n_perms} seed={seed} \
         threads={} rows={} wall_s={wall:.2} rss_kb={}",
        method.name(),
        adata.x.n_rows,
        adata.x.n_cols,
        pairs.len(),
        rayon::current_num_threads(),
        rows,
        vmhwm_kb(),
    );
}

/// The process's peak resident set size, in kB.
fn vmhwm_kb() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").expect("/proc/self/status");
    status
        .lines()
        .find_map(|line| line.strip_prefix("VmHWM:"))
        .and_then(|value| value.split_whitespace().next())
        .and_then(|value| value.parse().ok())
        .expect("VmHWM")
}
