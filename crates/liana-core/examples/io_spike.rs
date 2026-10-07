//! W2-T1 spike: can `anndata` 0.7 + `anndata-hdf5` 0.5 read the h5ad files we need?
//!
//! Usage: `cargo run -p liana-core --example io_spike -- <file.h5ad> [...]`
//!
//! Prints, per file: shape, X kind (dense/sparse) + dtype + nnz, obs categorical
//! label categories, `var_names` head, `obsm` keys + first values. One file's
//! failure does not stop the others — the point is to see exactly what works.

use anndata::data::array::{DynArray, DynCscMatrix, DynCsrMatrix};
use anndata::{AnnData, AnnDataOp, ArrayData, ArrayElemOp, AxisArraysOp, Backend, HasShape};
use anndata_hdf5::H5;
use anyhow::{Context, Result};
use polars::prelude::*;

fn main() -> Result<()> {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    anyhow::ensure!(!paths.is_empty(), "usage: io_spike <file.h5ad> [...]");
    let mut failures = 0;
    for path in &paths {
        match report(path) {
            Ok(()) => {}
            Err(e) => {
                failures += 1;
                println!("== {path}\n  FAILED: {e:#}");
                // The high-level `open` reads every element eagerly. Probe the one we
                // actually need (X) through the low-level container API, to separate
                // "this HDF5 file is unreadable" from "one element's encoding is".
                match probe_x(path) {
                    Ok(msg) => println!("  low-level X probe: {msg}"),
                    Err(e) => println!("  low-level X probe FAILED: {e:#}"),
                }
            }
        }
    }
    println!("\n{} of {} files read", paths.len() - failures, paths.len());
    Ok(())
}

fn report(path: &str) -> Result<()> {
    let ad = AnnData::<H5>::open(H5::open(path)?).with_context(|| format!("open {path}"))?;
    println!("== {path}");
    println!("  shape: {} x {}", ad.n_obs(), ad.n_vars());

    // --- X ---
    let x = ad.get_x();
    println!("  X dtype: {:?}, shape: {:?}", x.dtype(), x.shape());
    let xdata: ArrayData = x.get()?.context("X is empty")?;
    match &xdata {
        ArrayData::Array(a) => {
            let nnz = match a {
                DynArray::F64(m) => m.iter().filter(|v| **v != 0.0).count(),
                DynArray::F32(m) => m.iter().filter(|v| **v != 0.0).count(),
                DynArray::I32(m) => m.iter().filter(|v| **v != 0).count(),
                DynArray::I64(m) => m.iter().filter(|v| **v != 0).count(),
                _ => 0,
            };
            println!("  X kind: dense, nnz={nnz}");
        }
        ArrayData::CsrMatrix(m) => println!("  X kind: CSR, nnz={}", csr_nnz(m)),
        ArrayData::CsrNonCanonical(m) => {
            println!("  X kind: CSR(non-canonical), shape={:?}", m.shape())
        }
        ArrayData::CscMatrix(m) => println!("  X kind: CSC, nnz={}", csc_nnz(m)),
        ArrayData::DataFrame(df) => println!("  X kind: DataFrame, shape={:?}", df.shape()),
    }

    // --- obs ---
    let obs: DataFrame = ad.read_obs()?;
    println!("  obs columns: {:?}", obs.get_column_names());
    for name in obs.get_column_names() {
        let col = obs.column(name)?;
        match col.dtype() {
            DataType::Categorical(_, _) => {
                let n_cats = col.n_unique()?;
                let uniques = col.unique()?;
                println!(
                    "    obs[{name}]: categorical, {n_cats} categories, head={:?}",
                    uniques.head(Some(12))
                );
            }
            dt => println!("    obs[{name}]: {dt:?}, head={:?}", col.head(Some(3))),
        }
    }

    // --- var / names ---
    let var: DataFrame = ad.read_var()?;
    println!("  var columns: {:?}", var.get_column_names());
    let var_names: Vec<String> = ad.var_names().into_iter().take(12).collect();
    println!("  var_names head: {var_names:?}");
    let obs_names: Vec<String> = ad.obs_names().into_iter().take(3).collect();
    println!("  obs_names head: {obs_names:?}");

    // --- obsm ---
    let obsm = ad.obsm();
    println!("  obsm keys: {:?}", obsm.keys());
    for key in obsm.keys() {
        if let Some(elem) = obsm.get(&key) {
            println!(
                "    obsm[{key}]: dtype={:?}, shape={:?}",
                elem.dtype(),
                elem.shape()
            );
            if let Ok(Some(ArrayData::Array(a))) = elem.get::<ArrayData>() {
                let first: Vec<f64> = match a {
                    DynArray::F64(m) => m.iter().take(3).copied().collect(),
                    DynArray::F32(m) => m.iter().take(3).map(|v| *v as f64).collect(),
                    DynArray::I64(m) => m.iter().take(3).map(|v| *v as f64).collect(),
                    DynArray::I32(m) => m.iter().take(3).map(|v| *v as f64).collect(),
                    _ => vec![],
                };
                println!("      first values: {first:?}");
            }
        }
    }

    Ok(())
}

/// Read only `X` through `DataContainer` + `ArrayElem`, skipping `obs`/`var` entirely.
fn probe_x(path: &str) -> Result<String> {
    use anndata::backend::DataContainer;
    use anndata::container::ArrayElem;
    let file = H5::open(path)?;
    let container = DataContainer::open(&file, "X")?;
    let elem = ArrayElem::try_from(container)?;
    let data: ArrayData = elem.get()?.context("X is empty")?;
    Ok(match &data {
        ArrayData::Array(a) => format!("dense, shape={:?}", a.shape()),
        ArrayData::CsrMatrix(m) => format!("CSR, shape={:?}, nnz={}", m.shape(), csr_nnz(m)),
        ArrayData::CsrNonCanonical(m) => format!("CSR(non-canonical), shape={:?}", m.shape()),
        ArrayData::CscMatrix(m) => format!("CSC, shape={:?}, nnz={}", m.shape(), csc_nnz(m)),
        ArrayData::DataFrame(df) => format!("DataFrame, shape={:?}", df.shape()),
    })
}

fn csr_nnz(m: &DynCsrMatrix) -> usize {
    match m {
        DynCsrMatrix::F32(x) => x.nnz(),
        DynCsrMatrix::F64(x) => x.nnz(),
        DynCsrMatrix::I32(x) => x.nnz(),
        DynCsrMatrix::I64(x) => x.nnz(),
        _ => 0,
    }
}

fn csc_nnz(m: &DynCscMatrix) -> usize {
    match m {
        DynCscMatrix::F32(x) => x.nnz(),
        DynCscMatrix::F64(x) => x.nnz(),
        _ => 0,
    }
}
