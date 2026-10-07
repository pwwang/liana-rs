//! `liana_rs._core` — the PyO3 module behind `python/liana_rs/__init__.py`.
//!
//! One function: [`run`], a thin wrapper over `liana_core::run::run_file`
//! returning the result column-typed (numeric columns as `float`s, the rest as
//! `str`s). The DataFrame belongs to the Python shim.

use std::path::Path;

use liana_core::run::{Settings, run_file};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Run one liana method over one `.h5ad`.
///
/// The columnar result: `{column_name: [values]}` in the CSV contract's column
/// order, a column's values all `float` when every cell parses as one, else all
/// `str`. Errors are `ValueError`s, as liana raises them.
#[pyfunction]
#[pyo3(signature = (h5ad, label_key, resource, method, n_perms = 1000, seed = 1337, threads = 0))]
#[allow(clippy::too_many_arguments)] // the eight parameters are the Python signature
fn run<'py>(
    py: Python<'py>,
    h5ad: &str,
    label_key: &str,
    resource: &str,
    method: &str,
    n_perms: usize,
    seed: u64,
    threads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let settings = Settings {
        n_perms,
        seed,
        threads,
        ..Settings::default()
    };
    // The run does not touch Python; let other threads have the interpreter.
    let output = py
        .detach(|| run_file(Path::new(h5ad), label_key, resource, method, &settings))
        .map_err(|error| PyValueError::new_err(format!("{error:#}")))?;

    let dict = PyDict::new(py);
    for (index, name) in output.header.split(',').enumerate() {
        let cells: Vec<&str> = output.rows.iter().map(|row| row[index].as_str()).collect();
        match cells
            .iter()
            .map(|cell| cell.parse::<f64>().ok())
            .collect::<Option<Vec<f64>>>()
        {
            Some(values) => dict.set_item(name, values)?,
            None => dict.set_item(name, cells)?,
        }
    }
    Ok(dict)
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(run, module)?)?;
    Ok(())
}
