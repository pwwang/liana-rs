//! Reading `.h5ad` inputs.
//!
//! [`read_h5ad`] returns X as a `f32` CSR matrix — the layout every downstream
//! step assumes (`prep_check_adata` casts to float32 csr) — plus cell/feature
//! names, a categorical label vector, and `obsm["spatial"]` when present.

mod h5ad;

pub use h5ad::read_h5ad;

/// A `f32` matrix in scipy's `csr_matrix` layout: row `i` is
/// `indices[indptr[i]..indptr[i + 1]]` / `data[indptr[i]..indptr[i + 1]]`.
///
/// Entry order and sparsity are preserved exactly as stored in the file, so a
/// row-order `f32` accumulation over `data` reproduces scipy bit for bit.
#[derive(Debug, Clone, PartialEq)]
pub struct Csr {
    pub n_rows: usize,
    pub n_cols: usize,
    /// `n_rows + 1` offsets, ascending, starting at 0.
    pub indptr: Vec<usize>,
    /// Column index of each stored entry, `< n_cols`.
    pub indices: Vec<u32>,
    /// Value of each stored entry; explicit zeros are never stored.
    pub data: Vec<f32>,
}

/// A `.h5ad` file reduced to what the pipeline consumes.
#[derive(Debug, Clone, PartialEq)]
pub struct Adata {
    pub x: Csr,
    pub obs_names: Vec<String>,
    pub var_names: Vec<String>,
    /// `obs[label_key]` as categorical codes, one per cell, indexing `label_names`.
    pub labels: Vec<u32>,
    pub label_names: Vec<String>,
    /// `obsm["spatial"]`, row-major and widened to `f64`.
    pub obsm_spatial: Option<Spatial>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Spatial {
    /// Columns per cell (2 for the usual `(x, y)`).
    pub n_cols: usize,
    /// `n_obs * n_cols` values, row-major.
    pub data: Vec<f64>,
}
