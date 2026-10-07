//! `.h5ad` reader on `hdf5-metno` (see `docs/io-spike.md` for why not anndata-rs).
//!
//! Only the encodings the pipeline needs are read: `X` as a dense `array`
//! dataset or a `csr_matrix` group, `obs`/`var` indices as `string-array`
//! datasets or `nullable-string-array` groups, categorical label columns
//! (`categories` + `codes`), and a numeric `obsm["spatial"]`. Anything else is
//! a hard error rather than a silently wrong value.

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use hdf5_metno::types::{FloatSize, IntSize, TypeDescriptor, VarLenUnicode};
use hdf5_metno::{Dataset, File, Group, Location, LocationType};

use super::{Adata, Csr, Spatial};

/// Read `path`, taking `obs[label_key]` (a categorical column) as the cell labels.
pub fn read_h5ad(path: &Path, label_key: &str) -> Result<Adata> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let x = read_x(&file).context("X")?;
    let obs_names = read_names(&file, "obs", x.n_rows).context("obs names")?;
    let var_names = read_names(&file, "var", x.n_cols).context("var names")?;
    let (label_names, labels) = read_labels(&file, label_key, x.n_rows)?;
    let obsm_spatial = read_spatial(&file, x.n_rows)?;
    Ok(Adata {
        x,
        obs_names,
        var_names,
        labels,
        label_names,
        obsm_spatial,
    })
}

fn read_x(file: &File) -> Result<Csr> {
    match file.loc_type_by_name("X")? {
        LocationType::Dataset => dense_to_csr(&file.dataset("X")?),
        LocationType::Group => {
            let group = file.group("X")?;
            let encoding = encoding(&group)?;
            ensure!(
                encoding == "csr_matrix",
                "unsupported X encoding-type {encoding:?}: only a dense array or `csr_matrix` is read"
            );
            group_to_csr(&group)
        }
        other => bail!("X is a {other:?}, expected a dense dataset or a csr_matrix group"),
    }
}

/// Dense `X` → CSR keeping only nonzeros, i.e. what `scipy.sparse.csr_matrix(X)` does.
fn dense_to_csr(dset: &Dataset) -> Result<Csr> {
    let (n_rows, n_cols) = matrix_shape(&dset.shape(), "X")?;
    ensure!(
        n_cols <= u32::MAX as usize,
        "X has {n_cols} columns, more than u32 indices address"
    );
    let values = read_f32(dset).context("X")?;
    ensure!(
        values.len() == n_rows * n_cols,
        "X holds {} values for {n_rows}x{n_cols}",
        values.len()
    );
    let mut indptr = Vec::with_capacity(n_rows + 1);
    let mut indices = Vec::new();
    let mut data = Vec::new();
    indptr.push(0);
    for row in 0..n_rows {
        for col in 0..n_cols {
            let value = values[row * n_cols + col];
            if value != 0.0 {
                indices.push(col as u32);
                data.push(value);
            }
        }
        indptr.push(data.len());
    }
    Ok(Csr {
        n_rows,
        n_cols,
        indptr,
        indices,
        data,
    })
}

fn group_to_csr(group: &Group) -> Result<Csr> {
    let shape: Vec<i64> = group
        .attr("shape")
        .context("attribute shape")?
        .read_raw()
        .context("read X/shape")?;
    let (n_rows, n_cols) = shape_pair(&shape, "X/shape")?;
    let data = read_f32(&group.dataset("data")?).context("X/data")?;
    let indices = read_indices::<u32>(&group.dataset("indices")?, "X/indices")?;
    let indptr = read_indices::<usize>(&group.dataset("indptr")?, "X/indptr")?;
    let csr = Csr {
        n_rows,
        n_cols,
        indptr,
        indices,
        data,
    };
    validate(&csr)?;
    Ok(csr)
}

/// The CSR invariants downstream code relies on (`indptr[i]` slicing rows,
/// indices in range) — checked once, at the boundary.
fn validate(csr: &Csr) -> Result<()> {
    ensure!(
        csr.indptr.len() == csr.n_rows + 1,
        "X/indptr has {} offsets, expected {}",
        csr.indptr.len(),
        csr.n_rows + 1
    );
    ensure!(
        csr.indices.len() == csr.data.len(),
        "X/indices and X/data differ in length ({} vs {})",
        csr.indices.len(),
        csr.data.len()
    );
    ensure!(
        csr.indptr.first() == Some(&0) && csr.indptr.last() == Some(&csr.data.len()),
        "X/indptr does not start at 0 and end at nnz"
    );
    ensure!(
        csr.indptr.windows(2).all(|w| w[0] <= w[1]),
        "X/indptr is not ascending"
    );
    ensure!(
        csr.indices.iter().all(|&c| (c as usize) < csr.n_cols),
        "X/indices holds a column index out of range"
    );
    Ok(())
}

fn read_names(file: &File, elem: &str, n: usize) -> Result<Vec<String>> {
    let group = file.group(elem)?;
    let index = attr_str(&group, "_index")?;
    let names = match group.loc_type_by_name(&index)? {
        LocationType::Dataset => read_strings(&group.dataset(&index)?)?,
        LocationType::Group => read_nullable_strings(&group.group(&index)?)?,
        other => bail!("{elem}/{index} is a {other:?}, expected a dataset or a group"),
    };
    ensure!(
        names.len() == n,
        "{elem}/{index} holds {} names, expected {n}",
        names.len()
    );
    Ok(names)
}

/// `nullable-string-array`: a `values` dataset plus a `mask` that is true where
/// the value is missing. Missing index values cannot be represented, so they
/// are an error.
fn read_nullable_strings(group: &Group) -> Result<Vec<String>> {
    let encoding = encoding(group)?;
    ensure!(
        encoding == "nullable-string-array",
        "unsupported index encoding {encoding:?}"
    );
    let mask: Vec<u8> = group.dataset("mask")?.read_raw().context("read mask")?;
    ensure!(
        mask.iter().all(|&m| m == 0),
        "index contains missing values"
    );
    read_strings(&group.dataset("values")?)
}

fn read_labels(file: &File, key: &str, n: usize) -> Result<(Vec<String>, Vec<u32>)> {
    let group = file
        .group("obs")?
        .group(key)
        .with_context(|| format!("obs/{key}"))?;
    let encoding = encoding(&group)?;
    ensure!(
        encoding == "categorical",
        "obs[{key}] has encoding-type {encoding:?}, expected \"categorical\""
    );
    let names = read_strings(&group.dataset("categories")?).context("read categories")?;
    let codes: Vec<i64> = group.dataset("codes")?.read_raw().context("read codes")?;
    ensure!(
        codes.len() == n,
        "obs[{key}] holds {} codes, expected {n}",
        codes.len()
    );
    let labels = codes
        .into_iter()
        .map(|code| u32::try_from(code).with_context(|| format!("obs[{key}] has a null code")))
        .collect::<Result<Vec<u32>>>()?;
    ensure!(
        labels.iter().all(|&c| (c as usize) < names.len()),
        "obs[{key}] holds a code outside the categories"
    );
    Ok((names, labels))
}

fn read_spatial(file: &File, n_obs: usize) -> Result<Option<Spatial>> {
    if !file.link_exists("obsm/spatial") {
        return Ok(None);
    }
    let dset = file.dataset("obsm/spatial")?;
    let (n_rows, n_cols) = matrix_shape(&dset.shape(), "obsm[\"spatial\"]")?;
    ensure!(
        n_rows == n_obs,
        "obsm[\"spatial\"] has {n_rows} rows, expected {n_obs}"
    );
    let data = read_f64(&dset)?;
    ensure!(
        data.len() == n_rows * n_cols,
        "obsm[\"spatial\"] holds {} values for {n_rows}x{n_cols}",
        data.len()
    );
    Ok(Some(Spatial { n_cols, data }))
}

/// Read a numeric dataset as `f32` the way numpy's `astype(np.float32)` does:
/// round the source value once. `f32` sources are not converted at all.
fn read_f32(dset: &Dataset) -> Result<Vec<f32>> {
    macro_rules! widen {
        ($t:ty) => {
            dset.read_raw::<$t>()?
                .into_iter()
                .map(|v| v as f32)
                .collect()
        };
    }
    let dtype = dset.dtype()?.to_descriptor()?;
    Ok(match dtype {
        TypeDescriptor::Float(FloatSize::U4) => dset.read_raw()?,
        TypeDescriptor::Float(FloatSize::U8) => widen!(f64),
        TypeDescriptor::Integer(IntSize::U1) => widen!(i8),
        TypeDescriptor::Integer(IntSize::U2) => widen!(i16),
        TypeDescriptor::Integer(IntSize::U4) => widen!(i32),
        TypeDescriptor::Integer(IntSize::U8) => widen!(i64),
        TypeDescriptor::Unsigned(IntSize::U1) => widen!(u8),
        TypeDescriptor::Unsigned(IntSize::U2) => widen!(u16),
        TypeDescriptor::Unsigned(IntSize::U4) => widen!(u32),
        TypeDescriptor::Unsigned(IntSize::U8) => widen!(u64),
        other => bail!("unsupported data dtype {other}"),
    })
}

fn read_f64(dset: &Dataset) -> Result<Vec<f64>> {
    macro_rules! widen {
        ($t:ty) => {
            dset.read_raw::<$t>()?
                .into_iter()
                .map(|v| v as f64)
                .collect()
        };
    }
    let dtype = dset.dtype()?.to_descriptor()?;
    Ok(match dtype {
        TypeDescriptor::Float(FloatSize::U8) => dset.read_raw()?,
        TypeDescriptor::Float(FloatSize::U4) => widen!(f32),
        TypeDescriptor::Integer(IntSize::U1) => widen!(i8),
        TypeDescriptor::Integer(IntSize::U2) => widen!(i16),
        TypeDescriptor::Integer(IntSize::U4) => widen!(i32),
        TypeDescriptor::Integer(IntSize::U8) => widen!(i64),
        TypeDescriptor::Unsigned(IntSize::U1) => widen!(u8),
        TypeDescriptor::Unsigned(IntSize::U2) => widen!(u16),
        TypeDescriptor::Unsigned(IntSize::U4) => widen!(u32),
        TypeDescriptor::Unsigned(IntSize::U8) => widen!(u64),
        other => bail!("unsupported data dtype {other}"),
    })
}

fn read_indices<T>(dset: &Dataset, what: &str) -> Result<Vec<T>>
where
    T: TryFrom<i64>,
    T::Error: std::error::Error + Send + Sync + 'static,
{
    let dtype = dset.dtype()?.to_descriptor()?;
    let raw: Vec<i64> = match dtype {
        TypeDescriptor::Integer(_) => dset.read_raw()?,
        TypeDescriptor::Unsigned(_) => dset
            .read_raw::<u64>()?
            .into_iter()
            .map(|v| v as i64)
            .collect(),
        other => bail!("{what} has dtype {other}, expected an integer"),
    };
    raw.into_iter()
        .map(|v| T::try_from(v).with_context(|| format!("{what} holds {v}, not a valid index")))
        .collect()
}

fn read_strings(dset: &Dataset) -> Result<Vec<String>> {
    match dset.dtype()?.to_descriptor()? {
        TypeDescriptor::VarLenAscii | TypeDescriptor::VarLenUnicode => Ok(dset
            .read_raw::<VarLenUnicode>()?
            .into_iter()
            .map(|s| s.as_str().to_owned())
            .collect()),
        other => bail!("expected variable-length strings, found {other}"),
    }
}

fn attr_str(loc: &Location, name: &str) -> Result<String> {
    let attr = loc
        .attr(name)
        .with_context(|| format!("attribute {name:?}"))?;
    Ok(attr
        .read_scalar::<VarLenUnicode>()
        .with_context(|| format!("read attribute {name:?}"))?
        .as_str()
        .to_owned())
}

fn encoding(loc: &Location) -> Result<String> {
    attr_str(loc, "encoding-type")
}

/// A 2-D dataspace shape as `(rows, cols)`.
fn matrix_shape(shape: &[usize], what: &str) -> Result<(usize, usize)> {
    match shape {
        [rows, cols] => Ok((*rows, *cols)),
        _ => bail!("{what} is {}-dimensional, expected 2", shape.len()),
    }
}

/// The `shape` attribute of a sparse `X` group as `(rows, cols)`.
fn shape_pair(shape: &[i64], what: &str) -> Result<(usize, usize)> {
    let [rows, cols] = shape else {
        bail!("{what} is {}-dimensional, expected 2", shape.len())
    };
    Ok((
        usize::try_from(*rows).with_context(|| format!("{what} row count {rows}"))?,
        usize::try_from(*cols).with_context(|| format!("{what} column count {cols}"))?,
    ))
}
