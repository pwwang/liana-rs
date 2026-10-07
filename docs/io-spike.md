# W2-T1 — h5ad I/O spike: `anndata` 0.7 + `anndata-hdf5` 0.5

Date 2026-10-06 · verdict: **partial — do not adopt; use the hdf5-metno fallback for T2.**

The spike is `crates/liana-core/examples/io_spike.rs`, behind the off-by-default
`io-spike` feature (anndata-rs + polars are not library dependencies):

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cargo run -p liana-core --example io_spike --features io-spike -- \
    testdata/fixtures/synthetic.h5ad /home/pwwang/p0a/data/sc_10000.h5ad
```

Resolved versions: `anndata 0.7.0`, `anndata-hdf5 0.5.3`, `hdf5-metno 0.12.5`,
`hdf5-metno-sys 0.11.4`, `hdf5-metno-src 0.10.4`, `blosc-src 0.3.8`,
`polars 0.53.0`, `ndarray 0.17.2`.

## Verdict

| File | Result |
|---|---|
| `testdata/fixtures/synthetic.h5ad` | **reads completely** — dense f64 X, obs categorical (materialized to strings), `var_names`, `obsm["spatial"]` |
| `/home/pwwang/p0a/data/sc_10000.h5ad` | **cannot be opened** — `obs/_index` / `var/_index` use the `nullable-string-array` encoding; X itself is readable through the low-level API |

The blocker is not HDF5 access but one encoding type: anndata ≥ 0.11 writes string
indices with missing values as a *group* (`mask` + `values`), and anndata-rs both

- lacks the encoding in its map — `backend.rs` accepts `nullable-integer` /
  `nullable-boolean`, and rejects everything else with
  `unsupported encoding type` (there is a `nullable-string` writer, but no
  `nullable-string-array` reader), and
- reads a dataframe index as a *dataset*: `DataFrameIndex::read` does
  `container.as_group()?.open_dataset("_index")` on a group → `H5Dopen2(): not a dataset`.

`AnnData::open` reads every element eagerly, so that one gap makes the whole file
unopenable: **every dataset produced by the `bench/` generator** (`sc_1000`,
`sc_10000`, `sc_50000`, `sc_100000` — written by anndata ≥ 0.11) is affected, while
the older vendored fixture is not. That is exactly the wrong way round: the crate
covers the toy fixture and fails on the production data.

Upstream README states `.layers` and `.raw` are unsupported too; neither is needed
by `liana-rs` today, but neither is covered.

### What the crate does cover (for the record)

`synthetic.h5ad` read through `AnnData::<H5>::open`:

```
shape: 4205 x 11
X dtype: Some(Array(F64)), shape: Some(Shape([4205, 11]))
X kind: dense, nnz=31069
obs columns: ["cell_type"]
  obs[cell_type]: String, head=[A, B, B]          # categorical materialized to strings
var columns: []
var_names head: ["ECM", "ligA", "ligB", "ligC", "ligD", "protE", "protF", "prodA", "prodB", "prodC", "prodD"]
obs_names head: ["0", "1", "2"]
obsm keys: ["spatial"]
  obsm[spatial]: dtype=Some(Array(I64)), shape=Some(Shape([4205, 2]))
    first values: [1.0, 100.0, 1.0]               # row 0 = (1, 100), matching anndata's [1, 100]
```

Verified against the oracle (`anndata.read_h5ad`): shapes, `nnz=31069`, `cell_type`
∈ {A,B}, `spatial[0] = [1, 100]`, `obs_names = 0,1,2,…` all match.

### The exact failure on `sc_10000.h5ad`

```
== /home/pwwang/p0a/data/sc_10000.h5ad
  FAILED: open …: H5Dopen2(): unable to synchronously open dataset: not a dataset
  low-level X probe: CSR, shape=Shape([10000, 2000]), nnz=1998938
```

The low-level probe (`DataContainer::open(&file, "X")` + `ArrayElem::try_from`,
skipping obs/var) reads the CSR correctly: 10000 × 2000, `nnz=1998938` — the exact
nnz in the dataset manifest (W1-A T4). So the HDF5 stack works; only the index
encoding does not. `obs/cell_type`, `obsm/spatial` and the `X` group are all
ordinary datasets/groups in this file (see below), i.e. nothing else in it is exotic.

## Build requirements (measured, this machine)

- A C compiler (`cc`/gcc — conda-forge gcc 12.1.0 here) **and** cmake (3.31.4):
  `hdf5-metno-sys` builds bundled HDF5 C sources from `hdf5-metno-src`, and
  `blosc-src` builds Blosc, both through the `cmake` crate wrapper.
- `anndata-hdf5` pins `hdf5-metno-sys` with `static, zlib, threadsafe` and
  `hdf5-metno` with `blosc, blosc-zstd` — a **static** HDF5 with threadsafe locking.
- 282 crates compile in ~2 min wall (first build, warm cache); polars dominates.
  For reference, `liana-rs`'s current dependency set is 100+ crates *lighter* and
  builds in ~15 s.
- The build succeeded here with no extra system packages beyond cmake + gcc.

## Fallback plan (adopted by T2)

Hand-roll a minimal reader directly on **`hdf5-metno`** — the same HDF5 bindings
`anndata-hdf5` uses, minus anndata-rs, polars, and the parts of the spec we do not
touch. Both files were inspected at the HDF5 level; the subset we need is:

| Element | Encodings present in our files | Needed by |
|---|---|---|
| `X` | dataset `array` (f64, dense) **or** group `csr_matrix` (`data`/`indices`/`indptr`) | always |
| `obs/<label>` | group `categorical` (`categories` + `codes`) | labels |
| `obs/_index`, `var/_index` | dataset `string-array` (fixture) **or** group `nullable-string-array` (`values` + `mask`) | names |
| `obsm/spatial` | dataset `array` (int64) | optional |

Neither file uses dataset compression (`compression=None` throughout), so default
`hdf5-metno` features suffice; `zlib` is the upgrade path if that changes. This is
~200 lines against a stable, fully specified encoding set — smaller than the
polars/anndata dependency graph it replaces, and it is the *only* option that reads
`p0a/data`.

`anndata`/`anndata-hdf5` stay in `Cargo.lock` behind the `io-spike` feature so this
verdict can be re-tested cheaply when anndata-rs gains `nullable-string-array`
support.
