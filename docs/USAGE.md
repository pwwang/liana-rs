# Usage

All three surfaces run the same engine over the same inputs: one `.h5ad`, one
`.obs` label column, one ligand–receptor resource, one method. This page
documents each of them; every example below was executed against this
repository's fixtures and the printed output is verbatim.

- [Command line](#command-line)
- [Python module](#python-module)
- [Rust library](#rust-library)

## Command line

`crates/liana-cli` builds the `liana-rs` binary. The fixture used throughout
this page (`testdata/fixtures/synthetic.h5ad`, 4205 cells × 11 features, a
two-label `cell_type` column) pairs with the toy resource
`testdata/expected/synthetic__resource.csv` (110 `ligand,receptor` pairs):

```bash
cargo build -p liana-rs          # -> target/debug/liana-rs; `--help` lists every flag
target/debug/liana-rs run \
    --h5ad testdata/fixtures/synthetic.h5ad \
    --label-key cell_type \
    --resource-file testdata/expected/synthetic__resource.csv \
    --method cellphonedb \
    --n-perms 1000 \
    --seed 1337 \
    --threads 0 \
    --out target/usage_out/out.csv
```

```
liana-rs: 440 rows x 12 columns -> target/usage_out/out.csv
```

That line is the entire runtime output, and it goes to **stderr** — the run
writes nothing to stdout, so the CSV is what you redirect or pipe.

### Flags

| Flag | Meaning |
|---|---|
| `--h5ad <path>` | Input `.h5ad`. Required. |
| `--label-key <col>` | `obs` column holding the cluster labels. Required. |
| `--resource <name>` | Resource name in the vendored omni resource. Default `consensus`; conflicts with `--resource-file`. |
| `--resource-file <csv>` | A `ligand,receptor` CSV to score against instead of `--resource`. |
| `--method <name>` | One of the nine below. Required. |
| `--n-perms <n>` | Permutations for the permutation-scored methods. Default `1000`. |
| `--seed <u64>` | RNG seed. Default `1337`. |
| `--threads <n>` | Worker threads; `0` = one per core. Default `0`. |
| `--expr-prop <f64>` | liana's `expr_prop`. Default `0.05`. |
| `--min-cells <usize>` | liana's `min_cells`. Default `5`. |
| `--out <csv>` | Output CSV path. Required. |

`--resource` and `--resource-file` are **two different flags**, not one flag
taking either form: `--resource` is a name in the vendored resource only, and
passing both is a usage error. `--resource consensus` needs a dataset whose
features are in the resource — on the 11-gene synthetic fixture all 2016
consensus symbols are missing, so the fixture examples use `--resource-file`.

### Methods

The nine names, with each method's output shape on the fixture and the columns
it adds beyond the shared `ligand`/`ligand_complex`/`…props`/`source`/`target`
block (the method-specific columns, in CSV order):

| `--method` | rows × cols | method-specific columns |
|---|---|---|
| `cellphonedb` | 440 × 12 | `lr_means`, `cellphone_pvals` |
| `geometric_mean` | 440 × 12 | `lr_gmeans`, `gmean_pvals` |
| `cellchat` | 440 × 13 | `lr_probs`, `cellchat_pvals` |
| `connectome` | 440 × 14 | `expr_prod`, `scaled_weight` |
| `logfc` | 440 × 13 | `lr_logfc` |
| `natmi` | 440 × 14 | `expr_prod`, `spec_weight` |
| `scseqcomm` | 440 × 13 | `inter_score` |
| `singlecellsignalr` | 440 × 12 | `lrscore` |
| `rank_aggregate` | 440 × 19 | `lr_means`, `cellphone_pvals`, `expr_prod`, `scaled_weight`, `lr_logfc`, `spec_weight`, `lrscore`, `specificity_rank`, `magnitude_rank` |

`connectome` interleaves `ligand_zscores`/`receptor_zscores` and `scseqcomm`
`ligand_cdf`/`receptor_cdf` behind their sides' means; `rank_aggregate` puts
`source,target` first. The authoritative header is each method's own
`csv_header()`; the exact header the run above wrote is:

```
ligand,ligand_complex,ligand_means,ligand_props,receptor,receptor_complex,receptor_means,receptor_props,source,target,lr_means,cellphone_pvals
```

### The output table

One row per `(ligand_complex, receptor_complex, source, target)` — the 110
resource pairs crossed with the 4 ordered label pairs the fixture's two labels
give, which is the 440 rows above. `source`/`target` are cluster labels;
the score columns carry the method's own statistic and, for the
permutation-scored methods (`cellphonedb`, `geometric_mean`, `cellchat`,
`rank_aggregate`), a p-value. The file is a header line plus one line per row,
no quoting — 441 lines for the run above:

```bash
wc -l target/usage_out/out.csv
head -2 target/usage_out/out.csv
```

```
441 target/usage_out/out.csv
ligand,ligand_complex,ligand_means,ligand_props,receptor,receptor_complex,receptor_means,receptor_props,source,target,lr_means,cellphone_pvals
protE,protE,0.5501153,0.3955938697318008,protF,protF,0.4817482,1,A,A,0.5159317,0.118
```

`--n-perms` only reaches the four permutation-scored methods; the other five
accept it and ignore it (verified: their output at `p100` and `p1000` is
byte-identical, while all four scored methods' output changes). `--threads` is
a scheduling knob, not a result knob: `--threads 1`, `4` and `0` write
byte-identical CSVs.

### Exit codes

| Code | When | What is printed |
|---|---|---|
| `0` | the run wrote the CSV | `liana-rs: <rows> rows x <cols> columns -> <out>` on stderr |
| `1` | a runtime failure — unreadable `.h5ad`, a resource name that does not exist, a label key or resource that does not cover the data | `Error: …` plus a `Caused by:` chain on stderr |
| `2` | a usage error — an unknown `--method`, a missing required flag, `--resource` with `--resource-file` | clap's message and `try '--help'` |

The runtime case, verbatim (exit `1`):

```bash
target/debug/liana-rs run --h5ad nope.h5ad --label-key cell_type \
    --resource consensus --method cellphonedb --out target/usage_out/x.csv
```

```
Error: read nope.h5ad

Caused by:
    0: open nope.h5ad
    1: H5Fopen(): unable to synchronously open file: unable to open file: name = 'nope.h5ad', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0
```

The usage cases exit `2`, e.g. an unknown `--method`
(`error: invalid value 'nope' for '--method <METHOD>': unknown method "nope";
choose from ["cellphonedb", …, "rank_aggregate"]`), a missing required flag, or
both resource flags
(`error: the argument '--resource <RESOURCE>' cannot be used with
'--resource-file <RESOURCE_FILE>'`).

## Python module

`crates/liana-py` builds the `liana_rs` module. Its one function is
`liana_rs.run(h5ad, label_key, resource, method, n_perms=1000, seed=1337,
threads=0)` — the CLI's run with the same defaults minus `expr_prop`/
`min_cells`, which stay at liana's `0.05`/`5`. `resource` here is **one
string**: the path when a file by that name exists, else a resource name.

### Environment setup

A venv of its own, never the `scripts/oracle.sh` one:

```bash
uv venv target/py-venv
uv pip install --python target/py-venv/bin/python maturin pandas
VIRTUAL_ENV=$PWD/target/py-venv target/py-venv/bin/maturin develop --uv -m crates/liana-py/Cargo.toml
```

`uv venv` refuses a directory that already holds a venv — add `--clear` to
rebuild one from scratch. `maturin build --release -m crates/liana-py/Cargo.toml`
writes the wheel to `target/wheels/` instead of installing it.

### The call

```python
import liana_rs

df = liana_rs.run(
    "testdata/fixtures/synthetic.h5ad",
    "cell_type",
    "testdata/expected/synthetic__resource.csv",   # path, or a name like "consensus"
    "cellphonedb",
    n_perms=1000,
)
print(type(df).__name__)
print(df.shape)
print(df.head(3).to_string())
```

Executed in `target/py-venv` (Python 3.14.4, pandas 3.0.6):

```
DataFrame
(440, 12)
  ligand ligand_complex  ligand_means  ligand_props receptor receptor_complex  receptor_means  receptor_props source target  lr_means  cellphone_pvals
0  protE          protE      0.550115      0.395594    protF            protF        0.481748        1.000000      A      A  0.515932            0.118
1  protF          protF      0.481748      1.000000    protE            protE        0.550115        0.395594      A      A  0.515932            0.118
2  protF          protF      0.471731      1.000000    protE            protE        0.550115        0.395594      B      A  0.510923            0.181
```

The same 440 × 12 table the CLI writes, value for value — the gate
`scripts/parity_diff.py --rtol 0` passes it against the CLI's CSV with 0
differing cells. The two files are not byte-identical: pandas spells an
integral float `1.0` where the Rust writer emits `1`, so a `df.to_csv()`
differs textually on those cells while comparing equal by value. `run` also
releases the GIL for the duration of the engine call, so other Python threads
keep running. Errors surface as `ValueError` (e.g. `"nope" is neither an
existing resource file nor a resource name: resource "nope" not found; choose
from […]`).

### The no-pandas fallback

With pandas unimportable, `run` returns the columnar dict the Rust module
produced instead of a `DataFrame` — `{column: [values]}`, a column's values all
`float` when every cell parses as one, else all `str`:

```python
import sys
sys.modules["pandas"] = None          # makes `import pandas` raise ImportError

import liana_rs

result = liana_rs.run(
    "testdata/fixtures/synthetic.h5ad",
    "cell_type",
    "testdata/expected/synthetic__resource.csv",
    "cellphonedb",
    n_perms=1000,
)
print(type(result).__name__)
print(sorted(result))
print({k: (type(v).__name__, v[:2]) for k, v in list(result.items())[:3]})
```

```
dict
['cellphone_pvals', 'ligand', 'ligand_complex', 'ligand_means', 'ligand_props', 'lr_means', 'receptor', 'receptor_complex', 'receptor_means', 'receptor_props', 'source', 'target']
{'ligand': ('list', ['protE', 'protF']), 'ligand_complex': ('list', ['protE', 'protF']), 'ligand_means': ('list', [0.5501153, 0.4817482])}
```

## Rust library

`crates/liana-core` is the library the CLI and the Python module both drive.
`crates/liana-core/examples/run_synthetic.rs` is a 47-line program over its
public API — read the fixture, resolve the resource, run one method, print the
top rows:

```bash
cargo run -p liana-core --example run_synthetic
```

```
cellphonedb: 4205 cells x 11 features, 2 labels; resource of 110 pairs
440 rows x 12 columns (top 5)

ligand,ligand_complex,ligand_means,ligand_props,receptor,receptor_complex,receptor_means,receptor_props,source,target,lr_means,cellphone_pvals
protE,protE,0.5501153,0.3955938697318008,protF,protF,0.4817482,1,A,A,0.5159317,0.118
protF,protF,0.4817482,1,protE,protE,0.5501153,0.3955938697318008,A,A,0.5159317,0.118
protF,protF,0.47173086,1,protE,protE,0.5501153,0.3955938697318008,B,A,0.5109231,0.181
protE,protE,0.5501153,0.3955938697318008,protF,protF,0.47173086,1,A,B,0.5109231,0.181
protE,protE,0.5188592,0.38639584317430326,protF,protF,0.4817482,1,B,A,0.5003037,0.815
```

### Entry points

The example uses these, in the order it calls them. This block is a
**signature listing, not a program** — each signature below was compiled and
asserted against the built library for this page:

```rust
// liana_core::io — X as an f32 CSR matrix plus names, labels and obsm["spatial"]
pub fn read_h5ad(path: &Path, label_key: &str) -> Result<Adata>

// liana_core::resource — the vendored omni resource, or a toy `ligand,receptor` CSV
pub fn select(name: &str) -> Result<Vec<LrPair>>
pub fn read_pairs(path: &Path) -> Result<Vec<LrPair>>

// liana_core::run — the nine methods
pub const METHOD_NAMES: &[&str]
impl Method {
    pub fn parse(name: &str) -> Result<Self>      // unknown name -> error listing all nine
    pub fn name(self) -> &'static str
    pub fn csv_header(self) -> &'static str       // the oracle CSV header, column for column
    pub fn run(self, adata: &Adata, resource: &[LrPair], settings: &Settings) -> Result<Output>
}
pub struct Settings {                    // Default = liana's DefaultValues
    pub expr_prop: f64,                  // 0.05
    pub min_cells: usize,                // 5
    pub n_perms: usize,                  // 1000
    pub seed: u64,                       // 1337
    pub threads: usize,                  // 0 = one per core
}
pub struct Output {                      // the CSV contract's raw fields
    pub header: &'static str,
    pub rows: Vec<Vec<String>>,
}
impl Output {
    pub fn to_csv(&self) -> String       // header + rows, comma-separated, no quoting
}
```

`Adata` is `{ x: Csr, obs_names: Vec<String>, var_names: Vec<String>,
labels: Vec<u32>, label_names: Vec<String>, obsm_spatial: Option<Spatial> }`;
`LrPair` is `{ ligand: String, receptor: String }`. The cells in `Output::rows`
are strings because they are the CSV's fields — each is spelled so that parsing
it back recovers the same value.

For the whole CLI call in one function there is the convenience path (again a
**signature listing**), which is what `liana_rs.run` wraps:

```rust
// path-if-exists else resource name, then read -> resolve -> parse -> run
pub fn run_file(h5ad: &Path, label_key: &str, resource: &str, method: &str,
                settings: &Settings) -> Result<Output>
pub fn resolve_resource(resource: &str) -> Result<Vec<LrPair>>
```

A complete program with it (run from the repository root):

```rust
use std::path::Path;

use anyhow::Result;
use liana_core::run::{Settings, run_file};

fn main() -> Result<()> {
    let output = run_file(
        Path::new("testdata/fixtures/synthetic.h5ad"),
        "cell_type",
        "testdata/expected/synthetic__resource.csv",
        "rank_aggregate",
        &Settings::default(),
    )?;
    print!("{}", output.to_csv());
    Ok(())
}
```

It prints the 440-row CSVs' 19-column header and every row — the same file the
CLI writes for `--method rank_aggregate`.
