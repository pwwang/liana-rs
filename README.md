# liana-rs

Rust reimplementation of the single-cell path of [LIANA+](https://github.com/scverse/liana), targeting bit-exact parity with liana 2.0.0 (tag `V2.0.0`, commit `c59472ccc9de8360dbbf5016db75f8abde08dd3e`) while removing its memory ceiling.

## Install

Toolchain requirements (edition 2024, bundled static HDF5, cmake + C compiler)
and a verified install path for each surface — the `liana-rs` binary, the
`liana_rs` Python wheel, and `liana-core` as a git dependency — are in
[docs/INSTALL.md](docs/INSTALL.md). Nothing is published on crates.io or PyPI.

## Usage

One call runs one method over one `.h5ad` and writes liana's result table — the
same engine behind all three surfaces. [docs/USAGE.md](docs/USAGE.md) has the
verified examples, flag reference and exit codes.

```bash
liana-rs run --h5ad data.h5ad --label-key cell_type --resource consensus \
    --method cellphonedb --n-perms 1000 --seed 1337 --threads 0 --out out.csv
```

```python
import liana_rs; df = liana_rs.run("data.h5ad", "cell_type", "consensus", "cellphonedb")
```

```rust
use liana_core::run::{Settings, run_file};
let output = run_file("data.h5ad".as_ref(), "cell_type", "consensus", "cellphonedb", &Settings::default())?;
```

## Layout

- `crates/liana-core` — library: ligand–receptor scoring (cellphonedb, geometric_mean, rank_aggregate, …)
- `crates/liana-cli` — `liana-rs` command-line binary
- `crates/liana-py` — the `liana_rs` Python module (PyO3 + maturin)
- `testdata/` — vendored parity fixtures and oracle outputs (see `testdata/FIXTURES.md`)
- `scripts/oracle.sh` — generates expected outputs from the pinned Python oracle
- `bench/` — paper-recipe dataset generator and benchmark runner

## Python module

`crates/liana-py` builds the `liana_rs` module, a thin PyO3 wrapper over the
same engine the CLI drives (`liana_rs.run(h5ad, label_key, resource, method,
n_perms=1000, seed=1337, threads=0)` returns a pandas DataFrame, or the
columnar dict when pandas is unimportable). Wheels for linux x86_64, macos
arm64 and (best-effort) windows are built by `.github/workflows/wheels.yml`.

Local build + gate, in a venv of its own — not the `scripts/oracle.sh` one:

```bash
uv venv target/py-venv
uv pip install --python target/py-venv/bin/python maturin pandas
VIRTUAL_ENV=$PWD/target/py-venv target/py-venv/bin/maturin develop --uv -m crates/liana-py/Cargo.toml
target/py-venv/bin/python scripts/check_py_parity.py   # 19/19 PASS
```

`maturin build --release -m crates/liana-py/Cargo.toml` writes the wheel to
`target/wheels/` (the workspace target dir) instead of installing it.

## CLI

```bash
cargo build -p liana-rs   # target/debug/liana-rs --help
liana-rs run --h5ad data.h5ad --label-key cell_type --resource consensus \
    --method cellphonedb --n-perms 1000 --seed 1337 --out out.csv
```

`scripts/check_cli_parity.sh` runs the built binary over all nine methods on
the synthetic fixture and diffs against the oracle CSVs at `rtol 0`.

## License

BSD-3-Clause (see `LICENSE`). This project is an independent reimplementation; it vendors test fixtures from [scverse/liana](https://github.com/scverse/liana) (BSD-3-Clause) — see `testdata/FIXTURES.md` for provenance.
