# Installation

`liana-rs` has three surfaces, all built from this repository:

- the `liana-rs` **command-line binary** (`crates/liana-cli`),
- the `liana_rs` **Python module** (`crates/liana-py`, PyO3 + maturin),
- the `liana-core` **Rust library** (`crates/liana-core`) the other two drive.

Nothing is published on crates.io or PyPI (checked 2026-10-07: crates.io
returns `404 crate 'liana-core' does not exist` for `liana-core` and `liana-rs`;
`https://pypi.org/pypi/liana-rs/json` is `404`). Every route below starts from
a source checkout, a locally built wheel, or a wheel artifact from CI.

Every command on this page was executed on **2026-10-07** on the machine in
[Requirements](#requirements), and the outputs shown are verbatim — paths
shortened to `<repo>` where noted, and any elision marked with `...`.
Anything not executed here carries an explicit **not verified here** note
with the reason.

- [Requirements](#requirements)
- [Command line](#command-line)
- [Python module](#python-module)
- [Rust library](#rust-library)
- [Smoke test](#smoke-test)

## Requirements

### Rust toolchain

**rustc ≥ 1.85.** The workspace declares `edition = "2024"`
(`Cargo.toml`, `[workspace.package]`), and edition 2024 requires 1.85 or
newer. There is no `rust-toolchain.toml`, so any stable toolchain ≥ 1.85
should work — but only one has been exercised, this box's:

```
$ rustc --version
rustc 1.93.1 (01f6ddf75 2026-02-11)
$ cargo --version
cargo 1.93.1 (083ac5135 2025-12-15)
```

(The 1.85 floor is what the manifest implies; whether the transitive
dependencies allow a compile that old was **not verified here** — only 1.93.1
was used.)

`rustfmt` and `clippy` are required for development (CI fails without them),
not for installing.

<!--
> **WSL/conda gotcha (hit during this run).** On this machine
> `/home/pwwang/miniconda3/bin` precedes `~/.cargo/bin` on `PATH`, and its
> `cargo` is 1.79 — too old for edition 2024. The build then stops with
> (verbatim, from a maturin run):
> ```
> error: failed to parse manifest at `/home/pwwang/github/liana-rs/crates/liana-py/Cargo.toml`
>
> Caused by:
>   feature `edition2024` is required
>
>   The package requires the Cargo feature called `edition2024`, but that feature is not stabilized in this version of Cargo (1.79.0 (ffa9cf99a 2024-06-03)).
>   ...
> 💥 maturin failed
>   Caused by: Cargo metadata failed. Does your crate compile with `cargo build`?
> ```
> Put rustup first and it builds:
> ```bash
> export PATH="$HOME/.cargo/bin:$PATH"
> cargo --version   # must print 1.93.x, not 1.79.0
> ```
> Toolchains that resolve a modern `cargo` themselves (rustup shims already
> first on `PATH`, or a fresh container) do not need this.
-->

### C toolchain and HDF5

`liana-core` depends on `hdf5-metno 0.12` with
`features = ["static", "zlib"]` (`Cargo.toml`, `[workspace.dependencies]`),
so HDF5 is **compiled from source and linked statically**. There is **no
system HDF5 requirement** — no `libhdf5-dev`/`hdf5-devel` package needed.
Verified against the built binary:

```
$ ldd target/release/liana-rs | grep -i hdf5
$ # (empty)
```

After that grep the whole `ldd` list is just the C runtime —
`linux-vdso.so.1`, `libgcc_s.so.1`, `librt.so.1`, `libpthread.so.0`,
`libm.so.6`, `libdl.so.2`, `libc.so.6`, `ld-linux-x86-64.so.2` — no `libhdf5`
anywhere, and consequently nothing HDF5-related to install on the target
machine either.

What you do need at build time is **cmake and a C compiler**, to build the
bundled HDF5. On this box:

```
$ cmake --version | head -1
cmake version 3.31.4
$ cc --version | head -1
cc (conda-forge gcc 12.1.0-17) 12.1.0
```

The `zlib` feature compiles deflate into the bundled HDF5. That is not
optional in practice: real-world `.h5ad` files (h5py's default) are
gzip-chunked, and without it HDF5 tries to load the filter as a plugin and
fails — the failure and its fix are in `ops/logs/w6-report.md`.

Linking against a *system* HDF5 instead would be a change to the workspace
feature set — **not verified here**; the tested configuration is the bundled
static build above.

### Python

- `requires-python = ">=3.9"` (`crates/liana-py/pyproject.toml`).
- Build backend: maturin `>=1.0,<2.0`; this run used maturin 1.15.0.
- Wheels are built **per interpreter** (not `abi3`), so a wheel matches one
  CPython version: the wheel built on this box is `cp314`, the CI Linux
  artifact is `cp39`, the CI macOS artifact `cp314`, the CI Windows artifact
  `cp312`. Interpreters exercised here: CPython 3.14.4 and CPython 3.9
  (both uv-managed).
- `pandas` is optional: with it you get a `DataFrame`, without it the
  columnar dict (`docs/USAGE.md`).

### Machine resources

| Item | This box |
|---|---|
| CPU | 32 threads (`nproc` → 32) |
| RAM | 47 GiB total, ~42 GiB available |
| OS | Ubuntu 24.04 (WSL2), kernel `6.18.33.2-microsoft-standard-WSL2`, x86_64 |

`--threads` (CLI) / `threads=` (Python) sets the worker-thread count; `0` —
the default — means one worker per core. Threads affect scheduling only: the
output is byte-identical across `--threads 1`, `4`, `0` (`docs/USAGE.md`).
Measured here with `/usr/bin/time -v`:

| Run | Wall | Peak RSS | CPU |
|---|---|---|---|
| `rank_aggregate` p1000 on the fixture (440 rows) | 0.09 s | 9.7 MB | — |
| `rank_aggregate` p1000 on kang (24 673 cells × 15 706 genes, 8 labels, consensus resource) | 7.3 s | **396 MB** | 1265% (~12.6 cores) |

The kang row is this exact command (dataset provenance and its sha256 are in
`testdata/real/README.md`; it was verified identical to the recorded
`eacdf3a2…` before the run):

```bash
/usr/bin/time -v target/release/liana-rs run \
    --h5ad testdata/real/kang_lognorm.h5ad --label-key cell_abbr \
    --resource consensus --method rank_aggregate --n-perms 1000 \
    --seed 1337 --threads 0 --out /tmp/w13_kang_ra.csv
# liana-rs: 3509 rows x 19 columns -> /tmp/w13_kang_ra.csv
```

The CLI's own dependency footprint is small because HDF5 is static: the
release binary is 8.4 MB (`target/release/liana-rs`).

### Platforms

The repository has been **built and run only on Linux x86_64**, specifically
WSL2 / Ubuntu 24.04 on the box above, 32 threads. Concretely:

- **Linux x86_64** — everything in this document was executed here.
- **macOS (arm64)** — **not verified here.** The wheels workflow built a
  `liana_rs-0.1.0-cp314-cp314-macosx_11_0_arm64.whl` artifact (downloaded and
  inspected below), but this machine cannot execute it.
- **Windows (x64)** — **not verified here.** Same: the CI artifact
  `liana_rs-0.1.0-cp312-cp312-win_amd64.whl` exists, nothing was executed.
  The workflow marks the Windows job `continue-on-error` on purpose (the
  static HDF5 build under MSVC is the risk).
- CI's `fmt + clippy + test` job runs on `ubuntu-latest` only
  (`.github/workflows/ci.yml`), so the Rust test suite runs on no other OS.

## Command line

### Build from source

```bash
export PATH="$HOME/.cargo/bin:$PATH"   # the WSL/conda gotcha above
cargo build --release -p liana-rs
```

→ `target/release/liana-rs`. On this box the workspace was already built, so
the command finished in 1.62 s doing nothing; a **from-scratch release build**
of the CLI — dependencies included, i.e. the bundled HDF5 C library — took
**96 s** on 32 cores, measured via the `cargo install` below (which always
builds fresh).

### Install into a cargo bin dir

```bash
cargo install --path crates/liana-cli                 # -> ~/.cargo/bin/liana-rs
```

To keep `~/.cargo/bin` untouched, the same install into a scratch root — the
form executed here:

```bash
$ cargo install --path crates/liana-cli --root /tmp/liana-install-root
   Installing /tmp/liana-install-root/bin/liana-rs
   Installed package `liana-rs v0.1.0 (/home/pwwang/github/liana-rs/crates/liana-cli)` (executable `liana-rs`)
warning: be sure to add `/tmp/liana-install-root/bin` to your PATH to be able to run the installed binaries
```

To use it in a shell session, either `export PATH="/tmp/liana-install-root/bin:$PATH"`
or call it by path as below. The plain `~/.cargo/bin` form differs only in the
destination directory (already on `PATH` for rustup users) — **not verified
here**, skipped deliberately to avoid writing into the user's bin dir.

### Smoke test

`--help` and a real run over the vendored fixture
(`testdata/fixtures/synthetic.h5ad`, 4205 cells × 11 features, paired with the
toy resource `testdata/expected/synthetic__resource.csv`):

```bash
$ /tmp/liana-install-root/bin/liana-rs --version
liana-rs 0.1.0

$ cd <repo>
$ /tmp/liana-install-root/bin/liana-rs run \
    --h5ad testdata/fixtures/synthetic.h5ad \
    --label-key cell_type \
    --resource-file testdata/expected/synthetic__resource.csv \
    --method cellphonedb \
    --n-perms 1000 \
    --seed 1337 \
    --threads 0 \
    --out /tmp/liana-install-run.csv
liana-rs: 440 rows x 12 columns -> /tmp/liana-install-run.csv
```

That status line goes to **stderr**; the CSV is the only file written:

```bash
$ wc -l /tmp/liana-install-run.csv
441 /tmp/liana-install-run.csv
$ head -1 /tmp/liana-install-run.csv
ligand,ligand_complex,ligand_means,ligand_props,receptor,receptor_complex,receptor_means,receptor_props,source,target,lr_means,cellphone_pvals
```

The top-level `--help` lists the one subcommand:

```
$ liana-rs --help
liana's single-cell ligand–receptor methods, in Rust

Usage: liana-rs <COMMAND>

Commands:
  run   Run one method and write the result CSV
  help  Print this message or the help of the given subcommand(s)
```

`liana-rs run --help` lists every flag with its default; `docs/USAGE.md` has
the flag table, exit codes (0 success, 1 runtime error, 2 usage error) and
the nine `--method` names.

## Python module

### (a) From a source checkout (maturin develop)

A venv of its own — never the `scripts/oracle.sh` one — as in
`docs/USAGE.md`:

```bash
$ uv venv target/py-venv          # first-time setup; on this box the venv already existed
$ uv pip install --python target/py-venv/bin/python maturin pandas
Using Python 3.14.4 environment at: target/py-venv
Checked 2 packages in 10ms
$ VIRTUAL_ENV=$PWD/target/py-venv target/py-venv/bin/maturin develop --uv -m crates/liana-py/Cargo.toml
🍹 Building a mixed python/rust project
🐍 Found CPython 3.14 at /home/pwwang/github/liana-rs/target/py-venv/bin/python
🔗 Found pyo3 bindings
   Compiling liana-py v0.1.0 (/home/pwwang/github/liana-rs/crates/liana-py)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 4.49s
📦 Built wheel for CPython 3.14 to /tmp/.tmpPQ4ZJK/liana_rs-0.1.0-cp314-cp314-linux_x86_64.whl
✏️ Setting installed package as editable
🛠 Installed liana-rs-0.1.0
```

The parity gate against the module (9 methods × 2 `n_perms`, plus the
no-pandas fallback) passes on this install:

```bash
$ target/py-venv/bin/python scripts/check_py_parity.py
...
GATE: PASS — no-pandas fallback returns the dict
py gate: 19/19 PASS
```

`VIRTUAL_ENV=...` matters: maturin installs into the venv it is told about
(or the one activated), and if it finds neither it silently targets the first
`python` on `PATH` — here that is conda's 3.12, which is how the same build
command once produced a `cp312` wheel instead of `cp314`.

### (b) Build a wheel, install it into a fresh venv

```bash
$ VIRTUAL_ENV=$PWD/target/py-venv target/py-venv/bin/maturin build --release -m crates/liana-py/Cargo.toml
📦 Built wheel for CPython 3.14 to /home/pwwang/github/liana-rs/target/wheels/liana_rs-0.1.0-cp314-cp314-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

The wheel lands in the workspace `target/wheels/`. Installing it into a
**fresh** venv — created just now, nothing but the wheel in it — proves the
wheel is self-contained:

```bash
$ uv venv --python 3.14 /tmp/liana-wheel-venv
Creating virtual environment at: /tmp/liana-wheel-venv
Activate with: source /tmp/liana-wheel-venv/bin/activate
$ uv pip install --python /tmp/liana-wheel-venv/bin/python target/wheels/liana_rs-0.1.0-cp314-cp314-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
Installed 1 package in 0.81ms
 + liana-rs==0.1.0 (from file:///home/pwwang/github/liana-rs/target/wheels/liana_rs-0.1.0-cp314-cp314-manylinux_2_17_x86_64.manylinux2014_x86_64.whl)

$ cd <repo>   # the fixture path below is relative
$ /tmp/liana-wheel-venv/bin/python -c "
import liana_rs
r = liana_rs.run('testdata/fixtures/synthetic.h5ad', 'cell_type',
                 'testdata/expected/synthetic__resource.csv', 'cellphonedb', n_perms=1000)
print(type(r).__name__)
print(len(r['ligand']), 'rows,', len(r), 'columns')
"
dict
440 rows, 12 columns
```

(`dict` — no pandas in this venv, so the module returned its columnar
fallback; that is the documented behavior, not a fault.) Adding pandas in
the same venv flips it to the `DataFrame` path:

```bash
$ uv pip install --python /tmp/liana-wheel-venv/bin/python pandas
$ /tmp/liana-wheel-venv/bin/python -c "
import liana_rs
df = liana_rs.run('testdata/fixtures/synthetic.h5ad', 'cell_type',
                  'testdata/expected/synthetic__resource.csv', 'cellphonedb')
print(type(df).__name__, df.shape)
"
DataFrame (440, 12)
```

### (c) The GitHub Actions wheels workflow

`.github/workflows/wheels.yml` builds wheels on `workflow_dispatch` and on
tags `v*`, with a three-entry matrix (plus `fail-fast: false`):

| `${{ matrix.target }}` | runs-on | artifact |
|---|---|---|
| `x86_64` | `ubuntu-latest` (maturin-action, `manylinux: auto`) | `wheels-x86_64` |
| `aarch64-apple-darwin` | `macos-latest` | `wheels-aarch64-apple-darwin` |
| `x64` | `windows-latest` (`continue-on-error`) | `wheels-x64` |

The workflow ran on 2026-10-07 (run `37649832985`, `workflow_dispatch` on
master, 4m30s, success): all three jobs passed — `wheel (x86_64)` 2m10s,
`wheel (aarch64-apple-darwin)` 59s, `wheel (x64)` 4m24s. Downloading an
artifact with the `gh` CLI (or from the run page's Artifacts section):

```bash
$ gh run download 37649832985 -n wheels-x86_64 -D /tmp/liana-ci-wheel
$ ls /tmp/liana-ci-wheel
liana_rs-0.1.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

The Linux artifact is a `cp39` manylinux wheel; it installs and runs on this
box in a fresh Python 3.9 venv, with no build step — so the artifact is
genuinely self-contained:

```bash
$ uv venv --python 3.9 /tmp/liana-ci-venv
Creating virtual environment at: /tmp/liana-ci-venv
Activate with: source /tmp/liana-ci-venv/bin/activate
$ uv pip install --python /tmp/liana-ci-venv/bin/python /tmp/liana-ci-wheel/liana_rs-0.1.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
 + liana-rs==0.1.0 (from file:///tmp/liana-ci-wheel/liana_rs-0.1.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl)
$ cd <repo>
$ /tmp/liana-ci-venv/bin/python -c "
import liana_rs
r = liana_rs.run('testdata/fixtures/synthetic.h5ad', 'cell_type',
                 'testdata/expected/synthetic__resource.csv', 'cellphonedb', n_perms=1000)
print(type(r).__name__, len(r['ligand']), 'rows,', len(r), 'columns; lr_means[0] =', r['lr_means'][0])
"
dict 440 rows, 12 columns; lr_means[0] = 0.5159317
```

The other two artifacts were downloaded and inspected but not executed here
(no macOS, no Windows on this machine): `wheels-aarch64-apple-darwin` holds
`liana_rs-0.1.0-cp314-cp314-macosx_11_0_arm64.whl` and `wheels-x64` holds
`liana_rs-0.1.0-cp312-cp312-win_amd64.whl`.

**Nothing is published on PyPI** — the workflow only uploads build artifacts.
`pip install liana-rs` has nothing to fetch (`https://pypi.org/pypi/liana-rs/json`
→ `404`, checked 2026-10-07). Until that changes, install from a source
checkout, a locally built wheel, or an artifact as above.

## Rust library

There is no crates.io release (`https://crates.io/api/v1/crates/liana-core`
→ `404 crate 'liana-core' does not exist`, checked 2026-10-07), so the way to
consume `liana-core` is the git repository.

Probe crate, created **outside** the repo at `/tmp/liana-dep-probe` — the
whole of it:

```toml
# /tmp/liana-dep-probe/Cargo.toml
[package]
name = "liana-dep-probe"
version = "0.1.0"
edition = "2024"

[dependencies]
anyhow = "1"
liana-core = { git = "https://github.com/pwwang/liana-rs" }
```

```rust
// /tmp/liana-dep-probe/src/main.rs
use std::path::Path;

use liana_core::run::{Settings, run_file};

fn main() -> anyhow::Result<()> {
    let out = run_file(
        Path::new("/home/pwwang/github/liana-rs/testdata/fixtures/synthetic.h5ad"),
        "cell_type",
        "/home/pwwang/github/liana-rs/testdata/expected/synthetic__resource.csv",
        "cellphonedb",
        &Settings::default(),
    )?;
    println!("{} rows x {} columns", out.rows.len(), out.header.split(',').count());
    println!("{}", out.rows[0].join(","));
    Ok(())
}
```

Build and run (rustup's `cargo`, C toolchain present as in
[Requirements](#requirements)):

```bash
$ cd /tmp/liana-dep-probe
$ cargo run
   Compiling liana-core v0.1.0 (https://github.com/pwwang/liana-rs#37fbe3a2)
   Compiling liana-dep-probe v0.1.0 (/tmp/liana-dep-probe)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 2m 44s
     Running `target/debug/liana-dep-probe`
440 rows x 12 columns
protE,protE,0.5501153,0.3955938697318008,protF,protF,0.4817482,1,A,A,0.5159317,0.118
```

The 2m44s is the bundled-HDF5 build happening once inside the probe's own
`target/`; consuming the crate from the git URL on macOS or Windows is
**not verified here** (no such machine). The git dependency resolved the
remote's default branch at the time to
`37fbe3a2344c6ef66b7b2cc63a4b028dd4c0178e`, recorded in the probe's
`Cargo.lock`:

```
source = "git+https://github.com/pwwang/liana-rs#37fbe3a2344c6ef66b7b2cc63a4b028dd4c0178e"
```

A bare `git = ...` tracks that default branch. To pin a revision, add `rev`
(verified to resolve here — the lock then records
`git+https://github.com/pwwang/liana-rs?rev=37fbe3a2344c6ef66b7b2cc63a4b028dd4c0178e#37fbe3a2`):

```toml
liana-core = { git = "https://github.com/pwwang/liana-rs", rev = "37fbe3a2344c6ef66b7b2cc63a4b028dd4c0178e" }
```

`liana-core`'s default features are all you need — the optional `io-spike`
feature (anndata + polars coverage spike) is off by default and not used by
the CLI or the Python module.

## Smoke test

One command per surface, all executed here; each is a complete proof that
its install works.

**CLI** — run one method over the vendored fixture:

```bash
$ /tmp/liana-install-root/bin/liana-rs run \
    --h5ad testdata/fixtures/synthetic.h5ad --label-key cell_type \
    --resource-file testdata/expected/synthetic__resource.csv \
    --method cellphonedb --out /tmp/smoke.csv
liana-rs: 440 rows x 12 columns -> /tmp/smoke.csv
```

**Python** — import the installed module and run the same call:

```bash
$ /tmp/liana-wheel-venv/bin/python -c "
import liana_rs
df = liana_rs.run('testdata/fixtures/synthetic.h5ad', 'cell_type',
                  'testdata/expected/synthetic__resource.csv', 'cellphonedb')
print(type(df).__name__, df.shape)
"
DataFrame (440, 12)
```

**Rust** — the library example shipped in this repo:

```bash
$ cargo run -q -p liana-core --example run_synthetic
cellphonedb: 4205 cells x 11 features, 2 labels; resource of 110 pairs
440 rows x 12 columns (top 5)

ligand,ligand_complex,ligand_means,ligand_props,receptor,receptor_complex,receptor_means,receptor_props,source,target,lr_means,cellphone_pvals
protE,protE,0.5501153,0.3955938697318008,protF,protF,0.4817482,1,A,A,0.5159317,0.118
protF,protF,0.4817482,1,protE,protE,0.5501153,0.3955938697318008,A,A,0.5159317,0.118
protF,protF,0.47173086,1,protE,protE,0.5501153,0.3955938697318008,B,A,0.5109231,0.181
protE,protE,0.5501153,0.3955938697318008,protF,protF,0.47173086,1,A,B,0.5109231,0.181
protE,protE,0.5188592,0.38639584317430326,protF,protF,0.4817482,1,B,A,0.5003037,0.815
...
```

All three print the same 440 × 12 table — one engine, three front doors.
(The example prints the header and the top five rows; the CLI and the Python
module write or return all 440.)
