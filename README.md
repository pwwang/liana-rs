# liana-rs

Rust reimplementation of the single-cell path of [LIANA+](https://github.com/scverse/liana), targeting bit-exact parity with liana 2.0.0 (tag `V2.0.0`, commit `c59472ccc9de8360dbbf5016db75f8abde08dd3e`) while removing its memory ceiling.

## Layout

- `crates/liana-core` — library: ligand–receptor scoring (cellphonedb, geometric_mean, rank_aggregate, …)
- `crates/liana-cli` — `liana-rs` command-line binary
- `testdata/` — vendored parity fixtures and oracle outputs (see `testdata/FIXTURES.md`)
- `scripts/oracle.sh` — generates expected outputs from the pinned Python oracle
- `bench/` — paper-recipe dataset generator and benchmark runner

## License

BSD-3-Clause (see `LICENSE`). This project is an independent reimplementation; it vendors test fixtures from [scverse/liana](https://github.com/scverse/liana) (BSD-3-Clause) — see `testdata/FIXTURES.md` for provenance.
