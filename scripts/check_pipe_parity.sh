#!/usr/bin/env bash
# End-to-end pipe parity gate.
#
# Runs `crates/liana-core/tests/{pipe,cellchat}_parity.rs` — which assert the
# values bit-exactly and write target/pipe_out/synthetic__<method>__p<N>.csv —
# and then re-checks the written CSVs against the oracle with
# scripts/parity_diff.py, keyed on the four key columns (the row order is not
# part of the contract; see ops/logs/w3-report.md).
#
# Tolerances: cellphonedb and cellchat are value-exact, so their cross-checks
# run at rtol=0.
# For geometric_mean, `lr_gmeans` is the one column the Rust side cannot make
# bit-exact — numpy 2.5.3 evaluates `exp((log l + log r)/2)` in f32 through
# Google Highway's kernels, which differ from Rust's libm on 182/440 rows by
# up to 4 ulp (2.34e-7 relative, measured); the Rust test pins that count and
# bound, and this looser cross-check bound (still below liana's own 1e-6 tie
# rtol) covers it. All other gmean columns are bit-exact in that test.
#
# Usage: scripts/check_pipe_parity.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PATH="$HOME/.cargo/bin:$PATH"

cargo test -p liana-core --test pipe_parity
cargo test -p liana-core --test cellchat_parity

for n in 100 1000; do
    python3 scripts/parity_diff.py \
        --expected "testdata/expected/synthetic__cellphonedb__p${n}.csv" \
        --actual "target/pipe_out/synthetic__cellphonedb__p${n}.csv" \
        --rtol 0
    python3 scripts/parity_diff.py \
        --expected "testdata/expected/synthetic__cellchat__p${n}.csv" \
        --actual "target/pipe_out/synthetic__cellchat__p${n}.csv" \
        --rtol 0
    python3 scripts/parity_diff.py \
        --expected "testdata/expected/synthetic__geometric_mean__p${n}.csv" \
        --actual "target/pipe_out/synthetic__geometric_mean__p${n}.csv" \
        --rtol 5e-7
done
