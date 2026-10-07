#!/usr/bin/env bash
# End-to-end pipe parity gate.
#
# Runs `crates/liana-core/tests/{pipe,cellchat}_parity.rs` — which assert the
# values bit-exactly and write target/pipe_out/synthetic__<method>__p<N>.csv —
# and then re-checks the written CSVs against the oracle with
# scripts/parity_diff.py, keyed on the four key columns (the row order is not
# part of the contract; see ops/logs/w3-report.md).
#
# Tolerances: all nine methods are value-exact, so every cross-check runs at
# rtol=0. geometric_mean's `lr_gmeans` (`exp((log l + log r)/2)` in f32
# through numpy's kernels) used to be the exception at rtol 5e-7 (W3 D2);
# `crates/liana-core/src/math/` now ports those kernels bit-exactly — and the
# W5a kernels behind the five remaining methods, and the `betainc` behind
# rank_aggregate's Robust Rank Aggregation, too.
#
# Usage: scripts/check_pipe_parity.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PATH="$HOME/.cargo/bin:$PATH"

cargo test -p liana-core --test pipe_parity
cargo test -p liana-core --test cellchat_parity

methods=(cellphonedb geometric_mean cellchat connectome logfc natmi \
         scseqcomm singlecellsignalr rank_aggregate)
for method in "${methods[@]}"; do
    for n in 100 1000; do
        python3 scripts/parity_diff.py \
            --expected "testdata/expected/synthetic__${method}__p${n}.csv" \
            --actual "target/pipe_out/synthetic__${method}__p${n}.csv" \
            --rtol 0
    done
done
