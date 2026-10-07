#!/usr/bin/env bash
# CLI parity gate: `liana-rs run` on the synthetic fixture + the toy resource
# must reproduce every oracle CSV value-exactly (`parity_diff.py --rtol 0`), for
# all nine methods x n_perms {100, 1000} — the same contract
# scripts/check_pipe_parity.sh pins at the library level.
#
# Usage: scripts/check_cli_parity.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PATH="$HOME/.cargo/bin:$PATH"

cargo build -p liana-rs

mkdir -p target/cli_out
methods=(cellphonedb geometric_mean cellchat connectome logfc natmi \
         scseqcomm singlecellsignalr rank_aggregate)
for method in "${methods[@]}"; do
    for n in 100 1000; do
        target/debug/liana-rs run \
            --h5ad testdata/fixtures/synthetic.h5ad \
            --label-key cell_type \
            --resource-file testdata/expected/synthetic__resource.csv \
            --method "$method" \
            --n-perms "$n" \
            --out "target/cli_out/synthetic__${method}__p${n}.csv"
        python3 scripts/parity_diff.py \
            --expected "testdata/expected/synthetic__${method}__p${n}.csv" \
            --actual "target/cli_out/synthetic__${method}__p${n}.csv" \
            --rtol 0
    done
done
