#!/usr/bin/env bash
# End-to-end cellphonedb parity gate.
#
# Runs `crates/liana-core/tests/pipe_parity.rs` — which asserts the values
# bit-exactly and writes target/pipe_out/synthetic__cellphonedb__p<N>.csv —
# and then re-checks the written CSVs against the oracle with
# scripts/parity_diff.py at rtol=0, keyed on the four key columns (the row
# order is not part of the contract; see ops/logs/w3-report.md).
#
# Usage: scripts/check_pipe_parity.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PATH="$HOME/.cargo/bin:$PATH"

cargo test -p liana-core --test pipe_parity

for n in 100 1000; do
    python3 scripts/parity_diff.py \
        --expected "testdata/expected/synthetic__cellphonedb__p${n}.csv" \
        --actual "target/pipe_out/synthetic__cellphonedb__p${n}.csv" \
        --rtol 0
done
