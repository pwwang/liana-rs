#!/usr/bin/env bash
# The T3 sweep: the full pipeline on the p0a benchmark datasets, at 1/4/8/32
# threads (RAYON_NUM_THREADS) and the n_perms the acceptance names.
#
# Each run is wrapped in `/usr/bin/time -v` — its "Maximum resident set size"
# cross-checks the bin's own `/proc/self/status` VmHWM — and prints one line:
# the bin's `method=... wall_s=... rss_kb=...` fields plus the time(1) peak.
# Raw stdout/stderr land in <outdir>/<label>.{out,err}.
#
# Usage: bench/run_engine_bench.sh [outdir] [data_dir]
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="${1:-$REPO_ROOT/target/bench}"
data="${2:-/home/pwwang/p0a/data}"
export PATH="$HOME/.cargo/bin:$PATH"

mkdir -p "$out"
cargo build --release -p engine-bench --manifest-path "$REPO_ROOT/Cargo.toml" || exit 1
bin="$REPO_ROOT/target/release/engine-bench"

run() { # label method n_obs n_perms threads
    local label=$1 method=$2 n_obs=$3 n_perms=$4 threads=$5
    RAYON_NUM_THREADS=$threads timeout 3600 /usr/bin/time -v "$bin" \
        --adata "$data/sc_${n_obs}.h5ad" --resource "$data/resource_${n_obs}.csv" \
        --method "$method" --n-perms "$n_perms" --seed 1337 \
        > "$out/$label.out" 2> "$out/$label.err"
    local peak
    peak=$(grep "Maximum resident set size" "$out/$label.err" | grep -o "[0-9]*")
    printf '%-24s %s  time_v_rss_kb=%s\n' "$label" "$(cat "$out/$label.out")" "$peak"
}

# thread sweep on the permutation path: the 1000-perm run, with the 1-perm run
# as the same-thread prep/read reference (`perms_s` = the difference)
for threads in 1 4 8 32; do
    run "cpd50k_p1000_t$threads" cellphonedb 50000 1000 "$threads"
    run "cpd50k_p1_t$threads" cellphonedb 50000 1 "$threads"
done

# n_perms flatness (100 vs the t4 1000 above) and repeat spread at t4
run cpd50k_p100_t4 cellphonedb 50000 100 4
run cpd50k_p1000_t4_rep2 cellphonedb 50000 1000 4
run cpd50k_p1000_t4_rep3 cellphonedb 50000 1000 4

# the 10k point of the scaling curve
run cpd10k_p1000_t4 cellphonedb 10000 1000 4
run cpd10k_p1_t4 cellphonedb 10000 1 4

# the trimean path end-to-end (cellchat), and its flatness
run cc50k_p1000_t4 cellchat 50000 1000 4
run cc50k_p100_t4 cellchat 50000 100 4
