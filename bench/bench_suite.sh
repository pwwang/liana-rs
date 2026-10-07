#!/usr/bin/env bash
# W7 — the paper's benchmark suite: three arms, one harness, one box.
#
#   rust     bench/engine_bench (liana_core::run::Method — the `liana-rs run`
#            CLI's own dispatch), RAYON_NUM_THREADS = --threads
#   release  liana 2.0.0 in the pinned oracle venv (`/home/pwwang/p0a/venv`)
#   patched  the same venv with PYTHONPATH=/home/pwwang/p0a/patched (W4/W5
#            memory patch), `bench/run_arm.py`
#
# Every run is serial (no two benchmarks overlap — one measurement at a time on
# this box) and wrapped in `/usr/bin/time -v` for peak RSS and whole-process
# elapsed; the rust/release/patched rows additionally carry the in-process wall
# (`wall_s=`, `WALL_S=`). Raw stdout/stderr land in <outdir>/<label>.{out,err};
# one TSV row per run lands in <outdir>/results_<stage>.tsv (rewritten per
# stage, so re-running a stage is idempotent). `bench/collect_results.py`
# assembles the TSVs into bench/results.json.
#
# Usage: bench/bench_suite.sh [stage ...]        stages: t1 t2 t3 t4 t5
#        OUTDIR=... DATA_DIR=... bench/bench_suite.sh [stage ...]
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="${OUTDIR:-$REPO_ROOT/target/bench7}"
data="${DATA_DIR:-/home/pwwang/p0a/data}"
venv_py="${ORACLE_PYTHON:-/home/pwwang/p0a/venv/bin/python}"
patched="${PATCHED_DIR:-/home/pwwang/p0a/patched}"
seed=1337
export PATH="$HOME/.cargo/bin:$PATH"

stages=("$@")
[ ${#stages[@]} -eq 0 ] && stages=(t1 t2 t3 t4 t5)

mkdir -p "$out/resources"
log="$out/suite.log"
eng="$REPO_ROOT/target/release/engine-bench"
cli="$REPO_ROOT/target/release/liana-rs"

say() { echo "=== $(date +%T) $* ===" | tee -a "$log"; }

# --- parsing -----------------------------------------------------------------

field() { # "<space-separated key=value line>" <key>
    echo "$1" | tr ' ' '\n' | grep "^$2=" | head -1 | cut -d= -f2
}

elapsed_s() { # time -v stderr file -> whole-process wall, seconds
    local e
    e=$(grep "Elapsed (wall clock)" "$1" | sed 's/.*): //')
    awk -F: '{ if (NF == 3) printf "%.2f", $1 * 3600 + $2 * 60 + $3; else printf "%.2f", $1 * 60 + $2 }' <<<"$e"
}

rss_kb() { grep -o "Maximum resident set size (kbytes): [0-9]*" "$1" | grep -o "[0-9]*"; }

cmdline() {
    grep "Command being timed" "$1" | sed 's/^[[:space:]]*Command being timed: //; s/^"//; s/"$//'
}

# emit_row <tsv> <label> <arm> <method> <n_obs> <n_lrs> <n_perms> <threads> \
#          <numba> <rows> <wall> <rc> <errfile> <note>
emit_row() {
    local tsv=$1 label=$2 arm=$3 method=$4 n_obs=$5 n_lrs=$6 n_perms=$7 \
        threads=$8 numba=$9 rows=${10} wall=${11} rc=${12} err=${13} note=${14}
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$label" "$arm" "$method" "$n_obs" "$n_lrs" "$n_perms" "$threads" "$numba" \
        "$rows" "$wall" "$(elapsed_s "$err")" "$(rss_kb "$err")" "$rc" "$note" \
        >> "$tsv"
    printf '%-34s rc=%s  %s  wall=%s  elapsed=%s  rss_kb=%s\n' \
        "$label" "$rc" "$(cat "$out/$label.out" 2>/dev/null | head -1)" \
        "$wall" "$(elapsed_s "$err")" "$(rss_kb "$err")" | tee -a "$log"
    # the exact command, for the manifest (one file per stage, rewritten with it)
    printf 'cmd\t%s\t%s\n' "$label" "$(cmdline "$err")" >> "$out/cmds_${label%%_*}.tsv"
}

# --- the three arms ----------------------------------------------------------

# rust_run <tsv> <label> <method> <n_obs> <n_perms> <threads> <resource basename>
rust_run() {
    local tsv=$1 label=$2 method=$3 n_obs=$4 n_perms=$5 threads=$6 resource=$7
    local o="$out/$label.out" e="$out/$label.err"
    RAYON_NUM_THREADS="$threads" timeout 1800 /usr/bin/time -v "$eng" \
        --adata "$data/sc_${n_obs}.h5ad" --resource "$out/resources/$resource" \
        --method "$method" --n-perms "$n_perms" --seed "$seed" \
        > "$o" 2> "$e"
    local rc=$? line
    line=$(cat "$o")
    emit_row "$tsv" "$label" rust "$method" "$n_obs" "$(field "$line" n_lrs)" \
        "$n_perms" "$(field "$line" threads)" "" "$(field "$line" rows)" \
        "$(field "$line" wall_s)" "$rc" "$e" ""
}

# cli_run — the same dispatch through the `liana-rs run` binary, cross-checking
# the harness (rows from the written CSV, wall/RSS from time(1) alone, since the
# CLI prints no timings).
# cli_run <tsv> <label> <method> <n_obs> <n_perms> <threads> <resource basename>
cli_run() {
    local tsv=$1 label=$2 method=$3 n_obs=$4 n_perms=$5 threads=$6 resource=$7
    local o="$out/$label.out" e="$out/$label.err" csv="$out/$label.csv"
    RAYON_NUM_THREADS="$threads" timeout 1800 /usr/bin/time -v "$cli" run \
        --h5ad "$data/sc_${n_obs}.h5ad" --label-key cell_type \
        --resource-file "$out/resources/$resource" \
        --method "$method" --n-perms "$n_perms" --seed "$seed" --threads "$threads" \
        --out "$csv" > "$o" 2> "$e"
    local rc=$? rows="" n_lrs
    n_lrs=$(( $(wc -l < "$out/resources/$resource") - 1 ))
    [ -s "$csv" ] && rows=$(( $(wc -l < "$csv") - 1 ))
    emit_row "$tsv" "$label" rust-cli "$method" "$n_obs" "$n_lrs" "$n_perms" "$threads" \
        "" "$rows" "" "$rc" "$e" "wall=time(1) elapsed only"
}

# py_run <tsv> <label> <arm:release|patched> <method> <n_obs> <n_perms> <n_jobs> <resource basename>
py_run() {
    local tsv=$1 label=$2 arm=$3 method=$4 n_obs=$5 n_perms=$6 n_jobs=$7 resource=$8
    local o="$out/$label.out" e="$out/$label.err" pp=""
    [ "$arm" = patched ] && pp="$patched"
    PYTHONPATH="$pp" timeout 1800 /usr/bin/time -v "$venv_py" \
        "$REPO_ROOT/bench/run_arm.py" \
        --h5ad "$data/sc_${n_obs}.h5ad" --resource "$out/resources/$resource" \
        --method "$method" --n_perms "$n_perms" --n_jobs "$n_jobs" \
        > "$o" 2> "$e"
    local rc=$? wall rows
    wall=$(grep "^WALL_S=" "$o" | cut -d= -f2)
    rows=$(grep "^RES_INPLACE" "$o" | sed 's/.*rows=//')
    emit_row "$tsv" "$label" "$arm" "$method" "$n_obs" \
        "$(field "$(grep '^RUN ' "$o")" n_lrs)" "$n_perms" "$n_jobs" \
        "$(grep -o "^NUMBA_THREADS=[0-9]*" "$o" | cut -d= -f2)" "$rows" "$wall" \
        "$rc" "$e" "wall=WALL_S (method call only)"
}

# --- inputs ------------------------------------------------------------------

# The truncated 50k resources of the law sweep (`head(n_lrs)`, the mem_probe
# convention), the two 10k anchor resources, and t5's 4,620-LR resource.
#
# t5's resource is the consensus resource's *size* (liana ships 4,620 pairs)
# drawn from this synthetic data's gene universe with the paper's own
# `_sample_resource` recipe (`/home/pwwang/p0a/benchmark.py`), the same recipe
# that built `resource_{n_obs}.csv`.  The literal `select_resource('consensus')`
# symbols cannot be used here: they do not exist in the synthetic `Gene{i}`
# var_names, and all three arms reject that dump ("2016 of 2016 resource symbols
# are missing"; rc=101/1 -- the failed run is kept in
# `target/bench7/miss_t5_resource/`).  The law being tested is in
# `n_perms x n_lrs` only, so the resource's size is what this stage varies.
prepare_resources() {
    local n n_obs
    for n_obs in 1000 10000 50000 100000; do
        cp "$data/resource_${n_obs}.csv" "$out/resources/resource_${n_obs}.csv"
    done
    for n in 200 1000 2000; do
        head -n $((n + 1)) "$data/resource_50000.csv" > "$out/resources/resource_50000_lrs${n}.csv"
        head -n $((n + 1)) "$data/resource_10000.csv" > "$out/resources/resource_10000_lrs${n}.csv"
    done
    "$venv_py" -c "
import sys
sys.path.insert(0, '/home/pwwang/p0a')
import anndata as ad
from benchmark import _sample_resource
adata = ad.read_h5ad('$data/sc_50000.h5ad')
r = _sample_resource(adata, n_lrs=4620)
assert len(r) == 4620, len(r)
r.to_csv('$out/resources/resource_50k_lrs4620.csv', index=False)
" || { say "FATAL: 4620-LR resource build failed"; exit 1; }
}

# --- stages ------------------------------------------------------------------

t1() { # the headline: rank_aggregate, 10k/50k/100k x n_perms {100,1000} x 4 threads
    local tsv="$out/results_t1.tsv" n_obs n_perms arm
    : > "$tsv" "$out/cmds_t1.tsv"
    for n_obs in 10000 50000 100000; do
        for n_perms in 100 1000; do
            local tag="ra_${n_obs%000}k_p${n_perms}" res="resource_${n_obs}.csv"
            for arm in rust release patched; do
                say "t1 $tag $arm"
                if [ "$arm" = rust ]; then
                    rust_run "$tsv" "t1_${tag}_rust" rank_aggregate "$n_obs" "$n_perms" 4 "$res"
                else
                    py_run "$tsv" "t1_${tag}_${arm}" "$arm" rank_aggregate "$n_obs" "$n_perms" 4 "$res"
                fi
            done
        done
    done
    say "t1 cli cross-check 50k p1000"
    cli_run "$tsv" "t1_ra_50k_p1000_cli" rank_aggregate 50000 1000 4 resource_50000.csv
}

t2() { # the memory law: peak RSS vs n_perms {10,100,1000} x n_lrs {200,1000,2000} at 50k
    local tsv="$out/results_t2.tsv" n_perms n_lrs arm
    : > "$tsv" "$out/cmds_t2.tsv"
    for n_perms in 10 100 1000; do
        for n_lrs in 200 1000 2000; do
            for arm in rust release patched; do
                say "t2 50k p${n_perms} lrs${n_lrs} $arm"
                if [ "$arm" = rust ]; then
                    rust_run "$tsv" "t2_ra_50k_p${n_perms}_lrs${n_lrs}_rust" rank_aggregate \
                        50000 "$n_perms" 4 "resource_50000_lrs${n_lrs}.csv"
                else
                    py_run "$tsv" "t2_ra_50k_p${n_perms}_lrs${n_lrs}_${arm}" "$arm" rank_aggregate \
                        50000 "$n_perms" 4 "resource_50000_lrs${n_lrs}.csv"
                fi
            done
        done
    done
    # The law's own anchor: the recorded mem_probe configs at 10k (release arm).
    for cfg in "10 2000" "100 2000" "1000 2000" "1000 200"; do
        set -- $cfg
        say "t2 anchor 10k p$1 lrs$2 release"
        py_run "$tsv" "t2_anchor_10k_p$1_lrs$2_release" release rank_aggregate \
            10000 "$1" 4 "resource_10000_lrs$2.csv"
    done
}

t3() { # thread scaling on the headline method, 50k x 1000, 4 threads = t1's point
    local tsv="$out/results_t3.tsv" threads
    : > "$tsv" "$out/cmds_t3.tsv"
    for threads in 1 4 8 32; do
        say "t3 50k p1000 t${threads} rust"
        rust_run "$tsv" "t3_ra_50k_p1000_t${threads}_rust" rank_aggregate \
            50000 1000 "$threads" resource_50000.csv
    done
}

t4() { # startup: the binary vs `python -c "import liana"` (recorded 1.78 s)
    local tsv="$out/results_t4.tsv" i o e rc
    : > "$tsv" "$out/cmds_t4.tsv"
    for i in 1 2 3 4 5; do
        o="$out/t4_lianars_version_r$i.out"; e="$out/t4_lianars_version_r$i.err"
        timeout 60 /usr/bin/time -v "$cli" --version > "$o" 2> "$e"
        rc=$?
        emit_row "$tsv" "t4_lianars_version_r$i" rust-cli none 0 "" "" "" "" "" "" \
            "$rc" "$e" "liana-rs --version: $(cat "$o")"
    done
    for i in 1 2 3 4 5; do
        o="$out/t4_python_import_r$i.out"; e="$out/t4_python_import_r$i.err"
        timeout 300 /usr/bin/time -v "$venv_py" -c "import liana" > "$o" 2> "$e"
        rc=$?
        emit_row "$tsv" "t4_python_import_r$i" python none 0 "" "" "" "" "" "" \
            "$rc" "$e" "python -c 'import liana' (oracle venv)"
    done
    # `/usr/bin/time -v` resolves only to 10 ms, and the binary starts in less
    # than that: 20 back-to-back `--version` runs under bash's own clock.
    local n=20 t_start t_end mean
    t_start=$EPOCHREALTIME
    for ((i = 0; i < n; i++)); do "$cli" --version > /dev/null; done
    t_end=$EPOCHREALTIME
    mean=$(awk -v a="$t_start" -v b="$t_end" -v n="$n" 'BEGIN { printf "%.6f", (b - a) / n }')
    echo "mean_s=$mean n=$n" > "$out/t4_lianars_version_hr.out"
    emit_row "$tsv" "t4_lianars_version_hr" rust-cli none 0 "" "" "" "" "" "$mean" \
        0 "$out/t4_lianars_version_r1.err" "mean of $n invocations; elapsed/rss from r1"
    say "t4 liana-rs --version mean of $n: ${mean}s"
    # first run of a real command on the smallest dataset, end to end
    cli_run "$tsv" "t4_firstrun_connectome_1k" connectome 1000 1000 1 "resource_1000.csv"
}

t5() { # 4,620-LR resource, 50k x 1000: the release arm's ~23 GB bet
    local tsv="$out/results_t5.tsv" arm
    : > "$tsv" "$out/cmds_t5.tsv"
    say "t5 free before: $(free -m | sed -n 2p)"
    for arm in rust patched release; do
        say "t5 4620 LRs 50k p1000 $arm"
        if [ "$arm" = rust ]; then
            rust_run "$tsv" "t5_ra_50k_p1000_lrs4620_rust" rank_aggregate \
                50000 1000 4 resource_50k_lrs4620.csv
        elif [ "$arm" = patched ]; then
            py_run "$tsv" "t5_ra_50k_p1000_lrs4620_patched" patched rank_aggregate \
                50000 1000 4 resource_50k_lrs4620.csv
        else
            # 40 GB address-space cap: a memory miss becomes a clean MemoryError
            # instead of an OOM kill on the box. The law's prediction is ~23 GB.
            ( ulimit -v $((40 * 1024 * 1024)); py_run "$tsv" \
                "t5_ra_50k_p1000_lrs4620_release" release rank_aggregate \
                50000 1000 4 resource_50k_lrs4620.csv )
        fi
    done
    say "t5 free after: $(free -m | sed -n 2p)"
}

# --- main --------------------------------------------------------------------

say "suite start: stages=${stages[*]} out=$out"
say "build: cargo build --release -p engine-bench -p liana-rs"
cargo build --release -p engine-bench -p liana-rs --manifest-path "$REPO_ROOT/Cargo.toml" \
    >> "$log" 2>&1 || { say "FATAL: build failed"; exit 1; }
prepare_resources

for stage in "${stages[@]}"; do
    say "$stage begin"
    "$stage"
    say "$stage done"
done
say "suite done"
