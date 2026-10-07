#!/bin/bash
# usage: run_one.sh <label> <method> <n_obs> <n_jobs> <timeout_s> [extra run_bench args...]
set -u
cd /home/pwwang/p0a
source venv/bin/activate
label=$1; method=$2; n_obs=$3; n_jobs=$4; tmo=$5; shift 5
echo "=== START $label method=$method n_obs=$n_obs n_jobs=$n_jobs tmo=$tmo extra=$* $(date +%T) ==="
timeout --signal=INT $tmo /usr/bin/time -v python run_bench.py --method $method --n_obs $n_obs --n_jobs $n_jobs "$@" > logs/$label.log 2>&1
rc=$?
echo "=== END $label rc=$rc $(date +%T) ==="
