"""One liana method on one `.h5ad` + one resource CSV, through whichever `liana`
is importable — the Python arms of `bench/bench_suite.sh`.

`bench/run_bench.py` (imported verbatim from the scratch workspace) hardcodes
`/home/pwwang/p0a/data/sc_{n_obs}.h5ad` + `resource_{n_obs}.csv`; this sibling
takes the two paths, which is what the memory-law sweep (truncated resources)
and the consensus-resource run need. Same measurement contract as `run_bench.py`:
`WALL_S` is the method call only (inputs already read) and `RES_INPLACE rows=` is
the reassembled result's row count. Wrap in `/usr/bin/time -v` for peak RSS.

The release arm is the pinned oracle venv; the patched arm adds
`PYTHONPATH=/home/pwwang/p0a/patched` (the W4/W5 memory patch). Which one ran is
printed as `LIANA_MODULE=`, next to `NUMBA_THREADS` — the numba kernels' own
thread count, which `n_jobs` does not set (W4 D4).
"""
import argparse, time

import anndata as ad
import liana as li
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--h5ad", required=True)
p.add_argument("--resource", required=True)
p.add_argument("--method", default="rank_aggregate")
p.add_argument("--n_jobs", type=int, default=4)
p.add_argument("--n_perms", type=int, default=1000)
a = p.parse_args()

adata = ad.read_h5ad(a.h5ad)
resource = pd.read_csv(a.resource)

import numba

print(f"NUMBA_THREADS={numba.get_num_threads()}", flush=True)
print(f"LIANA_MODULE={li.__file__}", flush=True)
print(
    f"RUN method={a.method} h5ad={a.h5ad} resource={a.resource} "
    f"n_lrs={len(resource)} n_jobs={a.n_jobs} n_perms={a.n_perms}",
    flush=True,
)

f = getattr(li.mt, a.method)
t0 = time.time()
res = f(
    adata=adata,
    groupby="cell_type",
    resource=resource,
    use_raw=False,
    verbose=False,
    n_jobs=a.n_jobs,
    n_perms=a.n_perms,
)
wall = time.time() - t0
print(f"WALL_S={wall:.2f}", flush=True)
out = adata.uns.get("liana_res") if res is None else res
print(f"RES_INPLACE rows={None if out is None else out.shape[0]}", flush=True)
