"""Run one liana method on one synthetic dataset; prints WALL_S. Wrap with /usr/bin/time -v."""
import sys, os, time, argparse
import numpy as np, pandas as pd, anndata as ad, liana as li

p = argparse.ArgumentParser()
p.add_argument("--method", required=True)
p.add_argument("--n_obs", type=int, required=True)
p.add_argument("--n_jobs", type=int, default=4)
p.add_argument("--n_perms", type=int, default=None)
p.add_argument("--verbose", type=int, default=0)
p.add_argument("--layer", default=None)
a = p.parse_args()

adata = ad.read_h5ad(f"/home/pwwang/p0a/data/sc_{a.n_obs}.h5ad")
resource = pd.read_csv(f"/home/pwwang/p0a/data/resource_{a.n_obs}.csv")

try:
    import numba
    print(f"NUMBA_THREADS={numba.get_num_threads()}", flush=True)
except Exception as e:
    print("NUMBA_THREADS=na", flush=True)

kwargs = dict(groupby="cell_type", resource=resource, use_raw=False,
              verbose=bool(a.verbose), n_jobs=a.n_jobs)
if a.n_perms is not None:
    kwargs["n_perms"] = a.n_perms
if a.layer is not None:
    kwargs["layer"] = a.layer

f = getattr(li.mt, a.method)
print(f"RUN method={a.method} n_obs={a.n_obs} n_jobs={a.n_jobs} "
      f"n_perms={kwargs.get('n_perms', 'default(1000)')}", flush=True)
t0 = time.time()
res = f(adata=adata, **kwargs)
wall = time.time() - t0
print(f"WALL_S={wall:.2f}", flush=True)
if res is None:
    key = "liana_res"
    out = adata.uns.get(key)
    print(f"RES_INPLACE key={key} rows={None if out is None else out.shape[0]}", flush=True)
else:
    print(f"RES_SHAPE={res.shape}", flush=True)
