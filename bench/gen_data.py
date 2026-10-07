"""Generate synthetic data with the LIANA+ manuscript's own recipe (benchmark.py)."""
import sys, os, time
import numpy as np, pandas as pd, scanpy as sc, anndata as ad

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from benchmark import _sample_anndata, _sample_resource

n_obs = int(sys.argv[1])
n_lrs = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
outdir = "/home/pwwang/p0a/data"

t0 = time.time()
adata = _sample_anndata(n_obs=n_obs)
t_gen = time.time() - t0
print(f"GEN_ADATA_S={t_gen:.1f}", flush=True)

resource = _sample_resource(adata, n_lrs=n_lrs)
print(f"GEN_RESOURCE_S={time.time()-t0:.1f}", flush=True)

adata.write_h5ad(f"{outdir}/sc_{n_obs}.h5ad")
resource.to_csv(f"{outdir}/resource_{n_obs}.csv", index=False)
print(f"TOTAL_S={time.time()-t0:.1f}", flush=True)
print(f"SHAPE={adata.shape} NNZ={adata.X.nnz} "
      f"NNZ_FRAC={adata.X.nnz/(adata.shape[0]*adata.shape[1]):.4f}", flush=True)
print("CELLTYPE_COUNTS=" + str(adata.obs["cell_type"].value_counts().to_dict()), flush=True)
print(f"RESOURCE_ROWS={resource.shape[0]}", flush=True)
