#!/usr/bin/env python
# coding: utf-8

# This is an extension of step1 to run additional methods.
import os
import time
import numpy as np
import scanpy as sc
import torch
from pathlib import Path
from scipy import sparse
import warnings
import subprocess
import concord as ccd
from benchmark_utils import add_embeddings, run_scib_benchmark

warnings.filterwarnings('ignore')

# ---------------------------- Settings ----------------------------

proj_name = "endothelium_subset_TabulaSapiens"
file_name = "endothelium_subset_TabulaSapiens"
file_suffix = time.strftime('%b%d-%H%M')
seed = 0

save_dir = Path(f"../save/{proj_name}")
save_dir.mkdir(parents=True, exist_ok=True)

data_dir = Path(f"../data/{proj_name}")
data_dir.mkdir(parents=True, exist_ok=True)

ccd.ul.set_seed(seed)

# ---------------------------- Load and Preprocess ----------------------------

# adata = sc.read_h5ad(data_dir / f"{file_name}.h5ad")

# batch_key = 'donor_id'
# state_key = 'cell_type'

# sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="cell_ranger", batch_key=batch_key)

# if adata.raw is not None:
#     raw_X = adata.raw[:, adata.var_names].X
#     dense_X = raw_X.toarray() if sparse.issparse(raw_X) else raw_X

#     if np.allclose(dense_X, np.round(dense_X)):
#         adata.layers["counts"] = raw_X.astype(np.int32)
#         print("✅ counts layer set from adata.raw (converted to int32)")
#     else:
#         raise ValueError("❌ adata.raw.X contains non-integer-like values.")
# else:
#     raise ValueError("❌ adata.raw is None — cannot set counts layer.")

# adata = adata[:, adata.var.highly_variable].copy()
# adata = adata[adata.layers["counts"].sum(axis=1) > 0, :]

# sc.tl.pca(adata, n_comps=30, use_highly_variable=True)
# adata.write_h5ad(data_dir / f"{file_name}_preprocessed.h5ad")
# print(f"✅ Preprocessed data saved to {data_dir / f'{file_name}_preprocessed.h5ad'}")

adata = sc.read_h5ad(data_dir / f"{file_name}_preprocessed.h5ad")
print(f"✅ Loaded preprocessed data from {data_dir / f'{file_name}_preprocessed.h5ad'}")

batch_key = 'donor_id'
state_key = 'cell_type'

# ---------------------------- Create Integration Jobs ----------------------------

# methods = ["scvi", "harmony", "scanorama", "liger", "unintegrated"]
methods = ["concord_knn", "concord_hcl", "contrastive"]
output_dir = '../jobs'
device = 'auto'
conda_env = 'scenv'

subprocess.run([
    "python", "../notebook/generate_py_sh_jobs.py",
    "--proj_name", proj_name,
    "--adata_filename", f"{file_name}_preprocessed.h5ad",
    "--methods", *methods,
    "--batch_key", batch_key,
    "--state_key", state_key,
    "--output_dir", output_dir,
    "--device", device,
])

# ---------------------------- Submit Jobs ----------------------------

job_dir = Path(output_dir) / f"benchmark_{proj_name}"
method_list = methods

for method in method_list:
    sh_path = job_dir / f"benchmark_{proj_name}_{method}.sh"
    if sh_path.exists():
        print(f"✅ Submitting: {sh_path}")
        os.system(f"qsub {sh_path}")
    else:
        print(f"⚠️  Skipped: {sh_path} not found")

# ---------------------------- Collect Results ----------------------------

# adata = add_embeddings(adata, proj_name=proj_name, methods=method_list)
# adata.write_h5ad(data_dir / f"{file_name}_final.h5ad")

# ---------------------------- Evaluate Embeddings ----------------------------

# embedding_keys = [f"X_{m}" for m in method_list]
# bm = run_scib_benchmark(
#     adata=adata,
#     embedding_keys=embedding_keys,
#     batch_key=batch_key,
#     label_key=state_key,
#     n_jobs=4
# )

# bm.plot_results_table()
# bm.plot_results_table(min_max_scale=False)

# ccd.pl.plot_all_embeddings(
#     adata=adata,
#     combined_keys=[f"{m}_UMAP" for m in method_list if m != "concord"],
#     color_bys=[batch_key, state_key],
#     basis_types=["UMAP"],
# )

# df = bm.get_results(min_max_scale=False)
# print(df)
