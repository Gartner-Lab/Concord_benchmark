#!/usr/bin/env python
# coding: utf-8

# ---------------------------------------------
# Benchmark Script for Tabula Sapiens Dataset
# ---------------------------------------------

import os
import time
from pathlib import Path
import warnings

import numpy as np
import torch
import scanpy as sc
import matplotlib as mpl
from scipy import sparse

import concord as ccd

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# ----------------------------
# Configuration
# ----------------------------
proj_name = "TabulaSapiens"
file_name = "TabulaSapiens"
file_suffix = time.strftime('%b%d-%H%M')

save_dir = Path(f"../save/{proj_name}-{file_suffix}")
save_dir.mkdir(parents=True, exist_ok=True)

data_dir = Path(f"../data/{proj_name}")
data_dir.mkdir(parents=True, exist_ok=True)

seed = 0
ccd.ul.set_seed(seed)

# load the data
adata = sc.read_h5ad(data_dir / f"{file_name}.h5ad")

# ### preprocessing

# --------------------- Config ----------------------
MIN_GENES = 300
MIN_CELLS_PER_GENE = 5
N_TOP_HVG = 5000
N_PCS = 300

# ---------------------- Restart from Raw ----------------------
adata = adata.raw.to_adata()
print(f"✅ Restarted from raw: {adata.shape}")

# ---------------------- Add 'counts' Layer ----------------------
# Ensure integer counts and store them
if not np.issubdtype(adata.X.dtype, np.integer):
    adata.X = adata.X.astype("int32")
adata.layers["counts"] = adata.X.copy()
print("✅ Stored integer count matrix in `.layers['counts']`")

# ---------------------- QC Metrics ----------------------
sc.pp.calculate_qc_metrics(adata, inplace=True)

# ---------------------- Cell Filtering ----------------------
n_before = adata.n_obs
adata = adata[adata.obs.n_genes_by_counts > MIN_GENES].copy()
print(f"✅ Filtered low-quality cells: {n_before:,} → {adata.n_obs:,}")

adata.write(data_dir / f"{file_name}_preprocessed.h5ad")

# ---------------------- Gene Filtering ----------------------
n_genes_before = adata.n_vars
sc.pp.filter_genes(adata, min_cells=MIN_CELLS_PER_GENE)
print(f"✅ Filtered low-expressed genes: {n_genes_before:,} → {adata.n_vars:,}")

# ---------------------- Set Filtered Raw ----------------------
adata.raw = adata.copy()
print("✅ Stored filtered data in `.raw`")

# ---------------------- Normalize & Log Transform ----------------------
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print("✅ Normalized total counts and log-transformed")

# ---------------------- Highly Variable Genes ----------------------
sc.pp.highly_variable_genes(
    adata,
    flavor='seurat_v3',
    n_top_genes=N_TOP_HVG,
    subset=False
)
n_hvgs = adata.var.highly_variable.sum()
print(f"✅ Identified {n_hvgs:,} highly variable genes")

# ---------------------- PCA ----------------------
print(f"🔄 Running PCA with n_comps={N_PCS} using highly variable genes...")
sc.tl.pca(
    adata,
    n_comps=N_PCS,
    svd_solver='arpack',
    use_highly_variable=True
)
print("✅ PCA complete. Result stored in `adata.obsm['X_pca']`")

# ---------------------- Subset to HVGs ----------------------
n_vars_before = adata.n_vars
adata = adata[:, adata.var.highly_variable].copy()
print(f"✅ Subset to highly variable genes: {n_vars_before:,} → {adata.n_vars:,}")

# --------------------- Save HVG-subsetted AnnData ----------------------
output_path = data_dir / f"{file_name}_preprocessed_HVG.h5ad"
adata.write(output_path)
print(f"💾 Saved HVG-filtered AnnData to: {output_path}")

print(f"🎉 Preprocessing complete: {adata.shape[0]:,} cells by {adata.shape[1]:,} HVGs")

import subprocess, json
methods = ["unintegrated", "harmony", "scvi", "contrastive", "concord_knn", "concord_hcl"]

# concord_args = {
#     "encoder_dims": [512, 256],
#     "beta": 0.1,
#     "n_epochs": 10
# }

output_dir = '../jobs'
device = 'auto'
conda_env = 'concord_env'
batch_key = 'donor_assay'  # Use the same batch key as in the notebook  
state_key = 'cell_type'
latent_dim = '50'  # Adjust as needed, but should match the encoder_dims in concord_args

mode = 'wynton'

subprocess.run([
    "python", "./generate_py_jobs.py",
    "--proj_name", proj_name,
    "--adata_filename", f"{file_name}_preprocessed_HVG.h5ad",
    "--methods", *methods,
    "--batch_key", batch_key,
    "--state_key", state_key,
    "--latent_dim", latent_dim,
    "--output_dir", output_dir,
    "--device", device,
    "--conda_env", conda_env,
    "--runtime", "04:00:00",
    "--mode", mode,
    # "--concord_kwargs", json.dumps(concord_args)
], check=True)

proj_folder = Path(output_dir) / f"benchmark_{proj_name}"
proj_folder.mkdir(parents=True, exist_ok=True)  # defensive

submit_all = proj_folder / f"submit_all_{proj_name}.sh"

with submit_all.open("w") as f:
    f.write("#!/bin/bash\n")
    f.write("# Auto-generated — sequentially runs every benchmark Python file\n")
    f.write('cd "$(dirname "$0")"\n\n')

    f.write("source ~/.bashrc\n")
    f.write("conda activate concord_env\n\n")    
    f.write("timestamp=$(date +'%m%d-%H%M')\n\n")

    # Set the Python executable to use
    f.write('py_exec="${PYTHON_EXEC:-python}"\n\n')

    for py_file in sorted(proj_folder.glob(f"benchmark_{proj_name}_*.py")):
        base = py_file.stem
        f.write(f"echo '🔄 Running: {py_file.name} (log: {base}_${{timestamp}}.log)'\n")
        f.write(f"${{py_exec}} {py_file.name} > {base}_${{timestamp}}.log 2>&1\n")
        f.write("echo '✅ Done.'\n\n")

submit_all.chmod(0o755)
print(f"📌  Next step: Run “{submit_all}” to execute all batch integration methods sequentially.")