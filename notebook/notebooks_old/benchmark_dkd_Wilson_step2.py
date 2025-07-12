#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path
import scanpy as sc
from benchmark_utils import add_embeddings

# ---------------------------- Settings ----------------------------

proj_name = "dkd_Wilson"
file_name = "dkd_Wilson"
data_dir = Path(f"../data/{proj_name}")
# output_path is the preprocessed data from step 1 or data with embeddings already added
output_path = data_dir / f"{file_name}_preprocessed.h5ad"
# final_path is the final output with new embeddings added
final_path = data_dir / f"{file_name}_final.h5ad"

methods = ['scvi', 'harmony', 'scanorama', 'liger', 'unintegrated']

# ---------------------------- Load Preprocessed Data ----------------------------

if not output_path.exists():
    raise FileNotFoundError(f"❌ Cannot find preprocessed file: {output_path}")

adata = sc.read_h5ad(output_path)
print(f"✅ Loaded: {output_path} with {adata.shape[0]} cells")

# ---------------------------- Add Embeddings ----------------------------

adata = add_embeddings(adata, proj_name=proj_name, methods=methods)

# ---------------------------- Save Final AnnData ----------------------------

adata.write_h5ad(final_path)
print(f"✅ Final AnnData saved to: {final_path}")
