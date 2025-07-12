#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path
import scanpy as sc
from benchmark_utils import add_embeddings

# ---------------------------- Settings ----------------------------

proj_name = "endothelium_subset_TabulaSapiens"
file_name = "endothelium_subset_TabulaSapiens"
data_dir = Path(f"../data/{proj_name}")
# input_path is the data with previous embeddings
input_path = data_dir / f"{file_name}_final.h5ad"
# output_path is the data with extended embeddings
output_path = data_dir / f"{file_name}_final_extend.h5ad"

new_methods = ['concord_knn', 'concord_hcl', 'contrastive']

# ---------------------------- Load ----------------------------

if not input_path.exists():
    raise FileNotFoundError(f"❌ Cannot find final file: {input_path}")

adata = sc.read_h5ad(input_path)
print(f"✅ Loaded: {input_path} with {adata.shape[0]} cells")

# ---------------------------- Add Embeddings ----------------------------

adata = add_embeddings(adata, proj_name=proj_name, methods=new_methods)

# ---------------------------- Save ----------------------------

adata.write_h5ad(output_path)
print(f"✅ Extended AnnData saved to: {output_path}")
