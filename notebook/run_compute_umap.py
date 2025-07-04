#!/usr/bin/env python

import time
import argparse
from pathlib import Path
import scanpy as sc
import concord as ccd
from benchmark_utils import compute_umap_and_save

# ───────────── Parse arguments ─────────────
parser = argparse.ArgumentParser(description="Run UMAP for a given dataset.")
parser.add_argument("--dataset", type=str, required=True, help="Dataset name (prefix of *_final.h5ad)")
parser.add_argument("--proj_name", type=str, default="concord_1-0-3_0603", help="Project name / folder under data/ and save/")
parser.add_argument("--methods", type=str, default="unintegrated,harmony,liger,scanorama,scvi,contrastive,concord_knn,concord_hcl",
                    help="Comma-separated list of integration method keys in obsm")
parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite existing UMAPs")
args = parser.parse_args()

# ───────────── Setup paths and seed ─────────────
proj_name = args.proj_name
data_root = Path(f"../data/{proj_name}")
save_root = Path(f"../save/{proj_name}")
save_root.mkdir(parents=True, exist_ok=True)

seed = 0
ccd.ul.set_seed(seed)

methods = [m.strip() for m in args.methods.split(",")]
file_suffix = time.strftime("%b%d-%H%M")

# ───────────── Load data and run ─────────────
h5ad_file = data_root / f"{args.dataset}_final.h5ad"
if not h5ad_file.exists():
    raise FileNotFoundError(f"[❌] File does not exist: {h5ad_file}")

print(f"📂 Processing dataset: {args.dataset}")
adata = sc.read_h5ad(h5ad_file)

compute_umap_and_save(
    adata=adata,
    methods=methods,
    save_dir=save_root,
    file_suffix=file_suffix,
    data_dir=data_root,
    file_name=args.dataset,
    seed=seed,
    overwrite=args.overwrite,
)
