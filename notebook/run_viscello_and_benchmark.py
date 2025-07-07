#!/usr/bin/env python

import argparse
import time
from pathlib import Path
import scanpy as sc
import pandas as pd
import pickle
import concord as ccd

# ---------------------------- Parse CLI ----------------------------
parser = argparse.ArgumentParser(description="Run VisCello export and benchmark for a given project")
parser.add_argument("--proj_name", required=True, help="Short name of the dataset (e.g. dkd_Wilson, HypoMap_Steuernagel)")
parser.add_argument("--file_name", required=True, help="Base filename (e.g. dkd_Wilson, HypoMap_Steuernagel). This should match the prefix of your .h5ad file.")
parser.add_argument("--data_version_folder", required=True, help="Name of the subfolder within 'data/' containing the .h5ad files (e.g. concord_1-0-3_0603)")
parser.add_argument("--organism", default="hsa", choices=["hsa", "mmu"], help="Organism code (default: hsa)")
parser.add_argument("--batch_key", required=True, help="obs column indicating batch (e.g. donor_id)")
parser.add_argument("--state_key", required=True, help="obs column for biological state (e.g. cell_type)")
parser.add_argument("--methods", nargs="+", required=True, help="List of embedding keys to benchmark")
parser.add_argument("--file_suffix", default=None, help="Optional suffix (e.g. Jul06-1800). If not provided, auto-generated")
args = parser.parse_args()

# ---------------------------- Set up paths ----------------------------
file_suffix = args.file_suffix or time.strftime('%b%d-%H%M')

notebook_dir = Path(__file__).parent.resolve()
project_root = notebook_dir.parent

data_input_base_dir = (project_root / "data" / args.data_version_folder).resolve()
adata_path = data_input_base_dir / f"{args.file_name}_final.h5ad"

save_dir = (project_root / f"save/{args.proj_name}-{file_suffix}").resolve()
viscello_out_dir = save_dir / "viscello_exports" / f"cello_{args.proj_name}_{file_suffix}"
benchmark_pkl_path = save_dir / f"benchmark_{args.state_key}_{file_suffix}.pkl"

# ---------------------------- Create output folders ----------------------------
# The data_input_base_dir is an input directory, so no need to create it here.
# It is assumed to already exist.
save_dir.mkdir(parents=True, exist_ok=True)
viscello_out_dir.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------- Step 1: Load AnnData ----------------------------
print(f"📂 Loading AnnData from: {adata_path}")
try:
    adata = sc.read_h5ad(adata_path)
except FileNotFoundError:
    print(f"ERROR: AnnData file not found at {adata_path}")
    print(f"Please ensure '{args.data_version_folder}' exists in 'data/' and '{args.file_name}_final.h5ad' is inside it.")
    exit(1) # Exit the script if the file is not found
except Exception as e:
    print(f"An error occurred while loading AnnData: {e}")
    exit(1)

# ---------------------------- Step 2: Update var_names ----------------------------
if "feature_name" in adata.var.columns:
    adata.var_names = pd.Index(adata.var["feature_name"].astype(str))
    adata.var_names_make_unique()
    print("✅ Updated var_names from 'feature_name'")
else:
    print("⚠️  'feature_name' column missing in adata.var. Skipping rename.")

# ---------------------------- Step 3: VisCello Export ----------------------------
print(f"🚀 Starting VisCello export for project: {args.proj_name}")
ccd.ul.anndata_to_viscello(
    adata,
    viscello_out_dir,
    project_name=args.proj_name,
    organism=args.organism
)
print(f"✅ VisCello exported to: {viscello_out_dir}")

# ---------------------------- Step 4: Benchmark ----------------------------
print(f"📊 Running benchmark pipeline for state_key='{args.state_key}' and batch_key='{args.batch_key}'")
out = ccd.bm.run_benchmark_pipeline(
    adata,
    embedding_keys=args.methods,
    state_key=args.state_key,
    batch_key=args.batch_key,
    save_dir=save_dir / "benchmarks_celltype", # Creates a subdirectory for benchmark specific outputs
    file_suffix=file_suffix,
    run=("scib", "probe"), # Specify which benchmark modules to run
    plot_individual=False, # Set to True if you want individual plots for each metric/embedding
)

try:
    with open(benchmark_pkl_path, "wb") as f:
        pickle.dump(out, f)
    print(f"✅ Benchmark results saved to: {benchmark_pkl_path}")
except Exception as e:
    print(f"ERROR: Could not save benchmark results to {benchmark_pkl_path}: {e}")
    exit(1)

print("\n✨ Script finished successfully! ✨")