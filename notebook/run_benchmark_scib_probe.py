#!/usr/bin/env python
"""run_benchmark_scib_probe.py

Benchmark multiple embeddings.

Typical usage (example):

python run_benchmark_scib_probe.py \
  --proj_name dkd_Wilson \
  --file_name dkd_Wilson \
  --batch_key donor_id \
  --state_key cell_type \
  --methods unintegrated harmony liger scanorama scvi

* ``--proj_name`` is used to locate the subfolder under ``data/``
* ``--file_name`` should match the base name (no suffix) of the `.h5ad` file.
"""

import argparse
import time
from pathlib import Path
import scanpy as sc
import pandas as pd
import pickle
import concord as ccd

# -----------------------------------------------------------------------------
# Parse CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Benchmark embeddings."
)

parser.add_argument(
    "--proj_name",
    required=True,
    help="Folder under data/ that contains the .h5ad file (e.g. dkd_Wilson)",
)
parser.add_argument(
    "--file_name",
    required=True,
    help="Base name of the .h5ad file (e.g. dkd_Wilson, without '_final.h5ad')",
)
parser.add_argument(
    "--batch_key",
    required=True,
    help="obs column indicating batch (e.g. donor_id)",
)
parser.add_argument(
    "--state_key",
    required=True,
    help="obs column for biological state (e.g. cell_type)",
)
parser.add_argument(
    "--methods",
    nargs="+",
    required=True,
    help="Space-separated list of embedding keys to benchmark",
)
parser.add_argument(
    "--file_suffix",
    default=None,
    help="Optional run suffix (e.g. Jul06-1800). If omitted, auto-generated",
)

args = parser.parse_args()

# -----------------------------------------------------------------------------
# Paths & folders
# -----------------------------------------------------------------------------
file_suffix = args.file_suffix or time.strftime("%b%d-%H%M")

script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent

# File lives in: <project_root>/data/<proj_name>/<file_name>_final.h5ad
data_dir = (project_root / "data" / args.proj_name).resolve()
adata_path = data_dir / f"{args.file_name}_final.h5ad"

# Output locations
save_dir = project_root / "save" / f"{args.proj_name}-{file_suffix}"

benchmark_pkl_path = save_dir / f"benchmark_{args.state_key}_{file_suffix}.pkl"

# -----------------------------------------------------------------------------
# Create output folders
# -----------------------------------------------------------------------------
save_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Step 1  Load AnnData
# -----------------------------------------------------------------------------
print(f"📂 Loading AnnData from: {adata_path}")
try:
    adata = sc.read_h5ad(adata_path)
except FileNotFoundError:
    print("❌ AnnData file not found.")
    print(
        f"Expected: {adata_path}\n"
        "Make sure your .h5ad file is under 'data/<proj_name>/' and named '<file_name>_final.h5ad'."
    )
    exit(1)
except Exception as e:
    print(f"❌ An error occurred while loading AnnData: {e}")
    exit(1)
adata.obs[args.batch_key] = adata.obs[args.batch_key].astype(str).astype("category")

# -----------------------------------------------------------------------------
# Step 4  Benchmark
# -----------------------------------------------------------------------------
print(
    f"📊 Running benchmark: state_key='{args.state_key}', "
    f"batch_key='{args.batch_key}', embeddings={args.methods}"
)
out = ccd.bm.run_benchmark_pipeline(
    adata,
    embedding_keys=args.methods,
    state_key=args.state_key,
    batch_key=args.batch_key,
    save_dir=save_dir / "benchmarks_celltype",
    file_suffix=file_suffix,
    run=("scib", "probe"),
    plot_individual=False,
)

# Save results
try:
    with open(benchmark_pkl_path, "wb") as f:
        pickle.dump(out, f)
    print(f"✅ Benchmark results saved → {benchmark_pkl_path}")
except Exception as e:
    print(f"❌ Could not save benchmark results: {e}")
    exit(1)

print("\n✨ All done! ✨")
