
import numpy as np
import scanpy as sc
import time
from pathlib import Path
import torch
import Concord as ccd
import warnings
warnings.filterwarnings('ignore')

print("Import successful")

proj_name = "benchmark_Huycke"
save_dir = f"../../save/dev_{proj_name}-{time.strftime('%b%d')}/"
save_dir = Path(save_dir)
save_dir.mkdir(parents=True, exist_ok=True)
seed = 0
ccd.ul.set_seed(seed)

file_suffix = f"{time.strftime('%b%d-%H%M')}"


data_dir = Path('../../data/intestine_dev/')
data_path = data_dir / 'adata_huycke_Jan05-1211.h5ad'
adata = sc.read(
    data_path
)

print("Data loading successful")
#ccd.ul.anndata_to_viscello(adata, data_dir / f"cello_{proj_name}_{file_suffix}", project_name = proj_name, organism='mmu')

unique_broad_cell_types = adata.obs['broad_cell_type'].unique()
import re
adata_subsets = {}
for ct in unique_broad_cell_types:
    sanitized_ct = re.sub(r'[^\w\-]', '_', ct)
    adata_subset = sc.read(data_dir / f"adata_huycke_{sanitized_ct}_Jan05-1844.h5ad")
    adata_subsets[sanitized_ct] = adata_subset

viscello_dir = str(data_dir / "cello_benchmark_Huycke_Jan05-2033")
ccd.ul.update_clist_with_subsets(global_adata = adata, adata_subsets = adata_subsets, viscello_dir = viscello_dir)
print("Conversion successful")