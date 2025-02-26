
import numpy as np
import scanpy as sc
import time
from pathlib import Path
import torch
import Concord as ccd
import warnings
warnings.filterwarnings('ignore')

print("Import successful")

proj_name = "benchmark_CBCE"
data_dir = Path('../../data/CE_CB/')
save_dir = f"../../save/dev_{proj_name}-{time.strftime('%b%d')}/"
save_dir = Path(save_dir)
save_dir.mkdir(parents=True, exist_ok=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
seed = 0
ccd.ul.set_seed(seed)
file_suffix = f"{time.strftime('%b%d-%H%M')}"


adata = sc.read(data_dir / "adata_cbce_Dec19-0624.h5ad")
print("Data loading successful")

file_suffix = f"{time.strftime('%b%d-%H%M')}"


adata.obsm['Concord'] = adata.obsm['Concord_Dec17-0930']
adata.obsm['Concord_UMAP'] = adata.obsm['Concord_Dec17-0930_UMAP']
adata.obsm['Concord_UMAP_3D'] = adata.obsm['Concord_Dec17-0930_UMAP_3D']
adata.obsm['Concord-decoder'] = adata.obsm['Concord-decoder_Dec18-1358']
adata.obsm['Concord-decoder_UMAP'] = adata.obsm['Concord-decoder_Dec18-1358_UMAP']
adata.obsm['Concord-decoder_UMAP_3D'] = adata.obsm['Concord-decoder_Dec18-1358_UMAP_3D']
adata.obsm['Seurat'] = adata.obsm['integrated.rpca']
adata.obsm['Seurat_UMAP'] = adata.obsm['umap.rpca']
adata.obsm['Unintegrated_UMAP'] = adata.obsm['Unintegrated_umap']
del adata.obsm['Unintegrated_umap']

adata.obs['ct_or_lin'] = adata.obs['cell_type'].astype(str)
adata.obs['ct_or_lin'][adata.obs['cell_type']=='unassigned'] = adata.obs['lineage_complete'][adata.obs['cell_type']=='unassigned'].astype(str)
adata.obs['ct_or_lin'][adata.obs['ct_or_lin']=='unassigned'] = np.NaN
adata.obs['lin_or_ct'] = adata.obs['lineage_complete'].astype(str)
adata.obs['lin_or_ct'][adata.obs['lineage_complete']=='unassigned'] = adata.obs['cell_type'][adata.obs['lineage_complete']=='unassigned'].astype(str)
adata.obs['lin_or_ct'][adata.obs['lin_or_ct']=='unassigned'] = np.NaN

adata.obs['cell_type'][adata.obs['cell_type']=='unassigned'] = np.NaN
adata.obs['lineage_complete'][adata.obs['lineage_complete']=='unassigned'] = np.NaN


concord_keys = ["Concord", 'Concord-decoder']
other_keys = ["Unintegrated", "Scanorama", "Liger", "Harmony", "scVI", "Seurat"]
combined_keys = other_keys + concord_keys

# Remove cells which has np.NaN in 'ct_or_lin'
adata = adata[adata.obs['ct_or_lin'].astype(str) != 'NaN']

adata.obs['ct_or_lin'] = adata.obs['ct_or_lin'].astype(str)
adata.obs['dataset3'] = adata.obs['dataset3'].astype(str)

from scib_metrics.benchmark import Benchmarker
bm = Benchmarker(
    adata,
    batch_key='dataset3',
    label_key='ct_or_lin',
    embedding_obsm_keys=combined_keys,
    n_jobs=4,
)
bm.benchmark()

scib_scores = bm.get_results(min_max_scale=False)
scib_scores.to_csv(save_dir / f"scib_scores_{file_suffix}.csv")
import pickle
with open(save_dir / f"bm_{file_suffix}.pkl", 'wb') as f:
    pickle.dump(bm, f)

