import scanpy as sc
import Concord as ccd

import time
from pathlib import Path
import torch
proj_name = "concord_treg"
save_dir = f"/wynton/home/gartner/zhuqin/Concord/save/dev_{proj_name}-{time.strftime('%b%d')}/"
save_dir = Path(save_dir)
save_dir.mkdir(parents=True, exist_ok=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
seed = 0
ccd.ul.set_seed(seed)

data_path = "/wynton/home/gartner/zhuqin/Concord_benchmark/data/treg_data/R_treg_graph_ad.h5ad"
adata = sc.read(
    data_path
)
adata.obs['cell_type']=adata.obs['T_subtype_refined'].cat.remove_unused_categories()
adata.obs['batch'] = adata.obs['dataset.ident']

feature_list = ccd.ul.select_features(adata, n_top_features=3000, flavor='seurat_v3')

# Initialize Concord with an AnnData object
cur_ccd = ccd.Concord(adata=adata, input_feature=feature_list, domain_key='batch', device=device)
cur_ccd.encode_adata(input_layer_key='X_log1p', output_key='Concord')

ccd.ul.run_umap(adata, source_key='Concord', umap_key='Concord_UMAP', n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean')

# Plot the UMAP embeddings
color_by = ['batch', 'cell_type'] # Choose which variables you want to visualize
ccd.pl.plot_embedding(
    adata, basis='Concord_UMAP', color_by=color_by, figsize=(10, 5), dpi=600, ncols=2, font_size=6, point_size=3, legend_loc='on data',
    save_path='Concord_UMAP.png'
)

ccd.ul.run_umap(adata, source_key='Concord', umap_key='Concord_UMAP_3D', n_components=3, n_epochs=300, n_neighbors=15, min_dist=0.1, metric='euclidean')

# Plot the 3D UMAP embeddings
col = 'cell_type'
ccd.pl.plot_embedding_3d(
    adata, basis='Concord_UMAP_3D', color_by=col,
    save_path='Concord_UMAP_3D.html',
    point_size=1, opacity=0.8, width=1000, height=800
)
