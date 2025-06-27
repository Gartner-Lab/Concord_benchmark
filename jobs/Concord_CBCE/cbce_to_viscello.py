import numpy as np
import scanpy as sc
import time
from pathlib import Path
import torch
import Concord as ccd
import warnings
warnings.filterwarnings('ignore')
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib import font_manager, rcParams
custom_rc = {
    'font.family': 'Arial',  # Set the desired font for this plot
}

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['pdf.fonttype'] = 42


proj_name = "benchmark_CBCE"
data_dir = Path('../../data/CE_CB/')
save_dir = f"../../save/dev_{proj_name}-{time.strftime('%b%d')}/"
save_dir = Path(save_dir)
save_dir.mkdir(parents=True, exist_ok=True)
#device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
device = torch.device('mps')
print(device)
seed = 0
ccd.ul.set_seed(seed)
file_suffix = 'Jan30-1028'

adata = sc.read(data_dir / "adata_cbce_Jan30-1028.h5ad")

ct_dict = {
    'Neuron_ASE_ASJ_AUA': data_dir/'adata_subsub_aseasjaua_Jan30-1028.h5ad',
    'AB_nonpharynx': data_dir/'adata_cbce_Dec26-1019_AB broad.h5ad',
    'Mesoderm_nonpharynx': data_dir/'adata_cbce_Dec21-0244_Mesoderm.h5ad',
    'Pharynx': data_dir/'adata_cbce_Dec23-1049_Pharynx.h5ad',
    'Intestine': data_dir/'adata_cbce_Dec21-0244_Intestine.h5ad',
    'Early200min': data_dir/'adata_cbce_Dec23-1707_early200.h5ad',
}

# Load adata subsets into dict
adata_subsets = {}
for ct, path in ct_dict.items():
    adata_subsets[ct] = sc.read(path)

viscello_dir = str(data_dir / f"cello_{proj_name}_Jan30-1028")

ccd.ul.update_clist_with_subsets(global_adata = adata, adata_subsets = adata_subsets, viscello_dir = viscello_dir)



