
import numpy as np
import scanpy as sc
import time
from pathlib import Path
import torch
import Concord as ccd
import warnings
warnings.filterwarnings('ignore')

print("Import successful")

proj_name = "cbce_1217"
data_dir = Path('../../data/CE_CB/')
save_dir = f"../../save/dev_{proj_name}-{time.strftime('%b%d')}/"
save_dir = Path(save_dir)
save_dir.mkdir(parents=True, exist_ok=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
seed = 0
ccd.ul.set_seed(seed)

file_suffix = f"{time.strftime('%b%d-%H%M')}"


adata = sc.read(data_dir / "adata_cbce_Dec17-1824.h5ad")
adata.obsm = ccd.ul.load_obsm_from_hdf5(Path('../../save/dev_cbce_1217-Dec18/') / f"obsm_Dec18-1227.h5")
print("Data loading successful")

file_suffix = f"{time.strftime('%b%d-%H%M')}"

ccd.set_verbose_mode(True)
timer = ccd.ul.Timer()
time_log = {}


# Run methods
feature_list = ccd.ul.select_features(adata, n_top_features=10000, flavor='seurat_v3')
adata = adata[:,feature_list].copy()
output_key = 'Liger'
with timer:
    ccd.ul.run_liger(adata, batch_key="dataset3", count_layer="counts", output_key=output_key, k=300, return_corrected=False)

time_log[output_key] = timer.interval
ccd.ul.save_obsm_to_hdf5(adata, save_dir / f"obsm_{output_key}_{file_suffix}.h5")
    
# Save time_log as well using pickle
import pickle
import pickle
with open(save_dir / f"time_log_{output_key}_{file_suffix}.pkl", 'wb') as f:
    pickle.dump(time_log, f)

