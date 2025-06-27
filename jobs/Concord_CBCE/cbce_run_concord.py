
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

file_suffix = f"{time.strftime('%b%d-%H%M')}"
ccd.set_verbose_mode(True)
timer = ccd.ul.Timer()
time_log = {}


# Run methdo
feature_list = ccd.ul.select_features(adata, n_top_features=10000, flavor='seurat_v3')
file_suffix = f"{time.strftime('%b%d-%H%M')}"
concord_args = {
        'adata': adata,
        'input_feature': feature_list,
        'batch_size': 128,
        'latent_dim': 300,
        'encoder_dims':[1000],
        'decoder_dims':[1000],
        'augmentation_mask_prob': 0.3, 
        'clr_temperature': 0.5,
        'p_intra_knn': 0.3,
        'sampler_knn': 300,
        'min_p_intra_domain': .95,
        'n_epochs': 15,
        'domain_key': 'dataset3',
        'verbose': False,
        'inplace': False,
        'seed': seed,
        'device': device,
        'save_dir': save_dir
    }

with timer:
    output_key = f'Concord-decoder_{file_suffix}'
    cur_ccd = ccd.Concord(use_decoder=True, **concord_args)
    cur_ccd.encode_adata(input_layer_key='X_log1p', output_key=output_key)
    
time_log[output_key] = timer.interval
# Save the latent embedding to a file, so that it can be loaded later
ccd.ul.save_obsm_to_hdf5(cur_ccd.adata, save_dir / f"obsm_{output_key}_{file_suffix}.h5")

# Save time_log as well using pickle
import pickle
import pickle
with open(save_dir / f"time_log_{output_key}_{file_suffix}.pkl", 'wb') as f:
    pickle.dump(time_log, f)


