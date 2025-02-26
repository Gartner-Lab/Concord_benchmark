
import numpy as np
import scanpy as sc
import time
from pathlib import Path
import torch
import Concord as ccd
import warnings
warnings.filterwarnings('ignore')

print("Import successful")

proj_name = "huycke_1230"
save_dir = f"../../save/dev_{proj_name}-{time.strftime('%b%d')}/"
save_dir = Path(save_dir)
#save_dir.mkdir(parents=True, exist_ok=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
seed = 0
ccd.ul.set_seed(seed)

file_suffix = f"{time.strftime('%b%d-%H%M')}"


data_dir = Path('../../data/intestine_dev/')
#data_path = data_dir / "intestine_adata_processed_concord_Huycke_intestine_Nov20.h5ad"
data_path = data_dir/'adata_huycke_Jan04-1936.h5ad'
adata = sc.read(
    data_path
)

print("Data loading successful")


file_suffix = f"{time.strftime('%b%d-%H%M')}"
ccd.set_verbose_mode(True)
timer = ccd.ul.Timer()
time_log = {}


# Run methdods
feature_list = ccd.ul.select_features(adata, n_top_features=10000, flavor='seurat_v3')
concord_args = {
        'adata': adata,
        'input_feature': feature_list,
        'batch_size': 64,
        'latent_dim': 50,
        'encoder_dims':[300],
        'decoder_dims':[300],
        'augmentation_mask_prob': 0.3, 
        'clr_temperature': 0.3,
        'p_intra_knn': 0.3,
        'sampler_knn': None,
        'min_p_intra_domain': 1.0,
        'n_epochs': 10,
        'domain_key': 'LaneID',
        'verbose': False,
        'inplace': False,
        'seed': seed,
        'device': device,
        'save_dir': save_dir
    }

with timer:
    output_key = f'Concord_{file_suffix}'
    cur_ccd = ccd.Concord(use_decoder=False, **concord_args)
    cur_ccd.encode_adata(input_layer_key='X_log1p', output_key=output_key)

time_log[output_key] = timer.interval

# Save the latent embedding to a file, so that it can be loaded later
ccd.ul.save_obsm_to_hdf5(cur_ccd.adata, save_dir / f"obsm_{output_key}_{file_suffix}.h5")

# Save time_log as well using pickle
import pickle
with open(save_dir / f"time_log_{output_key}_{file_suffix}.pkl", 'wb') as f:
    pickle.dump(time_log, f)


