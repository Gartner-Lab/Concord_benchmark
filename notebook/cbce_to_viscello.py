#!/usr/bin/env python
# coding: utf-8
import concord as ccd
import scanpy as sc
from pathlib import Path
import time

data_dir = Path('../data/CBCE')
data_dir.mkdir(parents=True, exist_ok=True)

proj_name = "CBCE"
save_dir = f"../save/dev_{proj_name}-{time.strftime('%b%d')}/"
save_dir = Path(save_dir)
save_dir.mkdir(parents=True, exist_ok=True)
proj_name = "CBCE"
file_name = "CBCE"
file_suffix = f"{time.strftime('%b%d-%H%M')}"
seed = 0

adata = sc.read_h5ad("../data/CBCE/CBCE_Jul16-1955.h5ad")

ccd.ul.anndata_to_viscello(adata,
                        output_dir=data_dir / f"viscello_{proj_name}",
                        project_name=proj_name,
                        organism='cel')
