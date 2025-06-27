#!/usr/bin/env python
# coding: utf-8

import os
import time
import torch
import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path
import concord as ccd
import logging
import sys
import argparse

# ------------------- Argument Parsing -------------------
parser = argparse.ArgumentParser()
parser.add_argument('--timestamp', required=True)
args = parser.parse_args()
FILE_SUFFIX = args.timestamp

# ------------------- Logger Setup -------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Console logging
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ------------------- Config -------------------
CONFIG = {
    "GENERAL_SETTINGS": {
        "PROJ_NAME": "endothelium_subset_TabulaSapiens",
        "SEED": 0,
    },
    "COMPUTATION_SETTINGS": {
        "AUTO_SELECT_DEVICE": True
        
    },
    "DATA_SETTINGS": {
        "ADATA_FILENAME": "endothelium_subset_TabulaSapiens_preprocessed.h5ad",
        "BATCH_KEY": "donor_id",
        "STATE_KEY": "cell_type",
        "COUNT_LAYER": "counts",
    },
    "INTEGRATION_SETTINGS": {
        "METHODS": ["contrastive"],
        "LATENT_DIM": 30,
        "RETURN_CORRECTED": False,
        "TRANSFORM_BATCH": None,
        "VERBOSE": True,
    },
    "UMAP_SETTINGS": {
        "COMPUTE_UMAP": False,
        "N_COMPONENTS": 2,
        "N_NEIGHBORS": 30,
        "MIN_DIST": 0.5,
    },
}

# Set seed
ccd.ul.set_seed(CONFIG["GENERAL_SETTINGS"]["SEED"])

# Select device
if CONFIG["COMPUTATION_SETTINGS"]["AUTO_SELECT_DEVICE"]:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    DEVICE = torch.device(CONFIG["COMPUTATION_SETTINGS"].get("MANUAL_DEVICE", "cpu"))

logger.info(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    gpu_index = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(gpu_index)
    logger.info(f"Using GPU {gpu_index}: {gpu_name}")
else:
    gpu_index = None
    gpu_name = "CPU"

# ------------------- Paths -------------------
method = CONFIG["INTEGRATION_SETTINGS"]["METHODS"][0]
BASE_SAVE_DIR = Path(f"../../save/{CONFIG['GENERAL_SETTINGS']['PROJ_NAME']}/{method}_{FILE_SUFFIX}/")
BASE_DATA_DIR = Path(f"../../data/{CONFIG['GENERAL_SETTINGS']['PROJ_NAME']}/")
BASE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)

# File logger after save dir created
log_file_path = BASE_SAVE_DIR / "run.log"
file_handler = logging.FileHandler(log_file_path)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.info(f"Logging to: {log_file_path}")

class StreamToLogger:
    def __init__(self, logger, log_level):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line)

    def flush(self):
        pass

sys.stdout = StreamToLogger(logger, logging.INFO)
sys.stderr = StreamToLogger(logger, logging.INFO)

def main():
    logger.info("Starting integration pipeline...")

    adata_path = BASE_DATA_DIR / CONFIG["DATA_SETTINGS"]["ADATA_FILENAME"]

    try:
        adata = sc.read_h5ad(adata_path)
        logger.info(f"Loaded AnnData from: {adata_path}")
    except FileNotFoundError:
        logger.error(f"AnnData file not found at: {adata_path}")
        return
    except Exception as e:
        logger.error(f"Error loading AnnData: {e}")
        return

    time_log, ram_log, vram_log = ccd.bm.run_integration_methods_pipeline(
        adata=adata,
        methods=CONFIG["INTEGRATION_SETTINGS"]["METHODS"],
        batch_key=CONFIG["DATA_SETTINGS"]["BATCH_KEY"],
        count_layer=CONFIG["DATA_SETTINGS"]["COUNT_LAYER"],
        class_key=CONFIG["DATA_SETTINGS"]["STATE_KEY"],
        latent_dim=CONFIG["INTEGRATION_SETTINGS"]["LATENT_DIM"],
        device=DEVICE,
        return_corrected=CONFIG["INTEGRATION_SETTINGS"]["RETURN_CORRECTED"],
        transform_batch=CONFIG["INTEGRATION_SETTINGS"]["TRANSFORM_BATCH"],
        seed=CONFIG["GENERAL_SETTINGS"]["SEED"],
        compute_umap=CONFIG["UMAP_SETTINGS"]["COMPUTE_UMAP"],
        umap_n_components=CONFIG["UMAP_SETTINGS"]["N_COMPONENTS"],
        umap_n_neighbors=CONFIG["UMAP_SETTINGS"]["N_NEIGHBORS"],
        umap_min_dist=CONFIG["UMAP_SETTINGS"]["MIN_DIST"],
        verbose=CONFIG["INTEGRATION_SETTINGS"]["VERBOSE"]
    )
    logger.info("Integration complete.")

    methods_to_save = CONFIG["INTEGRATION_SETTINGS"]["METHODS"]
    for obsm_key in methods_to_save:
        if obsm_key in adata.obsm:
            df = pd.DataFrame(adata.obsm[obsm_key], index=adata.obs_names)
            out_path = BASE_SAVE_DIR / f"{obsm_key}_embedding_{FILE_SUFFIX}.tsv"
            df.to_csv(out_path, sep='\t')
            logger.info(f"Saved embedding for '{obsm_key}' to: {out_path}")
        else:
            logger.warning(f"obsm['{obsm_key}'] not found. Skipping.")

    log_data = []
    for k in methods_to_save:
        if k in time_log and k in ram_log and k in vram_log:
            log_data.append({
                "method": k,
                "gpu_name": gpu_name,
                "runtime_sec": time_log[k],
                "RAM_MB": ram_log[k],
                "VRAM_MB": vram_log[k]
            })
        else:
            logger.warning(f"Missing performance logs for '{k}'")

    if log_data:
        log_df = pd.DataFrame(log_data)
        log_file_path = BASE_SAVE_DIR / f"benchmark_log_{method}_{FILE_SUFFIX}.tsv"
        log_df.to_csv(log_file_path, sep='\t', index=False)
        logger.info(f"Saved performance log to: {log_file_path}")
    else:
        logger.warning("No complete log data to save.")

    logger.info("All tasks finished successfully.")

if __name__ == "__main__":
    main()
