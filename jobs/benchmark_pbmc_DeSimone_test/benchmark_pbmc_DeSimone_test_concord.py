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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------- GLOBAL SETTINGS / CONFIGURATION -------------------
# This dictionary holds all configurable parameters, logically grouped.
# For running with multiple datasets, you might iterate over a list of
# project-specific settings in a larger script, or pass them via command-line arguments.
CONFIG = {
    "GENERAL_SETTINGS": {
        "PROJ_NAME": "pbmc_Darmanis", # Name of the project/dataset being processed
        "SEED": 0,                    # Random seed for reproducibility
        "FILE_SUFFIX_FORMAT": '%m%d-%H%M', # Format for timestamp in output filenames/directories
    },
    "COMPUTATION_SETTINGS": {
        "AUTO_SELECT_DEVICE": True,   # If True, script will automatically use CUDA if available, else CPU
        # "MANUAL_DEVICE": "cpu",     # Uncomment and set if you want to force a specific device (e.g., "cpu" or "cuda:0")
    },
    "DATA_SETTINGS": {
        "ADATA_FILENAME": 'pbmc_Darmanis_subset_9K.h5ad', # Name of the AnnData file
        "BATCH_KEY": 'dataset',       # Key in adata.obs for batch information
        "STATE_KEY": 'cell_type',     # Key in adata.obs for cell type/state information
        "COUNT_LAYER": "counts",      # Layer in adata.layers containing raw counts
    },
    "INTEGRATION_SETTINGS": {
        "METHODS": ["concord"],       # List of integration methods to run (e.g., ["concord", "scvi"])
        "LATENT_DIM": 30,             # Dimensionality of the latent space for integration
        "RETURN_CORRECTED": False,    # Whether to return corrected data (Concord specific)
        "TRANSFORM_BATCH": None,      # Batch transformation (Concord specific)
        "VERBOSE": True,              # Print detailed progress messages
    },
    "UMAP_SETTINGS": {
        "COMPUTE_UMAP": False,        # Whether to compute UMAP embedding after integration
        "N_COMPONENTS": 2,            # Number of UMAP components
        "N_NEIGHBORS": 30,            # Number of neighbors for UMAP
        "MIN_DIST": 0.5,              # Minimum distance for UMAP
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

# File suffix for timestamp
FILE_SUFFIX = time.strftime(CONFIG["GENERAL_SETTINGS"]["FILE_SUFFIX_FORMAT"])

# Paths
BASE_SAVE_DIR = Path(f"../../save/{CONFIG['GENERAL_SETTINGS']['PROJ_NAME']}-{FILE_SUFFIX.split('-')[0]}/")
BASE_DATA_DIR = Path(f"../../data/{CONFIG['GENERAL_SETTINGS']['PROJ_NAME']}/")
BASE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ----------------- MAIN FUNCTION -------------------

def main():
    """
    Main function to load scRNA-seq data, perform Concord integration,
    and save results and performance logs.
    """
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

    # Save obs_names for optional verification later
    obs_name_path = BASE_SAVE_DIR / "obs_names.csv"
    adata.obs_names.to_series().to_csv(obs_name_path)
    logger.info(f"Saved obs_names to: {obs_name_path}")

    # Run integration
    time_log, ram_log, vram_log = ccd.ul.run_integration_methods_pipeline(
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
        verbose=CONFIG["INTEGRATION_SETTINGS"]["VERBOSE"],
    )
    logger.info("Integration complete.")

    # Save embeddings with obs_names as index
    methods_to_save = CONFIG["INTEGRATION_SETTINGS"]["METHODS"]
    for obsm_key in methods_to_save:
        if obsm_key in adata.obsm:
            df = pd.DataFrame(adata.obsm[obsm_key], index=adata.obs_names)
            embedding_file_path = BASE_SAVE_DIR / f"{obsm_key}_embedding_{FILE_SUFFIX}.tsv"
            df.to_csv(embedding_file_path, sep='\t')
            logger.info(f"Saved embedding for '{obsm_key}' to: {embedding_file_path}")
        else:
            logger.warning(f"obsm['{obsm_key}'] not found. Skipping.")

    # Save performance logs
    log_data = []
    for k in methods_to_save:
        if k in time_log and k in ram_log and k in vram_log:
            log_data.append({
                "method": k,
                "runtime_sec": time_log[k],
                "RAM_GB": ram_log[k] / 1024**3,
                "VRAM_GB": vram_log[k] / 1024**3
            })
        else:
            logger.warning(f"Missing performance logs for '{k}'")

    if log_data:
        log_df = pd.DataFrame(log_data)
        log_file_path = BASE_SAVE_DIR / f"benchmark_log_{FILE_SUFFIX}.tsv"
        log_df.to_csv(log_file_path, sep='\t', index=False)
        logger.info(f"Saved performance log to: {log_file_path}")
    else:
        logger.warning("No complete log data to save.")

    logger.info("All tasks finished successfully.")

if __name__ == "__main__":
    main()
