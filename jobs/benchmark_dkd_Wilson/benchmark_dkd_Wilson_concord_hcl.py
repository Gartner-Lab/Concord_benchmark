#!/usr/bin/env python
# coding: utf-8
import torch, scanpy as sc, pandas as pd, logging, sys
from pathlib import Path
import concord as ccd
import argparse as _arg
import time as _t

# ------------------ Argument Parsing ------------------ #
p = _arg.ArgumentParser()
p.add_argument("--timestamp")
args = p.parse_args()
FILE_SUFFIX = args.timestamp or _t.strftime("%m%d-%H%M")

# ------------------ Logger ---------------------------- #
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
h = logging.StreamHandler(); h.setFormatter(fmt); logger.addHandler(h)

# ------------------ Config ---------------------------- #
CONFIG = {
    "GENERAL_SETTINGS": {"PROJ_NAME": "dkd_Wilson", "SEED": 0},
    "COMPUTATION_SETTINGS": {
        "AUTO_SELECT_DEVICE": True
        
    },
    "DATA_SETTINGS": {
        "ADATA_FILENAME": "dkd_Wilson_preprocessed_HVG.h5ad",
        "BATCH_KEY": "sample_uuid",
        "STATE_KEY": "cell_type",
        "COUNT_LAYER": "counts",
    },
    "INTEGRATION_SETTINGS": {
        "METHODS": ["concord_hcl"],
        "LATENT_DIM": 50,
        "RETURN_CORRECTED": False,
        "TRANSFORM_BATCH": None,
        "VERBOSE": False,
    },
    "UMAP_SETTINGS": {"COMPUTE_UMAP": False, "N_COMPONENTS": 2,
                       "N_NEIGHBORS": 30, "MIN_DIST": 0.1},
    "CONCORD_SETTINGS": {"CONCORD_KWARGS": {}},
}

ccd.ul.set_seed(CONFIG["GENERAL_SETTINGS"]["SEED"])

# --------------- Device selection -------------------- #
if CONFIG["COMPUTATION_SETTINGS"]["AUTO_SELECT_DEVICE"]:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    DEVICE = torch.device(CONFIG["COMPUTATION_SETTINGS"].get("MANUAL_DEVICE","cpu"))
logger.info(f"Using device: {DEVICE}")

# --------------- Paths -------------------------------- #
METHOD = CONFIG["INTEGRATION_SETTINGS"]["METHODS"][0]
OUT_KEY = CONFIG["CONCORD_SETTINGS"]["CONCORD_KWARGS"].get("output_key", METHOD)

BASE_SAVE_DIR = Path(f"../../save/{CONFIG['GENERAL_SETTINGS']['PROJ_NAME']}/{OUT_KEY}_{FILE_SUFFIX}")
BASE_DATA_DIR = Path(f"../../data/{CONFIG['GENERAL_SETTINGS']['PROJ_NAME']}")
BASE_SAVE_DIR.mkdir(parents=True, exist_ok=True); BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)

# add file logger after dirs exist
file_handler = logging.FileHandler(BASE_SAVE_DIR / "run.log")
file_handler.setFormatter(fmt); logger.addHandler(file_handler)

# --------------- Stream redirect ---------------------- #
class _Stream:  # redirect stdout/err to logger
    def __init__(self, level): self.level = level
    def write(self, msg): 
        for line in msg.rstrip().splitlines(): logger.log(self.level, line)
    def flush(self): pass
sys.stdout, sys.stderr = _Stream(logging.INFO), _Stream(logging.INFO)

# --------------- Main --------------------------------- #
def main():
    adata_path = BASE_DATA_DIR / CONFIG["DATA_SETTINGS"]["ADATA_FILENAME"]
    try:
        adata = sc.read_h5ad(adata_path); logger.info(f"Loaded {adata_path}")
    except Exception as e:
        logger.error(f"Unable to read AnnData: {e}"); return

    log_df = ccd.bm.run_integration_methods_pipeline(
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
        concord_kwargs=CONFIG["CONCORD_SETTINGS"]["CONCORD_KWARGS"],
    )
    logger.info("Integration complete.")

    # ------- save embedding --------------------------- #
    if OUT_KEY in adata.obsm:
        df = pd.DataFrame(adata.obsm[OUT_KEY], index=adata.obs_names)
        out = BASE_SAVE_DIR / f"{OUT_KEY}_embedding_{FILE_SUFFIX}.tsv"
        df.to_csv(out, sep="\t"); logger.info(f"Saved embedding → {out}")
    else:
        logger.warning(f"obsm['{OUT_KEY}'] missing, skip save")

    # ------- save performance log --------------------- #
    log_df.insert(0, "method", log_df.index)
    log_df.to_csv(BASE_SAVE_DIR / f"benchmark_log_{FILE_SUFFIX}.tsv", sep="\t", index=False)
    logger.info("All tasks finished successfully.")

if __name__ == "__main__":
    main()
