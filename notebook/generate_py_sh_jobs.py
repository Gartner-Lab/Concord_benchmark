#!/usr/bin/env python
# coding: utf-8
"""
Generate benchmark *.py wrappers + submission *.sh launchers.

Example
-------
# Local/AWS EC2
python generate_py_sh_jobs.py \
  --proj_name lung_SM \
  --adata_filename submucosal.h5ad \
  --methods scanorama harmony \
  --batch_key donor_id \
  --state_key cell_type \
  --mode local \
  --conda_env concord_env

# Wynton HPC
python generate_py_sh_jobs.py \
  --proj_name lung_SM \
  --adata_filename submucosal.h5ad \
  --methods scanorama harmony \
  --batch_key donor_id \
  --state_key cell_type \
  --mode wynton
"""
import argparse, json, textwrap, re
from pathlib import Path

# --------------------------------------------------------------------- #
# ---------------------------  PYTHON TEMPLATE  ----------------------- #
# --------------------------------------------------------------------- #
PYTHON_TEMPLATE = r"""#!/usr/bin/env python
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
CONFIG = {{
    "GENERAL_SETTINGS": {{"PROJ_NAME": "{proj_name}", "SEED": 0}},
    "COMPUTATION_SETTINGS": {{
        "AUTO_SELECT_DEVICE": {auto_device}{manual_device_comma}
        {manual_device}
    }},
    "DATA_SETTINGS": {{
        "ADATA_FILENAME": "{adata_filename}",
        "BATCH_KEY": "{batch_key}",
        "STATE_KEY": "{state_key}",
        "COUNT_LAYER": "counts",
    }},
    "INTEGRATION_SETTINGS": {{
        "METHODS": ["{method}"],
        "LATENT_DIM": {latent_dim},
        "RETURN_CORRECTED": False,
        "TRANSFORM_BATCH": None,
        "VERBOSE": {verbose},
    }},
    "UMAP_SETTINGS": {{"COMPUTE_UMAP": False, "N_COMPONENTS": 2,
                       "N_NEIGHBORS": 30, "MIN_DIST": 0.1}},
    "CONCORD_SETTINGS": {{"CONCORD_KWARGS": {concord_kwargs_repr}}},
}}

ccd.ul.set_seed(CONFIG["GENERAL_SETTINGS"]["SEED"])

# --------------- Device selection -------------------- #
if CONFIG["COMPUTATION_SETTINGS"]["AUTO_SELECT_DEVICE"]:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    DEVICE = torch.device(CONFIG["COMPUTATION_SETTINGS"].get("MANUAL_DEVICE","cpu"))
logger.info(f"Using device: {{DEVICE}}")

# --------------- Paths -------------------------------- #
METHOD = CONFIG["INTEGRATION_SETTINGS"]["METHODS"][0]
OUT_KEY = CONFIG["CONCORD_SETTINGS"]["CONCORD_KWARGS"].get("output_key", METHOD)

BASE_SAVE_DIR = Path(f"../../save/{{CONFIG['GENERAL_SETTINGS']['PROJ_NAME']}}/{{OUT_KEY}}_{{FILE_SUFFIX}}")
BASE_DATA_DIR = Path(f"../../data/{{CONFIG['GENERAL_SETTINGS']['PROJ_NAME']}}")
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
        adata = sc.read_h5ad(adata_path); logger.info(f"Loaded {{adata_path}}")
    except Exception as e:
        logger.error(f"Unable to read AnnData: {{e}}"); return

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
        out = BASE_SAVE_DIR / f"{{OUT_KEY}}_embedding_{{FILE_SUFFIX}}.tsv"
        df.to_csv(out, sep="\\t"); logger.info(f"Saved embedding → {{out}}")
    else:
        logger.warning(f"obsm['{{OUT_KEY}}'] missing, skip save")

    # ------- save performance log --------------------- #
    log_df.insert(0, "method", log_df.index)
    log_df.to_csv(BASE_SAVE_DIR / f"benchmark_log_{{FILE_SUFFIX}}.tsv", sep="\\t", index=False)
    logger.info("All tasks finished successfully.")

if __name__ == "__main__":
    main()
"""

# --------------------------------------------------------------------- #
# --------------------------  SH TEMPLATES  --------------------------- #
# --------------------------------------------------------------------- #
SH_TEMPLATE_WYNTON = """#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -r y
#$ -q gpu.q
#$ -pe smp 1
#$ -l mem_free={mem}
#$ -l scratch={scratch}
#$ -l h_rt={runtime}
#$ -l compute_cap=86

echo "Host: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

module load cuda/11.8
source /wynton/home/cbi/shared/software/CBI/miniforge3-24.3.0-0/etc/profile.d/conda.sh
conda activate {conda_env}

TIMESTAMP=$(date +'%m%d-%H%M')
python {script_name}.py --timestamp $TIMESTAMP
"""

SH_TEMPLATE_LOCAL = """#!/usr/bin/env bash
set -euo pipefail

echo "Job: {script_name}"
echo "Start: $(date -Is)"
echo "Host: $(hostname)"
nvidia-smi || true

# ---- conda ----
source ~/miniconda3/etc/profile.d/conda.sh   # adjust if conda lives elsewhere
conda activate {conda_env}

TIMESTAMP=$(date +'%m%d-%H%M')
python {script_name}.py --timestamp $TIMESTAMP

echo "End: $(date -Is)"
"""

# --------------------------------------------------------------------- #
# ----------------------------  DRIVER  ------------------------------- #
# --------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--proj_name', required=True)
    p.add_argument('--adata_filename', required=True)
    p.add_argument('--methods', nargs='+', required=True)
    p.add_argument('--batch_key', required=True)
    p.add_argument('--state_key', required=True)
    p.add_argument('--latent_dim', type=int, default=50)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--mem', default='8G')
    p.add_argument('--scratch', default='50G')
    p.add_argument('--runtime', default='01:00:00')
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--conda_env', default='scenv')
    p.add_argument('--output_dir', default='./generated_scripts')
    p.add_argument('--concord_kwargs', default='{}',
                   help='JSON string or @file path with Concord kwargs')
    p.add_argument('--mode', choices=['wynton', 'local'], default='local',
                   help='wynton = Wynton HPC, local = local/AWS')
    args = p.parse_args()

    out_root = Path(args.output_dir) / f"benchmark_{args.proj_name}"
    out_root.mkdir(parents=True, exist_ok=True)

    # ------------ concord_kwargs: parse ONCE ------------------------- #
    if args.concord_kwargs.startswith('@'):
        concord_kwargs = json.loads(Path(args.concord_kwargs[1:]).read_text())
    else:
        concord_kwargs = json.loads(textwrap.dedent(args.concord_kwargs))

    suffix = ''
    tag_or_key = concord_kwargs.get('output_key') or concord_kwargs.get('tag')
    if tag_or_key:
        suffix = '_' + re.sub(r'[^A-Za-z0-9_-]', '_', tag_or_key)

    sh_tpl = SH_TEMPLATE_WYNTON if args.mode == 'wynton' else SH_TEMPLATE_LOCAL

    for method in args.methods:
        script_name = f"benchmark_{args.proj_name}_{method}{suffix}"

        auto_device     = str(args.device == 'auto')
        manual_device   = f'"MANUAL_DEVICE": "{args.device}"' if args.device != 'auto' else ''
        manual_comma    = ',' if manual_device else ''
        verbose_flag    = 'True' if args.verbose else 'False'

        # --------------- write *.py ----------------------- #
        py_text = PYTHON_TEMPLATE.format(
            proj_name=args.proj_name,
            adata_filename=args.adata_filename,
            batch_key=args.batch_key,
            state_key=args.state_key,
            method=method,
            auto_device=auto_device,
            manual_device=manual_device,
            manual_device_comma=manual_comma,
            latent_dim=args.latent_dim,
            concord_kwargs_repr=repr(concord_kwargs),
            verbose=verbose_flag,
        )
        py_path = out_root / f"{script_name}.py"
        py_path.write_text(py_text)

        # --------------- write *.sh ----------------------- #
        sh_text = sh_tpl.format(
            mem=args.mem, scratch=args.scratch, runtime=args.runtime,
            conda_env=args.conda_env, script_name=script_name,
        )
        sh_path = out_root / f"{script_name}.sh"
        sh_path.write_text(sh_text)
        sh_path.chmod(0o755)  # make executable

        rel = lambda p: p.relative_to(Path(args.output_dir))
        print(f"✅ {rel(py_path)}")
        print(f"✅ {rel(sh_path)}\n")

if __name__ == '__main__':
    main()
