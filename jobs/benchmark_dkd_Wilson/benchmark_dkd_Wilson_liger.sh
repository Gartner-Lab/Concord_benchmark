#!/usr/bin/env bash
set -eo pipefail

<<<<<<< HEAD
echo "Job: benchmark_dkd_Wilson_liger"
echo "Start: $(date -Is)"
echo "Host: $(hostname)"
nvidia-smi || true

# ---- conda ------------------------------------------------
source ~/miniconda3/etc/profile.d/conda.sh     # update if conda lives elsewhere
conda activate concord_env

TIMESTAMP=$(date +'%m%d-%H%M')
python benchmark_dkd_Wilson_liger.py --timestamp $TIMESTAMP

echo "End: $(date -Is)"
=======
echo "Running on: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true

source ~/.bashrc
conda activate concord

TIMESTAMP=$(date +'%m%d-%H%M')
python benchmark_dkd_Wilson_liger.py --timestamp $TIMESTAMP
>>>>>>> origin
