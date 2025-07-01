#!/usr/bin/env bash
set -eo pipefail

echo "Job: benchmark_dkd_Wilson_scanorama"
echo "Start: $(date -Is)"
echo "Host: $(hostname)"
nvidia-smi || true

# ---- conda ------------------------------------------------
source ~/miniconda3/etc/profile.d/conda.sh     # update if conda lives elsewhere
conda activate concord_env

TIMESTAMP=$(date +'%m%d-%H%M')
python benchmark_dkd_Wilson_scanorama.py --timestamp $TIMESTAMP

echo "End: $(date -Is)"
