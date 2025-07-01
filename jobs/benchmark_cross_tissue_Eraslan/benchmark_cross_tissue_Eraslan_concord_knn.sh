#!/usr/bin/env bash
set -eo pipefail

echo "Running on: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true

source ~/.bashrc
conda activate concord

TIMESTAMP=$(date +'%m%d-%H%M')
python benchmark_cross_tissue_Eraslan_concord_knn.py --timestamp $TIMESTAMP
