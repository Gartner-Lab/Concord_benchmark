#!/usr/bin/env bash
set -eo pipefail

echo "Running on: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true

source ~/.bashrc
conda activate concord_env

TIMESTAMP=$(date +'%m%d-%H%M')
python benchmark_HypoMap_Steuernagel_concord_knn.py --timestamp $TIMESTAMP
