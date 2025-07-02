#!/usr/bin/env bash
set -eo pipefail

echo "Running on: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true

source ~/.bashrc
conda activate cellpath

TIMESTAMP=$(date +'%m%d-%H%M')
python benchmark_cel_packerN2_hyperparam_concord_knn_concord_knn_p_intra_domain-0_9.py --timestamp $TIMESTAMP
