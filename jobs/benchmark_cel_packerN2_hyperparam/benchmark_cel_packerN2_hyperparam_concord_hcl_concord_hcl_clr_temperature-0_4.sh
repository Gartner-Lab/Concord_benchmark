#!/usr/bin/env bash
set -eo pipefail

echo "Running on: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true

source ~/.bashrc
conda activate cellpath

TIMESTAMP=$(date +'%m%d-%H%M')
python benchmark_cel_packerN2_hyperparam_concord_hcl_concord_hcl_clr_temperature-0_4.py --timestamp $TIMESTAMP
