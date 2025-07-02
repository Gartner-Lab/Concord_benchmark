#!/usr/bin/env bash
set -eo pipefail

echo "Running on: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true

source ~/.bashrc
conda activate concord

TIMESTAMP=$(date +'%m%d-%H%M')
python benchmark_dkd_Wilson_liger.py --timestamp $TIMESTAMP
