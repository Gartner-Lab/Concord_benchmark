#!/usr/bin/env bash
set -eo pipefail

echo "Running on: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true

source ~/.bashrc
conda activate concord

<<<<<<< HEAD:jobs/benchmark_pbmc_DeSimone_10K/benchmark_pbmc_DeSimone_10K_concord_knn.sh
# Initialize conda and activate environment
source /wynton/home/cbi/shared/software/CBI/miniforge3-24.3.0-0/etc/profile.d/conda.sh
conda activate scenv

=======
>>>>>>> origin:jobs/benchmark_cross_tissue_Eraslan/benchmark_cross_tissue_Eraslan_concord_knn.sh
TIMESTAMP=$(date +'%m%d-%H%M')
python benchmark_pbmc_DeSimone_10K_concord_knn.py --timestamp $TIMESTAMP
