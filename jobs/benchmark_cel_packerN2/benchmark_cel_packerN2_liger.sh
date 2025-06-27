#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -r y
#$ -q gpu.q
#$ -pe smp 1
#$ -l mem_free=8G
#$ -l scratch=50G
#$ -l h_rt=02:00:00
#$ -l compute_cap=86

echo "Running on: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

module load cuda/11.8

# Initialize conda and activate environment
source /wynton/home/cbi/shared/software/CBI/miniforge3-24.3.0-0/etc/profile.d/conda.sh
conda activate cellpath

cd $(dirname ../jobs/benchmark_cel_packerN2/benchmark_cel_packerN2_liger.py)
TIMESTAMP=$(date +'%m%d-%H%M')
python benchmark_cel_packerN2_liger.py --timestamp $TIMESTAMP
