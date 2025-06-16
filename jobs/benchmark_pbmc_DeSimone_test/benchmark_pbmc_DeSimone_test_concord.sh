#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -r y
#$ -q gpu.q
#$ -pe smp 1
# #$ -l hostname=qb3-atgpu25
#$ -l mem_free=4G
#$ -l scratch=20G
#$ -l h_rt=00:05:00

# -------- Environment Info --------
echo "Running on: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

# -------- Load Conda and CUDA --------
module load cuda/11.8
source activate scenv || conda activate scenv

# -------- Run Python Benchmark --------
python benchmark_pbmc_DeSimone_test_concord.py
