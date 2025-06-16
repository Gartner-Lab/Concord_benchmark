#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -r y
#$ -q gpu.q
#$ -pe smp 1
#$ -l mem_free=16G
#$ -l scratch=50G
#$ -l h_rt=02:00:00

echo "Running on: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

module load cuda/11.8
source activate scenv || conda activate scenv

python benchmark_dkd_Wilson_scanorama.py
