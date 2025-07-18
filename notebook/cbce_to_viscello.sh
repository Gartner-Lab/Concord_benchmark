#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -r y
#$ -pe smp 1
#$ -l mem_free=100G
#$ -l scratch=50G
#$ -l h_rt=3:00:00

echo "Running on: $(hostname)"

source /wynton/home/cbi/shared/software/CBI/miniforge3-24.3.0-0/etc/profile.d/conda.sh
conda activate cellpath

TIMESTAMP=$(date +'%m%d-%H%M')
python cbce_to_viscello.py
