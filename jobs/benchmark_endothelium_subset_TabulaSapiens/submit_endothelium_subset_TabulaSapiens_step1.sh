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
#$ -N bmk_TabulaSapiens_s1

# -------- System Info --------
echo "Running on: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

# -------- Load CUDA --------
module load cuda/11.8

# -------- Conda Setup --------
source /wynton/home/cbi/shared/software/CBI/miniforge3-24.3.0-0/etc/profile.d/conda.sh
conda activate scenv

# -------- Logging --------
LOG_DIR="../save/endothelium_subset_TabulaSapiens/logs"
mkdir -p "$LOG_DIR"
exec > >(tee -a "${LOG_DIR}/job_${JOB_ID}.log") 2>&1

# -------- Run Step 1 --------
python ../notebook/benchmark_endothelium_subset_TabulaSapiens_step1.py

echo "✅ Step 1 finished at $(date)"
