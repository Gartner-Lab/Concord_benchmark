#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -r y
#$ -q gpu.q
#$ -pe smp 1
#$ -l compute_cap=61         # Minimum SM version: GTX 1080 or better
#$ -l gpu_mem=10000M         # At least 10 GiB GPU memory
#$ -l mem_free=8G            # 8 GiB system RAM
#$ -l scratch=50G            # 50 GiB scratch space
#$ -l h_rt=02:00:00          # Max 2 hours runtime

# -------- GPU Profiling Start --------
gpuprof=$(dcgmi group --create mygpus --add "$SGE_GPU" | awk '{print $10}')
dcgmi stats --group "$gpuprof" --enable
dcgmi stats --group "$gpuprof" --jstart "$JOB_ID"

# -------- Environment Info --------
echo "Running on: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

# -------- Load Conda and CUDA --------
module load cuda/11.8
source activate scenv

# -------- Run Python Benchmark --------
python benchmark_pbmc_DeSimone_test_concord.py

# -------- GPU Profiling End --------
dcgmi stats --group "$gpuprof" --jstop "$JOB_ID"
dcgmi stats --group "$gpuprof" --verbose --job "$JOB_ID"
dcgmi group --delete "$gpuprof"

# -------- Optional: Show job info --------
[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID"
