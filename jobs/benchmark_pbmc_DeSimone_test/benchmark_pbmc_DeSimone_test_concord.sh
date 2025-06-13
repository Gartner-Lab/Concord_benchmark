#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -r y
#$ -q gpu.q
#$ -pe smp 4
# #$ -l compute_cap=61        # Minimum SM version (GTX 1080 or better)
# #$ -l gpu_mem=8000M        # Needs at least 8 GiB GPU memory
#$ -l mem_free=4G           # 4 GiB system RAM per CPU
#$ -l scratch=50G           # 50 GiB scratch space
#$ -l h_rt=02:00:00         # Max 2 hours runtime

# Use the same save path format as the Python script
DATESTAMP=$(date +%Y%m%d)
SAVE_DIR="../../save/pbmc_Darmanis-${DATESTAMP}"
mkdir -p "$SAVE_DIR"

# Redirect output log to the save directory
#$ -o $SAVE_DIR/job_output_concord.txt

# -------- GPU Profiling Start --------
gpuprof=$(dcgmi group --create mygpus --add "$SGE_GPU" | awk '{print $10}')
dcgmi stats --group "$gpuprof" --enable
dcgmi stats --group "$gpuprof" --jstart "$JOB_ID"

# -------- Job Logging --------
echo "Running on: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

# -------- Run Python Benchmark --------
module load cuda/11.8
source activate scenv
python benchmark_pbmc_DeSimone_test_concord.py

# -------- GPU Profiling End --------
dcgmi stats --group "$gpuprof" --jstop "$JOB_ID"
dcgmi stats --group "$gpuprof" --verbose --job "$JOB_ID"
dcgmi group --delete "$gpuprof"

[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID"
