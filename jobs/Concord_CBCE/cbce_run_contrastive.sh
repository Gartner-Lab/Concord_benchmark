#! /bin/bash
#$ -S /bin/bash       # Specifies the script language (bash)
#$ -cwd               # Run in the current working directory
#$ -N contrastive      # Job name
#$ -q gpu.q           # Specify the GPU queue
#$ -l h_rt=0:60:00   # Maximum run time (24 hours in this case)
#$ -l mem_free=64G    # Request memory (32GB here)
#$ -l compute_cap=80
#$ -l gpu_mem=20000M

current_time=$(date "+%Y%m%d-%H%M%S")
# Define log file names
output_log="output_$current_time.log"
error_log="error_$current_time.log"

exec > $output_log 2> $error_log

# Load necessary modules (e.g., CUDA)
module load CBI miniforge3/24.3.0-0
module load cuda/11.8
conda activate cellpath
export CUDA_VISIBLE_DEVICES=$SGE_GPU
export LD_LIBRARY_PATH=/wynton/home/gartner/zhuqin/.conda/envs/cellpath/lib:$LD_LIBRARY_PATH

# Run your Python script
gpuprof=$(dcgmi group -c mygpus -a $SGE_GPU | awk '{print $10}')
dcgmi stats -g $gpuprof -e
dcgmi stats -g $gpuprof -s $JOB_ID
python cbce_run_contrastive.py

dcgmi stats -g $gpuprof -x $JOB_ID
dcgmi stats -g $gpuprof -v -j $JOB_ID
dcgmi group -d $gpuprof
