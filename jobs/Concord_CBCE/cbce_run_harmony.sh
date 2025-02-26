#! /bin/bash
#$ -S /bin/bash       # Specifies the script language (bash)
#$ -cwd               # Run in the current working directory
#$ -q gpu.q           # Specify the GPU queue
#$ -l h_rt=1:00:00   # Maximum run time (24 hours in this case)
#$ -l mem_free=64G    # Request memory (32GB here)
#$ -o output.log      # Output file
#$ -e error.log       # Error log file

current_time=$(date "+%Y%m%d-%H%M%S")
# Define log file names
script_name="cbce_run_harmony"

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
python $script_name.py

dcgmi stats -g $gpuprof -x $JOB_ID
dcgmi stats -g $gpuprof -v -j $JOB_ID
dcgmi group -d $gpuprof
