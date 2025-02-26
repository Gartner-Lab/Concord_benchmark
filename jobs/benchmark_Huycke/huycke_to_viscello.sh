#! /bin/bash
#$ -S /bin/bash       # Specifies the script language (bash)
#$ -cwd               # Run in the current working directory
#$ -l h_rt=1:00:00   # Maximum run time (24 hours in this case)
#$ -l mem_free=64G    # Request memory (32GB here)

current_time=$(date "+%Y%m%d-%H%M%S")
# Define log file names
script_name="huycke_to_viscello"

# Load necessary modules (e.g., CUDA)
module load CBI miniforge3/24.3.0-0
module load cuda/11.8
conda activate cellpath
export CUDA_VISIBLE_DEVICES=$SGE_GPU
export LD_LIBRARY_PATH=/wynton/home/gartner/zhuqin/.conda/envs/cellpath/lib:$LD_LIBRARY_PATH

python $script_name.py
