#! /bin/bash
#$ -S /bin/bash       # Specifies the script language (bash)
#$ -cwd               # Run in the current working directory
#$ -N cbce_vis      # Job name
#$ -l h_rt=0:60:00   # Maximum run time (24 hours in this case)
#$ -l mem_free=100G    # Request memory (32GB here)

current_time=$(date "+%Y%m%d-%H%M%S")
# Define log file names
output_log="output_$current_time.log"
error_log="error_$current_time.log"

exec > $output_log 2> $error_log

# Load necessary modules (e.g., CUDA)
module load CBI miniforge3/24.3.0-0
conda activate cellpath
export LD_LIBRARY_PATH=/wynton/home/gartner/zhuqin/.conda/envs/cellpath/lib:$LD_LIBRARY_PATH

# Run your Python script
python cbce_to_viscello.py
