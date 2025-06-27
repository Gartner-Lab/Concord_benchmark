#! /bin/bash
#$ -S /bin/bash       # Specifies the script language (bash)
#$ -cwd               # Run in the current working directory
#$ -pe smp 1
#$ -l h_rt=48:00:00   # Maximum run time (24 hours in this case)
#$ -l mem_free=250G    # Request memory (32GB here)
#$ -o seurat_output.log      # Output file
#$ -e seurat_error.log       # Error log file

current_time=$(date "+%Y%m%d-%H%M%S")
# Define log file names
script_name="cbce_run_seurat"
# include the script name in the log file name to differentiate between different scripts
output_log="output_$script_name_$current_time.log"
error_log="error_$script_name_$current_time.log"

exec > $output_log 2> $error_log

# Load necessary modules (e.g., CUDA)
module load CBI miniforge3/24.3.0-0
conda activate cellpath
export LD_LIBRARY_PATH=/wynton/home/gartner/zhuqin/.conda/envs/cellpath/lib:$LD_LIBRARY_PATH

# Run your cript

Rscript cbce_run_seurat_rPCA.R

## End-of-job summary, if running as a job
[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID"  # This is useful for debugging and usage purposes,
                                          # e.g. "did my job exceed its memory request?"

