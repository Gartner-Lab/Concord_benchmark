#! /bin/bash
#$ -S /bin/bash       # Specifies the script language (bash)
#$ -cwd               # Run in the current working directory
#$ -q gpu.q           # Specify the GPU queue
#$ -l h_rt=12:00:00   # Maximum run time (24 hours in this case)
#$ -l mem_free=64G    # Request memory (32GB here)
#$ -o scib_output.log      # Output file
#$ -e scib_error.log       # Error log file

# Load necessary modules (e.g., CUDA)
module load CBI miniforge3/24.3.0-0
module load cuda/11.8
conda activate cellpath
export CUDA_VISIBLE_DEVICES=$SGE_GPU
export LD_LIBRARY_PATH=/wynton/home/gartner/zhuqin/.conda/envs/cellpath/lib:$LD_LIBRARY_PATH
script_name="cbce_scib"
python $script_name.py

## End-of-job summary, if running as a job
[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID"  # This is useful for debugging and usage purposes,
                                          # e.g. "did my job exceed its memory request?"


