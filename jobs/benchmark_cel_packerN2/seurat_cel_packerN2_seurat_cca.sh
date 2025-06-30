#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -r y
#$ -pe smp 1
#$ -l mem_free=32G
#$ -l scratch=50G
#$ -l h_rt=02:00:00

echo "Running on: $(hostname)"

module load CBI miniforge3/24.3.0-0
conda activate cellpath
export LD_LIBRARY_PATH=/wynton/home/gartner/zhuqin/.conda/envs/cellpath/lib:$LD_LIBRARY_PATH

Rscript cbce_run_seurat_rPCA.R

TIMESTAMP=$(date +'%m%d-%H%M')
Rscript seurat_cel_packerN2_seurat_cca.R --timestamp $TIMESTAMP
