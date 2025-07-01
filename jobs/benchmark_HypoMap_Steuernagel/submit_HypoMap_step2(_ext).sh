#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -r y
#$ -l mem_free=32G
#$ -l scratch=50G
#$ -l h_rt=03:00:00

source ~/.bashrc
conda activate scenv

python ../notebook/benchmark_HypoMap_Steuernagel_step2_extend.py

