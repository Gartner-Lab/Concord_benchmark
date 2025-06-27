#!/bin/bash
# Auto-generated — submits every job for this project
# Run from this folder, or let the script cd into it.

cd "$(dirname "$0")"

qsub "benchmark_cel_packerN2_concord_hcl.sh"
qsub "benchmark_cel_packerN2_concord_knn.sh"
qsub "benchmark_cel_packerN2_contrastive.sh"
qsub "benchmark_cel_packerN2_harmony.sh"
qsub "benchmark_cel_packerN2_liger.sh"
qsub "benchmark_cel_packerN2_scanorama.sh"
qsub "benchmark_cel_packerN2_scvi.sh"
qsub "benchmark_cel_packerN2_unintegrated.sh"
