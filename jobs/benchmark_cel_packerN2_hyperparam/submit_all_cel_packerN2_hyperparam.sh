#!/bin/bash
# Auto-generated — submits every job for this project
# Run from this folder, or let the script cd into it.

cd "$(dirname "$0")"

qsub "benchmark_cel_packerN2_hyperparam_concord_hcl_concord_hcl_batch_size-1024.sh"
qsub "benchmark_cel_packerN2_hyperparam_concord_hcl_concord_hcl_batch_size-64.sh"
