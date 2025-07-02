#!/bin/bash
# Auto-generated — sequentially runs every benchmark Python file
cd "$(dirname "$0")"

source ~/.bashrc
conda activate concord_env

timestamp=$(date +'%m%d-%H%M')

py_exec="${PYTHON_EXEC:-python}"

echo '🔄 Running: benchmark_dkd_Wilson_concord_hcl.py (log: benchmark_dkd_Wilson_concord_hcl_${timestamp}.log)'
${py_exec} benchmark_dkd_Wilson_concord_hcl.py > benchmark_dkd_Wilson_concord_hcl_${timestamp}.log 2>&1
echo '✅ Done.'

echo '🔄 Running: benchmark_dkd_Wilson_concord_knn.py (log: benchmark_dkd_Wilson_concord_knn_${timestamp}.log)'
${py_exec} benchmark_dkd_Wilson_concord_knn.py > benchmark_dkd_Wilson_concord_knn_${timestamp}.log 2>&1
echo '✅ Done.'

echo '🔄 Running: benchmark_dkd_Wilson_contrastive.py (log: benchmark_dkd_Wilson_contrastive_${timestamp}.log)'
${py_exec} benchmark_dkd_Wilson_contrastive.py > benchmark_dkd_Wilson_contrastive_${timestamp}.log 2>&1
echo '✅ Done.'

echo '🔄 Running: benchmark_dkd_Wilson_harmony.py (log: benchmark_dkd_Wilson_harmony_${timestamp}.log)'
${py_exec} benchmark_dkd_Wilson_harmony.py > benchmark_dkd_Wilson_harmony_${timestamp}.log 2>&1
echo '✅ Done.'

echo '🔄 Running: benchmark_dkd_Wilson_liger.py (log: benchmark_dkd_Wilson_liger_${timestamp}.log)'
${py_exec} benchmark_dkd_Wilson_liger.py > benchmark_dkd_Wilson_liger_${timestamp}.log 2>&1
echo '✅ Done.'

echo '🔄 Running: benchmark_dkd_Wilson_scanorama.py (log: benchmark_dkd_Wilson_scanorama_${timestamp}.log)'
${py_exec} benchmark_dkd_Wilson_scanorama.py > benchmark_dkd_Wilson_scanorama_${timestamp}.log 2>&1
echo '✅ Done.'

echo '🔄 Running: benchmark_dkd_Wilson_scvi.py (log: benchmark_dkd_Wilson_scvi_${timestamp}.log)'
${py_exec} benchmark_dkd_Wilson_scvi.py > benchmark_dkd_Wilson_scvi_${timestamp}.log 2>&1
echo '✅ Done.'

echo '🔄 Running: benchmark_dkd_Wilson_unintegrated.py (log: benchmark_dkd_Wilson_unintegrated_${timestamp}.log)'
${py_exec} benchmark_dkd_Wilson_unintegrated.py > benchmark_dkd_Wilson_unintegrated_${timestamp}.log 2>&1
echo '✅ Done.'

