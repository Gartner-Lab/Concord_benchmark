#!/bin/bash
# Auto-generated — sequentially runs every benchmark Python file
cd "$(dirname "$0")"

source ~/.bashrc
conda activate concord_env

timestamp=$(date +'%m%d-%H%M')

py_exec="${PYTHON_EXEC:-python}"

echo '🔄 Running: benchmark_HypoMap_Steuernagel_concord_hcl.py (log: benchmark_HypoMap_Steuernagel_concord_hcl_${timestamp}.log)'
${py_exec} benchmark_HypoMap_Steuernagel_concord_hcl.py > benchmark_HypoMap_Steuernagel_concord_hcl_${timestamp}.log 2>&1
echo '✅ Done.'

echo '🔄 Running: benchmark_HypoMap_Steuernagel_concord_knn.py (log: benchmark_HypoMap_Steuernagel_concord_knn_${timestamp}.log)'
${py_exec} benchmark_HypoMap_Steuernagel_concord_knn.py > benchmark_HypoMap_Steuernagel_concord_knn_${timestamp}.log 2>&1
echo '✅ Done.'

echo '🔄 Running: benchmark_HypoMap_Steuernagel_contrastive.py (log: benchmark_HypoMap_Steuernagel_contrastive_${timestamp}.log)'
${py_exec} benchmark_HypoMap_Steuernagel_contrastive.py > benchmark_HypoMap_Steuernagel_contrastive_${timestamp}.log 2>&1
echo '✅ Done.'

echo '🔄 Running: benchmark_HypoMap_Steuernagel_harmony.py (log: benchmark_HypoMap_Steuernagel_harmony_${timestamp}.log)'
${py_exec} benchmark_HypoMap_Steuernagel_harmony.py > benchmark_HypoMap_Steuernagel_harmony_${timestamp}.log 2>&1
echo '✅ Done.'

echo '🔄 Running: benchmark_HypoMap_Steuernagel_liger.py (log: benchmark_HypoMap_Steuernagel_liger_${timestamp}.log)'
${py_exec} benchmark_HypoMap_Steuernagel_liger.py > benchmark_HypoMap_Steuernagel_liger_${timestamp}.log 2>&1
echo '✅ Done.'

echo '🔄 Running: benchmark_HypoMap_Steuernagel_scanorama.py (log: benchmark_HypoMap_Steuernagel_scanorama_${timestamp}.log)'
${py_exec} benchmark_HypoMap_Steuernagel_scanorama.py > benchmark_HypoMap_Steuernagel_scanorama_${timestamp}.log 2>&1
echo '✅ Done.'

echo '🔄 Running: benchmark_HypoMap_Steuernagel_scvi.py (log: benchmark_HypoMap_Steuernagel_scvi_${timestamp}.log)'
${py_exec} benchmark_HypoMap_Steuernagel_scvi.py > benchmark_HypoMap_Steuernagel_scvi_${timestamp}.log 2>&1
echo '✅ Done.'

echo '🔄 Running: benchmark_HypoMap_Steuernagel_unintegrated.py (log: benchmark_HypoMap_Steuernagel_unintegrated_${timestamp}.log)'
${py_exec} benchmark_HypoMap_Steuernagel_unintegrated.py > benchmark_HypoMap_Steuernagel_unintegrated_${timestamp}.log 2>&1
echo '✅ Done.'

