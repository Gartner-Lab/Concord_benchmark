#!/bin/bash
# Auto-generated — sequentially runs every benchmark Python file
cd "$(dirname "$0")"

source ~/.bashrc
conda activate concord_env

timestamp=$(date +'%m%d-%H%M')

py_exec="${PYTHON_EXEC:-python}"

echo '🔄 Running: benchmark_immune_DominguezConde_concord_hcl.py (log: benchmark_immune_DominguezConde_concord_hcl_${timestamp}.log)'
${py_exec} benchmark_immune_DominguezConde_concord_hcl.py > benchmark_immune_DominguezConde_concord_hcl_${timestamp}.log 2>&1
echo '✅ Done.'

# echo '🔄 Running: benchmark_immune_DominguezConde_concord_knn.py (log: benchmark_immune_DominguezConde_concord_knn_${timestamp}.log)'
# ${py_exec} benchmark_immune_DominguezConde_concord_knn.py > benchmark_immune_DominguezConde_concord_knn_${timestamp}.log 2>&1
# echo '✅ Done.'

echo '🔄 Running: benchmark_immune_DominguezConde_contrastive.py (log: benchmark_immune_DominguezConde_contrastive_${timestamp}.log)'
${py_exec} benchmark_immune_DominguezConde_contrastive.py > benchmark_immune_DominguezConde_contrastive_${timestamp}.log 2>&1
echo '✅ Done.'

echo '🔄 Running: benchmark_immune_DominguezConde_harmony.py (log: benchmark_immune_DominguezConde_harmony_${timestamp}.log)'
${py_exec} benchmark_immune_DominguezConde_harmony.py > benchmark_immune_DominguezConde_harmony_${timestamp}.log 2>&1
echo '✅ Done.'

echo '🔄 Running: benchmark_immune_DominguezConde_liger.py (log: benchmark_immune_DominguezConde_liger_${timestamp}.log)'
${py_exec} benchmark_immune_DominguezConde_liger.py > benchmark_immune_DominguezConde_liger_${timestamp}.log 2>&1
echo '✅ Done.'

echo '🔄 Running: benchmark_immune_DominguezConde_scanorama.py (log: benchmark_immune_DominguezConde_scanorama_${timestamp}.log)'
${py_exec} benchmark_immune_DominguezConde_scanorama.py > benchmark_immune_DominguezConde_scanorama_${timestamp}.log 2>&1
echo '✅ Done.'

echo '🔄 Running: benchmark_immune_DominguezConde_scvi.py (log: benchmark_immune_DominguezConde_scvi_${timestamp}.log)'
${py_exec} benchmark_immune_DominguezConde_scvi.py > benchmark_immune_DominguezConde_scvi_${timestamp}.log 2>&1
echo '✅ Done.'

echo '🔄 Running: benchmark_immune_DominguezConde_unintegrated.py (log: benchmark_immune_DominguezConde_unintegrated_${timestamp}.log)'
${py_exec} benchmark_immune_DominguezConde_unintegrated.py > benchmark_immune_DominguezConde_unintegrated_${timestamp}.log 2>&1
echo '✅ Done.'

