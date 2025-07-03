#!/bin/bash
# Auto-generated — sequentially runs every benchmark Python file
cd "$(dirname "$0")"

source ~/.bashrc
conda activate concord_env

timestamp=$(date +'%m%d-%H%M')

py_exec="${PYTHON_EXEC:-python}"

echo '🔄 Running: benchmark_pancreatic_islet_Hrovatin_concord_hcl.py (log: benchmark_pancreatic_islet_Hrovatin_concord_hcl_${timestamp}.log)'
${py_exec} benchmark_pancreatic_islet_Hrovatin_concord_hcl.py > benchmark_pancreatic_islet_Hrovatin_concord_hcl_${timestamp}.log 2>&1
echo '✅ Done.'

# echo '🔄 Running: benchmark_pancreatic_islet_Hrovatin_concord_knn.py (log: benchmark_pancreatic_islet_Hrovatin_concord_knn_${timestamp}.log)'
# ${py_exec} benchmark_pancreatic_islet_Hrovatin_concord_knn.py > benchmark_pancreatic_islet_Hrovatin_concord_knn_${timestamp}.log 2>&1
# echo '✅ Done.'

echo '🔄 Running: benchmark_pancreatic_islet_Hrovatin_contrastive.py (log: benchmark_pancreatic_islet_Hrovatin_contrastive_${timestamp}.log)'
${py_exec} benchmark_pancreatic_islet_Hrovatin_contrastive.py > benchmark_pancreatic_islet_Hrovatin_contrastive_${timestamp}.log 2>&1
echo '✅ Done.'

echo '🔄 Running: benchmark_pancreatic_islet_Hrovatin_harmony.py (log: benchmark_pancreatic_islet_Hrovatin_harmony_${timestamp}.log)'
${py_exec} benchmark_pancreatic_islet_Hrovatin_harmony.py > benchmark_pancreatic_islet_Hrovatin_harmony_${timestamp}.log 2>&1
echo '✅ Done.'

echo '🔄 Running: benchmark_pancreatic_islet_Hrovatin_liger.py (log: benchmark_pancreatic_islet_Hrovatin_liger_${timestamp}.log)'
${py_exec} benchmark_pancreatic_islet_Hrovatin_liger.py > benchmark_pancreatic_islet_Hrovatin_liger_${timestamp}.log 2>&1
echo '✅ Done.'

echo '🔄 Running: benchmark_pancreatic_islet_Hrovatin_scanorama.py (log: benchmark_pancreatic_islet_Hrovatin_scanorama_${timestamp}.log)'
${py_exec} benchmark_pancreatic_islet_Hrovatin_scanorama.py > benchmark_pancreatic_islet_Hrovatin_scanorama_${timestamp}.log 2>&1
echo '✅ Done.'

echo '🔄 Running: benchmark_pancreatic_islet_Hrovatin_scvi.py (log: benchmark_pancreatic_islet_Hrovatin_scvi_${timestamp}.log)'
${py_exec} benchmark_pancreatic_islet_Hrovatin_scvi.py > benchmark_pancreatic_islet_Hrovatin_scvi_${timestamp}.log 2>&1
echo '✅ Done.'

echo '🔄 Running: benchmark_pancreatic_islet_Hrovatin_unintegrated.py (log: benchmark_pancreatic_islet_Hrovatin_unintegrated_${timestamp}.log)'
${py_exec} benchmark_pancreatic_islet_Hrovatin_unintegrated.py > benchmark_pancreatic_islet_Hrovatin_unintegrated_${timestamp}.log 2>&1
echo '✅ Done.'

