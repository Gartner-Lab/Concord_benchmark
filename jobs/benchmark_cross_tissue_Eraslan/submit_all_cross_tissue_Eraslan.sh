#!/bin/bash
# Auto-generated — sequentially runs every benchmark Python file
cd "$(dirname "$0")"

source ~/.bashrc
conda activate concord_env

timestamp=$(date +'%m%d-%H%M')

py_exec="${PYTHON_EXEC:-python}"

echo '🔄 Running: benchmark_cross_tissue_Eraslan_concord_hcl.py (log: benchmark_cross_tissue_Eraslan_concord_hcl_${timestamp}.log)'
${py_exec} benchmark_cross_tissue_Eraslan_concord_hcl.py > benchmark_cross_tissue_Eraslan_concord_hcl_${timestamp}.log 2>&1
echo '✅ Done.'

echo '🔄 Running: benchmark_cross_tissue_Eraslan_contrastive.py (log: benchmark_cross_tissue_Eraslan_contrastive_${timestamp}.log)'
${py_exec} benchmark_cross_tissue_Eraslan_contrastive.py > benchmark_cross_tissue_Eraslan_contrastive_${timestamp}.log 2>&1
echo '✅ Done.'

echo '🔄 Running: benchmark_cross_tissue_Eraslan_harmony.py (log: benchmark_cross_tissue_Eraslan_harmony_${timestamp}.log)'
${py_exec} benchmark_cross_tissue_Eraslan_harmony.py > benchmark_cross_tissue_Eraslan_harmony_${timestamp}.log 2>&1
echo '✅ Done.'

echo '🔄 Running: benchmark_cross_tissue_Eraslan_liger.py (log: benchmark_cross_tissue_Eraslan_liger_${timestamp}.log)'
${py_exec} benchmark_cross_tissue_Eraslan_liger.py > benchmark_cross_tissue_Eraslan_liger_${timestamp}.log 2>&1
echo '✅ Done.'

echo '🔄 Running: benchmark_cross_tissue_Eraslan_scanorama.py (log: benchmark_cross_tissue_Eraslan_scanorama_${timestamp}.log)'
${py_exec} benchmark_cross_tissue_Eraslan_scanorama.py > benchmark_cross_tissue_Eraslan_scanorama_${timestamp}.log 2>&1
echo '✅ Done.'

echo '🔄 Running: benchmark_cross_tissue_Eraslan_scvi.py (log: benchmark_cross_tissue_Eraslan_scvi_${timestamp}.log)'
${py_exec} benchmark_cross_tissue_Eraslan_scvi.py > benchmark_cross_tissue_Eraslan_scvi_${timestamp}.log 2>&1
echo '✅ Done.'

echo '🔄 Running: benchmark_cross_tissue_Eraslan_unintegrated.py (log: benchmark_cross_tissue_Eraslan_unintegrated_${timestamp}.log)'
${py_exec} benchmark_cross_tissue_Eraslan_unintegrated.py > benchmark_cross_tissue_Eraslan_unintegrated_${timestamp}.log 2>&1
echo '✅ Done.'

