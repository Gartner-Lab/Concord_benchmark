#!/bin/bash
# Auto-generated — sequentially runs every job script
cd "$(dirname "$0")"

timestamp=$(date +'%m%d-%H%M')

echo '🔄 Running: benchmark_dkd_Wilson_concord_hcl.sh (log: benchmark_dkd_Wilson_concord_hcl_${timestamp}.log)'
bash benchmark_dkd_Wilson_concord_hcl.sh > benchmark_dkd_Wilson_concord_hcl_${timestamp}.log 2>&1
echo '✅ Done.'

echo '🔄 Running: benchmark_dkd_Wilson_concord_knn.sh (log: benchmark_dkd_Wilson_concord_knn_${timestamp}.log)'
bash benchmark_dkd_Wilson_concord_knn.sh > benchmark_dkd_Wilson_concord_knn_${timestamp}.log 2>&1
echo '✅ Done.'

echo '🔄 Running: benchmark_dkd_Wilson_contrastive.sh (log: benchmark_dkd_Wilson_contrastive_${timestamp}.log)'
bash benchmark_dkd_Wilson_contrastive.sh > benchmark_dkd_Wilson_contrastive_${timestamp}.log 2>&1
echo '✅ Done.'

echo '🔄 Running: benchmark_dkd_Wilson_harmony.sh (log: benchmark_dkd_Wilson_harmony_${timestamp}.log)'
bash benchmark_dkd_Wilson_harmony.sh > benchmark_dkd_Wilson_harmony_${timestamp}.log 2>&1
echo '✅ Done.'

echo '🔄 Running: benchmark_dkd_Wilson_liger.sh (log: benchmark_dkd_Wilson_liger_${timestamp}.log)'
bash benchmark_dkd_Wilson_liger.sh > benchmark_dkd_Wilson_liger_${timestamp}.log 2>&1
echo '✅ Done.'

echo '🔄 Running: benchmark_dkd_Wilson_scanorama.sh (log: benchmark_dkd_Wilson_scanorama_${timestamp}.log)'
bash benchmark_dkd_Wilson_scanorama.sh > benchmark_dkd_Wilson_scanorama_${timestamp}.log 2>&1
echo '✅ Done.'

echo '🔄 Running: benchmark_dkd_Wilson_scvi.sh (log: benchmark_dkd_Wilson_scvi_${timestamp}.log)'
bash benchmark_dkd_Wilson_scvi.sh > benchmark_dkd_Wilson_scvi_${timestamp}.log 2>&1
echo '✅ Done.'

echo '🔄 Running: benchmark_dkd_Wilson_unintegrated.sh (log: benchmark_dkd_Wilson_unintegrated_${timestamp}.log)'
bash benchmark_dkd_Wilson_unintegrated.sh > benchmark_dkd_Wilson_unintegrated_${timestamp}.log 2>&1
echo '✅ Done.'

