#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"            # work inside this folder (../jobs)
shopt -s nullglob

for folder in benchmark_cel_packerN2_downsample_*; do
  [[ -d "$folder" ]] || continue
  echo "===== entering $folder  $(date) ====="

  for job in "$folder"/*.py; do
    [[ -e "$job" ]] || continue

    base=${job%.py}
    log="${base}.log"

    # ───────────────────────────────────────────────────────────────
    # skip if a previous run finished successfully
    #   • If you only care that the log exists (no success check),
    #     drop the grep clause.
    # ───────────────────────────────────────────────────────────────
    if [[ -f "$log" ]] && grep -q "finished OK" "$log"; then
        echo ">>> SKIP $job  — already completed"
        continue
    fi

    echo ">>> $job   $(date)" | tee -a "$log"
    if python "$job" >>"$log" 2>&1; then
        echo ">>> finished OK" | tee -a "$log"
    else
        echo ">>> FAILED"      | tee -a "$log"
    fi
  done
done
