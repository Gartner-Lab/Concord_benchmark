#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"            # work inside this folder
shopt -s nullglob               # empty globs expand to nothing

for job in *.py; do
    base=${job%.py}
    log="${base}.log"

    # ── 1. decide whether we need to run ──────────────────────────────
    if [[ -f $log ]] && tail -n 1 "$log" | grep -q ">>> finished OK"; then
        printf '⏭  %-60s  (already done)\n' "$job"
        continue
    fi

    # ── 2. (re)run the job ────────────────────────────────────────────
    printf '▶︎   %-60s  %(%F %T)T\n' "$job" -1 | tee -a "$log"
    if python "$job" >>"$log" 2>&1; then
        echo ">>> finished OK" | tee -a "$log"
    else
        echo ">>> FAILED"      | tee -a "$log"
    fi
done

