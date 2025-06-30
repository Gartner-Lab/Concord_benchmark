#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"           # work inside this folder
shopt -s nullglob              # empty globs expand to nothing

for job in *.py; do            # <<──  changed from jobs/*.py to *.py
  echo ">>> $job   $(date)" | tee -a all_runs.log
  if python "$job" >> all_runs.log 2>&1; then
      echo ">>> finished OK"   | tee -a all_runs.log
  else
      echo ">>> FAILED"        | tee -a all_runs.log
  fi
done
