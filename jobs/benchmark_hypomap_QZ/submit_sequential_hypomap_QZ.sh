#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"           # work inside this folder
shopt -s nullglob              # empty globs expand to nothing

for job in *.py; do
  base=${job%.py}              # strip the .py suffix
  log="${base}.log"            # e.g. train_model.py → train_model.log

  echo ">>> $job   $(date)" | tee -a "$log"
  if python "$job" >>"$log" 2>&1; then
      echo ">>> finished OK"   | tee -a "$log"
  else
      echo ">>> FAILED"        | tee -a "$log"
  fi
done
