#!/bin/bash
# Periodic R2 checkpoint sync for the distill sweep — run detached alongside run_distill.sh.
# Syncs both sweep layouts from /vol/runs to s3://monica-training/ckpt/.
# Usage: nohup bash scripts/runpod/m10/run_sync.sh &
set -euo pipefail

cd /workspace/monica
set -a; . /workspace/monica/.env; set +a

while true; do
  sleep 1800
  for M in student-1b-attn-hi student-1b-attn-lo; do
    if [ -d "/vol/runs/$M" ]; then
      python -m src.data.r2_sync up /vol/runs/$M s3://monica-training/ckpt/$M \
        >> /workspace/sync.log 2>&1
    fi
  done
  echo "synced $(date)" >> /workspace/sync.log
done
