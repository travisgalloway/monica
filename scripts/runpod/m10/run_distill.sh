#!/bin/bash
# Two-layout student sweep (Step 4) — runs after run_precompute.sh completes.
# Run detached alongside run_sync.sh: nohup bash scripts/runpod/m10/run_distill.sh &
#
# MOHAWK stage-1 OOM risk (O(L²)): mixing-match materializes a (B,H,L,L) mixing matrix per
# layer for student AND teacher. At seq 8192 this is ~64x tighter than the seq-1024 Path-A
# run that already OOM'd an 80 GB card at batch 8 — hence batch-size 1 + PYTORCH_CUDA_ALLOC_CONF.
# Later stages are looser. If OOM persists on stage-1, reduce --batch-size to 1 and raise
# --grad-accum to maintain effective batch tokens. Re-run with --resume after any interruption.
set -euo pipefail

cd /workspace/monica
set -a; . /workspace/monica/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

mkdir -p /vol/runs

for M in student-1b-attn-hi student-1b-attn-lo; do
  LOG=/vol/runs/$M/distill.log
  mkdir -p /vol/runs/$M

  echo "DISTILL_START manifest=$M $(date)" | tee -a "$LOG"

  python scripts/distill.py \
    --manifest config/manifests/$M.yaml \
    --corpus /vol/split8k \
    --teacher-outputs /vol/teacher-outputs/topk-logits \
    --backend cuda \
    --out /vol/runs/$M \
    --batch-size 1 \
    --grad-accum 16 \
    --steps-per-stage 1000 \
    --k 50 \
    --temperature 2.0 \
    --ce-weight 0.1 \
    --kl-weight 0.9 \
    --eval-every 200 \
    --ckpt-every 500 \
    2>&1 | tee -a "$LOG"

  echo "DISTILL_EXITED manifest=$M code=${PIPESTATUS[0]} $(date)" | tee -a "$LOG"

  echo "SYNC_START manifest=$M $(date)" | tee -a "$LOG"
  python -m src.data.r2_sync up /vol/runs/$M s3://monica-training/ckpt/$M \
    2>&1 | tee -a "$LOG"
  echo "SYNC_DONE manifest=$M $(date)" | tee -a "$LOG"
done
