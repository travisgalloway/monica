#!/bin/bash
# Teacher top-k logit precompute (Step 3) — run once; both sweep layouts reuse it.
# Run detached: nohup bash scripts/runpod/m10/run_precompute.sh &
# Needs an Ampere+ 80 GB card (A100/H100) — run preflight.sh first.
set -euo pipefail

cd /workspace/monica
set -a; . /workspace/monica/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

mkdir -p /vol/corpus8k /vol/split8k /vol/teacher-outputs/topk-logits /workspace/runs

LOG=/workspace/runs/precompute.log
echo "PRECOMPUTE_START $(date)" | tee -a "$LOG"

echo "[precompute] r2_sync down qwen3-8k corpus" | tee -a "$LOG"
python -m src.data.r2_sync down \
  s3://monica-training/poc-distill/corpus/tokenized/qwen3-8k /vol/corpus8k \
  2>&1 | tee -a "$LOG"

echo "[precompute] split --shards" | tee -a "$LOG"
python -m src.data.split \
  --shards /vol/corpus8k --out /vol/split8k --val-tokens 10000000 \
  2>&1 | tee -a "$LOG"

echo "[precompute] precompute_teacher start $(date)" | tee -a "$LOG"
python scripts/precompute_teacher.py \
  --manifest config/manifests/student-1b-attn-hi.yaml \
  --data /vol/split8k \
  --splits train,val \
  --backend cuda \
  --k 50 \
  --batch-size 1 \
  --out /vol/teacher-outputs/topk-logits \
  --push s3://monica-training/poc-distill/teacher-outputs/topk-logits \
  2>&1 | tee -a "$LOG"

echo "PRECOMPUTE_EXITED code=${PIPESTATUS[0]} $(date)" | tee -a "$LOG"
