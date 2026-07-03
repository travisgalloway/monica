#!/bin/bash
# Cluster-mode teacher precompute shard — one pod of N in parallel.
# Each pod processes 1/NUM_SHARDS of the train chunks independently.
# Shard 0 also computes the val split (cheap; no need to parallelize).
#
# Usage: SHARD_ID=0 NUM_SHARDS=4 bash run_precompute_shard.sh
# Or pass as positional args: bash run_precompute_shard.sh 0 4
#
# After all N shards complete, run merge_teacher_shards.py on any pod or locally:
#   python scripts/merge_teacher_shards.py \
#     --source s3://monica-training/poc-distill/teacher-outputs/topk-logits \
#     --num-shards 4 \
#     --out /vol/teacher-outputs/topk-logits-merged \
#     --push s3://monica-training/poc-distill/teacher-outputs/topk-logits-merged
set -euo pipefail

SHARD_ID="${1:-${SHARD_ID:-}}"
NUM_SHARDS="${2:-${NUM_SHARDS:-}}"

if [ -z "$SHARD_ID" ] || [ -z "$NUM_SHARDS" ]; then
  echo "ERROR: set SHARD_ID and NUM_SHARDS (or pass as positional args)" >&2
  exit 1
fi

cd /workspace/monica
set -a; . /workspace/monica/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

mkdir -p /vol/corpus8k /vol/split8k \
         "/vol/teacher-outputs/topk-logits/shard-${SHARD_ID}" /workspace/runs

LOG="/workspace/runs/precompute-shard-${SHARD_ID}.log"
echo "PRECOMPUTE_SHARD_START shard=${SHARD_ID}/${NUM_SHARDS} $(date)" | tee -a "$LOG"

echo "[shard-${SHARD_ID}] r2_sync down qwen3-8k corpus" | tee -a "$LOG"
python -m src.data.r2_sync down \
  s3://monica-training/poc-distill/corpus/tokenized/qwen3-8k /vol/corpus8k \
  2>&1 | tee -a "$LOG"

# Restore any partial precompute output from R2 — enables resume after pod billing stop.
# precompute_teacher.py's auto-resume logic detects existing bytes and skips done chunks.
echo "[shard-${SHARD_ID}] r2_sync down partial precompute output (resume checkpoint)" | tee -a "$LOG"
python -m src.data.r2_sync down \
  "s3://monica-training/poc-distill/teacher-outputs/topk-logits/shard-${SHARD_ID}" \
  "/vol/teacher-outputs/topk-logits/shard-${SHARD_ID}" \
  2>&1 | tee -a "$LOG" || echo "[shard-${SHARD_ID}] no prior checkpoint on R2 (fresh start)" | tee -a "$LOG"

echo "[shard-${SHARD_ID}] split --shards" | tee -a "$LOG"
python -m src.data.split \
  --shards /vol/corpus8k --out /vol/split8k --val-tokens 10000000 \
  2>&1 | tee -a "$LOG"

# Shard 0 also handles val (cheap compared to train; no need to parallelize)
SPLITS_ARG="train"
if [ "$SHARD_ID" -eq 0 ]; then
  SPLITS_ARG="train,val"
  echo "[shard-0] will also process val split" | tee -a "$LOG"
else
  # Guard against a reused /vol carrying stale val artifacts from an earlier shard-0 run —
  # this shard won't rewrite them, but the R2 checkpoint loop uploads the whole shard dir,
  # which would silently contaminate this shard's R2 prefix with val data.
  rm -f "/vol/teacher-outputs/topk-logits/shard-${SHARD_ID}"/teacher-val.* 2>/dev/null || true
fi

# Background R2 checkpoint: sync output every 2 hours so a billing stop loses at most 2h.
# Runs in a subshell so set -e in main script doesn't kill it on transient upload errors.
(
  while sleep 7200; do
    echo "[r2-ckpt] shard-${SHARD_ID} syncing to R2 $(date)" >> "$LOG"
    python -m src.data.r2_sync up \
      "/vol/teacher-outputs/topk-logits/shard-${SHARD_ID}" \
      "s3://monica-training/poc-distill/teacher-outputs/topk-logits/shard-${SHARD_ID}" \
      >> "$LOG" 2>&1 || echo "[r2-ckpt] upload failed (non-fatal)" >> "$LOG"
  done
) &
R2_CKPT_PID=$!
trap 'kill "$R2_CKPT_PID" 2>/dev/null || true' EXIT INT TERM

echo "[shard-${SHARD_ID}] precompute_teacher start $(date)" | tee -a "$LOG"
python scripts/precompute_teacher.py \
  --manifest config/manifests/student-1b-attn-hi.yaml \
  --data /vol/split8k \
  --splits "$SPLITS_ARG" \
  --backend cuda \
  --k 50 \
  --batch-size 1 \
  --shard-id "$SHARD_ID" \
  --num-shards "$NUM_SHARDS" \
  --out "/vol/teacher-outputs/topk-logits" \
  --push s3://monica-training/poc-distill/teacher-outputs/topk-logits \
  2>&1 | tee -a "$LOG"

echo "PRECOMPUTE_SHARD_DONE shard=${SHARD_ID}/${NUM_SHARDS} $(date)" | tee -a "$LOG"
