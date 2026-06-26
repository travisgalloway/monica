#!/bin/bash
# Post-training for the sweep winner (Step 5): instruct SFT -> reasoning SFT -> GRPO.
# Usage: bash scripts/runpod/m10/run_posttrain.sh <winner-manifest>
#   e.g. bash scripts/runpod/m10/run_posttrain.sh student-1b-attn-hi
#
# Reads SFT corpora from s3://monica-training/shared/sft/tokenized/qwen3-8k (staged R2).
# Syncs portable safetensors to s3://monica-training/models/m10-winner/ after each step.
# Re-run with the same command to resume (sft.py auto-detects <out>/resume).
set -euo pipefail

cd /workspace/monica
set -a; . /workspace/monica/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

M="${1:?usage: $0 <winner-manifest> e.g. student-1b-attn-hi}"

case "$M" in
  student-1b-attn-hi) CFG=config/student-1b.yaml ;;
  student-1b-attn-lo) CFG=config/student-1b-attn-lo.yaml ;;
  *) echo "ERROR: unknown manifest '$M'; expected student-1b-attn-hi or student-1b-attn-lo"; exit 1 ;;
esac

BASE_WEIGHTS=/vol/runs/$M/weights.safetensors
OUT=/vol/runs/$M
LOG=$OUT/posttrain.log
mkdir -p "$OUT"

echo "POSTTRAIN_START manifest=$M config=$CFG $(date)" | tee -a "$LOG"

# ─── Download SFT corpora ─────────────────────────────────────────────────────
echo "[posttrain] r2_sync down SFT corpora" | tee -a "$LOG"
mkdir -p /vol/sft
python -m src.data.r2_sync down \
  s3://monica-training/shared/sft/tokenized/qwen3-8k /vol/sft \
  2>&1 | tee -a "$LOG"

# sft.py expects train.jsonl / val.jsonl in the --data dir; the corpus builder writes
# flat instruct.jsonl / reasoning.jsonl — split 90/10 by line count here.
python -c "
from pathlib import Path
import math

for name in ('instruct', 'reasoning'):
    src = Path('/vol/sft') / f'{name}.jsonl'
    if not src.exists():
        print(f'  {name}.jsonl not found — skipping split')
        continue
    lines = src.read_text(encoding='utf-8').splitlines()
    n = len(lines)
    cut = max(1, math.floor(n * 0.9))
    out_dir = Path(f'/vol/sft/{name}')
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'train.jsonl').write_text('\n'.join(lines[:cut]) + '\n', encoding='utf-8')
    (out_dir / 'val.jsonl').write_text('\n'.join(lines[cut:]) + '\n', encoding='utf-8')
    print(f'  {name}: {cut} train, {n - cut} val')
" 2>&1 | tee -a "$LOG"

# ─── Phase 1: instruct SFT ────────────────────────────────────────────────────
echo "INSTRUCT_SFT_START $(date)" | tee -a "$LOG"
python scripts/sft.py \
  --config "$CFG" \
  --data /vol/sft/instruct \
  --init "$BASE_WEIGHTS" \
  --out "$OUT/sft-instruct" \
  --backend cuda \
  --epochs 2 \
  --batch-size 2 \
  --grad-accum 16 \
  --base-lr 2e-5 \
  --eval-every 100 \
  --ckpt-every 200 \
  2>&1 | tee -a "$LOG"
echo "INSTRUCT_SFT_EXITED code=${PIPESTATUS[0]} $(date)" | tee -a "$LOG"

python -m src.data.r2_sync up "$OUT/sft-instruct" \
  s3://monica-training/models/m10-winner/sft-instruct \
  2>&1 | tee -a "$LOG"

# ─── Phase 2: reasoning SFT ───────────────────────────────────────────────────
echo "REASONING_SFT_START $(date)" | tee -a "$LOG"
python scripts/sft.py \
  --config "$CFG" \
  --data /vol/sft/reasoning \
  --init "$OUT/sft-instruct/weights.safetensors" \
  --out "$OUT/sft-reasoning" \
  --backend cuda \
  --epochs 2 \
  --batch-size 2 \
  --grad-accum 16 \
  --base-lr 1e-5 \
  --eval-every 100 \
  --ckpt-every 200 \
  2>&1 | tee -a "$LOG"
echo "REASONING_SFT_EXITED code=${PIPESTATUS[0]} $(date)" | tee -a "$LOG"

python -m src.data.r2_sync up "$OUT/sft-reasoning" \
  s3://monica-training/models/m10-winner/sft-reasoning \
  2>&1 | tee -a "$LOG"

# ─── Phase 3: GRPO ────────────────────────────────────────────────────────────
echo "[posttrain] r2_sync down RL problems" | tee -a "$LOG"
mkdir -p /vol/rl
python -m src.data.r2_sync down s3://monica-training/shared/rl /vol/rl \
  2>&1 | tee -a "$LOG"

echo "GRPO_START $(date)" | tee -a "$LOG"
python scripts/rlvr.py \
  --config "$CFG" \
  --init "$OUT/sft-reasoning/weights.safetensors" \
  --problems /vol/rl/problems.jsonl \
  --out "$OUT/rlvr" \
  --steps 200 \
  --ckpt-every 50 \
  2>&1 | tee -a "$LOG"
echo "GRPO_EXITED code=${PIPESTATUS[0]} $(date)" | tee -a "$LOG"

python -m src.data.r2_sync up "$OUT/rlvr" \
  s3://monica-training/models/m10-winner/rlvr \
  2>&1 | tee -a "$LOG"

echo "POSTTRAIN_DONE manifest=$M $(date)" | tee -a "$LOG"
