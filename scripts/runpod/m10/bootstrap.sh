#!/bin/bash
# One-time pod bootstrap for the M10 distillation chain (Phase B).
# Run once after the pod is up; idempotent (safe to re-run).
#
# Prereqs on the pod:
#   - /workspace/ exists (RunPod default)
#   - SSH key authorised (pod_create JSON → PUBLIC_KEY)
#   - .env placed at /workspace/monica/.env (R2 creds + HF_TOKEN) before this
#     script sources it (or set manually in the shell first)
#
# Mirrors the proven poc-qwen pattern from ~/.claude/monica-runpod-ops/run_train.sh.
set -euo pipefail

REPO="https://github.com/travisgalloway/monica"
WORKSPACE=/workspace/monica

echo "=== M10 bootstrap: $(date) ==="

# --- 1. Clone (skip if already present) ---
if [ ! -d "$WORKSPACE/.git" ]; then
  echo "[bootstrap] cloning $REPO -> $WORKSPACE"
  git clone "$REPO" "$WORKSPACE"
fi
cd "$WORKSPACE"
echo "[bootstrap] repo at $(git rev-parse --short HEAD)"

# --- 2. Install: cuda-fast for fused Mamba Triton scan + causal-conv1d (#40) ---
echo "[bootstrap] pip install cuda-fast"
pip install -e ".[dev,data,cuda-fast]"

# --- 3. s3fs pin — a bare install upgrades fsspec and breaks datasets (#memory: s3fs-fsspec-pin) ---
echo "[bootstrap] pin s3fs==2026.2.0"
pip install "s3fs==2026.2.0"

# --- 4. Source R2 creds (.env must already exist at $WORKSPACE/.env) ---
if [ -f "$WORKSPACE/.env" ]; then
  set -a; . "$WORKSPACE/.env"; set +a
  echo "[bootstrap] .env sourced (R2 creds + HF_TOKEN)"
else
  echo "[bootstrap] WARNING: $WORKSPACE/.env not found — r2_sync and HF downloads will fail"
fi

# --- 5. Pod-wide env vars (write to /etc/environment for nohup/tmux sessions) ---
echo "[bootstrap] setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
# Make these survive nohup / new shells
grep -q "PYTORCH_CUDA_ALLOC_CONF" /etc/environment 2>/dev/null || \
  echo "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" >> /etc/environment
grep -q "PYTHONUNBUFFERED" /etc/environment 2>/dev/null || \
  echo "PYTHONUNBUFFERED=1" >> /etc/environment

echo "=== M10 bootstrap: DONE $(date) ==="
