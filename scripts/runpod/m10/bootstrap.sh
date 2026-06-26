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

# --- 2. Install: base first, then mamba-ssm/causal-conv1d with --no-build-isolation.
#    mamba-ssm's setup.py imports torch at build time; pip's isolated build env lacks it,
#    so the plain `pip install -e ".[cuda-fast]"` form fails. Installing the base first
#    (which resolves torch from the image) then building the Triton kernels against it works.
echo "[bootstrap] pip install base (cuda)"
pip install -e ".[dev,data,cuda]" -q
echo "[bootstrap] pip install mamba-ssm + causal-conv1d (no-build-isolation)"
pip install "mamba-ssm>=2.0" "causal-conv1d>=1.0" --no-build-isolation

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
