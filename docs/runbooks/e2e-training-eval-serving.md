# End-to-End Operational Runbook: Training, Evaluation, and Serving (POC to MVP)

[← Design Docs](../design/README.md) | [Target Configuration Matrix](../design/16-target-configuration-matrix.md) | [Feature Matrix](../feature-matrix.md)

This runbook documents the complete end-to-end lifecycle for Monica across training, verification, evaluation, quantization, and model serving. It covers all three target tiers specified in [16-target-configuration-matrix.md](../design/16-target-configuration-matrix.md):
- **Tier 1: Mac-Trained POC** (~100M params, ~64k context window)
- **Tier 2: CUDA-Trained POC** (~1B active params, 128k context window)
- **Tier 3: CUDA-Trained MVP** (~4B active params, 128k–256k context window)

All tiers standardize on two deployment precision regimes:
- **Native FP16 / BF16** (unquantized reference weights & KV cache)
- **Mixed Precision W4 + KV8** (4-bit affine hidden weights, 8-bit tied embedding head, 8-bit attention KV cache, and continuous FP32 recurrent SSM state)

---

## 1. Prerequisites and Environment Setup

### 1.1 macOS / Apple Silicon (Tier 1 Mac POC)

Requires macOS 14+, Xcode CLI tools, and Python 3.11+.

```bash
# Clone repository
git clone https://github.com/travisgalloway/monica.git
cd monica

# Create and activate virtualenv
python3 -m venv .venv
source .venv/bin/activate

# Install with Apple Silicon MLX and development dependencies
pip install -e ".[dev,data,mlx]"

# Build native Swift tokenizer and engine tools
(cd swift && swift build -c release --build-system native)

# Install Task runner (optional, or use make)
# brew install go-task
```

### 1.2 Linux / CUDA Host (Tier 2 POC & Tier 3 MVP)

Requires Ubuntu 22.04+, NVIDIA Driver 550+, CUDA 12.4+, PyTorch 2.4+.

```bash
# Create and activate virtualenv
python3 -m venv .venv
source .venv/bin/activate

# Install with CUDA and Triton fast-paths
pip install -e ".[dev,data,cuda,cuda-fast,eval]"

# Verify GPU visibility and Triton kernels
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
```

---

## 2. Stage 1: Data Preparation & Tokenization

Monica uses an in-house byte-level BPE tokenizer (`vocab_size=49,152`, uint16 packing) built natively in Swift (`monica-tokenize`) with full TypeScript and Python keyword pre-tokenization.

### 2.1 Build Decontamination Blocklist

Before tokenizing training sets, generate the contamination blocklist from held-out evaluation sets to prevent benchmark memorization:

```bash
# Generate blocklist and manifest
python scripts/build_decontam_blocklist.py \
  --out eval_sets/decontam/blocklist.txt \
  --manifest eval_sets/decontam/blocklist.manifest.json
```

### 2.2 Tokenize and Pack Training Corpus

Tokenize Essential-Web and Stack-v2 corpus shards into packed binary uint16 arrays:

```bash
# Using native Swift tokenizer CLI
.build/release/monica-tokenize pack \
  --vocab configs/tokenizer/vocab.json \
  --merges configs/tokenizer/merges.txt \
  --input data/raw/corpus.jsonl \
  --output data/packed/train_shard_000.bin \
  --max-seq-len 8192 \
  --decontam-blocklist eval_sets/decontam/blocklist.txt
```

For local validation and smoke tests, generate a synthetic pre-packed fixture:

```bash
python scripts/build_synthetic_corpus.py \
  --out data/synthetic/train.bin \
  --tokens 500000 \
  --vocab-size 49152
```

---

## 3. Stage 2: Model Training Execution

### 3.1 Tier 1: Mac-Trained POC (~100M, 64k Window)

Runs locally on Apple Silicon (M1–M4) using MLX.

```bash
# Quick validation using Taskfile or Makefile
task smoke
# Or: make smoke

# Full local training run
python scripts/train.py \
  --config config/code-small-dense.yaml \
  --data data/split \
  --out runs/tier1-poc \
  --total-steps 50000 \
  --base-lr 3e-4 \
  --batch-size 4 \
  --grad-accum 8 \
  --ckpt-every 1000 \
  --eval-every 500
```

To resume from an existing checkpoint:

```bash
python scripts/train.py \
  --config config/code-small-dense.yaml \
  --data data/split \
  --resume runs/tier1-poc/resume \
  --out runs/tier1-poc
```

### 3.2 Tier 2: CUDA-Trained POC (~1B Active, 128k Window)

Runs on single-node 4× or 8× NVIDIA A100/H100 GPUs using PyTorch FSDP2.

```bash
torchrun --nproc_per_node=8 scripts/train.py \
  --config config/1b.yaml \
  --data "data/packed/train_shard_*.bin" \
  --out /mnt/checkpoints/tier2-poc-1b \
  --total-steps 100000 \
  --base-lr 2e-4 \
  --global-batch-size 256 \
  --seq-len 4096 \
  --curriculum-length-schedule "1024:10000,4096:50000,16384:100000" \
  --ckpt-every 2500 \
  --backend cuda
```

### 3.3 Tier 3: CUDA-Trained MVP (~4B Active, 128k–256k Window)

Runs on multi-node H100 clusters (e.g. 8 nodes × 8 H100s) with sparse upcycling from the Tier 1/2 dense baseline.

```bash
# Step 1: Upcycle dense checkpoint to sparse MoE
python scripts/sparse_upcycle.py \
  --source /mnt/checkpoints/tier2-poc-1b/final.safetensors \
  --target-config config/code-large-a.yaml \
  --out /mnt/checkpoints/tier3-mvp-init.safetensors \
  --num-experts 64 \
  --shared-experts 1

# Step 2: Distributed multi-node training with FSDP2 and loss-free balancing
torchrun \
  --nnodes=8 \
  --nproc_per_node=8 \
  --rdzv_id=monica-mvp \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:29500 \
  scripts/train.py \
    --config config/code-large-a.yaml \
    --init-weights /mnt/checkpoints/tier3-mvp-init.safetensors \
    --data "/mnt/data/packed/shard_*.bin" \
    --out /mnt/checkpoints/tier3-mvp-4b \
    --total-steps 250000 \
    --base-lr 1.5e-4 \
    --global-batch-size 1024 \
    --seq-len 8192 \
    --curriculum-length-schedule "2048:25000,8192:150000,32768:250000" \
    --loss-free-balancing \
    --ckpt-every 5000
```

---

## 4. Stage 3: Evaluation and Verification Suite

Monica incorporates comprehensive structural syntax, diagnostic guidance, and cross-path evaluation suites.

### 4.1 Automated Structural Syntax & AR Ablation

Evaluates syntax clean-rate and rollback metrics across 4 cells (Baseline, Fast Loop, Slow Loop, Both):

```bash
# Run AR ablation suite
python scripts/eval_ar_ablation.py \
  --checkpoint checkpoints/tier1-poc/final.safetensors \
  --eval-set eval_sets/ts_error_injection/eval.jsonl \
  --out results/ar_ablation.json

# Or via task runner:
task eval-ar
```

### 4.2 Diagnostic Supervision Verification

Evaluates rejection-sampled fine-tuning and contrastive negative margins:

```bash
# Run diagnostic supervision suite
python scripts/eval_diagnostic_supervision.py \
  --checkpoint checkpoints/tier1-poc/final.safetensors \
  --eval-set eval_sets/ts_error_injection/eval.jsonl \
  --out results/diagnostic_supervision.json

# Or via task runner:
task eval-diagnostic
```

### 4.3 Cross-Path Comparison (In-Generation vs. Tool-Call Feedback)

Benchmarks in-generation distribution-level feedback against the traditional multi-turn tool-call baseline:

```bash
# Run cross-path evaluation
python scripts/eval_cross_path.py \
  --checkpoint checkpoints/tier1-poc/final.safetensors \
  --eval-set eval_sets/ts_error_injection/eval.jsonl \
  --out results/cross_path_comparison.json

# Or via task runner:
task eval-cross-path
```

### 4.4 Held-Out Code Benchmark Suite

Runs recall, FIM completion, and external benchmarks (MultiPL-E, CrossCodeEval):

```bash
python scripts/eval_code_suite.py \
  --config config/code-small-dense.yaml \
  --weights runs/tier1-poc/weights.safetensors \
  --suites recall,needle,fim,external \
  --output results/code_suite.json \
  --transcript results/code_suite.jsonl
```

---

## 5. Stage 4: Quantization & Compression (Mixed Precision W4 + KV8)

To deploy with 75% memory bandwidth reduction while preserving exact code syntax generation, apply 4-bit affine hidden weight quantization with an 8-bit tied embedding head:

```bash
# Quantize checkpoint
python scripts/quantize_checkpoint.py \
  --weights runs/tier1-poc/weights.safetensors \
  --out runs/tier1-poc/weights.q4.safetensors \
  --bits 4 \
  --group-size 64 \
  --head-bits 8

# Inspect resulting configuration sidecar
cat runs/tier1-poc/weights.q4.safetensors.config.json
```

The output JSON sidecar records:
- `quant.mode`: "affine"
- `quant.group_size`: 64
- `quant.targets.embedding`: 8
- `quant.targets.layers.*`: 4
- `state_quantization.mode`: "fp32" (unquantized continuous SSM states)

---

## 6. Stage 5: Serving & Generation

### 6.1 Python / MLX Inference Serving

Run interactive generation or batch completions:

```bash
# Native FP16 serving (Mac POC)
python scripts/generate.py \
  --config config/code-small-dense.yaml \
  --weights runs/tier1-poc/weights.safetensors \
  --prompt "function mergeSort(arr: number[]): number[] {" \
  --max-new-tokens 256 \
  --temperature 0.2

# Mixed Precision W4 + KV8 serving
python scripts/generate.py \
  --config config/code-small-dense.yaml \
  --weights runs/tier1-poc/weights.q4.safetensors \
  --prompt "interface DistributedCache<K, V> {" \
  --max-new-tokens 512
```

### 6.2 High-Throughput Native Swift Engine

The native Swift inference engine leverages parallel-scan SSD kernels and decoupled KV caching:

```bash
# Benchmark prefill speedup (sequential vs. parallel-scan)
.build/release/monica-engine bench-prefill \
  --model checkpoints/tier1-poc/model_w4_kv8.safetensors \
  --prompt-len 2048 \
  --iterations 10

# Launch local HTTP OpenAI-compatible completions server
.build/release/monica-engine serve \
  --model checkpoints/tier1-poc/model_w4_kv8.safetensors \
  --port 8080 \
  --max-context 65536 \
  --kv-cache-dtype int8
```

### 6.3 Ultra-Long Context Serving (128k to 256k Window)

When serving Tier 2 (~1B) or Tier 3 (~4B) at 128k–256k context lengths:

```bash
# Launch long-context server with dynamic NTK-aware RoPE and scaled SSM delta
.build/release/monica-engine serve \
  --model /mnt/checkpoints/tier3-mvp-4b/model_w4_kv8.safetensors \
  --port 8080 \
  --max-context 262144 \
  --kv-cache-dtype int8 \
  --long-ctx-factor 4.0 \
  --rope-base-freq 1000000.0
```

Memory consumed at 256k tokens:
- **Weights**: 2.15 GB (W4 + Head 8)
- **KV Cache**: 5.37 GB (KV8 across 7 attention layers)
- **SSM State**: 6.23 MB (FP32 across 49 Mamba layers)
- **Total RAM**: **7.52 GB** (runs comfortably on unified 16 GB/24 GB Apple Silicon or single 24 GB GPU).

---

## 7. Diagnostics, Monitoring, and Troubleshooting

### 7.1 Loss Divergence or Spike During MoE Training
- **Symptom**: Step loss jumps abruptly from ~2.5 to >12.0 or NaN.
- **Cause**: Auxiliary-loss-free router imbalance or gradient overflow in SwiGLU experts.
- **Remedy**:
  1. Inspect routing entropy metrics logged in wandb / tensorboard.
  2. Verify that `--loss-free-balancing` is enabled.
  3. Lower learning rate warm-up multiplier by 0.7x or increase gradient clipping to 1.0.

### 7.2 Memory OOM at Ultra-Long Context Prefill (>64k)
- **Symptom**: `CUDA out of memory` or `MLX buffer allocation failed` during prompt prefill.
- **Cause**: Dense quadratic attention matrix materialization during prefill.
- **Remedy**:
  1. Pass `--prefill-chunk-size 4096` to enable chunked flash attention.
  2. Ensure recurrent SSM layers use parallel scan chunking (`scan_chunk_size: 256`).

### 7.3 Checkpoint State Resume Parity
- **Symptom**: Resumed model loss does not match uninterrupted run.
- **Cause**: Mamba-2 recurrent state buffers (`prev_state`) or optimizer momentum missing from checkpoint.
- **Verification**: Run `task smoke` to execute the exact resume parity check. The numerical difference must strictly be `0.0`.
