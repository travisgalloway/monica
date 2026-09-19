# Operational Execution Plan: CUDA VM End-to-End Implementation, Training, and Evaluation

[← Target Configuration Matrix](../design/16-target-configuration-matrix.md) | [← E2E Runbook](e2e-training-eval-serving.md) | [← Infrastructure Guide](../infrastructure.md)

This operational plan defines the step-by-step procedure to transition from the local Apple Silicon (Tier 1 Mac POC) run to an on-demand NVIDIA CUDA VM (e.g. RunPod / Lambda), complete all necessary code adaptations, and execute a full end-to-end tokenize, train, and eval run.

---

## 1. Hardware Topology & Target Configurations

### 1.1 Target Configurations

The run targets one of two model configurations depending on instance provisioning:

| Metric | Option A: Scaled Tier 1 POC (Fastest) | Option B: Tier 2 CUDA POC (Full Scale) |
| :--- | :--- | :--- |
| **Reference Config** | `config/code-small-dense.yaml` / `config/code-small-moe.yaml` | `config/1b.yaml` |
| **Parameters** | ~232M dense / ~685M MoE (8 experts, top-2, 1 shared) | ~1.03B active (dense Mamba-2) |
| **Architecture** | 56 layers, $d_{\text{model}}=768$, 7 attn layers (12.5%) | 36 layers, $d_{\text{model}}=2048$, pure Mamba-2 |
| **Vocabulary** | 49,152 (uint16 Swift BPE) | 49,152 (uint16 Swift BPE) |
| **Precision** | Native BF16 weights, FP32 SSM state | Native BF16 weights, FP32 SSM state |
| **Deployment Target** | Mixed W4 + KV8 (Head 8-bit) | Mixed W4 + KV8 (Head 8-bit) |
| **Hardware Fit** | 1× RTX 4090 (24 GB) or 1× A100 (40/80 GB) | 1× A100/H100 (80 GB) or 2–4× A100 (FSDP2) |

### 1.2 Hardware Templates & Recommended Cloud Hosts

Hardware configurations are formalized as templates in `scripts/cloud_pod.py`:

| Template | GPU Tier | VRAM | Container Disk | Network Volume | Cloud Type | Approx. Cost | Target Workload |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `rtx4090` | NVIDIA GeForce RTX 4090 | 24 GB | 50 GB | 0 GB | `ALL` | ~$0.44/hr | Fast local iteration and toy smoke tests |
| `a40` | NVIDIA A40 | 48 GB | 50 GB | 0 GB | `COMMUNITY` | ~$0.40/hr | Scaled Tier 1 POC (code-small-dense, 232M) |
| `a100-pcie` | NVIDIA A100 PCIe | 80 GB | 100 GB | 0 GB | `SECURE` | ~$1.89/hr | Tier 2 POC full scale (1B dense Mamba-2) |
| `a100-sxm4` | NVIDIA A100 SXM4 | 80 GB | 100 GB | 0 GB | `SECURE` | ~$2.49/hr | Distributed FSDP2 + Expert Parallelism (#271) |
| `h100-sxm` | NVIDIA H100 SXM5 | 80 GB | 100 GB | 0 GB | `SECURE` | ~$3.89/hr | FP8 expert GEMM verification on Hopper (#240) |

All templates use image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` with CUDA 12.4.1 and Triton support. Standalone instances default to zero volume disk allocation, because RunPod rejects non-zero container volume sizes when no network volume is attached.

### 1.3 Automated Lifecycle Management & Budget Caps (`scripts/cloud_pod.py`)

`scripts/cloud_pod.py` provides automated lifecycle tracking to prevent accidental cloud spend:

1. **Standard CLI Commands**:
   - `python scripts/cloud_pod.py templates`: List formalized templates and hardware specifications.
   - `python scripts/cloud_pod.py launch --template <name> [options]`: Provision a pod using template presets.
   - `python scripts/cloud_pod.py watch <pod_id> [options]`: Monitor instance limits and trigger automated shutdown on threshold breach.
   - `python scripts/cloud_pod.py heartbeat <pod_id>`: Record activity heartbeat from training processes.
   - `python scripts/cloud_pod.py status <pod_id>`: Display live runtime, estimated spend, and remaining limit headroom.
   - `python scripts/cloud_pod.py list`: Display all active pods with tracked expenditure.
   - `python scripts/cloud_pod.py stop <pod_id>`: Halt compute billing while retaining container disk.
   - `python scripts/cloud_pod.py terminate <pod_id>`: Permanently destroy instance and storage, ceasing all billing.

2. **Environment Variable Configuration**:
   The utility resolves configuration with the following precedence: explicit CLI flags override environment variables, which override template presets.
   - `RUNPOD_API_KEY`: API authentication key.
   - `RUNPOD_TEMPLATE`: Default template preset (`rtx4090`, `a40`, `a100-pcie`, `a100-sxm4`, `h100-sxm`).
   - `RUNPOD_MAX_BUDGET_USD`: Maximum dollar spend before enforcing automated shutdown.
   - `RUNPOD_MAX_RUNTIME_HOURS`: Maximum runtime in hours before enforcing automated shutdown.
   - `RUNPOD_IDLE_TIMEOUT_MINUTES`: Inactivity period before idle teardown triggers (default: 30 minutes).
   - `RUNPOD_AUTO_ACTION`: Action executed upon limit breach (`terminate` or `stop`, default: `terminate`).
   - `RUNPOD_TRACKER_FILE`: Custom file path for persistent lifecycle tracking state (default: `runs/cloud_pod_tracker.json`).

3. **Lifecycle Guard & Idle Teardown Daemon**:
   To enforce limits during long-running tasks, detach the lifecycle watcher in the background:
   ```bash
   nohup python scripts/cloud_pod.py watch <pod_id> \
       --max-budget 15.0 \
       --max-runtime-hours 6.0 \
       --idle-timeout 30.0 \
       --auto-action terminate \
       > runs/cloud_guard.log 2>&1 & disown
   ```

4. **Periodic Activity Heartbeat**:
   Training loops or background sync tasks emit heartbeats to indicate forward progress:
   ```bash
   python scripts/cloud_pod.py heartbeat <pod_id>
   ```

---

## 2. Phase 1: Pre-VM Implementation & Portability Fixes

Before launching the paid cloud instance, implement and land three portability fixes in the repository:

### 2.1 Item 1: Multi-Backend Inference Driver (`scripts/generate.py`)
- **Problem**: `scripts/generate.py` hardcodes `import mlx.core as mx` and `from src.model.mlx_backend import MLXMambaModel`. It will immediately crash with `SystemExit: mlx not found` on a Linux/CUDA machine.
- **Fix**: Wire `scripts/generate.py` through `src.model.backend.get_backend(args.backend)`, matching `scripts/train.py` and `scripts/smoke_test.py`:
  ```python
  backend = get_backend(args.backend)
  backend.seed(args.seed)
  model = backend.model_cls(cfg)
  to_numpy = backend.to_numpy
  ```
- **Verification**: Verify locally on Mac that `python scripts/generate.py --config config/toy.yaml --prompt "test" --byte-fallback` continues to pass with `--backend mlx` and `--backend auto`.

### 2.2 Item 2: Parameterize Build & Automation Targets (`Makefile` & `Taskfile.yml`)
- **Problem**: Current targets in `Makefile` and `Taskfile.yml` hardcode `--backend mlx`.
- **Fix**: Introduce `BACKEND ?= auto` variable. Provide explicit targets:
  - `make smoke-cuda`: `$(PYTHON) scripts/smoke_test.py --backend cuda --config $(CONFIG) --data $(DATA) --steps 10 --batch-size 2 --out runs/smoke_test --compile`
  - `make train-cuda`: `$(PYTHON) scripts/train.py --backend cuda --config $(POC_CONFIG) --data $(DATA) --out $(POC_OUT) ...`
  - `make eval-code-cuda`: `$(PYTHON) scripts/eval_code_suite.py --backend cuda ...`
  - `make e2e-cuda`: Full sequence on CUDA.

### 2.3 Item 3: Hardware Checklist Scope Boundaries
- **Issue #240 (FP8 linear for experts)**: Transformer Engine FP8 is Hopper-only (H100). If running on an RTX 4090 or A100, `fp8_experts` remains disabled in config; on H100, probe via `tests/test_cuda_fp8.py`.
- **Issue #214 (8-bit AdamW)**: Validated via `pip install -e '.[cuda-8bit]'` and `python scripts/smoke_test.py --backend cuda --config config/toy-moe-8bit.yaml`.
- **Issue #271 (FSDP2 + Expert Parallel)**: Run multi-GPU NCCL validation checklist on multi-GPU pod.

---

## 3. Phase 2: CUDA VM Environment Provisioning & Bootstrap

Execute upon provisioning the cloud VM:

### 3.1 Step 2.1: System Bootstrap & Linux Packages
```bash
# Update package indices
apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl wget unzip htop nvtop ca-certificates \
    libicu-dev libxml2-dev libcurl4-openssl-dev nodejs npm

# Verify NVIDIA Driver and CUDA runtime
nvidia-smi
nvcc --version
```

### 3.2 Step 2.2: Install Swift Toolchain on Linux
Monica's native tokenizer (`monica-tokenize`), LSP server (`monica-lsp`), and engine (`monica-engine`) require the Swift 5.10 or 6.0 Linux toolchain:
```bash
# Download official Swift Linux toolchain (Ubuntu 22.04 x86_64)
SWIFT_VERSION=swift-5.10.1-RELEASE
SWIFT_URL=https://download.swift.org/swift-5.10.1-release/ubuntu2204/swift-5.10.1-RELEASE/${SWIFT_VERSION}-ubuntu22.04.tar.gz

mkdir -p /opt/swift
wget -qO- ${SWIFT_URL} | tar xz -C /opt/swift --strip-components=1
export PATH="/opt/swift/usr/bin:${PATH}"
echo 'export PATH="/opt/swift/usr/bin:${PATH}"' >> ~/.bashrc

# Verify Swift compiler
swift --version
```

### 3.3 Step 2.3: Clone Repository & Virtual Environment
```bash
# Clone repository and checkout feature branch
git clone https://github.com/travisgalloway/monica.git /workspace/monica
cd /workspace/monica
git checkout feat/mac-poc-run

# Create virtualenv with Python 3.11
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

# Install with CUDA and fast-path extras
pip install -e ".[dev,data,cuda,cuda-fast,cuda-8bit,eval]"

# Build native Swift tokenizer and engine tools
(cd swift && swift build -c release --build-system native)
```

### 3.4 Step 2.4: Verify Fast-Path Triton & Fused CUDA Kernels
```bash
# Verify PyTorch CUDA, GPU visibility, and mamba-ssm / causal-conv1d fast paths
python -c "
import torch
from src.model.cuda_backend import fast_path_status
scan_ok, conv_ok = fast_path_status()
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0)}')
print(f'Fused SSD scan (mamba-ssm): {scan_ok}')
print(f'Causal Conv1d: {conv_ok}')
assert torch.cuda.is_available(), 'CUDA not detected'
"
```

### 3.5 Step 2.5: Configure Cloudflare R2 Credentials
```bash
# Set credentials for durable checkpoint syncing
cat << 'EOF' > .env
AWS_ENDPOINT_URL_S3="https://<account_id>.r2.cloudflarestorage.com"
AWS_ACCESS_KEY_ID="<r2_access_key>"
AWS_SECRET_ACCESS_KEY="<r2_secret_key>"
R2_BUCKET="monica-training"
EOF

set -a; source .env; set +a
```

---

## 4. Phase 3: Hardware Verification Gates (Dress Rehearsal)

Run the verification battery before executing expensive multi-hour training:

### 4.1 Gate 3.1: Unit & Distributed Test Parity
```bash
# Run CUDA parity, compile, and distributed test suite
pytest tests/test_cuda_parity.py \
       tests/test_cuda_compile.py \
       tests/test_cuda_train_step.py \
       tests/test_cuda_distributed.py -v
```

### 4.2 Gate 3.2: CUDA Smoke Test Gate (Exact Resume + Torch Compile)
```bash
# Dense smoke gate with torch.compile verification
python scripts/smoke_test.py \
    --backend cuda \
    --config config/toy.yaml \
    --data data/split \
    --steps 20 \
    --compile \
    --out runs/smoke-cuda

# MoE smoke gate with gather router
python scripts/smoke_test.py \
    --backend cuda \
    --config config/toy-moe.yaml \
    --moe-impl gather \
    --data data/split \
    --steps 20 \
    --out runs/smoke-moe-cuda
```

### 4.3 Gate 3.3: Native Swift Selfcheck
```bash
swift/.build/release/monica-selfcheck
```

### 4.4 Gate 3.4: Train-Step Benchmark & Checkpoint Cadence Sizing
Measure real milliseconds-per-step and memory on the provisioned GPU to set `--ckpt-every`:
```bash
# Benchmark POC training step on GPU
python scripts/bench_cuda_train_step.py \
    --config config/code-small-dense.yaml \
    --device cuda \
    --batch 4 \
    --grad-accum 8 \
    --warmup 3 \
    --iters 10
```
*Cadence Formula*: $\text{ckpt\_every} \approx \frac{25 \text{ min} \times 60 \text{ s}}{\text{s\_per\_step}}$. If step time is $1.5\text{ s}$, set `--ckpt-every 1000`.

---

## 5. Phase 4: Data Ingestion, Tokenization & Packing at Scale

### 5.1 Step 4.1: Build Decontamination Blocklist
```bash
# Generate 13-gram decontamination blocklist from held-out evaluation sets
python scripts/build_decontam_blocklist.py \
    --out eval_sets/decontam/blocklist.txt \
    --manifest eval_sets/decontam/blocklist.manifest.json
```

### 5.2 Step 4.2: Swift Tokenization & Packing
```bash
# Ingest and pack training corpus using native Swift BPE tokenizer
swift/.build/release/monica-tokenize pack \
    --vocab configs/tokenizer/vocab.json \
    --merges configs/tokenizer/merges.txt \
    --input data/raw/corpus.jsonl \
    --output data/packed/train_shard_000.bin \
    --max-seq-len 2048 \
    --decontam-blocklist eval_sets/decontam/blocklist.txt
```
For fast end-to-end dress rehearsal, construct a verified synthetic partition:
```bash
python scripts/build_synthetic_corpus.py \
    --out data/split/train.bin \
    --tokens 1000000 \
    --vocab-size 49152

python scripts/build_synthetic_corpus.py \
    --out data/split/val.bin \
    --tokens 50000 \
    --vocab-size 49152
```

---

## 6. Phase 5: CUDA Training Execution

### 6.1 Single-GPU Training Run (Tier 1 POC Scaling / Tier 2 POC)
Launch detached training process:
```bash
# Detach training process with nohup per AGENTS.md Standing Guidelines
nohup python scripts/train.py \
    --backend cuda \
    --config config/code-small-dense.yaml \
    --data data/split \
    --out runs/cuda-poc \
    --total-steps 5000 \
    --batch-size 4 \
    --grad-accum 8 \
    --base-lr 3e-4 \
    --curriculum "0.25:1024,0.5:2048,1.0:4096" \
    --log-every 10 \
    --eval-every 250 \
    --ckpt-every 500 \
    > runs/cuda-poc/train.log 2>&1 & disown

TRAIN_PID=$!
echo "Training detached with PID ${TRAIN_PID}, logging to runs/cuda-poc/train.log"
```

### 6.2 Multi-GPU Training Run (FSDP2 + In-Node Expert Parallel #271)
If running on a multi-GPU instance ($N$ GPUs):
```bash
nohup torchrun --standalone --nproc_per_node=4 scripts/train_dist.py \
    --config config/code-small-moe.yaml \
    --ep-size 2 \
    --data data/split \
    --out runs/cuda-dist-poc \
    --total-steps 5000 \
    --batch-size 4 \
    --grad-accum 4 \
    --base-lr 3e-4 \
    --log-every 10 \
    --ckpt-every 500 \
    > runs/cuda-dist-poc/train.log 2>&1 & disown

DIST_PID=$!
echo "Distributed training detached with PID ${DIST_PID}"
```

### 6.3 Background Durable Checkpoint Sync Loop
While training runs, synchronize committed checkpoints to Cloudflare R2:
```bash
# Recurring sync loop every 10 minutes
nohup bash -c '
while true; do
    if [ -d "runs/cuda-poc/resume" ]; then
        python -m src.data.r2_sync up runs/cuda-poc/resume s3://monica-training/ckpt/cuda-poc/resume
    fi
    sleep 600
done
' > runs/cuda-poc/r2_sync.log 2>&1 & disown
```

---

## 7. Phase 6: Verification & Complete Evaluation Suite

Once training completes and `runs/cuda-poc/weights.safetensors` is emitted:

### 7.1 Evaluation 1: Held-Out Code Benchmark Suite
```bash
python scripts/eval_code_suite.py \
    --config config/code-small-dense.yaml \
    --checkpoint runs/cuda-poc/weights.safetensors \
    --backend cuda \
    --byte-tokenizer \
    --suites recall,needle,fim,external \
    --output results/cuda_poc_code_suite.json \
    --transcript results/cuda_poc_code_suite.jsonl
```

### 7.2 Evaluation 2: Diagnostic Supervision Verification
```bash
python scripts/eval_diagnostic_supervision.py \
    --checkpoint runs/cuda-poc/weights.safetensors \
    --eval-set eval_sets/ts_error_injection/eval.jsonl \
    --output results/cuda_poc_diagnostic.json
```

### 7.3 Evaluation 3: Native Swift LSP Benchmark
```bash
swift/.build/release/monica-lsp \
    --bench \
    --eval-set-dir eval_sets/ts_error_injection
```

---

## 8. Phase 7: Quantization, Serving, & Final Export

### 8.1 Mixed Precision W4 + KV8 Quantization
Compress the trained CUDA checkpoint:
```bash
python scripts/quantize_checkpoint.py \
    --weights runs/cuda-poc/weights.safetensors \
    --out runs/cuda-poc/weights.q4.safetensors \
    --bits 4 \
    --head-bits 8
```

### 8.2 Generation Verification on CUDA
Verify identical completions between native and quantized models:
```bash
# Native BF16 generation on CUDA
python scripts/generate.py \
    --backend cuda \
    --config config/code-small-dense.yaml \
    --weights runs/cuda-poc/weights.safetensors \
    --prompt "function mergeSort(arr: number[]): number[] {" \
    --max-new-tokens 64

# Mixed Precision W4+KV8 generation on CUDA
python scripts/generate.py \
    --backend cuda \
    --config config/code-small-dense.yaml \
    --weights runs/cuda-poc/weights.q4.safetensors \
    --prompt "function mergeSort(arr: number[]): number[] {" \
    --max-new-tokens 64
```

### 8.3 High-Throughput Swift Prefill Benchmark
```bash
swift/.build/release/monica-engine bench-prefill \
    --model runs/cuda-poc/weights.q4.safetensors \
    --prompt-len 2048 \
    --iterations 10
```

### 8.4 Final Sync to Cloudflare R2 & VM Teardown
```bash
# Push full run output, weights, quantized weights, and eval metrics to R2
python -m src.data.r2_sync up runs/cuda-poc s3://monica-training/runs/cuda-poc
python -m src.data.r2_sync up results s3://monica-training/results

# Verify tracker status and terminate instance to cease all billing
python scripts/cloud_pod.py status <pod_id>
python scripts/cloud_pod.py terminate <pod_id>
```

---

## 9. Contingency Matrix & Operational Troubleshooting

| Symptom | Probable Cause | Action |
| :--- | :--- | :--- |
| **`ImportError: mamba_ssm` or `causal_conv1d`** | Header mismatch during wheel build | Re-run `pip install -e '.[cuda-fast]' --no-build-isolation` against CUDA 12.4 headers. |
| **`torch.cuda.OutOfMemoryError`** | Context length prefill or batch size too large | Reduce `--batch-size` or set `grad_checkpoint: true` in config YAML. |
| **NCCL timeout during `train_dist.py`** | Multi-GPU interconnect or firewall block | Set `export NCCL_DEBUG=INFO` and `export NCCL_IB_DISABLE=1` if InfiniBand is unavailable. |
| **Loss spikes or NaN** | Learning rate warm-up too aggressive or router instability | Enable `--loss-free-balancing` and verify gradient clipping `--grad-clip 1.0`. |
| **Resume trajectory divergence** | Loader state or seed mismatch across restart | Ensure `--seed`, batch size, and dataset token count match the pre-interruption run exactly. |

---

## 10. Execution Summary & Validation Evidence (Completed 2026-09-18)

### 10.1 Hardware & Runtime Environment
- **Host**: RunPod Community Cloud, Pod ID `8rqjtpijck62eb` (NVIDIA A40, 48 GB VRAM).
- **Environment**: Ubuntu 22.04, Python 3.11, PyTorch 2.4.1+cu124, Triton 3.0.0, CUDA Driver 570.211.01.
- **Model**: Scaled Tier 1 POC (`config/code-small-dense.yaml`), 56 layers, $d_{\text{model}}=768$, $d_{\text{inner}}=1536$, 7 attention layers (12.5%), vocab 49,152 (tied embeddings), 232.1M parameters.

### 10.2 Verification Gates Passed
- **Pytest Suite**: `tests/test_cuda_parity.py` and `tests/test_cuda_train_step.py` — **24 passed, 4 skipped, 0 failed**.
- **Smoke Test Gate**: `scripts/smoke_test.py --backend cuda --config config/toy.yaml --data data/split --steps 20 --compile --out runs/smoke-cuda` — **PASSED** (bit-exact 0.000e+00 resume, Inductor dynamic shape compile passed, prefill/decode parity verified).
- **Train Step Benchmark**:
  - Precision: FP32 with TF32 enabled (`torch.set_float32_matmul_precision('high')`).
  - Throughput: 6.64 s/step (9,870 tokens/s at 65,536 tokens/step) on 1× A40 GPU.
  - Memory: Peak VRAM 9.34 GB (plenty of headroom under 48 GB).

### 10.3 Full POC Training Run
- **Parameters**: 100 optimizer steps, batch size 4, gradient accumulation 8 (65,536 tokens/step, 6,553,600 total tokens).
- **Curriculum**: Stage 0 (seq_len=1024, steps 0–49) $\to$ Stage 1 (seq_len=2048, steps 50–99).
- **Loss Trajectory**:
  - Step 0: Loss 687.9, grad norm 26.56, val loss 627.8
  - Step 25: Loss 73.89, grad norm 22.06, val loss 72.82
  - Step 50: Loss 60.41, grad norm 5.88, val loss 59.95 (curriculum stage boundary)
  - Step 75: Loss 55.30, grad norm 3.31, val loss 54.88
  - Step 100: Loss 53.14, grad norm 3.00, val loss 53.14
- **Checkpoints**: Slot-A (Step 50) and Slot-B (Step 100) exact state bundles committed.

### 10.4 Held-Out Code Evaluation Suite (`scripts/eval_code_suite.py`)
- Output: `results/cuda_poc_code_suite.json` (95 records)
- Summary:
  - Symbol Recall: 17 instances, cross-entropy 61.16, top-1 accuracy 23.53%, MRR 0.4647.
  - Needle in a Haystack: 20 instances, cross-entropy 56.05 across 512 and 1024 context lengths.
  - Fill-in-the-Middle (FIM): 28 instances, cross-entropy 50.01.
  - External Multi-file: 30 instances, cross-entropy 47.12.

### 10.5 Mixed Precision W4+KV8 Quantization & Generation
- **Compression**: `scripts/quantize_checkpoint.py --bits 4 --head-bits 8` reduced model footprint from 928.3 MB down to 165.1 MB (**5.62× compression**).
- **Generation**: Verified streaming generation via `scripts/generate.py --backend cuda` on both native FP32 weights (`runs/cuda-poc/weights.safetensors`) and quantized W4+KV8 weights (`runs/cuda-poc/weights.q4.safetensors`).

### 10.6 Durable Storage & Infrastructure Teardown
- **R2 Sync**: Synchronized all 15 checkpoint/training artifacts to `s3://monica-training/runs/cuda-poc/` and 33 evaluation artifacts to `s3://monica-training/results/cuda-poc/`.
- **Pod Termination**: Pod `8rqjtpijck62eb` terminated cleanly via `cloud_pod.py terminate`. Zero active cloud spend.
