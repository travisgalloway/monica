# Target Configuration Matrix: POC to MVP (#198 / #272)

[← Index](README.md)

This specification defines the authoritative target matrix for the Monica code model project across development, scale verification, and production tiers, standardizing on two deployment precision regimes: **Native FP16 / BF16** and **Mixed Precision W4 + KV8**.

---

## 1. Executive Summary

Monica's Mamba-2 Hybrid MoE (MHM) architecture decouples token-generation compute from context length by combining:
1. **Mamba-2 / SSD State-Space Layers (87.5%)**: Constant $O(1)$ memory footprint across arbitrary sequence lengths with linear $O(L)$ prefill.
2. **Causal Multi-Head Attention Layers (12.5%)**: Scaled KV cache capturing cross-file symbol resolution and in-context retrieval.
3. **Sparse Mixture-of-Experts (MoE)**: SwiGLU experts with Loss-Free Balancing and additive shared experts (DeepSeek-V2/V3 style).

The project targets three milestone tiers:
- **Tier 1: Mac-Trained POC** (~100M params, ~64k context window) — Local Apple Silicon development and architecture validation.
- **Tier 2: CUDA-Trained POC** (~1B active params, 128k context window) — Single-node / cloud scale proof-of-concept.
- **Tier 3: CUDA-Trained MVP** (~4B active params, 128k – 256k context window) — Production-grade multi-file code model.

---

## 2. Target Configuration Matrix

| Specification | Tier 1: Mac-Trained POC | Tier 2: CUDA-Trained POC | Tier 3: CUDA-Trained MVP |
| :--- | :--- | :--- | :--- |
| **Role** | Local Apple Silicon POC & gate | Single-node scale proof-of-concept | Flagship production code model |
| **Training Hardware** | Apple Silicon (MLX / Metal) | 1–4× NVIDIA A100/H100 (CUDA) | Multi-node H100 cluster (FSDP2) |
| **Model Class** | Mamba-2 Hybrid Dense / Small MoE | Mamba-2 Hybrid MoE | Mamba-2 Hybrid MoE (MHM) |
| **Total Parameters** | ~100M – 232M | ~1.03B – 1.8B | ~3.86B – 16B |
| **Active Parameters** | **~100M** (100% active) | **~1.0B active** | **~4.0B active** |
| **Target Context Window** | **~64k tokens** (65,536) | **128k tokens** (131,072) | **128k – 256k tokens** (262,144) |
| **Base Training Length** | 2,048 (Curriculum: 512 → 8k) | 4,096 (Curriculum: 1k → 16k) | 8,192 (Curriculum: 2k → 32k) |
| **Backbone Dimensions** | $d_{\text{model}}=768$, 24–56 layers | $d_{\text{model}}=1536–2048$, 36–48 layers | $d_{\text{model}}=2048–3072$, 56–64 layers |
| **Attention Layers** | 12.5% (`attn_every: 8`) | 12.5% (`attn_every: 8`) | 12.5% (`attn_every: 8`) |
| **MoE Routing** | Dense or 8/top-2 (1 shared) | 16–32 experts / top-4 (1 shared) | 64 experts / top-8 (1 shared) |
| **Vocabulary** | 49,152 (uint16 Swift BPE) | 49,152 (uint16 Swift BPE) | 49,152 (uint16 Swift BPE) |
| **Primary Reference Config**| `config/code-small-dense.yaml` | `config/1b.yaml` | `config/code-large-a.yaml` |

---

## 3. Dual Deployment Precision Regimes

Every tier targets two standardized runtime precision configurations:

### Regime A: Native Precision (FP16 / BF16)
- **Weights**: Native 16-bit float (`fp16` on Apple Silicon Metal; `bf16` on CUDA).
- **SSM Recurrent State**: 32-bit float (`fp32`) for numerical integration stability.
- **Attention KV Cache**: Native 16-bit float (`fp16` / `bf16`).
- **Objective**: Reference ground-truth quality, zero quantization noise, maximum fidelity for benchmark verification.

### Regime B: Mixed Precision (W4 + KV8)
- **Model Hidden Weights**: 4-bit group-wise affine quantization (`group_size: 64`).
- **Tied Embedding / Output Head**: **Kept at 8-bit** (`--head-bits 8`) to eliminate syntax token logit distortion.
- **Attention KV Cache**: **8-bit quantized** (`KV8`), preserving sharp attention softmax distributions and needle-in-a-haystack recall across 64k–256k context tokens.
- **SSM Recurrent State**: Uncompressed 32-bit float (`fp32`), maintaining exact continuous-time state transitions at zero memory cost (under 10 MB total).
- **Objective**: 75% memory bandwidth reduction, 2x–3x decode speedup, and deployment on consumer unified memory.

---

## 4. Quantitative Memory Footprint & Sizing

Because 87.5% of the model layers are Mamba-2 SSMs with fixed-size state, memory growth at ultra-long context is bounded strictly to the 12.5% attention layers.

### Tier 1: Mac-Trained POC (~100M params, 64k window)
- **SSM Recurrent State**: **2.67 MB** (Constant $O(1)$)
- **Attention Layers**: 2 to 7 layers, $d_{\text{model}}=768$

| Context Length | Weights (Native FP16) | Weights (W4 + Head 8) | KV Cache (FP16) | KV Cache (KV8) | Total (Native FP16) | Total (Mixed W4+KV8) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2k tokens** | 0.20 GB | 0.06 GB | 0.04 GB | 0.02 GB | **0.24 GB** | **0.08 GB** |
| **16k tokens** | 0.20 GB | 0.06 GB | 0.33 GB | 0.17 GB | **0.53 GB** | **0.23 GB** |
| **64k tokens** | 0.20 GB | 0.06 GB | 1.34 GB | 0.67 GB | **1.54 GB** | **0.73 GB** |

*Target Hardware*: Fits comfortably on any baseline Mac (8 GB / 16 GB Apple Silicon M1–M4).

---

### Tier 2: CUDA-Trained POC (~1B active params, 128k window)
- **SSM Recurrent State**: **4.50 MB** (Constant $O(1)$)
- **Attention Layers**: 4 to 6 layers, $d_{\text{model}}=2048$

| Context Length | Weights (Native BF16) | Weights (W4 + Head 8) | KV Cache (BF16) | KV Cache (KV8) | Total (Native BF16) | Total (Mixed W4+KV8) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **4k tokens** | 2.06 GB | 0.58 GB | 0.16 GB | 0.08 GB | **2.22 GB** | **0.66 GB** |
| **32k tokens** | 2.06 GB | 0.58 GB | 1.34 GB | 0.67 GB | **3.40 GB** | **1.25 GB** |
| **128k tokens** | 2.06 GB | 0.58 GB | 5.37 GB | 2.68 GB | **7.43 GB** | **3.26 GB** |

*Target Hardware*: Fits on single 16 GB / 24 GB GPU (NVIDIA L4 / A10G / RTX 4090) or 16 GB / 24 GB MacBook Pro.

---

### Tier 3: CUDA-Trained MVP (~4B active params, 128k – 256k window)
- **SSM Recurrent State**: **6.23 MB** (Constant $O(1)$)
- **Attention Layers**: 7 to 8 layers, $d_{\text{model}}=2048–3072$
- **Total Parameters**: ~3.86B (MoE 64/8/1) to ~16B total capacity

| Context Length | Weights (Native BF16) | Weights (W4 + Head 8) | KV Cache (BF16) | KV Cache (KV8) | Total (Native BF16) | Total (Mixed W4+KV8) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **32k tokens** | 7.71 GB | 2.15 GB | 1.34 GB | 0.67 GB | **9.05 GB** | **2.82 GB** |
| **128k tokens** | 7.71 GB | 2.15 GB | 5.37 GB | 2.68 GB | **13.08 GB** | **4.83 GB** |
| **256k tokens (Max)** | 7.71 GB | 2.15 GB | 10.74 GB | 5.37 GB | **18.45 GB** | **7.52 GB** |

*Target Hardware*:
- **128k window**: Fits natively in FP16 on 24 GB / 36 GB Mac or single 24 GB GPU. In W4+KV8, consumes under **5 GB RAM**.
- **256k window (Max Ceiling)**: Native FP16 consumes **18.45 GB**, running on a 24 GB / 32 GB Mac or single 24 GB GPU (RTX 4090 / A10G / L4). In Mixed W4+KV8, consumes **only 7.52 GB RAM**, running comfortably on standard 16 GB / 24 GB MacBook Pro hardware.

---

## 5. Long-Context Extension Architecture

To scale from nominal training lengths to 64k, 128k, and 256k tokens without quality collapse, the tiers employ:

1. **Continuous SSM Receptive Field Scaling (`long_ctx_factor`)**:
   $$\Delta t_{\text{eval}} = \frac{\Delta t}{\text{long\_ctx\_factor}}$$
   Scales the discretization interval to preserve long-horizon state memory in the 87.5% Mamba layers.
2. **Dynamic NTK-Aware RoPE Interpolation**:
   Applies base frequency scaling ($base = 10000 \cdot \alpha$) to the 12.5% attention layers to extrapolate positional representations beyond the base training length.
3. **Repository-Context Packing**:
   Multi-file TypeScript AST and import-graph packing in the Swift data packer (`monica-tokenize pack`), bundling entire project modules into contiguous 32k–128k shards.
4. **FlashAttention / Metal SDPA Chunking**:
   Intra-chunk attention tiling prevents materialization of $L \times L$ attention matrices in unified memory during prefill.
