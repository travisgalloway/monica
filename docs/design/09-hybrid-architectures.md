# Hybrid SSM + Transformer architectures (model choice)

[← Index](README.md)

Why the scale-up model is a **Mamba-2 *hybrid*** — mostly SSD blocks with a small fraction
of attention layers — and how it is sized. This hybrid rationale is **foundation** that the live
M12 code model builds on (its MoE spine + sizing are in
[13-code-model-moe.md](13-code-model-moe.md), tracker
[issue #198](https://github.com/travisgalloway/monica/issues/198)); the SSD block itself is in
[model: the Mamba block](02-model-ssm.md). The single-~1B-model sizing discussed below was the
**reserve M10** target (arch children #66–#68, epic
[#65](https://github.com/travisgalloway/monica/issues/65), dropped 2026-07-19) — read it for the
attention-fraction reasoning, which carries over to the M12 MoE model.

## Why a hybrid, not pure SSD

Pure SSMs compress history into a fixed-size state, so they lag Transformers on **exact
copying** and **in-context retrieval** — few-shot recall, copying long identifiers, resolving
long-range references. That is exactly the **code** failure mode this POC cares about
(TypeScript / Rust / SQL emphasis). The fix is a few **attention** layers interleaved among
the Mamba-2 blocks: attention restores precise token-level lookup where the SSM is weak,
while the SSM keeps the sequence-length cost linear and the inference footprint KV-cache-free
for the bulk of the depth.

- **Jamba** pattern: ~**1 attention layer per 7 Mamba-2 layers** (every 8th). Good starting
  ratio.
- **Nemotron-H** pushes further (~92% Mamba). Tune the fraction empirically: raise attention
  if the retrieval probes (#79) lag, lower it for speed.
- Reported result that motivates this: an 8B Mamba-2-hybrid beat an 8B Transformer on all
  twelve standard tasks tested while generating up to ~8× faster, where the pure SSM trailed.

The attention fraction is **config-gated and behind the [seam](01-architecture-seam.md)**
(#67): `MambaConfig` gains `attn_every: int|None` (and `n_attn_heads`); `attn_every=None`
is pure Mamba (today's default). The attention block (causal MHA + RoPE) is implemented in
both backends; the recurrent-state container becomes mixed — a per-attention-layer **KV
cache** rides alongside the per-layer SSM `(conv_state, ssm_state)` in `step`/`init_state`/
`get_state`. forward/step parity and backend parity (#67) must stay green in fp32 ~1e-4.

## The packing hazard (why long-seq is its own concern)

Mamba's linear memory makes **8K+** sequences affordable, which keeps long code files intact.
But packing concatenates documents into fixed-length sequences and the SSD recurrent state
**carries across positions**, so without care one document's state bleeds into the next.
Either **reset the SSM state at document boundaries** or use **packing-aware kernels**
(PackMamba reports up to ~3× throughput). This is tracked separately (#68) and verified
conformance-style: two documents packed into one sequence must produce identical per-document
logits to running them apart (fp32 ~1e-4).

## Sizing the model

Mamba has **no KV cache**, so fp16/bf16 weights ≈ total inference footprint. The target is a
**single ~1B model**; the 100M `poc` is the cheap architecture-validation rung below it. (The
earlier 2B/4B scale tiers — and a 16B candidate before them — were dropped for this POC.)

| tier | d_model | n_layers | head_dim | ≈ params | bf16 weights | train GPU |
|---|---|---|---|---|---|---|
| **100M** (poc — OLMo, reserve) | 768 | 24 | 64 | ~127M | ~0.25 GB | T4 16 GB / L4 24 GB |
| **205M** (poc-qwen — Qwen2.5, completed POC run, reserve) | 768 | 24 | 64 | ~205M | ~0.4 GB | T4 16 GB / L4 24 GB |
| **1B** (target — from scratch) | 2048 | 36 | 64 | ~1.03B | ~2 GB | L4 / A10 24 GB |
| **1B** (reserve distillation student, hybrid) | 2048 | 28 | 64 | ~1.03B | ~2 GB | L4 / A10 24 GB |

Both ~1B configs land within ±5% of the 1B target (verify with the sizing tool). The **reserve
distillation student** ([`config/student-1b.yaml`](../../config/student-1b.yaml)) uses **28
layers** — matching the teacher's 28 transformer layers so the Mamba-in-the-Llama init maps
layer-to-layer (see [distillation, reserve](../reserve/10-distillation.md)) — while the
**from-scratch** 1B ([`config/1b.yaml`](../../config/1b.yaml), OLMo vocab) uses 36 layers for the
same param budget (its smaller vocab leaves more room in the layer stack). The 100M tier is the
validated [`poc.yaml`](07-configs-and-decisions.md) (OLMo). The now-**completed POC run** (reserve)
used [`poc-qwen.yaml`](../../config/poc-qwen.yaml) — the same layers retargeted to the Qwen2.5
vocab (151,646) so the POC exercised a tokenizer/data path token-aligned with the (reserve)
distillation student's (the student is on the Qwen3 vocab 151,669, which is token-aligned with
Qwen2.5); the larger tied embedding (~116M) then dominates, making it ~205M, embedding-heavy. (At
the student's d_model 2048 that same embedding is a negligible ~11%, so the vocab is "free" there
— it only dominates a narrow 768-wide model.) The sizing tool is a portable closed-form
param/memory calculator (`src/model/sizing.py` + `scripts/model_size.py`, #66), cross-checked
against the built model's portable state-dict param sum. Training memory ≈ 8 B/param
(weights+grad+AdamW), or **~10 B/param with 8-bit Adam** — the VRAM-tight lever used in
Phase 5 (#75).

### Estimating training time

Sizing answers "does it fit?"; the companion estimator answers "how long?".
`src/model/train_time.py` + `scripts/train_time.py` turn a param count and a token
budget into estimated wall-clock via the standard `6·N·D` training-FLOPs model
divided by each machine's achieved throughput. The **M1 Pro** figure is *calibrated*
from the one measured point we have — ~99 s/step at 131,072 tokens/step for `poc`
(~1.0 TFLOP/s effective), which reproduces the "3B tokens ≈ 26 days" rule of thumb.
The **H100** figures are bf16 dense peak (~990 TFLOPS) × an assumed 40% MFU (single)
and ×85% scaling (8-GPU); there is no in-repo H100 bench yet, so treat them as
planning estimates and retune with `--mfu` / `--scaling`.

Example (`python scripts/train_time.py`, Chinchilla-optimal 20 tokens/param budget):

| model | params | tokens (20×) | M1 Pro | 1× H100 | 8× H100 |
|---|---|---|---|---|---|
| poc | ~127M | 2.53B | 22.2 d | 1.4 h | 11.9 m |
| — | 270M | 5.40B | 100.6 d | 6.1 h | 54.1 m |
| 1b | ~1.03B | 20.67B | 4.0 y | 3.7 d | 13.2 h |
| — | 3B | 60.00B | 34.0 y | 31.6 d | 4.6 d |
| — | 7B | 140.00B | 185.2 y | 171.9 d | 25.3 d |

The spread is the headline: a from-scratch 1B run is years on the laptop but days on
a single H100, and the laptop is only viable at the `poc` rung. Pass `--tokens 3B`
for a fixed budget across sizes, or `--config <yaml>` to estimate an exact config.

**Chinchilla is an upper bound, not a POC run.** `20×params` is the *compute-optimal*
budget; exploratory POC runs use **far fewer** tokens (just enough to watch the loss
curve drop), so the table above over-states what we actually do on the laptop. The
honest laptop question is the inverse — "how many tokens fit in a day?" — which
`--hours N` answers and which matches measured experience (a ~200M model trains
**~72M tokens in 24h on M1 Pro**, a normal short POC, *not* the 4B Chinchilla budget):

| model | params | M1 Pro / 24h | 1× H100 / 24h | 8× H100 / 24h |
|---|---|---|---|---|
| poc | ~127M | 114M | 45.0B | 306B |
| — | 200M | 72M | 28.5B | 194B |
| 1b | ~1.03B | 14M | 5.5B | 37.5B |

These are first-principles estimates (generic `6·N·D`, not the SSM/attention split);
the real per-step cost is what `scripts/bench_train_step.py` (MLX) and
`scripts/bench_cuda_train_step.py` (CUDA) measure. The short (reserve) distillation runs in
particular needed ≪ Chinchilla tokens — see [distillation, reserve](../reserve/10-distillation.md).

### Precision differs from the POC

`poc.yaml` uses **fp16 + loss scaling** because fp16 is ~18% faster than bf16 *on MLX/Metal*
(see [configs & decisions](07-configs-and-decisions.md)). The scale configs train on **CUDA**,
where **bf16 is native** and needs no loss scaling — so `config/1b.yaml` and
`config/student-1b.yaml` set `precision: bf16` (`scaler_for_precision` already returns `None`
for bf16). `tie_embeddings`
and `grad_checkpoint` stay on; vocab follows the chosen tokenizer and sets the packed dtype —
uint16 below 65536 (POC), uint32 for Qwen3 (the reserve distillation student, #90; see
[distillation, reserve](../reserve/10-distillation.md)).

## Verifying the attention fraction

The whole point of the hybrid is to close the retrieval gap, so it must be measured directly
(#79): needle-in-a-haystack (swept to 8K+), phonebook lookup (exact key→value copying — the
code analog), and 5-shot MMLU. A retrieval probe must measurably improve vs pure Mamba at the
same scale, first at the 100M smoke gate (#81), then per tier in Phase 5. Decontaminate these
benchmarks out of the corpus at the dedup stage (#73).

## Tooling note

NVIDIA NeMo / Megatron-Bridge ship first-class Mamba-2 + Nemotron-H hybrid configs with the
attention ratio preset, which can save wiring the hybrid by hand on the CUDA side. At train
time the fused `mamba-ssm` + `causal-conv1d` kernels (#40) are the throughput path; the
pure-PyTorch SSD scan in `src/model/cuda_backend.py` stays the conformance reference and CPU
fallback.

## Related

- [Model: the Mamba block + selective SSM](02-model-ssm.md) — the SSD block the attention layers interleave with.
- [Corpus pipeline](08-corpus-pipeline.md) — the data the family trains on, and the sizing table.
- [Configs & locked decisions](07-configs-and-decisions.md) — `poc.yaml` and the precision benchmark.
- [Conformance: fp32 parity](03-conformance.md) — the parity checks the hybrid must keep green.
