# Hybrid SSM + Transformer architectures (model choice)

[← Index](README.md)

Why the scale-up model is a **Mamba-2 *hybrid*** — mostly SSD blocks with a small fraction
of attention layers — and how the **100M → 1B → 2B → 4B** family is sized. This is the model
companion to the [corpus pipeline](08-corpus-pipeline.md); the GitHub tracker is
[issue #65](https://github.com/travisgalloway/monica/issues/65) (arch children #66–#68). The
SSD block itself is documented in [model: the Mamba block](02-model-ssm.md); this doc is
about what to add to it at scale.

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

## Sizing the family

Mamba has **no KV cache**, so fp16/bf16 weights ≈ total inference footprint — the ×4 ladder
maps cleanly to GPU/RAM tiers and to the RunPod [sizing table](08-corpus-pipeline.md). 16B
(an earlier candidate) is dropped; the series tops out at 4B for this POC.

| tier | d_model | n_layers | head_dim | ≈ params | bf16 weights | train GPU |
|---|---|---|---|---|---|---|
| **100M** (poc, exists) | 768 | 24 | 64 | ~127M | ~0.25 GB | T4 16 GB / L4 24 GB |
| **1B** | 2048 | 36 | 64 | ~1.0B | ~2 GB | L4 / A10 24 GB |
| **2B** | 2560 | 44 | 64 | ~2.0B | ~4 GB | A100 40 GB / L40S 48 GB |
| **4B** | 3584 | 48 | 64 | ~4.0B | ~8 GB | A100 80 GB / H100 80 GB |

The 100M tier is the validated [`poc.yaml`](07-configs-and-decisions.md); the 2B dims are
provisional until the sizing tool (#66) sets them to hit the target within ±5%. The sizing
tool is a portable closed-form param/memory calculator (`src/model/sizing.py` +
`scripts/model_size.py`, #66), cross-checked against the built model's portable state-dict
param sum. Training memory ≈ 8 B/param (weights+grad+AdamW), or **~10 B/param with 8-bit
Adam** — the VRAM-tight lever used in Phase 5 (#75).

### Precision differs from the POC

`poc.yaml` uses **fp16 + loss scaling** because fp16 is ~18% faster than bf16 *on MLX/Metal*
(see [configs & decisions](07-configs-and-decisions.md)). The scale configs train on **CUDA**,
where **bf16 is native** and needs no loss scaling — so `config/{1b,2b,4b}.yaml` set
`precision: bf16` (`scaler_for_precision` already returns `None` for bf16). `tie_embeddings`
and `grad_checkpoint` stay on; vocab follows the chosen tokenizer (must stay < 65536 — see
[corpus pipeline](08-corpus-pipeline.md)).

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
