# Distillation (teacher → hybrid student)

[← Index](README.md)

Why the POC **distills** a compact Mamba-2 hybrid from a frozen transformer teacher instead of
pretraining from scratch, and how that makes an architecture search cheap. The tracker is
[issue #65](https://github.com/travisgalloway/monica/issues/65); the corpus + teacher-output
storage is in [corpus pipeline](08-corpus-pipeline.md); the model is in
[hybrid architectures](09-hybrid-architectures.md).

## Why distill, not pretrain

The POC is an **architecture search** over attention fraction, layer placement, and state size,
and each candidate is a fresh student. From-scratch pretraining makes that search impossible —
each trial would cost hundreds of billions of tokens. Distilling against a **frozen teacher
signal** reaches capability for **under 1% of the from-scratch token count** (MOHAWK produced a
capable student from ~3B tokens, a hybrid from ~5B), so each candidate is cheap. From-scratch
pretraining is demoted to a **production-reserve** stage (#75) that begins only after a layout
validates.

## Two conversion methods

**Mamba-in-the-Llama** (default, #99) initializes the Mamba layers **directly from the teacher's
attention projections** — mapping **Q, K, V, O** onto the SSM's **C, B, input, and output**
projections — keeps a fraction of attention layers, **freezes the MLPs**, and runs progressive
distillation followed by SFT and DPO. The reference run took under five days on 8 A100s.
Reference: [Mamba in the Llama](https://arxiv.org/abs/2408.15237),
[`jxiw/MambaInLlama`](https://github.com/jxiw/MambaInLlama).

**MOHAWK** (alternative, #99) distills a Mamba-2 student through **progressive matching**: first
the mixing matrices, then hidden states, then final logits. Reference:
[MOHAWK](https://arxiv.org/abs/2408.10189).

Both reuse the teacher's attention projections and layer structure, which is why the **conversion
teacher must be close to the student's size (~1.5B)** and **fixes the tokenizer**. The matching
runs as the distillation loss + train step (#100): KL on the teacher's top-k logits combined with
cross-entropy, staged per the manifest's `stages` list.

## Two teacher roles — keep them separate

| Role | Used for | Constrained by | Choice |
|---|---|---|---|
| **Conversion teacher** | init + matching (the student is built from it) | **must** be ~student size + fix the tokenizer | **DeepSeek-R1-Distill-Qwen-1.5B** (MIT, already reasoning, Qwen tokenizer) |
| **Trace-generation teacher** | producing reasoning traces in post-training | **unconstrained** — traces are re-tokenized | the strongest Qwen-tokenizer R1 distill you can run (14B/32B) |

Code/math conversion alternatives on the same tokenizer: Qwen2.5-Coder-1.5B, Qwen2.5-Math-1.5B
(Apache-2.0). Avoid Qwen2.5 3B/72B (Qwen license, not Apache-2.0) and Llama / StarCoder2 as a
tokenizer source (use restrictions). The MOHAWK lineage's demonstrated teacher family was Phi
(Phi-4-mini, MIT) — usable, but on a different tokenizer.

## The tokenizer is fixed by the conversion teacher (#90, #91)

The student must **share a vocabulary with the conversion teacher** for logit and hidden-state
matching, so the tokenizer is **Qwen2.5 (vocab 151,646)** — shared across Qwen2.5/-Coder/-Math and
every DeepSeek-R1-Distill-Qwen variant, so those teachers are interchangeable. Adopted for the
production model too, which collapses the POC-to-production tokenizer question (no re-tokenization).
This exceeds the uint16 bound → **uint32 packing** (#90); see
[corpus pipeline](08-corpus-pipeline.md).

## Precompute once, sweep students cheaply (#94, #98)

Everything that depends only on the **teacher + corpus** — not the student — is computed a single
time and reused by every trial:

- The tokenized distillation corpus (#92).
- **Teacher outputs** over it: top-50..100 logits + indices per token (#94); optionally hidden
  states for MOHAWK matching. The teacher forward pass is the dominant cost — paid **once**.
- The shared SFT corpora and verifiable RL sets ([post-training](11-post-training.md)).

Each student trial is then a lightweight **manifest** naming the frozen artifacts + the layout:

```yaml
student: 1b-attn12pct
conversion_teacher: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
tokenizer: qwen25               # Qwen2.5 vocab, 151646
seq_len: 8192
layout: { d_model: 2048, n_layers: 48, attention_every: 8, state_size: 128 }
init: mamba-in-the-llama        # or mohawk
stages: [mixing-match, hidden-align, logit-distill, instruct-sft, reasoning-sft, tool-sft, grpo]
corpus: poc-distill/corpus/tokenized/qwen25-8k
teacher_outputs: poc-distill/teacher-outputs/topk-logits
sft: shared/sft/tokenized/qwen25-8k
rl: shared/rl
schedule: { lr: 3.0e-4, warmup: 0.02, batch_tokens: 1_000_000 }
```

The `layout` keys (`attention_every`, `state_size`) are the manifest's own sweep-schema
names; the #98 harness maps them onto the model config fields (`MambaConfig.attn_every` /
`d_state`). A sweep over architectures is a set of sibling manifests pointing at the **same**
teacher signal.

## What invalidates the precompute

- **Change the teacher** → invalidates the teacher outputs and the traces (the largest artifacts).
  So the teacher is **fixed first** (#91).
- **Change the tokenizer** → invalidates the tokenized corpus and the logits (indices shift) —
  which is why the teacher's tokenizer is the natural pinned choice.
- **Change the student layout** → invalidates **nothing** upstream. *This is the point: the
  student is free, so layout sweeps are cheap.*
- **Add RL problems** → appends to the verifiable sets rather than rebuilding anything.

## Reading the POC

It validates the architecture — the attention fraction, the layer placement, and the state size —
which are **tokenizer-independent and transfer**. Absolute capability numbers still come from the
from-scratch production run, so the POC is a **layout decision and a feasibility check**, sealed
by the local-hardware headline metric (#104).

## Related

- [Corpus pipeline](08-corpus-pipeline.md) — the distillation corpus, teacher outputs, storage layout.
- [Hybrid architectures](09-hybrid-architectures.md) — the student the teacher converts into.
- [Post-training](11-post-training.md) — the three capability layers applied after conversion.
- [Training](05-training.md) — the backend-free loop the distill train step plugs into.
