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

Both reuse the teacher's attention projections and layer structure, so a conversion teacher
**close to the student's size is ideal** and it **fixes the tokenizer**. We trade exact size-match
for full openness: the chosen teacher is the 7B `open-r1/OpenR1-Distill-7B`, and the 7B→~1B gap is
bridged by the adaptive `_fit` cropping in student init (`src/model/mlx_student_init.py`).

To keep that cropping coherent, the init also **transfers the teacher's token embedding and
lm_head** (`_init_embeddings`, cropped with the same `_fit`), so the student's residual stream *is*
the teacher's first-`d_model` residual coordinates end-to-end — the per-layer corner-crop becomes a
subspace restriction rather than an arbitrary slice, and it matches the first-`min(d)` channel
alignment the hidden-state matching loss already uses (`mlx_distill._hidden_mse`). The subspace is
the **first** `d_model` coordinates; if the early distillation curve shows that is a poor warm
start, the alternative is a PCA-of-activations basis (keep the highest-variance directions and apply
that change of basis consistently across embeddings, every projection, and the matching loss) —
deferred until the first-k baseline is measured. The matching runs as the distillation loss + train
step (#100): KL on the teacher's top-k logits combined with cross-entropy, staged per the
manifest's `stages` list.

## Two teacher roles — keep them separate

| Role | Used for | Constrained by | Choice |
|---|---|---|---|
| **Conversion teacher** | init + matching (the student is built from it) | ideally ~student size + fix the tokenizer | **`open-r1/OpenR1-Distill-7B`** (fully open: SFT of Qwen2.5-Math-7B on open R1 traces, Apache-2.0; already reasoning, Qwen tokenizer) |
| **Trace-generation teacher** | producing reasoning traces in post-training | **unconstrained** — traces are re-tokenized | the strongest Qwen-tokenizer R1 distill you can run (14B/32B) |

**Why fully open:** OpenR1-Distill-7B is HuggingFace's Open-R1 reproduction of R1 distillation —
the reasoning training data and recipe DeepSeek kept closed are openly released here
(Mixture-of-Thoughts ~350k verified traces, OpenR1-Math-220k, CodeForces-CoTs), Apache-2.0. It
keeps the Qwen2 architecture and Qwen2.5 tokenizer, so it is a drop-in for the existing teacher
forward — no re-tokenization, no new architecture. (It does not open Qwen's *pretraining* corpus;
OLMo 2 would, but as a non-reasoning teacher needing a tokenizer + architecture rewrite.)

Smaller / alternative conversion teachers on the same tokenizer: the original
DeepSeek-R1-Distill-Qwen-1.5B (MIT, weights-only), Qwen2.5-Coder/-Math-1.5B (Apache-2.0). Avoid
Qwen2.5 3B/72B (Qwen license, not Apache-2.0) and Llama / StarCoder2 as a tokenizer source (use
restrictions). The MOHAWK lineage's demonstrated teacher family was Phi (Phi-4-mini, MIT) —
usable, but on a different tokenizer.

### The teacher loader behind the seam (#93)

The conversion teacher is loaded **frozen and forward-only** behind the hardware seam, exactly
like DPO's frozen reference (`mlx_train_step.make_dpo_train_step` holds a distinct `ref_model`
the optimizer never touches). The portable protocol is `ConversionTeacher`
(`src/model/teacher.py`): `forward(return_hidden=...)` (logits + optional per-layer hidden
states), `topk_logits(tokens, k)` (the cached signal #94/#100 match against), and
`attention_projection(layer)` (the Q/K/V/O the #99 init maps onto the student SSM's
C/B/input/output). Above the seam, callers see only opaque arrays plus a `to_numpy` converter —
never a backend array type — and the teacher reports **no** trainable parameters, so it is
structurally excluded from the optimizer and the resume bundle.

The MLX implementation (`src/model/mlx_teacher.py`, a minimal self-contained Qwen2 forward — no
`mlx-lm` dependency) loads real HF weights via `from_pretrained` (a checkpoint dir or, lazily, an
HF repo id) and builds a tiny **synthetic** teacher via `from_config` for offline tests and small
local checks. Build one through `get_backend(...).make_teacher(...)`; the CUDA teacher is deferred
(the branch raises `NotImplementedError`).

### Student init from the teacher (#99)

A trial is a **manifest** (`config/manifests/*.yaml`); `src/train/distill_manifest.py` parses it
(portable, above the seam): `init:` resolves to an `InitMethod`, `stages:` is validated against
the canonical list (the distill stages `mixing-match → hidden-align → logit-distill` plus the
post-training ones), and `manifest_to_config` resolves the `layout` sweep-schema
(`d_model`, `n_layers`, `attention_every → attn_every`, `state_size → d_state`) + `tokenizer →
vocab_size` onto a `MambaConfig`.

`get_backend(...).init_student(student, teacher, method)`
(`src/model/mlx_student_init.py`) then performs the conversion. **Mamba-in-the-Llama** maps the
teacher attention onto the student SSM — **Q → C**, **K → B** (the two `d_state` slices of the SSM
`x_proj`), **V → input** (`in_proj` main half), **O → output** (`out_proj`) — copies the kept
attention layers from the teacher and **freezes** them (the student has no MLP blocks, so the
retained attention plays the role the paper's frozen MLPs do; the trainable set is the new Mamba
layers). **MOHAWK** is a lighter init (copy attention, leave Mamba at default, freeze nothing);
the matching happens in the staged distill loss (#100). Because teacher and student widths differ,
the mapping is **adaptive** (exact copy where dims align, else copy-overlap + zero-pad/truncate);
init quality is judged by the downstream distillation curve, not by exactness. Freezing uses MLX's
native `nn.Module.freeze`, which `nn.value_and_grad` already honors, so the train step is
unchanged. The CUDA initializer is deferred.

### The staged distillation loss + train step (#100)

The student trains against the **cached** teacher signal through the manifest's distillation
stages (`distill_stages(manifest)` → `mixing-match → hidden-align → logit-distill`, in order).
Each stage is a separate injected `TrainStepFn` from
`get_backend(...).make_distill_train_step(model, opt, stage=...)`
(`src/model/mlx_distill.py`), mirroring SFT/DPO/GRPO and funnelling through the shared
`_accumulate_and_step`, so the backend-free loop is unchanged:

- **`logit-distill`** — compound `ce_weight·CE + kl_weight·KL_topk`, where `KL_topk` is the
  `T²`-scaled KL between the teacher's cached **top-k** distribution and the student's logits
  renormalized over the same support. No teacher inference in the loop (acceptance: KL+CE from
  cached top-k).
- **`hidden-align`** — MSE between per-layer hidden states, with **cached** teacher hidden states
  in the micro-batch *or* **on-the-fly** recompute (`teacher.forward(return_hidden=True)`,
  stop-gradient); compared over the overlapping `min(d)` channels (the width mismatch).
- **`mixing-match`** — MSE between the student's head-averaged SSM **mixing matrix**
  (`SelectiveSSM.mixing_matrix`, the materialized 1-semiseparable matrix, verified to reproduce
  the scan) and the teacher's head-averaged causal attention matrix. A tractable simplification of
  MOHAWK's strict per-layer teacher-forced orientation (each runs on its own forward, since the
  widths differ).

The compound loss is a single scalar, so the dynamic fp16 loss scaler
(`train.loss_scale.DynamicLossScaler`) covers it unchanged — the combined term is scaled before
backprop and overflowing steps skip cleanly. The CUDA distill step is deferred.

## The tokenizer is fixed by the conversion teacher (#90, #91)

The student must **share a vocabulary with the conversion teacher** for logit and hidden-state
matching, so the tokenizer is **Qwen2.5 (vocab 151,646)** — shared across Qwen2.5/-Coder/-Math,
OpenR1-Distill-7B, and every DeepSeek-R1-Distill-Qwen variant, so those teachers are
interchangeable without re-tokenizing. Adopted for the
production model too, which collapses the POC-to-production tokenizer question (no re-tokenization).
This exceeds the uint16 bound → **uint32 packing** (#90); see
[corpus pipeline](08-corpus-pipeline.md).

## Precompute once, sweep students cheaply (#94, #98)

Everything that depends only on the **teacher + corpus** — not the student — is computed a single
time and reused by every trial:

- The tokenized distillation corpus (#92): `src/data/distill_corpus.py` orchestrates the existing
  clean → Qwen2.5-tokenize → uint32-pack stages into `poc-distill/corpus/{cleaned,tokenized/qwen25-8k}`
  with doc-boundary sidecars and a corpus manifest — the exact path the manifests below name.
- **Teacher outputs** over it: top-50..100 logits + indices per token (#94); optionally hidden
  states for MOHAWK matching. The teacher forward pass is the dominant cost — paid **once**.
- The shared SFT corpora and verifiable RL sets ([post-training](11-post-training.md)).

Each student trial is then a lightweight **manifest** naming the frozen artifacts + the layout:

```yaml
student: 1b-attn12pct
conversion_teacher: open-r1/OpenR1-Distill-7B
tokenizer: qwen25               # Qwen2.5 vocab, 151646
seq_len: 8192
layout: { d_model: 2048, n_layers: 28, attention_every: 8, state_size: 128 }   # ~1.03B; matches the teacher's 28 layers
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
