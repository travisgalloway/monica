# Monica — Mamba-2 Hybrid POC

A proof-of-concept **Mamba-2 hybrid** language model, developed and validated on
**Apple Silicon with MLX**, architected behind **one hardware seam** so a successful POC
migrates to **CUDA** for a larger run with minimal rewrite. The current program (**M12**) is a
from-scratch, **TypeScript-first Mamba-2 hybrid Mixture-of-Experts (MoE) code model**: a mostly
Mamba-2/SSD backbone with ~12.5% attention layers for cross-file recall and Jamba-style MoE on the
MLPs, trained on a general multilingual Essential-Web + Stack-v2 corpus with its own byte-level
BPE and fill-in-the-middle, at two sizes (small ~120M-active/700M-total; large "Large A"
~700M-active/3.5B-total, sparse-upcycled from the small dense checkpoint). A **secondary axis**
studies feeding language-server / static-analysis signal into the model. See
[`docs/design/13-code-model-moe.md`](docs/design/13-code-model-moe.md) and
[issue #198](https://github.com/travisgalloway/monica/issues/198).

> The earlier **M10 distillation program** (distil a ~1B student from a frozen
> `Qwen/Qwen3-4B-Thinking-2507` teacher) was dropped 2026-07-19; its machinery is built and its
> design record is kept under [`docs/reserve/`](docs/reserve/10-distillation.md). The from-scratch
> OLMo pretrain path is complete and stays in reserve.

**Usage:** [`docs/usage.md`](docs/usage.md) — end-to-end commands (install → data →
train → serve/chat → eval). **Cloud:** [`docs/infrastructure.md`](docs/infrastructure.md)
— running the data + training pipeline on object storage + rented GPUs (R2 + RunPod).
**Design & rationale:** [`docs/design/`](docs/design/README.md).

> **Licensing note.** This project keeps its training corpus entirely **third-party/non-Claude**
> (Essential-Web, Stack-v2, Wikipedia, etc.). Anthropic's Usage Policy restricts training a model
> on *Claude's own* inputs/outputs — a different thing than using Claude Code as a coding
> assistant to build this pipeline, which is what happens here. No Claude-generated text enters the
> training corpus (this must hold for any M12 SSI reward / SFT data too). See `CLAUDE.md`'s
> "Licensing / usage-policy compliance" section for the full note, including the reserve M10
> teacher-license assessment.

---

## TL;DR — what is this?

**The model.** Monica is a small language model whose backbone is **Mamba-2**, a
_state-space model_. Where a normal Transformer re-reads the whole conversation for every
new word (a cost and memory footprint that grow with length), a state-space model keeps a
small fixed-size running summary and updates it one token at a time. That means **constant
memory per generated token and no growing KV cache** — it stays fast and cheap on a local
Mac, even for long inputs (whole files, long reasoning traces). Pure state-space models are
weak at _exact_ recall (copying a variable name, quoting a number), so Monica is a
**hybrid**: it mixes in a _few_ ordinary attention layers (roughly one in eight) to recover
precise lookup where it matters — math and code.

**How it's built.** Monica is a **code model trained from scratch**. To get strong capability
per parameter it uses a **Mixture-of-Experts (MoE)**: many small "expert" sub-networks of which
only a few fire per token, so the model holds a lot of knowledge while staying cheap to run. It's
**TypeScript-first**, trained with **fill-in-the-middle** (so it learns to complete code from both
sides, not just left-to-right), and comes in two sizes — a small rung and a larger one
**sparse-upcycled** from it (the small dense model is grown into the big sparse one instead of
training the big one from zero).

**The intended process, in order:**

1. **Build the corpus + tokenizer** — a general multilingual code+text mixture (Essential-Web +
   Stack-v2) and Monica's own byte-level BPE trained on it.
2. **Build the MoE architecture & harness** — load-balancing router, the CUDA MoE backend,
   fill-in-the-middle, a length curriculum, and the evals (bits-per-byte is the primary metric).
3. **Sweep small designs, then run** — cheaply ablate attention ratio / state size / routing,
   train the small model, then sparse-upcycle to the large one.
4. **Run it locally** — serve it on Apple Silicon, where the constant-memory design pays off.

Alongside this, a **secondary axis** asks whether feeding a language server's diagnostics into
generation helps — a validated way to raise type-cleanliness, though (so far) not functional
correctness; see [`docs/design/13-code-model-moe.md`](docs/design/13-code-model-moe.md).

This is a **proof of concept**: success is a smoothly improving learning curve plus a clear
local-hardware win (context length and tokens/sec that a same-size Transformer can't match), not a
leaderboard score. The tracker is [issue #198](https://github.com/travisgalloway/monica/issues/198).

---

## The seam (most important rule)

All hardware-specific code lives behind `src/model/interface.py` (`ModelInterface`).
Everything above it — `data/`, `train/`, `serve/`, `eval/`, `conformance/` — is portable
Python that **never imports MLX or CUDA**. Only the backend modules —
`src/model/mlx_backend.py`, `src/model/mlx_train_step.py`, `src/model/cuda_backend.py`, and
`src/model/cuda_train_step.py` — touch a hardware library. `tests/test_import_guard.py`
enforces this.

`ModelInterface`: `forward` (parallel training path) · `step` (recurrence inference path) ·
`init_state` · `get_state`/`set_state` · `save`/`load` · `config`.

## Layout

```
config/                    model dims + run params (single source of truth)
  toy.yaml toy-hybrid.yaml toy-moe.yaml      tiny smoke/correctness configs
  poc.yaml                                   ~127M from-scratch POC / dev rung (OLMo vocab, fp16)
  1b.yaml                                    ~1B from-scratch target (OLMo vocab, bf16, CUDA)
  student-1b.yaml  manifests/student-1b-*.yaml   ~1B distillation student + sweep manifests
src/model/                 interface (seam) · blocks (config + hybrid/MoE gating) · mlx/cuda backends
                           teacher · mlx_teacher · mlx_student_init · mlx_distill (distillation)
src/data/                  download · tokenize (olmo/qwen3/qwen25) · pack(uint16/uint32) · split · loader
                           corpus · shard (doc-boundary bounds) · distill_corpus · storage (R2 layout)
                           sft_*/dpo_*/reasoning_* (post-training corpora)
src/train/                 loop · schedule (warmup+cosine) · checkpoint · loss_scale
                           sft · dpo · grpo · verifiers · distill_manifest · sweep
src/eval/                  val_loss (Tier-1) · olmes_adapter (Tier-2 lm-eval) · long_context · probes
src/conformance/           forward_step_parity · backend_parity (fp32 guards)
src/serve/                 sessions · rewind · sampling · generate · spec_decode
scripts/train.py           pretrain driver       scripts/sweep.py      student sweep table
scripts/smoke_test.py      milestone-4 gate      scripts/generate.py   serve + chat CLI
scripts/sft.py dpo.py rlvr.py   post-training    scripts/eval_olmes.py lm-eval harness
tests/                     unit tests
docs/usage.md              usage guide   docs/infrastructure.md  cloud runbook   docs/design/  rationale
```

## Status

The POC core is **implemented and verified on Apple Silicon**, and the **CUDA backend is now
done and verified on a rented A40** (full suite green on both). The active program is **M12 — the
from-scratch Mamba-2 hybrid MoE code model**
([issue #198](https://github.com/travisgalloway/monica/issues/198)); the M1–M8 core was tracked in
[issue #2](https://github.com/travisgalloway/monica/issues/2). The earlier M10 distillation program
(#65) was **dropped 2026-07-19** — machinery built, kept as reserve under
[`docs/reserve/`](docs/reserve/10-distillation.md).

| Milestone                 | State                                                                             |
| ------------------------- | --------------------------------------------------------------------------------- |
| 1 Seam + MLX model        | done; MLX backend + forward/step parity verified                                  |
| 2 Data pipeline           | done; FineWeb-Edu / Wikipedia / instruction sources, unit-tested                  |
| 3 Minimal training loop   | done; `train_step` + loop exercised by the smoke gate                             |
| 4 Smoke test (gate)       | passing — resume exact, eval runs                                                 |
| 5 POC scale run           | infra done; ~1.9B-token Qwen2.5 corpus built → R2; ~205M `poc-qwen` run on RunPod; prior OLMo run hit val-ppl ~77 |
| 6 OLMES / lm-eval         | done — loglikelihood + generative (`generate_until`) tasks                        |
| 7 Serving + chat + rewind | done — CLI generate/chat over `SessionStore` + `RewindTree`                       |
| 8 CUDA backend            | **done** — pure-PyTorch Mamba-2/SSD + optional mamba-ssm fast paths; A40-verified |
| 9 Post-training           | **done** — SFT / DPO / GRPO machinery on MLX (+ CUDA step-factory parity)         |
| 10 Distillation           | machinery **built, then dropped** 2026-07-19 (reserve; see `docs/reserve/`)        |
| 12 MoE code model         | **active** — MoE build (#213/#214) is the net-new work; SSI signal secondary       |

**M12 — the active program.** A from-scratch **TypeScript-first Mamba-2 hybrid MoE code model**
(tracker [#198](https://github.com/travisgalloway/monica/issues/198); design record
[`docs/design/13-code-model-moe.md`](docs/design/13-code-model-moe.md)). The spine: own byte-level
BPE (#191) → Essential-Web + Stack-v2 corpus (#193) → aux-loss-free MoE router (#213) → CUDA MoE
backend (#214) → FIM / length curriculum / evals → ablation sweep (#219) → small full run (#222) →
sparse-upcycled large run (#223). The bulk of the net-new work is the MoE build — MoE is
MLX-toy-only today. A **secondary axis (SSI)** studies language-server / static-analysis signal:
a validated clean-rate tool with a found functional ceiling (#225/#226/#227/#230).

**Reserve — M10 distillation (dropped 2026-07-19).** The frozen-teacher machinery is built (teacher
loader `src/model/mlx_teacher.py`, student init `src/model/mlx_student_init.py`, staged loss
`src/model/mlx_distill.py`, manifest parser, sweep table, `scripts/distill.py`) but the program is
no longer active; the design record and run playbook are kept under
[`docs/reserve/`](docs/reserve/path-b-run.md).

**Two training paths (both reserve now).** The original **from-scratch pretrain** path
(`scripts/train.py`, OLMo tokenizer, uint16) is complete and validated — the POC's foundation and
production-reserve route ([#75](https://github.com/travisgalloway/monica/issues/75)). The
**distillation** path (Qwen3 tokenizer, uint32, `config/student-1b.yaml`) is reserve. M12 trains a
new code model from scratch on its own BPE.

## Hybrid architecture

The backbone is **Mamba-2 / SSD** (scalar A per head; chunked-matmul scan) + gradient
checkpointing — migrated from Mamba-1 for training throughput/memory. "Hybrid" means a small
fraction of layers are **causal attention** instead of Mamba, config-gated by `attn_every`
(layer `i` is attention iff `(i+1) % attn_every == 0`; Jamba-style ~1 attention layer per 8).
Optional **sparse MoE** FFN layers are likewise gated by `moe_every` / `n_experts` / `top_k`.
Both are off by default (pure Mamba) and live behind the seam. Sizing and the attention-fraction
sweep are in [`docs/design/09-hybrid-architectures.md`](docs/design/09-hybrid-architectures.md);
the M12 MoE code-model plan is in
[`docs/design/13-code-model-moe.md`](docs/design/13-code-model-moe.md), and the reserve
distillation rationale in [`docs/reserve/10-distillation.md`](docs/reserve/10-distillation.md).

Reference configs: `config/poc.yaml` = d_model 768 / 24 layers / d_state 16 / head_dim 64 (24
heads) / seq 1024 / ~3B tokens / OLMo vocab (pure Mamba). `config/student-1b.yaml` = d_model
2048 / 28 layers / d_state 128 / `attn_every` 8 / seq 8192 / Qwen3 vocab 151,669 / bf16 (the
~1B distillation student — a sweep seed, not a locked size; the 36-layer teacher's depth is
bridged by the adaptive Mamba-in-the-Llama init mapping). `config/toy*.yaml` are tiny fp32 smoke configs (`toy-hybrid` adds an
attention layer; `toy-moe` adds MoE). Conformance compares **fp32** at ~1e-4 rel.

## Experimental snapshotting — session rewind

The SSM carries a **small, fixed-size explicit state** per session, which makes whole-session
**snapshot / rewind** cheap and exact at turn boundaries — a feature that's awkward on a
KV-cache Transformer. Two portable primitives in [`src/serve/`](src/serve/) (no MLX/torch):

- **`SessionStore`** ([`src/serve/sessions.py`](src/serve/sessions.py)) — maps `session_id →`
  that session's recurrent state, with **constant memory per session**
  (`max_concurrent ≈ memory_budget / per_session_state`). Sessions run independently; no
  cross-session coupling.
- **`RewindTree`** ([`src/serve/rewind.py`](src/serve/rewind.py)) — snapshots the full
  cross-section of state at each turn as a node in a tree; `rewind` to any retained node makes
  it the new branch point, so a later `commit` **forks history** there. Retained nodes are
  LRU-capped (states are uniform size, so a flat count is the right budget). It rewinds the
  running summary in the fixed state — not exact per-token recall (an architectural limit, not
  a bug).

See the embedding example in [`docs/usage.md`](docs/usage.md#embedding-it-in-an-app). Because
both are portable, the same code runs unchanged on the CUDA backend.

## Quickstart

```bash
# Apple Silicon (full backend):
pip install -e ".[dev,data,mlx]"  # the mlx extra installs only on Apple Silicon

# Linux / CUDA host (training backend; runs CPU-only for conformance too):
pip install -e ".[dev,data,cuda]"        # base CUDA backend (pure-PyTorch)
pip install -e ".[dev,data,cuda-fast]"   # + mamba-ssm Triton scan + causal-conv1d

pytest                            # Mac: full suite (MLX paths included).
                                  # Linux: MLX-only tests skip (importorskip), not fail.

# Data pipeline offline smoke (no network/tokenizer):
python -m src.data.download --dummy --out data/raw --max-docs 2000
python -m src.data.tokenize --in data/raw/dummy.txt --out data/ids.npy --byte-fallback
python -m src.data.pack --in data/ids.npy --out data/packed.bin
python -m src.data.split --packed data/packed.bin --out data/split --val-tokens 2000
```

For a **real corpus** and the full train → serve → eval flow, see
[`docs/usage.md`](docs/usage.md); for the **cloud** (R2 + RunPod) pipeline, see
[`docs/infrastructure.md`](docs/infrastructure.md). To **validate every stage locally** in one
offline command — and for the small local-training configs (`config/small.yaml`,
`config/poc-small.yaml`) — see
[`docs/local-development.md`](docs/local-development.md):

```bash
scripts/local_validate.sh      # data → smoke gate → small.yaml train → distill smoke → teacher precompute
```

Run the smoke gate (the M4 gate) to confirm the backend, training loop, and exact-resume all
pass before scaling:

```bash
python scripts/smoke_test.py --data data/split
```

### Scale run (from-scratch pretrain)

[`scripts/train.py`](scripts/train.py) is the from-scratch run driver: config → model → data →
loop, with fp16 dynamic loss scaling, gradient accumulation, JSONL metrics, periodic
checkpoints, and held-out val-perplexity. The recommended `config/poc.yaml` invocation (and the
one-time ~3B-token data prep) is recorded as comments in that file. Sketch:

```bash
# ... data prep into data/split (see config/poc.yaml comments and docs/usage.md) ...
python scripts/train.py --config config/poc.yaml --data data/split --out runs/poc \
    --total-tokens 3000000000 --batch-size 32 --grad-accum 4
# resume after an interruption (auto-detects runs/poc/resume if --resume omitted):
python scripts/train.py --config config/poc.yaml --data data/split --out runs/poc \
    --total-tokens 3000000000 --batch-size 32 --grad-accum 4 --resume runs/poc/resume
```

**POC success = a smoothly decreasing `val_perplexity` in `runs/poc/metrics.jsonl`** with a
stable `grad_norm` — not a benchmark score.

### Distillation (reserve — dropped 2026-07-19)

The distillation corpus, teacher loader, student init, and staged loss exist but the program is
**no longer active**. The machinery still runs — a candidate sweep over attention fraction /
placement / state size:

```bash
python scripts/sweep.py                    # all of config/manifests/
python scripts/sweep.py config/manifests/student-1b-attn-lo.yaml config/manifests/student-1b-attn-hi.yaml
```

See [`docs/reserve/10-distillation.md`](docs/reserve/10-distillation.md) for the full (reserve)
picture.

### Post-training (M9)

Once a base exists, instruction-tune, preference-align, and RL-polish it:

```bash
python scripts/sft.py  --config config/poc.yaml --data data/sft  --init runs/poc/weights.safetensors --out runs/sft
python scripts/dpo.py  --config config/poc.yaml --data data/dpo  --init runs/sft/weights.safetensors --out runs/dpo
python scripts/rlvr.py --config config/poc.yaml --init runs/sft/weights.safetensors --problems math.jsonl --out runs/rlvr
```

### Serve & chat (M7)

[`scripts/generate.py`](scripts/generate.py) is the CLI front-end over the trained weights —
completion and an instruction-template chat REPL:

```bash
# completion
python scripts/generate.py --config config/poc.yaml --weights runs/poc/weights.safetensors \
    --prompt "Water is a chemical compound that " --max-new-tokens 80 --temperature 0.7 --top-p 0.9
# chat REPL (Ctrl-D to exit)
python scripts/generate.py --config config/poc.yaml --weights runs/poc/weights.safetensors --chat
```

At ~100M params expect roughly grammatical English, weak semantics — the goal is the learning
curve, not answer quality. See [`docs/usage.md`](docs/usage.md) for flags, sampling tips, and
embedding the generation core in an app.

### Eval (M6, Tier-2)

[`scripts/eval_olmes.py`](scripts/eval_olmes.py) runs the lm-evaluation-harness
(`pip install -e ".[eval]"`) over loglikelihood and generative tasks:

```bash
HF_DATASETS_TRUST_REMOTE_CODE=1 python scripts/eval_olmes.py \
    --config config/poc.yaml --weights runs/poc/weights.safetensors \
    --tasks hellaswag,arc_easy,arc_challenge,piqa --limit 500
```

Scores sit near chance at this scale — judge by "the harness runs end-to-end."
