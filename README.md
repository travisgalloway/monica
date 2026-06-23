# Monica — Mamba-2 Hybrid POC

A proof-of-concept **Mamba-2 hybrid** language model, developed and validated on
**Apple Silicon with MLX**, architected behind **one hardware seam** so a successful POC
migrates to **CUDA** for a larger run with minimal rewrite. The current program is to
**distil** a compact **~1B** hybrid student from a larger frozen teacher, sweep a few
architecture layouts cheaply, then post-train the winner for reasoning. (1B is the single
target model — the `poc` is the cheap architecture-validation rung, run at ~205M with the
Qwen2.5 tokenizer (`config/poc-qwen.yaml`) to mirror the student's data path; a ~127M OLMo
variant, `config/poc.yaml`, stays in reserve.)

**Usage:** [`docs/usage.md`](docs/usage.md) — end-to-end commands (install → data →
train/distil → serve/chat → eval). **Cloud:** [`docs/infrastructure.md`](docs/infrastructure.md)
— running the data + training pipeline on object storage + rented GPUs (R2 + RunPod).
**Design & rationale:** [`docs/design/`](docs/design/README.md).

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

**How it learns.** Training a model from scratch is enormously expensive. Instead, Monica
**learns from a bigger, already-trained "teacher" model** — a technique called
**distillation**. The student watches what the teacher would predict and learns to match it,
reaching useful capability for a _tiny fraction_ of the data a from-scratch run needs. Because
each student is cheap to train, we can **try several small designs** (how much attention, where
to place it, how big the state is) and keep the best one.

**The intended process, in order:**

1. **Precompute the teacher's signal once** — tokenize a corpus and record the teacher's
   predictions. This is the expensive part, and it's done a single time.
2. **Sweep student layouts** — train several cheap candidate students against that frozen
   signal and pick the layout that wins on math/code _and_ on the local speed/long-context
   advantage.
3. **Post-train the winner** — teach it to follow instructions and to reason
   (`<think>…</think>` style), with optional tool-use and a final reinforcement-learning polish.
4. **Run it locally** — serve it on Apple Silicon, where the constant-memory design pays off.

This is a **proof of concept**: success is a smoothly improving learning curve plus a clear
local-hardware win (context length and tokens/sec that a same-size Transformer can't match), not a
leaderboard score. The from-scratch training path is fully built and validated; the
**distillation stage is in progress** (the building blocks exist; the end-to-end cloud run is
being wired up — see [issue #65](https://github.com/travisgalloway/monica/issues/65)).

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
done and verified on a rented A40** (full suite green on both). The active program is **M10 —
distillation** ([issue #65](https://github.com/travisgalloway/monica/issues/65)); the M1–M8
core was tracked in [issue #2](https://github.com/travisgalloway/monica/issues/2).

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
| 10 Distillation           | **in progress** — teacher loader, student init, staged loss, sweep harness built  |

**M10 distillation — where it stands.** The frozen-artifact strategy and its building blocks
are in place: the frozen conversion teacher (`Qwen/Qwen3-4B-Thinking-2507`, loaded by
`src/model/mlx_teacher.py` / `cuda_teacher.py`), student initialization
(`src/model/mlx_student_init.py` — Mamba-in-the-Llama / MOHAWK), the staged distillation loss
(`src/model/mlx_distill.py` — mixing-match → hidden-align → logit-distill), the manifest parser
(`src/train/distill_manifest.py`), the sweep table (`scripts/sweep.py`), and the run driver
(`scripts/distill.py`). **Pending:** the corpus re-tokenize to the Qwen3 vocab + corpus-scale
teacher-logit precompute ([#94](https://github.com/travisgalloway/monica/issues/94)), the R2 +
RunPod plumbing ([#80](https://github.com/travisgalloway/monica/issues/80)), and the end-to-end
cloud distill run at full scale ([#81](https://github.com/travisgalloway/monica/issues/81); the
runbook is [`docs/path-b-run.md`](docs/path-b-run.md)).

**Two training paths.** The original **from-scratch pretrain** path (`scripts/train.py`,
OLMo tokenizer, uint16) is complete and validated — it's the POC's foundation and the
production-reserve route ([#75](https://github.com/travisgalloway/monica/issues/75)). The
**distillation** path (Qwen3 tokenizer, uint32, `config/student-1b.yaml`) is the current
focus and the cheaper route to a capable model.

## Hybrid architecture

The backbone is **Mamba-2 / SSD** (scalar A per head; chunked-matmul scan) + gradient
checkpointing — migrated from Mamba-1 for training throughput/memory. "Hybrid" means a small
fraction of layers are **causal attention** instead of Mamba, config-gated by `attn_every`
(layer `i` is attention iff `(i+1) % attn_every == 0`; Jamba-style ~1 attention layer per 8).
Optional **sparse MoE** FFN layers are likewise gated by `moe_every` / `n_experts` / `top_k`.
Both are off by default (pure Mamba) and live behind the seam. Sizing and the attention-fraction
sweep are in [`docs/design/09-hybrid-architectures.md`](docs/design/09-hybrid-architectures.md);
the distillation rationale is in [`docs/design/10-distillation.md`](docs/design/10-distillation.md).

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
[`docs/infrastructure.md`](docs/infrastructure.md).

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

### Distillation (in progress)

The distillation corpus, teacher loader, student init, and staged loss exist; the cloud run
harness is being wired up. Inspect a candidate sweep over attention fraction / placement / state
size:

```bash
python scripts/sweep.py                    # all of config/manifests/
python scripts/sweep.py config/manifests/student-1b-attn-lo.yaml config/manifests/student-1b-attn-hi.yaml
```

See [`docs/usage.md`](docs/usage.md#3-train) and
[`docs/design/10-distillation.md`](docs/design/10-distillation.md) for the full picture.

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
