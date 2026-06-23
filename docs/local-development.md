# Local development on Apple Silicon (MLX)

MLX is the **dev/validation backend**: it exists so you can prove a change correct, generate
small amounts of teacher signal, and train test-scale models *locally* before paying for a CUDA
cloud run. Scale training itself runs on CUDA/RunPod (see
[`docs/infrastructure.md`](infrastructure.md)) — so this page is about **developer velocity**, not
throughput. (The MLX *train-step throughput* optimizations were deliberately retired; see issue
#30's decision record before reopening that.)

Three things live here:

1. [Validate every stage locally in one command](#1-validate-every-stage-locally)
2. [Train test models locally (`small.yaml`, `poc-small.yaml`)](#2-train-test-models-locally)
3. [Generate teacher signal locally (Qwen3 via MLX, or LM Studio)](#3-generate-teacher-signal-locally)

All commands assume the Apple-Silicon install (`pip install -e ".[dev,data,mlx]"`) and the venv
at `.venv`.

---

## 1. Validate every stage locally

```bash
scripts/local_validate.sh
```

One offline command (no network / HF / weights) that fails fast through every pipeline stage:

| Stage | What it runs | What it proves |
|---|---|---|
| 1 data | `download --dummy` → `tokenize --byte-fallback` → `pack` → `split` | the data pipeline end-to-end |
| 2 smoke | `scripts/smoke_test.py` on a **fresh** byte split | resume is bit-exact + val eval runs (fp32) |
| 3 train | `scripts/train.py --config config/small.yaml` | the real fp16 + loss-scaling training path |
| 4 distill | `scripts/distill_smoke.py` | the 3 staged losses (mixing-match → hidden-align → logit-distill) |
| 5 teacher | `scripts/precompute_teacher.py --backend mlx --synthetic --compile` | the #94 precompute + the `mx.compile` lever |

Knobs (env vars): `PYTHON` (default `.venv/bin/python`), `WORK` (default `runs/local-validate`),
`STEPS` (default 20), `KEEP=1` to keep the work dir. Use this as the pre-push gate for any change
to the loop, the SSD scan, mixed precision, checkpointing, or the distill stages.

> The smoke gate must run on a **freshly built byte split**, not the real `data/split` (which is
> the OLMo-vocab corpus) — `local_validate.sh` builds one for you.

---

## 2. Train test models locally

There are now four rungs, so you can pick the one that matches your iteration speed:

| Config | Params | Use | Cost (measured, M-series) |
|---|---|---|---|
| `config/toy.yaml` | ~64K | correctness / exact-resume gate | instant |
| `config/small.yaml` | ~2.6M | **fast code-path iteration** (byte vocab, fp16) | **~0.08 s/step** @ 2,048 tok, ~0.8 GB |
| `config/poc-small.yaml` | ~97M | **largest "real" model trainable locally** (OLMo vocab) | **~18.8 s/step** @ 32,768 tok, ~12.9 GB |
| `config/poc.yaml` | ~127M | cloud / reserve scale run | ~99 s/step @ 131,072 tok |

`small.yaml` is for *validating that training/distill code works*, in seconds. `poc-small.yaml` is
the ≤100M "trainable locally" target — real Mamba-2/SSD architecture and a real tokenizer, but a
Chinchilla-ish run is still **days** of local compute (it's for short POC runs; use CUDA cloud for
scale). Both carry their measured step-time in the YAML header.

```bash
# fast loop (byte corpus from stage 1 above):
.venv/bin/python scripts/train.py --config config/small.yaml --data <byte-split> \
    --out runs/small --total-steps 200 --batch-size 8 --grad-accum 1

# ~97M local POC (needs a real OLMo-tokenized corpus):
.venv/bin/python scripts/train.py --config config/poc-small.yaml --data data/split \
    --out runs/poc-small --total-tokens 200000000 --batch-size 16 --grad-accum 2

# measure step-time / peak memory for any config:
.venv/bin/python scripts/bench_train_step.py --config config/poc-small.yaml --batch 16 --grad-accum 2
```

---

## 3. Generate teacher signal locally

The distillation student trains against **cached teacher top-k logits** (the #94 precompute), then
the `logit-distill` stage matches them with KL. You can produce that signal locally two ways.

### 3a. Real Qwen3 teacher via MLX — *the recommended path*

`MLXConversionTeacher.from_pretrained` runs the actual Qwen3 weights on Apple Silicon (white-box:
full forward, hidden states, and Q/K/V/O projections — so it supports init #99 and **all** matching
stages #100, not just logit-distill):

```bash
.venv/bin/python scripts/precompute_teacher.py \
    --manifest config/manifests/student-1b-attn-hi.yaml \
    --data data/poc-distill/split --backend mlx \
    --pretrained Qwen/Qwen3-4B-Thinking-2507 \
    --teacher-dtype fp16 --compile --k 50 --out runs/teacher-local
```

Two **local levers** (both opt-in; default is the bit-identical eager fp32 path, so conformance and
the smoke gate are untouched):

- `--teacher-dtype fp16` — hold/compute the teacher in fp16. Halves teacher memory (a 4B teacher
  ~16 GB → ~8 GB), so it fits comfortably on a 32 GB Mac. RMSNorm/softmax stay fp32 internally.
- `--compile` — `mx.compile` the teacher's logits-only forward (fixed-shape, forward-only — *not*
  the student forward/eval path #30 rejected). Fuses the per-layer op stream; the cached top-k
  **indices are identical** to eager (values within fp tolerance).

> The `--pretrained` id must be one `precompute_teacher.py` knows (`_teacher_config_for`), so the
> teacher's `effective_vocab_size` matches the manifest tokenizer vocab — otherwise it fails loudly
> rather than caching unusable indices. At real corpus scale the footprint is large (k=50 ≈ 300
> B/token); keep local runs to a small split.

### 3b. LM Studio / OpenAI-compatible endpoint — *partial, convenience only*

If a Qwen3 model is already loaded in [LM Studio](https://lmstudio.ai) (or llama.cpp `--server`,
vLLM, …), point the precompute at its endpoint:

```bash
.venv/bin/python scripts/precompute_teacher.py \
    --manifest config/toy-distill.yaml --data <split> --backend mlx \
    --teacher-endpoint http://localhost:1234/v1 --endpoint-model qwen3-4b --k 10 \
    --out runs/teacher-lmstudio
```

**This path is partial and approximate** (`src/model/api_teacher.py` documents it in full):

- It implements **only** `topk_logits`, so it feeds the **logit-distill** stage only. An HTTP
  endpoint exposes no weights/hidden states, so init (#99), `hidden-align`, and `mixing-match`
  (#100) are **not** available — use 3a for those.
- Values are **log-probs, not logits** (the KL temperature scaling is exact only at T=1), the
  top-k count is **server-capped** (often ≤10–20), and top tokens are mapped string→id **best
  effort** with the Qwen3 tokenizer (re-tokenization can drift the per-position alignment; the
  teacher warns once).

Prefer 3a for any real local validation; reach for 3b only as a quick convenience when the model
is already serving.

---

See also: [`docs/usage.md`](usage.md) (full flow), [`docs/design/10-distillation.md`](design/10-distillation.md)
(distillation design), [`docs/infrastructure.md`](infrastructure.md) (cloud R2 + RunPod).
