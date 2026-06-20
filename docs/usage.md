# Usage guide

End-to-end commands for the Mamba-2 Hybrid POC: **install → data → train/distil →
serve/chat → eval**. For *why* the project is built this way, see
[`design/`](design/README.md); for running this pipeline on cloud storage + rented GPUs
(R2 + RunPod), see [`infrastructure.md`](infrastructure.md).

The project has **two training paths**:

- **Distillation (current focus)** — build a compact **~1B** Mamba-2 hybrid student from a
  larger frozen teacher (Qwen2.5 tokenizer → uint32 packing). *In progress:* the building
  blocks exist; the end-to-end run harness is being wired up.
- **From-scratch pretrain (validated foundation / production reserve)** — train a ~100M POC
  from scratch (OLMo tokenizer → uint16). Complete and exercised by the smoke gate.

Everything here runs on **Apple Silicon via MLX**; the same portable code runs on the
**CUDA** backend (`pip install -e ".[dev,data,cuda]"`). Commands use the project venv
(`.venv/bin/python`); from a fresh shell substitute `python` once the venv is activated. All
paths are relative to the repo root.

---

## 1. Install

```bash
# Apple Silicon (full backend — required for training/serving/eval on a Mac):
pip install -e ".[dev,data,mlx]"     # the mlx extra installs only on Apple Silicon

# Linux / CUDA host (training backend; also runs CPU-only for conformance on a Mac):
pip install -e ".[dev,data,cuda]"        # base CUDA backend (pure-PyTorch)
pip install -e ".[dev,data,cuda-fast]"   # + mamba-ssm Triton scan + causal-conv1d fast paths

# Optional, for the Tier-2 benchmark harness:
pip install -e ".[eval]"             # pulls lm-eval (and transitively torch)
```

Tokenizers auto-download from the HF Hub on first use (cached under `~/.cache/huggingface`,
needs network once):

- **OLMo** (`allenai/OLMo-7B-hf`, vocab 50,280 < 65,536) → **uint16** packing — the
  from-scratch POC path.
- **Qwen2.5** (vocab 151,646 ≥ 65,536) → **uint32** packing — the distillation path, fixed by
  the conversion teacher (`open-r1/OpenR1-Distill-7B`).

Run the tests to confirm the install (on Linux the MLX-only tests skip, not fail):

```bash
.venv/bin/python -m pytest -q -rs
```

---

## 2. Build a corpus

### 2a. Distillation corpus (current focus — Qwen2.5, uint32)

[`src/data/distill_corpus.py`](../src/data/distill_corpus.py) is a thin orchestrator that
builds the **frozen** distillation corpus every student trial consumes: it cleans text into
durable Parquet, then Qwen2.5-tokenizes and packs it into fixed-length training shards with a
**document-boundary sidecar** (`.bounds`, for SSM state reset at doc edges, #68). It adds no
new logic over the stages below — only the `poc-distill/` layout and a corpus-level manifest.

```bash
# From a one-document-per-line text file (use --source dummy for an offline smoke).
# --out-root is the poc-distill class root; it defaults to data/poc-distill (omit to use it):
.venv/bin/python -m src.data.distill_corpus --source text --in data/raw/slice.txt \
    --tokenizer qwen25 --seq-len 8192 --out-root data/poc-distill
```

This writes:

```
data/poc-distill/corpus/
  cleaned/      part-*.parquet          durable, re-mixable text
  tokenized/qwen25-8k/
    part-*.bin      uint32 tokens (Qwen2.5 vocab 151,646 → uint32, #90)
    part-*.bounds   uint8 doc-start flags
    manifest.json   {seq_len, dtype, tokenizer, n_tokens, ...}
  manifest.json     two-stage summary (the artifact teacher-logit precompute freezes against)
```

The corpus is **precomputed once**: the teacher-logit precompute and every student layout
sweep read it unchanged. The three-class storage layout (`poc-distill/` · `shared/` ·
`reserve-pretrain/`) is defined in [`src/data/storage.py`](../src/data/storage.py) and doubles
as the R2 prefix scheme — see [`infrastructure.md`](infrastructure.md). For reasoning traces
that must not straddle a sequence boundary, [`src/data/shard.py`](../src/data/shard.py)'s
`pack_atomic` mode bin-packs each document whole (#96).

### 2b. From-scratch corpus (secondary — OLMo, uint16)

The classic pipeline is **one document per line → tokenize (uint16) → pack → split**. Three
sources via `--source`; mix them by concatenating the text files.

```bash
# English Wikipedia (clean encyclopedic prose) — the bulk of the corpus
.venv/bin/python -m src.data.download --source wikipedia --out data/raw/wiki.txt --max-docs 420000

# Instruction pairs (Dolly-15k), oversampled so the chat format is learned
.venv/bin/python -m src.data.download --source instruct --out data/raw/instruct.txt --repeat 4

# Concatenate INSTRUCT FIRST so the token cap / val split only ever trims Wikipedia
cat data/raw/instruct.txt data/raw/wiki.txt > data/raw/corpus.txt

# Tokenize + pack (OLMo tokenizer, EOS per line), capped near a target token count
.venv/bin/python -m src.data.tokenize --in data/raw/corpus.txt --out data/packed.bin --max-tokens 110000000

# Split off a held-out validation shard (disjoint tail)
.venv/bin/python -m src.data.split --packed data/packed.bin --out data/split --val-tokens 5000000
```

This yields `data/split/{train.bin,val.bin}` (+ `.meta.json` sidecars recording the dtype).
`--source fineweb` (FineWeb-Edu web text) is also available; `--dummy` produces synthetic
offline text for pipeline smoke tests. The dtype is **vocab-driven**: vocab < 65,536 → uint16,
otherwise uint32 (`packing_dtype_for` in [`src/data/pack.py`](../src/data/pack.py)).

> **Tip — sizing:** ~257 OLMo tokens/Wikipedia article, so ~420k docs ≈ ~108M tokens. Pick
> `--max-docs` and `--max-tokens` for your token budget. At ~99 s/step the M1 Pro does
> ~114M tokens/day, so a ~100M-token run is ~1 day.

---

## 3. Train

### 3a. From-scratch pretrain — `scripts/train.py`

[`scripts/train.py`](../scripts/train.py) wires config → model → data → loop, with fp16
dynamic loss scaling, gradient accumulation, JSONL metrics, periodic checkpoints, and held-out
perplexity. Run it under `caffeinate`/`nohup` for a long unattended run.

```bash
caffeinate -i nohup .venv/bin/python scripts/train.py \
    --config config/poc.yaml --data data/split --out runs/poc \
    --total-tokens 100000000 --batch-size 8 --grad-accum 16 \
    --base-lr 3e-4 --warmup-steps 40 --grad-clip 1.0 \
    --log-every 10 --eval-every 50 --ckpt-every 100 --seed 0 \
    > runs/poc/train.log 2>&1 &
```

- **Effective batch = `batch-size × grad-accum × seq_len` tokens/step** (here 8 × 16 × 1024 =
  131,072). `batch 8 / accum 16` keeps peak RAM ~12–17 GB; `batch 32 / accum 4` is faster
  (~25 GB peak) — use the smaller batch if you're also using the machine, the larger if it's
  dedicated.
- **`--backend {auto,mlx,cuda}`** (default `auto`) selects the hardware backend.
- **Resume** after any interruption (auto-detects `runs/poc/resume`): re-run the same command,
  optionally adding `--resume runs/poc/resume`.
- **Checkpoints** are a crash-safe, double-buffered store under `runs/poc/resume/` (atomic
  `LATEST` pointer; the previous checkpoint always survives until the next is durably written)
  plus a portable `runs/poc/weights.safetensors`. To keep a good checkpoint, `cp` it aside at a
  val-perplexity low.

**Monitor:**

```bash
tail -f runs/poc/metrics.jsonl     # {step, lr, loss, grad_norm, val_perplexity, tokens_per_sec, ...}
```

**POC success = a smoothly decreasing `val_perplexity`** with stable `grad_norm`. (A reference
~100M-param run on ~100M tokens reached val-perplexity ~77.)

The smaller `config/toy.yaml` (vocab 256, fp32) backs the exact-resume smoke gate:

```bash
.venv/bin/python scripts/smoke_test.py --data data/split    # use a byte-fallback split
```

### 3b. Distillation (in progress)

Distillation builds a hybrid student from a **frozen teacher**, reaching capability at <1% of
from-scratch tokens — so several architecture layouts can be swept cheaply. The pieces exist
today; the **single end-to-end run driver and the cloud plumbing are still being built**
([#80](https://github.com/travisgalloway/monica/issues/80),
[#81](https://github.com/travisgalloway/monica/issues/81)):

| Piece | Module | Role |
|---|---|---|
| Conversion teacher | [`src/model/mlx_teacher.py`](../src/model/mlx_teacher.py) | frozen forward; cached top-k logits + attention projections |
| Student init | [`src/model/mlx_student_init.py`](../src/model/mlx_student_init.py) | Mamba-in-the-Llama / MOHAWK; maps teacher attention → student SSM |
| Staged loss | [`src/model/mlx_distill.py`](../src/model/mlx_distill.py) | mixing-match → hidden-align → logit-distill (KL on top-k + CE) |
| Manifest | [`src/train/distill_manifest.py`](../src/train/distill_manifest.py) | parse `config/manifests/*.yaml` → resolved `MambaConfig` |
| Sweep table | [`scripts/sweep.py`](../scripts/sweep.py) | per-trial param/memory + layout table (not a runner) |

Inspect a sweep over the three architecture variables (attention fraction, layer placement,
state size). Sibling manifests must share one frozen teacher signal:

```bash
.venv/bin/python scripts/sweep.py                    # all of config/manifests/
.venv/bin/python scripts/sweep.py \
    config/manifests/student-1b-attn-lo.yaml config/manifests/student-1b-attn-hi.yaml
```

A manifest names the frozen artifacts (corpus, teacher outputs, SFT/RL sets) and the swept
layout; the student layout is **downstream of every frozen artifact**, so changing it
invalidates nothing upstream. See
[`design/10-distillation.md`](design/10-distillation.md) for the full rationale.

### 3c. Post-training (M9) — SFT → DPO → RLVR

Once a base exists (a pretrained POC or a distilled student), layer capabilities in order. Each
driver initializes from the previous stage's weights and uses the shared training loop.

```bash
# Prep SFT data once (response-masked instruction JSONL), then instruction-tune:
.venv/bin/python -m src.data.sft_data --split train --out data/sft/train.jsonl
.venv/bin/python -m src.data.sft_data --split test  --out data/sft/val.jsonl
.venv/bin/python scripts/sft.py --config config/poc.yaml --data data/sft \
    --init runs/poc/weights.safetensors --out runs/sft

# Preference-align (DPO): SFT weights are BOTH the policy init and the frozen reference
.venv/bin/python -m src.data.dpo_data --split train_prefs --out data/dpo/train.jsonl
.venv/bin/python -m src.data.dpo_data --split test_prefs  --out data/dpo/val.jsonl
.venv/bin/python scripts/dpo.py --config config/poc.yaml --data data/dpo \
    --init runs/sft/weights.safetensors --out runs/dpo

# RLVR / GRPO with a verifiable reward (math exact-match; no sandbox)
.venv/bin/python scripts/rlvr.py --config config/poc.yaml \
    --init runs/sft/weights.safetensors --problems math.jsonl --out runs/rlvr --steps 200 --ckpt-every 50
```

`--problems` for `rlvr.py` is JSONL with `{"prompt": "...", "answer": "..."}` per line;
`--reward math` (default) uses final-number exact-match. On-policy DPO pairs (clean — generated
by our own model) can be produced with
[`scripts/gen_onpolicy_prefs.py`](../scripts/gen_onpolicy_prefs.py). SFT health = falling masked
`val_perplexity`; DPO health = a rising chosen-minus-rejected reward margin. See
[`design/11-post-training.md`](design/11-post-training.md).

---

## 4. Serve & chat

[`scripts/generate.py`](../scripts/generate.py) is the CLI front-end (completion + interactive
chat). It needs trained `--weights`; without them it random-inits and emits gibberish.

**Completion** — continue a prompt (streams token-by-token):

```bash
.venv/bin/python scripts/generate.py \
    --config config/poc.yaml --weights runs/poc/weights.safetensors \
    --prompt "Water is a chemical compound that " \
    --max-new-tokens 80 --temperature 0.7 --top-p 0.9
```

**Chat** — an instruction-template REPL (type a line, Enter to send, Ctrl-D to exit):

```bash
.venv/bin/python scripts/generate.py \
    --config config/poc.yaml --weights runs/poc/weights.safetensors --chat --temperature 0.7
```

Each chat line is wrapped in the **same instruction template the model was trained on**
([`src/data/instruct_format.py`](../src/data/instruct_format.py)), and generation stops at the
next `### Instruction:` marker or end-of-text.

| Flag | Default | Meaning |
|---|---|---|
| `--config` | `config/poc.yaml` | Model config (must match the weights) |
| `--weights` | — (random init) | Path to a `.safetensors` checkpoint |
| `--prompt "…"` / `--chat` | — | Completion prompt **or** chat REPL (one required) |
| `--max-new-tokens` | 100 | Max tokens to generate |
| `--temperature` | 0.8 | 0 = greedy/deterministic; higher = more random |
| `--top-k` / `--top-p` | none | Top-k / nucleus filtering (e.g. `--top-p 0.9`) |
| `--repetition-penalty` | 1.0 | >1 discourages repeats; `--no-repeat-ngram-size N` bans repeated n-grams |
| `--seed` | 0 | RNG seed for reproducible sampling |
| `--byte-fallback` | off | Offline byte tokenizer — **toy configs only** |

**Sampling tips:** deterministic → `--temperature 0`; most coherent → `--temperature 0.6
--top-p 0.9`; more varied → higher temp + `--top-k 40`.

> **Expectation:** at ~100M params output is roughly grammatical English with weak semantics,
> and chat replies are *template-shaped but not reliably correct*. The POC goal is the learning
> curve, not answer quality.

### Embedding it in an app

The CLI is thin glue over portable primitives in [`src/serve/`](../src/serve/) (no MLX/torch).
Use them directly for multi-session serving with snapshot/rewind:

```python
import numpy as np
from functools import partial
from src.model.blocks import load_config
from src.model.mlx_backend import MLXMambaModel
from src.data.tokenize import load_olmo_tokenizer
from src.serve.sessions import SessionStore
from src.serve.sampling import sample
from src.serve.generate import generate

cfg = load_config("config/poc.yaml")
model = MLXMambaModel(cfg); model.load("runs/poc/weights.safetensors")
tok = load_olmo_tokenizer()
store = SessionStore(model, max_concurrent=8)          # LRU-evicted, constant RAM/session

store.create("user1")
ids = tok.encode("The history of ", add_special_tokens=False)
out = generate(store, "user1", ids,
               sampler=partial(sample, temperature=0.8, top_p=0.9, rng=np.random.default_rng(0)),
               to_numpy=lambda a: np.array(a),
               max_new_tokens=80, eos_id=tok.eos_token_id)
print(tok.decode(out))
```

`SessionStore` holds each conversation's recurrent state with bounded memory; `RewindTree`
([`src/serve/rewind.py`](../src/serve/rewind.py)) snapshots/undoes turns and branches history —
the "experimental snapshotting" the SSM's small fixed-size state makes cheap. Both are portable,
so the same code runs unchanged on the CUDA backend.

---

## 5. Evaluate (Tier-2 benchmarks)

Beyond the Tier-1 held-out perplexity (logged during training),
[`scripts/eval_olmes.py`](../scripts/eval_olmes.py) runs the **lm-evaluation-harness** (needs
the `[eval]` extra). It supports loglikelihood (multiple-choice) tasks and generative tasks
(via `generate_until`).

```bash
# Multiple-choice (set HF_DATASETS_TRUST_REMOTE_CODE=1 for piqa's loader)
HF_DATASETS_TRUST_REMOTE_CODE=1 .venv/bin/python scripts/eval_olmes.py \
    --config config/poc.yaml --weights runs/poc/weights.safetensors \
    --tasks hellaswag,arc_easy,arc_challenge,piqa --limit 500 --output runs/poc/eval_mc.json

# Generative (exercises the generate_until path)
HF_DATASETS_TRUST_REMOTE_CODE=1 .venv/bin/python scripts/eval_olmes.py \
    --config config/poc.yaml --weights runs/poc/weights.safetensors \
    --tasks gsm8k --limit 50 --output runs/poc/eval_gsm8k.json
```

`--limit N` caps examples per task (each is a batch-1 forward, so full sets are slow); omit for
a full run. At ~100M params scores sit near chance — **judge by "the harness runs end-to-end
and returns numbers,"** not leaderboard position. Long-context and retrieval probes live in
[`scripts/long_context.py`](../scripts/long_context.py) /
[`scripts/retrieval_probe.py`](../scripts/retrieval_probe.py) /
[`scripts/probes.py`](../scripts/probes.py) — these become the headline metric vs a same-size
Transformer once a student layout validates.

---

## Troubleshooting

- **`mlx not found`** — you're not on Apple Silicon, or not using the venv interpreter. Use
  `.venv/bin/python` (the `[mlx]` extra installs only on Apple Silicon). On a CUDA host use
  `--backend cuda` with the `[cuda]`/`[cuda-fast]` extras.
- **Empty / immediate-EOS generation** — ensure prompts are encoded with
  `add_special_tokens=False` (the CLI does this); appending EOS makes the model stop at once.
- **Swapping / slow steps near the RAM ceiling** — drop to `--batch-size 8 --grad-accum 16` (or
  lower); the run is resume-safe, so stop and resume if it thrashes.
- **`transformers`/HF warnings** ("PyTorch not found", "clean_up_tokenization_spaces",
  unauthenticated Hub) — harmless; only the tokenizer is needed.
