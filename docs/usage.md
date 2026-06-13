# Usage guide

End-to-end commands for the Mamba POC: **install → data → train → serve/chat →
eval**. For *why* the project is built this way, see [`design/`](design/README.md).

Everything runs on **Apple Silicon via MLX**. Commands use the project venv
(`.venv/bin/python`); from a fresh shell you can substitute `python` once the venv is
activated. All paths are relative to the repo root.

---

## 1. Install

```bash
# Apple Silicon (full backend — required for training/serving/eval):
pip install -e ".[dev,data,mlx]"     # the mlx extra installs only on Apple Silicon

# Linux / CUDA host (portable layers only; no MLX runtime):
pip install -e ".[dev,data]"

# Optional, for the Tier-2 benchmark harness:
pip install -e ".[eval]"             # pulls lm-eval (and transitively torch)
```

The **OLMo tokenizer** (`allenai/OLMo-7B-hf`, vocab 50280) auto-downloads from the
HF Hub on first use and is cached under `~/.cache/huggingface` (needs network once).

Run the tests to confirm the install (on Linux the MLX-only tests skip, not fail):

```bash
.venv/bin/python -m pytest -q
```

---

## 2. Build a training corpus

The pipeline is **one document per line → tokenize (uint16) → pack → split**. Three
sources are available via `--source`; mix them by concatenating the text files.

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

This yields `data/split/{train.bin,val.bin}` (+ `.meta.json` sidecars). `--source
fineweb` (FineWeb-Edu web text) is also available; `--dummy` produces synthetic
offline text for pipeline smoke tests.

> **Tip — sizing:** ~257 OLMo tokens/Wikipedia article, so ~420k docs ≈ ~108M tokens.
> Pick `--max-docs` and `--max-tokens` for your token budget. At ~99 s/step the M1 Pro
> does ~114M tokens/day, so a ~100M-token run is ~1 day.

---

## 3. Train

[`scripts/train.py`](../scripts/train.py) wires config → model → data → loop, with
fp16 dynamic loss scaling, gradient accumulation, JSONL metrics, periodic
checkpoints, and held-out perplexity. Run it under `caffeinate`/`nohup` for a long
unattended run.

```bash
caffeinate -i nohup .venv/bin/python scripts/train.py \
    --config config/poc.yaml --data data/split --out runs/poc \
    --total-tokens 100000000 --batch-size 8 --grad-accum 16 \
    --base-lr 3e-4 --warmup-steps 40 --grad-clip 1.0 \
    --log-every 10 --eval-every 50 --ckpt-every 100 --seed 0 \
    > runs/poc/train.log 2>&1 &
```

- **Effective batch = `batch-size × grad-accum × seq_len` tokens/step** (here
  8 × 16 × 1024 = 131,072). `batch 8 / accum 16` keeps peak RAM ~12–17 GB; `batch
  32 / accum 4` is faster (~25 GB peak) — use the smaller batch if you're also using
  the machine, the larger if it's dedicated.
- **Resume** after any interruption (auto-detects `runs/poc/resume`): re-run the same
  command, optionally adding `--resume runs/poc/resume`.
- **Checkpoints overwrite a single path** (`runs/poc/weights.safetensors` +
  `runs/poc/resume/`). To keep a good checkpoint, `cp` it aside at a val-perplexity low.

**Monitor:**

```bash
tail -f runs/poc/metrics.jsonl     # {step, lr, loss, grad_norm, val_perplexity, tokens_per_sec, ...}
```

**POC success = a smoothly decreasing `val_perplexity`** with stable `grad_norm`.
(A reference ~100M-param run on ~100M tokens reached val-perplexity ~77.)

The smaller `config/toy.yaml` (vocab 256, fp32) is for the exact-resume smoke gate:

```bash
.venv/bin/python scripts/smoke_test.py --data data/split    # use a byte-fallback split
```

---

## 4. Serve & chat

[`scripts/generate.py`](../scripts/generate.py) is the CLI front-end (completion +
interactive chat). It needs trained `--weights`; without them it random-inits and
emits gibberish.

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
([`src/data/instruct_format.py`](../src/data/instruct_format.py)), and generation stops
at the next `### Instruction:` marker or end-of-text.

| Flag | Default | Meaning |
|---|---|---|
| `--config` | `config/poc.yaml` | Model config (must match the weights) |
| `--weights` | — (random init) | Path to a `.safetensors` checkpoint |
| `--prompt "…"` / `--chat` | — | Completion prompt **or** chat REPL (one required) |
| `--max-new-tokens` | 100 | Max tokens to generate |
| `--temperature` | 0.8 | 0 = greedy/deterministic; higher = more random |
| `--top-k` / `--top-p` | none | Top-k / nucleus filtering (e.g. `--top-p 0.9`) |
| `--seed` | 0 | RNG seed for reproducible sampling |
| `--byte-fallback` | off | Offline byte tokenizer — **toy configs only** |

**Sampling tips:** deterministic → `--temperature 0`; most coherent → `--temperature
0.6 --top-p 0.9`; more varied → higher temp + `--top-k 40`.

> **Expectation:** at ~100M params output is roughly grammatical English with weak
> semantics, and chat replies are *template-shaped but not reliably correct*. The POC
> goal is the learning curve, not answer quality.

### Embedding it in an app

The CLI is thin glue over portable primitives in [`src/serve/`](../src/serve/) (no
MLX/torch). Use them directly for multi-session serving:

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

`SessionStore` holds each conversation's recurrent state with bounded memory;
`RewindTree` ([`src/serve/rewind.py`](../src/serve/rewind.py)) snapshots/undoes turns.
Both are portable, so the same code runs unchanged on a future CUDA backend.

---

## 5. Evaluate (Tier-2 benchmarks)

Beyond the Tier-1 held-out perplexity (logged during training),
[`scripts/eval_olmes.py`](../scripts/eval_olmes.py) runs the **lm-evaluation-harness**
(needs the `[eval]` extra). It supports loglikelihood (multiple-choice) tasks and
generative tasks (via `generate_until`).

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

`--limit N` caps examples per task (each is a batch-1 forward, so full sets are slow);
omit for a full run. At ~100M params scores sit near chance — **judge by "the harness
runs end-to-end and returns numbers,"** not leaderboard position.

---

## Troubleshooting

- **`mlx not found`** — you're not on Apple Silicon, or not using the venv interpreter.
  Use `.venv/bin/python` (the `[mlx]` extra installs only on Apple Silicon).
- **Empty / immediate-EOS generation** — ensure prompts are encoded with
  `add_special_tokens=False` (the CLI does this); appending EOS makes the model stop at once.
- **Swapping / slow steps near the RAM ceiling** — drop to `--batch-size 8 --grad-accum 16`
  (or lower); the run is resume-safe, so stop and resume if it thrashes.
- **`transformers`/HF warnings** ("PyTorch not found", "clean_up_tokenization_spaces",
  unauthenticated Hub) — harmless; only the tokenizer is needed.
