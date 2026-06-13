# monica — Mamba POC (SSM on Mac Experiment)

A proof-of-concept Mamba (selective state-space) language model, developed and
validated on **Apple Silicon with MLX**, architected behind **one hardware seam**
so a successful POC migrates to **CUDA** for a larger run with minimal rewrite.

**Usage:** [`docs/usage.md`](docs/usage.md) has end-to-end commands —
install → data → train → serve/chat → eval.

## The seam (most important rule)

All hardware-specific code lives behind `src/model/interface.py`
(`ModelInterface`). Everything above it — `data/`, `train/`, `serve/`, `eval/`,
`conformance/` — is portable Python that **never imports MLX or CUDA**. Only the
backend modules — `src/model/mlx_backend.py`, `src/model/mlx_train_step.py`, and
`src/model/cuda_backend.py` — touch a hardware library.
`tests/test_import_guard.py` enforces this.

`ModelInterface`: `forward` (parallel training path) · `step` (recurrence inference
path) · `init_state` · `get_state`/`set_state` · `save`/`load` · `config`.

## Layout

```
config/{toy,poc}.yaml      model dims + run params (single source of truth)
src/model/                 interface (seam) · blocks (config) · mlx/cuda backends
src/data/                  download (fineweb/wikipedia/instruct) · tokenize · pack(uint16) · split · loader
src/train/                 loop · schedule (warmup+cosine) · checkpoint
src/eval/                  val_loss (Tier-1) · olmes_adapter (Tier-2 lm-eval)
src/conformance/           forward_step_parity · backend_parity (fp32 guards)
src/serve/                 sessions · rewind · sampling · generate (generation core)
scripts/train.py           training driver         scripts/generate.py   serve + chat CLI
scripts/smoke_test.py      milestone-4 gate        scripts/eval_olmes.py  lm-eval harness
tests/                     unit tests
docs/usage.md              usage guide             docs/design/           design + rationale
```

## Status — M1–M7 implemented; M8 (CUDA) deferred

The seam, configs, data pipeline, MLX model/backend, training loop, smoke gate,
serving/chat, and the Tier-2 eval harness are **implemented and verified on Apple
Silicon** — the full suite (~97 tests) runs on a Mac (incl. the real MLX paths; on
Linux the MLX-only tests skip and the rest pass), and the M4 smoke gate passes end to
end (exact save/kill/resume + held-out perplexity eval). A reference **~100M-param run
on ~100M tokens** reached held-out val-perplexity ~77. Progress is tracked in
[issue #2](https://github.com/travisgalloway/monica/issues/2).

| Milestone | State |
|---|---|
| 1 Seam + toy MLX model | done; MLX backend + forward/step parity verified |
| 2 Data pipeline | done; FineWeb-Edu / Wikipedia / instruction sources, unit-tested |
| 3 Minimal training loop | done; `train_step` + loop exercised by the smoke gate |
| 4 Smoke test (gate) | passing — resume exact, eval runs |
| 5 POC scale run | infra done; short ~100M-token run completed (val-ppl ~77); full 2–5B deferred |
| 6 OLMES / lm-eval | working — loglikelihood + generative (`generate_until`) tasks |
| 7 Serving + chat | working — CLI generate/chat over `SessionStore` + `RewindTree` |
| 8 CUDA backend | deferred |

**Locked decisions:** SSM is **Mamba-2 / SSD** (scalar A per head; chunked-matmul
scan) + gradient checkpointing — migrated from Mamba-1 for training throughput/memory.
poc = d_model 768 / 24 layers / d_state 16 / head_dim 64 (24 heads) / seq 1024 /
~3B tokens (tied embedding mandatory). toy = d_model 64 / 2 layers / seq 128 /
fp32. Precision for poc (fp16 vs bf16) **confirmed on MLX in M1** — not assumed.
Conformance compares in **fp32** (~1e-4 rel). OLMES + serving/rewind deferred;
**POC success = a smoothly decreasing held-out val-perplexity curve**.

## Quickstart

```bash
# Apple Silicon (full backend):
pip install -e ".[dev,data,mlx]"  # the mlx extra installs only on Apple Silicon

# Linux / CUDA host (portable layers only — omit the mlx extra):
pip install -e ".[dev,data]"

pytest                            # Mac: full suite (~97 tests, MLX paths included).
                                  # Linux: MLX-only tests (pytest.importorskip(
                                  # "mlx.core")) skip, not fail — portable suite runs.

# Data pipeline offline smoke (no network/tokenizer):
python -m src.data.download --dummy --out data/raw --max-docs 2000
python -m src.data.tokenize --in data/raw/dummy.txt --out data/ids.npy --byte-fallback
python -m src.data.pack --in data/ids.npy --out data/packed.bin
python -m src.data.split --packed data/packed.bin --out data/split --val-tokens 2000
```

For a **real corpus** (Wikipedia + instruction data) and the full
train → serve → eval flow, see [`docs/usage.md`](docs/usage.md).

Run the smoke gate (the M4 gate) to confirm the MLX backend, training loop, and
exact-resume all pass before scaling to `config/poc.yaml`:

```bash
python scripts/smoke_test.py --data data/split
```

### Scale run (M5)

[`scripts/train.py`](scripts/train.py) is the real run driver: config → model → data →
loop, with fp16 dynamic loss scaling, gradient accumulation, JSONL metrics, periodic
checkpoints, and held-out val-perplexity. The recommended `config/poc.yaml` invocation
(and the one-time ~3B-token data prep) is recorded as comments in that file. Sketch:

```bash
# ... data prep into data/split (see config/poc.yaml comments) ...
python scripts/train.py --config config/poc.yaml --data data/split --out runs/poc \
    --total-tokens 3000000000 --batch-size 32 --grad-accum 4
# resume after an interruption (auto-detects runs/poc/resume if --resume omitted):
python scripts/train.py --config config/poc.yaml --data data/split --out runs/poc \
    --total-tokens 3000000000 --batch-size 32 --grad-accum 4 --resume runs/poc/resume
```

**POC success = a smoothly decreasing `val_perplexity` in `runs/poc/metrics.jsonl`**
with a stable `grad_norm` — not a benchmark score.

### Serve & chat (M7)

[`scripts/generate.py`](scripts/generate.py) is the CLI front-end over the trained
weights — completion and an instruction-template chat REPL:

```bash
# completion
python scripts/generate.py --config config/poc.yaml --weights runs/poc/weights.safetensors \
    --prompt "Water is a chemical compound that " --max-new-tokens 80 --temperature 0.7 --top-p 0.9
# chat REPL (Ctrl-D to exit)
python scripts/generate.py --config config/poc.yaml --weights runs/poc/weights.safetensors --chat
```

At ~100M params expect roughly grammatical English, weak semantics — the goal is the
learning curve, not answer quality. See [`docs/usage.md`](docs/usage.md) for flags,
sampling tips, and embedding the generation core in an app.

### Eval (M6, Tier-2)

[`scripts/eval_olmes.py`](scripts/eval_olmes.py) runs the lm-evaluation-harness
(`pip install -e ".[eval]"`) over loglikelihood and generative tasks:

```bash
HF_DATASETS_TRUST_REMOTE_CODE=1 python scripts/eval_olmes.py \
    --config config/poc.yaml --weights runs/poc/weights.safetensors \
    --tasks hellaswag,arc_easy,arc_challenge,piqa --limit 500
```

Scores sit near chance at this scale — judge by "the harness runs end-to-end."
