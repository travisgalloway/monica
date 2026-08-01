# Local development on Apple Silicon (MLX)

MLX is the **dev/validation backend**: it exists so you can prove a change correct and train
test-scale models *locally* before paying for a CUDA cloud run. Scale training itself runs on
CUDA/RunPod (see [`docs/infrastructure.md`](infrastructure.md)) — so this page is about
**developer velocity**, not throughput. (The MLX *train-step throughput* optimizations were
deliberately retired; see issue #30's decision record before reopening that.)

Three things live here:

1. [Validate every stage locally in one command](#1-validate-every-stage-locally)
2. [Train test models locally (`small.yaml`, `poc-small.yaml`)](#2-train-test-models-locally)
3. [Generate teacher signal locally (Qwen3 via MLX, or LM Studio) — reserve, M10 distillation](#3-generate-teacher-signal-locally)

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

(The former stages 4–5 exercised the M10 distillation/teacher machinery, removed with #189.)

Knobs (env vars): `PYTHON` (default `.venv/bin/python`), `WORK` (default `runs/local-validate`),
`STEPS` (default 20), `KEEP=1` to keep the work dir. Use this as the pre-push gate for any change
to the loop, the SSD scan, mixed precision, checkpointing, or eval.

> The smoke gate must run on a **freshly built byte split**, not the real `data/split` (which is
> the OLMo-vocab corpus) — `local_validate.sh` builds one for you.

### Debug a forward pass on MPS (no pod)

The torch/CUDA backend's SSD scan is pure PyTorch by default; the fused `mamba-ssm` scan and
`causal-conv1d` only engage when `x.device.type == "cuda"`, and `torch.compile`'s AUTO mode stays
eager off CUDA. So `CUDAMambaModel(cfg, device="mps")` runs the same math on Apple Silicon GPU as
the CPU/no-`mamba-ssm` path — use it to catch a forward-pass typo or shape/logic bug locally
instead of spinning up a pod (#218).

```bash
# forward/step parity on Apple Silicon GPU — the pure-PyTorch path, no CUDA:
.venv/bin/python -m pytest tests/test_cuda_parity.py -q -rs
```

```python
from src.model.blocks import load_config
from src.model.cuda_backend import CUDAMambaModel
from src.conformance.forward_step_parity import check_forward_step_parity

cfg = load_config("config/toy.yaml")
model = CUDAMambaModel(cfg, device="mps")
model.eval()
# ... build a (B, L) int token batch, then:
check_forward_step_parity(model, tokens, to_numpy=lambda a: a.detach().cpu().numpy())
```

Measured (fp32, `rtol=1e-4, atol=1e-5`): `config/toy.yaml` → `max_abs_diff ≈ 2.3e-5`, `ok=True`;
`config/toy-hybrid.yaml` (adds the attention block) → `≈ 3.1e-5`, `ok=True`.

**Caveat:** this validates the *pure-PyTorch* path only. It does **not** validate the fused CUDA
kernel — that's `tests/test_cuda_parity.py::test_fused_scan_matches_vanilla`, which needs a real
GPU with `mamba-ssm` installed and skips everywhere else. MPS is for catching typos and
shape/logic bugs, not for signing off kernel numerics or performance — for that, see
[`docs/infrastructure.md`](infrastructure.md) for the pod path.

**CI mirrors this recipe** (`.github/workflows/ci.yml`, #249): a Linux `portable` job runs
`pytest -q -rs` with no mlx/torch installed (the unambiguous seam-guard environment); a Linux
`smoke-linux` job installs CPU-only torch and runs the same fresh-toy-split →
`smoke_test.py --backend cuda` steps as above, under `$RUNNER_TEMP` instead of `data/`; and a
macOS `full-macos` job runs the full `pytest -q -rs` plus `smoke_test.py --backend mlx`. The
smoke-gate data build itself is fully offline (dummy corpus + byte fallback — no network, no HF
token, no corpus or model weights ever fetched); CI as a whole does need network for dependency
installs, and a few tests opportunistically fetch an HF tokenizer. Those degrade gracefully —
`tests/test_chat_template.py` wraps the load in `except Exception: pytest.skip("OLMo tokenizer
unavailable")`, so an HF outage produces a skip, not a red build.

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

## 3. Generate teacher signal locally (reserve — removed with #189)

> **Reserve, code removed.** The M10 distillation program (issue #65, **dropped 2026-07-19**) had a
> local teacher-signal precompute (`scripts/precompute_teacher.py` — a real Qwen3 MLX teacher, or an
> LM-Studio/OpenAI-compatible endpoint). That code was **removed from the tree** with #189; the
> design record is preserved at [`reserve/10-distillation.md`](reserve/10-distillation.md) and the
> code is recoverable from git history. See [`design/13-code-model-moe.md`](design/13-code-model-moe.md)
> for the live M12 program.

---

See also: [`docs/usage.md`](usage.md) (full flow),
[`docs/design/13-code-model-moe.md`](design/13-code-model-moe.md) (the live M12 plan),
[`docs/reserve/10-distillation.md`](reserve/10-distillation.md) (reserve distillation design),
[`docs/infrastructure.md`](infrastructure.md) (cloud R2 + RunPod).
