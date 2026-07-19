# RunPod POC run — session handoff (`poc-qwen`, ~205M)

> **⛔ Reserve / historical (M10 distillation, superseded 2026-07-19).** This program is no longer
> active — see [`../design/13-code-model-moe.md`](../design/13-code-model-moe.md) and
> [issue #198](https://github.com/travisgalloway/monica/issues/198) for the live M12 code-model
> arc. The ~205M `poc-qwen` run this documents is **complete** (val-ppl 75.7); kept as a RunPod
> provisioning/run record and R2 asset inventory.

Self-contained runbook to train the ~205M Qwen2.5 POC on a RunPod GPU pod, eval it, and sync
checkpoints to R2. Written so a fresh session (or a clean pod clone) can execute it with no prior
context. Related: [`infrastructure.md`](../infrastructure.md) (general pod flow),
[`config/poc-qwen.yaml`](../../config/poc-qwen.yaml) (the config + its header runbook).

## Current state (already done)

- **Corpus on R2** — `s3://monica-training/reserve-pretrain/`:
  - `tokenized/v1-qwen25-1k/` — **1.91B Qwen2.5 tokens**, uint32, seq_len 1024, 15 shards
    (`part-000NN.{bin,bounds}` + `manifest.json`), 9.56 GB.
  - `cleaned/` — durable tokenizer-agnostic JSONL (3.54 GB), for cheap OLMo re-tokenization later.
- **Config** — `config/poc-qwen.yaml`: ~205M (Qwen2.5 vocab 151,646, fp16). `poc.yaml` (OLMo
  ~127M) is the reserve variant. The ~205M is embedding-dominated by design (tokenizer alignment
  with the 1B distillation student) — accepted. (The distillation student has since moved to the
  Qwen3 vocab ~151,669, #65; Qwen3 is token-aligned with Qwen2.5, so this from-scratch POC run
  stays on its already-built Qwen2.5 corpus.)
- **Code** — in **PR #134** (branch `feature/80-r2-datatrove-corpus-pipeline`): `r2_sync`,
  datatrove pipeline, `split.py --shards`, `poc-qwen.yaml`. CUDA backend is A40-verified.
- **R2 creds** work (bucket-scoped read/write on `monica-training`). **RunPod auth not yet set up.**

## Decisions locked

- Model **~205M** Qwen2.5 (embedding-heavy; deliberate). Tokens **~1.9B** (≈Chinchilla for the
  ~100M-class layers). Precision **fp16** + dynamic loss scaling (T4/L4 OK; bf16 needs Ampere+).

## Step 0 — get the code onto the pod

The pod needs the #134 code. Either **merge PR #134 to `main`** first (then clone `main`), or on
the pod checkout the branch: `git checkout feature/80-r2-datatrove-corpus-pipeline`.

## Step 1 — provision the GPU pod

- RunPod **`-devel`** image, **py3.11**, e.g. `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`.
- fp16 runs on **T4 16GB / L4 24GB**; only need **Ampere+** if you switch to bf16.
- Attach a **network volume (~50 GB)** for the corpus + checkpoints (`/vol` below).
- Keep the pod region network-close to R2.

## Step 2 — install + secrets

```bash
pip install -e ".[dev,data,cuda]"          # torch CUDA backend
pip install "s3fs==2026.2.0"               # for R2; pin to match fsspec (datasets caps fsspec<=2026.2.0)
# secrets in the pod env (HF_TOKEN is NOT needed — training reads packed shards, no tokenizer):
export AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...
export AWS_ENDPOINT_URL_S3=https://<account-id>.r2.cloudflarestorage.com
export R2_BUCKET=monica-training
```

## Step 3 — pull the corpus + make a train/val split

```bash
python -m src.data.r2_sync down \
    s3://monica-training/reserve-pretrain/tokenized/v1-qwen25-1k /vol/corpus
python -m src.data.split --shards /vol/corpus --out /vol/split --val-tokens 10000000
#   -> /vol/split/{train.bin,val.bin} (+ .meta.json), ~1.9B train / 10M val, uint32. No re-tokenize.
```

## Step 4 — prove the backend BEFORE the long (paid) run

```bash
# tiny offline toy split for the CUDA smoke gate (uses config/toy.yaml, fp32, vocab 256):
python -m src.data.download --dummy --out data/raw --max-docs 2000
python -m src.data.tokenize  --in data/raw/dummy.txt --out data/ids.npy --byte-fallback
python -m src.data.pack      --in data/ids.npy --out data/packed.bin
python -m src.data.split     --packed data/packed.bin --out data/toy-split --val-tokens 2000
python scripts/smoke_test.py --backend cuda --data data/toy-split    # bit-exact resume + eval

# throughput + PEAK GPU memory on the real config (catch OOM/throughput before paying):
python scripts/bench_cuda_train_step.py --config config/poc-qwen.yaml --batch 32 --grad-accum 4
```

The run is ~**14,500 steps** (1.9e9 tokens ÷ (32×4×1024 = 131,072 tok/step)). Multiply by the
bench's s/step to estimate wall-clock + cost and pick the GPU.

## Step 5 — train (checkpoint to R2 on a cadence — pods are ephemeral)

```bash
python scripts/train.py --backend cuda --config config/poc-qwen.yaml --data /vol/split \
    --out /vol/runs/poc-qwen --total-tokens 1900000000 --batch-size 32 --grad-accum 4 \
    --eval-every 200 --ckpt-every 500
# sync checkpoints to R2 periodically AND at the end (pod is not durable):
python -m src.data.r2_sync up /vol/runs/poc-qwen s3://monica-training/ckpt/poc-qwen
# resume after an interruption: re-run with  --resume /vol/runs/poc-qwen/resume
```

**Success = a smoothly decreasing `val_perplexity`** in `/vol/runs/poc-qwen/metrics.jsonl` with a
stable `grad_norm`. (Benchmark scores are NOT the bar — see below.)

## Step 6 — post-eval

```bash
python scripts/eval_olmes.py ...        # OLMES / lm-eval (loglikelihood + generative)
python scripts/retrieval_probe.py ...   # retrieval probes (#79)
python scripts/long_context.py ...      # long-context behavior
```

At ~205M, benchmark scores sit near chance — judge by the **val-ppl curve** + the harness running
end-to-end (#13), and the **local-hardware headline metric** vs a same-size transformer (#104).

## Gotchas

- **HF_TOKEN not needed for training** (no tokenizer at train time); only for re-tokenizing the
  `cleaned/` corpus (e.g. to OLMo for a true-127M `poc.yaml` run).
- **fp16** + dynamic loss scaling is on; switch the config to bf16 only on an Ampere+ card.
- **s3fs pin**: must match `fsspec` (`datasets` caps `fsspec<=2026.2.0`) or `datasets` breaks.
- **Bench first** — it sizes the GPU and surfaces OOM before the long run.
- **Checkpoint to R2 frequently** — RunPod pods are not durable; the volume can be lost.

## Open items / context for the new session

- **Merge PR #134** (or checkout its branch on the pod) — the run depends on its code.
- **RunPod auth / provisioning** is not yet done (`RUNPOD_API_KEY` is still a placeholder in `.env`).
- This POC validates the **architecture + the Qwen2.5 data path**; the real program is **M10
  distillation** (#65) — teacher-logit precompute (#94) is the next GPU-heavy step after this.
  The full-scale execution runbook for that program is [`path-b-run.md`](path-b-run.md).
- Background memories that load automatically: `poc-corpus-r2`, `datatrove-venv`, `s3fs-fsspec-pin`,
  `poc-step-time-baseline` (the ~99 s/step figure is the MLX baseline — CUDA differs; trust the bench).
