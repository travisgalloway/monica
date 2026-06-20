# Cloud infrastructure — running the pipeline on object storage + rented GPUs

This is the operational runbook for taking the data and training pipeline off a laptop and onto
**durable object storage + on-demand GPU hosts**. It is written **generically first** (any
S3-compatible store + any CUDA host), then with the **specifics we use: Cloudflare R2 + RunPod**.

For the *why* behind the corpus design, see
[`design/08-corpus-pipeline.md`](design/08-corpus-pipeline.md); for the distillation strategy
these artifacts feed, see [`design/10-distillation.md`](design/10-distillation.md).

> **Status.** The storage **layout** is implemented and is the single source of truth
> ([`src/data/storage.py`](../src/data/storage.py)); the same path strings are valid local
> directories *and* object-store prefixes. The actual **R2/RunPod readers/writers and the cloud
> run harness are in progress** ([#80](https://github.com/travisgalloway/monica/issues/80),
> [#81](https://github.com/travisgalloway/monica/issues/81)). What follows is the **intended**
> flow — build and unit-test it locally first, then rent a pod for the few stages that need one.

---

## Principle: cloud is on-demand

Almost the entire stack is **Mac-doable today** (MLX, or CUDA-on-torch-CPU for conformance):
the data pipeline on a slice, the manifest/sizing tooling, the teacher loader and student init
at toy scale, the distillation loss + train step, and the SFT/DPO/GRPO machinery. **Build and
unit-test all of it locally before renting anything.**

Rent a pod only for the handful of stages that genuinely need one:

| Paid stage | Why it needs a pod | Issue |
|---|---|---|
| Teacher top-k logit precompute (corpus scale) | the dominant compute cost; runs the 7B teacher over the whole corpus | [#94](https://github.com/travisgalloway/monica/issues/94) |
| R2 + storage plumbing | wiring the `s3fs` readers/writers + secrets | [#80](https://github.com/travisgalloway/monica/issues/80) |
| Cloud distill smoke run | full flow dress-rehearsal on a cheap GPU | [#81](https://github.com/travisgalloway/monica/issues/81) |
| ≥1B distill / pretrain runs | throughput needs the card; relies on the `state-spaces/mamba` CUDA kernels | [#75](https://github.com/travisgalloway/monica/issues/75) |

**Training** runs on CUDA (where the fused Mamba kernels live); the **inference** target stays
Apple Silicon / MLX. No pod stands idle — bring it up, run the stage, sync results to durable
storage, tear it down.

---

## Generic overview (any S3-compatible store + CUDA host)

The pipeline is **storage-URI agnostic**: every artifact path is produced by
[`src/data/storage.py`](../src/data/storage.py), which returns plain path strings that work both
as local directories and as object-store prefixes (the data drivers go through `fsspec`/`s3fs`,
so `file://` today swaps to `s3://` later with no path changes). The shape is the same on any
provider:

1. **Durable object store** holds every artifact: the cleaned corpus, the tokenized training
   shards, the precomputed teacher outputs, the SFT/RL sets, and **checkpoints** (compute hosts
   are ephemeral — checkpoints must be synced off them).
2. **A CPU host** runs the heavy data stages (ingest / clean / dedup / tokenize), reading and
   writing the object store directly.
3. **A GPU host** runs training only: it **pulls** the relevant tokenized subset + teacher
   outputs to a fast local/volume disk, trains, and **pushes checkpoints back** to the store.
4. **Keep compute network-close to storage** so the train-time pull is fast and egress is cheap.

### The three-class storage layout

One layout keeps the **student architecture downstream of every frozen artifact**, so a layout
sweep invalidates nothing upstream (the whole point of the distillation strategy):

```
<store>://<bucket>/
  poc-distill/      corpus/{cleaned, tokenized/<tok>-<k>}/      # frozen distillation corpus (#92)
                    teacher-outputs/{topk-logits, hidden-states}/   # precomputed teacher signal (#94)
                    manifests/
  shared/           sft/{cleaned/<kind>, tokenized/<tok>-<k>}/  # instruct/reasoning/tool SFT (#95/#96/#102)
                    rl/{math-verifiable, code-verifiable}/      # verifiable RL sets (#103)
                    eval/
  reserve-pretrain/ cleaned/  tokenized/<ver>-<tok>-<k>/  manifests/   # from-scratch corpus (#70/#71)
  ckpt/             <run>/...                                   # checkpoints synced off the GPU host
```

Two invariants the layout enforces (encoded in `storage.py`):

- **Cleaned text and RL problems are tokenizer-agnostic and durable** — re-tokenize cheaply when
  the tokenizer or `seq_len` changes; never re-clean.
- **Every tokenized folder name-pins `<tokenizer>-<seqlen_k>`** (e.g. `qwen25-8k`), so multiple
  tokenized views coexist without collision.

**What invalidates what:** changing the **teacher** invalidates `teacher-outputs/`; changing the
**tokenizer** invalidates the tokenized views (ids shift); changing the **student layout**
invalidates **nothing** — fix the teacher and tokenizer first, then sweep students freely.

### Cost shape (provider-independent)

- Prefer **few large shards** (high-hundreds-of-MB to low-GB) over many small files — per-request
  ("Class A") operations dominate at small sizes.
- Keep secrets (store key/secret, HF token) in the host's secret store / env — **never committed**.
- Treat the GPU host as **non-durable**: checkpoint to the object store on a cadence, not just at
  the end.

---

## Cloudflare R2 specifics

R2 is our durable store — **S3-compatible with no egress fees**, which suits the repeated
train-time pulls. Concretely:

- **Access:** the `datatrove`/`fsspec` S3 reader/writer address R2 through the **`s3://`**
  scheme (R2 exposes an S3 API; point the S3 client at the R2 endpoint). The three artifact
  prefixes — `poc-distill/` · `shared/` · `reserve-pretrain/` — are exactly the strings from
  `src/data/storage.py`. `ckpt/` is a separate checkpoint-sync prefix (a run-output convention,
  not a `storage.py` constant).
- **Secrets:** R2 key/secret + HF token live in the pod's secrets/env, never in the repo.
- **Sizing:** target ~1–2 TB working set, growing with the reserve corpus. Few large shards
  (R2 Class A ops cost per million).
- **Checkpoints:** synced to `s3://<bucket>/ckpt/<run>/` (the R2 bucket) from the GPU host on a
  cadence — the durable copy, since the pod is ephemeral.
- **Install:** the data extras pull `fsspec`/`pyarrow` — `pip install -e ".[data]"` — but the
  S3 filesystem backend is separate: also `pip install s3fs` so `s3://` URLs resolve. The cloud
  corpus engine adds `pip install -e ".[datatrove]"`.

---

## RunPod specifics

RunPod provides the on-demand compute. Two roles, kept separate:

- **CPU pod** — the data stages (ingest / clean / dedup / tokenize). Install `datatrove` +
  `s3fs` + tokenizer deps; its S3 reader/writer point at R2.
- **GPU pod** — training only. Pull the tokenized subset + teacher outputs from R2 to a network
  volume, train, checkpoint back to R2.

**Region:** RunPod network volumes are region-locked — keep the pod region network-close to R2
so the pull is fast and free.

**GPU pod spec.** The card choice is driven by **precision**, not a blanket rule: the **1B
training** configs (`config/student-1b.yaml`, `config/1b.yaml`) are **bf16**, which needs an
**Ampere-or-newer** card (a T4/Turing has no bf16). The cheaper **smoke gate** (`config/toy.yaml`,
fp32) and **train-step bench** (`config/poc.yaml`, fp16) below run fine on a **T4/L4** — so a
T4/L4 is enough to dry-run the flow, and you only need Ampere+ for the actual bf16 run. Use a
RunPod **`-devel`** image so the build sees the preinstalled CUDA torch (e.g.
`runpod/pytorch:2.4.0-...-devel-ubuntu22.04` or the `2.8.0-...-cudnn-devel` image). Then:

```bash
# 1. Backend install (the [cuda] extra pulls torch; mlx is Mac-only).
pip install -e ".[dev,data,cuda]"
#    Optional fused kernels (#40) — mamba-ssm Triton SSD scan + causal-conv1d:
pip install -e ".[dev,data,cuda-fast]"

# 2. CUDA smoke gate — prove the torch backend resumes bit-exactly through the
#    double-buffered CheckpointStore. Build a tiny toy split on the pod, then:
python scripts/smoke_test.py --backend cuda --data <toy-split>     # use config/toy.yaml (dense; MoE is MLX-only)

# 3. Train-step bench — s/step, tokens/s, and PEAK GPU MEMORY for the real path,
#    BEFORE paying for big cards:
python scripts/bench_cuda_train_step.py --config config/poc.yaml --batch 32 --grad-accum 4

# 4. Pull the subset from R2 to the network volume, then train (checkpoints → R2):
python scripts/train.py --backend cuda --config config/<student-or-poc>.yaml \
    --data <local-volume-split> --out <run-dir> --total-tokens <N> --batch-size 32 --grad-accum 4
```

Bring the pod up **in that order** so a config or throughput problem surfaces *before* the long
run. The CUDA backend is already done and A40-verified (the full suite is green on a rented
A40); the fused kernels auto-detect at runtime and degrade gracefully when absent.

---

## End-to-end intended flow (once #80/#81 land)

1. **Local (Mac):** build + unit-test the data pipeline on a slice, the teacher loader, student
   init, distillation loss, and the manifest/sweep — all at toy scale.
2. **CPU pod:** build the frozen distillation corpus and SFT/RL sets to `poc-distill/` and
   `shared/` in R2.
3. **GPU pod (precompute):** run the teacher over the corpus → `poc-distill/teacher-outputs/`.
4. **GPU pod (sweep):** train the candidate student layouts against the frozen signal; checkpoint
   to `ckpt/`; pick the layout that wins on math/code **and** the local-hardware target.
5. **GPU pod (post-train):** instruct SFT → reasoning SFT → optional tool-use → GRPO on the
   chosen student (re-targets the M9 machinery).
6. **Local (Mac / MLX):** serve the winner and measure the headline metric (context length +
   tokens/sec vs a same-size Transformer).
