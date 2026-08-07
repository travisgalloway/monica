# Cloud infrastructure — running the pipeline on object storage + rented GPUs

This is the operational runbook for taking the data and training pipeline off a laptop and onto
**durable object storage + on-demand GPU hosts**. It is written **generically first** (any
S3-compatible store + any CUDA host), then with the **specifics we use: Cloudflare R2 + RunPod**.
The generic topology, R2 specifics, and RunPod pod-role split below are reusable foundation for
any cloud run, including the live M12 program; the **distillation-specific stages** (frozen
teacher-signal precompute, the three-class `poc-distill/` layout, the Path B run) are **reserve**,
inherited from the dropped M10 program.

For the *why* behind the corpus design, see
[`design/08-corpus-pipeline.md`](design/08-corpus-pipeline.md); for the (reserve) distillation
strategy these artifacts fed, see [`reserve/10-distillation.md`](reserve/10-distillation.md); for
the live M12 plan, see [`design/13-code-model-moe.md`](design/13-code-model-moe.md).

> **Status.** The storage **layout** is implemented and is the single source of truth
> ([`src/data/storage.py`](../src/data/storage.py)); the same path strings are valid local
> directories *and* object-store prefixes. The **R2/RunPod readers/writers and the M10 cloud run
> harness** described below were **never finished** — they were mid-build
> ([#80](https://github.com/travisgalloway/monica/issues/80),
> [#81](https://github.com/travisgalloway/monica/issues/81)) when M10 was dropped 2026-07-19 —
> treat those pieces as reserve, not a live build target. What follows is the **M10-era intended**
> flow, kept for its generic R2/RunPod topology; build and unit-test locally first, then rent a
> pod for the few stages that need one.

---

## Principle: cloud is on-demand

Almost the entire stack is **Mac-doable today** (MLX, or CUDA-on-torch-CPU for conformance):
the data pipeline on a slice, the manifest/sizing tooling, the SFT/DPO/GRPO machinery, and — for
the reserve M10 path — the teacher loader, student init at toy scale, and the distillation loss +
train step. **Build and unit-test all of it locally before renting anything.** This principle
carries over to M12; the paid-stage table below is the M10-era example (reserve).

Rent a pod only for the handful of stages that genuinely need one (M10-era example, reserve):

| Paid stage | Why it needs a pod | Issue |
|---|---|---|
| Teacher top-k logit precompute (corpus scale) | the dominant compute cost; runs the Qwen3-4B teacher over the whole corpus | [#94](https://github.com/travisgalloway/monica/issues/94) |
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

### The three-class storage layout (reserve — M10 distillation)

One layout keeps the **student architecture downstream of every frozen artifact**, so a layout
sweep invalidates nothing upstream (the whole point of the M10 distillation strategy — kept here
as reserve/example; the live M12 corpus build, #193, has no frozen-teacher class to isolate):

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
- **Every tokenized folder name-pins `<tokenizer>-<seqlen_k>`** (e.g. `qwen3-8k`), so multiple
  tokenized views coexist without collision (the reserve-pretrain corpus stays `qwen25`).

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
- **Checkpoints:** the durable copy belongs at `s3://<bucket>/ckpt/<run>/` — but nothing syncs it
  there automatically. `scripts/train.py` only writes checkpoints to local/volume disk
  (`store.save(...)`); pushing that tree to R2 is an **operator-driven** step via
  `src/data/r2_sync.py up` (see [Checkpoint cadence on interruptible
  pods](#checkpoint-cadence-on-interruptible-pods) below), not a cadence the trainer runs itself.
- **Install:** the data extras pull `fsspec`/`pyarrow` — `pip install -e ".[data]"` — but the
  S3 filesystem backend is separate: also `pip install "s3fs==<fsspec-pin>"` so `s3://` URLs
  resolve (pin `s3fs` to the same release as `fsspec`, since `datasets` caps `fsspec<=2026.2.0`;
  a bare `pip install s3fs` upgrades `fsspec` and breaks `datasets`). The cloud corpus engine
  adds `pip install -e ".[datatrove]"`.

### Syncing an artifact tree to R2 (#80, first piece)

The builders write **local directory trees**; `src/data/r2_sync.py` mirrors one to/from any
fsspec backend (`file://` locally, `s3://` on R2), reading R2 creds from the env and the endpoint
from `AWS_ENDPOINT_URL_S3` (see `.env.example`). Build locally, then push:

```bash
set -a; . ./.env; set +a                                   # load HF/R2 secrets
python -m src.data.r2_sync up   data/poc-distill s3://<bucket>/poc-distill   # mirror a built tree to R2
python -m src.data.r2_sync down s3://<bucket>/poc-distill data/poc-distill   # pull on a pod
```

### Building the scale corpus with datatrove (#80)

The full-source build (FineWeb-Edu + supplements, cross-source MinHash) runs the datatrove port
in `src/data/datatrove_pipeline.py` + `scripts/build_corpus.py`. It reuses the project filter/dedup
*logic* (`filters.py`/`dedup.py`) as datatrove blocks and writes **cleaned text shards**; the
existing `src/data/shard.py` tokenizes them to the Qwen2.5 uint32 trainer shards.

> **M12 code corpus differs.** The reserve build above tokenizes in Python (`src/data/shard.py`,
> Qwen2.5). The **M12 code path** (`scripts/build_ts_clean_corpus.py`) instead stops at cleaned
> text (`cleaned.jsonl`); the native Swift `monica-tokenize pack` (`swift/`, #191/#245) tokenizes
> + packs it into the same uint16 shard layout. So for M12: **Python cleans → Swift
> tokenizes+packs** — `src/data/shard.py`'s `--tokenizer code` path no longer exists.
> `scripts/build_ts_clean_corpus.py` itself is unchanged by #247 (still emits `cleaned.jsonl`
> directly, no Parquet stage). Separately, `src/data/corpus.py`'s `build_corpus`/`write_shards`
> Parquet shards (the general pipeline used above) are now **directly** readable by
> `monica-tokenize` (a minimal pure-Swift Parquet reader, #247) — no `cleaned.jsonl` round trip
> needed for that path either, as long as the shards are written `compression="snappy"`
> (`--compression snappy`; the pure-Swift reader has no zstd decoder).

**Environment caveat.** datatrove supports Python ≤3.12 and pulls C-extension/`spacy` deps, so it
runs in a **dedicated py3.11 venv** matching the RunPod `py3.11` images — *not* the main py3.14
`.venv`:

```bash
python3.11 -m venv .venv-dt && .venv-dt/bin/pip install -e ".[dev,data,datatrove]" && .venv-dt/bin/pip install spacy
set -a; . ./.env; set +a
# CPU pod: clean + cross-source MinHash dedup straight to R2 (cleaned/ and dedup/deduplicated/):
.venv-dt/bin/python scripts/build_corpus.py --source fineweb-edu \
    --out s3://monica-training/reserve-pretrain --executor slurm --tasks 200 \
    --quality --license-filter --scrub --dedup
# then tokenize the cleaned shards (trainer format unchanged):
.venv-dt/bin/python -m src.data.shard --in <out>/dedup/deduplicated \
    --out <out>/tokenized/v1-qwen25-8k --tokenizer qwen25 --seq-len 8192
```

A bounded local smoke (`--limit N --executor local`) validates the wiring; note that `--limit`
truncates the HF streaming reader early, which leaves the process hanging at interpreter exit
(a non-daemon datasets thread) — harmless, and absent on full (no-`limit`) pod runs.

---

## RunPod specifics

RunPod provides the on-demand compute. Two roles, kept separate:

- **CPU pod** — the data stages (ingest / clean / dedup / tokenize). Install `datatrove` +
  `s3fs` + tokenizer deps; its S3 reader/writer point at R2.
- **GPU pod** — training only. Pull the tokenized subset + teacher outputs from R2 to a network
  volume, train, checkpoint back to R2.

### Region pin — a one-way decision

RunPod network volumes are **region-locked** and cannot be moved once created — every later pod
that needs the corpus + checkpoints must be schedulable in that same region, so the pin is
effectively permanent. Choose the region by **where the card you will actually train on has
capacity** — H100 for the M12 large run (#223) — not by proximity to R2. R2 has no egress fee
(above), so "network-close to R2" is a *latency*, not a *cost*, consideration: worth weighing once
the H100-capacity region is chosen, but subordinate to it.

### Instance tiering — never pay H100 rates for CPU work

The pod-role split above says *which stage* runs where; this is *which tier*. Every CPU-bound
stage — datatrove clean/dedup (`scripts/build_corpus.py`), the M12 TS corpus build
(`scripts/build_ts_clean_corpus.py`), and BPE train/encode/pack via the native Swift
`monica-tokenize` CLI (`swift/`, #191/#245) — is integer/CPU work with **no CUDA path at all** (no
MLX either — BPE is CPU/integer work). Run these on an **A40/4090-class** pod, or a plain CPU pod;
an H100 there buys nothing and burns the most expensive hour on the menu.

The same rule picks the card for the dress rehearsal in the bring-up order below: steps 2–3 (toy
smoke gate, train-step bench) are deliberately cheap-card work before step 4 commits to the real
card — tier the pod to the step, not to the run.

Ordering only, no $/hr figures: **H100 ≫ A40/4090 ≫ CPU**. Rates are an external market that
changes; re-price at rental time (the same discipline the reserve run's cost notes keep,
[`reserve/path-b-run.md`](reserve/path-b-run.md)).

### Billing hygiene — stop is not terminate; the Startup Program

**Stop ≠ terminate.** A *stopped* RunPod pod still bills — its container disk is retained. The
state that must survive a pod is the **network volume** plus whatever was pushed to R2; neither
needs the pod alive. So the concrete mechanics of "no pod stands idle" (above): sync off, then
**terminate** — let the volume carry state to the next pod, not a stopped instance.

**Network volume storage bills independently of any pod**, for as long as the volume exists —
the other half of why the region pin above is effectively permanent: a wrong region pin costs
money until the volume is deleted, not just until the pod is stopped.

**RunPod Startup Program** offers eligible projects up to **≤1000 free H100-hours** — check
eligibility and apply *before* spending on the M12 large run (#223), the program's dominant GPU
cost. Terms are external and change; verify at application time, same as the external-rate
discipline above.

### Checkpoint cadence on interruptible pods

1. **Target: a committed checkpoint every 20–30 minutes.** Interruptible/spot capacity can vanish
   without warning; the exposure window between commits is the loss budget.
2. **The knob is a step count, not wall-clock.** `--ckpt-every` (`scripts/train.py:58`, CLI
   default **500**) sets `TrainConfig.ckpt_every` (`src/train/loop.py:38`, dataclass default
   **100** — the CLI default wins for `scripts/train.py`; the two must not be conflated), and the
   loop fires on `step % cfg.ckpt_every == 0` (`src/train/loop.py:120`). There is **no timer** —
   the operator must convert minutes to steps.
3. **The arithmetic:**

   ```
   ckpt_every ≈ target_minutes × 60 / measured_s_per_step
   ```

   Worked against the repo's only published baseline — **~99 s/step on an M1 Pro / MLX** run,
   batch 32 × grad_accum 4 × seq 1024 = 131,072 tok/step, fp16 (`scripts/bench_train_step.py`,
   #31/#30) — **not** a CUDA or H100 number: 20 min → `1200/99 ≈ 12` steps; 30 min →
   `1800/99 ≈ 18` steps. At that rate the shipped CLI default of 500 is **~13.7 hours of
   exposure** and must be lowered explicitly. Invert it: on a card fast enough to hit ~2 s/step,
   the same 20–30 min window is ~600–900 steps and 500 is already about right. The number is
   **per-pod and must be computed from a measurement**, never copied from this doc — measure with
   `scripts/bench_cuda_train_step.py` (step 3 of the bring-up order below) before setting the
   flag.
4. **Cost of the write.** Each commit rewrites the inactive slot in full — weights, optimizer
   state, and `resume_meta.json`, every file fsync'd, then the `LATEST` flip
   (`src/train/checkpoint.py:179-200`). Time one `store.save` on the pod and keep the interval a
   comfortable multiple of it; the repo has not measured this cost, so don't quote a number here.
5. **Where the checkpoint lands — and does not.** `store` roots at `<out>/resume`, on the pod's
   local disk / volume (`scripts/train.py:117-118`), and `on_checkpoint` calls only
   `store.save(...)` (`scripts/train.py:130-134`). **`scripts/train.py` does not push to R2.**
   Getting the checkpoint durable is an operator responsibility: a side loop calling
   `python -m src.data.r2_sync up <out>/resume s3://<bucket>/ckpt/<run>/`
   (`src/data/r2_sync.py`) — the correction to the R2-specifics checkpoint note above. The
   double-buffered layout makes a mid-sync copy safe to interpret: `LATEST` is the single commit
   point (`src/train/checkpoint.py:126-139`).

> **Status.** A checkpoint restores **model weights, optimizer state, step, and fp16 loss-scale
> state — and nothing else.** `resume_meta.json` holds exactly `{step, loss_scale_state}`
> (`src/train/checkpoint.py:189`), and resume reads back only those
> (`scripts/train.py:120-126`). **No dataloader/stream state is saved.** The data position is
> *re-derived* from `start_step * grad_accum` (`src/train/loop.py:95-96`) against the loader's
> epoch length (`src/train/loop.py:60-67`, `src/data/loader.py:46-48`).

The operational consequence is the whole point of the callout: the stream is reproduced **only if
every implicit input is byte-identical** across the kill —

- `--seed`
- `--batch-size`
- `--grad-accum`
- `seq_len` (from the config)
- the packed `train.bin` itself (its token count sets `n_chunks`, hence `len(train_loader)`)

Change any one of these — resize the batch to fit a different card, repack the corpus, resume
against a different split — and the resumed run silently reads a **different** data order,
re-reading seen data and corrupting epoch accounting. **Nothing detects this.** That is precisely
the gap [#216](https://github.com/travisgalloway/monica/issues/216) is open to close
("interruptible pods need explicit dataloader state saved alongside model/optimizer"); until it
lands, the invariance is an **operator discipline**, enforced by the checklist below.

One genuine positive, verified: the fast-forward itself is **free** —
`epoch(skip_batches=...)` slices the already-shuffled index (`src/data/loader.py:64-65`) without
reading any chunk, so a resume deep into the corpus costs no replay time.

### Shard dtype

Trainer shards pack as **uint16** when the vocab is `< 65536` (`src/data/pack.py:30-33`, #90),
with the dtype recorded in the `.meta.json` sidecar so the loader reads it back without parsing
during training (`pack.py:86`, `pack.py:94-99`). M12's own BPE is well under the ceiling, so M12
shards are uint16 — half the bytes to move and mmap per pull. Already implemented, and the Swift
packer emits the same layout (above); nothing to do here.

**GPU pod spec.** The card choice is driven by **precision**, not a blanket rule: the **1B
training** configs (`config/student-1b.yaml`, `config/1b.yaml`) are **bf16**, which needs an
**Ampere-or-newer** card (a T4/Turing has no bf16). The cheaper **smoke gate** (`config/toy.yaml`,
fp32) and **train-step bench** (`config/poc.yaml`, fp16) below run fine on a **T4/L4** — so a
T4/L4 is enough to dry-run the flow, and you only need Ampere+ for the actual bf16 run. Use a
RunPod **`-devel`** image so the build sees the preinstalled CUDA torch (e.g.
`runpod/pytorch:2.4.0-...-devel-ubuntu22.04` or the `2.8.0-...-cudnn-devel` image). Then:

```bash
# 1. Backend install (the [cuda] extra pulls torch; mlx is Mac-only).
#    For any GPU TRAINING/precompute run use the [cuda-fast] extra — it adds the fused
#    mamba-ssm Triton SSD scan + causal-conv1d (#40). Without them the SSD scan/conv fall
#    back to pure PyTorch (much slower); the CUDA backend logs a RuntimeWarning at model
#    build if it's running on GPU without them, so you catch a missing install early.
pip install -e ".[dev,data,cuda-fast]"
#    [cuda] alone (no fused kernels) is fine only for CPU-parity tests / data-prep:
#    pip install -e ".[dev,data,cuda]"

# 2. CUDA smoke gate — prove the torch backend resumes bit-exactly through the
#    double-buffered CheckpointStore. Build a tiny toy split on the pod, then:
python scripts/smoke_test.py --backend cuda --data <toy-split>     # config/toy.yaml (dense)
#    Or exercise the CUDA MoE path (#214 — MoE is no longer MLX-only) the same way:
python scripts/smoke_test.py --backend cuda --config config/toy-moe.yaml --moe-impl gather \
    --data <toy-split>

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

### Manual verification: 8-bit optimizer + fp8 experts (#214)

Both levers are real code paths, not stubs, but neither is CI-testable: `cuda-cpu` runs on CPU
torch with no `bitsandbytes`/`transformer-engine` installed at all, and `full-macos`/`portable`
have no CUDA. What CI DOES cover (`tests/test_cuda_8bit_optimizer.py`, `tests/test_cuda_fp8.py`):
config surface, `optimizer_sizing_key`, the `SystemExit`/`NotImplementedError` seam guards, and
the lazy-import guarantee (`tests/test_import_guard.py`'s `FORBIDDEN_ROOTS`). Everything below
needs real hardware and is a merge-adjacent checklist item, not a blocking gate.

**8-bit AdamW (`optimizer_8bit: true`, any Ampere+ card):**

```bash
pip install -e ".[dev,data,cuda-8bit]"     # bitsandbytes — CUDA-only wheels, no macOS build
python scripts/smoke_test.py --backend cuda --config config/toy-moe-8bit.yaml --data <toy-split>
```
- [ ] Model builds and trains without the `SystemExit`/`ImportError` path firing.
- [ ] `nvidia-smi`/`torch.cuda.max_memory_allocated()` shows a lower optimizer-state footprint
      than the same config with `optimizer_8bit: false` (the whole point of the lever).
- [ ] The resume-exactness phase of the smoke gate still passes — a bitsandbytes state-dict
      round trip through `CheckpointStore` must not silently drop or requantize state.
- [ ] Confirm the tied embedding's optimizer state actually stayed 32-bit: inspect
      `bnb.optim.GlobalOptimManager.get_instance().pid2config` for `id(model.embedding.weight)`,
      or diff its per-param memory footprint against a non-embedding AdamW8bit param of similar
      size.
- [ ] `muon` + `optimizer_8bit` stacked: confirm the memory savings are small relative to plain
      `adamw` + `optimizer_8bit` — Muon already owns the expert matrices and the embedding is
      forced to 32-bit either way, so this is a sanity check, not a real lever.

**fp8 MoE experts (`fp8_experts: true`, Hopper+/sm_90 only — H100, H200, etc.):**

```bash
pip install -e ".[dev,data,cuda-fp8]"      # transformer-engine — Hopper-only prebuilt wheels
pytest -q tests/test_cuda_fp8.py           # un-skips the two Hopper-gated acceptance tests here
python scripts/smoke_test.py --backend cuda --config config/toy-moe.yaml --moe-impl gather \
    --data <toy-split>   # then repeat with fp8_experts: true set in the config
```
- [ ] `[cuda] fp8 MoE experts ACTIVE (Transformer Engine, Hopper+).` printed at model build
      (`_report_fp8_status_once`) — a silent bf16 fallback here is a throughput trap, not a
      correctness bug, so it is easy to miss without checking this line.
- [ ] `tests/test_cuda_fp8.py::test_fp8_expert_forward_matches_bf16` and
      `::test_fp8_expert_checkpoint_backward_finite_grads` pass (both skip everywhere else).
- [ ] A real training step under `grad_checkpoint: true` + `fp8_experts: true` runs several
      hundred steps without the fp8 amax history diverging into non-finite loss — the `te.
      checkpoint` vs plain-checkpoint distinction this PR wires in is exactly the failure mode
      a short smoke run would not catch (amax drift compounds over many steps).
- [ ] `torch.compile` (`--compile` on `smoke_test.py`, or `torch_compile: true`) still graph-breaks
      cleanly around `te.Linear`/`te.checkpoint` rather than hard-erroring.

---

## Resumable-job checklist

A rented pod is interruptible; everything above exists to make an interruption a non-event. Work
through these before/during/after a run — grouped cheapest-first, so a failure surfaces before the
dominant spend.

### A. Before renting

- [ ] RunPod Startup Program eligibility checked (up to ≤1000 free H100-hours).
- [ ] Region chosen by **H100 availability**, not R2 proximity; network volume created there
      (effectively permanent once created).
- [ ] Instance tier picked per stage — CPU-bound work (corpus/tokenizer) kept off H100.

### B. Pod bring-up

- [ ] `[cuda-fast]` installed and the fused kernels actually engaged (#40) — the CUDA backend
      warns at model build if they're missing.
- [ ] `s3fs` pinned to the same release as `fsspec` (above).
- [ ] `r2_sync down` verified working in-region.
- [ ] Toy CUDA smoke gate green: `scripts/smoke_test.py --backend cuda`.

### C. Cadence sizing

- [ ] `scripts/bench_cuda_train_step.py` run at the real shape; `s/step` and peak memory recorded.
- [ ] `--ckpt-every` computed for the 20–30 min target and **explicitly passed** (not left at the
      500 default).
- [ ] One `store.save` timed and confirmed small against the checkpoint interval.

### D. Resume invariance (until #216 lands)

- [ ] The exact `scripts/train.py` command line recorded verbatim alongside the run.
- [ ] `--seed` / `--batch-size` / `--grad-accum` / config `seq_len` / `--data` pinned and reused
      byte-identically on resume.
- [ ] A deliberate kill-and-resume rehearsed once, early in the run — confirm
      `[resume] from step N` (`scripts/train.py:127`) and a continuous `metrics.jsonl`.

### E. Durability + teardown

- [ ] Checkpoints synced to `s3://<bucket>/ckpt/<run>/` via `r2_sync up` (manual — not automatic).
- [ ] Final `weights.safetensors` materialized (`scripts/train.py:161`).
- [ ] Pod **terminated**, not stopped.
- [ ] Volume kept only if the next stage needs it — otherwise deleted (it bills independently).

---

## End-to-end intended flow (M10-era, reserve — #80/#81 never fully landed before the pivot)

This flow is specific to the (dropped) M10 distillation program; kept as reserve/history, not a
live target. For the concrete, command-by-command Path B execution of this flow (the full-scale
~1B distillation run — exact commands, pod sizing, R2 paths, cost, and the Path A gotchas), see
[`reserve/path-b-run.md`](reserve/path-b-run.md); for the completed ~205M `poc-qwen` run's pod
recipe + asset inventory, see [`reserve/runpod-poc-run.md`](reserve/runpod-poc-run.md). The steps
below are the generic shape.

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
