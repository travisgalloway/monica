# M10 pod-chain — ready-to-fire staging (steps 3–4)

> **⛔ Reserve / historical (M10 distillation, superseded 2026-07-19).** This program is no longer
> active — the "ready-to-fire" chain below is **parked**, not live. See
> [`../../design/13-code-model-moe.md`](../../design/13-code-model-moe.md) and
> [issue #198](https://github.com/travisgalloway/monica/issues/198) for the live M12 code-model
> arc. Retained as the inventory of R2 assets (corpus + ~566 GB teacher cache) that may still
> occupy paid storage.

> **Update (2026-07-03).** Step 3 (teacher precompute) below completed for the **base** corpus on
> 2026-07-02. The corpus was then extended with new domains (#176); extending the teacher cache to
> cover those new chunks is a separate **append** step (3′) that must run **before** Step 4, reusing
> the base cache rather than re-running Step 3 from scratch — see
> [`m10-phase-bprime-append.md`](m10-phase-bprime-append.md) (#177).

The pod-gated stages of the M10 distillation chain — **teacher precompute** (#94) and the
**two-layout student sweep** (#81) — are the only steps that need a rented CUDA pod. This doc is
the *staging* companion: it records the plumbing validations that were run **without** a pod, and
gives the exact copy-paste block to fire steps 3–4 the moment a pod is up.

It is intentionally short and defers all reasoning (cost, sizing, gotchas, the full step-by-step) to
**[`../path-b-run.md`](../path-b-run.md)**, the authoritative Path B runbook. Read that for *why*;
read this to *go fast*. See also [`../infrastructure.md`](../../infrastructure.md) (generic R2 + RunPod).

## Status: staging validated (no pod, no GPU run)

Verified locally on the dev Mac — the off-pod half of the chain is green:

| Check | Command | Result |
|---|---|---|
| datatrove env (CPU-pod data stages) | `.venv-dt/bin/python -c "import datatrove; from datatrove.pipeline.readers import JsonlReader"` | OK — Python 3.11.15 (matches the RunPod py3.11 image; not the main py3.14 `.venv`) |
| R2 reachable + writable | `r2_sync up`/`down` round-trip to a throwaway `s3://monica-training/_smoke/staging-check` | 2 files up, 2 down, `diff -r` byte-identical; prefix deleted after |
| Manifest ↔ corpus alignment | `grep seq_len / tokenized / teacher_outputs config/manifests/student-1b-attn-{hi,lo}.yaml` | both: `seq_len 8192`, `poc-distill/corpus/tokenized/qwen3-8k`, `.../teacher-outputs/topk-logits` |

The teacher precompute is **positionally aligned** to the corpus split, so the split's `seq_len`
**must** equal the manifest's **8192** — confirmed for both sweep manifests. Changing the corpus,
tokenizer, or `seq_len` invalidates the precompute; the student `layout` does not (sweep is free).

R2 bucket `monica-training` currently holds `ckpt/`, `models/`, `reserve-pretrain/`. `poc-distill/`
is reserved/empty — it is created by step 1 of `path-b-run.md` (corpus tokenize) and the precompute.

## When a pod is up: fire steps 3–4

Prereqs on the pod (one time — see `path-b-run.md` Step 0 for detail):
```bash
git clone https://github.com/travisgalloway/monica && cd monica
pip install -e ".[dev,data,cuda-fast]"      # fused Mamba Triton scan + causal-conv1d (#40); without them the SSD scan silently falls back to slow pure-PyTorch
pip install "s3fs==2026.2.0"                 # pin to fsspec; a bare install upgrades fsspec and breaks datasets
set -a; . ./.env; set +a                     # R2 creds + HF_TOKEN (HF_TOKEN optional — Qwen3-4B-Thinking-2507 is Apache-2.0/ungated)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # required for the MOHAWK O(L^2) stage-1 (see below)
```

> **`cuda` vs `cuda-fast`, not a contradiction.** `scripts/runpod/m10/bootstrap.sh` actually runs
> `pip install -e ".[dev,data,cuda]"` followed by a separate `pip install "mamba-ssm>=2.0"
> "causal-conv1d>=1.0" --no-build-isolation` — `mamba-ssm`'s `setup.py` imports `torch` at build
> time, which pip's isolated build env for a plain `.[cuda-fast]` extra doesn't have, so the
> one-line extras form fails there. `bootstrap.sh`'s two-step sequence is the documented
> workaround for that pip build-isolation issue, not a different/lesser install — it lands the
> same fused Triton scan + `causal-conv1d` as `.[dev,data,cuda-fast]` above.

### Step 3 — teacher precompute (the dominant $, run ONCE; both layouts reuse it)
Needs an **Ampere+ 80 GB** card (A100/H100) for the bf16 / seq-8192 path. Bench a small slice first.
**Prefer H100** — a single card run in sequence for cost simplicity, or an **8× H100 cluster for
this phase only** (embarrassingly parallel over chunks) when wall-clock matters. See
[`../path-b-run.md`](../path-b-run.md) §"Time & \$ estimate" for the full time/$ table
(~\$150–1,100 depending on measured MFU — bench first).
```bash
python -m src.data.r2_sync down s3://monica-training/poc-distill/corpus/tokenized/qwen3-8k /vol/corpus8k
python -m src.data.split --shards /vol/corpus8k --out /vol/split8k --val-tokens 10000000   # seq_len stays 8192
python scripts/precompute_teacher.py \
    --manifest config/manifests/student-1b-attn-hi.yaml \
    --data /vol/split8k --splits train,val --backend cuda --k 50 --batch-size 8 \
    --out /vol/teacher-outputs/topk-logits \
    --push s3://monica-training/poc-distill/teacher-outputs/topk-logits
```
Footprint ≈ 6·k bytes/token (k=50 → ~300 B/token). The manifest here only supplies
`conversion_teacher` + `seq_len` (identical across hi/lo), which is why one precompute serves both.

### Step 4 — the two-layout sweep (reuses the single precompute)
> Run **Step 3′ — the #177 append** first if the corpus extension (#176) hasn't been merged into
> the teacher cache yet: [`m10-phase-bprime-append.md`](m10-phase-bprime-append.md). Otherwise this
> sweep trains only against the base FineWeb-derived corpus.

Two sibling manifests, **same** corpus + teacher outputs, only `layout` differs. The three distill
stages run in order (mixing-match → hidden-align → logit-distill). This is the "sweep" — two
manifests × three stages → a Phase-2 winner pick; it is one team-Workflow call away once the pod
exists (no repo runner needed — it's the loop below).

> **`--teacher-outputs` is the sole source, not an override.** `scripts/distill.py` never reads a
> `teacher_outputs` field out of the manifest at all — the CLI flag below is mandatory whenever
> the `logit-distill` stage runs and is the *only* place the path comes from. More generally, the
> manifest fields `corpus`, `teacher_outputs`, `sft`, `rl`, `schedule` are parsed by the manifest
> loader (`src/train/distill_manifest.py`) but currently **ignored** by `distill.py` (decorative —
> everything it actually needs is passed on the CLI: `--corpus`, `--teacher-outputs`, etc.). Also
> note there are **two separate per-layout file families** that must be kept manually in sync when
> tuning a layout: `config/manifests/student-1b-attn-{hi,lo}.yaml` (read by this Step-4 sweep) vs.
> the flat `config/student-1b-attn-{hi,lo}.yaml` at the config root (resolved `MambaConfig` YAMLs
> read by Step-5's `scripts/sft.py`/`scripts/rlvr.py`, **not** by `distill.py`).
```bash
for M in student-1b-attn-hi student-1b-attn-lo; do
  python scripts/distill.py --manifest config/manifests/$M.yaml \
      --corpus /vol/split8k --teacher-outputs /vol/teacher-outputs/topk-logits \
      --backend cuda --out /vol/runs/$M \
      --batch-size 1 --grad-accum 16 \
      --steps-per-stage 1000 --k 50 --temperature 2.0 --ce-weight 0.1 --kl-weight 0.9 \
      --eval-every 200 --ckpt-every 500
  python -m src.data.r2_sync up /vol/runs/$M s3://monica-training/ckpt/$M   # pods are ephemeral — sync each layout
done
# interrupted? re-add --resume (resumes the furthest-progressed stage of that manifest).
```

> **MOHAWK stage-1 OOM (O(L²)).** `mixing-match` materializes a per-layer `(B,H,L,L)` mixing matrix
> for student **and** teacher. At seq 8192 this is ~64× tighter than the seq-1024 Path-A run that
> already OOM'd an 80 GB card at batch 8 — so use a **micro-batch of 1–2 + large grad-accum** and
> keep `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Later stages are looser.

**Phase-2 gate (winner pick).** Compare the two layouts on the val-perplexity / logit-distill curve
**plus** the local-hardware target (context length + tok/s) — **not** benchmark scores. Keep the
winning manifest; post-train it (Step 5 in `path-b-run.md`).

## Re-running the staging validations
```bash
.venv-dt/bin/python -c "import datatrove; from datatrove.pipeline.readers import JsonlReader; print('datatrove OK')"
set -a; . ./.env; set +a
T=$(mktemp -d); printf 'x\n' > "$T/a.txt"
python -m src.data.r2_sync up "$T" s3://monica-training/_smoke/staging-check
python -m src.data.r2_sync down s3://monica-training/_smoke/staging-check "$T-down" && diff -r "$T" "$T-down" && echo OK
python -c "from src.data.r2_sync import _fs_for; fs,_=_fs_for('s3://monica-training/'); [fs.rm_file(f) for f in fs.find('monica-training/_smoke')]"
```

## Verification checklist (confirm on the CUDA pod)

Everything above was built and unit-tested on a Mac; these are the **real-hardware risks that a
Mac cannot exercise**. Ordered cheapest-first, so a failure surfaces *before* the dominant GPU
spend (step 3). The corpus + SFT corpora already passed their Mac-side manifest assertions and
are staged on R2 (see #65) — the items below re-verify them in the CUDA environment.

### A. Environment — before any GPU spend
- [ ] `pip install -e ".[dev,data,cuda-fast]"` succeeds (`mamba-ssm>=2.0` + `causal-conv1d>=1.0`).
- [ ] **The fused Triton SSD scan actually engages** — not the silent pure-PyTorch fallback. Import
  `mamba_ssm` / `causal_conv1d` and confirm a forward uses them (else throughput tanks; this is the
  whole point of `cuda-fast`, #40).
- [ ] `s3fs==2026.2.0` pin holds (a bare `pip install s3fs` upgrades fsspec and breaks `datasets`).
- [ ] `r2_sync down` works from the pod in-region (zero-egress); `.env` creds resolve.
- [ ] bf16 path ⇒ **Ampere+** card (A100/H100 80 GB). Confirm `nvidia-smi`.

### B. Corpus integrity (cheap — before precompute)
- [ ] `r2_sync down s3://monica-training/poc-distill/corpus/tokenized/qwen3-8k /vol/corpus8k`, then
  `split --shards /vol/corpus8k --out /vol/split8k --val-tokens 10000000`.
- [ ] Assert: dtype **uint32**, train/val provably disjoint, doc-boundary `.bounds` present, and
  **max token id < `len(tokenizer)`** — the Qwen3 special-token bound (`<|im_start|>`/`<|im_end|>`
  exceed `vocab_size`) fixed in #153/#154. Confirm the on-pod data clears it.

### C. Pre-flight CUDA smoke (before the dominant-cost run)
- [ ] `python scripts/distill_smoke.py --backend cuda` — student init (MOHAWK / Mamba-in-the-Llama)
  + all three stages (mixing-match → hidden-align → logit-distill) run end-to-end at toy scale.
- [ ] `python scripts/precompute_teacher.py --synthetic --backend cuda` then a tiny **real** slice —
  confirms **Qwen3-4B-Thinking loads on CUDA** (per-head QK RMSNorm, no QKV bias, `rope_theta` 5e6),
  top-k cache writes + `--push` round-trip.
- [ ] **Fill in `src/conformance/backend_parity.py`** (currently a skeleton — runs only where CUDA
  is present) and run `tests/test_cuda_parity.py`: MLX↔CUDA agreement in **fp32 at ~1e-4 rel**. This
  is the conformance check explicitly deferred to the CUDA scale-up.

### D. Step 3 — teacher precompute (dominant cost)
- [ ] Bench tok/s + $/Mtoken on a small slice and **extrapolate to ~1.9B before committing**.
- [ ] Positional alignment: the split's `seq_len` (8192) **equals** the manifest's — the precompute
  is positionally aligned to the split.

### E. Step 4 — student sweep (top failure risk)
- [ ] **MOHAWK stage-1 O(L²) OOM at seq 8192** — `mixing-match` materializes a per-layer
  `(B,H,L,L)` matrix for student **and** teacher. The seq-1024 Path-A rehearsal already OOM'd an
  80 GB card at batch 8; at 8192 it is ~64× tighter. Verify **micro-batch 1–2 + high grad-accum +
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** actually fits. (Most likely failure.)
- [ ] `torch.compile` of the student forward (#145/#147) and SDPA (#144/#146) engage on CUDA with
  no graph-break errors.
- [ ] **Checkpoint-to-R2 + `--resume` survives a kill** (pods are ephemeral) — resumes the
  furthest-progressed stage of that manifest.
- [ ] All three stage losses drop for **both** layouts (the Path-A success signal) → Phase-2 winner.

### F. Step 5 — post-train the winner
- [ ] `scripts/sft.py` (instruct → reasoning) and `scripts/rlvr.py` (GRPO) run on CUDA.
- [ ] CUDA seg_ids doc-boundary reset (#111) holds under packing.

### G. Step 6 — headline (back on Mac / MLX)
- [ ] The CUDA-trained **portable safetensors load in the MLX backend** (the seam round-trip is the
  whole point) — then context length + tok/s vs a same-size transformer = the POC success gate (#104).
