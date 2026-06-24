# M10 pod-chain — ready-to-fire staging (steps 3–4)

The pod-gated stages of the M10 distillation chain — **teacher precompute** (#94) and the
**two-layout student sweep** (#81) — are the only steps that need a rented CUDA pod. This doc is
the *staging* companion: it records the plumbing validations that were run **without** a pod, and
gives the exact copy-paste block to fire steps 3–4 the moment a pod is up.

It is intentionally short and defers all reasoning (cost, sizing, gotchas, the full step-by-step) to
**[`../path-b-run.md`](../path-b-run.md)**, the authoritative Path B runbook. Read that for *why*;
read this to *go fast*. See also [`../infrastructure.md`](../infrastructure.md) (generic R2 + RunPod).

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

### Step 3 — teacher precompute (the dominant $, run ONCE; both layouts reuse it)
Needs an **Ampere+ 80 GB** card (A100/H100) for the bf16 / seq-8192 path. Bench a small slice first.
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
Two sibling manifests, **same** corpus + teacher outputs, only `layout` differs. The three distill
stages run in order (mixing-match → hidden-align → logit-distill). This is the "sweep" — two
manifests × three stages → a Phase-2 winner pick; it is one team-Workflow call away once the pod
exists (no repo runner needed — it's the loop below).
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
