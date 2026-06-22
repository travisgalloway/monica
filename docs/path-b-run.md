# Path B run — session handoff (full-scale M10 distillation, ~1B student)

Self-contained runbook to execute the **real M10 deliverable** ([issue #141](https://github.com/travisgalloway/monica/issues/141),
parent tracker [#65](https://github.com/travisgalloway/monica/issues/65)): distil a compact ~1B
Mamba-2 hybrid student from the frozen `open-r1/OpenR1-Distill-7B` teacher, sweep the two attention
layouts against **one** precomputed teacher signal, post-train the winner, and eval. Written so a
fresh session (or a clean pod clone) can run it with no prior context.

The *pipeline* is already proven end-to-end on cloud — **Path A** ran the real 7B teacher + real ~1B
student through all three distill stages on an A100 (PRs #139, #140). Path B is the same pipeline at
**full scale and seq_len 8192**. All code is merged: precompute (#135/#94), distill driver (#137/#81),
eval tokenizer fix (#139). **There is no code to write — this is execution.**

Related: [`infrastructure.md`](infrastructure.md) (generic R2 + RunPod flow; this doc is its concrete
Path B companion), [`design/10-distillation.md`](design/10-distillation.md) (why distil / the staged
loss), [`config/manifests/student-1b-attn-hi.yaml`](../config/manifests/student-1b-attn-hi.yaml) +
[`student-1b-attn-lo.yaml`](../config/manifests/student-1b-attn-lo.yaml) (the two sweep trials).

## Current state (already done)

- **Path A validated** (2026-06-21, A100 80GB, ~1hr, pod terminated): real 7B teacher + real ~1B
  student, all three stages' losses dropped (mixing-match 0.00114→0.00025, hidden-align 186.9→49.6,
  logit-distill 5.71→4.22); portable ~1B `weights.safetensors` (4.12 GB) written; init froze the 3
  attention layers [7,15,23], ~979.8M trainable. Used a 4M/1M-token slice of the **1k** val data
  (no 8k corpus built yet). Manifest `config/manifests/student-1b-attn-hi-1k.yaml` (PR #140).
- **On R2** — `s3://monica-training/`:
  - `reserve-pretrain/cleaned/` — durable **tokenizer-agnostic JSONL** (3.54 GB). Re-tokenize from
    here cheaply; never re-clean.
  - `reserve-pretrain/tokenized/v1-qwen25-1k/` — the 1.91B-token **seq_len 1024** Qwen2.5 view (used
    by the poc-qwen run and Path A; *not* the 8k view Path B needs).
  - `poc-distill/` — deliberately **empty/clean**, reserved for this run's artifacts.
- **Manifests** — `student-1b-attn-{hi,lo}.yaml` (the seq_len **8192** sweep, ~1.03B each) point at
  the **same** `poc-distill/corpus/tokenized/qwen25-8k` + `poc-distill/teacher-outputs/topk-logits`.
- **Pod ops dir** — `~/.claude/monica-runpod-ops/` (holds the prior pod's TERMINATED state).

## Decisions locked

- **Teacher fixed**: `open-r1/OpenR1-Distill-7B` (fully open, Qwen2.5 tokenizer). Changing it
  invalidates the teacher outputs — so it is pinned first (#91).
- **Tokenizer**: Qwen2.5, vocab 151,646 → **uint32** packing (#90). Shared with the teacher.
- **seq_len 8192**, **bf16** for the 1B student → needs an **Ampere-or-newer** card (A100/H100 80GB;
  bf16 does not exist on Turing/T4).
- **Precompute once, sweep cheap**: the corpus + teacher outputs are frozen; the student layout is
  downstream of every frozen artifact, so the hi/lo sweep invalidates nothing upstream.

## The dependency graph (why ordering matters)

```
  cleaned JSONL ──tokenize@8k──▶ corpus/tokenized/qwen25-8k ─┐         (frozen, built once)
                                                             ├─▶ 7B teacher precompute ─▶ teacher-outputs/topk-logits
                                                             │                                    │  (frozen, the dominant $ — built ONCE)
                                                             ▼                                    ▼
                                                     ┌── distill attn-hi ──┐   ┌── distill attn-lo ──┐   (cheap, reuse the above)
                                                     └─────────┬───────────┘   └──────────┬──────────┘
                                                               └──── pick winner ─────────┘
                                                                          │
                                  shared/sft/qwen25-8k ──▶ instruct-sft ─▶ reasoning-sft ─▶ grpo ─▶ eval
```

Build the two frozen artifacts (corpus, teacher outputs) before any student; build the SFT corpus
any time before post-training.

## Step 0 — code + venvs + secrets on the pod

```bash
# clone main (all Path B code is merged):
git clone https://github.com/travisgalloway/monica && cd monica

# main backend venv (training / precompute / data):
pip install -e ".[dev,data,cuda]"          # torch CUDA backend; add cuda-fast for fused Mamba kernels (#40)
pip install "s3fs==2026.2.0"               # R2; pin to fsspec (datasets caps fsspec<=2026.2.0) — a bare install breaks datasets

# separate py3.11 venv for datatrove ONLY (it caps at py<=3.12 + pulls spacy) — used only if you
# rebuild the corpus from raw sources (Step 1, option B):
python3.11 -m venv .venv-dt && .venv-dt/bin/pip install -e ".[dev,data,datatrove]" && .venv-dt/bin/pip install spacy

# secrets in the pod env (.env holds R2 creds + HF_TOKEN):
set -a; . ./.env; set +a
#   needs: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL_S3, R2_BUCKET=monica-training
#   HF_TOKEN is REQUIRED here (the 7B teacher is gated) — unlike the poc-qwen run which needed none.
```

## Step 1 — build the 8k distillation corpus (CPU pod)

The `cleaned/` JSONL already exists and is tokenizer-agnostic, so the **cheap, recommended path is to
re-tokenize it at seq_len 8192** — no re-clean, no datatrove.

```bash
# A. (recommended) re-tokenize the existing cleaned JSONL to the 8k Qwen view:
python -m src.data.r2_sync down s3://monica-training/reserve-pretrain/cleaned /vol/cleaned
python -m src.data.shard --in /vol/cleaned --out /vol/corpus/qwen25-8k \
    --tokenizer qwen25 --seq-len 8192 --shard-size-mb 512
python -m src.data.r2_sync up /vol/corpus/qwen25-8k \
    s3://monica-training/poc-distill/corpus/tokenized/qwen25-8k

# B. (only if changing sources per #70/#71) full rebuild from raw, in .venv-dt:
#   .venv-dt/bin/python scripts/build_corpus.py --source fineweb-edu --out s3://monica-training/poc-distill \
#       --executor slurm --tasks 200 --quality --license-filter --scrub --dedup
#   then shard the cleaned output exactly as in option A.

# train/val split (disjoint contiguous hold-out), seq_len MUST stay 8192:
python -m src.data.split --shards /vol/corpus/qwen25-8k --out /vol/split8k --val-tokens 10000000
#   -> /vol/split8k/{train.bin,val.bin} (+ .meta.json), uint32.
```

Data-sourcing decisions (which sources, supplements) are tracked in #70 / #71 — option A reuses the
already-decided poc-qwen corpus, which is the fast start.

## Step 2 — build the English chat SFT corpus (CPU pod)

Built with the **existing** machinery — no new code. English instructions + reasoning traces, both
rendered in **Qwen ChatML** (`src/data/chat_template.py`, the single source of truth — `<|im_end|>`
is the chat EOS, identical across SFT/RL/serving) and response-masked.

```bash
# English instruction SFT (OASST1 en + FLAN), response-masked:
python -m src.data.instruct_sft --sources oasst1 flan --tokenizer qwen25 \
    --seq-len 8192 --out-root /vol/shared
#   -> /vol/shared/sft/{cleaned/instruct/records.jsonl, tokenized/qwen25-8k/instruct.jsonl}

# reasoning traces (<think>…</think><answer>…</answer>):
python -m src.data.reasoning_sft --sources mot --tokenizer qwen25 \
    --seq-len 8192 --chunk-align 64 --out-root /vol/shared
#   -> /vol/shared/sft/tokenized/qwen25-8k/{reasoning.jsonl, reasoning-packed/}

python -m src.data.r2_sync up /vol/shared/sft s3://monica-training/shared/sft
```

Tool-use SFT (`src/data/tool_sft.py`) does **not** exist yet (#102) — optional, skip for the first
winner.

## Step 3 — teacher precompute at scale (GPU pod — the dominant cost)

The 7B forward over the whole corpus. Done **once**; both students reuse it (the manifests share
`teacher_outputs`). **Bench on a small slice first** to size the card and cost before the long run.

```bash
python -m src.data.r2_sync down s3://monica-training/poc-distill/corpus/tokenized/qwen25-8k /vol/corpus8k
python -m src.data.split --shards /vol/corpus8k --out /vol/split8k --val-tokens 10000000   # if not already split

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # see gotchas
python scripts/precompute_teacher.py \
    --manifest config/manifests/student-1b-attn-hi.yaml \
    --data /vol/split8k --splits train,val --backend cuda --k 50 --batch-size 8 \
    --out /vol/teacher-outputs/topk-logits \
    --push s3://monica-training/poc-distill/teacher-outputs/topk-logits
```

- Footprint ≈ **6·k bytes/token** (k=50 → ~300 B/token). Reused by both layouts (the manifest only
  supplies `conversion_teacher` + `seq_len` here, identical across hi/lo).
- **Positional alignment**: the cached outputs are aligned to whatever split is passed — the corpus
  seq_len **must** be 8192 to match the manifest. Re-run if the split changes.
- Path A measured the 7B forward at ~7 s/batch (seq **1024**, batch 8) on an A100; at seq 8192
  expect a much smaller batch — bench to get the real s/batch, then multiply by token count.

## Step 4 — student sweep (GPU pod — reuse the one precompute)

Two sibling manifests, **same** corpus + teacher outputs, only `layout` differs:
`attn-hi` (`attention_every 8` → 3/28 attn ≈10.7%, `state_size 128`) vs
`attn-lo` (`attention_every 14` → 2/28 attn ≈7.1%, `state_size 192`). Stages run in order:
mixing-match → hidden-align → logit-distill.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for M in student-1b-attn-hi student-1b-attn-lo; do
  python scripts/distill.py --manifest config/manifests/$M.yaml \
      --corpus /vol/split8k --teacher-outputs /vol/teacher-outputs/topk-logits \
      --backend cuda --out /vol/runs/$M \
      --batch-size 1 --grad-accum 16 \
      --steps-per-stage 1000 --k 50 --temperature 2.0 --ce-weight 0.1 --kl-weight 0.9 \
      --eval-every 200 --ckpt-every 500       # checkpoint cadence — pods are ephemeral
  python -m src.data.r2_sync up /vol/runs/$M s3://monica-training/ckpt/$M
done
# resume an interrupted run: re-add --resume (resumes the furthest-progressed stage).
```

**Pick the winner** on the val-perplexity / logit-distill curve plus the local-hardware target — not
benchmark scores (see Step 6). Then compare the two layouts and keep the winning manifest.

> **MOHAWK stage-1 OOM (carried from Path A).** mixing-match materializes a per-layer `(B,H,L,L)`
> mixing matrix for student **and** teacher — **O(L²)**. At seq 1024, batch 8 OOM'd an 80 GB card;
> at seq **8192** this is ~64× tighter, so expect a **very small micro-batch (1–2) + large
> grad-accum** and keep `expandable_segments:True` set. Tune `--batch-size`/`--grad-accum` down
> until stage 1 fits; later stages (hidden-align, logit-distill) are looser.

## Step 5 — post-train the winner (GPU pod)

On the Qwen ChatML SFT corpus from Step 2, applied to the winning student's `weights.safetensors`.

```bash
# instruct-SFT (sft.py supports --backend cuda; expects a dir with train.jsonl/val.jsonl):
python scripts/sft.py --backend cuda --config <winner config> \
    --init /vol/runs/<winner>/logit-distill/weights.safetensors \
    --data /vol/shared/sft/tokenized/qwen25-8k/instruct --out /vol/runs/sft-instruct \
    --epochs 2 --batch-size 8 --grad-accum 16 --base-lr 2e-5 --ckpt-every 100
```

**Open items to resolve before running the rest of Step 5** (these are why #101/#102/#103 are still
open — flag, don't improvise):

- **reasoning-SFT** — `src/data/reasoning_sft.py` builds the data, but there is **no dedicated
  reasoning-SFT training driver** in `scripts/`. It reuses the masked-CE SFT machinery; confirm how
  `sft.py` consumes the reasoning records (jsonl vs the `reasoning-packed/` form) before running, or
  track #101.
- **tool-SFT** — optional, code not built (#102). Skip for the first winner.
- **GRPO** — `scripts/rlvr.py` (`--init <sft weights> --problems <jsonl {prompt,answer}> --reward
  math`) is documented **MLX-only** (it uses the serving recurrence) and does **not** take a
  `--backend` flag. A ~1B GRPO run on Mac/MLX is slow; the GRPO *step factory* has CUDA parity
  (M9) but the driver does not yet wire a CUDA backend — resolve via #103 before the RL pass.

## Step 6 — eval (Mac/MLX + harness)

```bash
# long-context behavior — the headline SSM property. Path A's poc-qwen extended 1k→8k with NO
# degradation natively (knob OFF). Use --batch-size 2: at seq 8192 the (B,T,V=151646) logit tensor
# is ~40 GB fp32 at batch 8 → Metal OOM.
python scripts/long_context.py --config <winner config> --weights <winner weights> \
    --data /vol/split8k --mults 1 2 4 8 --batch-size 2

# OLMES / lm-eval (near-chance expected at ~1B — judge by the val-ppl curve, not the scores):
python scripts/eval_olmes.py --config <winner config> --weights <winner weights> \
    --tokenizer qwen25 --tasks hellaswag,arc_easy,arc_challenge,piqa --limit 200 \
    --output runs/eval/olmes-path-b.json
```

**POC success = a smoothly decreasing held-out val-perplexity curve + a local-hardware win** (context
length + tok/s vs a same-size transformer, #104) — *not* benchmark scores.

## Cost & sizing summary

| Stage | Pod | Why | Output prefix (R2) |
|---|---|---|---|
| 1 corpus tokenize@8k | CPU | re-tokenize cleaned JSONL | `poc-distill/corpus/tokenized/qwen25-8k` |
| 2 SFT corpus | CPU | instruct + reasoning, ChatML | `shared/sft/tokenized/qwen25-8k` |
| 3 teacher precompute | **GPU (A100/H100 80GB)** | **dominant \$** — 7B fwd over whole corpus, once | `poc-distill/teacher-outputs/topk-logits` |
| 4 student sweep (×2) | GPU 80GB bf16 | distil hi + lo, reuse Step 3 | `ckpt/student-1b-attn-{hi,lo}` |
| 5 post-train winner | GPU | instruct→reasoning→grpo | `ckpt/sft-*`, `ckpt/grpo-*` |
| 6 eval | **Mac/MLX** | long-context + OLMES | `runs/eval/` |

Whole run is **order \$100s** on A100/H100 (issue #141). Bench Step 3 before committing.

## Gotchas (carried from Path A)

- **MOHAWK stage-1 O(L²) OOM** — small micro-batch + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  (see Step 4). Tightest constraint of the whole run at seq 8192.
- **Positional alignment** — teacher outputs are aligned to the exact split passed; corpus seq_len
  **must** equal the manifest's 8192. Re-precompute if the split changes.
- **HF_TOKEN required** for the gated 7B teacher (Steps 3–4). Unlike the poc-qwen training run, which
  needed none (it read packed shards, no tokenizer).
- **bf16 needs Ampere+** (A100/H100 80GB). T4/L4 only suffice for a cheap fp16/fp32 dry-run.
- **`s3fs==2026.2.0` pin** — must match fsspec (`datasets` caps `fsspec<=2026.2.0`); a bare
  `pip install s3fs` upgrades fsspec and breaks `datasets`.
- **Checkpoint to R2 frequently** — RunPod pods/volumes are ephemeral; sync mid-run, not just at the
  end. `distill.py --resume` resumes the furthest-progressed stage.
- **datatrove in `.venv-dt`** (py3.11), never the main py3.14 `.venv` — only needed for a full
  corpus rebuild (Step 1, option B).

## Open items

- Data-sourcing decisions for the corpus: core sources #70, supplements #71 (option A skips these by
  reusing the poc-qwen corpus).
- reasoning-SFT training driver (#101) and tool-SFT code (#102) — see Step 5 caveats.
- GRPO on CUDA for a 1B model (#103) — `rlvr.py` is MLX-only today.
- Local-hardware headline metric vs a same-size transformer (#104).
