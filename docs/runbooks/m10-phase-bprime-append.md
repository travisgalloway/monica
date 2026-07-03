# M10 Phase B′ — append-only teacher precompute for the corpus extension (#177)

Ready-to-fire runbook for **#177**: extend the frozen 566 GB Qwen3-4B-Thinking teacher top-k cache
to cover the multi-domain corpus extension (#176, built 2026-07-03) **without re-precomputing the
~230,318 unchanged FineWeb chunks**. All machinery is merged
(`scripts/verify_teacher_alignment.py`, `scripts/append_new_chunks.py`, `scripts/precompute_teacher.py`)
— this is a **run**, not a build. Written so a fresh session (or a clean pod clone) can fire it with
no prior context.

Related: [`../path-b-run.md`](../path-b-run.md) (the authoritative Path B runbook — see its update
banner; this doc replaces its Step 3 for the *extension* leg), [`m10-pod-chain.md`](m10-pod-chain.md)
(precompute + sweep staging, steps 3–4 — this runbook's output feeds its Step 3),
[`../infrastructure.md`](../infrastructure.md) (generic R2 + RunPod flow).

## State (as of 2026-07-03)

- **Base corpus + base teacher precompute: DONE (2026-07-02).** 4× A100-SXM4-80GB pods ran the
  4B-teacher top-k forward (k=50, seq_len=8192) over the base 1.897B-token FineWeb-derived corpus
  and merged: **230,318 train chunks (~1.887B rows), 305 val chunks**, at
  `s3://monica-training/poc-distill/teacher-outputs/topk-logits-merged/` (IDX 377 GB + VALS 189 GB
  ≈ 566 GB total). Pod terminated.
- **Corpus extension: DONE (2026-07-03, #176).** ~1.9B new pretrain tokens blended across five
  domains (code, math, docs, conversation, reasoning) pushed to R2 alongside the base corpus
  (`s3://monica-training/poc-distill/corpus/tokenized/qwen3-8k`) — confirm the exact extension
  shard prefix from `provenance.json` written by #176's run (not hard-coded here since it wasn't
  captured in this session).
- **This runbook (#177): not yet run.** It is the remaining step before the two-layout sweep (#81)
  can train against the *full* extended ~3.8B blend.

## Why append, not re-precompute (enforced by code, not a guess)

The 566 GB cache is **positionally bound** to FineWeb's `train.bin` — chunk `i` reads row `i×8192`
(`teacher_outputs.py`'s `DistillLoader` asserts `n_chunks == 230318`). The extension is packed as
new shards **after** the frozen FineWeb prefix, so every FineWeb chunk keeps its position and its
already-computed logits — only the new chunks need a fresh teacher forward. Three guards abort into
a full re-precompute if violated:

1. The regenerated FineWeb `train.bin` must still yield exactly `n_chunks == 230318` (no silent
   corpus drift).
2. The frozen shard-0 must show `n_chunks == 230318` with no stray `start_chunk` offset before
   merging.
3. `verify_teacher_alignment.py` — a **live** teacher forward over probe chunks (0 / 100000 /
   230317) must show **≥0.99 top-k index-set agreement** with the cached logits, and
   `effective_vocab_size == 151669`.

**If the alignment gate fails**, the fallback is a full re-precompute over the combined corpus
(real $/time) — the gate exists to catch this cheaply (one teacher forward over 3 probe chunks)
*before* committing to the append.

## Prereqs (pod)

- **GPU: Ampere+ 80 GB (A100/H100)** — the alignment gate and the new-chunk precompute both need a
  live CUDA teacher forward; bf16 needs Ampere+.
- **Volume: ≥1.2 TB.** Combined footprint estimate ≈ 566 GB (existing) + ~475 GB (new chunks) ≈
  **~1.05 TB** — recompute exactly once the real new-chunk count is known from #176's
  `provenance.json`, and size the volume with headroom (the merge also needs scratch space for the
  regenerated/trimmed/combined `train.bin`).

```bash
git clone https://github.com/travisgalloway/monica && cd monica
pip install -e ".[dev,data,cuda-fast]"      # fused Mamba Triton scan + causal-conv1d (#40)
pip install "s3fs==2026.2.0"                 # pin to fsspec; a bare install upgrades fsspec and breaks datasets
set -a; . ./.env; set +a                     # AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL_S3, R2_BUCKET=monica-training
#   HF_TOKEN optional — Qwen/Qwen3-4B-Thinking-2507 is Apache-2.0 and openly downloadable.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Step 1 — pull inputs from R2

```bash
# regenerated FineWeb corpus (the ORIGINAL base corpus — must still tokenize to 230318 chunks):
python -m src.data.r2_sync down s3://monica-training/poc-distill/corpus/tokenized/qwen3-8k /vol/fineweb-shards

# the new-source corpus extension (#176) — confirm the exact prefix from #176's provenance.json:
python -m src.data.r2_sync down s3://monica-training/poc-distill/corpus/tokenized/<extension-prefix> /vol/extension-shards

# the frozen base teacher cache (shard-0 for the merge):
python -m src.data.r2_sync down s3://monica-training/poc-distill/teacher-outputs/topk-logits-merged /vol/topk-logits-merged
```

## Step 2 — the alignment gate (run FIRST, abort on FAIL)

`verify_teacher_alignment.py --data` takes an actual regenerated `train.bin` **file**, not the
shard dir — regenerate it first via `src.data.split` (the same call `path-b-run.md`'s Step 3 uses
for the base corpus):

```bash
python -m src.data.split --shards /vol/fineweb-shards --out /vol/fineweb-split --val-tokens 10000000
#   -> /vol/fineweb-split/{train.bin,val.bin} — MUST regenerate to exactly n_chunks == 230318;
#   if it doesn't, the corpus has drifted and the append is unsafe (STOP here, do not proceed).

python scripts/verify_teacher_alignment.py \
    --data /vol/fineweb-split/train.bin \
    --topk-dir /vol/topk-logits-merged \
    --split train --probe-chunks 0,100000,230317 \
    --backend cuda
#   --min-agreement 0.99 (default) — nonzero exit => STOP, fall back to a full re-precompute
#   over the combined corpus (see "Why append, not re-precompute" above). Do not proceed to
#   Step 3 on a FAIL.
```

## Step 3 — precompute teacher top-k for ONLY the new chunks

**Positional alignment warning:** Step 4's `append_new_chunks.py` builds its own flat `train.bin`
from `--extension-shards` internally (`build_flat_new_train`), concatenating **every** shard's
tokens with **no val holdout**. The split you precompute over here must produce the *exact same*
token stream in the *exact same* order, or shard1-local won't line up with the combined corpus.
Use `--val-tokens 0` — **not** the `10000000` used for the base FineWeb split — so nothing is held
out (the frozen 305 val chunks already come from the base corpus; the extension contributes train
data only):

```bash
python -m src.data.split --shards /vol/extension-shards --out /vol/extension-split --val-tokens 0
#   --val-tokens 0 is deliberate here (unlike the base-corpus split in Step 2, which holds out
#   10,000,000) — split_shards() reads shards in manifest order and holds out the tail
#   `val_tokens` into val.bin; build_flat_new_train() reads the SAME manifest order but flattens
#   ALL tokens with no holdout. A nonzero --val-tokens here would silently drop the tail chunks
#   from shard1-new relative to what Step 4 reconstructs, misaligning every chunk after the cut.

python scripts/precompute_teacher.py \
    --manifest config/manifests/student-1b-attn-hi.yaml \
    --data /vol/extension-split --splits train \
    --backend cuda --k 50 --batch-size 8 \
    --out /vol/teacher-outputs/shard1-new
#   the manifest here only supplies conversion_teacher + seq_len (identical across hi/lo) —
#   same reason one precompute serves both layouts in the base run.
```

`/vol/teacher-outputs/shard1-new` becomes `--shard1-local` in Step 4.

## Step 4 — append-merge

```bash
python scripts/append_new_chunks.py \
    --fineweb-shards /vol/fineweb-shards \
    --extension-shards /vol/extension-shards \
    --work-dir /vol/append-work \
    --topk-dir /vol/topk-logits-merged \
    --shard1-local /vol/teacher-outputs/shard1-new \
    --push s3://monica-training/poc-distill/teacher-outputs/topk-logits-ext-merged \
    --val-tokens 10000000
```

This single script performs the full recipe: regenerates + trims FineWeb's `train.bin` to the
frozen 230,318-chunk prefix (via **copy-prefix-then-rename**, not `os.truncate` — RunPod's `/vol`
MooseFS mount silently no-ops truncate, corrupting resumed writes), builds a flat `train.bin` from
the extension shards, concatenates the two into the combined corpus, and **stream-merges** the
frozen shard-0 (from R2) with the new shard-1 straight to the new R2 prefix — avoiding
materializing the ~1.1 TB combined set locally. It self-verifies at the end that `DistillLoader`
opens the combined `(train.bin, merged teacher-outputs)` pair.

## Step 5 — finalize

- Flip the local PHASE marker → `B′:append-done` (the ops-state convention recorded in
  `~/.claude/monica-runpod-ops/` from prior pod runs — reuse it, don't invent a new mechanism).
- **Repoint the sweep at the extended cache.** The two sweep manifests
  (`config/manifests/student-1b-attn-{hi,lo}.yaml`) currently pin
  `teacher_outputs: poc-distill/teacher-outputs/topk-logits` (the base-only cache). Two ways to
  point the Step 4 sweep (in `m10-pod-chain.md`) at the extended one instead:
  - **Recommended — CLI override, keeps manifests stable:** pass
    `--teacher-outputs s3://monica-training/poc-distill/teacher-outputs/topk-logits-ext-merged`
    (or the local synced copy) to `scripts/distill.py` instead of relying on the manifest field.
  - **Alternative:** edit both manifests' `teacher_outputs:` field to the new `-ext-merged` prefix.

## Sizing

| Stage | Cost driver | Output |
|---|---|---|
| Step 2 (alignment gate) | 1 live teacher forward × 3 probe chunks — cheap, the point of the gate | pass/fail |
| Step 3 (new-chunk precompute) | 4B forward over ~475 GB-equivalent new tokens — the real $ of this runbook | `/vol/teacher-outputs/shard1-new` |
| Step 4 (append-merge) | I/O-bound (streaming merge), not compute-bound | `topk-logits-ext-merged` on R2, ~1.05 TB |

## Gotchas

- **Positional alignment is everything** — if the regenerated FineWeb `train.bin` doesn't yield
  exactly 230,318 chunks, or shard-0 shows a stray `start_chunk` offset, the append is unsafe;
  the script's guards catch this, but don't skip Step 2.
- **`os.truncate` silently no-ops on RunPod's `/vol` MooseFS mount** — `append_new_chunks.py`
  already uses the safe copy-prefix-then-rename trim; don't hand-roll a truncate-based variant.
- **`s3fs==2026.2.0` pin** — must match fsspec (`datasets` caps `fsspec<=2026.2.0`); a bare
  `pip install s3fs` upgrades fsspec and breaks `datasets`.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** carries forward into the Step 4 sweep
  (MOHAWK stage-1 OOM risk at seq 8192) — set it once at the top of the session and keep it set.

## Exit criteria

- [ ] Merged extended teacher-outputs cache on R2 at the new `-ext-merged` prefix.
- [ ] PHASE marker flipped → `B′:append-done`.
- [ ] Unblocks the two-layout student sweep (#81), now trainable against the full extended ~3.8B
      blend — proceed to [`m10-pod-chain.md`](m10-pod-chain.md) Step 4.

## After this: the rest of the chain

This runbook covers only the append leg. Downstream:

- **Sweep (#81):** [`m10-pod-chain.md`](m10-pod-chain.md) Step 4 — repoint `--teacher-outputs` as
  above, then run the hi/lo sweep as documented there.
- **Post-train the winner (#101 instruct SFT, #103 GRPO):** see
  [`../path-b-run.md`](../path-b-run.md) Step 5. Use `--config config/student-1b-attn-hi.yaml` if
  the hi layout wins, `--config config/student-1b-attn-lo.yaml` if lo wins — both are resolved
  MambaConfig YAMLs (not manifests) for `scripts/sft.py`/`scripts/rlvr.py`'s `--config` flag.
  **Known gap, not fixed here:** `scripts/rlvr.py` (#103) wires only math/exact-match rewards and
  takes **no `--backend` flag** (MLX-only, via the serving recurrence) — the code-sandbox
  `CodeVerifier` mentioned in #103's scope is not wired into the driver. A ~1B GRPO run on Mac/MLX
  is slow; resolve the CUDA-backend gap via #103 before the RL pass, or accept the slower MLX run.
- **Headline eval (#104):** `docs/path-b-run.md` Step 6, plus `scripts/bench_context.py` (added in
  PR #179) for the throughput-vs-context-length measurement once the winner is post-trained.
