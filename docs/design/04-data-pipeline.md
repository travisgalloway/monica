# Data pipeline

[← Index](README.md)

The pipeline is four backend-free stages — **download → tokenize → pack → split** —
plus a loader, all under [`src/data/`](../../src/data/). It yields numpy arrays; the
backend converts them inside `forward`, keeping the [seam](01-architecture-seam.md)
intact.

## uint16 packing

Token ids are packed as a flat `uint16` array on disk. From
[`src/data/pack.py`](../../src/data/pack.py):

> uint16 because the OLMo vocab (~50k) fits under 65536 — confirm the actual vocab
> before committing (see MambaConfig.validate / tokenize.load_olmo_tokenizer). The
> loader reads this format directly at train time; no JSON parsing during training.

A sidecar `<name>.meta.json` records dtype and token count. The packer validates the
**original** ids before casting:

> Validate the ORIGINAL values: casting to uint16 first would silently wrap
> out-of-range / negative ids and defeat the check.

The same bound is enforced structurally by `MambaConfig.validate()`, which raises if
`vocab_size >= 65536`. This is *why* the tokenizer choice below matters.

## The tokenizer

From [`src/data/tokenize.py`](../../src/data/tokenize.py):

> Use the OLMo tokenizer (via HuggingFace) so the vocab matches AI2's, enabling later
> comparison.
>
> CONFIRMED (issue #4): `allenai/OLMo-7B-hf` is reachable on the HF Hub with
> vocab_size=50280 (eos_token_id=50279), which fits the uint16 packing requirement
> (< 65536). `allenai/OLMo-2-1124-7B` is deliberately NOT a candidate: its vocab is
> 100278 (> 65536) and can never satisfy the uint16 constraint enforced by
> `MambaConfig.validate()`.

So the tokenizer and the storage format are a linked decision: OLMo-7B-hf is chosen
partly *because* its vocab fits uint16, and OLMo-2 is rejected *because* it doesn't.

A byte-level fallback tokenizer exists, but only for plumbing tests:

> A byte-level fallback tokenizer is provided ONLY for offline pipeline testing; it
> is not vocab-compatible with OLMo and must not be used for a real run.

(The toy config uses `vocab_size: 256` precisely because it runs on this byte
fallback for the offline smoke path.)

## Disjoint validation split

Held-out perplexity is the POC's success metric ([eval](06-smoke-gate-and-eval.md)),
so the val shard must not leak into training. From
[`src/data/split.py`](../../src/data/split.py):

> The validation shard MUST NOT overlap the training stream — held-out perplexity on
> it is the primary pipeline-health signal (see eval/val_loss). We split by a single
> contiguous cut so train and val token ranges are provably disjoint.

`split_packed` cuts `val_tokens` contiguous tokens off one end; the two ranges are
disjoint by construction (no sampling, no overlap to reason about).

## The loader

From [`src/data/loader.py`](../../src/data/loader.py):

> Design: mmap + a chunk index, shuffled at the chunk level. No per-step parsing — a
> slow loader bottlenecks a small model. Each item is a contiguous `seq_len + 1`
> window so the training loop can form (input, target) by a one-token shift.
>
> This module is backend-free: it yields numpy arrays. The backend converts them to
> its own array type inside `forward`.

Key choices:

- **mmap, not load-into-RAM** — scales to multi-GB packed files.
- **Chunk-level shuffle** — cheap; full per-token shuffle is unnecessary and slow.
- **`seq_len + 1` windows** — the extra token is the shift, so `(inputs, targets) =
  (arr[:, :-1], arr[:, 1:])` with no separate target file.
- **No per-step parsing** — for a small model, a slow loader is the bottleneck; the
  on-disk format is read directly.

For exact-resume testing, the [smoke gate](06-smoke-gate-and-eval.md) pre-materializes
a fixed batch list (shuffle off) so the batch at global step *s* is identical across
runs.

## Related

- [Architecture: the hardware seam](01-architecture-seam.md) — why the loader yields numpy.
- [Smoke gate & eval](06-smoke-gate-and-eval.md) — how the val shard is consumed.
- [Configs & locked decisions](07-configs-and-decisions.md) — vocab_size per config.
