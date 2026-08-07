"""Distance-bucketed fill-in-the-middle (FIM) evaluation (#215).

ABOVE THE SEAM. Pure numpy — never imports `mlx` or `torch`. It touches a model only through
`ModelInterface.forward`, with a `to_numpy` converter supplied at the seam (identity by default;
on MLX pass a converter), exactly like `src/eval/val_loss.py` and `src/eval/long_context.py`.

What it measures
----------------
A PSM stream is `[fim_prefix] prefix [fim_suffix] suffix [fim_middle] middle`, so the whole
suffix is consumed *before* the first middle token is predicted. On a mostly-SSM backbone that
recall has to survive the fixed-width state — which is precisely the weakness the hybrid's
attention layers exist to patch. Reporting one aggregate FIM loss would hide it, so every score
here is bucketed by distance (short / medium / long by default) and every example also carries
its own record.

Scoring is **teacher-forced and loss-based**, not generative: cross-entropy over the middle span
only, plus token accuracy and exact match. No pass@1 gating — at the small rung a generative gate
is noise (see #221).

The architectural consequence — configuration guidance, not code
----------------------------------------------------------------
Because the middle is generated after the suffix has scrolled past, **the final block should be
an attention block**: `n_layers % attn_every == 0`, from the `layer i is attention iff
(i+1) % attn_every == 0` rule at `src/model/blocks.py:192`. `config/toy-hybrid.yaml` satisfies it
(4 layers, `attn_every 2` -> `[Mamba, Attn, Mamba, Attn]`); `config/student-1b.yaml` (28 layers,
`attn_every 8` -> attention at 7/15/23, top block Mamba) does not, so a future MHM config must
not copy that shape. This is an M12 training-recipe constraint, deliberately NOT a
`MambaConfig.validate()` rule: that validator is shared with reserve configs that legitimately
violate it. `attention_after_suffix()` below is the advisory check — arithmetic over two ints,
no model import, and it never raises.

Where the FIM streams come from
-------------------------------
Insertion itself lives in the Swift `pack` path (`swift/Sources/MonicaTokenizer/FIM.swift`, #215)
because that is where document boundaries still exist. Python has no code tokenizer any more
(retired with #245), so this module consumes **already-tokenized** documents:
`documents_from_shards` recovers them from a Swift-packed shard dir via `src.data.shard`. Point
it at the *shard* dir, not a split dir — `src/data/split.py` drops the `.bounds` sidecars that
carry the doc starts.

Ownership: **#215 owns this module; #221 (code eval suite) extends it.** Hence the split between
example *construction* and *scoring*, the callable bucket key, and the per-instance record schema
— #221's "FIM-by-prefix-distance" probe should import these, not re-implement them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..eval.val_loss import masked_cross_entropy, perplexity

# Default bucket edges over a distance in tokens: (name, lo_inclusive, hi_exclusive_or_None).
DEFAULT_BUCKETS: Tuple[Tuple[str, int, Optional[int]], ...] = (
    ("short", 0, 256),
    ("medium", 256, 1024),
    ("long", 1024, None),
)


@dataclass(frozen=True)
class FIMSentinels:
    """Reserved special-token ids. These match the Swift tokenizer's fixed layout
    (`swift/Sources/monica-tokenize/main.swift`: `<|endoftext|>`, `<|fim_prefix|>`,
    `<|fim_middle|>`, `<|fim_suffix|>`, `<|fim_pad|>`, `<mask>` at ids 0..5)."""

    prefix: int = 1
    suffix: int = 3
    middle: int = 2
    eos: int = 0
    pad: int = 4

    @property
    def all_sentinels(self) -> Tuple[int, int, int]:
        return (self.prefix, self.suffix, self.middle)


@dataclass(frozen=True)
class FIMExample:
    """One teacher-forcing example: the full PSM stream plus the spans needed to score and
    bucket it. `middle_start` is the index of the `fim_middle` sentinel — which is also the
    input position whose prediction is the first middle token."""

    tokens: np.ndarray
    middle_start: int
    prefix_len: int
    middle_len: int
    suffix_len: int
    doc_index: int

    @property
    def recall_distance(self) -> int:
        """Tokens the model must carry state across before emitting the first middle token:
        the whole prefix + suffix + their two sentinels, i.e. `middle_start`. This is the
        SSM-relevant distance, as distinct from `prefix_len` (the literal 'prefix distance')."""
        return self.middle_start


def make_fim_example(doc_ids: Sequence[int], cut_a: int, cut_b: int,
                     sentinels: FIMSentinels = FIMSentinels(),
                     doc_index: int = 0) -> FIMExample:
    """Build a PSM stream from an already-tokenized document and two cut points `0 <= a <= b <= n`.

    Cuts are on **token** indices here, whereas the Swift pack-time transform cuts on UTF-8 byte
    offsets so that middles can begin mid-token. That difference is deliberate and harmless for
    evaluation: this module measures loss as a function of distance, and the trained-on
    distribution is the packed corpus's, not this constructor's. Use `documents_from_shards` when
    you want to score the real, Swift-inserted streams instead.
    """
    ids = np.asarray(doc_ids, dtype=np.int64).reshape(-1)
    n = int(ids.size)
    if not (0 <= cut_a <= cut_b <= n):
        raise ValueError(f"cut points must satisfy 0 <= a <= b <= {n}, got a={cut_a}, b={cut_b}")

    prefix, middle, suffix = ids[:cut_a], ids[cut_a:cut_b], ids[cut_b:]
    stream = np.concatenate([
        np.array([sentinels.prefix], dtype=np.int64), prefix,
        np.array([sentinels.suffix], dtype=np.int64), suffix,
        np.array([sentinels.middle], dtype=np.int64), middle,
    ])
    middle_start = 1 + int(prefix.size) + 1 + int(suffix.size)
    return FIMExample(tokens=stream, middle_start=middle_start,
                      prefix_len=int(prefix.size), middle_len=int(middle.size),
                      suffix_len=int(suffix.size), doc_index=doc_index)


def build_fim_examples(docs: Iterable[Sequence[int]], rng: np.random.Generator, *,
                       sentinels: FIMSentinels = FIMSentinels(),
                       n_per_doc: int = 1, min_middle: int = 1) -> List[FIMExample]:
    """Sample `n_per_doc` PSM examples per document with uniform two-point cuts.

    `rng` is an explicit `np.random.Generator` — there is no module-level RNG, so a given seed
    reproduces a given eval set exactly. Documents shorter than `min_middle` are skipped (they
    cannot yield a non-empty middle); an all-too-short corpus therefore yields `[]`, which
    `evaluate_fim` turns into a loud failure rather than a perfect score.
    """
    if min_middle < 1:
        raise ValueError(f"min_middle must be >= 1, got {min_middle}")
    examples: List[FIMExample] = []
    for doc_index, doc in enumerate(docs):
        ids = np.asarray(doc, dtype=np.int64).reshape(-1)
        n = int(ids.size)
        if n < min_middle:
            continue
        for _ in range(n_per_doc):
            cut_a = int(rng.integers(0, n - min_middle + 1))
            cut_b = int(rng.integers(cut_a + min_middle, n + 1))
            examples.append(make_fim_example(ids, cut_a, cut_b, sentinels, doc_index))
    return examples


def documents_from_shards(shard_dir, *, min_len: int = 1,
                          max_docs: Optional[int] = None) -> List[np.ndarray]:
    """Recover per-document token arrays from a Swift-packed (or `src/data/shard.py`-packed)
    shard directory, using the `.bounds` doc-start flags.

    Must point at the **shard** dir, not a split dir: `src/data/split.py` drops `.bounds`.
    The final document of each shard is truncated at the shard edge (packing is a flat token
    stream), which is why `min_len` exists — it drops the fragments too small to be worth scoring.
    """
    from ..data.shard import doc_start_offsets, open_shard, read_manifest

    shard_dir = Path(shard_dir)
    manifest = read_manifest(shard_dir)
    docs: List[np.ndarray] = []
    for shard in manifest["shards"]:
        toks, bnds = open_shard(shard_dir, shard["name"])
        starts = doc_start_offsets(bnds)
        for i, start in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else len(toks)
            ids = np.asarray(toks[start:end], dtype=np.int64)
            if ids.size >= min_len:
                docs.append(ids)
            if max_docs is not None and len(docs) >= max_docs:
                return docs
    return docs


def bucket_of(value: int,
              buckets: Sequence[Tuple[str, int, Optional[int]]] = DEFAULT_BUCKETS) -> str:
    """Name of the first bucket whose `[lo, hi)` contains `value` (`hi=None` = open-ended)."""
    for name, lo, hi in buckets:
        if value >= lo and (hi is None or value < hi):
            return name
    raise ValueError(f"value {value} falls in no bucket of {[b[0] for b in buckets]}")


def attention_after_suffix(n_layers: int, attn_every: Optional[int]) -> bool:
    """True when the model's **final** block is an attention block.

    From `src/model/blocks.py:192`, layer `i` is attention iff `attn_every and (i+1) % attn_every
    == 0`; the last layer is `i = n_layers - 1`, so the condition is `n_layers % attn_every == 0`.
    Advisory only (see the module docstring): PSM consumes the suffix before the middle, so a
    top attention block is what lets the middle re-read it rather than recall it from SSM state.
    """
    if not attn_every or attn_every <= 0 or n_layers <= 0:
        return False
    return n_layers % attn_every == 0


def _score_batch(model, batch: Sequence[FIMExample], sentinels: FIMSentinels,
                 to_numpy) -> List[dict]:
    """Teacher-forced scores for one padded batch. One `forward` call, per-example results."""
    max_len = max(int(ex.tokens.size) for ex in batch)
    padded = np.full((len(batch), max_len), sentinels.pad, dtype=np.int64)
    for i, ex in enumerate(batch):
        padded[i, :ex.tokens.size] = ex.tokens

    inputs = padded[:, :-1]
    targets = padded[:, 1:]
    # Mask the positions that predict middle tokens. Target index j holds stream[j+1], and the
    # middle span is stream[middle_start+1 : middle_start+1+middle_len], so j runs over
    # [middle_start, middle_start + middle_len).
    mask = np.zeros_like(inputs, dtype=np.float64)
    for i, ex in enumerate(batch):
        mask[i, ex.middle_start:ex.middle_start + ex.middle_len] = 1.0

    # Right-padding is safe *because the model is causal*: a position can only attend to (or
    # accumulate state from) positions at or before itself, so tokens appended after an example's
    # own end cannot influence any of its scored positions. That is what makes a short example's
    # score identical whether it is batched alone or alongside a longer one.
    logits = np.asarray(to_numpy(model.forward(inputs)))

    records: List[dict] = []
    for i, ex in enumerate(batch):
        sel = mask[i] > 0
        n_middle = int(sel.sum())
        if n_middle == 0:
            # An empty middle is a legitimate draw at pack time but carries no signal; record it
            # with an explicit None rather than folding a 0-token example into the mean.
            records.append({"ce_nats": None, "n_middle_tokens": 0,
                            "token_accuracy": None, "exact_match": None})
            continue
        ce = masked_cross_entropy(logits[i:i + 1], targets[i:i + 1], mask[i:i + 1])
        pred = np.argmax(logits[i][sel], axis=-1)
        correct = pred == targets[i][sel]
        records.append({
            "ce_nats": float(ce),
            "n_middle_tokens": n_middle,
            "token_accuracy": float(correct.mean()),
            "exact_match": float(bool(correct.all())),
        })
    return records


def evaluate_fim(model, examples: Sequence[FIMExample], *, batch_size: int = 8,
                 buckets: Sequence[Tuple[str, int, Optional[int]]] = DEFAULT_BUCKETS,
                 key: Callable[[FIMExample], int] = lambda ex: ex.prefix_len,
                 sentinels: FIMSentinels = FIMSentinels(),
                 to_numpy=np.asarray) -> dict:
    """Score `examples` and aggregate by bucket.

    `key` picks the bucketed distance — `ex.prefix_len` by default (#215's literal
    "prefix distance"). Pass `lambda ex: ex.recall_distance` for the SSM-relevant distance;
    every record carries both either way, so #221 needs no second module.

    Returns `{"records": [...], "by_bucket": {name: agg | None}, "overall": agg, "advisory": str
    | None}`. A bucket with no scored example is `None`, not a zero row (the `long_context.py`
    convention). If *nothing* scored at all this raises — a silently empty eval would report a
    perfect model (the `val_loss.evaluate` precedent).
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    records: List[dict] = []
    for start in range(0, len(examples), batch_size):
        batch = list(examples[start:start + batch_size])
        for ex, scored in zip(batch, _score_batch(model, batch, sentinels, to_numpy)):
            records.append({
                "doc_index": ex.doc_index,
                "bucket": bucket_of(int(key(ex)), buckets),
                "prefix_len": ex.prefix_len,
                "middle_len": ex.middle_len,
                "suffix_len": ex.suffix_len,
                "recall_distance": ex.recall_distance,
                **scored,
            })

    scored_records = [r for r in records if r["n_middle_tokens"] > 0]
    if not scored_records:
        raise ValueError(
            "evaluate_fim(): no middle tokens scored — the example set is empty or every "
            "example has an empty middle span"
        )

    by_bucket: dict = {}
    for name, _lo, _hi in buckets:
        rows = [r for r in scored_records if r["bucket"] == name]
        by_bucket[name] = aggregate_rows(rows) if rows else None

    return {
        "records": records,
        "by_bucket": by_bucket,
        "overall": aggregate_rows(scored_records),
        "advisory": _architecture_advisory(model),
    }


def aggregate_rows(rows: List[dict]) -> dict:
    """Token-weighted aggregate of scored FIM records (was `evaluate_fim`'s inner closure;
    lifted to module level so #221's multi-key re-aggregation can reuse it verbatim).

    CE and token accuracy are weighted by `n_middle_tokens` so a long middle does not count
    the same as a one-token middle; exact match is a per-example rate.
    """
    n_tokens = sum(r["n_middle_tokens"] for r in rows)
    ce = sum(r["ce_nats"] * r["n_middle_tokens"] for r in rows) / n_tokens
    acc = sum(r["token_accuracy"] * r["n_middle_tokens"] for r in rows) / n_tokens
    return {
        "ce": float(ce),
        "perplexity": perplexity(float(ce)),
        "token_accuracy": float(acc),
        "exact_match_rate": float(np.mean([r["exact_match"] for r in rows])),
        "n_examples": len(rows),
        "n_tokens": int(n_tokens),
    }


def aggregate_by(scored_records: Sequence[dict], *, key_field: str,
                 buckets: Sequence[Tuple[str, int, Optional[int]]] = DEFAULT_BUCKETS) -> dict:
    """Re-bucket already-scored records on `key_field` (`"prefix_len"` or
    `"recall_distance"`) and aggregate each bucket.

    Pure re-aggregation — no model call. An empty bucket is `None`, not a zero row.
    """
    rows_by_bucket: dict = {name: [] for name, _lo, _hi in buckets}
    for r in scored_records:
        if not r.get("n_middle_tokens"):
            continue
        rows_by_bucket[bucket_of(int(r[key_field]), buckets)].append(r)
    return {name: (aggregate_rows(rows) if rows else None)
            for name, rows in rows_by_bucket.items()}


def evaluate_fim_multi_key(model, examples: Sequence[FIMExample], *, batch_size: int = 8,
                           buckets: Sequence[Tuple[str, int, Optional[int]]] = DEFAULT_BUCKETS,
                           keys: Sequence[str] = ("prefix_len", "recall_distance"),
                           sentinels: FIMSentinels = FIMSentinels(),
                           to_numpy=np.asarray) -> dict:
    """#221's FIM probe: **both** keyings from a single forward pass.

    `evaluate_fim` buckets on one key at a time, so getting FIM-by-prefix-distance *and*
    FIM-by-recall-distance meant scoring the same examples twice — the forward pass is the
    expensive part and it is identical in both. Records already carry both distances
    (`fim_eval` scores once, buckets later), so this scores once and re-aggregates.

    Returns `{"records", "by_key": {key: {bucket: agg | None}}, "overall", "advisory"}`.
    Each `by_key[k]` is byte-identical to `evaluate_fim(..., key=lambda ex: getattr(ex, k))`'s
    `by_bucket`, which `tests/test_fim_eval.py` pins down.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    for key in keys:
        if key not in ("prefix_len", "recall_distance", "middle_len", "suffix_len"):
            raise ValueError(f"unknown FIM bucket key {key!r}")

    records: List[dict] = []
    for start in range(0, len(examples), batch_size):
        batch = list(examples[start:start + batch_size])
        for ex, scored in zip(batch, _score_batch(model, batch, sentinels, to_numpy)):
            records.append({
                "doc_index": ex.doc_index,
                "prefix_len": ex.prefix_len,
                "middle_len": ex.middle_len,
                "suffix_len": ex.suffix_len,
                "recall_distance": ex.recall_distance,
                **scored,
            })

    scored_records = [r for r in records if r["n_middle_tokens"] > 0]
    if not scored_records:
        raise ValueError(
            "evaluate_fim_multi_key(): no middle tokens scored — the example set is empty "
            "or every example has an empty middle span"
        )

    return {
        "records": records,
        "by_key": {key: aggregate_by(scored_records, key_field=key, buckets=buckets)
                   for key in keys},
        "overall": aggregate_rows(scored_records),
        "advisory": _architecture_advisory(model),
    }


def _architecture_advisory(model) -> Optional[str]:
    """Non-fatal note when the model's top block is not attention (see the module docstring).
    Returns None when the config doesn't expose the two fields — this must never raise, and an
    unreadable config is 'unknown', never 'fine'."""
    config = getattr(model, "config", None)
    n_layers = getattr(config, "n_layers", None)
    if not isinstance(n_layers, int):
        return None
    attn_every = getattr(config, "attn_every", None)
    if attention_after_suffix(n_layers, attn_every):
        return None
    return (f"final block is not attention (n_layers={n_layers}, attn_every={attn_every}); "
            "in PSM the suffix is consumed before the middle, so the top block should be "
            "attention — see src/eval/fim_eval.py and docs/design/13-code-model-moe.md")


def format_fim_table(results: dict) -> str:
    """One line per bucket + an overall line, for logging / posting to the tracker."""
    lines = ["FIM loss by distance bucket:"]
    for name, agg in results["by_bucket"].items():
        if agg is None:
            lines.append(f"  {name:<8} (no examples in this bucket)")
        else:
            lines.append(f"  {name:<8} ce={agg['ce']:.4f}  ppl={agg['perplexity']:.4f}  "
                         f"tok_acc={agg['token_accuracy']:.4f}  "
                         f"exact={agg['exact_match_rate']:.4f}  "
                         f"({agg['n_examples']} ex, {agg['n_tokens']} tok)")
    o = results["overall"]
    lines.append(f"  {'overall':<8} ce={o['ce']:.4f}  ppl={o['perplexity']:.4f}  "
                 f"tok_acc={o['token_accuracy']:.4f}  exact={o['exact_match_rate']:.4f}  "
                 f"({o['n_examples']} ex, {o['n_tokens']} tok)")
    if results.get("advisory"):
        lines.append(f"  advisory: {results['advisory']}")
    return "\n".join(lines)


def format_fim_multi_table(results: dict) -> str:
    """`evaluate_fim_multi_key` results as one `format_fim_table` block per key."""
    blocks = []
    for key, by_bucket in results["by_key"].items():
        blocks.append(format_fim_table({
            "by_bucket": by_bucket,
            "overall": results["overall"],
            # The advisory is a property of the model, not of the keying — print it once,
            # on the last block, rather than repeating it per key.
            "advisory": None,
        }).replace("FIM loss by distance bucket:", f"FIM loss by {key}:", 1))
    if results.get("advisory"):
        blocks.append(f"  advisory: {results['advisory']}")
    return "\n".join(blocks)
