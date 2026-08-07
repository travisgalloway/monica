"""Shared plumbing for the M12 code eval suite (#221).

ABOVE THE SEAM. Pure numpy + stdlib — never imports `mlx` or `torch`. Like
`src/eval/fim_eval.py` and `src/eval/val_loss.py` it touches a model only through
`ModelInterface.forward`, with a `to_numpy` converter supplied at the seam.

Why this module exists
----------------------
#221 ships several probes (cross-file symbol recall, RULER-over-code needle, FIM by
prefix distance, per-domain BPB, external-suite adapters). The issue's acceptance bar is
**deterministic, per-instance records** — so every probe has to emit the *same* record
shape, written the *same* way, or "run it twice and diff the JSONL" stops being a usable
check. This module owns that shape, the writer, the aggregator, and the one batched
span-scorer they all share.

The record schema
-----------------
``{suite, id, bucket, distance, n_scored_tokens, ce_nats, token_accuracy, exact_match,
   rank_top1, mrr, meta}``

`ce_nats` is the **mean** cross-entropy over the scored span (nats/token) so it is
comparable across spans of different lengths; `n_scored_tokens` is the weight used when
aggregating. `rank_top1`/`mrr` are `None` for probes that have no candidate set.

Scoring is teacher-forced throughout — no generation, no pass@1 gating. At the small rung
a generative gate is noise (the LSP-in-the-loop assessment: functional pass@1 flat at
0.503 while clean-rate moved 0.887 -> 0.962), which is why the issue asks for loss- and
rank-based probes instead.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .fim_eval import DEFAULT_BUCKETS, bucket_of
from .val_loss import masked_cross_entropy, perplexity

#: Column order of one per-instance record. Every suite emits exactly these keys, so a
#: transcript can be read back without knowing which probe produced a line.
RECORD_FIELDS: Tuple[str, ...] = (
    "suite", "id", "bucket", "distance", "n_scored_tokens",
    "ce_nats", "token_accuracy", "exact_match", "rank_top1", "mrr", "meta",
)


def make_record(*, suite: str, id: str, bucket: str, distance: int,
                n_scored_tokens: int = 0,
                ce_nats: Optional[float] = None,
                token_accuracy: Optional[float] = None,
                exact_match: Optional[float] = None,
                rank_top1: Optional[bool] = None,
                mrr: Optional[float] = None,
                meta: Optional[dict] = None) -> dict:
    """One per-instance record with every field present (missing metrics are explicit
    `None`, never absent keys — an absent key reads as 'not measured' downstream in
    exactly the same way as a zero, which is the failure shape this suite avoids)."""
    return {
        "suite": suite,
        "id": id,
        "bucket": bucket,
        "distance": int(distance),
        "n_scored_tokens": int(n_scored_tokens),
        "ce_nats": None if ce_nats is None else float(ce_nats),
        "token_accuracy": None if token_accuracy is None else float(token_accuracy),
        "exact_match": None if exact_match is None else float(exact_match),
        "rank_top1": None if rank_top1 is None else bool(rank_top1),
        "mrr": None if mrr is None else float(mrr),
        "meta": dict(meta or {}),
    }


# --------------------------------------------------------------------------------------- #
# Deterministic IO
# --------------------------------------------------------------------------------------- #

def dumps_canonical(obj) -> str:
    """One canonical JSON line: sorted keys, no incidental whitespace.

    Byte-reproducibility is the point — `sort_keys=True` removes dict-insertion order from
    the output, so two runs that computed the same values produce the same bytes.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON-serializable: {type(o).__name__}")


def write_jsonl(records: Iterable[dict], path) -> int:
    """Write records as canonical JSONL **in the order given**. Returns the line count.

    Callers must pass an already-deterministic order (this module never sorts for you —
    sorting here would silently paper over a probe whose own iteration order is
    nondeterministic, which is exactly what the twice-and-diff check is meant to catch).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(dumps_canonical(rec) + "\n")
            n += 1
    return n


def read_jsonl(path) -> List[dict]:
    """Read a JSONL file back into a list of dicts (blank lines skipped)."""
    out: List[dict] = []
    with open(Path(path), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path) -> str:
    """Hex sha256 of a file, streamed. This is what turns "the blocklist is wired" into a
    verifiable claim rather than an assertion."""
    h = hashlib.sha256()
    with open(Path(path), "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------------------- #

def bucket_for_distance(distance: int,
                        buckets: Sequence[Tuple[str, int, Optional[int]]] = DEFAULT_BUCKETS
                        ) -> str:
    """Bucket name for a token distance — `fim_eval.bucket_of`, re-exported so every suite
    in #221 uses the same edges as #215's FIM probe."""
    return bucket_of(int(distance), buckets)


def _mean_of(rows: Sequence[dict], key: str) -> Optional[float]:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def summarize_records(records: Sequence[dict]) -> dict:
    """Token-weighted aggregate over records with at least one scored token.

    Mirrors `fim_eval.evaluate_fim`'s aggregate: CE and token accuracy are weighted by
    `n_scored_tokens` so a long span does not count the same as a one-token span, while
    exact-match / rank / MRR are per-instance means.
    """
    rows = [r for r in records if r.get("n_scored_tokens")]
    if not rows:
        raise ValueError(
            "summarize_records(): nothing scored — an empty eval reports a perfect model, "
            "so this is a failure, not a zero row"
        )
    n_tokens = sum(int(r["n_scored_tokens"]) for r in rows)
    ce = sum(float(r["ce_nats"]) * int(r["n_scored_tokens"]) for r in rows) / n_tokens
    acc_rows = [r for r in rows if r.get("token_accuracy") is not None]
    acc = (sum(float(r["token_accuracy"]) * int(r["n_scored_tokens"]) for r in acc_rows)
           / sum(int(r["n_scored_tokens"]) for r in acc_rows)) if acc_rows else None
    return {
        "ce": float(ce),
        "perplexity": perplexity(float(ce)),
        "token_accuracy": None if acc is None else float(acc),
        "exact_match_rate": _mean_of(rows, "exact_match"),
        "rank_top1_rate": (float(np.mean([bool(r["rank_top1"]) for r in rows
                                          if r.get("rank_top1") is not None]))
                           if any(r.get("rank_top1") is not None for r in rows) else None),
        "mrr": _mean_of(rows, "mrr"),
        "n_instances": len(rows),
        "n_tokens": int(n_tokens),
    }


def summarize_bucketed(records: Sequence[dict],
                       bucket_names: Optional[Sequence[str]] = None) -> dict:
    """`{"by_bucket": {name: agg | None}, "overall": agg}` grouped on each record's own
    `bucket` field.

    An empty bucket is `None`, **not** a zero row (`long_context.py`'s convention) — a zero
    CE row would read as a perfect model on a bucket that was never measured. If nothing at
    all scored, `summarize_records` raises.

    `bucket_names` fixes the reported order and forces empty buckets to appear (so a grid
    cell that produced no instances is visible as `None` rather than silently missing).
    Buckets present in the records but absent from `bucket_names` are appended in sorted
    order — a bucket is never dropped from the report.
    """
    names: List[str] = list(bucket_names or [])
    for r in records:
        if r.get("bucket") is not None and r["bucket"] not in names:
            names.append(str(r["bucket"]))
    if bucket_names is not None:
        head = list(bucket_names)
        tail = sorted(n for n in names if n not in head)
        names = head + tail

    by_bucket: Dict[str, Optional[dict]] = {}
    for name in names:
        rows = [r for r in records if r.get("bucket") == name and r.get("n_scored_tokens")]
        by_bucket[name] = summarize_records(rows) if rows else None
    return {"by_bucket": by_bucket, "overall": summarize_records(records)}


def format_bucket_table(title: str, results: dict) -> str:
    """One line per bucket + an overall line, matching `fim_eval.format_fim_table`."""
    def _fmt(v, spec=".4f"):
        return "  n/a " if v is None else format(v, spec)

    lines = [f"{title}:"]
    for name, agg in results["by_bucket"].items():
        if agg is None:
            lines.append(f"  {name:<16} (no instances in this bucket)")
            continue
        lines.append(
            f"  {name:<16} ce={_fmt(agg['ce'])}  ppl={_fmt(agg['perplexity'])}  "
            f"tok_acc={_fmt(agg['token_accuracy'])}  exact={_fmt(agg['exact_match_rate'])}  "
            f"top1={_fmt(agg['rank_top1_rate'])}  mrr={_fmt(agg['mrr'])}  "
            f"({agg['n_instances']} inst, {agg['n_tokens']} tok)")
    o = results["overall"]
    lines.append(
        f"  {'overall':<16} ce={_fmt(o['ce'])}  ppl={_fmt(o['perplexity'])}  "
        f"tok_acc={_fmt(o['token_accuracy'])}  exact={_fmt(o['exact_match_rate'])}  "
        f"top1={_fmt(o['rank_top1_rate'])}  mrr={_fmt(o['mrr'])}  "
        f"({o['n_instances']} inst, {o['n_tokens']} tok)")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------- #
# The shared batched span scorer
# --------------------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ScoreRow:
    """One teacher-forced row: score `tokens[span_start : span_start + span_len]`.

    `span_start >= 1` is required — target index `j` holds `tokens[j+1]`, so a span that
    starts at index 0 has no input position that predicts its first token.
    """

    tokens: np.ndarray
    span_start: int
    span_len: int

    def __post_init__(self) -> None:
        if self.span_start < 1:
            raise ValueError(f"span_start must be >= 1, got {self.span_start}")
        if self.span_len < 1:
            raise ValueError(f"span_len must be >= 1, got {self.span_len}")
        end = self.span_start + self.span_len
        if end > int(np.asarray(self.tokens).size):
            raise ValueError(f"span [{self.span_start}, {end}) runs past the {self.tokens.size}-"
                             "token row")


def score_rows(model, rows: Sequence[ScoreRow], *, pad_id: int = 0, batch_size: int = 8,
               to_numpy=np.asarray) -> List[dict]:
    """Teacher-forced scores for `rows`, one `forward` per batch, in input order.

    Right-padding is safe **because the model is causal**: a position can only attend to
    (or accumulate state from) positions at or before itself, so tokens appended after a
    row's own end cannot influence any of its scored positions. That is what makes a short
    row's score identical whether it is scored alone or alongside a longer one — the
    property `test_padding_does_not_change_a_short_instances_score` pins down.

    Each result is `{total_ce_nats, ce_nats, n_scored_tokens, token_accuracy, exact_match}`
    where `ce_nats` is the per-token mean and `total_ce_nats` the sum (the sum is what
    candidate ranking compares — see `code_recall`).
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    out: List[dict] = []
    for start in range(0, len(rows), batch_size):
        batch = list(rows[start:start + batch_size])
        max_len = max(int(np.asarray(r.tokens).size) for r in batch)
        padded = np.full((len(batch), max_len), int(pad_id), dtype=np.int64)
        for i, r in enumerate(batch):
            ids = np.asarray(r.tokens, dtype=np.int64).reshape(-1)
            padded[i, :ids.size] = ids

        inputs = padded[:, :-1]
        targets = padded[:, 1:]
        mask = np.zeros_like(inputs, dtype=np.float64)
        for i, r in enumerate(batch):
            mask[i, r.span_start - 1:r.span_start - 1 + r.span_len] = 1.0

        logits = np.asarray(to_numpy(model.forward(inputs)))
        for i, r in enumerate(batch):
            sel = mask[i] > 0
            n = int(sel.sum())
            ce = masked_cross_entropy(logits[i:i + 1], targets[i:i + 1], mask[i:i + 1])
            pred = np.argmax(logits[i][sel], axis=-1)
            correct = pred == targets[i][sel]
            out.append({
                "total_ce_nats": float(ce) * n,
                "ce_nats": float(ce),
                "n_scored_tokens": n,
                "token_accuracy": float(correct.mean()),
                "exact_match": float(bool(correct.all())),
            })
    return out


# --------------------------------------------------------------------------------------- #
# A deterministic offline stand-in for a trained model
# --------------------------------------------------------------------------------------- #

class StubCausalModel:
    """A pure-numpy causal fake: logits at position `t` depend ONLY on `inputs[b, t]`.

    There is **no trained MHM checkpoint yet** (#200/#222 are downstream of this issue), so
    every acceptance check here is a fixture/determinism check rather than a quality number.
    This model makes the whole suite runnable offline with no backend and no checkpoint,
    and it is causal by construction — which is the property the right-padding batching in
    `score_rows` relies on, so it also serves as the padding-safety test's model.

    Its numbers are meaningless as *quality*. Never report them as measured recall.
    """

    def __init__(self, vocab_size: int = 256, seed: int = 0,
                 n_layers: int = 4, attn_every: int = 2):
        self.vocab_size = int(vocab_size)

        class _Cfg:
            pass

        self.config = _Cfg()
        self.config.n_layers = n_layers
        self.config.attn_every = attn_every
        self._table = (np.random.default_rng(seed)
                       .standard_normal((self.vocab_size, self.vocab_size))
                       .astype(np.float32))

    def forward(self, inputs):
        return self._table[np.asarray(inputs) % self.vocab_size]


# --------------------------------------------------------------------------------------- #
# Fixture IO
# --------------------------------------------------------------------------------------- #

def make_byte_encoder():
    """`ByteTokenizer.encode` — the offline fixture encoder (`vocab_size=256`, one id per
    UTF-8 byte). Every text-consuming probe in this suite takes an injected encoder because
    Python has no code tokenizer any more (retired with #245)."""
    from ..data.tokenize import ByteTokenizer

    return ByteTokenizer().encode


def make_swift_encoder(tokenizer_json, binary: str = "monica-tokenize"):
    """Shell out to the native Swift tokenizer: `monica-tokenize encode --tokenizer X --json`.

    The real encoder for anything that has to match the packed training corpus. One
    subprocess per document (the CLI reads one text from stdin), so this is for eval-set
    construction, not for a hot loop. Raises `FileNotFoundError` with the fix when the
    binary is not on PATH — a missing tokenizer must never silently degrade to bytes, since
    that would report distances in the wrong unit.
    """
    import subprocess

    tokenizer_json = str(tokenizer_json)

    def encode(text: str) -> List[int]:
        try:
            proc = subprocess.run(
                [binary, "encode", "--tokenizer", tokenizer_json, "--json"],
                input=text, capture_output=True, text=True, check=True)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"{binary!r} not found on PATH — build it with "
                "`swift build -c release --package-path swift` and add "
                "`swift/.build/release` to PATH, or use the byte encoder for offline "
                "fixture runs") from e
        return list(json.loads(proc.stdout))

    return encode


def load_code_files(path) -> List[dict]:
    """Load a `{path, text}` JSONL fixture (the shape used by both the code-recall repo
    fixture and the needle haystack), sorted by `path` for a deterministic order."""
    rows = read_jsonl(path)
    for r in rows:
        if "path" not in r or "text" not in r:
            raise ValueError(f"{path}: every row needs 'path' and 'text', got {sorted(r)}")
    return sorted(rows, key=lambda r: r["path"])
