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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

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

        raw_out = model.forward(inputs)
        mod_name = type(raw_out).__module__
        if hasattr(raw_out, "argmax") and mod_name.startswith(("mlx", "torch")):
            B, L = inputs.shape
            V = raw_out.shape[-1]
            if mod_name.startswith("mlx"):
                import mlx.core as mx
                import mlx.nn as nn
                pred_np = np.asarray(raw_out.argmax(axis=-1))
                t_arr = mx.array(targets).reshape(-1).astype(mx.int32)
                ce_arr = nn.losses.cross_entropy(raw_out.reshape(-1, V).astype(mx.float32),
                                                 t_arr, reduction="none").reshape(B, L)
                ce_np = np.asarray(ce_arr)
            else:
                import torch
                import torch.nn.functional as F
                with torch.no_grad():
                    pred_np = raw_out.argmax(dim=-1).cpu().numpy()
                    t_arr = torch.as_tensor(targets, dtype=torch.long, device=raw_out.device).reshape(-1)
                    ce_arr = F.cross_entropy(raw_out.reshape(-1, V).float(),
                                             t_arr, reduction="none").reshape(B, L)
                    ce_np = ce_arr.cpu().numpy()

            for i, r in enumerate(batch):
                sel = mask[i] > 0
                n = int(sel.sum())
                ce = float(ce_np[i][sel].mean()) if n > 0 else 0.0
                pred = pred_np[i][sel]
                correct = pred == targets[i][sel]
                out.append({
                    "total_ce_nats": float(ce_np[i][sel].sum()),
                    "ce_nats": float(ce),
                    "n_scored_tokens": n,
                    "token_accuracy": float(correct.mean()) if n > 0 else 0.0,
                    "exact_match": float(bool(correct.all())) if n > 0 else 0.0,
                })
        else:
            logits = np.asarray(to_numpy(raw_out))
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

# --------------------------------------------------------------------------------------- #
# In-context associative recall model (#359)
# --------------------------------------------------------------------------------------- #

class InContextRecallModel:
    """A causal model with variable-order in-context associative recall (induction heads).

    Simulates the attention layer's capacity to retrieve previously defined symbols
    from the KV cache.
    Logits at position `t` depend causally on `inputs[b, :t+1]`:
    1. Base token-transition table (like `StubCausalModel`).
    2. Associative induction retrieval: prefix contexts that appeared in the past
       `inputs[b, :t+1]` boost predictions for their observed next tokens.
    """

    def __init__(self, vocab_size: int = 256, boost: float = 15.0, seed: int = 0,
                 max_order: int = 3, n_layers: int = 4, attn_every: int = 2):
        self.vocab_size = int(vocab_size)
        self.boost = float(boost)
        self.max_order = int(max_order)

        class _Cfg:
            pass

        self.config = _Cfg()
        self.config.n_layers = n_layers
        self.config.attn_every = attn_every
        self._table = (np.random.default_rng(seed)
                       .standard_normal((self.vocab_size, self.vocab_size))
                       .astype(np.float32))

    def forward(self, inputs):
        inputs = np.asarray(inputs, dtype=np.int64)
        B, L = inputs.shape
        base = self._table[inputs % self.vocab_size]
        out = base.copy()
        for b in range(B):
            row = (inputs[b] % self.vocab_size).tolist()
            ngram_next: Dict[Tuple[int, ...], Dict[int, int]] = {}
            for t in range(L):
                for order in range(1, min(t + 2, self.max_order + 1)):
                    ctx = tuple(row[t - order + 1 : t + 1])
                    if ctx in ngram_next:
                        for nxt, cnt in ngram_next[ctx].items():
                            out[b, t, nxt] += self.boost * order * min(cnt, 2)
                if t > 0:
                    curr = row[t]
                    for order in range(1, min(t + 1, self.max_order + 1)):
                        ctx = tuple(row[t - order : t])
                        if ctx not in ngram_next:
                            ngram_next[ctx] = {}
                        ngram_next[ctx][curr] = ngram_next[ctx].get(curr, 0) + 1
        return out


# --------------------------------------------------------------------------------------- #
# Repository recall probe (topological vs random ordering) (#359)
# --------------------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RepoRecallInstance:
    """One cross-file repository recall query instance."""

    id: str
    definer: str
    consumer: str
    symbol: str
    consumer_prefix: str
    candidates: Tuple[str, ...]
    candidate_tokens: Tuple[np.ndarray, ...]
    answer_index: int


def build_repo_recall_instances(files: Sequence[dict], encode: Callable[[str], Sequence[int]],
                                rng: Optional[np.random.Generator] = None,
                                n_candidates: int = 4) -> List[RepoRecallInstance]:
    """Extract symbol retrieval instances across dependencies in a multi-file repo."""
    if n_candidates < 2:
        raise ValueError(f"n_candidates must be >= 2, got {n_candidates}")
    from ..data.repo_graph import build_repo_graph

    graph = build_repo_graph(files)
    by_path = {f["path"]: f["text"] for f in files}
    all_exports = sorted({s for syms in graph.exports.values() for s in syms})
    instances: List[RepoRecallInstance] = []

    for consumer in graph.files:
        consumer_text = by_path[consumer]
        for definer in sorted(graph.dependencies.get(consumer, set())):
            exported_in_definer = graph.exports.get(definer, set())
            for sym in sorted(exported_in_definer):
                ts_match = re.search(r"import\s*\{[^}]*\b" + re.escape(sym) + r"\b[^}]*\}\s*from", consumer_text)
                py_match = re.search(r"from\s+[\.\w]+\s+import\s+[^#\n]*\b" + re.escape(sym) + r"\b", consumer_text)
                match = ts_match or py_match
                if not match:
                    continue
                start = match.start()
                inside = consumer_text[start : match.end()]
                sym_pos = inside.find(sym)
                if sym_pos < 0:
                    continue
                query_prefix = consumer_text[:start + sym_pos]

                distractors = [f"{sym}Helper", f"{sym}Impl", f"{sym}Alt", f"get{sym.capitalize()}"][:n_candidates - 1]
                candidates = tuple(sorted(set(distractors) | {sym}))
                answer_index = candidates.index(sym)
                candidate_tokens = tuple(
                    np.asarray(list(encode(c)), dtype=np.int64).reshape(-1) for c in candidates
                )
                if any(ct.size < 1 for ct in candidate_tokens):
                    continue

                instances.append(RepoRecallInstance(
                    id=f"{consumer}::{sym}",
                    definer=definer,
                    consumer=consumer,
                    symbol=sym,
                    consumer_prefix=query_prefix,
                    candidates=candidates,
                    candidate_tokens=candidate_tokens,
                    answer_index=answer_index,
                ))
    return instances


def evaluate_repo_recall(model, files: Sequence[dict], encode: Callable[[str], Sequence[int]],
                         rng: Optional[np.random.Generator] = None, *,
                         n_candidates: int = 4, n_random_orders: int = 10,
                         batch_size: int = 8, to_numpy=np.asarray,
                         file_sep: str = "\n<|file_sep|>\n",
                         rank_metric: str = "ce_nats") -> dict:
    """Compare symbol resolution accuracy across topologically sorted sequences vs random file sequences (#359).

    Emits records with suite='repo_recall', bucket='topological' vs bucket='random'.
    Returns summary dict containing `topo_top1_accuracy` and `random_top1_accuracy`.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    from ..data.repo_graph import build_repo_graph

    graph = build_repo_graph(files)
    topo_order = graph.topological_sort()
    by_path = {f["path"]: f["text"] for f in files}
    instances = build_repo_recall_instances(files, encode, rng, n_candidates=n_candidates)
    if not instances:
        raise ValueError("evaluate_repo_recall(): no cross-file recall instances found in repository")

    all_paths = list(graph.files)
    random_orders: List[List[str]] = []
    for _ in range(n_random_orders):
        p = list(rng.permutation(all_paths))
        if p != topo_order:
            random_orders.append(p)
        else:
            p.reverse()
            random_orders.append(p)

    def make_prefix(order: Sequence[str], consumer: str, consumer_prefix: str) -> str:
        c_idx = order.index(consumer)
        preceding = order[:c_idx]
        if preceding:
            return file_sep.join(by_path[p] for p in preceding) + file_sep + consumer_prefix
        return consumer_prefix

    records: List[dict] = []

    # 1. Score topological order
    for inst in instances:
        prefix_str = make_prefix(topo_order, inst.consumer, inst.consumer_prefix)
        prefix_toks = np.asarray(list(encode(prefix_str)), dtype=np.int64).reshape(-1)
        rows = [
            ScoreRow(tokens=np.concatenate([prefix_toks, ct]), span_start=int(prefix_toks.size), span_len=int(ct.size))
            for ct in inst.candidate_tokens
        ]
        scored = score_rows(model, rows, batch_size=batch_size, to_numpy=to_numpy)
        ordering = sorted(range(len(inst.candidates)), key=lambda j: (scored[j][rank_metric], inst.candidates[j]))
        rank = ordering.index(inst.answer_index) + 1
        ans = scored[inst.answer_index]
        records.append(make_record(
            suite="repo_recall", id=f"{inst.id}::topo", bucket="topological", distance=int(prefix_toks.size),
            n_scored_tokens=ans["n_scored_tokens"], ce_nats=ans["ce_nats"], token_accuracy=ans["token_accuracy"],
            exact_match=ans["exact_match"], rank_top1=(rank == 1), mrr=1.0 / rank,
            meta={"ordering": "topological", "definer": inst.definer, "consumer": inst.consumer, "symbol": inst.symbol, "rank": rank}
        ))

    # 2. Score random orders
    for r_idx, rand_order in enumerate(random_orders):
        for inst in instances:
            prefix_str = make_prefix(rand_order, inst.consumer, inst.consumer_prefix)
            prefix_toks = np.asarray(list(encode(prefix_str)), dtype=np.int64).reshape(-1)
            rows = [
                ScoreRow(tokens=np.concatenate([prefix_toks, ct]), span_start=int(prefix_toks.size), span_len=int(ct.size))
                for ct in inst.candidate_tokens
            ]
            scored = score_rows(model, rows, batch_size=batch_size, to_numpy=to_numpy)
            ordering = sorted(range(len(inst.candidates)), key=lambda j: (scored[j][rank_metric], inst.candidates[j]))
            rank = ordering.index(inst.answer_index) + 1
            ans = scored[inst.answer_index]
            records.append(make_record(
                suite="repo_recall", id=f"{inst.id}::rand_{r_idx}", bucket="random", distance=int(prefix_toks.size),
                n_scored_tokens=ans["n_scored_tokens"], ce_nats=ans["ce_nats"], token_accuracy=ans["token_accuracy"],
                exact_match=ans["exact_match"], rank_top1=(rank == 1), mrr=1.0 / rank,
                meta={"ordering": "random", "definer": inst.definer, "consumer": inst.consumer, "symbol": inst.symbol, "rank": rank, "perm_idx": r_idx}
            ))

    topo_records = [r for r in records if r["bucket"] == "topological"]
    rand_records = [r for r in records if r["bucket"] == "random"]

    topo_top1 = float(np.mean([r["rank_top1"] for r in topo_records]))
    rand_top1 = float(np.mean([r["rank_top1"] for r in rand_records]))
    topo_mrr = float(np.mean([r["mrr"] for r in topo_records]))
    rand_mrr = float(np.mean([r["mrr"] for r in rand_records]))

    summary = summarize_bucketed(records, ["topological", "random"])
    return {
        "records": records,
        "topo_top1_accuracy": topo_top1,
        "random_top1_accuracy": rand_top1,
        "topo_mrr": topo_mrr,
        "random_mrr": rand_mrr,
        "topological_order": topo_order,
        "n_instances": len(instances),
        "n_random_orders": len(random_orders),
        "by_bucket": summary["by_bucket"],
        "overall": summary["overall"],
    }

# --------------------------------------------------------------------------------------- #
# Data Engineering, Contracts & Query Verifier Evaluation (#342)
# --------------------------------------------------------------------------------------- #

DATA_CONTRACT_BUCKETS: Tuple[str, ...] = (
    "sql",
    "data_pipeline",
    "openapi",
    "graphql",
    "protobuf",
    "clean_architecture",
)


def evaluate_data_contracts(
    test_cases: Sequence[dict],
    *,
    verifier: Optional[Any] = None,
) -> dict:
    """Evaluate completions against data engineering, contract & query verifiers (#342).

    Each test case in `test_cases` is a dict containing:
      - `id`: unique case identifier
      - `contract`: contract name ('sql', 'data_pipeline', 'openapi', 'graphql', 'protobuf', 'clean_architecture')
      - `code`: candidate code/spec text
      - `reference`: optional reference code/spec or query for equivalence checks
      - `expected_clean`: bool indicating if the test case is clean (reward == 1.0) vs faulty/hacked

    Emits records conforming to RECORD_FIELDS:
      suite='data_contracts', bucket=contract, distance=len(code), etc.
    Returns summary dict with per-bucket and overall statistics.
    """
    if verifier is None:
        from ..train.verifiers.data_contracts import DataContractsVerifier
        verifier = DataContractsVerifier()

    records: List[dict] = []
    for inst in test_cases:
        inst_id = str(inst.get("id", "case"))
        contract = str(inst.get("contract", "sql"))
        code = str(inst.get("code", ""))
        reference = inst.get("reference")
        expected_clean = bool(inst.get("expected_clean", True))
        prompt = str(inst.get("prompt", ""))

        reward = verifier.reward(code, reference=reference, prompt=prompt, contract=contract)
        is_clean = (reward is not None and reward == 1.0)
        success = (is_clean == expected_clean)

        records.append(
            make_record(
                suite="data_contracts",
                id=inst_id,
                bucket=contract,
                distance=len(code),
                n_scored_tokens=len(code.split()) if code else 0,
                ce_nats=0.0 if is_clean else 1.0,
                token_accuracy=1.0 if success else 0.0,
                exact_match=1.0 if (reward == 1.0) else 0.0,
                rank_top1=success,
                mrr=1.0 if success else 0.0,
                meta={
                    "contract": contract,
                    "reward": reward,
                    "expected_clean": expected_clean,
                    "is_clean": is_clean,
                    "success": success,
                },
            )
        )

    buckets_present = [b for b in DATA_CONTRACT_BUCKETS if any(r["bucket"] == b for r in records)] or list(DATA_CONTRACT_BUCKETS)
    summary = summarize_bucketed(records, buckets_present)
    total = len(records)
    passed = sum(1 for r in records if r["rank_top1"])
    accuracy = (passed / total) if total else 0.0

    return {
        "records": records,
        "accuracy": accuracy,
        "n_cases": total,
        "n_passed": passed,
        "by_bucket": summary["by_bucket"],
        "overall": summary["overall"],
    }


# --------------------------------------------------------------------------- #
# Web & Backend Application Stack Suite (#343)
# --------------------------------------------------------------------------- #

WEB_BACKEND_BUCKETS: Tuple[str, ...] = (
    "typescript_web",
    "python_web",
    "go_web",
    "java_web",
    "csharp_web",
    "php_web",
    "ruby_web",
)


def evaluate_web_backend(
    test_cases: Sequence[dict],
    *,
    verifier: Optional[Any] = None,
) -> dict:
    """Evaluate completions against Web & Backend application stack verifiers (#343).

    Each test case in `test_cases` is a dict containing:
      - `id`: unique case identifier
      - `stack`: stack name ('typescript_web', 'python_web', 'go_web', 'java_web', 'csharp_web', 'php_web', 'ruby_web')
      - `code`: candidate code text
      - `reference`: optional reference code
      - `expected_clean`: bool indicating if the test case is clean (reward == 1.0) vs faulty/hacked
      - `prompt`: optional prompt string prefix

    Emits records conforming to RECORD_FIELDS:
      suite='web_backend', bucket=stack, distance=len(code), etc.
    Returns summary dict with per-bucket and overall statistics.
    """
    if verifier is None:
        from ..train.verifiers.web_backend import WebBackendVerifier
        verifier = WebBackendVerifier()

    records: List[dict] = []
    for inst in test_cases:
        inst_id = str(inst.get("id", "case"))
        stack = str(inst.get("stack", "typescript_web"))
        code = str(inst.get("code", ""))
        reference = inst.get("reference")
        expected_clean = bool(inst.get("expected_clean", True))
        prompt = str(inst.get("prompt", ""))

        reward = verifier.reward(code, reference=reference, prompt=prompt, stack=stack)
        is_clean = (reward is not None and reward == 1.0)
        success = (is_clean == expected_clean)

        records.append(
            make_record(
                suite="web_backend",
                id=inst_id,
                bucket=stack,
                distance=len(code),
                n_scored_tokens=len(code.split()) if code else 0,
                ce_nats=0.0 if is_clean else 1.0,
                token_accuracy=1.0 if success else 0.0,
                exact_match=1.0 if (reward == 1.0) else 0.0,
                rank_top1=success,
                mrr=1.0 if success else 0.0,
                meta={
                    "stack": stack,
                    "reward": reward,
                    "expected_clean": expected_clean,
                    "is_clean": is_clean,
                    "success": success,
                },
            )
        )

    buckets_present = [b for b in WEB_BACKEND_BUCKETS if any(r["bucket"] == b for r in records)] or list(WEB_BACKEND_BUCKETS)
    summary = summarize_bucketed(records, buckets_present)
    total = len(records)
    passed = sum(1 for r in records if r["rank_top1"])
    accuracy = (passed / total) if total else 0.0

    return {
        "records": records,
        "accuracy": accuracy,
        "n_cases": total,
        "n_passed": passed,
        "by_bucket": summary["by_bucket"],
        "overall": summary["overall"],
    }


# --------------------------------------------------------------------------- #
# Cloud, Infra, Containers & Automation Suite (#344)
# --------------------------------------------------------------------------- #

CLOUD_INFRA_BUCKETS: Tuple[str, ...] = (
    "terraform",
    "docker",
    "kubernetes",
    "shell",
    "html_tailwind",
)


def evaluate_cloud_infra(
    test_cases: Sequence[dict],
    *,
    verifier: Optional[Any] = None,
) -> dict:
    """Evaluate completions against Cloud, Infra, Containers & Automation verifiers (#344).

    Each test case in `test_cases` is a dict containing:
      - `id`: unique case identifier
      - `domain`: domain name ('terraform', 'docker', 'kubernetes', 'shell', 'html_tailwind')
      - `code`: candidate code/manifest text
      - `reference`: optional reference code
      - `expected_clean`: bool indicating if the test case is clean (reward == 1.0) vs faulty/hacked
      - `prompt`: optional prompt string prefix

    Emits records conforming to RECORD_FIELDS:
      suite='cloud_infra', bucket=domain, distance=len(code), etc.
    Returns summary dict with per-bucket and overall statistics.
    """
    if verifier is None:
        from ..train.verifiers.cloud_infra import CloudInfraVerifier
        verifier = CloudInfraVerifier()

    records: List[dict] = []
    for inst in test_cases:
        inst_id = str(inst.get("id", "case"))
        domain = str(inst.get("domain", "terraform"))
        code = str(inst.get("code", ""))
        reference = inst.get("reference")
        expected_clean = bool(inst.get("expected_clean", True))
        prompt = str(inst.get("prompt", ""))

        reward = verifier.reward(code, reference=reference, prompt=prompt, domain=domain)
        is_clean = (reward is not None and reward == 1.0)
        success = (is_clean == expected_clean)

        records.append(
            make_record(
                suite="cloud_infra",
                id=inst_id,
                bucket=domain,
                distance=len(code),
                n_scored_tokens=len(code.split()) if code else 0,
                ce_nats=0.0 if is_clean else 1.0,
                token_accuracy=1.0 if success else 0.0,
                exact_match=1.0 if (reward == 1.0) else 0.0,
                rank_top1=success,
                mrr=1.0 if success else 0.0,
                meta={
                    "domain": domain,
                    "reward": reward,
                    "expected_clean": expected_clean,
                    "is_clean": is_clean,
                    "success": success,
                },
            )
        )

    buckets_present = [b for b in CLOUD_INFRA_BUCKETS if any(r["bucket"] == b for r in records)] or list(CLOUD_INFRA_BUCKETS)
    summary = summarize_bucketed(records, buckets_present)
    total = len(records)
    passed = sum(1 for r in records if r["rank_top1"])
    accuracy = (passed / total) if total else 0.0

    return {
        "records": records,
        "accuracy": accuracy,
        "n_cases": total,
        "n_passed": passed,
        "by_bucket": summary["by_bucket"],
        "overall": summary["overall"],
    }


# --------------------------------------------------------------------------- #
# Repository-Scale Symbol Grounding & RULER-over-Code Suite (#345)
# --------------------------------------------------------------------------- #

REPO_GROUNDING_SYMBOL_KINDS: Tuple[str, ...] = (
    "type_annotation",
    "imported_symbol",
    "function_invocation",
)

REPO_GROUNDING_CONTEXT_BUCKETS: Tuple[str, ...] = (
    "8k",
    "16k",
    "24k",
    "32k",
)


@dataclass(frozen=True)
class RepoGroundingInstance:
    """One cross-file symbol grounding probe instance across scaled repository contexts (#345)."""

    id: str
    symbol_kind: str  # "type_annotation", "imported_symbol", "function_invocation"
    context_bucket: str  # "8k", "16k", "24k", "32k"
    symbol: str
    definer: str
    consumer: str
    prefix_tokens: np.ndarray
    candidates: Tuple[str, ...]
    candidate_tokens: Tuple[np.ndarray, ...]
    answer_index: int
    context_length: int
    meta: Dict[str, Any]


def _bucket_for_ctx_len(length: int) -> str:
    if length <= 8192:
        return "8k"
    elif length <= 16384:
        return "16k"
    elif length <= 24576:
        return "24k"
    return "32k"


def build_repo_grounding_instances(
    files: Sequence[dict],
    encode: Callable[[str], Sequence[int]],
    rng: Optional[np.random.Generator] = None,
    *,
    context_lengths: Sequence[int] = (8192, 16384, 32768),
    n_candidates: int = 4,
    max_instances: Optional[int] = None,
) -> List[RepoGroundingInstance]:
    """Extract symbol grounding instances across 8k-32k token context windows.

    Probes 3 distinct symbol usage categories:
    1. `imported_symbol`: imported identifier in import clauses.
    2. `type_annotation`: type references (param annotations, return types, interfaces).
    3. `function_invocation`: cross-file function/method calls.

    Pads the multi-file repository context with distractor/filler files so the symbol
    query appears at controlled 8k, 16k, or 32k context horizons.
    """
    if n_candidates < 2:
        raise ValueError(f"n_candidates must be >= 2, got {n_candidates}")
    if rng is None:
        rng = np.random.default_rng(0)

    from ..data.repo_graph import build_repo_graph

    graph = build_repo_graph(files)
    by_path = {f["path"]: f["text"] for f in files}
    all_exports = sorted({s for syms in graph.exports.values() for s in syms})
    paths = sorted(graph.files)

    instances: List[RepoGroundingInstance] = []

    for consumer in paths:
        consumer_text = by_path[consumer]
        dependencies = sorted(graph.dependencies.get(consumer, set()))
        for definer in dependencies:
            definer_text = by_path[definer]
            exported_syms = sorted(graph.exports.get(definer, set()))

            for sym in exported_syms:
                # 1. Imported Symbol probe
                ts_imp = re.search(r"import\s*\{[^}]*\b" + re.escape(sym) + r"\b[^}]*\}\s*from", consumer_text)
                py_imp = re.search(r"from\s+[\.\w]+\s+import\s+[^#\n]*\b" + re.escape(sym) + r"\b", consumer_text)
                imp_match = ts_imp or py_imp
                if imp_match:
                    start = imp_match.start()
                    inside = consumer_text[start: imp_match.end()]
                    sym_pos = inside.find(sym)
                    if sym_pos >= 0:
                        query_prefix = consumer_text[: start + sym_pos]
                        _make_grounding_instances(
                            instances=instances,
                            symbol_kind="imported_symbol",
                            sym=sym,
                            definer=definer,
                            consumer=consumer,
                            query_prefix=query_prefix,
                            definer_text=definer_text,
                            by_path=by_path,
                            paths=paths,
                            encode=encode,
                            all_exports=all_exports,
                            context_lengths=context_lengths,
                            n_candidates=n_candidates,
                            rng=rng,
                            max_instances=max_instances,
                        )
                        if max_instances is not None and len(instances) >= max_instances:
                            return instances

                # 2. Type Annotation probe
                ts_type = re.search(r"(?::\s*|\bextends\s+|\bimplements\s+|<)\s*" + re.escape(sym) + r"\b", consumer_text)
                py_type = re.search(r":\s*" + re.escape(sym) + r"\b", consumer_text)
                type_match = ts_type or py_type
                if type_match:
                    pos = consumer_text.find(sym, type_match.start())
                    if pos >= 0:
                        query_prefix = consumer_text[:pos]
                        _make_grounding_instances(
                            instances=instances,
                            symbol_kind="type_annotation",
                            sym=sym,
                            definer=definer,
                            consumer=consumer,
                            query_prefix=query_prefix,
                            definer_text=definer_text,
                            by_path=by_path,
                            paths=paths,
                            encode=encode,
                            all_exports=all_exports,
                            context_lengths=context_lengths,
                            n_candidates=n_candidates,
                            rng=rng,
                            max_instances=max_instances,
                        )
                        if max_instances is not None and len(instances) >= max_instances:
                            return instances

                # 3. Function Invocation probe
                fn_call = re.search(r"\b" + re.escape(sym) + r"\s*\(", consumer_text)
                if fn_call:
                    pos = fn_call.start()
                    query_prefix = consumer_text[:pos]
                    _make_grounding_instances(
                        instances=instances,
                        symbol_kind="function_invocation",
                        sym=sym,
                        definer=definer,
                        consumer=consumer,
                        query_prefix=query_prefix,
                        definer_text=definer_text,
                        by_path=by_path,
                        paths=paths,
                        encode=encode,
                        all_exports=all_exports,
                        context_lengths=context_lengths,
                        n_candidates=n_candidates,
                        rng=rng,
                        max_instances=max_instances,
                    )
                    if max_instances is not None and len(instances) >= max_instances:
                        return instances

    return instances


def _make_grounding_instances(
    *,
    instances: List[RepoGroundingInstance],
    symbol_kind: str,
    sym: str,
    definer: str,
    consumer: str,
    query_prefix: str,
    definer_text: str,
    by_path: Dict[str, str],
    paths: List[str],
    encode: Callable[[str], Sequence[int]],
    all_exports: List[str],
    context_lengths: Sequence[int],
    n_candidates: int,
    rng: np.random.Generator,
    max_instances: Optional[int],
) -> None:
    pool_distractors = [s for s in all_exports if s != sym]
    if not pool_distractors:
        pool_distractors = [f"{sym}Helper", f"{sym}Service", f"{sym}Handler", f"get{sym.capitalize()}"]
    take = min(n_candidates - 1, len(pool_distractors))
    chosen = [pool_distractors[i] for i in rng.choice(len(pool_distractors), size=take, replace=False)]
    candidates = tuple(sorted(set(chosen) | {sym}))
    answer_index = candidates.index(sym)
    candidate_tokens = tuple(np.asarray(list(encode(c)), dtype=np.int64).reshape(-1) for c in candidates)
    if any(ct.size < 1 for ct in candidate_tokens):
        return

    sep = "\n<|file_sep|>\n"
    base_prefix = definer_text + sep + query_prefix
    base_toks = np.asarray(list(encode(base_prefix)), dtype=np.int64).reshape(-1)

    distractor_pool = [by_path[p] for p in paths if p not in (definer, consumer)]
    if not distractor_pool:
        distractor_pool = [
            "// Unrelated repository module\nexport function calculateMetrics(x: number) { return x * 2; }\n"
        ]

    for target_len in context_lengths:
        bucket_name = _bucket_for_ctx_len(target_len)
        current_toks = base_toks
        if current_toks.size < target_len:
            pad_parts: List[str] = []
            cur_len = int(current_toks.size)
            idx = 0
            while cur_len < target_len:
                d_text = distractor_pool[idx % len(distractor_pool)]
                pad_parts.append(d_text)
                cur_len += len(d_text.split())
                idx += 1
                if idx > 2000:
                    break
            padded_text = sep.join(pad_parts) + sep + base_prefix
            current_toks = np.asarray(list(encode(padded_text)), dtype=np.int64).reshape(-1)
            if current_toks.size > target_len:
                current_toks = current_toks[-target_len:]

        inst = RepoGroundingInstance(
            id=f"{consumer}::{sym}::{symbol_kind}::{bucket_name}",
            symbol_kind=symbol_kind,
            context_bucket=bucket_name,
            symbol=sym,
            definer=definer,
            consumer=consumer,
            prefix_tokens=current_toks,
            candidates=candidates,
            candidate_tokens=candidate_tokens,
            answer_index=answer_index,
            context_length=int(current_toks.size),
            meta={"symbol": sym, "definer": definer, "consumer": consumer, "symbol_kind": symbol_kind, "bucket": bucket_name},
        )
        instances.append(inst)
        if max_instances is not None and len(instances) >= max_instances:
            break


def evaluate_repository_symbol_recall(
    model: Any,
    instances: Sequence[RepoGroundingInstance],
    *,
    batch_size: int = 8,
    pad_id: int = 0,
    to_numpy: Callable = np.asarray,
    rank_metric: str = "ce_nats",
) -> dict:
    """Evaluate cross-file symbol recall across type annotations, imports, and invocations (#345).

    Emits records conforming to RECORD_FIELDS with suite='repo_symbol_grounding'.
    Reports Top-1 Recall and Mean Reciprocal Rank (MRR) bucketed by symbol kind and context depth.
    """
    if not instances:
        raise ValueError("evaluate_repository_symbol_recall(): instances cannot be empty")

    rows: List[ScoreRow] = []
    owners: List[Tuple[int, int]] = []
    for i, inst in enumerate(instances):
        for j, ct in enumerate(inst.candidate_tokens):
            rows.append(ScoreRow(tokens=np.concatenate([inst.prefix_tokens, ct]), span_start=int(inst.prefix_tokens.size), span_len=int(ct.size)))
            owners.append((i, j))

    scored = score_rows(model, rows, pad_id=pad_id, batch_size=batch_size, to_numpy=to_numpy)

    per_instance: List[Dict[int, dict]] = [{} for _ in instances]
    for (i, j), s in zip(owners, scored):
        per_instance[i][j] = s

    records: List[dict] = []
    for i, inst in enumerate(instances):
        answer = per_instance[i][inst.answer_index]
        ordering = sorted(range(len(inst.candidates)), key=lambda j: (per_instance[i][j][rank_metric], inst.candidates[j]))
        rank = ordering.index(inst.answer_index) + 1
        records.append(make_record(
            suite="repo_symbol_grounding",
            id=inst.id,
            bucket=inst.symbol_kind,
            distance=inst.context_length,
            n_scored_tokens=answer["n_scored_tokens"],
            ce_nats=answer["ce_nats"],
            token_accuracy=answer["token_accuracy"],
            exact_match=answer["exact_match"],
            rank_top1=(rank == 1),
            mrr=1.0 / rank,
            meta={
                "symbol": inst.symbol,
                "symbol_kind": inst.symbol_kind,
                "context_bucket": inst.context_bucket,
                "definer": inst.definer,
                "consumer": inst.consumer,
                "rank": rank,
                "n_candidates": len(inst.candidates),
            },
        ))

    by_symbol_kind: Dict[str, dict] = {}
    for kind in REPO_GROUNDING_SYMBOL_KINDS:
        sub = [r for r in records if r["meta"]["symbol_kind"] == kind]
        by_symbol_kind[kind] = {
            "top1_recall": float(np.mean([r["rank_top1"] for r in sub])) if sub else 0.0,
            "mrr": float(np.mean([r["mrr"] for r in sub])) if sub else 0.0,
            "n_instances": len(sub),
        }

    by_context_bucket: Dict[str, dict] = {}
    for b in REPO_GROUNDING_CONTEXT_BUCKETS:
        sub = [r for r in records if r["meta"]["context_bucket"] == b]
        by_context_bucket[b] = {
            "top1_recall": float(np.mean([r["rank_top1"] for r in sub])) if sub else 0.0,
            "mrr": float(np.mean([r["mrr"] for r in sub])) if sub else 0.0,
            "n_instances": len(sub),
        }

    top1 = float(np.mean([r["rank_top1"] for r in records]))
    mrr = float(np.mean([r["mrr"] for r in records]))
    summary = summarize_bucketed(records, list(REPO_GROUNDING_SYMBOL_KINDS))

    return {
        "records": records,
        "top1_recall": top1,
        "mrr": mrr,
        "by_symbol_kind": by_symbol_kind,
        "by_context_bucket": by_context_bucket,
        "by_bucket": summary["by_bucket"],
        "overall": summary["overall"],
    }


# --------------------------------------------------------------------------- #
# Information-Gain Probe (Delta CE with vs without interface context) (#345)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class InfoGainInstance:
    """One information-gain probe instance measuring Delta CE on implementation tokens (#345)."""

    id: str
    target_file: str
    with_context_tokens: np.ndarray
    with_span_start: int
    without_context_tokens: np.ndarray
    without_span_start: int
    span_len: int
    interface_files: Tuple[str, ...]
    meta: Dict[str, Any]


def build_information_gain_instances(
    files: Sequence[dict],
    encode: Callable[[str], Sequence[int]],
    *,
    file_sep: str = "\n<|file_sep|>\n",
    max_instances: Optional[int] = None,
) -> List[InfoGainInstance]:
    """Extract information-gain probe instances from repository code.

    Measures model loss (cross-entropy) over implementation code blocks:
    - Condition A (with_context): Interface / type definitions prepended before implementation.
    - Condition B (without_context): Implementation code scored in isolation (no interface context).
    """
    from ..data.repo_graph import build_repo_graph

    graph = build_repo_graph(files)
    by_path = {f["path"]: f["text"] for f in files}
    instances: List[InfoGainInstance] = []

    for consumer in sorted(graph.files):
        consumer_text = by_path[consumer]
        dependencies = sorted(graph.dependencies.get(consumer, set()))
        if not dependencies:
            continue

        interface_context = file_sep.join(by_path[d] for d in dependencies) + file_sep
        if len(consumer_text.strip()) < 20:
            continue

        split_pt = max(1, len(consumer_text) // 2)
        prefix_str = consumer_text[:split_pt]
        body_str = consumer_text[split_pt:]
        if not body_str.strip():
            continue

        body_toks = np.asarray(list(encode(body_str)), dtype=np.int64).reshape(-1)
        if body_toks.size < 2:
            continue

        with_prefix = interface_context + prefix_str
        with_pre_toks = np.asarray(list(encode(with_prefix)), dtype=np.int64).reshape(-1)
        if with_pre_toks.size < 1:
            continue
        with_toks = np.concatenate([with_pre_toks, body_toks])

        without_pre_toks = np.asarray(list(encode(prefix_str)), dtype=np.int64).reshape(-1)
        if without_pre_toks.size < 1:
            without_pre_toks = np.asarray(list(encode("// local context\n")), dtype=np.int64).reshape(-1)
        without_toks = np.concatenate([without_pre_toks, body_toks])

        instances.append(InfoGainInstance(
            id=f"{consumer}::info_gain",
            target_file=consumer,
            with_context_tokens=with_toks,
            with_span_start=int(with_pre_toks.size),
            without_context_tokens=without_toks,
            without_span_start=int(without_pre_toks.size),
            span_len=int(body_toks.size),
            interface_files=tuple(dependencies),
            meta={"consumer": consumer, "dependencies": dependencies, "span_len": int(body_toks.size)},
        ))
        if max_instances is not None and len(instances) >= max_instances:
            break

    return instances


def evaluate_information_gain(
    model: Any,
    instances: Sequence[InfoGainInstance],
    *,
    batch_size: int = 8,
    pad_id: int = 0,
    to_numpy: Callable = np.asarray,
) -> dict:
    """Measure Delta CE on implementation tokens with and without repository interface context (#345).

    Delta CE = CE_without - CE_with (positive Delta CE indicates information gain from repo context).
    Emits records conforming to RECORD_FIELDS with suite='repo_information_gain'.
    """
    if not instances:
        raise ValueError("evaluate_information_gain(): instances cannot be empty")

    rows: List[ScoreRow] = []
    for inst in instances:
        rows.append(ScoreRow(tokens=inst.with_context_tokens, span_start=inst.with_span_start, span_len=inst.span_len))
        rows.append(ScoreRow(tokens=inst.without_context_tokens, span_start=inst.without_span_start, span_len=inst.span_len))

    scored = score_rows(model, rows, pad_id=pad_id, batch_size=batch_size, to_numpy=to_numpy)

    records: List[dict] = []
    delta_ces: List[float] = []

    for i, inst in enumerate(instances):
        s_with = scored[2 * i]
        s_without = scored[2 * i + 1]
        ce_with = float(s_with["ce_nats"])
        ce_without = float(s_without["ce_nats"])
        delta_ce = ce_without - ce_with
        delta_ces.append(delta_ce)

        records.append(make_record(
            suite="repo_information_gain",
            id=inst.id,
            bucket="information_gain",
            distance=int(inst.with_context_tokens.size),
            n_scored_tokens=inst.span_len,
            ce_nats=delta_ce,
            token_accuracy=float(s_with["token_accuracy"]),
            exact_match=float(s_with["exact_match"]),
            rank_top1=(delta_ce > 0),
            mrr=1.0 if (delta_ce > 0) else 0.0,
            meta={
                "target_file": inst.target_file,
                "interface_files": list(inst.interface_files),
                "ce_with": ce_with,
                "ce_without": ce_without,
                "delta_ce": delta_ce,
            },
        ))

    mean_delta = float(np.mean(delta_ces))
    summary = summarize_bucketed(records, ["information_gain"])

    return {
        "records": records,
        "mean_delta_ce": mean_delta,
        "n_instances": len(instances),
        "by_bucket": summary["by_bucket"],
        "overall": summary["overall"],
    }


# --------------------------------------------------------------------------- #
# Compiler-Grounded Blast Radius Verifier Evaluation (#345)
# --------------------------------------------------------------------------- #

BLAST_RADIUS_BUCKETS: Tuple[str, ...] = (
    "signature_mutation",
    "type_removal",
    "interface_rename",
    "required_param_added",
)


def evaluate_blast_radius(
    test_cases: Sequence[dict],
    *,
    verifier: Optional[Any] = None,
) -> dict:
    """Evaluate completions against compiler blast radius prediction verifier (#345).

    Each test case in `test_cases` is a dict containing:
      - `id`: unique case identifier
      - `bucket`: category name ('signature_mutation', 'type_removal', 'interface_rename', 'required_param_added')
      - `completion`: candidate prediction text or JSON
      - `repo`: repository file mapping `{path: content}`
      - `mutated_file`: mutated file path
      - `mutated_content`: mutated file content
      - `reference` / `ground_truth`: optional ground truth locations
      - `expected_clean`: bool indicating if reward == 1.0 is expected

    Emits records conforming to RECORD_FIELDS:
      suite='blast_radius', bucket=bucket, distance=len(completion), etc.
    Returns summary dict with per-bucket and overall statistics.
    """
    if verifier is None:
        from ..train.verifiers.repository_context import BlastRadiusVerifier
        verifier = BlastRadiusVerifier()

    records: List[dict] = []
    for inst in test_cases:
        inst_id = str(inst.get("id", "case"))
        bucket = str(inst.get("bucket", "signature_mutation"))
        completion = str(inst.get("completion", ""))
        repo = inst.get("repo")
        mutated_file = inst.get("mutated_file")
        mutated_content = inst.get("mutated_content")
        ground_truth = inst.get("ground_truth", inst.get("reference"))
        expected_clean = bool(inst.get("expected_clean", True))
        prompt = str(inst.get("prompt", ""))

        eval_res = verifier.evaluate(
            completion=completion,
            reference=ground_truth,
            prompt=prompt,
            repo=repo,
            mutated_file=mutated_file,
            mutated_content=mutated_content,
        )

        reward = float(eval_res["reward"])
        is_clean = (reward == 1.0)
        success = (is_clean == expected_clean)

        records.append(
            make_record(
                suite="blast_radius",
                id=inst_id,
                bucket=bucket,
                distance=len(completion),
                n_scored_tokens=len(completion.split()) if completion else 0,
                ce_nats=0.0 if is_clean else 1.0,
                token_accuracy=float(eval_res["precision"]),
                exact_match=1.0 if is_clean else 0.0,
                rank_top1=success,
                mrr=float(eval_res["f1"]),
                meta={
                    "bucket": bucket,
                    "reward": reward,
                    "precision": eval_res["precision"],
                    "recall": eval_res["recall"],
                    "f1": eval_res["f1"],
                    "expected_clean": expected_clean,
                    "is_clean": is_clean,
                    "success": success,
                },
            )
        )

    buckets_present = [b for b in BLAST_RADIUS_BUCKETS if any(r["bucket"] == b for r in records)] or list(BLAST_RADIUS_BUCKETS)
    summary = summarize_bucketed(records, buckets_present)
    total = len(records)
    passed = sum(1 for r in records if r["rank_top1"])
    accuracy = (passed / total) if total else 0.0

    return {
        "records": records,
        "accuracy": accuracy,
        "n_cases": total,
        "n_passed": passed,
        "by_bucket": summary["by_bucket"],
        "overall": summary["overall"],
    }
