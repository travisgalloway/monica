"""Cross-file symbol-resolution recall for TypeScript, bucketed by token distance (#221).

ABOVE THE SEAM. Pure numpy + stdlib; the optional `tree_sitter` toolchain is imported
lazily inside a function (the `src/lsp/ts_boundaries.py` precedent) and its absence only
removes a cross-check, never the eval.

What it measures and why it is not just FIM
-------------------------------------------
A symbol is *defined* in one file and *used* in another. Between the two we stack `k`
unrelated files, so the model must carry the definition across a controlled number of
tokens before predicting the identifier at the use site. On a mostly-SSM backbone that
recall has to survive a fixed-width state — the exact failure the hybrid's ~12.5%
attention layers exist to patch — and reporting one aggregate number would hide it. So
every instance is bucketed by its measured token distance, and every instance also emits
its own record.

Two metrics, both teacher-forced (no generation, no pass@1 gating):

* **CE / token-accuracy / exact-match** over the identifier's token span at the use site.
  Cheap, but confounded: a low loss on a common identifier is not recall.
* **Discriminative rank** — the real recall signal. The same position is scored once per
  *candidate* symbol (the correct one plus near-miss identifiers exported elsewhere in the
  bundle, e.g. `areaOfCircle` vs `areaOfSquare`), and the candidates are ranked by summed
  negative log-probability. `rank_top1`/`mrr` then answer "did the model resolve the
  cross-file definition?" rather than "is this identifier cheap?".

  Ranking on the **sum** (not the mean) of log-probs is the standard multiple-choice
  scoring rule and is what #221's plan specifies; it does carry a mild bias toward shorter
  candidates, which is why the candidate pool is drawn from same-shape near-miss exports
  rather than arbitrary vocabulary.

Extraction is deliberately shallow and **fail-closed**
------------------------------------------------------
`export`/`import` are matched with regexes over single lines. Anything ambiguous is
*skipped, never guessed*: `export *`, `export default`, re-export forms
(`export { x } from './y'`), aliased import specifiers (`{ a as b }`), and multi-line
destructured imports all yield no instance. A wrong instance is worse than a missing one —
it would silently score the model on a symbol that was never actually defined where we
claim it was.

Tokenization
------------
Python has no code tokenizer any more (retired with #245; the tokenizer is the native
Swift `swift/MonicaTokenizer` package), so this module takes an **injected**
`encode: Callable[[str], Sequence[int]]`. `src.data.tokenize.ByteTokenizer().encode` is the
offline fixture encoder; `monica-tokenize encode --json` is the real one.

Context segments are encoded **separately** and concatenated, so the scored identifier span
is exactly the candidate's own tokens. Under a byte-level tokenizer that is identical to
encoding the assembled string; under a merging BPE it can differ at the two segment seams
by a merge that would have spanned them. That is the deliberate trade: an exact, honest
scored span beats a span that drifts with the tokenizer's merge table.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .code_suite import (ScoreRow, make_record, score_rows, summarize_bucketed)
from .fim_eval import DEFAULT_BUCKETS, bucket_of

Encode = Callable[[str], Sequence[int]]

#: `export <kind> <Ident>` on one line. `export default ...` and `export * from ...` do not
#: match (no kind keyword follows `export`), which is the fail-closed behaviour we want.
EXPORT_RE = re.compile(
    r"^export\s+(?:async\s+)?(function|const|let|var|class|interface|type|enum)\s+"
    r"([A-Za-z_$][A-Za-z0-9_$]*)",
    re.MULTILINE,
)

#: `import { a, b } from "./mod";` on ONE line — `[^}\n]*` refuses to span a newline, so a
#: multi-line destructured import is skipped rather than half-parsed.
IMPORT_RE = re.compile(
    r"^import\s*\{([^}\n]*)\}\s*from\s*[\"']([^\"']+)[\"']\s*;?",
    re.MULTILINE,
)

#: Only bare specifiers are usable. `{ a as b }`, `{ type T }` and `{ default as x }` are
#: ambiguous about which identifier appears at the use site, so they are dropped.
_BARE_SPECIFIER_RE = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")

_MODULE_SUFFIXES = (".ts", ".tsx", ".mts", ".cts", "/index.ts", "/index.tsx")


# --------------------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------------------- #

def exported_symbols(source: str) -> Dict[str, str]:
    """`{name: kind}` for every unambiguously exported top-level declaration.

    A name declared twice is dropped entirely (ambiguous definition site — we could not say
    which occurrence the use site resolves to).
    """
    counts: Dict[str, int] = {}
    kinds: Dict[str, str] = {}
    for kind, name in EXPORT_RE.findall(source):
        counts[name] = counts.get(name, 0) + 1
        kinds[name] = kind
    return {name: kind for name, kind in kinds.items() if counts[name] == 1}


def imported_symbols(source: str) -> Dict[str, str]:
    """`{name: module_specifier}` for bare, single-line destructured imports.

    A name imported from two different modules is dropped (ambiguous resolution).
    """
    counts: Dict[str, int] = {}
    modules: Dict[str, str] = {}
    for specifiers, module in IMPORT_RE.findall(source):
        for raw in specifiers.split(","):
            name = raw.strip()
            if not _BARE_SPECIFIER_RE.match(name):
                continue                      # `a as b`, `type T`, empty — skip, never guess
            counts[name] = counts.get(name, 0) + 1
            if name in modules and modules[name] != module:
                counts[name] = 99             # force the drop below
            modules[name] = module
    return {name: mod for name, mod in modules.items() if counts[name] == 1}


def resolve_module(specifier: str, importer_path: str, known_paths: Sequence[str]
                   ) -> Optional[str]:
    """Resolve a relative module specifier against the fixture's own path set.

    Only relative specifiers resolve — a bare specifier (`"react"`) is an external package
    with no definition in the bundle, so it yields `None` rather than a guess.
    """
    if not specifier.startswith("."):
        return None
    base = PurePosixPath(importer_path).parent
    target = str((base / specifier).as_posix())
    # PurePosixPath does not normalise "..", so do it by hand over the parts.
    parts: List[str] = []
    for part in target.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    stem = "/".join(parts)
    known = set(known_paths)
    for suffix in ("",) + _MODULE_SUFFIXES:
        candidate = stem + suffix
        if candidate in known:
            return candidate
    return None


def _import_block_end(source: str) -> int:
    """Character offset past the last `import ... from ...` statement, so a use site is
    never matched inside the import that declares it."""
    end = 0
    for m in IMPORT_RE.finditer(source):
        end = max(end, m.end())
    return end


def _declaration_name_end(source: str, name: str) -> Optional[int]:
    """Offset one past the identifier in `name`'s exported declaration, or None."""
    for m in EXPORT_RE.finditer(source):
        if m.group(2) == name:
            return m.end()
    return None


def _first_use_offset(source: str, name: str, after: int) -> Optional[int]:
    """Offset of the first whole-word occurrence of `name` at or after `after`."""
    for m in re.finditer(r"\b" + re.escape(name) + r"\b", source):
        if m.start() >= after:
            return m.start()
    return None


def _tree_sitter_rejects(source: str) -> bool:
    """True when tree-sitter is available AND reports the file has no clean top-level
    statement boundary at all — i.e. the file does not parse. Advisory cross-check only:
    when the optional toolchain is absent this returns False and the regex path stands
    alone (CI/portable hosts do not have tree-sitter)."""
    from ..lsp.ts_boundaries import top_level_boundaries, tree_sitter_available

    if not tree_sitter_available():
        return False
    try:
        return not top_level_boundaries(source)
    except Exception:
        # An unavailable/broken grammar must not take the eval down; it just means the
        # cross-check did not run.
        return False


# --------------------------------------------------------------------------------------- #
# Instance construction
# --------------------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RecallInstance:
    """One cross-file recall instance: a token prefix ending immediately before the use
    site, plus the candidate identifiers to rank at that position."""

    id: str
    prefix_tokens: np.ndarray
    candidates: Tuple[str, ...]
    candidate_tokens: Tuple[np.ndarray, ...]
    answer_index: int
    distance: int
    bucket: str
    definer: str
    user: str
    symbol: str
    n_distractor_files: int


def _encode_array(encode: Encode, text: str) -> np.ndarray:
    return np.asarray(list(encode(text)), dtype=np.int64).reshape(-1)


def build_recall_instances(files: Sequence[dict], encode: Encode, rng: np.random.Generator,
                           *, buckets: Sequence[Tuple[str, int, Optional[int]]] = DEFAULT_BUCKETS,
                           max_distractor_files: int = 24,
                           n_candidates: int = 8,
                           max_instances: Optional[int] = None) -> List[RecallInstance]:
    """Build one instance per (symbol, target bucket) whose distance is reachable.

    For each usable `(definer, user, symbol)` triple and each bucket, the smallest number of
    padding files `k` whose *measured* token distance lands in that bucket is used. A bucket
    no `k <= max_distractor_files` can reach simply produces no instance for that triple —
    the bucket is then reported as `None`, never padded with a synthetic instance.

    `rng` is an explicit `np.random.Generator` (the `fim_eval.build_fim_examples`
    convention): a given seed reproduces a given instance set exactly, which is what makes
    "run the driver twice, diff the JSONL" a real check.
    """
    if n_candidates < 2:
        raise ValueError(f"n_candidates must be >= 2 (answer + >=1 distractor), got {n_candidates}")

    by_path = {f["path"]: f["text"] for f in files}
    paths = sorted(by_path)
    exports = {p: exported_symbols(by_path[p]) for p in paths}
    # Drop files that do not parse when the optional cross-check is available.
    exports = {p: syms for p, syms in exports.items() if not _tree_sitter_rejects(by_path[p])}

    all_symbols = sorted({name for syms in exports.values() for name in syms})
    encoded_files = {p: _encode_array(encode, by_path[p]) for p in paths}
    bucket_names = [b[0] for b in buckets]

    instances: List[RecallInstance] = []
    for user in paths:
        user_text = by_path[user]
        import_end = _import_block_end(user_text)
        for symbol, module in sorted(imported_symbols(user_text).items()):
            definer = resolve_module(module, user, paths)
            if definer is None or symbol not in exports.get(definer, {}):
                continue                                   # unresolved / not exported: skip
            decl_end = _declaration_name_end(by_path[definer], symbol)
            use_at = _first_use_offset(user_text, symbol, import_end)
            if decl_end is None or use_at is None:
                continue

            head = by_path[definer][:decl_end]
            definer_tail = by_path[definer][decl_end:]
            user_head = user_text[:use_at]

            pool = [p for p in paths if p not in (definer, user)]
            order = [pool[i] for i in rng.permutation(len(pool))]
            order = order[:max_distractor_files]

            head_tokens = _encode_array(encode, head)
            if head_tokens.size < 1:
                continue

            distractor_names = [s for s in all_symbols if s != symbol]
            if not distractor_names:
                continue
            take = min(n_candidates - 1, len(distractor_names))
            chosen = [distractor_names[i]
                      for i in rng.choice(len(distractor_names), size=take, replace=False)]
            candidates = tuple(sorted(set(chosen) | {symbol}))
            answer_index = candidates.index(symbol)
            candidate_tokens = tuple(_encode_array(encode, c) for c in candidates)
            if any(t.size < 1 for t in candidate_tokens):
                continue

            for bucket_name in bucket_names:
                inst = _instance_for_bucket(
                    encode=encode, bucket_name=bucket_name, buckets=buckets,
                    head_tokens=head_tokens, definer_tail=definer_tail,
                    user_head=user_head, order=order, encoded_files=encoded_files,
                    candidates=candidates, candidate_tokens=candidate_tokens,
                    answer_index=answer_index, definer=definer, user=user, symbol=symbol,
                )
                if inst is not None:
                    instances.append(inst)
                    if max_instances is not None and len(instances) >= max_instances:
                        return instances
    return instances


def _instance_for_bucket(*, encode: Encode, bucket_name: str,
                         buckets: Sequence[Tuple[str, int, Optional[int]]],
                         head_tokens: np.ndarray, definer_tail: str, user_head: str,
                         order: Sequence[str], encoded_files: Dict[str, np.ndarray],
                         candidates: Tuple[str, ...],
                         candidate_tokens: Tuple[np.ndarray, ...], answer_index: int,
                         definer: str, user: str, symbol: str) -> Optional[RecallInstance]:
    """Smallest `k` padding files whose measured distance lands in `bucket_name`, or None.

    The search is over a monotonically growing prefix of `order`, so it terminates at the
    first hit; distance is measured in **encoded tokens**, which makes the bucketing
    tokenizer-honest rather than a character-count proxy.
    """
    tail_tokens = _encode_array(encode, definer_tail)
    user_tokens = _encode_array(encode, user_head)
    sep_tokens = _encode_array(encode, "\n")

    mid_len = int(tail_tokens.size + user_tokens.size)
    parts: List[np.ndarray] = [tail_tokens]
    for k in range(len(order) + 1):
        if k > 0:
            block = encoded_files[order[k - 1]]
            parts.append(sep_tokens)
            parts.append(block)
            mid_len += int(sep_tokens.size + block.size)
        if bucket_of(mid_len, buckets) == bucket_name:
            mid = np.concatenate(parts + [user_tokens]) if parts else user_tokens
            prefix = np.concatenate([head_tokens, mid])
            return RecallInstance(
                id=f"{user}::{symbol}::{bucket_name}",
                prefix_tokens=prefix, candidates=candidates,
                candidate_tokens=candidate_tokens, answer_index=answer_index,
                distance=mid_len, bucket=bucket_name, definer=definer, user=user,
                symbol=symbol, n_distractor_files=k)
    return None


# --------------------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------------------- #

def evaluate_code_recall(model, instances: Sequence[RecallInstance], *, batch_size: int = 8,
                         pad_id: int = 0, to_numpy=np.asarray,
                         buckets: Sequence[Tuple[str, int, Optional[int]]] = DEFAULT_BUCKETS
                         ) -> dict:
    """Score `instances` and aggregate by distance bucket.

    Every candidate of every instance becomes one right-padded row; the rows are batched
    together so a candidate set costs one forward pass per `batch_size` rows rather than one
    per instance. Right-padding is safe because the model is causal — see
    `code_suite.score_rows`.

    Returns `{"records", "by_bucket", "overall"}`. An empty bucket is `None`; if nothing
    scored at all, `summarize_records` raises rather than reporting a perfect model.
    """
    rows: List[ScoreRow] = []
    owners: List[Tuple[int, int]] = []
    for i, inst in enumerate(instances):
        for j, cand in enumerate(inst.candidate_tokens):
            rows.append(ScoreRow(tokens=np.concatenate([inst.prefix_tokens, cand]),
                                 span_start=int(inst.prefix_tokens.size),
                                 span_len=int(cand.size)))
            owners.append((i, j))

    scored = score_rows(model, rows, pad_id=pad_id, batch_size=batch_size, to_numpy=to_numpy)

    per_instance: List[Dict[int, dict]] = [dict() for _ in instances]
    for (i, j), s in zip(owners, scored):
        per_instance[i][j] = s

    records: List[dict] = []
    for i, inst in enumerate(instances):
        answer = per_instance[i][inst.answer_index]
        # Rank ascending by total CE (= descending summed log-prob). Ties break on the
        # candidate string so the ordering — and therefore `mrr` — is deterministic.
        ordering = sorted(range(len(inst.candidates)),
                          key=lambda j: (per_instance[i][j]["total_ce_nats"], inst.candidates[j]))
        rank = ordering.index(inst.answer_index) + 1
        records.append(make_record(
            suite="code_recall", id=inst.id, bucket=inst.bucket, distance=inst.distance,
            n_scored_tokens=answer["n_scored_tokens"], ce_nats=answer["ce_nats"],
            token_accuracy=answer["token_accuracy"], exact_match=answer["exact_match"],
            rank_top1=(rank == 1), mrr=1.0 / rank,
            meta={"definer": inst.definer, "user": inst.user, "symbol": inst.symbol,
                  "n_distractor_files": inst.n_distractor_files,
                  "n_candidates": len(inst.candidates), "rank": rank},
        ))

    summary = summarize_bucketed(records, [b[0] for b in buckets])
    return {"records": records, **summary}
