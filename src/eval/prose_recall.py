"""Technical prose & general knowledge recall probe (prose_recall) (#363).

ABOVE THE SEAM. Pure numpy + stdlib; never imports `mlx`, `torch`, or hardware backends.

What it measures and architectural motivation
----------------------------------------------
Monica's MHM spine employs a mostly Mamba-2/SSD recurrent backbone with ~12.5% hybrid
attention layers. Pure state-space models compress context into a fixed-width hidden
state (d_state), leading to the "phone book problem" where associative recall decays
over long token distances. While issue #221 introduced `src/eval/code_recall.py` for
cross-file symbol resolution in code, 45% of Monica's pretraining mix is non-code (15%
curated web, 15% architecture & technical prose, 15% math & logic).

Factual and conceptual recall over specifications (RFCs, PEPs, system design docs)
must be validated under long contexts to verify that hybrid attention layers prevent
factual state washout.

Design & Scoring Discipline
---------------------------
Synthetic probe instances present a key specification invariant or factual definition,
followed by k tokens of unrelated distractor prose (bucketed at 512, 1024, 2048, 4096,
8192, 16384 tokens), followed by a focused query site.

Two metrics, both teacher-forced (no generation, no pass@1 gating):
* **Cross-Entropy loss / token accuracy / exact match** over the target entity span.
* **Discriminative rank** against a closed candidate pool of in-domain near-miss
  distractors (e.g. matching RFC status codes, methods, or headers), ranked by summed
  negative log-probabilities (ascending total CE in nats). `rank_top1` and `mrr`
  quantify whether the model reliably retrieves the factual definition across distance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .code_suite import (
    ScoreRow,
    load_code_files,
    make_record,
    read_jsonl,
    score_rows,
    summarize_bucketed,
)
from .fim_eval import bucket_of

Encode = Callable[[str], Sequence[int]]

#: Target separation distances (in tokens) between definition statement and query site.
DEFAULT_DISTANCES: Tuple[int, ...] = (512, 1024, 2048, 4096, 8192, 16384)

#: Distance buckets: (name, lo_inclusive, hi_exclusive_or_None).
#: Log2 geometric midpoint intervals partition token distances cleanly into standard buckets.
PROSE_DEFAULT_BUCKETS: Tuple[Tuple[str, int, Optional[int]], ...] = (
    ("512", 0, 768),
    ("1024", 768, 1536),
    ("2048", 1536, 3072),
    ("4096", 3072, 6144),
    ("8192", 6144, 12288),
    ("16384", 12288, None),
)

PROSE_BUCKET_NAMES: Tuple[str, ...] = tuple(b[0] for b in PROSE_DEFAULT_BUCKETS)


def prose_bucket_for_distance(
    distance: int,
    buckets: Sequence[Tuple[str, int, Optional[int]]] = PROSE_DEFAULT_BUCKETS,
) -> str:
    """Return bucket name for distance in tokens."""
    return bucket_of(int(distance), buckets)


# --------------------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ProseRecallSpec:
    """One specification definition or invariant with query and candidate pool."""

    id: str
    domain: str
    statement: str
    query: str
    target: str
    candidates: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("ProseRecallSpec: id must be non-empty")
        if not self.statement.strip():
            raise ValueError(f"ProseRecallSpec {self.id}: statement must be non-empty")
        if not self.query.strip():
            raise ValueError(f"ProseRecallSpec {self.id}: query must be non-empty")
        if not self.target:
            raise ValueError(f"ProseRecallSpec {self.id}: target must be non-empty")
        if self.target not in self.candidates:
            raise ValueError(
                f"ProseRecallSpec {self.id}: target {self.target!r} not in candidates {self.candidates}"
            )
        if len(self.candidates) < 2:
            raise ValueError(
                f"ProseRecallSpec {self.id}: candidates must have >= 2 items (answer + distractor), got {len(self.candidates)}"
            )


@dataclass(frozen=True)
class ProseRecallInstance:
    """One prose recall instance with prefix tokens, target candidates, and metadata."""

    id: str
    prefix_tokens: np.ndarray
    candidates: Tuple[str, ...]
    candidate_tokens: Tuple[np.ndarray, ...]
    answer_index: int
    distance: int
    bucket: str
    domain: str
    target: str
    statement: str
    query: str


def _encode_array(encode: Encode, text: str) -> np.ndarray:
    return np.asarray(list(encode(text)), dtype=np.int64).reshape(-1)


# --------------------------------------------------------------------------------------- #
# Fixture loading
# --------------------------------------------------------------------------------------- #

def load_prose_specs(path: Union[str, Path]) -> List[ProseRecallSpec]:
    """Load specification probes from JSONL, validating required fields."""
    rows = read_jsonl(path)
    specs: List[ProseRecallSpec] = []
    for r in rows:
        cands = tuple(r.get("candidates", []))
        spec = ProseRecallSpec(
            id=str(r.get("id", "")),
            domain=str(r.get("domain", "prose")),
            statement=str(r.get("statement", "")),
            query=str(r.get("query", "")),
            target=str(r.get("target", "")),
            candidates=cands,
        )
        specs.append(spec)
    return sorted(specs, key=lambda s: s.id)


def load_distractor_texts(path: Union[str, Path]) -> List[str]:
    """Load distractor prose texts from JSONL, sorted by path for determinism."""
    files = load_code_files(path)
    return [f["text"] for f in files if f.get("text")]


# --------------------------------------------------------------------------------------- #
# Instance construction
# --------------------------------------------------------------------------------------- #

def _tile_distractor_tokens(
    blocks: Sequence[np.ndarray],
    budget: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Concatenate distractor token blocks to exactly `budget` tokens."""
    if budget <= 0:
        return np.zeros((0,), dtype=np.int64)
    if not blocks:
        raise ValueError("_tile_distractor_tokens(): no distractor blocks provided")

    order = list(range(len(blocks)))
    if rng is not None:
        order = list(rng.permutation(len(blocks)))

    out: List[np.ndarray] = []
    total = 0
    idx = 0
    while total < budget:
        block = blocks[order[idx % len(order)]]
        if block.size == 0:
            idx += 1
            if idx > len(blocks) * 2:
                raise ValueError("_tile_distractor_tokens(): all distractor blocks are empty")
            continue
        take = min(int(block.size), budget - total)
        out.append(block[:take])
        total += take
        idx += 1
    return np.concatenate(out)


def build_prose_recall_instances(
    specs: Sequence[Union[dict, ProseRecallSpec]],
    distractors: Sequence[Union[dict, str]],
    encode: Encode,
    rng: np.random.Generator,
    *,
    distances: Sequence[int] = DEFAULT_DISTANCES,
    buckets: Sequence[Tuple[str, int, Optional[int]]] = PROSE_DEFAULT_BUCKETS,
    n_candidates: Optional[int] = None,
    max_instances: Optional[int] = None,
) -> List[ProseRecallInstance]:
    """Build recall instances across specification probes and target distance buckets.

    For each probe specification and each requested distance k, k tokens of unrelated
    distractor prose are placed between the specification definition and the query site.
    Scoring ranks candidate entities at the query position.
    """
    if not specs:
        return []
    if not distractors:
        raise ValueError("build_prose_recall_instances(): distractors cannot be empty")

    # Normalize specs
    norm_specs: List[ProseRecallSpec] = []
    for s in specs:
        if isinstance(s, ProseRecallSpec):
            norm_specs.append(s)
        else:
            cands = tuple(s.get("candidates", []))
            norm_specs.append(ProseRecallSpec(
                id=str(s.get("id", "")),
                domain=str(s.get("domain", "prose")),
                statement=str(s.get("statement", "")),
                query=str(s.get("query", "")),
                target=str(s.get("target", "")),
                candidates=cands,
            ))

    # Normalize distractor token blocks
    distractor_texts: List[str] = []
    for d in distractors:
        if isinstance(d, dict):
            distractor_texts.append(d.get("text", ""))
        else:
            distractor_texts.append(str(d))

    dist_blocks = [_encode_array(encode, t) for t in distractor_texts if t.strip()]
    if not dist_blocks:
        raise ValueError("build_prose_recall_instances(): no non-empty distractor texts")

    instances: List[ProseRecallInstance] = []

    for spec in norm_specs:
        # Candidate selection
        cands = list(spec.candidates)
        if n_candidates is not None and n_candidates >= 2 and len(cands) > n_candidates:
            # Keep target and randomly draw distractors
            other_cands = [c for c in cands if c != spec.target]
            drawn = [other_cands[i] for i in rng.choice(len(other_cands), size=n_candidates - 1, replace=False)]
            cands = sorted(set(drawn) | {spec.target})
        else:
            cands = sorted(set(cands))

        candidates_tuple = tuple(cands)
        answer_index = candidates_tuple.index(spec.target)
        candidate_tokens = tuple(_encode_array(encode, c) for c in candidates_tuple)
        if any(ct.size < 1 for ct in candidate_tokens):
            continue

        head_text = f"Specification Invariant:\n{spec.statement}\n\nTechnical Notes:\n"
        query_text = f"\n\nSpecification Query:\n{spec.query}"
        head_tokens = _encode_array(encode, head_text)
        query_tokens = _encode_array(encode, query_text)

        for dist in distances:
            bucket_name = prose_bucket_for_distance(dist, buckets)
            dist_tokens = _tile_distractor_tokens(dist_blocks, dist, rng=rng)
            prefix = np.concatenate([head_tokens, dist_tokens, query_tokens])

            inst_id = f"{spec.id}::{bucket_name}"
            inst = ProseRecallInstance(
                id=inst_id,
                prefix_tokens=prefix,
                candidates=candidates_tuple,
                candidate_tokens=candidate_tokens,
                answer_index=answer_index,
                distance=dist,
                bucket=bucket_name,
                domain=spec.domain,
                target=spec.target,
                statement=spec.statement,
                query=spec.query,
            )
            instances.append(inst)
            if max_instances is not None and len(instances) >= max_instances:
                return instances

    return instances


# --------------------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------------------- #

def evaluate_prose_recall(
    model,
    instances: Sequence[ProseRecallInstance],
    *,
    batch_size: int = 8,
    pad_id: int = 0,
    to_numpy=np.asarray,
    buckets: Sequence[Tuple[str, int, Optional[int]]] = PROSE_DEFAULT_BUCKETS,
) -> dict:
    """Score prose recall instances teacher-forced and aggregate by distance bucket.

    For each instance, all candidates are scored under right-padding causal batching.
    Candidates are ranked ascending by total CE (descending summed log-probabilities).
    Emits shared schema records and returns `{'records', 'by_bucket', 'overall'}`.
    """
    if not instances:
        raise ValueError("evaluate_prose_recall(): nothing scored — instances cannot be empty")

    rows: List[ScoreRow] = []
    owners: List[Tuple[int, int]] = []
    for i, inst in enumerate(instances):
        for j, cand in enumerate(inst.candidate_tokens):
            rows.append(ScoreRow(
                tokens=np.concatenate([inst.prefix_tokens, cand]),
                span_start=int(inst.prefix_tokens.size),
                span_len=int(cand.size),
            ))
            owners.append((i, j))

    scored = score_rows(model, rows, pad_id=pad_id, batch_size=batch_size, to_numpy=to_numpy)

    per_instance: List[Dict[int, dict]] = [dict() for _ in instances]
    for (i, j), s in zip(owners, scored):
        per_instance[i][j] = s

    records: List[dict] = []
    for i, inst in enumerate(instances):
        answer = per_instance[i][inst.answer_index]
        # Rank ascending by total CE (= descending summed log-prob). Ties break on candidate string.
        ordering = sorted(
            range(len(inst.candidates)),
            key=lambda j: (per_instance[i][j]["total_ce_nats"], inst.candidates[j]),
        )
        rank = ordering.index(inst.answer_index) + 1

        records.append(make_record(
            suite="prose_recall",
            id=inst.id,
            bucket=inst.bucket,
            distance=inst.distance,
            n_scored_tokens=answer["n_scored_tokens"],
            ce_nats=answer["ce_nats"],
            token_accuracy=answer["token_accuracy"],
            exact_match=answer["exact_match"],
            rank_top1=(rank == 1),
            mrr=1.0 / rank,
            meta={
                "domain": inst.domain,
                "target": inst.target,
                "n_candidates": len(inst.candidates),
                "rank": rank,
            },
        ))

    summary = summarize_bucketed(records, [b[0] for b in buckets])
    return {"records": records, **summary}
