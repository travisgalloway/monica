"""RULER-over-code: a needle-in-a-haystack retrieval probe in TypeScript space (#221).

ABOVE THE SEAM. Pure numpy + stdlib — never imports `mlx` or `torch`.

What it measures
----------------
A syntactically valid TS declaration carrying an opaque random value —

    export const MONICA_NEEDLE_<KEY> = "<VALUE>";

— is planted at a controlled depth inside a haystack of ordinary code, and the value is
queried at the very end. Scoring is teacher-forced over the value's token span: exact match
is the headline, with token accuracy and CE alongside. Nothing generates, so there is no
pass@1 gate.

The grid, not the scalar, is the point. Results are reported over
`context_len x depth` (RULER's grid), because a fixed-width recurrent state fails as a
*shape* — degrading with context length, and asymmetrically in depth — and a single
averaged retrieval number hides exactly that. `multikey` (plant N needles, query one) adds
capacity pressure on top of distance. `multivalue` (one key, several values) is
deliberately out of scope here; it is the natural next variant.

Relation to `src/eval/probes.py`
--------------------------------
`probes.make_needle_batch` plants its needle in a **synthetic disjoint id space**, which is
the right design for an architecture unit-probe and the wrong one for a code model: it
cannot tell you whether recall survives when the needle looks like the surrounding code.
Only `retrieval_probe.recall_accuracy` is genuinely shared (argmax accuracy over a masked
span); the rest is precedent, not reuse.

Tokenization
------------
Like `src/eval/code_recall.py`, this module takes an injected
`encode: Callable[[str], Sequence[int]]` (Python has no code tokenizer since #245) and
encodes segments separately so the scored value span is exactly the value's own tokens. The
haystack is truncated at a **token** boundary to hit `context_len` exactly, so its final
filler file may be cut mid-statement — harmless, since the filler is a distractor by
construction and is never scored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .code_suite import ScoreRow, make_record, score_rows, summarize_bucketed

Encode = Callable[[str], Sequence[int]]

#: RULER's classic depth grid: the needle's fractional position in the haystack.
DEFAULT_DEPTHS: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

#: Context lengths in tokens. Small by default so the fixture-only offline run is quick;
#: a real evaluation passes the model's actual context ladder.
DEFAULT_CONTEXT_LENS: Tuple[int, ...] = (512, 1024, 2048)

_KEY_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
_VALUE_ALPHABET = "abcdefghijkmnpqrstuvwxyz23456789"


def bucket_name(context_len: int, depth: float) -> str:
    """Grid-cell name. Fixed formatting so two runs produce identical bucket keys."""
    return f"ctx{int(context_len)}_d{float(depth):.2f}"


@dataclass(frozen=True)
class NeedleInstance:
    """One planted-needle instance; `span_start`/`span_len` locate the value's tokens."""

    id: str
    tokens: np.ndarray
    span_start: int
    span_len: int
    bucket: str
    distance: int
    context_len: int
    depth: float
    variant: str
    key: str
    value: str
    n_needles: int


def _encode_array(encode: Encode, text: str) -> np.ndarray:
    return np.asarray(list(encode(text)), dtype=np.int64).reshape(-1)


def _draw(rng: np.random.Generator, alphabet: str, n: int) -> str:
    return "".join(alphabet[i] for i in rng.integers(0, len(alphabet), size=n))


def needle_text(key: str, value: str) -> str:
    """The planted declaration. Valid TS, and shaped like the surrounding code so the model
    cannot find it by format alone."""
    return f'\nexport const MONICA_NEEDLE_{key} = "{value}";\n'


def query_text(key: str) -> str:
    """The trailing use site. The scored span is what follows the open quote."""
    return f'\nconst check = MONICA_NEEDLE_{key};\n// value: "'


def _tile_to(blocks: Sequence[np.ndarray], budget: int) -> np.ndarray:
    """Concatenate `blocks` cyclically to exactly `budget` tokens (truncating the last)."""
    if budget <= 0:
        return np.zeros((0,), dtype=np.int64)
    if not blocks:
        raise ValueError("_tile_to(): no filler blocks — the haystack fixture is empty")
    out: List[np.ndarray] = []
    total = 0
    i = 0
    while total < budget:
        block = blocks[i % len(blocks)]
        if block.size == 0:
            i += 1
            if i > len(blocks) * 2:
                raise ValueError("_tile_to(): every filler block is empty")
            continue
        take = min(int(block.size), budget - total)
        out.append(block[:take])
        total += take
        i += 1
    return np.concatenate(out)


def build_needle_instances(fillers: Sequence[dict], encode: Encode, rng: np.random.Generator,
                           *, context_lens: Sequence[int] = DEFAULT_CONTEXT_LENS,
                           depths: Sequence[float] = DEFAULT_DEPTHS,
                           variants: Sequence[str] = ("single",),
                           n_needles: int = 4,
                           n_per_cell: int = 1) -> List[NeedleInstance]:
    """Build the `context_len x depth x variant` grid of instances.

    A cell whose `context_len` is too small to hold the needles plus the query simply
    produces **no instance** — it is then reported as an empty bucket (`None`), never as a
    zero score. `rng` is explicit, so a seed reproduces the whole grid exactly.
    """
    for variant in variants:
        if variant not in ("single", "multikey"):
            raise ValueError(f"unknown needle variant {variant!r} (want 'single' or 'multikey')")
    if n_needles < 1:
        raise ValueError(f"n_needles must be >= 1, got {n_needles}")

    blocks = [_encode_array(encode, f["text"]) for f in fillers]
    instances: List[NeedleInstance] = []

    for context_len in context_lens:
        for depth in depths:
            for variant in variants:
                for rep in range(n_per_cell):
                    inst = _build_one(blocks, encode, rng, context_len=int(context_len),
                                      depth=float(depth), variant=variant,
                                      n_needles=(n_needles if variant == "multikey" else 1),
                                      rep=rep)
                    if inst is not None:
                        instances.append(inst)
    return instances


def _build_one(blocks: Sequence[np.ndarray], encode: Encode, rng: np.random.Generator, *,
               context_len: int, depth: float, variant: str, n_needles: int,
               rep: int) -> Optional[NeedleInstance]:
    keys = [_draw(rng, _KEY_ALPHABET, 8) for _ in range(n_needles)]
    values = [_draw(rng, _VALUE_ALPHABET, 12) for _ in range(n_needles)]
    needles = [_encode_array(encode, needle_text(k, v)) for k, v in zip(keys, values)]
    query = _encode_array(encode, query_text(keys[0]))
    value_tokens = _encode_array(encode, values[0])

    overhead = int(sum(int(n.size) for n in needles) + query.size + value_tokens.size)
    budget = context_len - overhead
    if budget < 1 or value_tokens.size < 1:
        return None                       # cell too small — no instance, no fake score

    haystack = _tile_to(blocks, budget)

    # Position each needle in haystack coordinates. The queried needle sits at `depth`;
    # the rest are spread evenly so `multikey` adds capacity pressure, not extra distance.
    placements: List[Tuple[int, int, np.ndarray]] = [
        (int(round(depth * budget)), 0, needles[0])]
    for i in range(1, n_needles):
        placements.append((int(round(budget * i / (n_needles + 1))), i, needles[i]))
    placements.sort(key=lambda p: (p[0], p[1]))

    parts: List[np.ndarray] = []
    total = 0
    prev = 0
    query_needle_end: Optional[int] = None
    for pos, order, toks in placements:
        chunk = haystack[prev:pos]
        parts.append(chunk)
        total += int(chunk.size)
        prev = pos
        parts.append(toks)
        total += int(toks.size)
        if order == 0:
            query_needle_end = total
    tail = haystack[prev:]
    parts.append(tail)
    total += int(tail.size)

    context = np.concatenate(parts + [query, value_tokens])
    span_start = total + int(query.size)
    assert query_needle_end is not None
    distance = span_start - query_needle_end

    return NeedleInstance(
        id=f"{bucket_name(context_len, depth)}_{variant}_{rep:02d}",
        tokens=context, span_start=span_start, span_len=int(value_tokens.size),
        bucket=bucket_name(context_len, depth), distance=int(distance),
        context_len=int(context_len), depth=float(depth), variant=variant,
        key=keys[0], value=values[0], n_needles=n_needles)


def evaluate_code_needle(model, instances: Sequence[NeedleInstance], *, batch_size: int = 4,
                         pad_id: int = 0, to_numpy=np.asarray,
                         context_lens: Sequence[int] = DEFAULT_CONTEXT_LENS,
                         depths: Sequence[float] = DEFAULT_DEPTHS) -> dict:
    """Score the grid. Returns `{"records", "by_bucket", "overall"}`.

    `by_bucket` is keyed by `ctx<N>_d<D>` and always lists every requested grid cell, so a
    cell that produced no instance shows up as `None` rather than vanishing from the table —
    a missing cell and a measured-zero cell must not look alike.
    """
    rows = [ScoreRow(tokens=i.tokens, span_start=i.span_start, span_len=i.span_len)
            for i in instances]
    scored = score_rows(model, rows, pad_id=pad_id, batch_size=batch_size, to_numpy=to_numpy)

    records = [
        make_record(suite="code_needle", id=inst.id, bucket=inst.bucket,
                    distance=inst.distance, n_scored_tokens=s["n_scored_tokens"],
                    ce_nats=s["ce_nats"], token_accuracy=s["token_accuracy"],
                    exact_match=s["exact_match"],
                    meta={"context_len": inst.context_len, "depth": inst.depth,
                          "variant": inst.variant, "n_needles": inst.n_needles,
                          "key": inst.key})
        for inst, s in zip(instances, scored)
    ]
    grid = [bucket_name(c, d) for c in context_lens for d in depths]
    return {"records": records, **summarize_bucketed(records, grid)}
