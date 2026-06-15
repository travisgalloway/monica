"""Stage 6: blend cleaned sources at natural size (#74).

No fixed-ratio downsampling — take the core sets whole and set **epoch counts** so the
small supplements show up enough to matter (``docs/design/08-corpus-pipeline.md`` line 55):
web/code 1 pass; wiki/math/docs 2–3 passes; keep any source under ~4 passes (past that
repetition stops helping). Priority code languages (TypeScript, Rust, SQL) get extra
passes — they're priority langs and the permissive filter (#72) trims TS/SQL harder.

ABOVE THE SEAM — stdlib + the corpus reader only; no ``mlx``/``torch``, no heavy deps.
Streams with a bounded shuffle buffer so the blend stays memory-light and deterministic
(fixed seed). The blended Record stream feeds tokenize -> ``shard.pack_sequences``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Mapping

from .corpus import Record, read_shards


@dataclass
class BlendSpec:
    """Per-source epoch counts + priority-language oversampling. Surfaced as config
    (the #74 tunables): `passes` per source, a `default_passes`, a `max_passes` cap, and
    `priority_langs` -> extra passes applied to matching code records within each source."""

    passes: Dict[str, int] = field(default_factory=dict)
    default_passes: int = 1
    max_passes: int = 4
    priority_langs: Dict[str, int] = field(default_factory=dict)

    def passes_for(self, source: str) -> int:
        """Epoch count for a source, clamped to `max_passes` (>=1)."""
        return max(1, min(self.passes.get(source, self.default_passes), self.max_passes))

    def priority_passes(self) -> Dict[str, int]:
        """Priority-language extra passes, each clamped to `max_passes`."""
        return {lang.lower(): max(0, min(n, self.max_passes))
                for lang, n in self.priority_langs.items()}


def _lang_filter(uri: str, lang: str) -> Callable[[], Iterator[Record]]:
    return lambda: (r for r in read_shards(uri) if (r.lang or "").lower() == lang)


def _interleave(streams: List[Iterator[Record]], rng: random.Random,
                buffer_size: int) -> Iterator[Record]:
    """Round-robin across streams into a bounded shuffle buffer; emit random buffer items."""
    active = list(streams)
    buf: List[Record] = []
    while active:
        still: List[Iterator[Record]] = []
        for s in active:
            try:
                buf.append(next(s))
                still.append(s)
            except StopIteration:
                continue
            if len(buf) >= buffer_size:
                yield buf.pop(rng.randrange(len(buf)))
        active = still
    rng.shuffle(buf)
    yield from buf


def blend(source_uris: Mapping[str, str], spec: BlendSpec, *, seed: int = 0,
          buffer_size: int = 2048) -> Iterator[Record]:
    """Interleave each source's cleaned shards at its epoch count (plus priority-language
    oversampling), with a deterministic bounded shuffle. `source_uris` maps a source name
    to its cleaned-shard URI (dir / `.parquet` / `file://` / `s3://`)."""
    rng = random.Random(seed)
    factories: List[Callable[[], Iterator[Record]]] = []
    prio = spec.priority_passes()
    for source, uri in source_uris.items():
        for _ in range(spec.passes_for(source)):
            factories.append(lambda u=uri: read_shards(u))
        for lang, extra in prio.items():
            for _ in range(extra):
                factories.append(_lang_filter(uri, lang))
    yield from _interleave([f() for f in factories], rng, buffer_size)
