"""Stage 4–5: cross-source dedup + benchmark decontamination (#73).

The heaviest corpus stage (``docs/design/08-corpus-pipeline.md`` lines 53–54): exact
dedup first, then **MinHash-LSH** near-dedup run *cross-source* (supplements overlap each
other and the web spine), then **decontamination** that strips docs overlapping the eval
benchmarks on a mixed 13-gram + 7-gram scheme.

ABOVE THE SEAM — numpy + stdlib ``hashlib`` only, no ``mlx``/``torch`` and no heavy data
deps. Everything streams (the dedup state — hash set / LSH buckets — is held in memory but
records flow through lazily), so it chains onto ``filter_records`` without materializing
the corpus. At scale (#80) ``datatrove``'s distributed exact-hash + MinHash stages replace
the in-process engine here; the schema and semantics match.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np

from .corpus import Record

# Mersenne prime modulus + 32-bit shingle hashes keep the affine MinHash arithmetic
# inside uint64 without overflow (a<2^31, hv<2^32 -> a*hv < 2^63).
_PRIME = (1 << 61) - 1
_MAX32 = (1 << 32) - 1


@dataclass
class DedupStats:
    """Per-stage drop counters (mirrors filters.FilterStats)."""

    dropped_exact: int = 0
    dropped_near: int = 0
    dropped_decontam: int = 0

    def as_dict(self) -> dict:
        return {"dropped_exact": self.dropped_exact, "dropped_near": self.dropped_near,
                "dropped_decontam": self.dropped_decontam}


# --------------------------------------------------------------------------- #
# Shingling + hashing helpers
# --------------------------------------------------------------------------- #
def _hash32(s: str) -> int:
    """Stable 32-bit hash of a shingle (process-independent, unlike ``hash()``)."""
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=4).digest(), "big")


def shingles(text: str, size: int = 5) -> Set[str]:
    """Word n-gram shingles (lowercased). Docs shorter than `size` become one shingle."""
    words = text.lower().split()
    if not words:
        return set()
    if len(words) < size:
        return {" ".join(words)}
    return {" ".join(words[i:i + size]) for i in range(len(words) - size + 1)}


# --------------------------------------------------------------------------- #
# Exact dedup
# --------------------------------------------------------------------------- #
def exact_dedup(records: Iterable[Record], stats: Optional[DedupStats] = None,
                ) -> Iterator[Record]:
    """Drop byte-identical texts (sha1), keeping the first occurrence."""
    seen: Set[str] = set()
    for r in records:
        h = hashlib.sha1(r.text.encode("utf-8")).hexdigest()
        if h in seen:
            if stats is not None:
                stats.dropped_exact += 1
            continue
        seen.add(h)
        yield r


# --------------------------------------------------------------------------- #
# MinHash LSH near-dedup
# --------------------------------------------------------------------------- #
class MinHasher:
    """Deterministic affine MinHash family: ``h_i(x) = (a_i*x + b_i) mod P``."""

    def __init__(self, num_perm: int = 128, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.num_perm = num_perm
        self.a = rng.integers(1, 1 << 31, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, 1 << 31, size=num_perm, dtype=np.uint64)

    def signature(self, shs: Set[str]) -> np.ndarray:
        """MinHash signature (uint64 array of length num_perm). Empty set -> all-max."""
        if not shs:
            return np.full(self.num_perm, _PRIME, dtype=np.uint64)
        hv = np.fromiter((_hash32(s) & _MAX32 for s in shs), dtype=np.uint64, count=len(shs))
        # (num_perm, k): affine-permute every shingle hash, then take the per-row min.
        mixed = (self.a[:, None] * hv[None, :] + self.b[:, None]) % _PRIME
        return mixed.min(axis=1)


def jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Estimated Jaccard similarity = fraction of agreeing MinHash slots."""
    return float(np.mean(sig_a == sig_b))


def _band_keys(sig: np.ndarray, bands: int, rows: int) -> List[bytes]:
    """LSH band keys: a doc is a near-dup candidate of another if any band matches."""
    return [i.to_bytes(2, "big") + sig[i * rows:(i + 1) * rows].tobytes()
            for i in range(bands)]


class MinHashLSH:
    """Streaming cross-source near-dedup via banded LSH over MinHash signatures."""

    def __init__(self, threshold: float = 0.8, num_perm: int = 128, bands: int = 16,
                 shingle_size: int = 5, seed: int = 0):
        if num_perm % bands:
            raise ValueError(f"num_perm ({num_perm}) must be divisible by bands ({bands})")
        self.threshold = threshold
        self.bands = bands
        self.rows = num_perm // bands
        self.shingle_size = shingle_size
        self.hasher = MinHasher(num_perm, seed)
        self._buckets: dict = {}        # band key -> kept-signature indices
        self._kept: List[np.ndarray] = []

    def _is_duplicate(self, sig: np.ndarray, keys: Sequence[bytes]) -> bool:
        seen_idx: Set[int] = set()
        for k in keys:
            seen_idx.update(self._buckets.get(k, ()))
        return any(jaccard(sig, self._kept[i]) >= self.threshold for i in seen_idx)

    def add(self, text: str) -> bool:
        """Register `text`; return True if kept (not a near-dup of an earlier doc)."""
        sig = self.hasher.signature(shingles(text, self.shingle_size))
        keys = _band_keys(sig, self.bands, self.rows)
        if self._is_duplicate(sig, keys):
            return False
        idx = len(self._kept)
        self._kept.append(sig)
        for k in keys:
            self._buckets.setdefault(k, []).append(idx)
        return True


def near_dedup(records: Iterable[Record], threshold: float = 0.8, num_perm: int = 128,
               bands: int = 16, shingle_size: int = 5, seed: int = 0,
               stats: Optional[DedupStats] = None) -> Iterator[Record]:
    """Stream records through MinHash-LSH near-dedup, dropping near-duplicates."""
    lsh = MinHashLSH(threshold, num_perm, bands, shingle_size, seed)
    for r in records:
        if lsh.add(r.text):
            yield r
        elif stats is not None:
            stats.dropped_near += 1


# --------------------------------------------------------------------------- #
# Benchmark decontamination (13-gram + 7-gram)
# --------------------------------------------------------------------------- #
class Decontaminator:
    """Holds the eval-benchmark n-gram set; flags training docs that overlap it."""

    def __init__(self, ngrams: Set[str], ngram_sizes: Tuple[int, ...] = (13, 7)):
        self.ngrams = ngrams
        self.sizes = ngram_sizes

    @classmethod
    def from_texts(cls, texts: Iterable[str], ngram_sizes: Tuple[int, ...] = (13, 7),
                   ) -> "Decontaminator":
        grams: Set[str] = set()
        for t in texts:
            words = t.lower().split()
            for n in ngram_sizes:
                for i in range(len(words) - n + 1):
                    grams.add(" ".join(words[i:i + n]))
        return cls(grams, ngram_sizes)

    def contaminated(self, text: str) -> bool:
        words = text.lower().split()
        for n in self.sizes:
            for i in range(len(words) - n + 1):
                if " ".join(words[i:i + n]) in self.ngrams:
                    return True
        return False


def decontaminate(records: Iterable[Record], decon: Decontaminator,
                  stats: Optional[DedupStats] = None) -> Iterator[Record]:
    """Drop docs whose n-grams overlap the eval benchmarks (re-run when evals change)."""
    for r in records:
        if decon.contaminated(r.text):
            if stats is not None:
                stats.dropped_decontam += 1
            continue
        yield r
