"""WSD decay-phase general knowledge experience replay buffer (#364).

ABOVE THE SEAM: pure numpy + stdlib, zero hardware backend imports.

Implements scheduled experience replay during the WSD learning-rate decay phase to
prevent catastrophic forgetting of foundational non-code knowledge (curated web,
math/logic, and technical prose).

The replay loader draws from domain shards corresponding to:
  * curated web (Essential-Web / FineWeb-Edu)
  * math and logic
  * technical prose and documentation

Exposes the duck-typed loader contract required by MicroBatchStream:
  `__len__`, `epoch(reseed, skip_batches)`, `.seq_len`, `.batch_size`, `.rng`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union

import numpy as np

from ..data.loader import PackedLoader

REPLAY_DOMAINS: dict[str, tuple[str, ...]] = {
    "web": ("fineweb", "fineweb-edu", "fineweb_edu", "essential-web", "essential_web", "web"),
    "math": ("math", "logic", "math-logic", "math_logic"),
    "prose": ("technical-prose", "technical_prose", "prose", "docs", "wikipedia", "arxiv"),
}


def _match_domain(name: str) -> Optional[str]:
    """Match a directory or filename against target general knowledge domains."""
    clean = name.lower().replace("_", "-")
    for domain, aliases in REPLAY_DOMAINS.items():
        for alias in aliases:
            norm_alias = alias.replace("_", "-")
            if norm_alias == clean or norm_alias in clean.split("-"):
                return domain
    return None


def discover_replay_shards(replay_dir: Union[Path, str]) -> Dict[str, List[Path]]:
    """Scan `replay_dir` for general knowledge domain shards.

    Searches for:
      1. Domain subdirectories (e.g. `web/train.bin`, `math/part-00000.bin`).
      2. Domain-tagged files in `replay_dir` (e.g. `fineweb.bin`, `math.bin`, `prose.bin`).
      3. Fallback to all `.bin` files in `replay_dir` if no explicit domain tags match.

    Raises FileNotFoundError if `replay_dir` does not exist or contains no `.bin` files.
    """
    root = Path(replay_dir)
    if not root.exists():
        raise FileNotFoundError(f"replay data directory does not exist: {root}")

    domain_shards: Dict[str, List[Path]] = {d: [] for d in REPLAY_DOMAINS}
    all_bin_files: List[Path] = []

    # Check for domain subdirectories first
    for child in sorted(root.iterdir()):
        if child.is_dir():
            d = _match_domain(child.name)
            bins = sorted(child.glob("*.bin"))
            if d is not None and bins:
                domain_shards[d].extend(bins)
            elif bins:
                all_bin_files.extend(bins)
        elif child.is_file() and child.suffix == ".bin":
            d = _match_domain(child.stem)
            if d is not None:
                domain_shards[d].append(child)
            else:
                all_bin_files.append(child)

    # If any specific domains matched, return those that have files
    active = {d: files for d, files in domain_shards.items() if files}
    if active:
        return active

    # If no domain-tagged structure, use all .bin files under a general category
    if all_bin_files:
        return {"general": sorted(all_bin_files)}

    # Check recursively for any .bin files in nested dirs
    recursive_bins = sorted(root.rglob("*.bin"))
    if recursive_bins:
        for b in recursive_bins:
            d = _match_domain(b.stem) or _match_domain(b.parent.name)
            if d is not None:
                domain_shards[d].append(b)
        active = {d: files for d, files in domain_shards.items() if files}
        if active:
            return active
        return {"general": recursive_bins}

    raise FileNotFoundError(f"no packed .bin files found in replay directory: {root}")


class ReplayLoader:
    """Composite loader drawing from general knowledge domains in a balanced, deterministic manner.

    Satisfies the duck-typed loader contract:
      * `__len__()`
      * `epoch(reseed, skip_batches)`
      * `seq_len`
      * `batch_size`
      * `rng`
    """

    def __init__(self, loaders: Sequence[Any], *, seed: int = 0):
        if not loaders:
            raise ValueError("ReplayLoader requires at least one constituent loader")
        self.loaders = list(loaders)
        self.seq_len = self.loaders[0].seq_len
        self.batch_size = self.loaders[0].batch_size
        for i, l in enumerate(self.loaders):
            if l.seq_len != self.seq_len or l.batch_size != self.batch_size:
                raise ValueError(
                    f"loader {i} shape ({l.seq_len}, {l.batch_size}) != "
                    f"expected ({self.seq_len}, {self.batch_size})")

        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def __len__(self) -> int:
        return sum(len(l) for l in self.loaders)

    def epoch(self, reseed: Optional[int] = None,
              skip_batches: int = 0) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (inputs, targets) balanced round-robin across constituent domain loaders.

        Exact `skip_batches` offsets are propagated into constituent loaders so resume
        is bit-level deterministic without re-reading skipped batches.
        """
        if reseed is not None:
            self.rng = np.random.default_rng(reseed)
            seed_val = reseed
        else:
            seed_val = self.seed

        n_loaders = len(self.loaders)
        # Compute how many batches each constituent loader should skip
        base_skips = skip_batches // n_loaders
        rem_skips = skip_batches % n_loaders

        iters: List[Iterator] = []
        for i, loader in enumerate(self.loaders):
            loader_skip = base_skips + (1 if i < rem_skips else 0)
            loader_seed = seed_val + (i * 1000)
            iters.append(iter(loader.epoch(reseed=loader_seed, skip_batches=loader_skip)))

        # Cycle round-robin across loaders, starting at the remainder offset
        loader_idx = rem_skips
        total_yielded = 0
        max_total = max(0, len(self) - skip_batches)

        while total_yielded < max_total:
            it = iters[loader_idx]
            batch = next(it, None)
            if batch is None:
                # Epoch exhausted for this domain; reopen fresh epoch for continuous replay
                loader = self.loaders[loader_idx]
                iters[loader_idx] = iter(loader.epoch(reseed=seed_val + loader_idx * 1000 + 1))
                batch = next(iters[loader_idx], None)
                if batch is None:
                    break
            yield batch
            total_yielded += 1
            loader_idx = (loader_idx + 1) % n_loaders


def build_replay_loader_factory(
    replay_dir: Union[Path, str],
    *,
    seed: int = 0,
    shuffle: bool = True,
) -> Callable[[int, int], Any]:
    """Construct a loader factory `(seq_len, batch_size) -> ReplayLoader` for replay data.

    Discovers shards across high-quality curated web, math, and technical prose.
    """
    domain_shards = discover_replay_shards(replay_dir)

    def factory(seq_len: int, batch_size: int) -> Any:
        loaders = []
        # Build one PackedLoader per domain (using the first shard of each domain or all shards)
        for domain, shards in sorted(domain_shards.items()):
            for shard_path in shards:
                loaders.append(PackedLoader(shard_path, seq_len=seq_len, batch_size=batch_size,
                                            shuffle=shuffle, seed=seed))
        return ReplayLoader(loaders, seed=seed)

    return factory
