"""Byte-for-byte comparison of two fixture trees (#298).

`swift/engine/Fixtures/` holds frozen Python-MLX oracles that the Swift port is gated
against. #298 is the failure mode that makes those oracles untrustworthy: on this repo's
MLX 0.32.0 Apple-Silicon build, repeated `MLXMambaModel` construction inside one process
can silently corrupt an unrelated later computation — correct shapes, no exception, no
log. An export that corrupts that way writes a *plausible* fixture, and every downstream
gate then happily agrees with the wrong numbers.

The guard is not a cure (the defect is upstream in MLX and this repo cannot fix it): it is
a **two-fresh-processes** cross-check. `scripts/export_parity_fixture.py` exports twice,
in two independent interpreters, and refuses to leave anything at `--out` unless the two
trees are identical byte-for-byte. #298 is deterministic *per process*, so two processes
would have to corrupt identically to slip past.

This module is the comparison half, and it is deliberately **portable** — hashlib +
pathlib only, no numpy, no mlx, no torch (see `tests/test_import_guard.py`'s
`PORTABLE_MODULES`). That is what lets the guard's own logic be unit-tested on the Linux
`portable` CI job, where no MLX exists to produce a fixture in the first place.

## The anti-blind rule

`digest_tree` **raises** on a missing or empty tree rather than returning `{}`. Two empty
trees comparing "identical" is precisely the BLIND failure this guard exists to prevent:
a check that cannot observe its target must report that it could not, never agreement.
`compare_trees` inherits the behaviour, since it calls `digest_tree` on both sides.

## Scope

Intra-machine determinism only: two processes on ONE host. Cross-machine byte-identity is
**not** claimed and is known to be false — CI's `train-fixture-oracle`/`-verify` pair
measures ~1e-8 drift in `train.safetensors` between runner instances
(`.github/workflows/ci.yml`). Use this to compare a tree against a second export of
itself, not against a tree produced elsewhere.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List

# Fixtures reach 571 MB at poc scale (`swift/engine/Fixtures/README.md` §poc), so every
# read here is streamed — never `Path.read_bytes()` on a fixture file.
_CHUNK = 1 << 20


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(_CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def digest_tree(root: Path | str) -> Dict[str, str]:
    """Map every file under `root` to its sha256, keyed by POSIX-style relative path.

    Raises `FileNotFoundError` if `root` is not a directory and `ValueError` if it holds
    no files. Both are the anti-blind rule: an unreadable or empty tree is *unknown*, and
    returning `{}` would make it compare clean against another unknown.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"cannot digest {root}: not a directory. An unreadable tree is BLIND, not "
            "clean — refusing to report agreement.")
    out: Dict[str, str] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        out[path.relative_to(root).as_posix()] = _sha256(path)
    if not out:
        raise ValueError(
            f"cannot digest {root}: it contains no files. Two empty trees comparing "
            "'identical' is the BLIND failure this check exists to prevent.")
    return out


def _first_difference(a: Path, b: Path) -> int:
    """Byte offset of the first difference between two files, or the length of the
    shorter one when it is a prefix of the longer. Streamed, like `_sha256`."""
    offset = 0
    with a.open("rb") as fa, b.open("rb") as fb:
        while True:
            ba, bb = fa.read(_CHUNK), fb.read(_CHUNK)
            if not ba and not bb:
                return offset
            if ba != bb:
                for i in range(min(len(ba), len(bb))):
                    if ba[i] != bb[i]:
                        return offset + i
                return offset + min(len(ba), len(bb))
            offset += len(ba)


def compare_trees(a: Path | str, b: Path | str) -> List[str]:
    """Compare two fixture trees byte-for-byte.

    Returns `[]` when they are identical, otherwise one human-readable verdict per
    offending file: present on only one side, a size mismatch, or the byte offset at
    which the contents first diverge. Never raises for a *disagreement* — a disagreement
    is the answer. It does raise (via `digest_tree`) when either side is missing or
    empty, because that is not an answer.
    """
    a, b = Path(a), Path(b)
    da, db = digest_tree(a), digest_tree(b)

    verdicts: List[str] = []
    for rel in sorted(set(da) | set(db)):
        if rel not in db:
            verdicts.append(f"{rel}: present in {a}, missing from {b}")
            continue
        if rel not in da:
            verdicts.append(f"{rel}: missing from {a}, present in {b}")
            continue
        if da[rel] == db[rel]:
            continue
        pa, pb = a / rel, b / rel
        sa, sb = pa.stat().st_size, pb.stat().st_size
        if sa != sb:
            verdicts.append(
                f"{rel}: size mismatch ({sa} vs {sb} bytes), sha256 "
                f"{da[rel][:12]}… vs {db[rel][:12]}…")
        else:
            verdicts.append(
                f"{rel}: {sa} bytes, first differing byte at offset "
                f"{_first_difference(pa, pb)}, sha256 {da[rel][:12]}… vs {db[rel][:12]}…")
    return verdicts
