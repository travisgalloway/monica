"""The #298 double-export comparison detects corruption — and refuses to be BLIND.

`src/conformance/fixture_digest.py` is the half of the #298 guard that decides whether two
fixture exports agree. A guard that cannot fail is not a guard, so the load-bearing cases
here are the negative ones: a single flipped byte, a dropped file, and — the one that
matters most — two empty trees, which must RAISE rather than report agreement.

Portable: hashlib/pathlib only, no MLX, so this runs on the Linux `portable` CI job where
no fixture can be produced at all.
"""

from __future__ import annotations

import pytest

from src.conformance.fixture_digest import compare_trees, digest_tree


def _tree(root, files):
    root.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return root


# A stand-in for a fixture tree: a large-ish binary blob (crosses the 1 MiB streaming
# chunk, so the chunked `_sha256`/`_first_difference` loops are actually exercised rather
# than short-circuited on a single read) plus the small sidecars a real fixture carries.
_BIG = bytes(range(256)) * 8192          # 2 MiB
_FILES = {
    "reference.safetensors": _BIG,
    "weights.safetensors": b"\x01\x02\x03\x04" * 64,
    "meta.json": b'{"forward_step_max_abs_diff": 1.43e-06}\n',
}


def test_identical_trees_agree(tmp_path):
    a = _tree(tmp_path / "a", _FILES)
    b = _tree(tmp_path / "b", _FILES)
    assert compare_trees(a, b) == []
    assert digest_tree(a) == digest_tree(b)
    assert set(digest_tree(a)) == set(_FILES)


def test_flipped_byte_is_reported(tmp_path):
    """The negative control the whole guard rests on: one corrupted byte in the middle of
    a multi-megabyte oracle — the shape #298's silent corruption would take."""
    a = _tree(tmp_path / "a", _FILES)
    corrupt = dict(_FILES)
    offset = 1_000_000
    blob = bytearray(_BIG)
    blob[offset] ^= 0x01
    corrupt["reference.safetensors"] = bytes(blob)
    b = _tree(tmp_path / "b", corrupt)

    verdicts = compare_trees(a, b)
    assert len(verdicts) == 1, verdicts
    assert "reference.safetensors" in verdicts[0]
    assert f"offset {offset}" in verdicts[0], verdicts[0]
    # Same-size corruption must NOT be reported as a size mismatch.
    assert "size mismatch" not in verdicts[0]


def test_missing_file_is_reported_from_both_sides(tmp_path):
    """A DROPPED oracle key is the same class of failure as a drifted one (the same
    reasoning `tests/test_parity_fixture_export.py` applies to safetensors keys), and it
    has to be caught whichever side lost it."""
    a = _tree(tmp_path / "a", _FILES)
    fewer = {k: v for k, v in _FILES.items() if k != "meta.json"}
    b = _tree(tmp_path / "b", fewer)

    forward = compare_trees(a, b)
    assert len(forward) == 1 and "meta.json" in forward[0]
    assert "missing from" in forward[0]

    reverse = compare_trees(b, a)
    assert len(reverse) == 1 and "meta.json" in reverse[0]
    assert "missing from" in reverse[0]


def test_size_mismatch_is_reported_as_such(tmp_path):
    a = _tree(tmp_path / "a", _FILES)
    truncated = dict(_FILES, **{"reference.safetensors": _BIG[:-4096]})
    b = _tree(tmp_path / "b", truncated)

    verdicts = compare_trees(a, b)
    assert len(verdicts) == 1
    assert "size mismatch" in verdicts[0], verdicts[0]


def test_nested_files_are_compared(tmp_path):
    """Fixture trees are not flat forever; a corruption one directory down must not be
    invisible to the guard."""
    a = _tree(tmp_path / "a", dict(_FILES, **{"sub/extra.bin": b"abc"}))
    b = _tree(tmp_path / "b", dict(_FILES, **{"sub/extra.bin": b"abd"}))
    verdicts = compare_trees(a, b)
    assert len(verdicts) == 1 and verdicts[0].startswith("sub/extra.bin")


def test_empty_trees_are_blind_not_clean(tmp_path):
    """THE anti-blind case. Two empty directories must never compare 'identical' — a
    check that cannot observe its target reports that it could not, and an export that
    wrote nothing at all is exactly the situation where a false 'clean' would let a
    missing oracle through."""
    a = (tmp_path / "a")
    b = (tmp_path / "b")
    a.mkdir()
    b.mkdir()
    with pytest.raises(ValueError, match="no files"):
        digest_tree(a)
    with pytest.raises(ValueError, match="no files"):
        compare_trees(a, b)


def test_missing_tree_raises(tmp_path):
    a = _tree(tmp_path / "a", _FILES)
    with pytest.raises(FileNotFoundError):
        digest_tree(tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        compare_trees(a, tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        compare_trees(tmp_path / "nope", a)


def test_a_file_where_a_tree_was_expected_raises(tmp_path):
    """`--out` pointing at a regular file is unreadable-as-a-tree, i.e. BLIND."""
    f = tmp_path / "not-a-dir"
    f.write_bytes(b"x")
    with pytest.raises(FileNotFoundError):
        digest_tree(f)


def test_empty_files_still_count_as_files(tmp_path):
    """A zero-byte file is *present*; only a tree with no files at all is BLIND."""
    a = _tree(tmp_path / "a", {"empty.bin": b""})
    b = _tree(tmp_path / "b", {"empty.bin": b""})
    assert compare_trees(a, b) == []
    assert list(digest_tree(a)) == ["empty.bin"]
