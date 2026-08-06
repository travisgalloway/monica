"""Differential / determinism test for FIM insertion in the Swift `pack` path (#215).

`monica-selfcheck` is the Swift package's own test runner and covers the round-trip reassembly
in-process. This file covers what only a shell-out can: that the **CLI flags** are wired to the
transform, that two packs with the same `--fim-seed` are byte-identical while a different seed
is not (anti-vacuity — a no-op transform would pass the identity check trivially), that
`--fim-rate 0` is byte-identical to omitting the flag entirely (the no-regression proof for the
existing pipeline and for `tests/test_swift_parquet.py`), and that the **Python trainer** sees
the sentinels where the PSM frame says they should be.

Note what this file cannot prove: a single machine cannot catch a platform-divergent RNG. That
is `swift-parity`'s job in `.github/workflows/ci.yml`, which `cmp`s macOS-packed shards against
Linux-packed ones.

Skips cleanly if `monica-tokenize` isn't built (`cd swift && swift build`).
"""
import json
import os
import subprocess
from pathlib import Path

import pytest

from src.data.shard import doc_start_offsets, open_shard, read_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "swift" / "Fixtures" / "parity-corpus.jsonl"

PREFIX_ID, MIDDLE_ID, SUFFIX_ID = 1, 2, 3


def _find_monica_tokenize() -> Path | None:
    env = os.environ.get("MONICA_TOKENIZE")
    if env and Path(env).exists():
        return Path(env)
    swift_dir = REPO_ROOT / "swift"
    for mode in ("release", "debug"):
        cand = swift_dir / ".build" / mode / "monica-tokenize"
        if cand.exists():
            return cand
    return None


@pytest.fixture(scope="module")
def monica_tokenize() -> Path:
    binary = _find_monica_tokenize()
    if binary is None:
        pytest.skip("monica-tokenize not built (cd swift && swift build)")
    return binary


@pytest.fixture(scope="module")
def tokenizer(monica_tokenize, tmp_path_factory) -> Path:
    """One tokenizer for the whole module — training it per test would dominate the runtime and
    prove nothing extra (training determinism is already `monica-selfcheck`'s job)."""
    if not FIXTURE.exists():
        pytest.skip(f"fixture corpus missing: {FIXTURE}")
    out = tmp_path_factory.mktemp("tok") / "tok.json"
    r = _run(monica_tokenize, "train", "--in", str(FIXTURE), "--out", str(out),
             "--vocab-size", "2000")
    assert r.returncode == 0, r.stderr
    return out


def _run(binary: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(binary), *args], capture_output=True, text=True)


def _pack(binary: Path, tok: Path, out: Path, *extra: str) -> subprocess.CompletedProcess:
    r = _run(binary, "pack", "--tokenizer", str(tok), "--in", str(FIXTURE), "--out", str(out),
             "--seq-len", "64", "--shard-size-mb", "1", *extra)
    assert r.returncode == 0, r.stderr
    # An empty pack would make every byte-comparison below vacuous — a check that cannot observe
    # its target reports BLIND, not healthy.
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["n_sequences"] > 0, f"pack produced no sequences: {manifest}"
    return r


def _dir_bytes(d: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(d.iterdir())}


def test_same_seed_packs_are_byte_identical(monica_tokenize, tokenizer, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for out in (a, b):
        _pack(monica_tokenize, tokenizer, out, "--fim-rate", "0.5", "--fim-seed", "1234")
    assert _dir_bytes(a) == _dir_bytes(b)


def test_a_different_seed_changes_the_tokens(monica_tokenize, tokenizer, tmp_path):
    """Anti-vacuity: without this, a transform that silently did nothing would pass the
    same-seed identity test."""
    a, c = tmp_path / "a", tmp_path / "c"
    _pack(monica_tokenize, tokenizer, a, "--fim-rate", "0.5", "--fim-seed", "1234")
    _pack(monica_tokenize, tokenizer, c, "--fim-rate", "0.5", "--fim-seed", "4321")
    assert (a / "part-00000.bin").read_bytes() != (c / "part-00000.bin").read_bytes()


def test_rate_zero_is_identical_to_omitting_the_flag(monica_tokenize, tokenizer, tmp_path):
    """The no-regression proof: the pre-#215 pipeline (and `tests/test_swift_parquet.py`'s
    byte-identity assertion) must not move because FIM exists."""
    off1, off2 = tmp_path / "off1", tmp_path / "off2"
    _pack(monica_tokenize, tokenizer, off1)
    _pack(monica_tokenize, tokenizer, off2, "--fim-rate", "0")
    assert _dir_bytes(off1) == _dir_bytes(off2)


def test_fim_changes_the_output_relative_to_no_fim(monica_tokenize, tokenizer, tmp_path):
    off, on = tmp_path / "off", tmp_path / "on"
    _pack(monica_tokenize, tokenizer, off)
    _pack(monica_tokenize, tokenizer, on, "--fim-rate", "0.5", "--fim-seed", "1234")
    assert (off / "part-00000.bin").read_bytes() != (on / "part-00000.bin").read_bytes()


def test_stats_line_reports_the_transform(monica_tokenize, tokenizer, tmp_path):
    r = _pack(monica_tokenize, tokenizer, tmp_path / "out",
              "--fim-rate", "0.5", "--fim-seed", "1234")
    line = next(ln for ln in r.stdout.splitlines() if ln.startswith("fim:"))
    assert "seed 1234" in line
    transformed = int(line.split()[1].split("/")[0])
    assert transformed > 0, f"no document was FIM-transformed: {line}"


def test_rate_outside_the_intended_band_warns_but_proceeds(monica_tokenize, tokenizer, tmp_path):
    r = _pack(monica_tokenize, tokenizer, tmp_path / "out", "--fim-rate", "0.9")
    assert "0.4" in r.stderr and "#215" in r.stderr


@pytest.mark.parametrize("bad", ["1.5", "-0.1", "abc"])
def test_invalid_rate_fails_fast(monica_tokenize, tokenizer, tmp_path, bad):
    r = _run(monica_tokenize, "pack", "--tokenizer", str(tokenizer), "--in", str(FIXTURE),
             "--out", str(tmp_path / f"out-{bad}"), "--seq-len", "64", "--fim-rate", bad)
    assert r.returncode != 0
    assert "fim-rate" in r.stderr


def test_python_reads_the_psm_frame_the_swift_packer_wrote(monica_tokenize, tokenizer, tmp_path):
    """End-to-end: the trainer's own shard reader must see `<|fim_prefix|>` at a document start
    and `<|fim_suffix|>` before `<|fim_middle|>` within that document. This is the assertion that
    a Swift-inserted FIM stream is what the Python side actually consumes."""
    out = tmp_path / "fim"
    _pack(monica_tokenize, tokenizer, out, "--fim-rate", "0.5", "--fim-seed", "1234")

    manifest = read_manifest(out)
    fim_docs = 0
    for shard in manifest["shards"]:
        toks, bnds = open_shard(out, shard["name"])
        starts = doc_start_offsets(bnds)
        for i, start in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else len(toks)
            doc = [int(t) for t in toks[start:end]]
            if not doc or doc[0] != PREFIX_ID:
                continue
            fim_docs += 1
            assert doc.count(PREFIX_ID) == 1
            # The doc may be truncated at the shard edge, so a sentinel can legitimately be
            # missing — but whenever both are present, suffix must precede middle (PSM order).
            if SUFFIX_ID in doc and MIDDLE_ID in doc:
                assert doc.index(SUFFIX_ID) < doc.index(MIDDLE_ID)
    assert fim_docs > 0, "no FIM-framed document reached the Python shard reader"
