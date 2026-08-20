"""Tests for the decontamination blocklist builder (#221, `scripts/build_decontam_blocklist.py`).

The acceptance claim being pinned here is "decontam blocklist wired": the artifact must be
byte-reproducible, must be consumable **unchanged** by `Decontaminator.from_texts` (which is
what `src/data/corpus.py --decontam-file` calls), and its manifest's sha256 must match the
file actually written.
"""

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/build_decontam_blocklist.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_decontam_blocklist", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO_ROOT))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    return mod


def _run(out: Path, *extra: str):
    return subprocess.run([sys.executable, str(SCRIPT), "--out", str(out), *extra],
                          cwd=REPO_ROOT, capture_output=True, text=True)


def test_flatten_collapses_newlines_into_one_line():
    """`Decontaminator` n-grams on `text.lower().split()`, so whitespace SHAPE is
    irrelevant — but line integrity is not: a record split across lines loses every n-gram
    that spanned the break."""
    mod = _load_module()
    assert mod.flatten("a\nb\t c\r\n") == "a b c"
    assert mod.flatten(None) == "" and mod.flatten("") == ""


def test_source_list_covers_the_builtin_and_external_sets():
    mod = _load_module()
    names = mod.all_source_names()
    assert "ts_error_injection" in names and "humaneval_ts" in names
    assert "code_recall" in names and "code_needle" in names
    assert "external:safim" in names


def test_blocklist_is_byte_reproducible(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    ra, rb = _run(a), _run(b)
    assert ra.returncode == 0, ra.stderr
    assert rb.returncode == 0, rb.stderr
    assert a.read_bytes() == b.read_bytes()
    assert a.stat().st_size > 0


def test_lines_are_sorted_deduplicated_and_free_of_embedded_newlines(tmp_path):
    out = tmp_path / "bl.txt"
    assert _run(out).returncode == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines == sorted(lines)
    assert len(lines) == len(set(lines))
    assert all(line.strip() == line and "\n" not in line for line in lines)


def test_manifest_sha256_matches_the_file_it_describes(tmp_path):
    out = tmp_path / "bl.txt"
    assert _run(out).returncode == 0
    from src.eval.code_suite import sha256_file

    manifest = json.loads((tmp_path / "bl.manifest.json").read_text())
    assert manifest["sha256"] == sha256_file(out)
    assert manifest["n_lines"] == len(out.read_text().splitlines())
    assert manifest["allow_network"] is False
    # Each external source records its pin status, so the exact revision a build drew on —
    # or the fact that it used the offline fixture instead — is visible in the artifact
    # itself rather than only in the source table (#304 filled every pin).
    assert manifest["sources"]["external:safim"]["fixture_only"] is True
    assert re.fullmatch(r"[0-9a-f]{40}", manifest["sources"]["external:safim"]["revision"])


def test_min_words_drops_texts_too_short_to_ever_match(tmp_path):
    """`Decontaminator`'s largest n-gram is 13 words; a shorter text cannot produce one, so
    keeping it would only inflate the file."""
    default_out, strict_out = tmp_path / "d.txt", tmp_path / "s.txt"
    assert _run(default_out).returncode == 0
    assert _run(strict_out, "--min-words", "400").returncode != 0    # empties it -> loud failure

    loose = tmp_path / "l.txt"
    assert _run(loose, "--min-words", "5").returncode == 0
    assert len(loose.read_text().splitlines()) > len(default_out.read_text().splitlines())


def test_the_artifact_is_consumable_by_the_decontaminator_unchanged(tmp_path):
    out = tmp_path / "bl.txt"
    assert _run(out).returncode == 0

    from src.data.corpus import Record
    from src.data.dedup import Decontaminator, decontaminate

    with open(out, encoding="utf-8") as f:
        decon = Decontaminator.from_texts([line.rstrip("\n") for line in f])

    lines = out.read_text(encoding="utf-8").splitlines()
    contaminated = Record(text=lines[0], source="t")
    clean = Record(text="a completely unrelated sentence about weather and tides " * 3,
                   source="t")
    kept = list(decontaminate([contaminated, clean], decon))
    assert [r.text for r in kept] == [clean.text]


def test_unknown_source_is_rejected(tmp_path):
    res = _run(tmp_path / "x.txt", "--sets", "not-a-set")
    assert res.returncode != 0
    assert "unknown source" in (res.stderr + res.stdout)


def test_list_sets_exits_cleanly(tmp_path):
    res = subprocess.run([sys.executable, str(SCRIPT), "--list-sets"], cwd=REPO_ROOT,
                         capture_output=True, text=True)
    assert res.returncode == 0
    assert "external:mceval" in res.stdout
