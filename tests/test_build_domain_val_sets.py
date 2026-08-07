"""Tests for the per-domain val-set builder (#221, `scripts/build_domain_val_sets.py`).

This is the "tagging required" step: packed shards carry no domain/language field, so these
val sets are built from a stage that still has one. The load-bearing guarantees are that
`n_bytes` is ALWAYS written (`domain_bpb` refuses a val set without it), that the held-out
selection is a pure function of `(seed, domain)`, and that a dropped or untagged record is
counted rather than silently absorbed.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/build_domain_val_sets.py"


def _module():
    spec = importlib.util.spec_from_file_location("build_domain_val_sets", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO_ROOT))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    return mod


def _corpus(tmp_path, rows):
    path = tmp_path / "cleaned"
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "part-000.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def _rows(n_per_lang=12):
    out = []
    for lang, filler in (("typescript", "const x{i} = {i};\n"), ("python", "x{i} = {i}\n")):
        for i in range(n_per_lang):
            out.append({"text": filler.format(i=i) * 40, "source": "synthetic", "lang": lang,
                        "meta": {"lang": lang}})
    return out


def _run(inp, out, *extra):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--in", str(inp), "--out", str(out),
         "--tokenizer", "byte", "--seed", "0", *extra],
        cwd=REPO_ROOT, capture_output=True, text=True)


def test_domain_of_reads_top_level_and_meta_fields():
    mod = _module()
    rec = {"text": "t", "source": "stack", "lang": "en", "meta": {"lang": "typescript"}}
    assert mod.domain_of(rec, "source") == "stack"
    assert mod.domain_of(rec, "lang") == "en"
    assert mod.domain_of(rec, "meta:lang") == "typescript"


def test_an_untagged_record_is_dropped_not_bucketed_into_a_catch_all():
    """An 'other' bucket silently mixes languages, which is exactly what a per-domain BPB
    report exists to separate."""
    mod = _module()
    assert mod.domain_of({"text": "t", "lang": ""}, "lang") is None
    assert mod.domain_of({"text": "t"}, "lang") is None
    assert mod.domain_of({"text": "t", "meta": {}}, "meta:lang") is None


def test_domain_dir_names_are_filesystem_safe():
    mod = _module()
    assert mod._safe_name("c++/17") == "c_17"
    assert mod._safe_name("///") == "unknown"


def test_builds_one_packed_val_set_per_domain_with_n_bytes(tmp_path):
    inp = _corpus(tmp_path, _rows())
    out = tmp_path / "domains"
    res = _run(inp, out, "--group-by", "lang", "--val-docs", "5")
    assert res.returncode == 0, res.stderr

    index = json.loads((out / "domains.json").read_text())
    assert set(index["domains"]) == {"typescript", "python"}
    assert index["n_records_read"] == 24 and index["n_records_untagged"] == 0
    for name, entry in index["domains"].items():
        assert entry["n_docs"] == 5 and entry["n_tokens"] > 0 and entry["n_bytes"] > 0
        meta = json.loads((out / entry["packed"]).with_suffix(".meta.json").read_text())
        # `domain_bpb` REFUSES a val set with no n_bytes, so this is not optional.
        assert meta["n_bytes"] == entry["n_bytes"]


def test_the_held_out_selection_is_reproducible(tmp_path):
    inp = _corpus(tmp_path, _rows())
    a, b = tmp_path / "a", tmp_path / "b"
    assert _run(inp, a, "--group-by", "lang", "--val-docs", "5").returncode == 0
    assert _run(inp, b, "--group-by", "lang", "--val-docs", "5").returncode == 0
    for domain in ("typescript", "python"):
        assert (a / domain / "val.bin").read_bytes() == (b / domain / "val.bin").read_bytes()


def test_adding_a_domain_does_not_perturb_the_others(tmp_path):
    """Each domain's shuffle is derived from a stable hash of (seed, name), not from one
    shared stream, so a rebuild stays comparable to the previous one."""
    base = _corpus(tmp_path / "base", _rows())
    extended = _corpus(tmp_path / "ext", _rows() + [
        {"text": "package main\n" * 60, "source": "synthetic", "lang": "go"}
        for _ in range(12)])
    a, b = tmp_path / "a", tmp_path / "b"
    assert _run(base, a, "--group-by", "lang", "--val-docs", "5").returncode == 0
    assert _run(extended, b, "--group-by", "lang", "--val-docs", "5").returncode == 0
    assert (b / "go" / "val.bin").exists()
    assert (a / "typescript" / "val.bin").read_bytes() == (
        b / "typescript" / "val.bin").read_bytes()


def test_min_docs_records_what_it_dropped(tmp_path):
    rows = _rows(n_per_lang=12) + [{"text": "fn main() {}\n" * 40, "source": "synthetic",
                                    "lang": "rust"}]
    inp = _corpus(tmp_path, rows)
    out = tmp_path / "domains"
    assert _run(inp, out, "--group-by", "lang", "--val-docs", "5",
                "--min-docs", "4").returncode == 0
    index = json.loads((out / "domains.json").read_text())
    assert "rust" not in index["domains"]
    assert index["dropped_domains"] == {"rust": 1}      # recorded, never silent


def test_val_bytes_budget_is_honoured(tmp_path):
    inp = _corpus(tmp_path, _rows())
    out = tmp_path / "domains"
    assert _run(inp, out, "--group-by", "lang", "--val-bytes", "1000").returncode == 0
    index = json.loads((out / "domains.json").read_text())
    for entry in index["domains"].values():
        # Stops at the first doc that reaches the budget, so it may overshoot by one doc.
        assert entry["n_bytes"] >= 1000


def test_a_corpus_with_no_such_field_fails_loudly(tmp_path):
    inp = _corpus(tmp_path, [{"text": "x" * 100, "source": "s", "lang": "en"}])
    res = _run(inp, tmp_path / "out", "--group-by", "meta:lang", "--val-docs", "5")
    assert res.returncode != 0
    assert "nothing to group" in (res.stderr + res.stdout)


def test_an_unbounded_val_set_is_rejected(tmp_path):
    inp = _corpus(tmp_path, _rows())
    res = _run(inp, tmp_path / "out", "--group-by", "lang")
    assert res.returncode != 0
    assert "--val-docs" in (res.stderr + res.stdout)


def test_swift_tokenizer_requires_a_tokenizer_json(tmp_path):
    inp = _corpus(tmp_path, _rows())
    res = subprocess.run(
        [sys.executable, str(SCRIPT), "--in", str(inp), "--out", str(tmp_path / "o"),
         "--tokenizer", "swift", "--val-docs", "5"],
        cwd=REPO_ROOT, capture_output=True, text=True)
    assert res.returncode != 0
    assert "--tokenizer-json" in (res.stderr + res.stdout)


def test_the_output_feeds_domain_bpb_end_to_end(tmp_path):
    from src.eval.code_suite import StubCausalModel
    from src.eval.domain_bpb import evaluate_domain_bpb, load_domain_index

    inp = _corpus(tmp_path, _rows(n_per_lang=40))
    out = tmp_path / "domains"
    assert _run(inp, out, "--group-by", "lang", "--val-docs", "30").returncode == 0

    index = load_domain_index(out / "domains.json")
    res = evaluate_domain_bpb(StubCausalModel(vocab_size=256, seed=0),
                              {n: e["packed"] for n, e in index.items()},
                              batch_size=2, seq_len=64)
    assert set(res["by_domain"]) == {"python", "typescript"}
    for agg in res["by_domain"].values():
        assert agg is not None and agg["val_bpb"] > 0
