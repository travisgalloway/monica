"""End-to-end OFFLINE integration test for the #193 TS "LSP-clean" pipeline orchestrator.

Drives `run_pipeline` (Stages 2-5: near-dedup -> prettier -> LSP-clean filter -> tokenize +
pack) directly against a tiny in-memory record list, a stub tsc runner, and prettier off --
no network, no SWH/AWS credentials, no local prettier/tsc toolchain. This is what makes
#193's acceptance ("pipeline runs end-to-end on a sample" + "packed corpus verified
uint16" + "manifest records the LSP-clean filter rate") verifiable in CI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

tokenizers = pytest.importorskip("tokenizers")

from scripts.build_ts_clean_corpus import iter_jsonl_records, run_pipeline  # noqa: E402
from src.data.corpus import Record  # noqa: E402
from src.data.dedup import DedupStats  # noqa: E402
from src.data.pack import packing_dtype_for  # noqa: E402
from src.data.shard import open_shard  # noqa: E402
from src.data.tokenizer_train import SPECIAL_TOKENS, save_tokenizer, train_code_bpe  # noqa: E402
from src.data.ts_clean import CleanRateStats  # noqa: E402

SAMPLE_SNIPPETS = [
    "export function add(a: number, b: number): number { return a + b; }\n",
    "export function sub(a: number, b: number): number { return a - b; }\n",
    "export function mul(a: number, b: number): number { return a * b; }\n",
    "export function div(a: number, b: number): number { return a / b; }\n",
    "export const greet = (name: string): string => `hello ${name}`;\n",
]


class _StubTsc:
    """Marks any text containing "gorblak" dirty; everything else clean."""

    def codes(self, source: str):
        return ["TS2339"] if "gorblak" in source else []


def _tiny_code_tokenizer(tmp_path):
    corpus = SAMPLE_SNIPPETS * 50   # repeat so BPE training has merges to learn
    tok = train_code_bpe(corpus, vocab_size=1000, special_tokens=SPECIAL_TOKENS)
    return save_tokenizer(tok, tmp_path / "code-tokenizer")


def _records(texts, license="mit"):
    return [Record(text=t, source="stack-v2", lang="typescript", license=license,
                   meta={"is_code": True}) for t in texts]


def test_run_pipeline_end_to_end_offline(tmp_path):
    tok_path = _tiny_code_tokenizer(tmp_path)
    records = _records(SAMPLE_SNIPPETS + [SAMPLE_SNIPPETS[0]])   # + one exact dup
    out_dir = tmp_path / "out"

    dedup_stats = DedupStats()
    clean_stats = CleanRateStats()
    manifest = run_pipeline(records, out_dir, tokenizer_path=tok_path, seq_len=64,
                            threshold=0.8, prettier_runner=None,
                            tsc_runner=_StubTsc(), dedup_stats=dedup_stats,
                            clean_stats=clean_stats)

    # uint16-verified packed corpus.
    assert manifest["dtype"] == "uint16"
    assert manifest["n_tokens"] > 0
    assert manifest["n_sequences"] >= 0

    # Manifest carries the LSP-clean filter rate.
    assert manifest["clean_rate"] == clean_stats.as_dict()
    assert manifest["clean_rate"]["n_seen"] == manifest["stage_counts"]["n_after_prettier"]
    assert manifest["tsc_clean_applied"] is True
    assert manifest["prettier_applied"] is False

    # Stage counts are internally consistent (near-dup dropped, nothing else here is dirty).
    assert manifest["stage_counts"]["n_source"] == 6
    assert manifest["stage_counts"]["n_after_dedup"] == 6 - dedup_stats.dropped_near

    # manifest.json was written to disk and matches the returned dict.
    on_disk = json.loads((out_dir / "manifest.json").read_text())
    assert on_disk == manifest

    # The packed shard itself is readable and dtype-correct if any sequence was emitted.
    if manifest["shards"]:
        shard_name = manifest["shards"][0]["name"]
        toks, bounds = open_shard(out_dir, shard_name)
        assert toks.dtype == np.uint16
        assert packing_dtype_for(1000) == np.uint16


def test_run_pipeline_dedup_drops_exact_repeat(tmp_path):
    tok_path = _tiny_code_tokenizer(tmp_path)
    records = _records(SAMPLE_SNIPPETS[:2] * 3)   # 3x duplicated pair -> 6 records
    dedup_stats = DedupStats()
    manifest = run_pipeline(records, tmp_path / "out2", tokenizer_path=tok_path, seq_len=32,
                            threshold=0.8, tsc_runner=_StubTsc(), dedup_stats=dedup_stats)
    assert manifest["stage_counts"]["n_source"] == 6
    assert manifest["stage_counts"]["n_after_dedup"] < 6
    assert dedup_stats.dropped_near > 0


def test_run_pipeline_ts_clean_drops_dirty_files(tmp_path):
    tok_path = _tiny_code_tokenizer(tmp_path)
    dirty = "export function bad(): number { return gorblak; }\n"
    records = _records(SAMPLE_SNIPPETS + [dirty])
    clean_stats = CleanRateStats()
    manifest = run_pipeline(records, tmp_path / "out3", tokenizer_path=tok_path, seq_len=64,
                            threshold=0.99, tsc_runner=_StubTsc(), clean_stats=clean_stats)
    assert clean_stats.n_dirty == 1
    assert manifest["stage_counts"]["n_after_clean"] == manifest["stage_counts"]["n_after_prettier"] - 1


def test_run_pipeline_skips_clean_filter_when_tsc_runner_is_none(tmp_path):
    tok_path = _tiny_code_tokenizer(tmp_path)
    records = _records(SAMPLE_SNIPPETS)
    manifest = run_pipeline(records, tmp_path / "out4", tokenizer_path=tok_path, seq_len=64,
                            threshold=0.99, tsc_runner=None)
    assert manifest["tsc_clean_applied"] is False
    assert manifest["clean_rate"] is None
    assert manifest["stage_counts"]["n_after_clean"] == manifest["stage_counts"]["n_after_prettier"]


def test_run_pipeline_applies_prettier_when_runner_given(tmp_path):
    tok_path = _tiny_code_tokenizer(tmp_path)
    records = _records(["const x=1"])

    class _UppercaseRunner:
        """Fake prettier: deterministic, obvious transform to prove Stage 3 ran."""

        def format(self, source: str) -> str:
            return source.upper()

    manifest = run_pipeline(records, tmp_path / "out5", tokenizer_path=tok_path, seq_len=8,
                            threshold=0.99, prettier_runner=_UppercaseRunner(),
                            tsc_runner=_StubTsc())
    assert manifest["prettier_applied"] is True
    assert manifest["n_tokens"] > 0


def test_manifest_written_even_with_zero_sequences(tmp_path):
    # seq_len far larger than the tiny corpus -> zero full sequences packed, but the
    # manifest must still be written and dtype-correct (n_sequences == 0 is valid, not
    # an error -- this is a SAMPLE run, not the full build).
    tok_path = _tiny_code_tokenizer(tmp_path)
    records = _records(["const x=1"])
    manifest = run_pipeline(records, tmp_path / "out6", tokenizer_path=tok_path,
                            seq_len=1_000_000, tsc_runner=_StubTsc())
    assert manifest["n_sequences"] == 0
    assert manifest["dtype"] == "uint16"
    assert (tmp_path / "out6" / "manifest.json").exists()


# --- iter_jsonl_records (the --from-jsonl offline Stage-1 source) ---------------------
def test_iter_jsonl_records_reads_text_and_content_fields(tmp_path):
    p = tmp_path / "sample.jsonl"
    p.write_text(
        '{"text": "const a = 1;", "license": "mit"}\n'
        '{"content": "const b = 2;", "path": "b.ts"}\n'
        '\n'
        '{"text": ""}\n'  # empty text -> skipped
    )
    out = list(iter_jsonl_records(p))
    assert [r.text for r in out] == ["const a = 1;", "const b = 2;"]
    assert out[0].license == "mit"
    assert out[1].meta["path"] == "b.ts"
    assert all(r.lang == "typescript" for r in out)
