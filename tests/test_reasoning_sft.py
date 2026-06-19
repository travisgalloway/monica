"""Reasoning-trace SFT driver (#96): masked JSONL + atomic packed .bounds artifact.
Offline via the handauthored trace set + ByteTokenizer (no backend, no network)."""

import json

from src.data.chat_template import IM_END
from src.data.distill_corpus import tokenized_subdir
from src.data.reasoning_sft import build_reasoning_sft
from src.data.reasoning_traces import handauthored_trace_records, trace_to_messages
from src.data.shard import doc_start_offsets, open_shard, read_manifest
from src.data.sft_loader import SFTLoader


def test_build_reasoning_sft_end_to_end(tmp_path):
    m = build_reasoning_sft(handauthored_trace_records(), tmp_path, tokenizer="qwen25",
                            byte_fallback=True, seq_len=1024, chunk_align=64)

    cleaned = tmp_path / "sft" / "cleaned" / "reasoning-traces" / "records.jsonl"
    tok_dir = tmp_path / "sft" / "tokenized" / tokenized_subdir("qwen25", 1024)
    packed_dir = tok_dir / "reasoning-packed"
    assert cleaned.exists()
    assert (tok_dir / "reasoning.jsonl").exists()
    assert (tok_dir / "reasoning-manifest.json").exists()
    assert (packed_dir / "manifest.json").exists()

    # Manifest summary: think-answer format, chat EOS, atomic-packing invariant.
    assert m["format"] == "think-answer" and m["chat_eos"] == IM_END
    assert m["n_masked_records"] == len(list(handauthored_trace_records()))
    assert m["packed_n_documents"] == m["n_masked_records"]   # every trace packed atomically

    # Atomic packing: each trace starts on a chunk boundary; #docs == kept traces; none split.
    pack_manifest = read_manifest(packed_dir)
    assert pack_manifest["n_documents"] == m["n_masked_records"]
    name = pack_manifest["shards"][0]["name"]
    _, bnds = open_shard(packed_dir, name)
    starts = doc_start_offsets(bnds)
    assert all(off % m["chunk_align"] == 0 for off in starts)  # chunk-aligned -> no mid-seq trace
    assert len(starts) == m["packed_n_documents"]


def test_masked_records_train_only_the_trace(tmp_path):
    tok_dir = tmp_path / "sft" / "tokenized" / tokenized_subdir("qwen25", 1024)
    build_reasoning_sft(handauthored_trace_records(), tmp_path, tokenizer="qwen25",
                        byte_fallback=True, seq_len=1024, chunk_align=64)
    recs = [json.loads(line) for line in (tok_dir / "reasoning.jsonl").read_text().splitlines()]
    assert recs
    from src.data.tokenize import ByteTokenizer
    tok = ByteTokenizer()
    rec = recs[0]
    trained = [rec["target_ids"][j] for j in range(len(rec["loss_mask"])) if rec["loss_mask"][j]]
    decoded = tok.decode(trained)
    # The trained span is the assistant trace ending on <|im_end|>; the user question is excluded.
    assert decoded.endswith(IM_END) and "<think>" in decoded and "<answer>" in decoded


def test_over_length_trace_dropped_from_both_artifacts(tmp_path):
    # One short trace + one trace longer than seq_len: the long one is dropped, never split.
    rows = [trace_to_messages("hi", "short", "ok"),
            trace_to_messages("big", "y " * 2000, "done")]
    m = build_reasoning_sft(rows, tmp_path, tokenizer="qwen25", byte_fallback=True,
                            seq_len=256, chunk_align=64)
    assert m["n_traces"] == 2
    assert m["n_masked_records"] == 1 and m["n_overlength_dropped"] == 1
    assert m["packed_n_documents"] == 1               # dropped trace absent from the packing too


def test_records_load_through_sft_loader(tmp_path):
    tok_dir = tmp_path / "sft" / "tokenized" / tokenized_subdir("qwen25", 1024)
    build_reasoning_sft(handauthored_trace_records(), tmp_path, tokenizer="qwen25",
                        byte_fallback=True, seq_len=1024, chunk_align=64)
    loader = SFTLoader(tok_dir / "reasoning.jsonl", seq_len=1024, batch_size=2)
    inputs, targets, mask = next(loader.epoch())
    assert inputs.shape == targets.shape == mask.shape and mask.sum() > 0
