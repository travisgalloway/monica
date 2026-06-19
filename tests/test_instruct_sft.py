"""Shared instruct SFT corpus driver (#95): builds the response-masked Qwen-ChatML records under
the `shared/sft/` prefix. Offline via the checked-in handauthored set + ByteTokenizer (no backend,
no network).
"""

import json

from src.data.chat_template import IM_END
from src.data.instruct_sft import build_chat_sft_records, build_instruct_sft
from src.data.sft_sources import handauthored_records
from src.data.sft_loader import SFTLoader
from src.data.tokenize import ByteTokenizer


def test_build_chat_sft_records_masks_only_assistant():
    tok = ByteTokenizer()
    rows = [{"messages": [{"role": "user", "content": "ping"},
                          {"role": "assistant", "content": "pong"}]}]
    recs = list(build_chat_sft_records(rows, tok))
    assert len(recs) == 1
    rec = recs[0]
    assert len(rec["input_ids"]) == len(rec["target_ids"]) == len(rec["loss_mask"])
    # The trained targets decode to exactly the assistant content + <|im_end|> (no extra EOS).
    trained = [rec["target_ids"][j] for j in range(len(rec["loss_mask"])) if rec["loss_mask"][j]]
    # Includes the trailing <|im_end|> (the chat EOS) so the model learns to stop, and no extra EOS.
    assert tok.decode(trained) == f"pong{IM_END}"


def test_over_length_examples_dropped():
    tok = ByteTokenizer()
    rows = [{"messages": [{"role": "user", "content": "x"},
                          {"role": "assistant", "content": "y" * 500}]}]
    assert list(build_chat_sft_records(rows, tok, max_seq_len=50)) == []


def test_build_instruct_sft_end_to_end(tmp_path):
    manifest = build_instruct_sft(handauthored_records(), tmp_path,
                                  tokenizer="qwen25", byte_fallback=True, seq_len=8192)

    # Two-artifact layout under shared/sft/.
    cleaned = tmp_path / "sft" / "cleaned" / "instruct" / "records.jsonl"
    tok_dir = tmp_path / "sft" / "tokenized" / "qwen25-8k"
    assert cleaned.exists()
    assert (tok_dir / "instruct.jsonl").exists() and (tok_dir / "manifest.json").exists()

    # Cleaned rows are tokenizer-agnostic {messages, source, license}.
    first = json.loads(cleaned.read_text().splitlines()[0])
    assert "messages" in first and "source" in first and "license" in first

    # Manifest records the documented fields incl. the chat-EOS convention.
    assert manifest["template"] == "qwen-chatml"
    assert manifest["chat_eos"] == IM_END
    assert manifest["tokenizer"] == "qwen25"
    assert manifest["n_records"] == len(list(handauthored_records()))
    assert manifest["n_tokens"] > 0
    assert manifest["sources"].get("handauthored") == manifest["n_records"]

    # Tokenized records drop straight into the M9 SFTLoader (same shape).
    loader = SFTLoader(tok_dir / "instruct.jsonl", seq_len=8192, batch_size=2)
    inputs, targets, mask = next(loader.epoch())
    assert inputs.shape == targets.shape == mask.shape
    assert mask.sum() > 0
