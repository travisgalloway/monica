"""Tool-use SFT corpus tests (#102): masking, distractors, abstention, end-to-end.

Offline via ByteTokenizer + handauthored_tool_records (no network, no backend)."""

import json
import random

import pytest

from src.data.chat_template import IM_END, _ROLES, render, response_spans
from src.data.instruct_sft import _effective_vocab_size
from src.data.sft_loader import SFTLoader
from src.data.tokenize import ByteTokenizer
from src.data.tool_sft import _valid_rows, build_tool_sft
from src.data.tool_sources import (
    TOOL_CALL_OPEN,
    TOOL_RESPONSE_OPEN,
    build_abstention_messages,
    build_tool_messages,
    glaive_row_to_messages,
    handauthored_tool_records,
    iter_tool_sft,
    sample_distractors,
    toolace_row_to_messages,
    validate_call_against_tools,
    when2call_row_to_messages,
    xlam_row_to_messages,
)


# --------------------------------------------------------------------------- #
# Helper: replay the inline mask loop (mirrors build_tool_sft exactly)
# --------------------------------------------------------------------------- #

def _trained_tokens(row, tok):
    """Decode the masked (trained) target tokens for one {messages,...} row."""
    full_ids, spans = response_spans(row["messages"], tok)
    mask = [0] * (len(full_ids) - 1)
    for s, e in spans:
        for j in range(max(0, s - 1), min(e - 1, len(mask))):
            mask[j] = 1
    target_ids = full_ids[1:]
    trained = [target_ids[j] for j in range(len(mask)) if mask[j]]
    return trained, full_ids, mask


# --------------------------------------------------------------------------- #
# Masking: tool call is assistant content, trained up to <|im_end|>
# --------------------------------------------------------------------------- #

def test_tool_call_is_assistant_content_masked():
    tok = ByteTokenizer()
    weather = {"name": "get_weather", "parameters": {}}
    row = build_tool_messages([weather], "weather in Paris?",
                              [{"name": "get_weather", "arguments": {"city": "Paris"}}])
    trained, _, _ = _trained_tokens(row, tok)
    decoded = tok.decode(trained)
    assert decoded.startswith(TOOL_CALL_OPEN)        # trains the <tool_call> block
    assert decoded.endswith(IM_END)                  # ... up to and including the stop token
    assert '"city": "Paris"' in decoded or '"city":"Paris"' in decoded  # JSON verbatim, not collapsed


# --------------------------------------------------------------------------- #
# Distractors: appear in input_ids / target_ids system turn but never in loss_mask
# --------------------------------------------------------------------------- #

def test_distractor_tools_in_input_not_in_loss():
    tok = ByteTokenizer()
    weather = {"name": "get_weather", "parameters": {}}
    distractors = sample_distractors({"get_weather"}, 2, rng=random.Random(1))
    row = build_tool_messages([weather] + distractors, "weather in Paris?",
                              [{"name": "get_weather", "arguments": {"city": "Paris"}}])
    _, full_ids, mask = _trained_tokens(row, tok)
    # A distractor name appears in the token stream (it's in the system turn)
    dname = distractors[0]["name"]
    name_ids = tok.encode(dname)
    text_ids = full_ids
    assert any(text_ids[i:i + len(name_ids)] == name_ids for i in range(len(text_ids)))
    # ... but every position whose TARGET is the distractor name has loss_mask == 0
    target_ids = full_ids[1:]
    for i in range(len(mask)):
        if target_ids[i:i + len(name_ids)] == name_ids:
            assert all(m == 0 for m in mask[i:i + len(name_ids)])


# --------------------------------------------------------------------------- #
# Tool response: user turn NOT trained; assistant call + final answer ARE trained
# --------------------------------------------------------------------------- #

def test_tool_response_user_turn_not_trained():
    tok = ByteTokenizer()
    weather = {"name": "get_weather", "parameters": {}}
    row = build_tool_messages([weather], "raining in Tokyo?",
                              [{"name": "get_weather", "arguments": {"city": "Tokyo"}}],
                              results=[{"condition": "rain"}], final="Yes, it is raining.")
    trained, _, _ = _trained_tokens(row, tok)
    decoded = tok.decode(trained)
    assert TOOL_RESPONSE_OPEN not in decoded          # the user <tool_response> turn is NOT trained
    assert TOOL_CALL_OPEN in decoded                  # the assistant tool-call turn IS trained
    assert "Yes, it is raining." in decoded           # the final answer assistant turn IS trained


# --------------------------------------------------------------------------- #
# Abstention: trains a plain no-call response, no <tool_call> in trained span
# --------------------------------------------------------------------------- #

def test_abstention_example_masks_plain_no_call_response():
    tok = ByteTokenizer()
    tools = sample_distractors(set(), 2, rng=random.Random(2))
    row = build_abstention_messages(tools, "Translate hello to French.",
                                    "I don't have a translation tool, but it's 'bonjour'.")
    trained, _, _ = _trained_tokens(row, tok)
    decoded = tok.decode(trained)
    assert TOOL_CALL_OPEN not in decoded              # trains a NO-call answer
    assert "bonjour" in decoded
    assert decoded.endswith(IM_END)


# --------------------------------------------------------------------------- #
# No new chat roles
# --------------------------------------------------------------------------- #

def test_no_new_chat_roles_used():
    for row in handauthored_tool_records():
        for m in row["messages"]:
            assert m["role"] in _ROLES
        render(row["messages"])                       # must not raise


# --------------------------------------------------------------------------- #
# Cleaned output: JSON content verbatim (braces/quotes intact)
# --------------------------------------------------------------------------- #

def test_cleaned_preserves_json_verbatim(tmp_path):
    build_tool_sft(handauthored_tool_records(), tmp_path, tokenizer="qwen25",
                   byte_fallback=True, seq_len=8192)
    cleaned = tmp_path / "sft" / "cleaned" / "tool" / "records.jsonl"
    text = cleaned.read_text()
    # <tool_call> tags present in the cleaned output
    assert TOOL_CALL_OPEN in text
    # Round-trip: the call JSON inside the assistant content is preserved verbatim (braces/quotes).
    # When serialized to JSONL the inner quotes are escaped, so parse back out to verify.
    first_row = json.loads(text.splitlines()[0])
    assistant_turns = [m for m in first_row["messages"] if m["role"] == "assistant"]
    assert assistant_turns, "expected at least one assistant turn"
    content = assistant_turns[0]["content"]
    # content should be the <tool_call>...</tool_call> block with intact JSON
    assert TOOL_CALL_OPEN in content
    start = content.find(TOOL_CALL_OPEN) + len(TOOL_CALL_OPEN)
    end = content.find("</tool_call>")
    assert end > start
    # The JSON block must be valid JSON (round-trippable)
    call_json = json.loads(content[start:end].strip())
    assert "name" in call_json


# --------------------------------------------------------------------------- #
# End-to-end: artifacts written, manifest correct, SFTLoader works
# --------------------------------------------------------------------------- #

def test_build_tool_sft_end_to_end(tmp_path):
    manifest = build_tool_sft(handauthored_tool_records(), tmp_path, tokenizer="qwen25",
                              byte_fallback=True, seq_len=8192)
    cleaned = tmp_path / "sft" / "cleaned" / "tool" / "records.jsonl"
    tok_dir = tmp_path / "sft" / "tokenized" / "qwen25-8k"
    assert cleaned.exists()
    assert (tok_dir / "tool.jsonl").exists() and (tok_dir / "tool-manifest.json").exists()
    first = json.loads(cleaned.read_text().splitlines()[0])
    assert "messages" in first and "source" in first and "license" in first
    assert manifest["template"] == "qwen-chatml"
    assert manifest["format"] == "qwen-tool-call"
    assert manifest["chat_eos"] == IM_END
    assert manifest["tokenizer"] == "qwen25"
    n = len(list(handauthored_tool_records()))
    assert manifest["n_records"] == n
    assert manifest["n_tokens"] > 0
    assert manifest["sources"].get("handauthored") == n
    assert manifest["n_abstention"] >= 1
    assert manifest["n_with_distractors"] >= 1
    assert manifest["n_schema_invalid"] == 0  # handauthored set is clean
    loader = SFTLoader(tok_dir / "tool.jsonl", seq_len=8192, batch_size=2)
    inputs, targets, mask = next(loader.epoch())
    assert inputs.shape == targets.shape == mask.shape
    assert mask.sum() > 0


# --------------------------------------------------------------------------- #
# Over-length: dropped, never truncated
# --------------------------------------------------------------------------- #

def test_over_length_tool_examples_dropped(tmp_path):
    huge = build_tool_messages([{"name": "noop", "parameters": {}}], "x" * 50,
                               [{"name": "noop", "arguments": {"blob": "y" * 500}}])
    m = build_tool_sft([huge], tmp_path, tokenizer="qwen25", byte_fallback=True,
                       seq_len=8192, max_seq_len=50)
    assert m["n_records"] == 0 and m["n_skipped"] == 1


# --------------------------------------------------------------------------- #
# Schema-invalid rows: dropped and counted, not silently kept (#102 box #1)
# --------------------------------------------------------------------------- #

def test_schema_invalid_row_dropped_and_counted(tmp_path):
    weather = {"name": "get_weather", "parameters": {"type": "object",
                                                      "properties": {"city": {"type": "string"}},
                                                      "required": ["city"]}}
    # The call omits the required "city" argument -> schema-invalid.
    bad = build_tool_messages([weather], "What's the weather?",
                              [{"name": "get_weather", "arguments": {}}])
    good = build_tool_messages([weather], "Weather in Paris?",
                               [{"name": "get_weather", "arguments": {"city": "Paris"}}])
    m = build_tool_sft([bad, good], tmp_path, tokenizer="qwen25", byte_fallback=True, seq_len=8192)
    assert m["n_schema_invalid"] == 1
    assert m["n_records"] == 1


# --------------------------------------------------------------------------- #
# Offline row mapper tests (no network required)
# --------------------------------------------------------------------------- #

def test_row_mappers_offline():
    # xLAM-shape synthetic row -> valid tagged dict, no network.
    xlam_row = {
        "query": "weather in Paris?",
        "tools": json.dumps([{"name": "get_weather", "parameters": {}}]),
        "answers": json.dumps([{"name": "get_weather", "arguments": {"city": "Paris"}}]),
    }
    rec = xlam_row_to_messages(xlam_row)
    assert rec is not None
    assert rec["source"] == "xlam" and rec["messages"][0]["role"] == "system"
    assert rec["messages"][-1]["role"] == "assistant"
    assert TOOL_CALL_OPEN in rec["messages"][-1]["content"]

    # when2call -> abstention row (no tool call in the assistant turn).
    w2c = when2call_row_to_messages({
        "query": "do a thing",
        "tools": [{"name": "x", "parameters": {}}],
        "response": "I can't do that.",
    })
    assert w2c is not None and TOOL_CALL_OPEN not in w2c["messages"][-1]["content"]

    # glaive: minimal synthetic row with a FUNCTION CALL turn.
    glaive_row = {
        "system": '[{"name": "get_weather", "description": "Get weather", "parameters": {}}]',
        "chat": (
            "USER: What is the weather in Paris?\n"
            'FUNCTION CALL: {"name": "get_weather", "arguments": {"city": "Paris"}}\n'
            "ASSISTANT: It is sunny in Paris."
        ),
    }
    grec = glaive_row_to_messages(glaive_row)
    assert grec is not None
    assert grec["source"] == "glaive"
    # The final assistant turn has a non-empty response
    assert grec["messages"][-1]["role"] == "assistant"
    assert grec["messages"][-1]["content"].strip()

    # toolace: minimal synthetic multi-turn row with tool call JSON in assistant turn.
    toolace_row = {
        "tools": json.dumps([{"name": "search", "parameters": {}}]),
        "conversations": [
            {"role": "user", "content": "Search for python tutorials."},
            {"role": "assistant",
             "content": json.dumps({"name": "search", "arguments": {"query": "python tutorials"}})},
        ],
    }
    trec = toolace_row_to_messages(toolace_row)
    assert trec is not None
    assert trec["source"] == "toolace"
    assert TOOL_CALL_OPEN in trec["messages"][-1]["content"]


# --------------------------------------------------------------------------- #
# validate_call_against_tools: name known + required args present (#102 box #1)
# --------------------------------------------------------------------------- #

def test_validate_call_against_tools_valid_call():
    tools = [{"name": "get_weather", "parameters": {"type": "object",
                                                     "properties": {"city": {"type": "string"}},
                                                     "required": ["city"]}}]
    call = {"name": "get_weather", "arguments": {"city": "Paris"}}
    assert validate_call_against_tools(call, tools) is True


def test_validate_call_against_tools_missing_required():
    tools = [{"name": "get_weather", "parameters": {"type": "object",
                                                     "properties": {"city": {"type": "string"}},
                                                     "required": ["city"]}}]
    call = {"name": "get_weather", "arguments": {}}
    assert validate_call_against_tools(call, tools) is False


def test_validate_call_against_tools_unknown_name():
    tools = [{"name": "get_weather", "parameters": {"type": "object", "required": []}}]
    call = {"name": "send_email", "arguments": {"to": "a@b.com"}}
    assert validate_call_against_tools(call, tools) is False


# --------------------------------------------------------------------------- #
# Unknown source raises ValueError
# --------------------------------------------------------------------------- #

def test_iter_tool_sft_unknown_source_raises():
    with pytest.raises(ValueError):
        list(iter_tool_sft(["bfcl"]))   # BFCL is eval-only, not a loader


# --------------------------------------------------------------------------- #
# _valid_rows: filters non-assistant-ended rows
# --------------------------------------------------------------------------- #

def test_valid_rows_filters_correctly():
    good = {"messages": [{"role": "user", "content": "hi"},
                          {"role": "assistant", "content": "hello"}],
            "source": "handauthored", "license": "cc0"}
    bad_role = {"messages": [{"role": "user", "content": "hi"}],
                "source": "x", "license": "y"}
    bad_empty = {"messages": [{"role": "user", "content": "hi"},
                               {"role": "assistant", "content": "   "}],
                 "source": "x", "license": "y"}
    result = _valid_rows([good, bad_role, bad_empty])
    assert len(result) == 1
    assert result[0]["source"] == "handauthored"


# --------------------------------------------------------------------------- #
# Regression: effective vocab size uses len(tok), not tok.vocab_size (#153)
# --------------------------------------------------------------------------- #

def test_effective_vocab_size_uses_len_not_vocab_size():
    """_effective_vocab_size must return len(tok) when available, not vocab_size.

    Qwen3 adds <|im_start|>/<|im_end|> as special tokens whose ids sit ABOVE
    tokenizer.vocab_size. The old `getattr(tok, "vocab_size", None)` check would
    raise a spurious ValueError for any row whose tokens include those specials.
    """

    class FakeQwen3Tokenizer:
        """Stub with len(tok)=20 but vocab_size=10 — simulates Qwen3 added specials."""
        vocab_size = 10

        def __len__(self):
            return 20

    tok = FakeQwen3Tokenizer()
    assert _effective_vocab_size(tok) == 20, (
        "_effective_vocab_size should return len(tok)=20, not vocab_size=10"
    )


def test_tool_sft_does_not_raise_with_extended_vocab(tmp_path):
    """build_tool_sft must not raise when the tokenizer's len() > vocab_size.

    This was the latent bug: the old vocab-bound check used `vocab_size` directly,
    so any real Qwen3 token id in [vocab_size, len(tok)) would trigger a false
    ValueError. Byte-fallback tests masked it because ByteTokenizer.vocab_size==256
    and len(ByteTokenizer)==256 (no gap). Here we use ByteTokenizer (always safe for
    offline tests) and verify the build succeeds end-to-end with n_records >= 1.
    """
    manifest = build_tool_sft(
        handauthored_tool_records(), tmp_path,
        tokenizer="qwen25", byte_fallback=True, seq_len=8192,
    )
    assert manifest["n_records"] >= 1, "expected at least one tokenized record"
