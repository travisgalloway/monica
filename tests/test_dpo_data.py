"""DPO record builder (portable). Verifies both sides share the prompt prefix, each
masks only its own response, and malformed rows are skipped — offline with
ByteTokenizer."""

from __future__ import annotations

import numpy as np

from src.data.dpo_data import build_dpo_records
from src.data.tokenize import ByteTokenizer


def _msgs(prompt, answer):
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]


def _row(prompt, chosen, rejected):
    return {"prompt": prompt, "chosen": _msgs(prompt, chosen),
            "rejected": _msgs(prompt, rejected)}


def _decode_response(tok, rec, side):
    ids = [t for t, m in zip(rec[f"{side}_target_ids"], rec[f"{side}_mask"]) if m]
    return tok.decode(ids).strip()


def test_builds_both_sides_with_response_masks():
    tok = ByteTokenizer()
    rows = [_row("Question", "Good answer", "Bad answer")]
    (rec,) = list(build_dpo_records(rows, tok))
    assert _decode_response(tok, rec, "chosen") == "Good answer"
    assert _decode_response(tok, rec, "rejected") == "Bad answer"


def test_prompt_tokens_are_masked_out_on_both_sides():
    tok = ByteTokenizer()
    rows = [_row("Secret prompt", "yes", "no")]
    (rec,) = list(build_dpo_records(rows, tok))
    for side in ("chosen", "rejected"):
        assert "Secret prompt" not in _decode_response(tok, rec, side)


def test_shared_prompt_prefix_is_identical():
    tok = ByteTokenizer()
    rows = [_row("Same prompt here", "alpha", "beta")]
    (rec,) = list(build_dpo_records(rows, tok))
    # The two sides differ only in the response, so the prompt prefix tokens match up to
    # the first differing position, which must be well past the start (a real prompt).
    c_in, r_in = rec["chosen_input_ids"], rec["rejected_input_ids"]
    common = next((i for i in range(min(len(c_in), len(r_in))) if c_in[i] != r_in[i]),
                  min(len(c_in), len(r_in)))
    assert common > 0 and c_in[:common] == r_in[:common]


def test_skips_rows_missing_a_side():
    tok = ByteTokenizer()
    rows = [{"prompt": "Q", "chosen": _msgs("Q", "a"), "rejected": []},  # no rejected
            {"prompt": "", "chosen": _msgs("Q", "a"), "rejected": _msgs("Q", "b")}]  # no prompt
    stats = {}
    out = list(build_dpo_records(rows, tok, stats=stats))
    assert out == [] and stats["skipped"] == 2


def test_over_length_is_skipped():
    tok = ByteTokenizer()
    rows = [_row("Q", "x" * 100, "y")]
    stats = {}
    out = list(build_dpo_records(rows, tok, max_seq_len=10, stats=stats))
    assert out == [] and stats["skipped"] == 1
