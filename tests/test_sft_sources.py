"""Clean-license SFT sources (#76).

Hermetic — no network. The OASST1 tree reconstruction and the format/license tagging are
the contract; verified end to end through `build_sft_records`.
"""

from src.data.sft_sources import (HANDAUTHORED, SOURCE_LICENSES, build_oasst1_threads,
                                  flan_to_messages, handauthored_records, iter_clean_sft,
                                  ultrachat_row_to_messages)
from src.data.sft_data import build_sft_records
from src.data.tokenize import ByteTokenizer


def _oasst_rows():
    # prompter -> assistant -> prompter -> assistant, plus a second assistant leaf.
    return [
        {"message_id": "a", "parent_id": None, "role": "prompter", "text": "Hi", "lang": "en"},
        {"message_id": "b", "parent_id": "a", "role": "assistant", "text": "Hello!", "lang": "en"},
        {"message_id": "c", "parent_id": "b", "role": "prompter", "text": "Bye", "lang": "en"},
        {"message_id": "d", "parent_id": "c", "role": "assistant", "text": "Goodbye!", "lang": "en"},
        {"message_id": "e", "parent_id": "a", "role": "assistant", "text": "Hey there", "lang": "en"},
    ]


def test_oasst1_reconstructs_multiturn_threads():
    recs = list(build_oasst1_threads(_oasst_rows(), lang="en"))
    # 3 assistant nodes (b, d, e) -> 3 examples; each ends in assistant, starts with user.
    assert len(recs) == 3
    for r in recs:
        assert r["source"] == "oasst1" and r["license"] == "apache-2.0"
        assert r["messages"][0]["role"] == "user"
        assert r["messages"][-1]["role"] == "assistant"
        roles = [m["role"] for m in r["messages"]]
        assert all(roles[i] != roles[i + 1] for i in range(len(roles) - 1))   # alternates
    # the deepest thread (b->...->d) is the 4-message one
    longest = max(recs, key=lambda r: len(r["messages"]))
    assert [m["content"] for m in longest["messages"]] == ["Hi", "Hello!", "Bye", "Goodbye!"]


def test_oasst1_language_filter():
    rows = _oasst_rows() + [
        {"message_id": "x", "parent_id": None, "role": "prompter", "text": "Hola", "lang": "es"},
        {"message_id": "y", "parent_id": "x", "role": "assistant", "text": "Buenas", "lang": "es"},
    ]
    en = list(build_oasst1_threads(rows, lang="en"))
    assert all(all("Hola" != m["content"] for m in r["messages"]) for r in en)
    assert len(en) == 3                                   # spanish thread excluded


def test_flan_to_messages():
    assert flan_to_messages({"inputs": "", "targets": "x"}) is None
    rec = flan_to_messages({"inputs": "Translate hi", "targets": "bonjour"})
    assert rec["source"] == "flan" and rec["license"] == "apache-2.0"
    assert rec["messages"] == [{"role": "user", "content": "Translate hi"},
                               {"role": "assistant", "content": "bonjour"}]


def test_handauthored_set_is_clean_and_multiturn():
    recs = list(handauthored_records())
    assert len(recs) == len(HANDAUTHORED)
    assert all(r["license"] == "cc0" for r in recs)
    assert any(len(r["messages"]) > 2 for r in recs)      # at least one multi-turn example


def test_sources_feed_build_sft_records():
    rows = list(iter_clean_sft(["handauthored"]))
    rows += list(build_oasst1_threads(_oasst_rows()))
    tok = ByteTokenizer()
    stats: dict = {}
    out = list(build_sft_records(rows, tok, stats=stats))
    assert out and stats["kept"] == len(out)
    r0 = out[0]
    assert set(r0) == {"input_ids", "target_ids", "loss_mask"}
    assert len(r0["input_ids"]) == len(r0["target_ids"]) == len(r0["loss_mask"])
    assert any(r["loss_mask"] for r in out)               # response tokens supervised


def test_source_licenses_cover_loaders():
    assert set(SOURCE_LICENSES) >= {"oasst1", "flan", "dolly", "handauthored", "ultrachat"}


def test_ultrachat_row_to_messages():
    row = {"messages": [{"role": "user", "content": "Hi"},
                        {"role": "assistant", "content": "Hello!"}]}
    rec = ultrachat_row_to_messages(row)
    assert rec is not None
    assert rec["source"] == "ultrachat" and rec["license"] == "mit"
    assert rec["messages"] == [{"role": "user", "content": "Hi"},
                               {"role": "assistant", "content": "Hello!"}]


def test_ultrachat_row_to_messages_skips_malformed():
    assert ultrachat_row_to_messages({}) is None
    assert ultrachat_row_to_messages({"messages": "not-a-list"}) is None
    assert ultrachat_row_to_messages(
        {"messages": [{"role": "user", "content": "only a question"}]}) is None
    assert ultrachat_row_to_messages(
        {"messages": [{"role": "user", "content": ""},
                      {"role": "assistant", "content": "ok"}]}) is None
