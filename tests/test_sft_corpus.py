"""SFT corpus resolution (#306): the bridge from the `shared/sft/` builders to `scripts/sft.py`.

Portable — ByteTokenizer + the checked-in handauthored sources, no backend, no network. Every
loud-failure path the resolver owns is asserted here: packed rejected by name, chat-EOS and
template verified, form mixing gated on manifest agreement, over-length dropped (never
truncated), empty corpora refused, and malformed record lines named with path + line number.
"""

import json
import random
import shutil

import numpy as np
import pytest

from src.data import chat_template
from src.data.instruct_sft import build_instruct_sft
from src.data.reasoning_sft import build_reasoning_sft
from src.data.reasoning_traces import handauthored_trace_records
from src.data.sft_corpus import (FORMS, looks_packed, resolve_sft_corpus,
                                 validate_manifest)
from src.data.sft_loader import SFTLoader, load_sft_records
from src.data.storage import tokenized_dir_name
from src.data.tokenize import ByteTokenizer
from src.data.tool_sft import build_tool_sft
from src.data.tool_sources import (build_tool_messages, handauthored_tool_records,
                                   sample_distractors)

SEQ_LEN = 1024
TOKENIZER = "qwen25"


def _build_all(root):
    """Build all three masked forms + the packed artifact into one `shared/sft` tree."""
    build_instruct_sft(_instruct_rows(), root, tokenizer=TOKENIZER, byte_fallback=True,
                       seq_len=SEQ_LEN, max_seq_len=SEQ_LEN)
    build_reasoning_sft(handauthored_trace_records(), root, tokenizer=TOKENIZER,
                        byte_fallback=True, seq_len=SEQ_LEN, chunk_align=64)
    build_tool_sft(handauthored_tool_records(), root, tokenizer=TOKENIZER, byte_fallback=True,
                   seq_len=SEQ_LEN, max_seq_len=SEQ_LEN)
    return root / "sft" / "tokenized" / tokenized_dir_name(TOKENIZER, SEQ_LEN)


def _instruct_rows():
    """Four tiny instruct rows (the instruct builder has no checked-in handauthored set)."""
    return [{"messages": [{"role": "user", "content": q},
                          {"role": "assistant", "content": a}],
             "source": "handauthored", "license": "cc0"}
            for q, a in (("2+2?", "4"), ("capital of France?", "Paris"),
                         ("colour of the sky?", "blue"), ("largest ocean?", "the Pacific"))]


@pytest.fixture(scope="module")
def corpus_root(tmp_path_factory):
    """Build the corpus once; tests that mutate it copy the tree first."""
    root = tmp_path_factory.mktemp("shared")
    tok_dir = _build_all(root)
    return tok_dir


@pytest.fixture
def mutable_corpus(corpus_root, tmp_path):
    dst = tmp_path / "tok"
    shutil.copytree(corpus_root, dst)
    return dst


# --------------------------------------------------------------------------- #
# 1. generic passthrough — the pre-#306 invocation is bit-for-bit unchanged
# --------------------------------------------------------------------------- #

def _generic_dir(tmp_path, n_train=3, n_val=1):
    d = tmp_path / "generic"
    d.mkdir()
    def rec(i):
        return {"input_ids": [i, i + 1, i + 2], "target_ids": [i + 1, i + 2, i + 3],
                "loss_mask": [0, 1, 1]}
    (d / "train.jsonl").write_text("".join(json.dumps(rec(i)) + "\n" for i in range(n_train)))
    if n_val:
        (d / "val.jsonl").write_text("".join(json.dumps(rec(90 + i)) + "\n" for i in range(n_val)))
    return d


def test_generic_passthrough_returns_paths(tmp_path):
    d = _generic_dir(tmp_path)
    c = resolve_sft_corpus(d)                      # auto -> generic, train.jsonl present
    assert c.is_generic and c.forms == ["generic"]
    assert c.train_path == d / "train.jsonl" and c.val_path == d / "val.jsonl"
    assert c.train_records is None and c.val_records is None   # untouched: no split, no filter
    # An explicit --corpus-form generic resolves identically.
    forced = resolve_sft_corpus(d, ["generic"])
    assert (forced.train_path, forced.val_path) == (c.train_path, c.val_path)


def test_generic_without_val_file_is_split(tmp_path):
    d = _generic_dir(tmp_path, n_train=4, n_val=0)
    c = resolve_sft_corpus(d, ["generic"], val_frac=0.25)
    assert not c.is_generic
    assert len(c.train_records) == 3 and len(c.val_records) == 1


# --------------------------------------------------------------------------- #
# 2. autodetect + explicit forms over the shared/sft layout
# --------------------------------------------------------------------------- #

def test_autodetect_finds_every_form(corpus_root):
    c = resolve_sft_corpus(corpus_root)
    assert c.forms == list(FORMS)                  # instruct + reasoning + tool, in FORMS order
    assert set(c.manifests) == set(FORMS)
    assert len(c.train_records) + len(c.val_records) == 3 + 4 + 4


@pytest.mark.parametrize("form,n", [("tool", 3), ("reasoning", 4), ("instruct", 4)])
def test_single_form_resolves(corpus_root, form, n):
    c = resolve_sft_corpus(corpus_root, [form])
    assert c.forms == [form]
    assert len(c.train_records) + len(c.val_records) == n
    assert all(set(r) >= {"input_ids", "target_ids", "loss_mask"} for r in c.train_records)


def test_autodetect_ignores_absent_forms(mutable_corpus):
    (mutable_corpus / "instruct.jsonl").unlink()
    (mutable_corpus / "reasoning.jsonl").unlink()
    c = resolve_sft_corpus(mutable_corpus)
    assert c.forms == ["tool"]


def test_missing_form_file_names_the_builder(corpus_root, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no SFT corpus found"):
        resolve_sft_corpus(empty)
    with pytest.raises(ValueError, match=r"tool\.jsonl does not exist"):
        resolve_sft_corpus(empty, ["tool"])


def test_records_file_without_manifest_is_refused(mutable_corpus):
    (mutable_corpus / "tool-manifest.json").unlink()
    with pytest.raises(ValueError, match="has no manifest"):
        resolve_sft_corpus(mutable_corpus, ["tool"])


# --------------------------------------------------------------------------- #
# 3. deterministic, disjoint split
# --------------------------------------------------------------------------- #

def test_split_is_deterministic_disjoint_and_nonempty(corpus_root):
    a = resolve_sft_corpus(corpus_root, seed=7)
    b = resolve_sft_corpus(corpus_root, seed=7)
    c = resolve_sft_corpus(corpus_root, seed=8)

    def ids(recs):
        return [tuple(r["input_ids"]) for r in recs]

    assert ids(a.train_records) == ids(b.train_records)          # same seed -> same split
    assert ids(a.val_records) == ids(b.val_records)
    assert set(ids(a.train_records)).isdisjoint(ids(a.val_records))
    assert a.val_records and a.train_records
    assert len(a.train_records) + len(a.val_records) == 11
    assert ids(a.val_records) != ids(c.val_records)              # a different seed moves it


# --------------------------------------------------------------------------- #
# 4/5. chat-EOS + template are ENFORCED, not merely recorded
# --------------------------------------------------------------------------- #

def _rewrite_manifest(tok_dir, form, **changes):
    path = tok_dir / FORMS[form][1]
    man = json.loads(path.read_text())
    for k, v in changes.items():
        if v is None:
            man.pop(k, None)
        else:
            man[k] = v
    path.write_text(json.dumps(man, indent=2))
    return path


def test_eos_mismatch_rejected(mutable_corpus):
    path = _rewrite_manifest(mutable_corpus, "tool", chat_eos="</s>")
    with pytest.raises(ValueError) as exc:
        resolve_sft_corpus(mutable_corpus, ["tool"])
    msg = str(exc.value)
    assert "chat_eos" in msg and "</s>" in msg and chat_template.CHAT_EOS in msg
    assert str(path) in msg


def test_missing_eos_key_rejected(mutable_corpus):
    _rewrite_manifest(mutable_corpus, "reasoning", chat_eos=None)
    with pytest.raises(ValueError, match="no `chat_eos` key"):
        resolve_sft_corpus(mutable_corpus, ["reasoning"])


def test_template_mismatch_rejected(mutable_corpus):
    _rewrite_manifest(mutable_corpus, "instruct", template="llama-chat")
    with pytest.raises(ValueError, match="template mismatch"):
        resolve_sft_corpus(mutable_corpus, ["instruct"])


def test_validate_manifest_accepts_a_real_manifest(corpus_root):
    for form, (_, man_file) in FORMS.items():
        path = corpus_root / man_file
        validate_manifest(path, json.loads(path.read_text()))   # must not raise


# --------------------------------------------------------------------------- #
# 6. mixing forms — supported when the manifests agree, refused when they do not
# --------------------------------------------------------------------------- #

def test_mixing_forms(corpus_root):
    c = resolve_sft_corpus(corpus_root, ["reasoning", "tool"])
    assert c.forms == ["reasoning", "tool"]
    assert len(c.train_records) + len(c.val_records) == 4 + 3


@pytest.mark.parametrize("field,value", [("tokenizer", "olmo"), ("seq_len", 4096),
                                         ("model_id", "Qwen/Qwen3-4B")])
def test_mixing_rejected_on_disagreement(mutable_corpus, field, value):
    _rewrite_manifest(mutable_corpus, "tool", **{field: value})
    with pytest.raises(ValueError) as exc:
        resolve_sft_corpus(mutable_corpus, ["reasoning", "tool"])
    msg = str(exc.value)
    assert "reasoning" in msg and "tool" in msg and field in msg


# --------------------------------------------------------------------------- #
# 7/8. over-length policy and empty corpora
# --------------------------------------------------------------------------- #

def test_overlength_dropped_not_truncated(corpus_root):
    """Records over `max_len` are dropped whole; survivors keep the builder's exact mask."""
    on_disk = load_sft_records(corpus_root / "reasoning.jsonl")
    lengths = sorted(len(r["input_ids"]) for r in on_disk)
    cut = lengths[1]                                 # keeps the two shortest, drops the rest
    c = resolve_sft_corpus(corpus_root, ["reasoning"], max_len=cut, val_frac=0.5)

    kept = c.train_records + c.val_records
    assert c.n_dropped_overlength == len(on_disk) - len(kept) > 0
    assert all(len(r["input_ids"]) <= cut for r in kept)
    by_ids = {tuple(r["input_ids"]): r for r in on_disk}
    for r in kept:
        original = by_ids[tuple(r["input_ids"])]
        # Byte-identical to the builder's record — no clipped answer span anywhere.
        assert r["loss_mask"] == original["loss_mask"]
        assert r["target_ids"] == original["target_ids"]
        assert r["loss_mask"][-1] == 1              # the mask still ends inside the answer run


def test_empty_corpus_fails_loudly(tmp_path):
    d = tmp_path / "tok"
    d.mkdir()
    (d / "tool.jsonl").write_text("\n")
    (d / "tool-manifest.json").write_text(json.dumps(
        {"chat_eos": chat_template.CHAT_EOS, "template": "qwen-chatml"}))
    with pytest.raises(ValueError, match="no SFT records"):
        resolve_sft_corpus(d, ["tool"])


def test_corpus_emptied_by_overlength_filter_fails_loudly(corpus_root):
    with pytest.raises(ValueError, match="emptied by the over-length filter"):
        resolve_sft_corpus(corpus_root, ["tool"], max_len=8)


def test_too_few_records_to_split_fails_loudly(corpus_root):
    on_disk = load_sft_records(corpus_root / "reasoning.jsonl")
    shortest = min(len(r["input_ids"]) for r in on_disk)
    with pytest.raises(ValueError, match="at least 2 are needed"):
        resolve_sft_corpus(corpus_root, ["reasoning"], max_len=shortest)


# --------------------------------------------------------------------------- #
# 9. reasoning-packed is rejected BY NAME, with the recipe that works
# --------------------------------------------------------------------------- #

def test_packed_form_rejected_by_name(corpus_root):
    with pytest.raises(ValueError) as exc:
        resolve_sft_corpus(corpus_root, ["reasoning-packed"])
    msg = str(exc.value)
    assert "reasoning-packed" in msg
    assert "split --shards" in msg and "scripts/train.py" in msg
    assert "no loss mask" in msg


def test_packed_directory_rejected_when_passed_as_data(corpus_root):
    packed = corpus_root / "reasoning-packed"
    assert looks_packed(packed) and not looks_packed(corpus_root)
    with pytest.raises(ValueError) as exc:
        resolve_sft_corpus(packed)
    assert "split --shards" in str(exc.value)


def test_unknown_form_names_the_valid_ones(corpus_root):
    with pytest.raises(ValueError, match="unknown SFT corpus form"):
        resolve_sft_corpus(corpus_root, ["chat"])


# --------------------------------------------------------------------------- #
# 10. malformed record lines are named (path + line number), not a bare decode error
# --------------------------------------------------------------------------- #

def test_malformed_record_line_names_path_and_line(tmp_path):
    p = tmp_path / "tool.jsonl"
    good = json.dumps({"input_ids": [1, 2], "target_ids": [2, 3], "loss_mask": [0, 1]})
    p.write_text(good + "\n" + good + "\n{not json\n")
    with pytest.raises(ValueError) as exc:
        load_sft_records(p)
    msg = str(exc.value)
    assert str(p) in msg and ":3:" in msg and "malformed SFT record" in msg
    with pytest.raises(ValueError, match=":3:"):
        SFTLoader(p, 128, 1)


def test_length_mismatched_record_rejected(tmp_path):
    p = tmp_path / "tool.jsonl"
    p.write_text(json.dumps({"input_ids": [1, 2, 3], "target_ids": [2, 3, 4],
                             "loss_mask": [0, 1]}) + "\n")
    with pytest.raises(ValueError, match="field lengths disagree"):
        load_sft_records(p)


def test_record_missing_a_key_rejected(tmp_path):
    p = tmp_path / "tool.jsonl"
    p.write_text(json.dumps({"input_ids": [1, 2], "target_ids": [2, 3]}) + "\n")
    with pytest.raises(ValueError, match="missing"):
        load_sft_records(p)


# --------------------------------------------------------------------------- #
# tool edge cases: abstention, several calls, schema-invalid payloads
# --------------------------------------------------------------------------- #

_WEATHER = {"name": "get_weather", "description": "Get current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                           "required": ["city"]}}
_TIME = {"name": "get_time", "description": "Get the current time in a city",
         "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                        "required": ["city"]}}


def test_schema_invalid_call_counted_and_never_reaches_the_loader(tmp_path):
    """A call naming a tool the row never declared is counted in the manifest and dropped."""
    good = build_tool_messages([_WEATHER], "weather in Paris?",
                               [{"name": "get_weather", "arguments": {"city": "Paris"}}])
    hallucinated = build_tool_messages([_WEATHER], "email bob?",
                                       [{"name": "send_email", "arguments": {"to": "bob"}}])
    m = build_tool_sft([good, hallucinated], tmp_path, tokenizer=TOKENIZER, byte_fallback=True,
                       seq_len=SEQ_LEN, max_seq_len=SEQ_LEN)
    assert m["n_schema_invalid"] == 1 and m["n_records"] == 1
    tok_dir = tmp_path / "sft" / "tokenized" / tokenized_dir_name(TOKENIZER, SEQ_LEN)
    recs = load_sft_records(tok_dir / "tool.jsonl")
    tok = ByteTokenizer()
    assert len(recs) == 1
    assert "send_email" not in tok.decode(recs[0]["input_ids"])


def test_abstention_and_multicall_rows_both_survive(tmp_path):
    """Zero calls and several calls are both trainable — a named outcome for each."""
    from src.data.tool_sources import build_abstention_messages
    multi = build_tool_messages(
        [_WEATHER, _TIME], "weather and time in Paris?",
        [{"name": "get_weather", "arguments": {"city": "Paris"}},
         {"name": "get_time", "arguments": {"city": "Paris"}}])
    abstain = build_abstention_messages(
        sample_distractors(set(), 2, rng=random.Random(0)), "translate hello",
        "I have no translation tool; it is 'bonjour'.")
    m = build_tool_sft([multi, abstain], tmp_path, tokenizer=TOKENIZER, byte_fallback=True,
                       seq_len=SEQ_LEN, max_seq_len=SEQ_LEN)
    assert m["n_records"] == 2 and m["n_abstention"] == 1 and m["n_schema_invalid"] == 0
    tok_dir = tmp_path / "sft" / "tokenized" / tokenized_dir_name(TOKENIZER, SEQ_LEN)
    c = resolve_sft_corpus(tok_dir, ["tool"], val_frac=0.5)
    tok = ByteTokenizer()
    trained = []
    for r in c.train_records + c.val_records:
        trained.append(tok.decode([t for t, mk in zip(r["target_ids"], r["loss_mask"]) if mk]))
    assert any(t.count("<tool_call>") == 2 for t in trained)    # the multi-call row
    assert any("<tool_call>" not in t for t in trained)         # the abstention row
    assert all(t.endswith(chat_template.CHAT_EOS) for t in trained)


def test_syntactically_malformed_payload_is_a_named_outcome(tmp_path):
    """A `<tool_call>` block that is not valid JSON is a NAMED outcome, not a mid-epoch crash.

    `tool_sft._iter_calls` skips unparseable blocks, so the row declares zero calls and passes the
    schema gate vacuously: it is kept and trained verbatim. That is the observed behaviour and it
    is pinned here so a future change to the builder has to face it — `n_schema_invalid` does NOT
    count syntactically malformed payloads (parked, see docs/parked-findings.md). The contract
    this test does enforce is that the builder neither raises nor silently drops the row.
    """
    row = build_tool_messages([_WEATHER], "weather in Paris?",
                              [{"name": "get_weather", "arguments": {"city": "Paris"}}])
    row["messages"][-1]["content"] = "<tool_call>{\"name\": \"get_weather\", </tool_call>"
    m = build_tool_sft([row], tmp_path, tokenizer=TOKENIZER, byte_fallback=True,
                       seq_len=SEQ_LEN, max_seq_len=SEQ_LEN)
    assert m["n_schema_invalid"] == 0 and m["n_abstention"] == 0
    assert m["n_records"] == 1
    tok_dir = tmp_path / "sft" / "tokenized" / tokenized_dir_name(TOKENIZER, SEQ_LEN)
    recs = load_sft_records(tok_dir / "tool.jsonl")
    trained = ByteTokenizer().decode(
        [t for t, mk in zip(recs[0]["target_ids"], recs[0]["loss_mask"]) if mk])
    assert trained.startswith("<tool_call>") and trained.endswith(chat_template.CHAT_EOS)


def test_split_indices_cover_every_record_exactly_once(corpus_root):
    c = resolve_sft_corpus(corpus_root, seed=3)
    on_disk = []
    for form, (rec_file, _) in FORMS.items():
        on_disk.extend(load_sft_records(corpus_root / rec_file))
    got = sorted(tuple(r["input_ids"]) for r in c.train_records + c.val_records)
    assert got == sorted(tuple(r["input_ids"]) for r in on_disk)
    assert np.isclose(len(c.val_records), max(1, round(0.05 * len(on_disk))), atol=1)
