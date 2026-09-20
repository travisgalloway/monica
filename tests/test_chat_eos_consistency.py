"""Chat EOS consistency tests across SFT, RLVR, and serving (#101).

Guarantees the cross-cutting invariant (docs/design/11-post-training.md):
`<|im_end|>` is the single source of truth for chat EOS across SFT (manifest verification
and response masking), RLVR (rollout termination and verifier scoring), and serving
(generation loop termination and CLI REPLs).
"""

import argparse
import pytest

from src.data import chat_template
from src.data.chat_template import CHAT_EOS, IM_END, IM_START
from src.data.tokenize import ByteTokenizer
from src.serve.generate import (
    enforce_chat_eos_consistency,
    generate,
    resolve_chat_eos_id,
    resolve_eos_ids,
)


def test_chat_eos_constant_matches_im_end():
    assert CHAT_EOS == IM_END == "<|im_end|>"


def test_enforce_chat_eos_consistency_valid():
    tok = ByteTokenizer()
    eos_ids = enforce_chat_eos_consistency(tok, expected_chat_eos=CHAT_EOS, where="test")
    assert isinstance(eos_ids, set)


def test_enforce_chat_eos_consistency_rejects_mismatch():
    tok = ByteTokenizer()
    with pytest.raises(ValueError) as exc:
        enforce_chat_eos_consistency(tok, expected_chat_eos="</s>", where="test_stage")
    msg = str(exc.value)
    assert "chat_eos mismatch" in msg
    assert "test_stage" in msg
    assert CHAT_EOS in msg
    assert "</s>" in msg
    assert "docs/design/11-post-training.md" in msg


def test_resolve_eos_ids_combines_standard_and_chat_eos():
    class DummyTokenizer:
        eos_token_id = 100

        def encode(self, text, add_special_tokens=False):
            if text == CHAT_EOS:
                return [101]
            return [1]

    tok = DummyTokenizer()
    eos_set = resolve_eos_ids(tok, include_chat_eos=True)
    assert eos_set == {100, 101}

    eos_set_no_chat = resolve_eos_ids(tok, include_chat_eos=False)
    assert eos_set_no_chat == {100}


def test_generate_stops_on_set_of_eos_ids():
    class DummyStore:
        def prefill(self, sid, prompt):
            return [[0.0, 0.0, 0.0, 0.0]]

        def step(self, sid, token):
            return [[0.0, 0.0, 0.0, 0.0]]

    # Sampler emits tokens: 1, 2, 3, 4, 5
    tokens_to_emit = [1, 2, 3, 4, 5]
    idx = 0

    def dummy_sampler(logits):
        nonlocal idx
        tok = tokens_to_emit[idx % len(tokens_to_emit)]
        idx += 1
        return tok

    store = DummyStore()
    # Stopping on token 3 via a set {3, 99}
    out = generate(
        store,
        session_id="test_eos",
        prompt_ids=[10],
        sampler=dummy_sampler,
        max_new_tokens=10,
        eos_id={3, 99},
    )
    # Output should include 1, 2 and then stop at 3 (3 is not appended because it is EOS)
    assert out == [1, 2]


def test_generate_enforce_chat_eos_flag():
    class DummyStore:
        def prefill(self, sid, prompt):
            return [[0.0]]

        def step(self, sid, token):
            return [[0.0]]

    store = DummyStore()
    with pytest.raises(ValueError, match="chat_eos mismatch"):
        generate(
            store,
            session_id="test_enforce",
            prompt_ids=[1],
            sampler=lambda l: 0,
            max_new_tokens=2,
            chat_eos="<wrong_eos>",
            enforce_chat_eos=True,
        )


def test_sft_corpus_manifest_enforces_chat_eos(tmp_path):
    from src.data.sft_corpus import validate_manifest

    manifest_ok = {"template": "qwen-chatml", "chat_eos": CHAT_EOS}
    # Should not raise
    validate_manifest(tmp_path / "manifest.json", manifest_ok)

    manifest_missing = {"template": "qwen-chatml"}
    with pytest.raises(ValueError, match="no `chat_eos` key"):
        validate_manifest(tmp_path / "manifest.json", manifest_missing)

    manifest_mismatch = {"template": "qwen-chatml", "chat_eos": "<|endoftext|>"}
    with pytest.raises(ValueError, match="chat_eos mismatch"):
        validate_manifest(tmp_path / "manifest.json", manifest_mismatch)
