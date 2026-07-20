"""Unit tests for the shared code BPE trainer + loader (#191)."""

import pytest

tokenizers = pytest.importorskip("tokenizers")

import numpy as np

from src.data.tokenizer_train import (
    train_code_bpe, save_tokenizer, SPECIAL_TOKENS, DEFAULT_VOCAB_SIZE)
from src.data.tokenize import (
    load_code_tokenizer, CodeTokenizer, tokenize_texts, _capped)
from src.data.pack import pack_ids, open_packed, packing_dtype_for

SAMPLE = [
    "function add(a: number, b: number): number { return a + b; }",
    "const greet = (name: string): string => `hello ${name}`;",
    "interface Point { x: number; y: number; }",
    "export class Vec { constructor(public x: number, public y: number) {} }",
] * 50   # repeat so BPE has merges to learn


def test_train_vocab_and_specials():
    tok = train_code_bpe(SAMPLE, vocab_size=2000)
    assert tok.get_vocab_size() <= 2000
    for t in SPECIAL_TOKENS:
        assert tok.token_to_id(t) is not None
    assert tok.token_to_id("<mask>") is not None
    assert tok.token_to_id("<|fim_prefix|>") is not None


def test_vocab_cap_16k():
    tok = train_code_bpe(SAMPLE)  # default vocab_size
    assert tok.get_vocab_size() <= DEFAULT_VOCAB_SIZE


def test_roundtrip_encode_decode():
    tok = train_code_bpe(SAMPLE, vocab_size=2000)
    s = "const x = 'π≈3.14'; // 数字"
    ids = tok.encode(s).ids
    assert tok.decode(ids) == s


def test_special_tokens_ids_stable():
    tok = train_code_bpe(SAMPLE, vocab_size=2000)
    assert tok.token_to_id("<|endoftext|>") == 0


def test_save_and_load_code_tokenizer(tmp_path):
    tok = train_code_bpe(SAMPLE, vocab_size=2000)
    save_tokenizer(tok, tmp_path)
    ct = load_code_tokenizer(tmp_path)
    assert isinstance(ct, CodeTokenizer)
    assert ct.vocab_size == tok.get_vocab_size()
    assert ct.eos_token_id == 0
    s = "const x = 'π≈3.14'; // 数字"
    assert ct.decode(ct.encode(s)) == s

    # Also test loading via the explicit tokenizer.json path.
    ct2 = load_code_tokenizer(tmp_path / "tokenizer.json")
    assert ct2.vocab_size == tok.get_vocab_size()


def test_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_code_tokenizer(tmp_path / "nope")


def test_uint16_pack_roundtrip(tmp_path):
    tok = train_code_bpe(SAMPLE, vocab_size=2000)
    save_tokenizer(tok, tmp_path)
    ct = load_code_tokenizer(tmp_path)

    texts = ["function f(x: number) { return x + 1; }", "const y = f(2);"]
    dtype = packing_dtype_for(ct.vocab_size)
    assert dtype == np.uint16

    out = tmp_path / "packed.bin"
    stream = _capped(tokenize_texts(texts, ct), None)
    n = pack_ids(stream, out, dtype=dtype)
    assert n > 0
    packed = np.asarray(open_packed(out))
    assert packed.dtype == np.uint16
    assert packed.size == n
