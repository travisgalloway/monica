"""Tokenize raw text to token-id streams using a configured HF tokenizer.

The packed token dtype follows the tokenizer vocab (`pack.packing_dtype_for`):
- **OLMo** (``allenai/OLMo-7B-hf``, vocab 50280 < 65536) — the original POC tokenizer,
  packs as uint16. (``allenai/OLMo-2-1124-7B`` at 100278 was rejected at POC scale.)
- **Qwen3** (vocab ~151,669) — fixed by the distillation conversion teacher
  (Qwen/Qwen3-4B-Thinking-2507); the unified Qwen3 BPE, token-aligned with Qwen2.5 plus
  a few added control tokens (incl. <think>/</think>). Exceeds uint16, so it packs as
  **uint32** (#90, see docs/design/10-distillation.md). This is the distillation default.
- **Qwen2.5** (vocab 151,646) — the prior teacher's tokenizer; uint32. Kept for back-compat
  (the DeepSeek-R1-Distill-Qwen variants share it).
- **StarCoder2** (vocab ~49,152) — a legacy code-corpus tokenizer (the old uint16 scale
  pick, now superseded by Qwen2.5).

A byte-level fallback tokenizer is provided ONLY for offline pipeline testing; it is not
vocab-compatible with any real tokenizer and must not be used for a real run.
"""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
from typing import Iterable, Iterator, List

# OLMo — the POC tokenizer (uint16-compatible, vocab < 65536).
OLMO_TOKENIZER_CANDIDATES = ("allenai/OLMo-7B-hf",)
# Qwen3 (#65) — fixed by the conversion teacher (Qwen3-4B-Thinking-2507); vocab ~151,669 ->
# uint32 packing. Token-aligned with Qwen2.5 plus the added <think>/</think> control tokens.
QWEN3_TOKENIZER_CANDIDATES = ("Qwen/Qwen3-4B-Thinking-2507", "Qwen/Qwen3-4B")
# Qwen2.5 (#90) — the prior teacher's tokenizer; vocab 151,646 -> uint32. The
# DeepSeek-R1-Distill-Qwen variants share this exact tokenizer.
QWEN25_TOKENIZER_CANDIDATES = ("Qwen/Qwen2.5-1.5B",
                               "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
# StarCoder2 — legacy code-corpus tokenizer (~49,152, uint16); superseded by Qwen2.5.
STARCODER2_TOKENIZER_CANDIDATES = ("bigcode/starcoder2-3b", "bigcode/starcoder2-15b")

#: Largest vocab we can pack (uint32 ceiling). uint16 vs uint32 is chosen per-vocab.
_MAX_PACKABLE_VOCAB = 1 << 32


def _load_hf_tokenizer(candidates, model_id, label, max_vocab: int = _MAX_PACKABLE_VOCAB):
    """Load an HF tokenizer, confirming its vocab fits the packed token dtype (uint32
    ceiling by default; uint16 vs uint32 is then chosen per-vocab by `pack`)."""
    from transformers import AutoTokenizer  # imported lazily

    ids = (model_id,) if model_id else candidates
    last_err = None
    for mid in ids:
        try:
            tok = AutoTokenizer.from_pretrained(mid)
            if tok.vocab_size > max_vocab:   # max id = vocab-1 must fit (uint32: 2**32-1)
                raise ValueError(f"{mid} vocab {tok.vocab_size} too large to pack "
                                 f"(> {max_vocab})")
            return tok
        except Exception as e:  # pragma: no cover - network/availability dependent
            last_err = e
    raise RuntimeError(f"No {label} tokenizer reachable from {ids}: {last_err}")


def load_olmo_tokenizer(model_id: str | None = None):
    """Load the OLMo tokenizer (POC path, uint16). Raises if unavailable."""
    return _load_hf_tokenizer(OLMO_TOKENIZER_CANDIDATES, model_id, "OLMo")


def load_qwen3_tokenizer(model_id: str | None = None):
    """Load the Qwen3 tokenizer — the distillation-student default (vocab ~151,669 ->
    uint32 packing, #65), shared with the Qwen3-4B-Thinking-2507 teacher. Raises if
    unavailable."""
    return _load_hf_tokenizer(QWEN3_TOKENIZER_CANDIDATES, model_id, "Qwen3")


def load_qwen25_tokenizer(model_id: str | None = None):
    """Load the Qwen2.5 tokenizer — the prior teacher's tokenizer (vocab 151,646 ->
    uint32 packing, #90). Kept for back-compat. Raises if unavailable."""
    return _load_hf_tokenizer(QWEN25_TOKENIZER_CANDIDATES, model_id, "Qwen2.5")


def load_starcoder2_tokenizer(model_id: str | None = None):
    """Load the StarCoder2 tokenizer (legacy code-corpus, uint16). Raises if unavailable."""
    return _load_hf_tokenizer(STARCODER2_TOKENIZER_CANDIDATES, model_id, "StarCoder2")


class ByteTokenizer:
    """Offline-only fallback. vocab_size=256, one id per byte."""

    vocab_size = 256

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: Iterable[int]) -> str:
        """Inverse of `encode` (lossy on invalid byte sequences). For offline serving."""
        return bytes(int(i) & 0xFF for i in ids).decode("utf-8", "replace")


def tokenize_texts(texts: Iterable[str], tokenizer) -> Iterable[int]:
    """Yield a flat stream of token ids across all documents (with EOS if available)."""
    eos = getattr(tokenizer, "eos_token_id", None)
    for text in texts:
        for tid in tokenizer.encode(text):
            yield tid
        if eos is not None:
            yield eos


def tokenize_docs(texts: Iterable[str], tokenizer) -> Iterator[List[int]]:
    """Yield one token-id list PER document (EOS appended if available). Unlike
    `tokenize_texts` this preserves document boundaries — `shard.pack_sequences` needs
    them to mark doc-starts so the SSM state can be reset across packed docs (#68/#74)."""
    eos = getattr(tokenizer, "eos_token_id", None)
    for text in texts:
        ids = list(tokenizer.encode(text))
        if eos is not None:
            ids.append(eos)
        if ids:
            yield ids


def _capped(stream: Iterable[int], max_tokens: int | None) -> Iterator[int]:
    """Truncate an id stream at `max_tokens` (pass-through when None)."""
    if max_tokens is None:
        yield from stream
        return
    for n, tid in enumerate(stream):
        if n >= max_tokens:
            break
        yield tid


@contextlib.contextmanager
def _open_texts(inp: Path):
    """Yield an iterator of doc texts from `inp`, transparently handling both inputs:
    a one-doc-per-line text file (the download.py output), or a directory / `.parquet`
    file of corpus Parquet shards (the corpus.py output) — so the cleaned shards feed
    this stage with no extra step. Parquet text is already normalized one-line-per-doc."""
    if inp.is_dir() or inp.suffix == ".parquet":
        from .corpus import iter_shard_texts
        yield iter_shard_texts(inp)
    else:
        # Explicit UTF-8 (corpus is UTF-8; avoids locale-dependent decoding) and strip
        # any trailing \r so \r\n-terminated input doesn't leak a carriage return.
        with open(inp, encoding="utf-8") as f:
            yield (line.rstrip("\r\n") for line in f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="uint16/uint32 ids; .bin streams "
                    "straight into the packed format, .npy goes through `pack` separately")
    ap.add_argument("--byte-fallback", action="store_true", help="offline testing only")
    ap.add_argument("--tokenizer", choices=("olmo", "qwen3", "qwen25", "starcoder2"),
                    default="olmo",
                    help="HF tokenizer; the packed dtype is uint16/uint32 per its vocab")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="stop after this many tokens (caps the scale run)")
    args = ap.parse_args()
    if args.max_tokens is not None and args.max_tokens <= 0:
        ap.error("--max-tokens must be positive (a value <= 0 silently yields 0 tokens)")

    from .pack import packing_dtype_for
    _loaders = {"olmo": load_olmo_tokenizer, "qwen3": load_qwen3_tokenizer,
                "qwen25": load_qwen25_tokenizer, "starcoder2": load_starcoder2_tokenizer}
    tok = ByteTokenizer() if args.byte_fallback else _loaders[args.tokenizer](args.model_id)
    dtype = packing_dtype_for(tok.vocab_size)   # uint16 (POC) / uint32 (Qwen3)
    # Input is either a one-doc-per-line text file or corpus Parquet shards (dir/.parquet).
    with _open_texts(args.inp) as texts:
        stream = _capped(tokenize_texts(texts, tok), args.max_tokens)
        if args.out.suffix == ".bin":
            # Stream straight into the packed format (chunked, bounded memory) — this
            # folds the `pack` stage in for the scale run and writes the .meta.json
            # sidecar, so `split` can consume the output directly.
            from .pack import pack_ids

            n = pack_ids(stream, args.out, dtype=dtype)
            print(f"tokenized+packed {n} ids ({dtype.name}, vocab {tok.vocab_size}) -> {args.out}")
        else:
            import numpy as np

            ids = np.fromiter(stream, dtype=dtype)
            np.save(args.out, ids)
            print(f"tokenized {ids.size} ids ({dtype.name}, vocab {tok.vocab_size}) -> {args.out}")


if __name__ == "__main__":
    main()
