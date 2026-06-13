"""Tokenize raw text to token-id streams using the OLMo tokenizer.

Use the OLMo tokenizer (via HuggingFace) so the vocab matches AI2's, enabling
later comparison.

CONFIRMED (issue #4): ``allenai/OLMo-7B-hf`` is reachable on the HF Hub with
vocab_size=50280 (eos_token_id=50279), which fits the uint16 packing requirement
(< 65536). ``allenai/OLMo-2-1124-7B`` is deliberately NOT a candidate: its vocab is
100278 (> 65536) and can never satisfy the uint16 constraint enforced by
``MambaConfig.validate()``.

A byte-level fallback tokenizer is provided ONLY for offline pipeline testing; it
is not vocab-compatible with OLMo and must not be used for a real run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Iterator, List

# Confirmed OLMo tokenizer ids on the HF Hub (uint16-compatible, vocab < 65536).
OLMO_TOKENIZER_CANDIDATES = ("allenai/OLMo-7B-hf",)


def load_olmo_tokenizer(model_id: str | None = None):
    """Load the OLMo tokenizer via transformers. Raises if unavailable."""
    from transformers import AutoTokenizer  # imported lazily

    ids = (model_id,) if model_id else OLMO_TOKENIZER_CANDIDATES
    last_err = None
    for mid in ids:
        try:
            tok = AutoTokenizer.from_pretrained(mid)
            # Confirm the vocab fits uint16 packing before committing to it.
            if tok.vocab_size >= 65536:
                raise ValueError(f"{mid} vocab {tok.vocab_size} too large for uint16")
            return tok
        except Exception as e:  # pragma: no cover - network/availability dependent
            last_err = e
    raise RuntimeError(f"No OLMo tokenizer reachable from {ids}: {last_err}")


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


def _capped(stream: Iterable[int], max_tokens: int | None) -> Iterator[int]:
    """Truncate an id stream at `max_tokens` (pass-through when None)."""
    if max_tokens is None:
        yield from stream
        return
    for n, tid in enumerate(stream):
        if n >= max_tokens:
            break
        yield tid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="uint16 ids; .bin streams "
                    "straight into the packed format, .npy goes through `pack` separately")
    ap.add_argument("--byte-fallback", action="store_true", help="offline testing only")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="stop after this many tokens (caps the scale run)")
    args = ap.parse_args()
    if args.max_tokens is not None and args.max_tokens <= 0:
        ap.error("--max-tokens must be positive (a value <= 0 silently yields 0 tokens)")

    tok = ByteTokenizer() if args.byte_fallback else load_olmo_tokenizer(args.model_id)
    # Explicit UTF-8 (corpus is UTF-8; avoids locale-dependent decoding) and strip any
    # trailing \r so \r\n-terminated input doesn't leak a stray carriage return into a doc.
    with open(args.inp, encoding="utf-8") as f:
        texts = (line.rstrip("\r\n") for line in f)
        stream = _capped(tokenize_texts(texts, tok), args.max_tokens)
        if args.out.suffix == ".bin":
            # Stream straight into the packed format (chunked, bounded memory) — this
            # folds the `pack` stage in for the scale run and writes the .meta.json
            # sidecar, so `split` can consume the output directly.
            from .pack import pack_ids

            n = pack_ids(stream, args.out)
            print(f"tokenized+packed {n} ids (vocab {tok.vocab_size}) -> {args.out}")
        else:
            import numpy as np

            ids = np.fromiter(stream, dtype=np.uint16)
            np.save(args.out, ids)
            print(f"tokenized {ids.size} ids (vocab {tok.vocab_size}) -> {args.out}")


if __name__ == "__main__":
    main()
