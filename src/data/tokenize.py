"""Tokenize raw text to token-id streams using the OLMo tokenizer.

Use the OLMo tokenizer (via HuggingFace) so the vocab matches AI2's, enabling
later comparison. VERIFY availability first — do not assume the exact tokenizer is
one import away; the model id may need adjusting.

A byte-level fallback tokenizer is provided ONLY for offline pipeline testing; it
is not vocab-compatible with OLMo and must not be used for a real run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

# Candidate OLMo tokenizer ids on the HF Hub. Confirm one is reachable before a run.
OLMO_TOKENIZER_CANDIDATES = ("allenai/OLMo-2-1124-7B", "allenai/OLMo-7B-hf")


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


def tokenize_texts(texts: Iterable[str], tokenizer) -> Iterable[int]:
    """Yield a flat stream of token ids across all documents (with EOS if available)."""
    eos = getattr(tokenizer, "eos_token_id", None)
    for text in texts:
        for tid in tokenizer.encode(text):
            yield tid
        if eos is not None:
            yield eos


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="raw .npy/.bin uint16 ids")
    ap.add_argument("--byte-fallback", action="store_true", help="offline testing only")
    ap.add_argument("--model-id", default=None)
    args = ap.parse_args()

    import numpy as np

    tok = ByteTokenizer() if args.byte_fallback else load_olmo_tokenizer(args.model_id)
    with open(args.inp) as f:
        texts = (line.rstrip("\n") for line in f)
        ids = np.fromiter(tokenize_texts(texts, tok), dtype=np.uint16)
    np.save(args.out, ids)
    print(f"tokenized {ids.size} ids (vocab {tok.vocab_size}) -> {args.out}")


if __name__ == "__main__":
    main()
