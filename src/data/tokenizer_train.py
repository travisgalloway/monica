"""Train the shared multilingual code BPE (#191, M11 foundation for #188).

A single shared byte-level BPE trained over the code corpus keeps every future
Branch-Train-Mix language branch in a compatible vocabulary (per-language BPEs can't
share embeddings/residual stream). ~16k vocab is compute-optimal for a ~1B/100M model
(Tao et al., NeurIPS 2024) and packs as **uint16** (`pack.packing_dtype_for`), a leaner
tied embedding than the retired Qwen3 distillation path's 151,669.

This module builds the trainer + CLI; the real training run over the final corpus
mixture is #193-gated. Here the artifact is only exercised on a sample corpus, so the
special-token list can still evolve before the real run.

`tokenizers` is imported LAZILY inside functions (never at module top-level) so this
module stays portable — it imports cleanly above the seam even where the `data` extra
is absent, matching the other guarded `src/data/*` modules (see
`tests/test_import_guard.py`).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Iterator

# Reserved up front (ids 0..N-1) so the SAME tokenizer serves the AR arm (FIM, #215) and
# the diffusion arm (<mask>, M13 WS5b) with no retrain. Standard StarCoder/OpenAI FIM set.
EOS_TOKEN = "<|endoftext|>"          # document separator / EOS -> wrapper.eos_token_id
MASK_TOKEN = "<mask>"                # diffusion arm (M13 WS5b)
SPECIAL_TOKENS = [
    EOS_TOKEN,
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|fim_pad|>",
    MASK_TOKEN,
]
DEFAULT_VOCAB_SIZE = 16384

_SOURCE_SUFFIXES = (".ts", ".tsx", ".js", ".jsx", ".py", ".txt", ".md")


def train_code_bpe(texts: Iterable[str], vocab_size: int = DEFAULT_VOCAB_SIZE,
                    special_tokens=SPECIAL_TOKENS):
    """Train a byte-level BPE over `texts` (iterable of str). Returns a tokenizers.Tokenizer.

    Byte-level + full 256-byte initial alphabet + unk_token=None => lossless round-trip on
    any UTF-8 input (no <unk>)."""
    from tokenizers import Tokenizer                       # lazy (seam/guard)
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder

    tok = Tokenizer(BPE(unk_token=None))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(special_tokens),
        initial_alphabet=ByteLevel.alphabet(),   # all 256 bytes -> no <unk>, full round-trip
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def save_tokenizer(tok, out_path) -> Path:
    """Write the tokenizer to a single JSON file. Accepts a dir (-> out/tokenizer.json) or a
    .json path. Returns the file path written."""
    out = Path(out_path)
    if out.suffix != ".json":
        out.mkdir(parents=True, exist_ok=True)
        out = out / "tokenizer.json"
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out))
    return out


def iter_corpus_texts(inp) -> Iterator[str]:
    """Yield document texts from `inp`: a directory (each source file = one doc) or a single
    file (whole-file = one doc). Offline, stdlib-only."""
    p = Path(inp)
    if p.is_dir():
        for f in sorted(p.rglob("*")):
            if f.is_file() and f.suffix in _SOURCE_SUFFIXES:
                yield f.read_text(encoding="utf-8", errors="replace")
    else:
        yield p.read_text(encoding="utf-8", errors="replace")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, required=True,
                    help="a directory of source files, or a single file")
    ap.add_argument("--out", type=Path, required=True,
                    help="output dir (-> out/tokenizer.json) or a .json path")
    ap.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    args = ap.parse_args()

    tok = train_code_bpe(iter_corpus_texts(args.inp), vocab_size=args.vocab_size)
    out = save_tokenizer(tok, args.out)
    print(f"trained {tok.get_vocab_size()} BPE ({len(SPECIAL_TOKENS)} special) -> {out}")


if __name__ == "__main__":
    main()
