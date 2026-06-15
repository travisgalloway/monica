"""Copying / retrieval / long-context probes (#79): is the attention fraction high enough?

Trains a small model on each probe (needle-in-a-haystack, phonebook exact-copy, 5-shot
recall) with a FIXED seed and reports accuracy; with `--compare` it trains a pure-Mamba and
a hybrid side by side, so the gap shows whether the config-gated attention earns its keep.
Pure SSMs lag on copying/retrieval and the gap widens with context length — raise the
attention fraction if the needle curve sags, lower it for speed. First run at the 100M gate
(#81), re-run on each Phase-5 tier (#75).

  python scripts/probes.py                       # hybrid, all three probes
  python scripts/probes.py --compare             # pure-Mamba vs hybrid
  python scripts/probes.py --probe needle --needle-lengths 64 256 1024 --steps 1500

MLX-only training (masked-CE), like scripts/retrieval_probe.py — an Apple-Silicon probe.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.model.blocks import MambaConfig
from src.model.backend import get_backend
from src.eval.probes import (fewshot_vocab, make_fewshot_copy_batch, make_needle_batch,
                            make_phonebook_batch, needle_vocab, phonebook_vocab,
                            probe_accuracy)

# (name, batch-generator, vocab, seq_len) factories — each keyed by a "size" knob.
def _needle(rng, bs, size):
    return make_needle_batch(rng, bs, size), needle_vocab(), size


def _phonebook(rng, bs, size):
    x, t, m = make_phonebook_batch(rng, bs, size)
    return (x, t, m), phonebook_vocab(), x.shape[1]


def _fewshot(rng, bs, size):
    x, t, m = make_fewshot_copy_batch(rng, bs, size)
    return (x, t, m), fewshot_vocab(), x.shape[1]


PROBES = {"needle": _needle, "phonebook": _phonebook, "fewshot": _fewshot}


def _cfg(vocab, slen, *, hybrid):
    cfg = MambaConfig(
        d_model=64, n_layers=4, head_dim=16, d_state=16, vocab_size=vocab,
        seq_len=slen, precision="fp32",
        attn_every=2 if hybrid else None, n_attn_heads=4 if hybrid else None)
    cfg.validate()
    return cfg


def train_eval(backend, gen, size, *, hybrid, steps, batch_size, lr, seed) -> float:
    """Train one model on a probe at the given size, return held-out accuracy."""
    rng = np.random.default_rng(seed)
    (_, _, _), vocab, slen = gen(rng, 1, size)        # peek vocab/seq_len for the config
    backend.seed(seed)
    model = backend.model_cls(_cfg(vocab, slen, hybrid=hybrid))
    opt = backend.make_optimizer(model, lr)
    train_step = backend.make_sft_train_step(model, opt)
    for _ in range(steps):
        batch, _, _ = gen(rng, batch_size, size)
        train_step(model, [batch], lr)
    ev = np.random.default_rng(seed + 999)
    (inputs, targets, mask), _, _ = gen(ev, 512, size)
    logits = backend.to_numpy(model.forward(inputs))
    return probe_accuracy(logits, targets, mask)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", choices=(*PROBES, "all"), default="all")
    ap.add_argument("--compare", action="store_true", help="pure-Mamba vs hybrid")
    ap.add_argument("--needle-lengths", type=int, nargs="+", default=[64, 256])
    ap.add_argument("--phonebook-entries", type=int, nargs="+", default=[16, 64])
    ap.add_argument("--fewshot-shots", type=int, nargs="+", default=[5])
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    backend = get_backend()
    sizes = {"needle": args.needle_lengths, "phonebook": args.phonebook_entries,
             "fewshot": args.fewshot_shots}
    probes = list(PROBES) if args.probe == "all" else [args.probe]
    archs = [("pure", False), ("hybrid", True)] if args.compare else [("hybrid", True)]

    print(f"# probe accuracy (steps={args.steps}, seed={args.seed})")
    for name in probes:
        for size in sizes[name]:
            cells = []
            for label, hybrid in archs:
                acc = train_eval(backend, PROBES[name], size, hybrid=hybrid,
                                 steps=args.steps, batch_size=args.batch_size,
                                 lr=args.lr, seed=args.seed)
                cells.append(f"{label}={acc:.3f}")
            print(f"{name:10s} size={size:<6d} " + "  ".join(cells))


if __name__ == "__main__":
    main()
