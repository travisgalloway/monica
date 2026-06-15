"""Retrieval probe (#67): does the attention fraction actually help recall?

Trains a **pure-Mamba** and a **hybrid** (same dims, a few attention layers) on the
multi-query associative-recall (MQAR) task from `src.eval.retrieval_probe`, with a
FIXED seed, and reports each model's recall accuracy. Pure SSMs have a fixed-width
state, so their recall degrades as the number of key-value pairs grows; the hybrid's
attention layers do not. Run at enough pairs, the hybrid wins — that gap is the
evidence the config-gated attention earns its keep.

  python scripts/retrieval_probe.py                      # default probe
  python scripts/retrieval_probe.py --n-pairs 64 --steps 2000
  python scripts/retrieval_probe.py --include-attn       # also a pure-attention upper bound

Runs on whatever backend is present (MLX on the Mac); SFT-style masked-CE training is
MLX-only today, so this is an Apple-Silicon probe (mirrors scripts/smoke_test.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.model.blocks import MambaConfig
from src.model.backend import get_backend
from src.eval.retrieval_probe import make_recall_batch, recall_accuracy, seq_len, vocab_size


def _make_cfg(d_model, n_layers, n_keys, n_values, slen, attn_every, n_attn_heads):
    cfg = MambaConfig(
        d_model=d_model, n_layers=n_layers, head_dim=16, d_state=16,
        vocab_size=vocab_size(n_keys, n_values), seq_len=slen, precision="fp32",
        attn_every=attn_every, n_attn_heads=n_attn_heads)
    cfg.validate()
    return cfg


def train_and_eval(backend, cfg, *, steps, batch_size, n_pairs, n_keys, n_values,
                   n_queries, lr, seed) -> float:
    """Train one model on MQAR and return its held-out recall accuracy."""
    backend.seed(seed)
    model = backend.model_cls(cfg)
    opt = backend.make_optimizer(model, lr)
    train_step = backend.make_sft_train_step(model, opt)        # masked CE over query positions
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        batch = make_recall_batch(rng, batch_size, n_pairs, n_keys, n_values, n_queries)
        train_step(model, [batch], lr)                          # batch is (inputs, targets, mask)
    ev = np.random.default_rng(seed + 999)
    inputs, targets, mask = make_recall_batch(ev, 512, n_pairs, n_keys, n_values, n_queries)
    logits = backend.to_numpy(model.forward(inputs))
    return recall_accuracy(logits, targets, mask)


def run_probe(*, steps=800, batch_size=64, n_pairs=16, n_keys=32, n_values=24,
              n_queries=None, d_model=64, n_layers=4, attn_every=2, n_attn_heads=4,
              lr=2e-3, seed=0, backend_name="auto", include_attn=False) -> dict:
    """Train pure-Mamba vs hybrid (optionally a pure-attention upper bound) on MQAR."""
    backend = get_backend(backend_name)
    n_queries = n_pairs if n_queries is None else n_queries
    slen = seq_len(n_pairs, n_queries)
    common = dict(steps=steps, batch_size=batch_size, n_pairs=n_pairs, n_keys=n_keys,
                  n_values=n_values, n_queries=n_queries, lr=lr, seed=seed)
    variants = {
        "mamba": _make_cfg(d_model, n_layers, n_keys, n_values, slen, None, n_attn_heads),
        "hybrid": _make_cfg(d_model, n_layers, n_keys, n_values, slen, attn_every, n_attn_heads),
    }
    if include_attn:
        variants["attn"] = _make_cfg(d_model, n_layers, n_keys, n_values, slen, 1, n_attn_heads)
    out = {f"{name}_acc": train_and_eval(backend, cfg, **common)
           for name, cfg in variants.items()}
    out.update(n_pairs=n_pairs, n_keys=n_keys, n_values=n_values, chance=1.0 / n_values)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-pairs", type=int, default=16)
    ap.add_argument("--n-keys", type=int, default=32)
    ap.add_argument("--n-values", type=int, default=24)
    ap.add_argument("--n-queries", type=int, default=None)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-attn", action="store_true",
                    help="also train a pure-attention model (recall upper bound)")
    args = ap.parse_args()

    r = run_probe(steps=args.steps, batch_size=args.batch_size, n_pairs=args.n_pairs,
                  n_keys=args.n_keys, n_values=args.n_values, n_queries=args.n_queries,
                  lr=args.lr, seed=args.seed, include_attn=args.include_attn)
    print(f"MQAR ({r['n_pairs']} pairs, {r['n_keys']} keys, {r['n_values']} values; "
          f"chance={r['chance']:.3f})")
    print(f"  pure-Mamba accuracy : {r['mamba_acc']:.3f}")
    if "attn_acc" in r:
        print(f"  pure-attn  accuracy : {r['attn_acc']:.3f}")
    print(f"  hybrid     accuracy : {r['hybrid_acc']:.3f}")
    print(f"  delta (hybrid-pure) : {r['hybrid_acc'] - r['mamba_acc']:+.3f}")


if __name__ == "__main__":
    main()
