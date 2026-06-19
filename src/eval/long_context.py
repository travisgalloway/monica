"""Long-sequence evaluation harness (#54 / #65 Phase 4 — usable context length).

Reports held-out perplexity as a function of sequence length, so the training-free
long-context knob (`MambaConfig.long_ctx_factor`, applied in the MLX backend's
`SelectiveSSM`) can be measured: a model trained at `seq_len` is evaluated at 1x / 2x /
4x that length, with the knob off (baseline degradation) and on (recovery).

PORTABLE: builds on `PackedLoader` + `val_loss.evaluate` and never imports a backend
(the knob lives on the model's config; this harness only chooses the eval length). An
SSM reads arbitrary lengths with no positional limit, so evaluating at >`seq_len` just
means packing longer chunks — `PackedLoader(seq_len=mult*base)` does exactly that.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

from ..data.loader import PackedLoader
from ..eval.val_loss import evaluate
from ..model.interface import ModelInterface


def long_context_eval(
    model: ModelInterface, val_packed_path, base_seq_len: int, batch_size: int,
    mults: Iterable[int] = (1, 2, 4), max_batches: Optional[int] = None,
    to_numpy=np.asarray,
) -> Dict[int, Optional[dict]]:
    """Evaluate `model` at each length `mult * base_seq_len`.

    Returns `{mult: {val_loss, val_perplexity, seq_len, n_batches} | None}`; `None`
    when the val file holds fewer than one chunk at that length (recorded, not raised —
    no silent drop). The knob is whatever `model.config.long_ctx_factor` already is, so
    call this twice (off vs on) to get both curves.
    """
    val_packed_path = Path(val_packed_path)
    results: Dict[int, Optional[dict]] = {}
    for mult in mults:
        seq_len = int(base_seq_len * mult)
        try:
            loader = PackedLoader(val_packed_path, seq_len, batch_size,
                                  shuffle=False, drop_last=False)
        except ValueError:
            results[mult] = None          # file too small for one chunk at this length
            continue
        res = evaluate(model, loader, max_batches=max_batches, to_numpy=to_numpy)
        res["seq_len"] = seq_len
        res["n_batches"] = min(len(loader), max_batches) if max_batches else len(loader)
        results[mult] = res
    return results


def format_curve(label: str, results: Dict[int, Optional[dict]]) -> str:
    """One-line-per-length perplexity-vs-length table for logging / posting to #65."""
    lines = [f"[{label}] perplexity vs sequence length:"]
    for mult, res in sorted(results.items()):
        if res is None:
            lines.append(f"  {mult}x  (skipped — val split too small for one chunk)")
        else:
            lines.append(f"  {mult}x  seq_len={res['seq_len']:>6}  "
                         f"val_loss={res['val_loss']:.4f}  "
                         f"perplexity={res['val_perplexity']:.4f}  "
                         f"({res['n_batches']} batches)")
    return "\n".join(lines)
