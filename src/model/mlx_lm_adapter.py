"""MLX-LM-backed `LMAdapter` (#199) — the model side of the LSP-in-the-loop harness.

BELOW THE SEAM. Imports `mlx`/`mlx_lm` inside `__init__`, mirroring
`scripts/smoke_test.py`'s local-import idiom, so this module is importable without
the packages present; it is deliberately **not** added to
`tests/test_import_guard.py::PORTABLE_MODULES`.

Rollback prefers `mlx_lm.models.cache.trim_prompt_cache` (exact, O(1) — just moves
the cache's write offset back) and falls back to a full re-prefill from the retained
token history when the cache type can't trim (`can_trim_prompt_cache` is False, e.g.
a `RotatingKVCache` model past its window — Qwen2.5-Coder's plain `KVCache` always
can). Both paths are accounted into `n_forward_tokens_nocache` identically
(`len(context_ids) + n_tokens_kept`, the cost of a hypothetical no-cache
implementation) regardless of which one actually ran, so the harness's cost table
compares the *algorithm* rather than which cache type a given model happens to use.
See `src/lsp/lm.py::LMAdapter` for the counter semantics.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


class MLXLMAdapter:
    """Implements `src.lsp.lm.LMAdapter` on `mlx_lm`.

    `dtype`, if given (e.g. `"float32"`), upcasts every floating-point parameter
    after load — the same "compare in fp32" idiom `CLAUDE.md` documents for
    `src/conformance/`'s parity tests. bf16's epsilon (~0.4% relative) is too
    coarse for a rollback-exactness gate: a batched prefill and an equivalent
    sequence of single-token steps are mathematically identical for a causal
    transformer, but bf16 accumulates rounding differently across the two
    (different matmul chunking) — confirmed empirically to a ~1e-5 max-abs-diff
    gap in fp32 vs. ~0.2 in bf16 at logit magnitude ~13. Production measurement
    runs leave `dtype=None` (the model's native bf16, for real inference speed);
    `test_mlx_lm_adapter.py`'s parity gate passes `dtype="float32"`.
    """

    def __init__(self, model_path: str, dtype: Optional[str] = None):
        import mlx.core as mx
        from mlx.utils import tree_map
        from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache
        from mlx_lm.utils import load

        self._mx = mx
        self._can_trim_prompt_cache = can_trim_prompt_cache
        self._make_prompt_cache = make_prompt_cache
        self._trim_prompt_cache = trim_prompt_cache

        self.model, self.tokenizer = load(model_path)
        if dtype is not None:
            target = getattr(mx, dtype)
            self.model.update(tree_map(
                lambda x: x.astype(target) if mx.issubdtype(x.dtype, mx.floating) else x,
                self.model.parameters()))

        self._cache = None
        self._context_ids: List[int] = []  # the immutable prefix set by reset()
        self._gen_ids: List[int] = []      # tokens step()-ped since reset()
        self.n_forward_tokens = 0
        self.n_forward_tokens_nocache = 0

    # --- tokenization ---
    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(token_ids))

    # --- stepwise state ---
    def reset(self, context: str) -> np.ndarray:
        self._context_ids = self.encode(context)
        self._cache = self._make_prompt_cache(self.model)
        self._gen_ids = []
        return self._forward(self._context_ids)

    def step(self, token_id: int) -> np.ndarray:
        logits = self._forward([token_id])
        self._gen_ids.append(token_id)
        return logits

    def rollback(self, n_tokens: int) -> None:
        if n_tokens <= 0:
            return
        keep = len(self._gen_ids) - n_tokens
        if keep < 0:
            raise ValueError(f"cannot roll back {n_tokens} tokens; only "
                              f"{len(self._gen_ids)} generated since reset()")

        # Uniform nocache accounting regardless of which physical path below runs:
        # a no-cache implementation always pays a full re-prefill of context + kept.
        self.n_forward_tokens_nocache += len(self._context_ids) + keep

        if self._can_trim_prompt_cache(self._cache):
            self._trim_prompt_cache(self._cache, n_tokens)
            self._gen_ids = self._gen_ids[:keep]
        else:
            kept_gen = self._gen_ids[:keep]
            self._cache = self._make_prompt_cache(self.model)
            self._forward(self._context_ids + kept_gen, count_nocache=False)
            self._gen_ids = kept_gen

    # --- internal ---
    def _forward(self, ids: Sequence[int], *, count_nocache: bool = True) -> np.ndarray:
        arr = self._mx.array(ids)[None]
        logits = self.model(arr, cache=self._cache)
        self._mx.eval(logits)
        n = len(ids)
        self.n_forward_tokens += n
        if count_nocache:
            self.n_forward_tokens_nocache += n
        last = logits[0, -1, :]
        return np.array(last.astype(self._mx.float32))
