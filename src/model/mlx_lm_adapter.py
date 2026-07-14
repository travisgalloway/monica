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


def _repair_config(model_path: str) -> Optional[dict]:
    """Supply `intermediate_size` for Mamba-2 checkpoints published without it.

    The `mlx-community` Mamba-2 conversions (e.g. `mamba2-130m`) predate `mlx_lm`'s
    current `mamba2.ModelArgs`, which requires `intermediate_size`; loading one raises
    `TypeError: missing 1 required positional argument`. The field is not a free
    parameter — it is `expand * hidden_size` by definition, and Mamba-2 independently
    requires `num_heads * head_dim` to equal the same value.

    So we derive it and then **check the two definitions against each other**, raising if
    they disagree. Silently guessing a load-bearing shape would be the sort of thing that
    produces a plausible model with subtly wrong dimensions — worse than a crash, because
    the SSM arm's numbers would look fine and mean nothing.

    Returns None (no override) for any model that loads normally.
    """
    import json
    from pathlib import Path

    from huggingface_hub import snapshot_download

    local = Path(model_path)
    root = local if local.is_dir() else Path(snapshot_download(model_path))
    cfg_path = root / "config.json"
    if not cfg_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text())

    if cfg.get("model_type") != "mamba2" or "intermediate_size" in cfg:
        return None

    expand, hidden = cfg.get("expand"), cfg.get("hidden_size")
    heads, head_dim = cfg.get("num_heads"), cfg.get("head_dim")
    if not (expand and hidden and heads and head_dim):
        return None

    derived = expand * hidden
    if heads * head_dim != derived:
        raise ValueError(
            f"{model_path}: cannot derive intermediate_size — expand*hidden_size="
            f"{derived} but num_heads*head_dim={heads * head_dim}. Refusing to guess a "
            f"load-bearing shape; supply intermediate_size in config.json explicitly.")

    cfg["intermediate_size"] = derived
    return cfg


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

    def __init__(self, model_path: str, dtype: Optional[str] = None,
                 rollback_strategy: str = "auto"):
        """`rollback_strategy` selects how `rollback()` physically unwinds state:

        - `"auto"` (default): trim if the cache supports it, else re-prefill. This is
          what a real deployment would do.
        - `"trim"`: same as auto (kept as an explicit label for the cost table).
        - `"reprefill"`: force the re-prefill path even on a trimmable transformer cache
          — the control that measures what rollback costs *without* cache support, so the
          SSM's re-prefill number can be compared against a like-for-like transformer one
          rather than against an O(1) trim.
        - `"snapshot"`: the harness drives `checkpoint()`/`restore()` instead of calling
          `rollback()` at all (#202).
        """
        if rollback_strategy not in ("auto", "trim", "reprefill", "snapshot"):
            raise ValueError(f"unknown rollback_strategy: {rollback_strategy!r}")
        import mlx.core as mx
        from mlx.utils import tree_map
        from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache
        from mlx_lm.utils import load

        self._mx = mx
        self._can_trim_prompt_cache = can_trim_prompt_cache
        self._make_prompt_cache = make_prompt_cache
        self._trim_prompt_cache = trim_prompt_cache

        self.model, self.tokenizer = load(model_path, model_config=_repair_config(model_path))
        if dtype is not None:
            target = getattr(mx, dtype)
            self.model.update(tree_map(
                lambda x: x.astype(target) if mx.issubdtype(x.dtype, mx.floating) else x,
                self.model.parameters()))

        self._cache = None
        self._context_ids: List[int] = []  # the immutable prefix set by reset()
        self._gen_ids: List[int] = []      # tokens step()-ped since reset()
        self._last_logits: Optional[np.ndarray] = None
        self._rollback_strategy = rollback_strategy
        self.n_forward_tokens = 0
        self.n_forward_tokens_nocache = 0
        # Which physical rollback path actually ran — the E3 cost table (#202).
        self.n_trim_rollbacks = 0
        self.n_reprefill_rollbacks = 0
        self.n_snapshot_rollbacks = 0
        self.n_reprefill_tokens = 0        # tokens re-forwarded by re-prefill rollbacks

    @property
    def rollback_strategy(self) -> str:
        return self._rollback_strategy

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

        if self._rollback_strategy in ("auto", "trim") and self._can_trim_prompt_cache(self._cache):
            self._trim_prompt_cache(self._cache, n_tokens)
            self._gen_ids = self._gen_ids[:keep]
            self.n_trim_rollbacks += 1
        else:
            kept_gen = self._gen_ids[:keep]
            self._cache = self._make_prompt_cache(self.model)
            self._forward(self._context_ids + kept_gen, count_nocache=False)
            self._gen_ids = kept_gen
            self.n_reprefill_rollbacks += 1
            self.n_reprefill_tokens += len(self._context_ids) + keep

    # --- chat (instruct models) --- #
    def render_chat(self, messages: Sequence[dict]) -> str:
        """Render `messages` through the tokenizer's chat template.

        Raises if the tokenizer has none — a base model has no chat template, and
        silently falling back to raw concatenation would quietly turn the "fair
        instruct opponent" experiment back into the base-model strawman it exists to
        replace.
        """
        tok = getattr(self.tokenizer, "_tokenizer", self.tokenizer)
        if getattr(tok, "chat_template", None) is None:
            raise ValueError(
                f"tokenizer for this model has no chat template — it is a base model, "
                f"not an instruct model. Chat-mode tool-call requires an instruct model.")
        return tok.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True)

    # --- #202: state snapshot / restore --- #
    def checkpoint(self) -> object:
        """Capture the full generation state (cache + token history).

        Why this exists, and why it is the interesting half of the SSM story: an SSM's
        cache (`ArraysCache`) is a fixed-size running summary with no per-token history,
        so `is_trimmable()` is False and rollback would otherwise degrade to a full
        re-prefill. But precisely *because* the state is fixed-size, it can be copied and
        put back — at a cost that does not grow with context length, where a
        transformer's KV cache does.

        `mx.array(c)` copies rather than aliases. That matters: `mamba2.py` rebinds
        (`cache[0] = ...`), so a shallow list copy would *happen* to be safe today, but a
        future in-place `+=` would silently corrupt every snapshot — and a corrupted
        rollback fails silently, poisoning the whole measurement rather than crashing.
        The parity gate in `tests/test_mlx_lm_adapter.py` is what proves this correct.
        """
        state = [
            [None if a is None else self._mx.array(a) for a in layer.state]
            for layer in self._cache
        ]
        return {
            "cache_state": state,
            "context_ids": list(self._context_ids),
            "gen_ids": list(self._gen_ids),
            "logits": None if self._last_logits is None else self._last_logits.copy(),
        }

    def restore(self, handle: object) -> np.ndarray:
        """Restore state captured by `checkpoint()`; return next-token logits.

        Rebuilds fresh cache objects and assigns the snapshotted arrays into them, so one
        handle can be restored repeatedly (a checkpoint stack replays the same frame
        across several retries) without a later restore inheriting a mutation from an
        earlier one.

        The next-token logits ride along in the handle rather than being recomputed: the
        restored cache *already* encodes every token through `gen_ids[-1]`, so replaying
        that token would double-process it and silently desync the state. Cache
        correctness is still fully under test — the very next `step()` runs through the
        restored cache, and that is what the parity gate compares against a re-prefill.
        """
        self._context_ids = list(handle["context_ids"])
        self._gen_ids = list(handle["gen_ids"])
        fresh = self._make_prompt_cache(self.model)
        for layer, saved in zip(fresh, handle["cache_state"]):
            layer.state = [None if a is None else self._mx.array(a) for a in saved]
        self._cache = fresh
        self._last_logits = handle["logits"]
        self.n_snapshot_rollbacks += 1
        if self._last_logits is None:
            raise ValueError("cannot restore a checkpoint taken before any forward pass")
        return self._last_logits.copy()

    def snapshot_bytes(self) -> int:
        """Bytes in one checkpoint handle — the memory price of a checkpoint stack.

        Constant in context length for an SSM (~138 MB for Mamba-Codestral-7B, ~9.7 MB
        for mamba2-130m); linear in context for a transformer's KV cache.
        """
        if self._cache is None:
            return 0
        return sum(
            a.nbytes for layer in self._cache for a in layer.state if a is not None
        )

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
        out = np.array(last.astype(self._mx.float32))
        self._last_logits = out
        return out
