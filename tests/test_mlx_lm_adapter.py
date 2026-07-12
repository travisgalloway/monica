"""Rollback-exactness gate for `src/model/mlx_lm_adapter.py` (#199).

**The gate that matters**: if cache trimming is inexact, every rollback in the
harness silently corrupts the run and the entire measurement table is garbage. This
is not a nice-to-have correctness test — it is the thing that makes the rest of the
measurement trustworthy.

Two things this file gets right that a naive version wouldn't:

1. **fp32, not the model's native bf16.** A batched prefill and an equivalent
   sequence of single-token steps are mathematically identical for a causal
   transformer, but bf16 accumulates rounding differently across the two paths
   (different matmul chunking) — confirmed empirically as a ~0.2 max-abs-diff gap
   at logit magnitude ~13 in bf16, vs. ~1e-5 in fp32. Same "compare in fp32"
   idiom `CLAUDE.md` documents for `src/conformance/`'s parity tests, applied here
   via `MLXLMAdapter(..., dtype="float32")`.
2. **Fixtures split one real encoding's token ids, not two independently-encoded
   strings.** Byte-level BPE is not prefix-stable under independent re-encoding —
   e.g. `encode("...u.") + encode("name);\\n")` can merge differently than
   `encode("...u.name);\\n")` at the boundary (confirmed empirically against this
   tokenizer). `decode(full_ids[:k])` re-encoding back to exactly `full_ids[:k]` is
   a genuine token-boundary round trip and doesn't have this problem.

Skips entirely if `mlx_lm` isn't installed or the pinned test model can't be reached
(no network / not yet downloaded) — this repo's harness falls back to
`hf_lm_adapter.py` in that case (see the design doc), and that fallback gets its own
gate when/if it exists.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

mlx_lm = pytest.importorskip("mlx_lm", reason="mlx_lm not installed")

from src.model.mlx_lm_adapter import MLXLMAdapter  # noqa: E402

MODEL_PATH = os.environ.get("LSP_TEST_MODEL", "mlx-community/Qwen2.5-Coder-0.5B-bf16")


def _model_available() -> bool:
    try:
        from mlx_lm.utils import hf_repo_to_path
        hf_repo_to_path(MODEL_PATH)  # local_files_only=True — no network fetch
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _model_available(),
                                 reason=f"test model {MODEL_PATH!r} not locally cached / reachable")

_FULL_TEXT = ('interface User { name: string; age: number; }\n'
              'const u: User = { name: "Ada", age: 32 };\n'
              'console.log(u.name);\n')


def _new_adapter() -> MLXLMAdapter:
    return MLXLMAdapter(MODEL_PATH, dtype="float32")


@pytest.fixture(scope="module")
def adapter() -> MLXLMAdapter:
    return _new_adapter()


def _token_split(adapter: MLXLMAdapter, text: str, k: int):
    """Split `text`'s encoding at token index `k`: return `(a_text, b_ids, full_ids)`
    where `a_text = decode(full_ids[:k])` is guaranteed to re-encode to exactly
    `full_ids[:k]` (a real token-boundary round trip) and `b_ids = full_ids[k:]`.
    """
    full_ids = adapter.encode(text)
    assert 0 < k < len(full_ids), f"split k={k} out of range for {len(full_ids)} tokens"
    a_text = adapter.decode(full_ids[:k])
    assert adapter.encode(a_text) == full_ids[:k], \
        "chosen split point is not a clean token-boundary round trip"
    return a_text, full_ids[k:], full_ids


def test_reset_ab_matches_reset_a_then_step_through_b(adapter: MLXLMAdapter):
    a_text, b_ids, full_ids = _token_split(adapter, _FULL_TEXT, k=20)

    logits_direct = adapter.reset(_FULL_TEXT)

    logits_stepped = adapter.reset(a_text)
    for tok in b_ids:
        logits_stepped = adapter.step(tok)

    assert np.allclose(logits_direct, logits_stepped, rtol=0, atol=1e-4), \
        f"max abs diff = {np.max(np.abs(logits_direct - logits_stepped))}"


def test_rollback_then_restep_reproduces_logits(adapter: MLXLMAdapter):
    a_text, b_ids, full_ids = _token_split(adapter, _FULL_TEXT, k=20)

    # Reference: reset(a), step through b once — this is what "generating b" means.
    adapter.reset(a_text)
    logits_direct = None
    for tok in b_ids:
        logits_direct = adapter.step(tok)

    # reset(a), step through b, roll ALL of it back, then re-step through b again —
    # must reproduce exactly the same final logits as the reference above.
    adapter.reset(a_text)
    for tok in b_ids:
        adapter.step(tok)
    adapter.rollback(len(b_ids))
    logits_rolled = None
    for tok in b_ids:
        logits_rolled = adapter.step(tok)

    assert np.allclose(logits_direct, logits_rolled, rtol=0, atol=1e-4), \
        f"max abs diff = {np.max(np.abs(logits_direct - logits_rolled))}"


def test_partial_rollback_reproduces_intermediate_logits(adapter: MLXLMAdapter):
    a_text, b_ids, full_ids = _token_split(adapter, _FULL_TEXT, k=15)
    assert len(b_ids) >= 4, "need enough tokens for a meaningful partial rollback"
    split = len(b_ids) // 2
    probe_tok = b_ids[split]

    # Reference: reset(a), step through the first `split` tokens of b, then the probe.
    ref = _new_adapter()
    ref.reset(a_text)
    for tok in b_ids[:split]:
        ref.step(tok)
    logits_ref_probe = ref.step(probe_tok)

    # reset(a), step through ALL of b, roll back past the probe, then re-step the probe.
    adapter.reset(a_text)
    for tok in b_ids:
        adapter.step(tok)
    adapter.rollback(len(b_ids) - split)
    logits_after_rollback = adapter.step(probe_tok)

    assert np.allclose(logits_after_rollback, logits_ref_probe, rtol=0, atol=1e-4), \
        f"max abs diff = {np.max(np.abs(logits_after_rollback - logits_ref_probe))}"


def test_rollback_updates_nocache_counter_uniformly(adapter: MLXLMAdapter):
    a_text, b_ids, full_ids = _token_split(adapter, _FULL_TEXT, k=20)
    context_len = len(adapter.encode(a_text))

    adapter.reset(a_text)
    for tok in b_ids:
        adapter.step(tok)

    before = adapter.n_forward_tokens_nocache
    adapter.rollback(len(b_ids))
    after = adapter.n_forward_tokens_nocache

    assert after - before == context_len  # keep == 0 here: only context re-prefill cost


def test_rollback_rejects_more_tokens_than_generated(adapter: MLXLMAdapter):
    adapter.reset("const x = ")
    adapter.step(adapter.encode("1")[0])
    with pytest.raises(ValueError):
        adapter.rollback(5)


def test_rollback_zero_is_a_noop():
    a = _new_adapter()
    a.reset("const x = ")
    tok = a.encode("1")[0]
    a.step(tok)
    n_before = a.n_forward_tokens
    a.rollback(0)
    assert a.n_forward_tokens == n_before
