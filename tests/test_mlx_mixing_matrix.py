"""MLX-only tests for the `mixing_matrix` interpretability accessor (#264).

This is the Python oracle that `swift/engine/`'s port is checked against, so these tests
must prove it correct BEFORE it is frozen into a fixture — a wrong oracle would gate Swift
against the wrong numbers. Three checks, none of which a shape check subsumes:

  * the defining identity `einsum('bhij,bjhp->bihp', M, X) == SelectiveSSM.parallel(x)`
    (pins the exact axis convention — a `(B,H,L,1)` `delta` broadcast instead of
    `(B,H,1,L)` compiles, has the same shape, and is wrong; this identity catches it);
  * strict causality — the upper triangle of `M` is exactly 0 (not just small);
  * `MLXMambaModel.mixing_matrices` skips attention AND MoE layers and returns the
    correct absolute layer indices (the MLX mirror of
    `tests/test_cuda_moe.py::test_mixing_matrices_skips_moe_layers`).

All comparisons at fp32 `rtol=1e-4, atol=1e-5` — not loosened.
"""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
# See scripts/export_parity_fixture.py's matching comment: this local MLX build (0.32.0)
# has been observed to silently corrupt results when many independently-constructed
# models allocate/free buffers of the same size class within one process — exactly what
# this file's three tests do. Disabling the buffer-cache pool (proven in isolation to
# take the observed failure rate to 0/10) trades allocator throughput, irrelevant here,
# for correctness.
mx.clear_cache()
mx.set_cache_limit(0)

from src.model.blocks import MambaConfig
from src.model.mlx_backend import MLXMambaModel, MambaBlock, _linear, _DTYPES


def _cfg(**over):
    base = dict(d_model=64, n_layers=4, head_dim=16, d_state=16, vocab_size=256,
                seq_len=32, precision="fp32")
    base.update(over)
    return MambaConfig(**base)


def _np(a):
    return np.array(a)


def test_mixing_matrix_identity_matches_parallel_scan():
    """einsum('bhij,bjhp->bihp', M, X) must equal SelectiveSSM.parallel(x), the defining
    property of the mixing matrix. This is what pins the `delta` axis convention: a
    `(B,H,L,1)` delta_col (instead of the correct `(B,H,1,L)`) broadcasts to the same
    shape and would only be caught here."""
    cfg = _cfg(n_layers=2)
    mx.random.seed(0)
    model = MLXMambaModel(cfg)
    block = model.layers[0]
    assert isinstance(block, MambaBlock)

    B, L = 2, 17
    x = mx.random.normal((B, L, cfg.d_model))
    xc = _ssm_input(block, x)
    M = block.ssm.mixing_matrix(xc)          # (B, H, L, L)
    H, N = cfg.n_heads, cfg.d_state
    P = cfg.head_dim
    Xh = mx.array(xc).reshape(B, L, H, P)
    y_from_matrix = mx.einsum("bhij,bjhp->bihp", M, Xh).reshape(B, L, H * P)
    y_from_parallel = block.ssm.parallel(xc)
    # Evaluate separately, not as one combined `mx.eval(a, b)` call: this MLX build
    # (0.32.0) has been observed to corrupt an UNRELATED later model's computation when
    # two independent graphs are realized in a single multi-arg eval here, even though
    # each value read back individually is correct. Splitting the eval avoids relying on
    # that behavior rather than working around a framework issue in the interpretability
    # accessor itself.
    mx.eval(y_from_matrix)
    mx.eval(y_from_parallel)

    a = np.array(y_from_matrix, dtype=np.float64)
    b = np.array(y_from_parallel, dtype=np.float64)
    assert np.allclose(a, b, rtol=1e-4, atol=1e-5), float(np.max(np.abs(a - b)))


def _ssm_input(block: MambaBlock, x):
    """Reproduce `MambaBlock.mixing_matrix`'s front-end (norm -> in_proj main half ->
    causal conv -> SiLU), using the module's own `_linear`/`_conv_seq` helpers -- the
    same code path `mixing_matrix` itself uses -- so this test exercises the real
    implementation rather than a hand-rolled duplicate of it."""
    cd = _DTYPES[block.config.precision]
    xn = block.norm(x)
    x_main, _ = mx.split(_linear(block.in_proj, xn, cd), 2, axis=-1)
    xc = block._conv_seq(x_main, cd)[:, : x.shape[1]]
    return mx.sigmoid(xc) * xc  # silu


def test_mixing_matrix_is_strictly_causal():
    """The upper triangle of M (j > i) must be EXACTLY zero -- segsum produces -inf
    above the diagonal and exp(-inf) == 0.0 exactly, not merely small. A transposed
    mixing matrix would have nonzero mass exactly where this must be zero."""
    cfg = _cfg(n_layers=1)
    mx.random.seed(1)
    model = MLXMambaModel(cfg)
    block = model.layers[0]
    B, L = 2, 13
    x = mx.random.normal((B, L, cfg.d_model))
    xc = _ssm_input(block, x)
    M = block.ssm.mixing_matrix(xc)     # (B, H, L, L)
    mx.eval(M)
    m = np.array(M)
    iu = np.triu_indices(L, k=1)
    upper = m[:, :, iu[0], iu[1]]
    assert np.max(np.abs(upper)) == 0.0
    # anti-degeneracy: the matrix is not all-zero either
    assert np.max(np.abs(m)) > 0.0


def test_mixing_matrices_skips_attention_and_moe_layers():
    """Mirrors tests/test_cuda_moe.py::test_mixing_matrices_skips_moe_layers: on a
    config with both attention and MoE layers, mixing_matrices must return only the
    plain Mamba layers, keyed by their ABSOLUTE layer index."""
    cfg = _cfg(n_layers=4, attn_every=4, moe_every=2, n_experts=4, top_k=2)
    # layer indices 0..3: is_attention_layer(3) True (attn_every=4 -> (i+1)%4==0 at i=3)
    # is_moe_layer(i) True at (i+1)%2==0 and not attention -> i=1 is MoE, i=3 is attention
    # (attention takes precedence per blocks.py:229-233) -> Mamba layers are 0, 2.
    assert cfg.is_moe_layer(1) and not cfg.is_attention_layer(1)
    assert cfg.is_attention_layer(3)
    mx.random.seed(2)
    model = MLXMambaModel(cfg)
    tokens = np.random.default_rng(0).integers(0, cfg.vocab_size, size=(2, 16)).astype(np.int32)
    out = model.mixing_matrices(mx.array(tokens))
    indices = [i for i, _ in out]
    assert indices == [0, 2]
    for i, m in out:
        mx.eval(m)
        arr = np.array(m)
        assert np.isfinite(arr).all()
        assert arr.shape == (2, 16, 16)
