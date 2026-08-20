"""Backend parity (#38): MLX and torch agree, and portable weights round-trip.

Because the CUDA backend runs on torch-CPU, the cross-backend checks are runnable
entirely on a Mac (mlx + torch both present) — no GPU. On a single-backend host they
SKIP cleanly, so the suite stays green:
  * a Linux container (torch present, mlx not installable) — cross-backend tests skip;
    the torch-only harness self-check below still runs;
  * a CUDA host without mlx — same;
  * a Mac without torch — all skip.

**One host is exempt from that, deliberately (#303).** Skipping is the right default,
but it is also how these five comparisons sat dormant in *every* CI job for the whole
M12 build: a check that cannot observe its target reads identical to the good outcome.
So the CI job designated as the cross-backend gate (`full-macos`, which installs
`.[dev,data,mlx,cuda]`) sets ``MONICA_REQUIRE_BOTH_BACKENDS=1``. Under that flag
``requires_both_backends`` attaches **no marker at all**, so a missing backend raises
ImportError from the test body — an error, not a silent `s`.

All comparisons are fp32, ~1e-4 rel (the documented tolerance; bf16/fp16 epsilon is too
coarse to be meaningful), per src/conformance/backend_parity.py.
"""

import os

import numpy as np
import pytest

try:
    import mlx.core  # noqa: F401
    HAVE_MLX = True
except ImportError:
    HAVE_MLX = False

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

from src.model.blocks import load_config
from src.conformance.backend_parity import check_backend_parity

CFG = "config/toy.yaml"

# #303: set by exactly one CI job (ci.yml's `full-macos`, guarded by
# tests/test_ci_backend_matrix.py). On that host a missing backend must be an ERROR.
REQUIRE_BOTH = os.environ.get("MONICA_REQUIRE_BOTH_BACKENDS") == "1"

# The contract: these five are the real MLX-vs-torch comparisons. Named literally so a
# rename that drops one from the gate is a loud KeyError, not a quiet coverage hole.
# (`test_parity_harness_torch_self` is deliberately NOT here — it compares torch against
# itself and keeps the Linux `cuda-cpu` job meaningful.)
CROSS_BACKEND_TESTS = (
    "test_backend_parity_mlx_vs_torch",
    "test_backend_parity_hybrid",
    "test_backend_parity_seg_ids",
    "test_portable_weights_roundtrip_both_directions",
    "test_moe_routing_entropy_parity_mlx_vs_torch",
)


def requires_both_backends(fn):
    """Skip on a single-backend host; on the designated both-backends job, do not.

    Returning ``fn`` unwrapped is the whole mechanism: with no skipif marker there is
    nothing that *can* skip, so an absent backend surfaces as the ImportError each test
    body raises on its first line.
    """
    if REQUIRE_BOTH:
        return fn
    return pytest.mark.skipif(
        not (HAVE_MLX and HAVE_TORCH),
        reason="needs both mlx and torch (set MONICA_REQUIRE_BOTH_BACKENDS=1 to make this an error)",
    )(fn)


def _tokens(cfg, B=2, L=24, seed=0):
    return np.random.default_rng(seed).integers(0, cfg.vocab_size, size=(B, L)).astype(np.int32)


def _mlx_np(a):
    return np.array(a)


def _torch_np(a):
    return a.detach().cpu().numpy()


@requires_both_backends
def test_backend_parity_mlx_vs_torch(tmp_path):
    """Identical portable weights in both backends -> `forward` agrees in fp32."""
    from src.model.mlx_backend import MLXMambaModel
    from src.model.cuda_backend import CUDAMambaModel

    cfg = load_config(CFG)
    # One source of weights -> both backends (torch is the source here; the round-trip
    # test below proves the other direction).
    torch.manual_seed(0)
    src = CUDAMambaModel(cfg)
    path = str(tmp_path / "weights.safetensors")
    src.save(path)

    mlx_m = MLXMambaModel(cfg)
    mlx_m.load(path)
    cuda_m = CUDAMambaModel(cfg)
    cuda_m.load(path)

    tokens = _tokens(cfg)
    with torch.no_grad():
        result = check_backend_parity(mlx_m, cuda_m, tokens,
                                      to_numpy_a=_mlx_np, to_numpy_b=_torch_np,
                                      rtol=1e-4, atol=1e-5)
    assert result["ok"], result


@requires_both_backends
def test_backend_parity_hybrid(tmp_path):
    """Hybrid (attention layers present): identical portable weights in both backends ->
    `forward` agrees in fp32. Proves the attention block ports MLX<->torch, including the
    qkv_proj/o_proj weights round-tripping through the portable state dict."""
    from src.model.mlx_backend import MLXMambaModel
    from src.model.cuda_backend import CUDAMambaModel

    cfg = load_config("config/toy-hybrid.yaml")
    torch.manual_seed(0)
    src = CUDAMambaModel(cfg)
    assert any(type(l).__name__ == "AttentionBlock" for l in src.layers)
    path = str(tmp_path / "weights.safetensors")
    src.save(path)

    mlx_m = MLXMambaModel(cfg); mlx_m.load(path)
    cuda_m = CUDAMambaModel(cfg); cuda_m.load(path)

    tokens = _tokens(cfg, B=2, L=24)
    with torch.no_grad():
        result = check_backend_parity(mlx_m, cuda_m, tokens,
                                      to_numpy_a=_mlx_np, to_numpy_b=_torch_np,
                                      rtol=1e-4, atol=1e-5)
    assert result["ok"], result


@requires_both_backends
def test_backend_parity_seg_ids(tmp_path):
    """Packed multi-doc forward with seg_ids (#68/#111) agrees MLX<->torch in fp32. Proves
    the CUDA seg_ids path (inter-chunk mask + boundary-aware conv + block-diagonal attn)
    matches the MLX reference, not just each backend's own doc-boundary self-consistency."""
    from src.model.mlx_backend import MLXMambaModel
    from src.model.cuda_backend import CUDAMambaModel

    cfg = load_config("config/toy-hybrid.yaml")        # exercises both Mamba + attention
    Q = cfg.chunk_size or 64
    torch.manual_seed(0)
    src = CUDAMambaModel(cfg)
    path = str(tmp_path / "weights.safetensors")
    src.save(path)
    mlx_m = MLXMambaModel(cfg); mlx_m.load(path)
    cuda_m = CUDAMambaModel(cfg); cuda_m.load(path)

    # Pack two chunk-aligned docs into one sequence with seg_ids (doc boundaries on chunks).
    rng = np.random.default_rng(0)
    packed, seg = [], []
    for d, n in enumerate([Q, 2 * Q]):
        packed.extend(rng.integers(0, cfg.vocab_size, size=n).tolist())
        seg.extend([d] * n)
    packed = np.asarray(packed, dtype=np.int32)[None]
    seg = np.asarray(seg, dtype=np.int32)[None]

    with torch.no_grad():
        a = _mlx_np(mlx_m.forward(packed, seg)).astype(np.float64)
        b = _torch_np(cuda_m.forward(packed, seg)).astype(np.float64)
    max_abs = float(np.abs(a - b).max())
    assert np.allclose(a, b, rtol=1e-4, atol=1e-5), f"seg_ids parity drift {max_abs:.3e}"


@requires_both_backends
def test_portable_weights_roundtrip_both_directions(tmp_path):
    """MLX save -> torch _load_portable -> torch save -> load back into MLX; the MLX
    logits are unchanged. Proves the cross-backend bridge in both directions (a
    CUDA-trained model can come back to the Mac)."""
    from src.model.mlx_backend import MLXMambaModel
    from src.model.cuda_backend import CUDAMambaModel

    cfg = load_config(CFG)
    import mlx.core as mx
    mx.random.seed(0)
    mlx_src = MLXMambaModel(cfg)
    tokens = _tokens(cfg)
    before = _mlx_np(mlx_src.forward(tokens))

    p_mlx = str(tmp_path / "from_mlx.safetensors")
    mlx_src.save(p_mlx)                         # MLX -> portable
    bridge = CUDAMambaModel(cfg)
    bridge.load(p_mlx)                          # portable -> torch
    p_torch = str(tmp_path / "from_torch.safetensors")
    bridge.save(p_torch)                        # torch -> portable

    mlx_back = MLXMambaModel(cfg)
    mlx_back.load(p_torch)                      # portable -> MLX
    after = _mlx_np(mlx_back.forward(tokens))

    max_abs = float(np.abs(before.astype(np.float64) - after.astype(np.float64)).max())
    assert np.allclose(before, after, rtol=1e-4, atol=1e-5), f"round-trip drift {max_abs:.3e}"


@requires_both_backends
def test_moe_routing_entropy_parity_mlx_vs_torch(tmp_path):
    """#217: the two backends' routing-entropy diagnostic must agree, not just the
    logits. Identical portable weights, identical input, load counting on in both --
    per-layer mean entropy compared at fp32 ~1e-4 (same tolerance as the logits parity
    tests above)."""
    from src.model.mlx_backend import MLXMambaModel
    from src.model.cuda_backend import CUDAMambaModel

    cfg = load_config("config/toy-moe.yaml")
    torch.manual_seed(0)
    src = CUDAMambaModel(cfg)
    path = str(tmp_path / "weights.safetensors")
    src.save(path)

    mlx_m = MLXMambaModel(cfg)
    mlx_m.load(path)
    cuda_m = CUDAMambaModel(cfg)
    cuda_m.load(path)
    mlx_m.set_moe_load_counting(True)
    cuda_m.set_moe_load_counting(True)

    tokens = _tokens(cfg)
    mlx_m.forward(tokens)
    with torch.no_grad():
        cuda_m.forward(tokens)
    mlx_stats = mlx_m.pop_moe_routing_stats()
    cuda_stats = cuda_m.pop_moe_routing_stats()

    assert len(mlx_stats) == len(cuda_stats) == cfg.n_moe_layers > 0
    for a, b in zip(mlx_stats, cuda_stats):
        assert a["entropy"] is not None and b["entropy"] is not None
        assert a["entropy"] == pytest.approx(b["entropy"], rel=1e-4, abs=1e-5)
        assert a["n_tokens"] == b["n_tokens"]


@pytest.mark.skipif(not HAVE_TORCH, reason="needs torch")
def test_parity_harness_torch_self(tmp_path):
    """Runnable without mlx: identical weights in two torch instances pass the parity
    harness (exercises check_backend_parity + the to_numpy plumbing on this host)."""
    from src.model.cuda_backend import CUDAMambaModel

    cfg = load_config(CFG)
    torch.manual_seed(0)
    src = CUDAMambaModel(cfg)
    path = str(tmp_path / "weights.safetensors")
    src.save(path)

    a = CUDAMambaModel(cfg)
    a.load(path)
    b = CUDAMambaModel(cfg)
    b.load(path)

    tokens = _tokens(cfg)
    with torch.no_grad():
        result = check_backend_parity(a, b, tokens,
                                      to_numpy_a=_torch_np, to_numpy_b=_torch_np,
                                      rtol=1e-4, atol=1e-5)
    assert result["ok"], result


# ── #303 guards: the gate must be observable, not merely present ──────────────
def test_designated_job_has_both_backends():
    """Layer 1. On the host that *declares* it carries both backends, not having them is
    a failure with a message naming the cause — never a skip.

    Without this, a botched install line on `full-macos` would leave the five comparisons
    erroring only via ImportError deep in a test body; this says it once, up front.
    """
    if not REQUIRE_BOTH:
        pytest.skip("not the designated both-backends host (MONICA_REQUIRE_BOTH_BACKENDS unset)")
    assert HAVE_MLX, (
        "MONICA_REQUIRE_BOTH_BACKENDS=1 but `import mlx.core` failed — this job is the "
        "cross-backend parity gate and its install step must resolve an mlx wheel "
        "(ci.yml `full-macos`: pip install -e '.[dev,data,mlx,cuda]')"
    )
    assert HAVE_TORCH, (
        "MONICA_REQUIRE_BOTH_BACKENDS=1 but `import torch` failed — this job is the "
        "cross-backend parity gate and its install step must resolve a torch wheel "
        "(the macOS arm64 PyPI wheel is CPU-only, which is the surface cuda_backend.py "
        "is compared on)"
    )


def test_cross_backend_tests_carry_no_skip_marker_when_required():
    """Layer 2. On the designated job the five must actually RUN, so none of them may
    carry a skip/skipif marker — re-adding one is a red test, not a quiet skip.

    ``globals()[name]`` raising KeyError on a rename is intentional: the tuple is the
    contract, and a rename that silently drops a comparison from the gate is precisely
    the regression #303 exists to prevent.
    """
    if not REQUIRE_BOTH:
        pytest.skip("not the designated both-backends host (MONICA_REQUIRE_BOTH_BACKENDS unset)")
    for name in CROSS_BACKEND_TESTS:
        fn = globals()[name]
        marks = {m.name for m in getattr(fn, "pytestmark", [])}
        assert not ({"skip", "skipif"} & marks), (
            f"{name} carries {sorted({'skip', 'skipif'} & marks)} and would skip on the "
            f"both-backends job — the cross-backend comparison must execute here"
        )
