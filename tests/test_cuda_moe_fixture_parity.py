"""The cross-backend gate for CUDA MoE (#214): the checked-in Swift/MLX parity fixtures
(`swift/engine/Fixtures/toy-moe*/`) are a frozen MLX oracle -- self-describing weights +
tokens + reference logits/hidden-states. This is the torch-only half of the #166 harness:
no CI job has BOTH mlx and torch, so this is the only place MLX<->CUDA MoE numerics are
actually gated. See `swift/engine/Fixtures/README.md`.

Parameterized over {toy-moe, toy-moe-biased} x {dense, gather}: the reconstructed
`MambaConfig` from each fixture's `.config.json` sidecar predates #214 (no `moe_impl`
key), so both compute strategies are exercised explicitly by overriding it, rather than
relying on whatever "auto" happens to resolve to.

Skipped where torch is unavailable (CPU-only, no GPU needed).
"""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from safetensors.numpy import load_file  # noqa: E402

from src.model.blocks import MambaConfig  # noqa: E402
from src.model.cuda_backend import CUDAMambaModel  # noqa: E402
from src.train.checkpoint import load_weights  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / "swift" / "engine" / "Fixtures"
RTOL, ATOL = 1e-4, 1e-5


def _load_fixture(name: str, moe_impl: str):
    fdir = FIXTURES / name
    raw = __import__("json").loads((fdir / "weights.safetensors.config.json").read_text())
    raw = {**raw, "moe_impl": moe_impl}       # pre-#214 sidecar has no moe_impl key
    cfg = MambaConfig(**raw)
    cfg.validate()

    model = CUDAMambaModel(cfg)
    model.eval()
    load_weights(model, str(fdir / "weights.safetensors"))

    tokens = load_file(str(fdir / "inputs.safetensors"))["tokens"]
    ref = load_file(str(fdir / "reference.safetensors"))
    meta = __import__("json").loads((fdir / "meta.json").read_text())
    return model, tokens, ref, meta


@pytest.mark.parametrize("fixture", ["toy-moe", "toy-moe-biased"])
@pytest.mark.parametrize("moe_impl", ["dense", "gather"])
def test_cuda_matches_mlx_reference_forward_and_step(fixture, moe_impl):
    model, tokens, ref, meta = _load_fixture(fixture, moe_impl)
    assert meta["precision"] == "fp32", (
        "the fixture must stay fp32 -- a future regeneration at fp16/bf16 would "
        "silently loosen this gate's precision without anyone noticing"
    )

    with torch.no_grad():
        forward_logits = model.forward(tokens).detach().cpu().numpy()

        state = model.init_state(tokens.shape[0])
        steps = []
        for t in range(tokens.shape[1]):
            logits_t, state = model.step(tokens[:, t], state)
            steps.append(logits_t.detach().cpu().numpy())
        step_logits = np.stack(steps, axis=1)

    fwd_diff = float(np.abs(forward_logits.astype(np.float64)
                            - ref["forward_logits"].astype(np.float64)).max())
    assert np.allclose(forward_logits, ref["forward_logits"], rtol=RTOL, atol=ATOL), (
        f"{fixture}/{moe_impl}: forward_logits max|diff|={fwd_diff:.3e}"
    )

    step_diff = float(np.abs(step_logits.astype(np.float64)
                             - ref["step_logits"].astype(np.float64)).max())
    assert np.allclose(step_logits, ref["step_logits"], rtol=RTOL, atol=ATOL), (
        f"{fixture}/{moe_impl}: step_logits max|diff|={step_diff:.3e}"
    )


@pytest.mark.parametrize("fixture", ["toy-moe", "toy-moe-biased"])
@pytest.mark.parametrize("moe_impl", ["dense", "gather"])
def test_cuda_matches_mlx_reference_hidden_states(fixture, moe_impl):
    """Per-layer localization: if the whole-model logits ever mismatch, this pins which
    layer diverged first instead of a week of manual bisection (the same reasoning as
    `scripts/export_parity_fixture.py`'s own comment on why `hidden.{i}` is worth the
    ~200 KB)."""
    model, tokens, ref, _ = _load_fixture(fixture, moe_impl)
    with torch.no_grad():
        hidden = model.hidden_states(tokens)
    n_layers = model.config.n_layers
    assert len(hidden) == n_layers + 1
    for i, h in enumerate(hidden):
        h_np = h.detach().cpu().numpy()
        ref_h = ref[f"hidden.{i}"]
        diff = float(np.abs(h_np.astype(np.float64) - ref_h.astype(np.float64)).max())
        assert np.allclose(h_np, ref_h, rtol=RTOL, atol=ATOL), (
            f"{fixture}/{moe_impl}: hidden.{i} first diverges here, max|diff|={diff:.3e}"
        )


def test_biased_fixture_bias_values_match():
    """The biased fixture's `moe_route_bias.*` load must reproduce as the SAME numbers
    `model.moe_biases()` reports -- the D3 portable round-trip, exercised end-to-end
    against a real cross-backend (MLX-written) checkpoint rather than a CUDA-written one."""
    model, _, _, _ = _load_fixture("toy-moe-biased", "dense")
    fdir = FIXTURES / "toy-moe-biased"
    weights = load_file(str(fdir / "weights.safetensors"))
    expected = {
        int(k[len("moe_route_bias."):]): weights[k].tolist()
        for k in weights if k.startswith("moe_route_bias.")
    }
    moe_layer_indices = [i for i in range(model.config.n_layers) if model.config.is_moe_layer(i)]
    got = model.moe_biases()
    assert len(got) == len(moe_layer_indices)
    for pos, layer_idx in enumerate(moe_layer_indices):
        assert layer_idx in expected, (layer_idx, expected)
        assert np.allclose(got[pos], expected[layer_idx], atol=1e-6)
