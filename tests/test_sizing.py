"""Param-count + sizing tests (portable; runs anywhere, no backend).

Guards the closed-form `MambaConfig.num_parameters()` and the `sizing` memory
estimates, and pins the scaling-config family (poc/1b/2b/4b) to its targets.

The exact-vs-real-tensors safety net (num_parameters == built model param sum)
lives in the MLX-gated test below — only a real backend has the actual tensors.
"""

import math
from pathlib import Path

import pytest

from src.model.blocks import MambaConfig, load_config
from src.model import sizing

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

# (name, approx target params) for the scaling ladder. poc is the existing ~127M.
FAMILY = [("poc", 127e6), ("1b", 1e9), ("2b", 2e9), ("4b", 4e9)]


def _cfg(name):
    return load_config(CONFIG_DIR / f"{name}.yaml")


def test_num_parameters_equals_breakdown_sum():
    # Toy/poc plus every ladder config: the total must equal the sum of named terms.
    cfgs = [load_config(CONFIG_DIR / "toy.yaml")] + [_cfg(n) for n, _ in FAMILY]
    for cfg in cfgs:
        bd = cfg.parameter_breakdown()
        assert cfg.num_parameters() == sum(bd.values())
        assert all(v > 0 for v in bd.values())


def test_hybrid_breakdown_sums_and_counts_attention():
    cfg = load_config(CONFIG_DIR / "toy-hybrid.yaml")
    cfg.validate()
    bd = cfg.parameter_breakdown()
    assert "attention" in bd and bd["attention"] > 0
    assert cfg.num_parameters() == sum(bd.values())
    assert cfg.n_attention_layers == cfg.n_layers // cfg.attn_every
    # Attention layers REPLACE Mamba layers: at toy dims a Mamba block is heavier than
    # an attention block, so the hybrid has fewer params than the all-Mamba twin.
    pure = MambaConfig(**{**cfg.to_dict(), "attn_every": None})
    assert cfg.num_parameters() < pure.num_parameters()


def test_attention_param_formula():
    # attn layer = norm(d_model) + qkv(3*d_model*d_attn) + o_proj(d_attn*d_model), d_attn=d_model.
    cfg = MambaConfig(d_model=64, n_layers=4, head_dim=16, vocab_size=256,
                      attn_every=2, n_attn_heads=4)
    d_model, d_attn = 64, 64
    expect_per = d_model + 3 * d_model * d_attn + d_attn * d_model
    assert cfg.parameter_breakdown()["attention"] == cfg.n_attention_layers * expect_per


def test_untied_adds_lm_head():
    tied = MambaConfig(d_model=64, n_layers=2, head_dim=16, vocab_size=256,
                       tie_embeddings=True)
    untied = MambaConfig(d_model=64, n_layers=2, head_dim=16, vocab_size=256,
                         tie_embeddings=False)
    bd = untied.parameter_breakdown()
    assert "lm_head" in bd and "lm_head" not in tied.parameter_breakdown()
    # The only difference is the extra vocab x d_model head.
    assert untied.num_parameters() - tied.num_parameters() == 256 * 64


def test_known_poc_count():
    # The poc config is the validated ~127M anchor; pin it tightly.
    poc = _cfg("poc")
    assert abs(poc.num_parameters() - 126.7e6) < 0.5e6


@pytest.mark.parametrize("name,target", FAMILY)
def test_family_config_valid_and_on_target(name, target):
    cfg = _cfg(name)
    cfg.validate()                                   # raises on any invariant break
    assert cfg.packing_dtype == "uint16"             # the POC family is uint16-packable
    assert cfg.d_inner % cfg.head_dim == 0
    dev = abs(cfg.num_parameters() - target) / target
    assert dev <= 0.05, f"{name}: {cfg.num_parameters()/1e6:.1f}M is {dev:.1%} off {target/1e6:.0f}M"


def test_family_param_counts_monotonic():
    counts = [_cfg(n).num_parameters() for n, _ in FAMILY]
    assert counts == sorted(counts), "ladder params must increase by tier"


def test_inference_bytes_no_kv_cache():
    cfg = _cfg("1b")
    # No KV cache -> inference footprint is exactly weights * dtype bytes.
    assert sizing.inference_bytes(cfg, "bf16") == cfg.num_parameters() * 2
    assert sizing.inference_bytes(cfg, "fp32") == cfg.num_parameters() * 4
    with pytest.raises(ValueError):
        sizing.inference_bytes(cfg, "int4")


def test_training_bytes_sane_and_optimizer_lever():
    cfg = _cfg("1b")
    adamw = sizing.training_bytes(cfg, optimizer="adamw")
    adam8 = sizing.training_bytes(cfg, optimizer="adam8bit")
    for t in (adamw, adam8):
        assert t["total"] == t["model_opt"] + t["activations"]
        # Training needs more than just the weights.
        assert t["total"] > sizing.inference_bytes(cfg, "bf16")
    # The 8-bit-Adam + fp32-master estimate is heavier per param than lean bf16 Adam.
    assert adam8["model_opt"] > adamw["model_opt"]
    with pytest.raises(ValueError):
        sizing.training_bytes(cfg, optimizer="sgd")


def test_grad_checkpoint_cuts_activations():
    base = _cfg("1b")
    off = MambaConfig(**{**base.to_dict(), "grad_checkpoint": False})
    on = MambaConfig(**{**base.to_dict(), "grad_checkpoint": True})
    a_off = sizing.activation_bytes(off, batch_size=8, seq_len=1024)
    a_on = sizing.activation_bytes(on, batch_size=8, seq_len=1024)
    assert a_on < a_off


def test_family_table_renders_all_tiers():
    table = sizing.format_family_table(sizing.load_family(CONFIG_DIR))
    for name, _ in FAMILY:
        assert name in table
    # Header columns present.
    assert "params" in table and "GPU (train)" in table
