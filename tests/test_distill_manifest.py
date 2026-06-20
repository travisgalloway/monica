"""Distillation manifest resolver (#99). Portable — runs without a backend."""

import pytest

from src.train.distill_manifest import (CANONICAL_STAGES, DistillManifest, DistillStage,
                                        InitMethod, distill_stages, load_manifest,
                                        manifest_to_config)

MANIFESTS = ["config/manifests/student-1b-attn-hi.yaml",
             "config/manifests/student-1b-attn-lo.yaml"]


@pytest.mark.parametrize("path", MANIFESTS)
def test_real_manifests_load(path):
    m = load_manifest(path)
    assert m.init == InitMethod.MAMBA_IN_THE_LLAMA       # both seeds default to MiL
    assert m.tokenizer == "qwen25" and m.vocab_size == 151646
    assert m.stages and all(s in CANONICAL_STAGES for s in m.stages)
    assert m.stages[:3] == ["mixing-match", "hidden-align", "logit-distill"]


@pytest.mark.parametrize("path", MANIFESTS)
def test_manifest_to_config(path):
    m = load_manifest(path)
    cfg = manifest_to_config(m)
    cfg.validate()
    assert cfg.d_model == m.layout["d_model"]
    assert cfg.n_layers == m.layout["n_layers"]
    assert cfg.attn_every == m.layout["attention_every"]    # sweep-schema -> model field
    assert cfg.d_state == m.layout["state_size"]
    assert cfg.vocab_size == 151646 and cfg.seq_len == m.seq_len


@pytest.mark.parametrize("path", MANIFESTS)
def test_distill_stages_order_and_filter(path):
    m = load_manifest(path)
    stages = distill_stages(m)
    # the three distillation stages, in manifest order; SFT/RL stages dropped
    assert stages == [DistillStage.MIXING_MATCH, DistillStage.HIDDEN_ALIGN,
                      DistillStage.LOGIT_DISTILL]
    assert all(isinstance(s, DistillStage) for s in stages)


def test_init_method_from_str():
    assert InitMethod.from_str("mamba-in-the-llama") == InitMethod.MAMBA_IN_THE_LLAMA
    assert InitMethod.from_str("mohawk") == InitMethod.MOHAWK
    with pytest.raises(ValueError):
        InitMethod.from_str("nope")


def _manifest(**over):
    base = dict(student="s", conversion_teacher="t", tokenizer="qwen25", seq_len=128,
                init=InitMethod.MOHAWK, stages=["mixing-match"], layout={})
    base.update(over)
    return DistillManifest(**base)


def test_validate_rejects_unknown_stage():
    with pytest.raises(ValueError):
        _manifest(stages=["mixing-match", "bogus-stage"]).validate()


def test_validate_rejects_empty_stages():
    with pytest.raises(ValueError):
        _manifest(stages=[]).validate()


def test_validate_rejects_unknown_tokenizer():
    with pytest.raises(ValueError):
        _manifest(tokenizer="gpt2").validate()
