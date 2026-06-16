"""Packing-aware document-boundary parity (#68), MLX (Apple Silicon only).

A packed multi-document forward (with seg_ids) must equal each document's standalone
forward — proving SSM/attention state doesn't bleed across chunk-aligned boundaries.
"""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from src.model.blocks import load_config
from src.model.mlx_backend import MLXMambaModel
from src.conformance.doc_boundary_parity import check_doc_boundary_parity


def _docs(cfg, lengths, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, cfg.vocab_size, size=n).tolist() for n in lengths]


def test_doc_boundary_parity_toy():
    cfg = load_config("config/toy.yaml")
    mx.random.seed(0)
    model = MLXMambaModel(cfg)
    Q = cfg.chunk_size or 64
    # exact chunk multiples, several chunks, and a short doc that needs padding
    docs = _docs(cfg, [Q, 2 * Q, 5])
    res = check_doc_boundary_parity(model, docs, Q, to_numpy=np.array)
    assert res["ok"] and res["max_abs_diff"] < 1e-3


def test_doc_boundary_parity_hybrid():
    cfg = load_config("config/toy-hybrid.yaml")
    mx.random.seed(0)
    model = MLXMambaModel(cfg)
    Q = cfg.chunk_size or 64
    docs = _docs(cfg, [Q, 7, Q + 3], seed=1)        # mixes attention + Mamba layers
    res = check_doc_boundary_parity(model, docs, Q, to_numpy=np.array)
    assert res["ok"] and res["max_abs_diff"] < 1e-3


def test_grad_checkpoint_path_supports_seg_ids():
    # poc uses grad_checkpoint=True; make sure seg_ids threads through the checkpointed
    # layer wrappers (build a tiny checkpointed config).
    cfg = load_config("config/toy.yaml")
    cfg.grad_checkpoint = True
    mx.random.seed(0)
    model = MLXMambaModel(cfg)
    Q = cfg.chunk_size or 64
    res = check_doc_boundary_parity(model, _docs(cfg, [Q, Q]), Q, to_numpy=np.array)
    assert res["ok"]
