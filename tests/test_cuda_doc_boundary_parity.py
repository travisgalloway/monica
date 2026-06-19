"""Packing-aware document-boundary parity (#68) on the CUDA/torch backend (#111).

Mirror of tests/test_doc_boundary_parity.py: a packed multi-document forward (with
seg_ids) must equal each document's standalone forward, proving SSM/attention state
doesn't bleed across chunk-aligned boundaries. Runs on torch-CPU — no GPU needed — so the
CUDA seg_ids path is verified on the Mac before it ships to a CUDA host.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.model.blocks import load_config
from src.model.cuda_backend import CUDAMambaModel
from src.conformance.doc_boundary_parity import check_doc_boundary_parity


def _torch_np(a):
    return a.detach().cpu().numpy()


def _docs(cfg, lengths, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, cfg.vocab_size, size=n).tolist() for n in lengths]


def test_doc_boundary_parity_toy():
    cfg = load_config("config/toy.yaml")
    torch.manual_seed(0)
    model = CUDAMambaModel(cfg)
    Q = cfg.chunk_size or 64
    # exact chunk multiples, several chunks, and a short doc that needs padding
    docs = _docs(cfg, [Q, 2 * Q, 5])
    res = check_doc_boundary_parity(model, docs, Q, to_numpy=_torch_np)
    assert res["ok"] and res["max_abs_diff"] < 1e-3


def test_doc_boundary_parity_hybrid():
    cfg = load_config("config/toy-hybrid.yaml")
    torch.manual_seed(0)
    model = CUDAMambaModel(cfg)
    Q = cfg.chunk_size or 64
    docs = _docs(cfg, [Q, 7, Q + 3], seed=1)        # mixes attention + Mamba layers
    res = check_doc_boundary_parity(model, docs, Q, to_numpy=_torch_np)
    assert res["ok"] and res["max_abs_diff"] < 1e-3


def test_grad_checkpoint_path_supports_seg_ids():
    # poc uses grad_checkpoint=True; make sure seg_ids threads through the checkpointed
    # layer wrappers (grad is enabled here, so _layer_forward takes the _checkpoint path).
    cfg = load_config("config/toy.yaml")
    cfg.grad_checkpoint = True
    torch.manual_seed(0)
    model = CUDAMambaModel(cfg)
    Q = cfg.chunk_size or 64
    res = check_doc_boundary_parity(model, _docs(cfg, [Q, Q]), Q, to_numpy=_torch_np)
    assert res["ok"]
