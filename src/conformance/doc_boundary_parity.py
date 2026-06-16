"""Packing-aware document-boundary parity (#68).

When several documents are packed into one training sequence, recurrent SSM state (and
attention) must not bleed across the boundaries — a packed multi-document forward has to
equal running each document on its own. This is the conformance gate for the `seg_ids`
path added to `ModelInterface.forward`: a silent leak across a boundary corrupts training
in a way ordinary losses don't surface (mirrors `forward_step_parity` for the SSD scan).

Document boundaries must be **chunk-aligned** (each doc starts at a multiple of
`chunk_size`); this helper pads each doc up to a chunk multiple, packs them with `seg_ids`,
and checks each document's logit slice against its standalone forward. Run in fp32 at
~1e-4 relative tolerance — bf16's epsilon is too coarse.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ..model.interface import ModelInterface


def check_doc_boundary_parity(model: ModelInterface, docs: Sequence[Sequence[int]],
                              chunk_size: int, *, to_numpy=np.asarray, pad_id: int = 0,
                              rtol: float = 1e-4, atol: float = 1e-5) -> dict:
    """Pack `docs` chunk-aligned into one sequence with `seg_ids` and assert each document's
    logits match its standalone forward. Returns `{max_abs_diff, ok}`; raises on mismatch.

    Each doc is padded up to a multiple of `chunk_size` (padding follows the real tokens, so
    causality keeps it from affecting them); only the real positions are compared.
    """
    packed: List[int] = []
    seg: List[int] = []
    spans: List[tuple] = []
    off = 0
    for d, doc in enumerate(docs):
        doc = [int(t) for t in doc]
        n = len(doc)
        plen = ((n + chunk_size - 1) // chunk_size) * chunk_size
        packed.extend(doc + [pad_id] * (plen - n))
        seg.extend([d] * plen)
        spans.append((off, off + n))
        off += plen

    packed_arr = np.asarray(packed, dtype=np.int64)[None]      # (1, Lp)
    seg_arr = np.asarray(seg, dtype=np.int64)[None]            # (1, Lp)
    packed_logits = to_numpy(model.forward(packed_arr, seg_arr))   # (1, Lp, V)

    max_abs = 0.0
    for d, doc in enumerate(docs):
        solo = to_numpy(model.forward(np.asarray([list(doc)], dtype=np.int64)))  # (1, n, V)
        s, e = spans[d]
        sub = packed_logits[:, s:e]
        diff = np.abs(sub.astype(np.float64) - solo.astype(np.float64))
        max_abs = max(max_abs, float(diff.max()))
        if not np.allclose(sub, solo, rtol=rtol, atol=atol):
            raise AssertionError(
                f"doc-boundary parity FAILED for document {d}: max|diff|={diff.max():.3e} "
                "— state leaked across a packed boundary")
    return {"max_abs_diff": max_abs, "ok": True}
