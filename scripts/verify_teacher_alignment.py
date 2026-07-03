"""Phase B' safety gate (#65): verify the live teacher still agrees with the frozen
`topk-logits-merged` cache before reusing it for the append-only corpus extension.

The 566 GB teacher top-k cache (k=50, 230,318 chunks, seq 8192) is positionally bound to the
exact FineWeb `train.bin` it was computed against (`teacher_outputs.py`'s `n_chunks` assert,
`teacher_outputs.py:152`). Before appending new-source chunks onto it
(`scripts/append_new_chunks.py`), spot-check that a FRESH teacher forward over a few probe
chunks still lands on (mostly) the same top-k index set as the cached one — catching a
regenerated `train.bin` that silently drifted (different tokenizer version, different doc
order, ...), which would otherwise misalign every downstream chunk.

    .venv/bin/python scripts/verify_teacher_alignment.py \\
        --data /vol/fineweb/train.bin --topk-dir /vol/teacher-outputs/topk-logits-merged \\
        --split train --probe-chunks 0,100000,230317 --backend cuda

Exits nonzero (abort -> full re-precompute) if agreement drops below --min-agreement on any
probe chunk. Needs the live teacher (torch/CUDA) — runs in Phase B' on a pod, not in Phase A'.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List


def _probe_agreement(teacher, backend, data, topk_dir: Path, split: str, chunk: int,
                     seq_len: int, k: int) -> float:
    """Fresh-forward one probe chunk and return the mean top-k index-set agreement
    (`|fresh ∩ stored| / k` per position, averaged over the chunk's `seq_len` positions)
    against the stored `topk-logits-merged` cache."""
    import numpy as np

    from src.data.teacher_outputs import read_teacher_meta, topk_outputs_paths

    stride = seq_len + 1
    tokens = np.asarray(data[chunk * stride: chunk * stride + seq_len], dtype=np.int64)
    _fresh_vals, fresh_idx = teacher.topk_logits(tokens[None, :], k)
    fresh_idx = backend.to_numpy(fresh_idx)[0]                    # (seq_len, k)

    meta = read_teacher_meta(topk_dir, split)
    paths = topk_outputs_paths(topk_dir, split)
    k_stored = int(meta["k"])
    r0 = chunk * seq_len
    stored_idx = np.memmap(paths["idx"], dtype=np.dtype(meta["idx_dtype"]), mode="r",
                           shape=(int(meta["n_rows"]), k_stored))[r0: r0 + seq_len, :]

    k_eff = min(k, fresh_idx.shape[-1], stored_idx.shape[-1])
    agreements = []
    for pos in range(seq_len):
        a = set(int(x) for x in fresh_idx[pos, :k_eff])
        b = set(int(x) for x in stored_idx[pos, :k_eff])
        agreements.append(len(a & b) / k_eff)
    return float(np.mean(agreements))


def verify_alignment(data_path: Path, topk_dir: Path, split: str, probe_chunks: List[int],
                     seq_len: int, k: int, backend_name: str = "cuda",
                     min_agreement: float = 0.99) -> dict:
    """Run the alignment spot-check over `probe_chunks`; raises `SystemExit` (abort -> full
    re-precompute) if any probe falls below `min_agreement`. Returns `{chunk: agreement}`."""
    from src.data.pack import open_packed
    from src.model.backend import get_backend
    from src.model.teacher import TeacherConfig

    backend = get_backend(backend_name)
    teacher = backend.make_teacher(config=TeacherConfig.qwen3_4b_thinking(),
                                   pretrained="Qwen/Qwen3-4B-Thinking-2507")
    if teacher.config.effective_vocab_size != 151669:
        raise SystemExit(
            f"teacher effective_vocab_size {teacher.config.effective_vocab_size} != 151669 "
            "(Qwen3 tokenizer vocab) — wrong teacher/config for this cache")

    data = open_packed(data_path)
    results: dict = {}
    for chunk in probe_chunks:
        agreement = _probe_agreement(teacher, backend, data, topk_dir, split, chunk, seq_len, k)
        results[chunk] = agreement
        status = "PASS" if agreement >= min_agreement else "FAIL"
        print(f"probe chunk {chunk}: top-k agreement {agreement:.4f} [{status}]")

    if not all(v >= min_agreement for v in results.values()):
        raise SystemExit(
            f"alignment check FAILED (min_agreement={min_agreement}): {results} — abort and "
            "run a full re-precompute instead of an append")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="regenerated FineWeb train.bin")
    ap.add_argument("--topk-dir", type=Path, required=True,
                    help="the frozen topk-logits-merged dir to verify against")
    ap.add_argument("--split", default="train")
    ap.add_argument("--probe-chunks", default="0,100000,230317",
                    help="comma-separated chunk indices to spot-check")
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--backend", choices=("auto", "mlx", "cuda"), default="cuda")
    ap.add_argument("--min-agreement", type=float, default=0.99,
                    help="minimum top-k index-set agreement to pass (default 0.99)")
    args = ap.parse_args()

    probe_chunks = [int(c) for c in args.probe_chunks.split(",") if c]
    verify_alignment(args.data, args.topk_dir, args.split, probe_chunks, args.seq_len, args.k,
                     backend_name=args.backend, min_agreement=args.min_agreement)


if __name__ == "__main__":
    main()
