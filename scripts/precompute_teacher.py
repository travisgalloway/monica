"""Teacher top-k logit precompute driver (#94) — the dominant, precompute-once cost of M10.

Runs the FROZEN conversion teacher over the flat packed distill corpus (`<data>/train.bin`,
`<data>/val.bin` — `src/data/split.py`'s output) and caches, per token, the top-`k` logits +
vocab indices, aligned positionally to the corpus tokens. Every student trial then reads these
back with ZERO teacher inference (`src.data.teacher_outputs.DistillLoader`), so the expensive
7B forward is paid once and reused across the layout sweep (docs/design/10-distillation.md).

The teacher forward is the program's heaviest GPU job, so this runs on the cloud GPU via
`--backend cuda` (the torch `CUDATeacher`); `--backend mlx` is the Apple-Silicon path. The
backend import stays behind `src.model.backend.get_backend`, so `--help` works on any host.

    # real run (cloud GPU): cache top-50 for the OpenR1-Distill-7B teacher, push to R2
    .venv/bin/python scripts/precompute_teacher.py --manifest config/manifests/student-1b-attn-hi.yaml \\
        --data data/poc-distill/split --backend cuda --k 50 \\
        --push s3://monica-training/poc-distill/teacher-outputs/topk-logits

    # offline smoke (byte vocab, synthetic teacher — no network/weights):
    .venv/bin/python scripts/precompute_teacher.py --manifest config/manifests/toy-distill.yaml \\
        --data /tmp/toy-split --out /tmp/teacher-outputs --backend mlx --k 8 --synthetic
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True,
                    help="student manifest (supplies conversion_teacher, seq_len)")
    ap.add_argument("--data", type=Path, required=True, help="dir with train.bin/val.bin")
    ap.add_argument("--splits", default="train,val", help="comma-separated splits (default train,val)")
    ap.add_argument("--out", type=Path, default=None,
                    help="teacher-outputs dir (default: <data>/teacher-outputs/topk-logits)")
    ap.add_argument("--backend", choices=("auto", "mlx", "cuda"), default="auto",
                    help="hardware backend (the 7B precompute uses cuda on the cloud GPU)")
    ap.add_argument("--k", type=int, default=50, help="logits cached per token (footprint ~6k B/token)")
    ap.add_argument("--batch-size", type=int, default=8, help="chunks per teacher forward")
    ap.add_argument("--pretrained", default=None,
                    help="HF checkpoint dir / repo id (default: manifest.conversion_teacher)")
    ap.add_argument("--synthetic", action="store_true",
                    help="build a synthetic toy teacher (TeacherConfig.tiny) — offline tests only")
    ap.add_argument("--push", default=None,
                    help="after writing, mirror --out to this fsspec URI / R2 prefix (#80)")
    return ap.parse_args()


def _teacher_config_for(model_id: str):
    """Map a known teacher repo id to its `TeacherConfig` (so logits slice to the tokenizer
    vocab — see `TeacherConfig.openr1_distill_7b`'s note). None lets `from_pretrained` read
    the checkpoint's `config.json`."""
    from src.model.teacher import TeacherConfig
    known = {"open-r1/OpenR1-Distill-7B": TeacherConfig.openr1_distill_7b,
             "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": TeacherConfig.qwen_1_5b}
    return known[model_id]() if model_id in known else None


def _topk_blocks(teacher, backend, data, n_chunks, stride, seq_len, batch_size, k):
    """Yield `(vals, idx)` numpy blocks over the packed file in on-disk chunk order (no shuffle
    — alignment is positional). Each block is `(b, seq_len, k_eff)`."""
    import numpy as np
    for c0 in range(0, n_chunks, batch_size):
        c1 = min(c0 + batch_size, n_chunks)
        rows = [np.asarray(data[c * stride: c * stride + seq_len], dtype=np.int64)
                for c in range(c0, c1)]
        inputs = np.stack(rows)                         # (b, seq_len)
        vals, idx = teacher.topk_logits(inputs, k)
        yield backend.to_numpy(vals), backend.to_numpy(idx)


def main() -> None:
    args = _parse_args()

    from src.model.backend import get_backend
    from src.model.teacher import TeacherConfig
    from src.train.distill_manifest import load_manifest
    from src.data import storage
    from src.data.pack import open_packed
    from src.data.teacher_outputs import write_teacher_topk, write_manifest

    manifest = load_manifest(args.manifest)
    seq_len = manifest.seq_len
    backend = get_backend(args.backend)

    if args.synthetic:
        teacher = backend.make_teacher(config=TeacherConfig.tiny())
        teacher_info = {"model_id": None, "synthetic": True,
                        "vocab_size": teacher.config.vocab_size}
    else:
        model_id = args.pretrained or manifest.conversion_teacher
        teacher = backend.make_teacher(config=_teacher_config_for(model_id), pretrained=model_id)
        teacher_info = {"model_id": model_id, "synthetic": False,
                        "vocab_size": teacher.config.vocab_size}

    eff_vocab = teacher.config.effective_vocab_size
    out_dir = args.out or storage.teacher_outputs_dir(args.data)
    out_dir = Path(out_dir)
    splits = [s for s in args.splits.split(",") if s]

    for split in splits:
        data = open_packed(args.data / f"{split}.bin")
        n_tokens = int(data.shape[0])
        stride = seq_len + 1
        n_chunks = n_tokens // stride
        if n_chunks == 0:
            raise ValueError(f"{split}.bin too small for one chunk (seq_len={seq_len})")
        blocks = _topk_blocks(teacher, backend, data, n_chunks, stride, seq_len,
                              args.batch_size, args.k)
        meta = write_teacher_topk(out_dir, split, blocks=blocks, n_chunks=n_chunks,
                                  seq_len=seq_len, vocab_size=eff_vocab,
                                  src_packed=str(args.data / f"{split}.bin"),
                                  src_n_tokens=n_tokens)
        print(f"teacher top-k [{split}]: {meta['n_rows']} rows x k={meta['k']} "
              f"({n_chunks} chunks x {seq_len}) -> {out_dir}")

    write_manifest(out_dir, k=args.k, seq_len=seq_len, effective_vocab_size=eff_vocab,
                   corpus_manifest=str(args.data), teacher=teacher_info, splits=splits)

    if args.push:
        from src.data.r2_sync import upload_dir
        written = upload_dir(out_dir, args.push)
        print(f"pushed {len(written)} file(s): {out_dir} -> {args.push}")


if __name__ == "__main__":
    main()
