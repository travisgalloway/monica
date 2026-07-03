"""Teacher top-k logit precompute driver (#94) — the dominant, precompute-once cost of M10.

Runs the FROZEN conversion teacher over the flat packed distill corpus (`<data>/train.bin`,
`<data>/val.bin` — `src/data/split.py`'s output) and caches, per token, the top-`k` logits +
vocab indices, aligned positionally to the corpus tokens. Every student trial then reads these
back with ZERO teacher inference (`src.data.teacher_outputs.DistillLoader`), so the expensive
teacher forward is paid once and reused across the layout sweep (docs/design/10-distillation.md).

The teacher forward is the program's heaviest GPU job, so this runs on the cloud GPU via
`--backend cuda` (the torch `CUDATeacher`); `--backend mlx` is the Apple-Silicon path. The
backend import stays behind `src.model.backend.get_backend`, so `--help` works on any host.

    # real run (cloud GPU): cache top-50 for the Qwen3-4B-Thinking-2507 teacher, push to R2
    .venv/bin/python scripts/precompute_teacher.py --manifest config/manifests/student-1b-attn-hi.yaml \\
        --data data/poc-distill/split --backend cuda --k 50 \\
        --push s3://monica-training/poc-distill/teacher-outputs/topk-logits

    # cluster mode (4 pods, each processes 1/4 of the train chunks):
    #   pod 0: --shard-id 0 --num-shards 4 --push .../topk-logits/shard-0
    #   pod 1: --shard-id 1 --num-shards 4 --push .../topk-logits/shard-1
    #   ... then run scripts/merge_teacher_shards.py to combine
    #
    # offline smoke (byte vocab, synthetic teacher — no network/weights):
    .venv/bin/python scripts/precompute_teacher.py --manifest config/toy-distill.yaml \\
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
                    help="hardware backend (the teacher precompute uses cuda on the cloud GPU)")
    ap.add_argument("--k", type=int, default=50,
                    help="logits cached per token; footprint 6*k B/token (fp16 vals + uint32 idx), "
                         "e.g. k=50 -> ~300 B/token")
    ap.add_argument("--batch-size", type=int, default=8, help="chunks per teacher forward")
    ap.add_argument("--pretrained", default=None,
                    help="HF checkpoint dir / repo id (default: manifest.conversion_teacher)")
    ap.add_argument("--synthetic", action="store_true",
                    help="build a synthetic toy teacher (TeacherConfig.tiny) — offline tests only")
    ap.add_argument("--teacher-endpoint", default=None,
                    help="OpenAI-compatible base URL (e.g. LM Studio http://localhost:1234/v1) to use "
                         "as a top-k-logit-only teacher. PARTIAL: feeds logit-distill only — no "
                         "hidden-align/mixing-match/init. Prefer --pretrained for full fidelity.")
    ap.add_argument("--endpoint-model", default=None,
                    help="model name to request from --teacher-endpoint (default: server's loaded model)")
    ap.add_argument("--teacher-dtype", choices=("fp32", "fp16"), default="fp32",
                    help="MLX teacher compute dtype; fp16 halves teacher memory for the local "
                         "Apple-Silicon precompute (default fp32 — the bit-identical path)")
    ap.add_argument("--compile", action="store_true",
                    help="mx.compile the MLX teacher's logits-only forward (fuses the per-layer "
                         "op stream; opt-in, eager fp32 stays the default for conformance)")
    ap.add_argument("--push", default=None,
                    help="after writing, mirror --out to this fsspec URI / R2 prefix (#80)")
    ap.add_argument("--shard-id", type=int, default=None,
                    help="0-indexed shard for cluster mode (use with --num-shards); output goes to "
                         "<out>/shard-<shard-id>/ and push to <push>/shard-<shard-id>/")
    ap.add_argument("--num-shards", type=int, default=None,
                    help="total number of cluster pods; splits n_chunks evenly across shards")
    return ap.parse_args()


def _teacher_config_for(model_id: str):
    """Map a known teacher repo id to its `TeacherConfig` (so logits slice to the tokenizer
    vocab — see `TeacherConfig.qwen3_4b_thinking`'s note). None lets `from_pretrained` read
    the checkpoint's `config.json`."""
    from src.model.teacher import TeacherConfig
    known = {"Qwen/Qwen3-4B-Thinking-2507": TeacherConfig.qwen3_4b_thinking}
    return known[model_id]() if model_id in known else None


def _topk_blocks(teacher, backend, data, n_chunks, stride, seq_len, batch_size, k,
                 start_chunk: int = 0):
    """Yield `(vals, idx)` numpy blocks over the packed file in on-disk chunk order (no shuffle
    — alignment is positional). Each block is `(b, seq_len, k_eff)`.
    `start_chunk` offsets into the file for cluster-mode sharding."""
    import numpy as np
    end_chunk = start_chunk + n_chunks
    for c0 in range(start_chunk, end_chunk, batch_size):
        c1 = min(c0 + batch_size, end_chunk)
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

    # MLX-local precompute levers (no-ops on the cuda backend, which has its own #145 path).
    mlx_opts = {"compute_dtype": args.teacher_dtype, "compile": args.compile}

    if args.teacher_endpoint:
        # Backend-free, top-k-logit-only teacher over an OpenAI-compatible server (LM Studio etc.).
        # PARTIAL: supports the logit-distill stage only; bypasses backend.make_teacher.
        from src.model.api_teacher import ApiTopkTeacher
        teacher = ApiTopkTeacher(base_url=args.teacher_endpoint, vocab_size=manifest.vocab_size,
                                 tokenizer=manifest.tokenizer, model=args.endpoint_model)
        teacher_info = {"model_id": args.endpoint_model, "endpoint": args.teacher_endpoint,
                        "synthetic": False, "partial": "logit-distill only",
                        "vocab_size": teacher.config.vocab_size}
    elif args.synthetic:
        teacher = backend.make_teacher(config=TeacherConfig.tiny(), **mlx_opts)
        teacher_info = {"model_id": None, "synthetic": True,
                        "vocab_size": teacher.config.vocab_size}
    else:
        model_id = args.pretrained or manifest.conversion_teacher
        teacher = backend.make_teacher(config=_teacher_config_for(model_id), pretrained=model_id,
                                       **mlx_opts)
        teacher_info = {"model_id": model_id, "synthetic": False,
                        "vocab_size": teacher.config.vocab_size}
        # Guard the foot-gun: if the teacher emits logits over a wider vocab than the student's
        # tokenizer vocab (e.g. `from_pretrained(config=None)` builds a TeacherConfig from HF
        # config.json, which omits `tokenizer_vocab_size`, so effective_vocab includes padded
        # rows), the cached top-k indices can point at rows the student cannot represent. Fail
        # loudly instead of silently writing unusable artifacts.
        if teacher.config.effective_vocab_size != manifest.vocab_size:
            raise SystemExit(
                f"teacher effective_vocab_size {teacher.config.effective_vocab_size} != manifest "
                f"tokenizer vocab {manifest.vocab_size} ({manifest.tokenizer}); the student cannot "
                f"consume these top-k indices. Use a --pretrained id known to _teacher_config_for "
                f"(it sets tokenizer_vocab_size), or extend it — see TeacherConfig.qwen3_4b_thinking.")

    eff_vocab = teacher.config.effective_vocab_size
    base_out = Path(args.out or storage.teacher_outputs_dir(args.data))
    splits = [s for s in args.splits.split(",") if s]

    # Cluster mode: each pod owns one shard of the chunk space. Output goes to
    # <out>/shard-<id>/; --push suffix gets /shard-<id>/ appended automatically.
    shard_id = args.shard_id
    num_shards = args.num_shards
    if (shard_id is None) != (num_shards is None):
        raise SystemExit("--shard-id and --num-shards must be used together")
    cluster = shard_id is not None
    if cluster and not (0 <= shard_id < num_shards):
        raise SystemExit(f"--shard-id {shard_id} out of range [0, {num_shards})")
    out_dir = base_out / f"shard-{shard_id}" if cluster else base_out
    push = (f"{args.push.rstrip('/')}/shard-{shard_id}" if cluster and args.push
            else args.push)

    # The teacher clamps k to min(k, effective_vocab) in topk_logits, so the actual cached k can
    # be smaller than args.k; record THAT (from the per-split meta) in the manifest so it agrees
    # with the per-split meta files downstream consumers read.
    actual_k = args.k
    for split in splits:
        data = open_packed(args.data / f"{split}.bin")
        n_tokens = int(data.shape[0])
        stride = seq_len + 1
        total_chunks = n_tokens // stride
        if total_chunks == 0:
            raise ValueError(f"{split}.bin too small for one chunk (seq_len={seq_len})")

        # `val` is never shard-divided: shard 0 owns the whole val split (the launcher only
        # gives val to shard 0). Other shards skip it entirely. `train` IS sharded across pods.
        split_cluster = cluster and split != "val"
        if cluster and split == "val" and shard_id != 0:
            print(f"[shard {shard_id}] skipping val (owned by shard 0)", flush=True)
            continue
        if split_cluster:
            per_shard, remainder = divmod(total_chunks, num_shards)
            start_chunk = shard_id * per_shard + min(shard_id, remainder)
            this_chunks = per_shard + (1 if shard_id < remainder else 0)
        else:
            start_chunk, this_chunks = 0, total_chunks

        # The shard's TRUE starting position, for merge_teacher_shards.py's contiguity check
        # (0, then previous_start+previous_n_chunks, ...). Resume below advances `start_chunk`
        # to the actual resume point, which must not leak into this recorded value.
        shard_start_chunk = start_chunk

        # Auto-resume: if output files already exist (e.g. after a pod billing stop),
        # skip already-written chunks and append from the last complete chunk boundary.
        _chunks_resume = 0
        _idx_file = out_dir / f"teacher-{split}.topk_idx"
        if _idx_file.exists() and _idx_file.stat().st_size > 0:
            import os as _os
            # Use the teacher-clamped k (min(args.k, effective_vocab)), not args.k directly —
            # the writer records rows at this width, so byte math must match what was actually
            # written or chunk-boundary detection undercounts/overcounts on resume.
            _on_disk_k = min(args.k, eff_vocab)
            _IB = _on_disk_k * 4   # bytes per row in idx (uint32 × k)
            _VB = _on_disk_k * 2   # bytes per row in vals (float16 × k)
            _vf = out_dir / f"teacher-{split}.topk_vals"
            _isz = _idx_file.stat().st_size
            _vsz = _vf.stat().st_size if _vf.exists() else 0
            _chunks_resume = min(_isz // (seq_len * _IB), _vsz // (seq_len * _VB))
            if 0 < _chunks_resume < this_chunks:
                # Trim both files to the last complete chunk boundary using delete+rewrite
                # rather than os.truncate(): network volumes (RunPod /vol, MooseFS) may
                # return 0 from truncate() but leave the file unchanged, causing the
                # append mode below to write at the wrong offset.
                _prefix_idx = _chunks_resume * seq_len * _IB
                _prefix_vf  = _chunks_resume * seq_len * _VB
                for _path, _keep in ((_idx_file, _prefix_idx), (_vf, _prefix_vf)):
                    _tmp = _path.with_suffix(".topk_resume_tmp")
                    try:
                        with open(_path, "rb") as _src, open(_tmp, "wb") as _dst:
                            _rem = _keep
                            while _rem > 0:
                                _buf = _src.read(min(8 * 1024 * 1024, _rem))
                                if not _buf:
                                    break
                                _dst.write(_buf)
                                _rem -= len(_buf)
                        _os.replace(_tmp, _path)
                    except Exception as _e:
                        try:
                            _tmp.unlink(missing_ok=True)
                        except Exception:
                            pass
                        raise RuntimeError(
                            f"[resume] failed to trim {_path} to {_keep} bytes: {_e}. "
                            "Delete the partial output and restart from scratch."
                        ) from _e
                start_chunk += _chunks_resume
                this_chunks -= _chunks_resume
                print(f"[resume] {_chunks_resume} chunks already done; "
                      f"continuing at global chunk {start_chunk}", flush=True)
            elif _chunks_resume >= this_chunks:
                print(f"[resume] split={split} already complete ({_chunks_resume} chunks) — skipping",
                      flush=True)
                continue

        blocks = _topk_blocks(teacher, backend, data, this_chunks, stride, seq_len,
                              args.batch_size, args.k, start_chunk=start_chunk)
        shard_info = {"shard_id": shard_id, "num_shards": num_shards,
                      "start_chunk": shard_start_chunk} if cluster else {}
        meta = write_teacher_topk(out_dir, split, blocks=blocks, n_chunks=this_chunks,
                                  seq_len=seq_len, vocab_size=eff_vocab,
                                  src_packed=str(args.data / f"{split}.bin"),
                                  src_n_tokens=n_tokens, extra=shard_info,
                                  _append=(_chunks_resume > 0),
                                  _rows_done=_chunks_resume * seq_len)
        actual_k = meta["k"]
        shard_label = f" [shard {shard_id}/{num_shards}]" if cluster else ""
        print(f"teacher top-k [{split}]{shard_label}: {meta['n_rows']} rows x k={meta['k']} "
              f"({this_chunks} chunks, start={start_chunk}) -> {out_dir}")

    write_manifest(out_dir, k=actual_k, seq_len=seq_len, effective_vocab_size=eff_vocab,
                   corpus_manifest=str(args.data), teacher=teacher_info, splits=splits)

    if push:
        from src.data.r2_sync import upload_dir
        written = upload_dir(out_dir, push)
        print(f"pushed {len(written)} file(s): {out_dir} -> {push}")


if __name__ == "__main__":
    main()
