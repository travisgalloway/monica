"""Offline end-to-end test of the teacher top-k precompute driver (#94).

Builds a tiny byte-vocab packed split, runs `scripts/precompute_teacher.py --synthetic` as a
subprocess (the real CLI), then asserts the cached files line up with the corpus and that a
student trial reads them back with ZERO teacher inference (AC#3).
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.data.pack import pack_ids
from src.data.teacher_outputs import DistillLoader, read_teacher_meta

REPO = Path(__file__).resolve().parents[1]
VOCAB = 256
SEQ_LEN = 4


def _backend():
    """Pick an available backend for the synthetic teacher (mlx preferred, else torch/cuda-CPU)."""
    try:
        import mlx.core  # noqa: F401
        return "mlx"
    except ImportError:
        pass
    try:
        import torch  # noqa: F401
        return "cuda"
    except ImportError:
        pytest.skip("no backend (mlx/torch) available for the synthetic teacher")


def _write_manifest(path: Path):
    path.write_text(
        "student: toy\n"
        "conversion_teacher: synthetic\n"
        "tokenizer: qwen25\n"
        f"seq_len: {SEQ_LEN}\n"
        "init: mohawk\n"
        "stages: [logit-distill]\n"
        "layout: {d_model: 16, n_layers: 2}\n")


def _build_split(data_dir: Path, n_chunks: int, val_chunks: int = 2):
    data_dir.mkdir(parents=True, exist_ok=True)
    stride = SEQ_LEN + 1
    for split, n in (("train", n_chunks), ("val", val_chunks)):
        ids = (np.arange(n * stride + 1) % VOCAB).astype(np.uint16)
        pack_ids(ids, data_dir / f"{split}.bin", dtype=np.uint16)


def test_precompute_cli_offline(tmp_path):
    backend = _backend()
    data = tmp_path / "split"
    out = tmp_path / "teacher-outputs"
    n_chunks = 5
    _build_split(data, n_chunks)
    manifest = tmp_path / "toy.yaml"
    _write_manifest(manifest)

    res = subprocess.run(
        [sys.executable, "scripts/precompute_teacher.py",
         "--manifest", str(manifest), "--data", str(data), "--out", str(out),
         "--backend", backend, "--k", "8", "--batch-size", "2", "--synthetic"],
        cwd=REPO, capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO), "PATH": __import__("os").environ.get("PATH", "")},
    )
    assert res.returncode == 0, f"precompute failed:\n{res.stdout}\n{res.stderr}"

    # per-split meta aligned to the packed corpus
    meta = read_teacher_meta(out, "train")
    assert meta["n_chunks"] == n_chunks and meta["seq_len"] == SEQ_LEN
    assert meta["n_rows"] == n_chunks * SEQ_LEN and meta["k"] == 8
    manifest_json = json.loads((out / "manifest.json").read_text())
    assert manifest_json["splits"] == ["train", "val"]
    assert manifest_json["teacher"]["synthetic"] is True

    # AC#3: a student trial reads the cached signal with NO teacher (zero teacher inference).
    loader = DistillLoader(data / "train.bin", out, "train", SEQ_LEN, batch_size=2, shuffle=False)
    inputs, targets, vals, idx = next(iter(loader.epoch()))
    assert inputs.shape == (2, SEQ_LEN) and vals.shape == (2, SEQ_LEN, 8)
    assert idx.min() >= 0 and idx.max() < VOCAB


def test_precompute_val_full_on_shard0(tmp_path):
    """(#174) `val` must never be shard-divided in cluster mode: shard 0 owns the full val
    split (the launcher only assigns val to shard 0), while `train` IS sharded across pods."""
    backend = _backend()
    data = tmp_path / "split"
    out = tmp_path / "teacher-outputs"
    n_chunks = 8
    val_chunks = 6
    _build_split(data, n_chunks, val_chunks=val_chunks)
    manifest = tmp_path / "toy.yaml"
    _write_manifest(manifest)

    res = subprocess.run(
        [sys.executable, "scripts/precompute_teacher.py",
         "--manifest", str(manifest), "--data", str(data), "--out", str(out),
         "--backend", backend, "--k", "8", "--batch-size", "2", "--synthetic",
         "--shard-id", "0", "--num-shards", "4"],
        cwd=REPO, capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO), "PATH": __import__("os").environ.get("PATH", "")},
    )
    assert res.returncode == 0, f"precompute failed:\n{res.stdout}\n{res.stderr}"

    shard0 = out / "shard-0"
    val_meta = read_teacher_meta(shard0, "val")
    assert val_meta["n_chunks"] == val_chunks, (
        f"val must be processed in full on shard 0, not sliced by num_shards "
        f"(got {val_meta['n_chunks']}, expected {val_chunks})")
    train_meta = read_teacher_meta(shard0, "train")
    assert train_meta["n_chunks"] == n_chunks // 4, "train IS still sharded across pods"
