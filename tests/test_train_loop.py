"""Portable training-loop orchestration: grad accumulation, logging, checkpoint, resume.

Uses fakes (no backend) so these run on any host. The loop's contract is exercised
through an injected `train_step` that records its calls.
"""

import numpy as np

from src.data.loader import PackedLoader
from src.data.pack import pack_ids
from src.train.loop import TrainConfig, train, _micro_batch_stream


class FakeLoader:
    """Minimal stand-in for PackedLoader: fixed batches, exposes batch_size/seq_len."""

    def __init__(self, n_batches: int, batch_size: int = 4, seq_len: int = 8):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_batches = n_batches

    def __len__(self):
        return self.n_batches

    def epoch(self, reseed=None, skip_batches=0):
        for i in range(skip_batches, self.n_batches):
            yield (("inp", i), ("tgt", i))


def test_grad_accum_passes_microbatches_and_steps_once():
    calls = []

    def fake_step(model, micro, lr):
        calls.append(len(micro))
        return {"loss": 1.0, "grad_norm": 0.5}

    loader = FakeLoader(n_batches=100)
    cfg = TrainConfig(total_steps=5, grad_accum=3, warmup_steps=0, log_every=1,
                      eval_every=100, ckpt_every=100)
    logs = []
    train(None, loader, cfg, fake_step, logger=logs.append)

    assert calls == [3, 3, 3, 3, 3]      # grad_accum micro-batches per optimizer step
    assert len(logs) == 5                # one step per call
    assert [p["step"] for p in logs] == [0, 1, 2, 3, 4]


def test_val_dict_merged_and_tokens_per_sec_present():
    def fake_step(model, micro, lr):
        return {"loss": 2.0, "grad_norm": 0.1}

    loader = FakeLoader(n_batches=100, batch_size=4, seq_len=8)
    cfg = TrainConfig(total_steps=4, grad_accum=1, warmup_steps=0, log_every=1,
                      eval_every=2, ckpt_every=100)
    logs = []
    val = lambda m: {"val_loss": 1.5, "val_perplexity": 4.48}
    train(None, loader, cfg, fake_step, val_eval=val, logger=logs.append)

    assert all("tokens_per_sec" in p for p in logs)
    evald = [p for p in logs if "val_perplexity" in p]
    assert {p["step"] for p in evald} == {0, 2}     # merged only at eval_every steps
    assert evald[0]["val_loss"] == 1.5


def test_checkpoint_fires_at_interval():
    ckpts = []

    def fake_step(model, micro, lr):
        return {"loss": 1.0, "grad_norm": 0.0}

    loader = FakeLoader(n_batches=100)
    cfg = TrainConfig(total_steps=10, grad_accum=1, warmup_steps=0, log_every=100,
                      eval_every=100, ckpt_every=3)
    train(None, loader, cfg, fake_step, on_checkpoint=ckpts.append)

    assert ckpts == [3, 6, 9]            # post-increment step hits the cadence


def test_start_step_resume_runs_only_remaining():
    calls = []

    def fake_step(model, micro, lr):
        calls.append(lr)
        return {"loss": 1.0, "grad_norm": 0.0}

    loader = FakeLoader(n_batches=100)
    cfg = TrainConfig(total_steps=10, grad_accum=1, warmup_steps=0, log_every=100,
                      eval_every=100, ckpt_every=100)
    train(None, loader, cfg, fake_step, start_step=7)

    assert len(calls) == 3               # steps 7, 8, 9 only


def _hashable(batch):
    inp, tgt = batch
    return (inp.tobytes(), tgt.tobytes())


def test_resume_stream_continues_data_not_restart(tmp_path):
    """P0.1 regression: resuming at step K must yield the SAME data the uninterrupted
    run would see at steps K.., not replay the corpus from the top. Pure numpy — runs
    without a backend. Uses distinct token values so every batch is identifiable, and a
    small enough loader that the resume point lands well past the first epoch boundary.
    """
    seq_len, batch_size, grad_accum, seed = 4, 2, 1, 123
    n_chunks = 20                                  # per_epoch = 20 // 2 = 10 micro-batches
    n_tokens = n_chunks * (seq_len + 1)
    packed = tmp_path / "train.bin"
    pack_ids(np.arange(n_tokens, dtype=np.uint16), packed, dtype=np.uint16)

    def fresh_loader():
        return PackedLoader(packed, seq_len, batch_size, shuffle=True, seed=seed)

    total_micro = 25                               # spans >2 epochs (10 micro/epoch)
    full = [_hashable(b) for b in
            _itertools_take(_micro_batch_stream(fresh_loader(), seed, start_micro=0),
                            total_micro)]

    resume_k = 13                                  # mid second epoch
    resumed = [_hashable(b) for b in
               _itertools_take(_micro_batch_stream(fresh_loader(), seed,
                                                   start_micro=resume_k),
                               total_micro - resume_k)]

    assert resumed == full[resume_k:], "resumed stream diverged from the uninterrupted one"
    # And sanity: the stream is NOT just repeating epoch 0 (would make the test vacuous).
    assert full[:10] != full[10:20]


def _itertools_take(it, n):
    out = []
    for x in it:
        out.append(x)
        if len(out) == n:
            break
    return out
