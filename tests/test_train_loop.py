"""Portable training-loop orchestration: grad accumulation, logging, checkpoint, resume.

Uses fakes (no backend) so these run on any host. The loop's contract is exercised
through an injected `train_step` that records its calls.
"""

from src.train.loop import TrainConfig, train


class FakeLoader:
    """Minimal stand-in for PackedLoader: fixed batches, exposes batch_size/seq_len."""

    def __init__(self, n_batches: int, batch_size: int = 4, seq_len: int = 8):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_batches = n_batches

    def epoch(self, reseed=None):
        for i in range(self.n_batches):
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
