"""Portable training-loop orchestration: grad accumulation, logging, checkpoint, resume.

Uses fakes (no backend) so these run on any host. The loop's contract is exercised
through an injected `train_step` that records its calls.
"""

import numpy as np
import pytest

from src.data.loader import PackedLoader
from src.data.pack import pack_ids
from src.train.curriculum import LengthCurriculum, Stage
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


def test_moe_diag_fires_only_on_its_cadence_and_merges_into_the_payload():
    """#217: `moe_diag` is merged like `val_eval`, but on its own `moe_diag_every`
    cadence (independent of `eval_every`)."""
    def fake_step(model, micro, lr):
        return {"loss": 1.0, "grad_norm": 0.5}

    calls = []

    def moe_diag(model):
        calls.append(model)
        return {"moe_domain_overlap": 0.5, "moe_kill_triggered": False}

    loader = FakeLoader(n_batches=100)
    cfg = TrainConfig(total_steps=6, grad_accum=1, warmup_steps=0, log_every=1,
                      eval_every=100, ckpt_every=100, moe_diag_every=2)
    logs = []
    train("the-model", loader, cfg, fake_step, moe_diag=moe_diag, logger=logs.append)

    diagged = [p for p in logs if "moe_domain_overlap" in p]
    assert {p["step"] for p in diagged} == {0, 2, 4}
    assert all(p["moe_kill_triggered"] is False for p in diagged)
    assert calls == ["the-model"] * 3           # moe_diag(model) called with the real model


def test_moe_diag_every_zero_never_fires_and_does_not_raise():
    def fake_step(model, micro, lr):
        return {"loss": 1.0, "grad_norm": 0.5}

    def moe_diag(model):
        raise AssertionError("moe_diag must not be called when moe_diag_every == 0")

    loader = FakeLoader(n_batches=100)
    cfg = TrainConfig(total_steps=4, grad_accum=1, warmup_steps=0, log_every=1,
                      eval_every=100, ckpt_every=100, moe_diag_every=0)   # default: off
    logs = []
    train(None, loader, cfg, fake_step, moe_diag=moe_diag, logger=logs.append)   # no raise

    assert not any("moe_domain_overlap" in p for p in logs)


def test_moe_diag_none_with_nonzero_every_is_a_noop():
    def fake_step(model, micro, lr):
        return {"loss": 1.0, "grad_norm": 0.5}

    loader = FakeLoader(n_batches=100)
    cfg = TrainConfig(total_steps=4, grad_accum=1, warmup_steps=0, log_every=1,
                      eval_every=100, ckpt_every=100, moe_diag_every=2)
    logs = []
    train(None, loader, cfg, fake_step, moe_diag=None, logger=logs.append)   # no raise

    assert not any("moe_domain_overlap" in p for p in logs)


def test_checkpoint_fires_at_interval():
    ckpts = []

    def fake_step(model, micro, lr):
        return {"loss": 1.0, "grad_norm": 0.0}

    loader = FakeLoader(n_batches=100)
    cfg = TrainConfig(total_steps=10, grad_accum=1, warmup_steps=0, log_every=100,
                      eval_every=100, ckpt_every=3)
    train(None, loader, cfg, fake_step,
          on_checkpoint=lambda step, data_state: ckpts.append((step, data_state)))

    assert [s for s, _ in ckpts] == [3, 6, 9]   # post-increment step hits the cadence
    # #216: every checkpoint carries a usable dataloader position, and it is positioned
    # at the START of the step it is filed under (step*grad_accum micro-batches consumed).
    for step, state in ckpts:
        assert state is not None and state["global_micro"] == step * cfg.grad_accum


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


def test_resume_via_explicit_data_state_continues_the_stream(tmp_path):
    """The #216 form of the P0.1 regression above: the same property, but carried by an
    EXPLICIT `data_state` rather than reconstructed from the step. This is the path a
    curriculum run takes, and the one that must hold when `len(loader)` is no longer a
    constant of the run."""
    seq_len, batch_size, grad_accum, seed = 4, 2, 1, 123
    n_chunks = 20                                  # per_epoch = 10 micro-batches
    packed = tmp_path / "train.bin"
    pack_ids(np.arange(n_chunks * (seq_len + 1), dtype=np.uint16), packed,
             dtype=np.uint16)
    factory = lambda sl, bs: PackedLoader(packed, sl, bs, shuffle=True, seed=seed)

    def recorder(store):
        def fake_step(model, micro, lr):
            store.extend(_hashable(b) for b in micro)
            return {"loss": 1.0, "grad_norm": 0.0}
        return fake_step

    cfg = TrainConfig(total_steps=25, grad_accum=grad_accum, warmup_steps=0,
                      log_every=100, eval_every=100, ckpt_every=13, seed=seed)
    loader = factory(seq_len, batch_size)

    full = []
    train(None, loader, cfg, recorder(full))

    saved = {}
    part = []
    cut = TrainConfig(**{**cfg.__dict__, "total_steps": 13})
    train(None, loader, cut, recorder(part),
          on_checkpoint=lambda step, ds: saved.update(step=step, data_state=ds))
    assert saved["data_state"] is not None

    rest = []
    train(None, factory(seq_len, batch_size), cfg, recorder(rest),
          start_step=saved["step"], start_data_state=saved["data_state"])

    assert part == full[:13]
    assert rest == full[13:], "resume from explicit data_state diverged"
    assert full[:10] != full[10:20], "test is vacuous: the stream never crossed an epoch"


def _two_stage_setup(tmp_path, steps_a=4, steps_b=6, grad_accum=1, seed=9):
    """A real two-stage curriculum over a packed corpus, with a matching loader factory."""
    packed = tmp_path / "train.bin"
    pack_ids(np.arange(600, dtype=np.uint16), packed, dtype=np.uint16)
    curriculum = LengthCurriculum(stages=(
        Stage(index=0, until_frac=0.5, seq_len=8, batch_size=4, steps=steps_a),
        Stage(index=1, until_frac=1.0, seq_len=16, batch_size=2, steps=steps_b),
    ), grad_accum=grad_accum)
    factory = lambda sl, bs: PackedLoader(packed, sl, bs, shuffle=True, seed=seed)
    return curriculum, factory


def test_resume_across_curriculum_boundary(tmp_path):
    """THE acceptance test, through the real loop: kill strictly AFTER a stage boundary
    and resume from the saved `data_state`; every micro-batch must match the
    uninterrupted run byte for byte."""
    grad_accum, seed = 2, 9
    curriculum, factory = _two_stage_setup(tmp_path, steps_a=4, steps_b=6,
                                           grad_accum=grad_accum, seed=seed)
    total = curriculum.total_steps                             # 10 steps, boundary at 4

    def recorder(store):
        def fake_step(model, micro, lr):
            store.extend((b[0].shape, _hashable(b)) for b in micro)
            return {"loss": 1.0, "grad_norm": 0.0}
        return fake_step

    def cfg_for(steps, ckpt_every=1000):
        return TrainConfig(total_steps=steps, grad_accum=grad_accum, warmup_steps=0,
                           log_every=100, eval_every=1000, ckpt_every=ckpt_every,
                           seed=seed)

    full = []
    train(None, None, cfg_for(total), recorder(full), curriculum=curriculum,
          loader_factory=factory)

    # Kill the way a preemptible pod does: mid-run, at a checkpoint, with the run's real
    # total_steps intact — NOT by shortening the budget (that would move the boundary).
    class Killed(Exception):
        pass

    kill = 6                                                   # boundary(4) + 2
    saved, part = {}, []

    def kill_at(step, data_state):
        saved.update(step=step, data_state=data_state)
        raise Killed

    with pytest.raises(Killed):
        train(None, None, cfg_for(total, ckpt_every=kill), recorder(part),
              curriculum=curriculum, loader_factory=factory, on_checkpoint=kill_at)

    rest = []
    train(None, None, cfg_for(total), recorder(rest), curriculum=curriculum,
          loader_factory=factory, start_step=saved["step"],
          start_data_state=saved["data_state"])

    assert part == full[:kill * grad_accum]
    assert rest == full[kill * grad_accum:], "resume across the boundary diverged"
    # Not vacuous: the shape really did change at the boundary.
    shapes = [s for s, _ in full]
    assert shapes[:4 * grad_accum] == [(4, 8)] * (4 * grad_accum)
    assert shapes[4 * grad_accum:] == [(2, 16)] * (6 * grad_accum)


def test_curriculum_payload_carries_stage_seq_len_and_cumulative_tokens(tmp_path):
    curriculum, factory = _two_stage_setup(tmp_path, steps_a=4, steps_b=6)
    cfg = TrainConfig(total_steps=curriculum.total_steps, grad_accum=1, warmup_steps=0,
                      log_every=1, eval_every=1000, ckpt_every=1000, seed=9)
    logs = []
    result = train(None, None, cfg, lambda m, mb, lr: {"loss": 1.0}, logger=logs.append,
                   curriculum=curriculum, loader_factory=factory)

    steps = [p for p in logs if "event" not in p]
    assert [p["seq_len"] for p in steps] == [8] * 4 + [16] * 6
    assert [p["stage"] for p in steps] == [0] * 4 + [1] * 6
    # tokens is cumulative over the REAL per-stage tokens/step, not steps * a constant.
    assert [p["tokens"] for p in steps] == [
        32, 64, 96, 128, 160, 192, 224, 256, 288, 320]
    assert result["tokens_seen"] == 4 * 32 + 6 * 32
    assert set(result) == {"step", "data_state", "tokens_seen"}
    assert result["step"] == curriculum.total_steps


def test_curriculum_logs_one_stage_event_per_boundary(tmp_path):
    """The anti-silent-regression measure: every CUDA recompile is preceded by a visible,
    timestamped line, so a stall at a boundary is attributable."""
    curriculum, factory = _two_stage_setup(tmp_path, steps_a=3, steps_b=3)
    cfg = TrainConfig(total_steps=6, grad_accum=1, warmup_steps=0, log_every=1,
                      eval_every=1000, ckpt_every=1000, seed=9)
    logs = []
    train(None, None, cfg, lambda m, mb, lr: {"loss": 1.0}, logger=logs.append,
          curriculum=curriculum, loader_factory=factory)

    events = [p for p in logs if p.get("event") == "stage"]
    assert [(e["step"], e["stage"], e["seq_len"]) for e in events] == [(0, 0, 8), (3, 1, 16)]


def test_no_curriculum_emits_no_stage_events():
    """A single-stage run has no boundary; the extra payload would be noise in every
    existing pretrain/SFT/DPO log."""
    cfg = TrainConfig(total_steps=4, grad_accum=1, warmup_steps=0, log_every=1,
                      eval_every=100, ckpt_every=100)
    logs = []
    train(None, FakeLoader(n_batches=100), cfg, lambda m, mb, lr: {"loss": 1.0},
          logger=logs.append)
    assert not [p for p in logs if "event" in p]


def test_curriculum_total_steps_must_match_cfg(tmp_path):
    curriculum, factory = _two_stage_setup(tmp_path, steps_a=4, steps_b=6)
    cfg = TrainConfig(total_steps=99, grad_accum=1, warmup_steps=0, log_every=100,
                      eval_every=100, ckpt_every=100)
    with pytest.raises(ValueError, match=r"curriculum.total_steps \(10\) != cfg"):
        train(None, None, cfg, lambda m, mb, lr: {"loss": 1.0}, curriculum=curriculum,
              loader_factory=factory)


def test_curriculum_requires_a_loader_factory(tmp_path):
    curriculum, _ = _two_stage_setup(tmp_path)
    cfg = TrainConfig(total_steps=curriculum.total_steps, grad_accum=1, warmup_steps=0)
    with pytest.raises(ValueError, match="needs a loader_factory"):
        train(None, None, cfg, lambda m, mb, lr: {"loss": 1.0}, curriculum=curriculum)


def test_resume_with_a_stale_stage_index_raises(tmp_path):
    """Edge case 4: resuming with a changed budget can move the stage boundary out from
    under the saved position. That must be loud and name the escape hatch, never a silent
    resample."""
    curriculum, factory = _two_stage_setup(tmp_path, steps_a=4, steps_b=6)
    cfg = TrainConfig(total_steps=curriculum.total_steps, grad_accum=1, warmup_steps=0,
                      log_every=100, eval_every=100, ckpt_every=6, seed=9)
    saved = {}
    train(None, None, cfg, lambda m, mb, lr: {"loss": 1.0}, curriculum=curriculum,
          loader_factory=factory,
          on_checkpoint=lambda step, ds: saved.update(step=step, data_state=ds))
    assert saved["data_state"]["stage_idx"] == 1        # step 6 is inside stage 2

    # Same SHAPES (so the fingerprint still matches) but a later boundary: step 6 now
    # belongs to stage 0.
    moved, _ = _two_stage_setup(tmp_path, steps_a=8, steps_b=2)
    cfg2 = TrainConfig(**{**cfg.__dict__, "total_steps": moved.total_steps})
    with pytest.raises(ValueError, match="--ignore-data-state"):
        train(None, None, cfg2, lambda m, mb, lr: {"loss": 1.0}, curriculum=moved,
              loader_factory=factory, start_step=saved["step"],
              start_data_state=saved["data_state"])


def test_ignore_data_state_falls_back_to_step_reconstruction(tmp_path):
    """The escape hatch reconstructs from the step instead of raising — and on an
    UNCHANGED curriculum it reproduces the same stream, so it is a safe fallback."""
    curriculum, factory = _two_stage_setup(tmp_path, steps_a=4, steps_b=6)

    def recorder(store):
        def fake_step(model, micro, lr):
            store.extend(_hashable(b) for b in micro)
            return {"loss": 1.0}
        return fake_step

    cfg = TrainConfig(total_steps=curriculum.total_steps, grad_accum=1, warmup_steps=0,
                      log_every=100, eval_every=100, ckpt_every=1000, seed=9)
    full = []
    train(None, None, cfg, recorder(full), curriculum=curriculum, loader_factory=factory)

    rest = []
    train(None, None, cfg, recorder(rest), curriculum=curriculum, loader_factory=factory,
          start_step=6, start_data_state={"version": 1, "bogus": True},
          ignore_data_state=True)
    assert rest == full[6:]


def test_legacy_bundle_without_data_state_reconstructs_from_step():
    """Edge case 5: a pre-#216 resume bundle has no `data_state` at all."""
    calls = []
    cfg = TrainConfig(total_steps=10, grad_accum=1, warmup_steps=0, log_every=100,
                      eval_every=100, ckpt_every=100)
    train(None, FakeLoader(n_batches=100), cfg,
          lambda m, mb, lr: calls.append(lr) or {"loss": 1.0},
          start_step=7, start_data_state=None)
    assert len(calls) == 3


def _itertools_take(it, n):
    out = []
    for x in it:
        out.append(x)
        if len(out) == n:
            break
    return out
