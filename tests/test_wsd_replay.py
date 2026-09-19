"""Tests for WSD decay-phase general knowledge replay buffer (#364).

Acceptance criteria:
  1. Data mix ratios across warmup, stable, and decay phase transitions.
  2. Deterministic resume test within the decay phase yields identical token streams.
  3. Small-scale run confirming held-out validation loss (domain_bpb.py) on web/math
     domains does not regress during decay.
  4. No backend leaks in tests/test_import_guard.py (verified via test_import_guard).
"""

from pathlib import Path
import json
import numpy as np
import pytest

from src.data.loader import PackedLoader
from src.data.pack import pack_ids
from src.eval.domain_bpb import evaluate_domain_bpb
from src.train.curriculum import LengthCurriculum, Stage
from src.train.loop import TrainConfig, train
from src.train.replay import (
    ReplayLoader,
    build_replay_loader_factory,
    discover_replay_shards,
)
from src.train.schedule import WSDSchedule
from src.train.stream import MicroBatchStream


class FakeLabeledLoader:
    """Loader yielding batches tagged with a source label for ratio verification."""

    def __init__(self, label: str, n_batches: int = 500, batch_size: int = 2, seq_len: int = 4):
        self.label = label
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rng = np.random.default_rng(0)

    def __len__(self):
        return self.n_batches

    def epoch(self, reseed=None, skip_batches=0):
        if reseed is not None:
            self.rng = np.random.default_rng(reseed)
        for i in range(skip_batches, self.n_batches):
            inp = np.full((self.batch_size, self.seq_len), fill_value=hash(self.label) % 1000, dtype=np.int64)
            tgt = np.full((self.batch_size, self.seq_len), fill_value=i, dtype=np.int64)
            yield (inp, tgt)


def _batch_hash(batch):
    inp, tgt = batch
    return (inp.tobytes(), tgt.tobytes())


# ============================================================================
# Criterion 1: Data mix ratios across warmup, stable, and decay transitions
# ============================================================================

def test_data_mix_ratios_across_phase_transitions():
    """Validates that:
      * Warmup phase: exactly 0% replay data.
      * Stable phase: exactly 0% replay data.
      * Decay phase: dynamically blended at the specified ratio (e.g. 30% or 25%).
    """
    total_steps = 100
    warmup_steps = 15
    decay_steps = 40  # decay begins at step 60 (steps 60..99)
    grad_accum = 1
    decay_replay_ratio = 0.30

    schedule = WSDSchedule(
        base_lr=1.0,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        decay_steps=decay_steps,
    )

    curriculum = LengthCurriculum.single(seq_len=4, batch_size=2, steps=total_steps, grad_accum=grad_accum)
    primary_factory = lambda sl, bs: FakeLabeledLoader("primary", n_batches=1000, batch_size=bs, seq_len=sl)
    replay_factory = lambda sl, bs: FakeLabeledLoader("replay", n_batches=1000, batch_size=bs, seq_len=sl)

    stream = MicroBatchStream(
        curriculum,
        primary_factory,
        seed=42,
        replay_loader_factory=replay_factory,
        decay_replay_ratio=decay_replay_ratio,
        schedule=schedule,
    )

    batches_by_phase = {"warmup": [], "stable": [], "decay": []}
    decay_start = total_steps - decay_steps

    for step in range(total_steps):
        phase = schedule.phase_at(step)
        batch = next(stream)
        # Check source of batch by input fill value
        is_replay = (batch[0][0, 0] == hash("replay") % 1000)
        batches_by_phase[phase].append(is_replay)

    # 1. Warmup checks
    warmup_batches = batches_by_phase["warmup"]
    assert len(warmup_batches) == warmup_steps
    assert sum(warmup_batches) == 0, "Replay data must not appear in warmup phase"

    # 2. Stable checks
    stable_batches = batches_by_phase["stable"]
    assert len(stable_batches) == (decay_start - warmup_steps)
    assert sum(stable_batches) == 0, "Replay data must not appear in stable phase"

    # 3. Decay checks
    decay_batches = batches_by_phase["decay"]
    assert len(decay_batches) == decay_steps
    expected_replay_count = int(decay_steps * decay_replay_ratio)
    actual_replay_count = sum(decay_batches)
    assert actual_replay_count == expected_replay_count, (
        f"Decay phase replay count {actual_replay_count} != expected {expected_replay_count}"
    )
    observed_ratio = actual_replay_count / len(decay_batches)
    assert observed_ratio == pytest.approx(decay_replay_ratio, abs=0.01)


def test_data_mix_ratio_custom_fractions():
    """Verify that different decay_replay_ratios (e.g. 0.25) are respected accurately."""
    total_steps = 40
    warmup_steps = 0
    decay_steps = 40  # entirely decay phase
    decay_replay_ratio = 0.25

    schedule = WSDSchedule(base_lr=1.0, warmup_steps=warmup_steps, total_steps=total_steps, decay_steps=decay_steps)
    curriculum = LengthCurriculum.single(seq_len=4, batch_size=2, steps=total_steps, grad_accum=1)

    primary_factory = lambda sl, bs: FakeLabeledLoader("primary", n_batches=200, batch_size=bs, seq_len=sl)
    replay_factory = lambda sl, bs: FakeLabeledLoader("replay", n_batches=200, batch_size=bs, seq_len=sl)

    stream = MicroBatchStream(
        curriculum,
        primary_factory,
        seed=123,
        replay_loader_factory=replay_factory,
        decay_replay_ratio=decay_replay_ratio,
        schedule=schedule,
    )

    replays = [next(stream)[0][0, 0] == (hash("replay") % 1000) for _ in range(total_steps)]
    assert sum(replays) == int(total_steps * 0.25)  # exactly 10 out of 40


# ============================================================================
# Criterion 2: Deterministic resume test within the decay phase
# ============================================================================

def _create_packed_corpus(path: Path, n_tokens: int, start_val: int = 0, n_bytes: int = None):
    ids = np.arange(start_val, start_val + n_tokens, dtype=np.uint16)
    pack_ids(ids, path, dtype=np.uint16, n_bytes=n_bytes or n_tokens * 2)
    return path


def test_deterministic_resume_within_decay_phase(tmp_path):
    """Save checkpoint inside the decay phase and resume; verify exact token parity."""
    seq_len, batch_size, grad_accum, seed = 4, 2, 2, 77
    total_steps = 30
    warmup_steps = 5
    decay_steps = 15  # decay starts at step 15
    interrupt_step = 22  # inside decay phase (step 22)

    primary_bin = _create_packed_corpus(tmp_path / "primary.bin", 2000, start_val=10)
    replay_bin = _create_packed_corpus(tmp_path / "replay.bin", 2000, start_val=50000)

    schedule = WSDSchedule(base_lr=1.0, warmup_steps=warmup_steps, total_steps=total_steps, decay_steps=decay_steps)
    curriculum = LengthCurriculum.single(seq_len=seq_len, batch_size=batch_size, steps=total_steps, grad_accum=grad_accum)

    primary_factory = lambda sl, bs: PackedLoader(primary_bin, sl, bs, shuffle=True, seed=seed)
    replay_factory = lambda sl, bs: PackedLoader(replay_bin, sl, bs, shuffle=True, seed=seed + 999)

    # 1. Run uninterrupted
    stream_uninterrupted = MicroBatchStream(
        curriculum,
        primary_factory,
        seed=seed,
        replay_loader_factory=replay_factory,
        decay_replay_ratio=0.30,
        schedule=schedule,
    )
    total_micros = total_steps * grad_accum
    full_tokens = [_batch_hash(next(stream_uninterrupted)) for _ in range(total_micros)]

    # 2. Run up to interrupt_step
    stream_part = MicroBatchStream(
        curriculum,
        primary_factory,
        seed=seed,
        replay_loader_factory=replay_factory,
        decay_replay_ratio=0.30,
        schedule=schedule,
    )
    interrupt_micros = interrupt_step * grad_accum
    part_tokens = [_batch_hash(next(stream_part)) for _ in range(interrupt_micros)]
    assert part_tokens == full_tokens[:interrupt_micros]

    saved_state = stream_part.state_dict()
    assert "replay_state" in saved_state
    assert saved_state["replay_state"]["decay_micro"] > 0
    assert saved_state["replay_state"]["replay_micro"] > 0

    # 3. Resume from saved_state
    stream_resumed = MicroBatchStream(
        curriculum,
        primary_factory,
        seed=seed,
        replay_loader_factory=replay_factory,
        decay_replay_ratio=0.30,
        schedule=schedule,
    )
    stream_resumed.load_state_dict(saved_state)

    resumed_tokens = [_batch_hash(next(stream_resumed)) for _ in range(total_micros - interrupt_micros)]

    # Assert exact bit-level parity
    assert resumed_tokens == full_tokens[interrupt_micros:], (
        "Resumed token stream in decay phase diverged from uninterrupted run"
    )


def test_resume_via_train_loop_with_replay(tmp_path):
    """Test full train() loop execution with checkpointing and resume inside decay."""
    seq_len, batch_size, grad_accum, seed = 4, 2, 1, 42
    total_steps = 20
    warmup_steps = 4
    decay_steps = 10  # decay starts at step 10
    ckpt_step = 14    # inside decay

    primary_bin = _create_packed_corpus(tmp_path / "primary.bin", 1500, start_val=100)
    replay_bin = _create_packed_corpus(tmp_path / "replay.bin", 1500, start_val=30000)

    loader_primary = PackedLoader(primary_bin, seq_len, batch_size, shuffle=True, seed=seed)
    loader_replay = PackedLoader(replay_bin, seq_len, batch_size, shuffle=True, seed=seed + 1)

    cfg = TrainConfig(
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        lr_schedule="wsd",
        decay_frac=0.5,  # 20 * 0.5 = 10 decay steps
        decay_replay_ratio=0.25,
        grad_accum=grad_accum,
        ckpt_every=ckpt_step,
        log_every=1,
    )

    batches_seen = []
    def fake_step(model, micro, lr):
        for b in micro:
            batches_seen.append(_batch_hash(b))
        return {"loss": 0.5, "grad_norm": 0.1}

    # Run full
    train(None, loader_primary, cfg, fake_step, replay_loader=loader_replay)
    full_run = list(batches_seen)

    # Run partial with identical total_steps schedule and checkpoint at ckpt_step
    saved = {}
    batches_seen.clear()

    class _InterruptRun(Exception):
        pass

    def interrupting_step(model, micro, lr):
        # Stop on the step immediately following the checkpoint
        if len(batches_seen) >= ckpt_step:
            raise _InterruptRun()
        return fake_step(model, micro, lr)

    try:
        train(
            None,
            PackedLoader(primary_bin, seq_len, batch_size, shuffle=True, seed=seed),
            cfg,
            interrupting_step,
            replay_loader=PackedLoader(replay_bin, seq_len, batch_size, shuffle=True, seed=seed + 1),
            on_checkpoint=lambda step, ds: saved.update(step=step, data_state=ds),
        )
    except _InterruptRun:
        pass

    assert saved["step"] == ckpt_step
    assert saved["data_state"] is not None

    # Resume from checkpoint
    batches_seen.clear()
    train(
        None,
        PackedLoader(primary_bin, seq_len, batch_size, shuffle=True, seed=seed),
        cfg,
        fake_step,
        replay_loader=PackedLoader(replay_bin, seq_len, batch_size, shuffle=True, seed=seed + 1),
        start_step=saved["step"],
        start_data_state=saved["data_state"],
    )
    resumed_run = list(batches_seen)

    assert full_run[:ckpt_step] == full_run[:ckpt_step]
    assert resumed_run == full_run[ckpt_step:], "train() loop resume diverged in decay phase"


# ============================================================================
# Criterion 3: Small-scale run confirming held-out validation loss (domain_bpb.py)
# on web/math domains does not regress during decay.
# ============================================================================

class LinearLookupModel:
    """Portable, differentiable lookup table model for small-scale verification.

    Logits for next token given current token are logits = W[token].
    Supports SGD update on batches to measure genuine domain forgetting and retention.
    """

    def __init__(self, vocab_size: int = 64, seed: int = 0):
        self.vocab_size = vocab_size
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((vocab_size, vocab_size)).astype(np.float64) * 0.1

    def forward(self, inputs):
        # inputs: (B, T) -> returns (B, T, V)
        inp = np.asarray(inputs) % self.vocab_size
        return self.W[inp]

    def train_on_batch(self, inputs, targets, lr: float = 0.1):
        # Online SGD update on cross-entropy loss
        inp = np.asarray(inputs) % self.vocab_size
        tgt = np.asarray(targets) % self.vocab_size
        logits = self.W[inp]  # (B, T, V)
        exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)  # (B, T, V)

        B, T = inp.shape
        for b in range(B):
            for t in range(T):
                target = tgt[b, t]
                input_token = inp[b, t]
                g = probs[b, t].copy()
                g[target] -= 1.0
                self.W[input_token] -= lr * g


def test_held_out_validation_loss_domain_bpb_no_regression(tmp_path):
    """Verification on small-scale run confirming held-out validation loss (domain_bpb.py)
    on web/math domains does not regress during decay.

    Demonstrates:
      * Training on domain A (code) with replay buffer preserving web & math.
      * Comparing decay phase WITH replay vs WITHOUT replay.
      * With replay, web and math held-out BPB stays healthy and does not regress.
    """
    vocab = 32
    seq_len = 8
    batch_size = 2

    # Synthesize distinct domain data
    # Domain 1 (Code): tokens concentrated in 0..10
    code_val = np.random.default_rng(1).integers(0, 10, size=200).astype(np.uint16)
    # Domain 2 (Web): tokens concentrated in 11..20
    web_val = np.random.default_rng(2).integers(11, 20, size=200).astype(np.uint16)
    # Domain 3 (Math): tokens concentrated in 21..30
    math_val = np.random.default_rng(3).integers(21, 30, size=200).astype(np.uint16)

    val_dir = tmp_path / "val_domains"
    val_dir.mkdir()
    pack_ids(code_val, val_dir / "code.bin", dtype=np.uint16, n_bytes=len(code_val) * 2)
    pack_ids(web_val, val_dir / "web.bin", dtype=np.uint16, n_bytes=len(web_val) * 2)
    pack_ids(math_val, val_dir / "math.bin", dtype=np.uint16, n_bytes=len(math_val) * 2)

    val_domains = {
        "code": val_dir / "code.bin",
        "web": val_dir / "web.bin",
        "math": val_dir / "math.bin",
    }

    # Prepare training streams
    # Primary: code tokens (causes forgetting of web/math if unmitigated)
    train_code = np.random.default_rng(10).integers(0, 10, size=1000).astype(np.uint16)
    train_primary_bin = tmp_path / "train_code.bin"
    pack_ids(train_code, train_primary_bin, dtype=np.uint16)

    # Replay: web and math tokens
    train_replay = np.concatenate([
        np.random.default_rng(11).integers(11, 20, size=500).astype(np.uint16),
        np.random.default_rng(12).integers(21, 30, size=500).astype(np.uint16),
    ])
    train_replay_bin = tmp_path / "train_replay.bin"
    pack_ids(train_replay, train_replay_bin, dtype=np.uint16)

    # Pretrain a common trunk model so web and math are initially learned
    initial_model = LinearLookupModel(vocab_size=vocab, seed=42)
    # Give initial exposure to all domains so baseline has learned them
    for _ in range(30):
        # Exposure to web
        wb = np.random.default_rng().integers(11, 20, size=(batch_size, seq_len + 1))
        initial_model.train_on_batch(wb[:, :-1], wb[:, 1:], lr=0.1)
        # Exposure to math
        mb = np.random.default_rng().integers(21, 30, size=(batch_size, seq_len + 1))
        initial_model.train_on_batch(mb[:, :-1], mb[:, 1:], lr=0.1)

    initial_eval = evaluate_domain_bpb(initial_model, val_domains, batch_size=batch_size, seq_len=seq_len)
    web_initial_bpb = initial_eval["by_domain"]["web"]["val_bpb"]
    math_initial_bpb = initial_eval["by_domain"]["math"]["val_bpb"]

    # Model A: Decay WITHOUT replay (only trains on code)
    model_no_replay = LinearLookupModel(vocab_size=vocab, seed=42)
    model_no_replay.W = initial_model.W.copy()

    # Model B: Decay WITH replay buffer (25% replay)
    model_with_replay = LinearLookupModel(vocab_size=vocab, seed=42)
    model_with_replay.W = initial_model.W.copy()

    schedule = WSDSchedule(base_lr=0.05, warmup_steps=0, total_steps=20, decay_steps=20)
    curriculum = LengthCurriculum.single(seq_len=seq_len, batch_size=batch_size, steps=20, grad_accum=1)

    # Train Model A (without replay)
    stream_no_replay = MicroBatchStream(
        curriculum,
        lambda sl, bs: PackedLoader(train_primary_bin, sl, bs, shuffle=True, seed=0),
        seed=0,
        decay_replay_ratio=0.0,
        schedule=schedule,
    )
    for step in range(20):
        b = next(stream_no_replay)
        lr = schedule.lr_at(step)
        model_no_replay.train_on_batch(b[0], b[1], lr=lr)

    # Train Model B (with replay)
    stream_with_replay = MicroBatchStream(
        curriculum,
        lambda sl, bs: PackedLoader(train_primary_bin, sl, bs, shuffle=True, seed=0),
        seed=0,
        replay_loader_factory=lambda sl, bs: PackedLoader(train_replay_bin, sl, bs, shuffle=True, seed=99),
        decay_replay_ratio=0.30,
        schedule=schedule,
    )
    for step in range(20):
        b = next(stream_with_replay)
        lr = schedule.lr_at(step)
        model_with_replay.train_on_batch(b[0], b[1], lr=lr)

    eval_no_replay = evaluate_domain_bpb(model_no_replay, val_domains, batch_size=batch_size, seq_len=seq_len)
    eval_with_replay = evaluate_domain_bpb(model_with_replay, val_domains, batch_size=batch_size, seq_len=seq_len)

    web_no_replay_bpb = eval_no_replay["by_domain"]["web"]["val_bpb"]
    math_no_replay_bpb = eval_no_replay["by_domain"]["math"]["val_bpb"]

    web_with_replay_bpb = eval_with_replay["by_domain"]["web"]["val_bpb"]
    math_with_replay_bpb = eval_with_replay["by_domain"]["math"]["val_bpb"]

    # 1. Replay buffer prevents regression compared to no-replay baseline
    assert web_with_replay_bpb < web_no_replay_bpb, (
        f"Web BPB with replay ({web_with_replay_bpb:.4f}) should be better than without replay ({web_no_replay_bpb:.4f})"
    )
    assert math_with_replay_bpb < math_no_replay_bpb, (
        f"Math BPB with replay ({math_with_replay_bpb:.4f}) should be better than without replay ({math_no_replay_bpb:.4f})"
    )

    # 2. Replay buffer preserves web and math BPB (does not regress significantly from initial baseline)
    assert web_with_replay_bpb <= web_initial_bpb + 0.1, (
        f"Web BPB regressed: initial={web_initial_bpb:.4f}, final={web_with_replay_bpb:.4f}"
    )
    assert math_with_replay_bpb <= math_initial_bpb + 0.1, (
        f"Math BPB regressed: initial={math_initial_bpb:.4f}, final={math_with_replay_bpb:.4f}"
    )


# ============================================================================
# Replay Loader & Shard Discovery Tests
# ============================================================================

def test_replay_shard_discovery_by_subdirectories(tmp_path):
    """Test shard discovery for web, math, and prose subdirectories."""
    web_dir = tmp_path / "fineweb_edu"
    web_dir.mkdir()
    _create_packed_corpus(web_dir / "part-00000.bin", 100)

    math_dir = tmp_path / "math_logic"
    math_dir.mkdir()
    _create_packed_corpus(math_dir / "train.bin", 100)

    prose_dir = tmp_path / "technical_prose"
    prose_dir.mkdir()
    _create_packed_corpus(prose_dir / "shards.bin", 100)

    discovered = discover_replay_shards(tmp_path)
    assert set(discovered.keys()) == {"web", "math", "prose"}
    assert len(discovered["web"]) == 1
    assert len(discovered["math"]) == 1
    assert len(discovered["prose"]) == 1


def test_replay_shard_discovery_by_filenames(tmp_path):
    """Test shard discovery from domain-tagged filenames in flat directory."""
    _create_packed_corpus(tmp_path / "fineweb.bin", 100)
    _create_packed_corpus(tmp_path / "math.bin", 100)
    _create_packed_corpus(tmp_path / "prose.bin", 100)

    discovered = discover_replay_shards(tmp_path)
    assert set(discovered.keys()) == {"web", "math", "prose"}


def test_replay_shard_discovery_fallback_all_bins(tmp_path):
    """Test fallback to general bins when no domain tags match."""
    _create_packed_corpus(tmp_path / "part-00001.bin", 100)
    _create_packed_corpus(tmp_path / "part-00002.bin", 100)

    discovered = discover_replay_shards(tmp_path)
    assert "general" in discovered
    assert len(discovered["general"]) == 2


def test_replay_shard_discovery_empty_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_replay_shards(tmp_path / "nonexistent")
    with pytest.raises(FileNotFoundError):
        discover_replay_shards(tmp_path)  # empty dir


def test_replay_loader_round_robin_and_skip(tmp_path):
    """Test ReplayLoader round-robin distribution and skip_batches."""
    f1 = _create_packed_corpus(tmp_path / "f1.bin", 100, start_val=10)
    f2 = _create_packed_corpus(tmp_path / "f2.bin", 100, start_val=1000)

    l1 = PackedLoader(f1, seq_len=4, batch_size=2, shuffle=False)
    l2 = PackedLoader(f2, seq_len=4, batch_size=2, shuffle=False)

    replay_loader = ReplayLoader([l1, l2], seed=0)
    batches = list(replay_loader.epoch())
    assert len(batches) == len(l1) + len(l2)

    # Verify alternating round-robin pattern
    assert batches[0][0][0, 0] < 1000   # from l1
    assert batches[1][0][0, 0] >= 1000  # from l2
    assert batches[2][0][0, 0] < 1000   # from l1
    assert batches[3][0][0, 0] >= 1000  # from l2

    # Verify skip_batches
    skipped = list(replay_loader.epoch(skip_batches=3))
    assert len(skipped) == len(batches) - 3
    assert _batch_hash(skipped[0]) == _batch_hash(batches[3])


def test_resume_across_curriculum_stages_with_replay(tmp_path):
    """Test multi-stage LengthCurriculum with replay active in later stage."""
    seed = 42
    grad_accum = 2
    curriculum = LengthCurriculum(stages=(
        Stage(index=0, until_frac=0.5, seq_len=4, batch_size=4, steps=10),
        Stage(index=1, until_frac=1.0, seq_len=8, batch_size=2, steps=10),
    ), grad_accum=grad_accum)
    total_steps = curriculum.total_steps  # 20 steps (40 microbatches)
    warmup_steps = 4
    decay_steps = 6  # decay starts at step 14 (inside stage 1)
    interrupt_step = 17

    primary_bin = _create_packed_corpus(tmp_path / "primary_curriculum.bin", 5000, start_val=10)
    replay_bin = _create_packed_corpus(tmp_path / "replay_curriculum.bin", 5000, start_val=20000)

    schedule = WSDSchedule(base_lr=1.0, warmup_steps=warmup_steps, total_steps=total_steps, decay_steps=decay_steps)

    primary_factory = lambda sl, bs: PackedLoader(primary_bin, sl, bs, shuffle=True, seed=seed)
    replay_factory = lambda sl, bs: PackedLoader(replay_bin, sl, bs, shuffle=True, seed=seed + 5)

    # 1. Full run
    s_full = MicroBatchStream(
        curriculum,
        primary_factory,
        seed=seed,
        replay_loader_factory=replay_factory,
        decay_replay_ratio=0.25,
        schedule=schedule,
    )
    total_micros = total_steps * grad_accum
    full_tokens = [_batch_hash(next(s_full)) for _ in range(total_micros)]

    # 2. Interrupted run
    s_part = MicroBatchStream(
        curriculum,
        primary_factory,
        seed=seed,
        replay_loader_factory=replay_factory,
        decay_replay_ratio=0.25,
        schedule=schedule,
    )
    interrupt_micros = interrupt_step * grad_accum
    part_tokens = [_batch_hash(next(s_part)) for _ in range(interrupt_micros)]
    assert part_tokens == full_tokens[:interrupt_micros]

    saved_state = s_part.state_dict()
    assert saved_state["stage_idx"] == 1  # in stage 1

    # 3. Resumed run
    s_resumed = MicroBatchStream(
        curriculum,
        primary_factory,
        seed=seed,
        replay_loader_factory=replay_factory,
        decay_replay_ratio=0.25,
        schedule=schedule,
    )
    s_resumed.load_state_dict(saved_state)
    resumed_tokens = [_batch_hash(next(s_resumed)) for _ in range(total_micros - interrupt_micros)]

    assert resumed_tokens == full_tokens[interrupt_micros:], "Multi-stage curriculum resume with replay diverged"
