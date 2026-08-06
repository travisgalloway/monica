"""MicroBatchStream: exact kill-and-resume of the data stream (#216).

This is the acceptance surface for the issue's central property — "kill-and-resume
reproduces the EXACT data stream". Pure numpy, no backend, so it runs on any host.

The failure this guards against is silent: a run resumes, resamples data it has already
seen, and its epoch accounting is quietly wrong while every log line looks healthy. So
the tests check the actual token bytes, not just counters, and every test that could pass
vacuously is paired with an assertion that the setup was non-degenerate.
"""

import copy

import numpy as np
import pytest

from src.data.loader import PackedLoader
from src.data.pack import pack_ids
from src.train.curriculum import LengthCurriculum, Stage
from src.train.loop import _micro_batch_stream
from src.train.stream import STATE_VERSION, MicroBatchStream


def _packed(tmp_path, n_tokens=2048):
    """A corpus of distinct ascending token ids, so every batch is identifiable."""
    path = tmp_path / "train.bin"
    pack_ids(np.arange(n_tokens, dtype=np.uint16) % 60000, path, dtype=np.uint16)
    return path


def _factory(path, seed=0):
    return lambda sl, bs: PackedLoader(path, sl, bs, shuffle=True, seed=seed)


def _take(stream, n):
    return [(inp.tobytes(), tgt.tobytes()) for inp, tgt in
            (next(stream) for _ in range(n))]


class FakeLoader:
    """A loader with NO `.rng` — pins the tripwire's `getattr` guard (edge case 9)."""

    def __init__(self, n_batches, batch_size=2, seq_len=4):
        self.n_batches, self.batch_size, self.seq_len = n_batches, batch_size, seq_len

    def __len__(self):
        return self.n_batches

    def epoch(self, reseed=None, skip_batches=0):
        for i in range(skip_batches, self.n_batches):
            yield (np.array([reseed, i]), np.array([reseed, i]))


# --- backward compatibility: single stage == the pre-#216 stream ------------
def test_single_stage_is_bit_identical_to_legacy_stream(tmp_path):
    """The pin: `epoch_idx` is GLOBAL and reseeds as `seed + epoch_idx`, which is what
    makes a one-stage curriculum reproduce `_micro_batch_stream` exactly. The legacy
    resume test and the M4 smoke gate both depend on this."""
    path = _packed(tmp_path)
    seq_len, batch_size, seed = 8, 4, 123
    legacy = _take(_micro_batch_stream(
        PackedLoader(path, seq_len, batch_size, shuffle=True, seed=seed), seed), 60)

    c = LengthCurriculum.single(seq_len, batch_size, steps=60, grad_accum=1)
    new = _take(MicroBatchStream(c, _factory(path, seed), seed), 60)

    assert new == legacy
    assert legacy[:20] != legacy[20:40], "corpus too big: never crossed an epoch"


def test_single_stage_seek_matches_legacy_seek(tmp_path):
    path = _packed(tmp_path)
    seq_len, batch_size, seed, k = 8, 4, 7, 37
    legacy = _take(_micro_batch_stream(
        PackedLoader(path, seq_len, batch_size, shuffle=True, seed=seed),
        seed, start_micro=k), 20)

    c = LengthCurriculum.single(seq_len, batch_size, steps=100, grad_accum=1)
    s = MicroBatchStream(c, _factory(path, seed), seed)
    s.seek_micro(k)
    assert _take(s, 20) == legacy


# --- the core property: state_dict -> load_state_dict reproduces the tail ---
def _curriculum(steps_a=6, steps_b=40, grad_accum=1):
    """Two stages with genuinely different shapes: (seq 8, batch 4) -> (seq 16, batch 2)."""
    return LengthCurriculum(stages=(
        Stage(index=0, until_frac=0.5, seq_len=8, batch_size=4, steps=steps_a),
        Stage(index=1, until_frac=1.0, seq_len=16, batch_size=2, steps=steps_b),
    ), grad_accum=grad_accum)


# On a 512-token corpus: stage 1 is (seq 8, batch 4) -> 56 chunks -> 14 batches/epoch;
# stage 2 is (seq 16, batch 2) -> 30 chunks -> 15 batches/epoch. The kill points below
# cover mid-epoch, an epoch boundary, the stage boundary itself, and several epochs in.
@pytest.mark.parametrize("kill_at, why", [
    (3, "mid-epoch, inside stage 1"),
    (6, "exactly ON the stage boundary"),
    (9, "mid-epoch, inside stage 2 after the crossing"),
    (21, "exactly on an epoch boundary inside stage 2"),
    (30, "deep into stage 2, several epochs later"),
])
def test_resume_reproduces_the_exact_tail(tmp_path, kill_at, why):
    path = _packed(tmp_path, n_tokens=512)
    seed, total = 5, 45
    c = _curriculum()

    uninterrupted = _take(MicroBatchStream(c, _factory(path, seed), seed), total)

    a = MicroBatchStream(c, _factory(path, seed), seed)
    head = _take(a, kill_at)
    state = copy.deepcopy(a.state_dict())
    del a                                       # "kill" the process state

    b = MicroBatchStream(c, _factory(path, seed), seed)
    b.load_state_dict(state)
    tail = _take(b, total - kill_at)

    assert head == uninterrupted[:kill_at], f"head diverged ({why})"
    assert tail == uninterrupted[kill_at:], f"resumed stream diverged ({why})"


def test_state_survives_a_json_round_trip(tmp_path):
    """The state is persisted through `resume_meta.json`, so it must survive json."""
    import json

    from src.train.checkpoint import _jsonable

    path = _packed(tmp_path, n_tokens=512)
    seed, total = 5, 30
    c = _curriculum()
    uninterrupted = _take(MicroBatchStream(c, _factory(path, seed), seed), total)

    a = MicroBatchStream(c, _factory(path, seed), seed)
    _take(a, 9)
    state = json.loads(json.dumps(_jsonable(a.state_dict())))

    b = MicroBatchStream(c, _factory(path, seed), seed)
    b.load_state_dict(state)
    assert _take(b, total - 9) == uninterrupted[9:]


# --- the curriculum actually engages (these tests are not vacuous) ----------
def test_shapes_change_at_the_stage_boundary(tmp_path):
    path = _packed(tmp_path, n_tokens=512)
    c = _curriculum(steps_a=6, steps_b=6)
    s = MicroBatchStream(c, _factory(path), 0)
    shapes = [next(s)[0].shape for _ in range(12)]
    assert shapes[:6] == [(4, 8)] * 6
    assert shapes[6:] == [(2, 16)] * 6


def test_counters_at_the_stage_boundary(tmp_path):
    """Edge case 7: killing exactly ON a boundary must name the NEW stage at
    micro_in_stage 0 with a fresh epoch — not the exhausted tail of the old one."""
    path = _packed(tmp_path, n_tokens=512)
    c = _curriculum(steps_a=6, steps_b=6)
    s = MicroBatchStream(c, _factory(path), 0)
    _take(s, 6)
    st = s.state_dict()
    assert st["stage_idx"] == 1 and st["micro_in_stage"] == 0
    assert st["batches_into_epoch"] == 0
    assert st["global_micro"] == 6


def test_epoch_idx_is_monotone_across_stages(tmp_path):
    path = _packed(tmp_path, n_tokens=512)
    c = _curriculum(steps_a=6, steps_b=20)
    s = MicroBatchStream(c, _factory(path), 0)
    seen = []
    for _ in range(26):
        next(s)
        seen.append(s.epoch_idx)
    assert seen == sorted(seen), "epoch_idx must never go backwards"
    assert seen[-1] > seen[0], "test is vacuous: no epoch ever rolled"


def test_stage_boundary_lands_on_a_step_boundary(tmp_path):
    """Edge case 1: stage lengths are in OPTIMIZER steps, so no single step ever mixes
    shapes — a mixed-shape step would silently corrupt the gradient."""
    path = _packed(tmp_path, n_tokens=512)
    ga = 3
    c = _curriculum(steps_a=4, steps_b=4, grad_accum=ga)
    s = MicroBatchStream(c, _factory(path), 0)
    for _ in range(8):                       # 8 optimizer steps
        shapes = {next(s)[0].shape for _ in range(ga)}
        assert len(shapes) == 1, f"a single optimizer step mixed shapes: {shapes}"


def test_last_stage_never_terminates(tmp_path):
    """`train()`'s `while step < total_steps` is the only stop condition."""
    path = _packed(tmp_path, n_tokens=512)
    c = _curriculum(steps_a=2, steps_b=2)
    s = MicroBatchStream(c, _factory(path), 0)
    batches = _take(s, 60)                   # far past the allocated 4 steps
    assert len(batches) == 60
    assert s.stage_idx == 1


# --- failure paths all RAISE; none degrade to a silent resample -------------
def test_fingerprint_mismatch_on_seed_raises(tmp_path):
    path = _packed(tmp_path, n_tokens=512)
    c = _curriculum()
    a = MicroBatchStream(c, _factory(path), 1)
    _take(a, 3)
    state = a.state_dict()

    b = MicroBatchStream(c, _factory(path), 2)          # different seed
    with pytest.raises(ValueError, match="does not match this run's configuration"):
        b.load_state_dict(state)


def test_fingerprint_mismatch_on_stage_shape_raises(tmp_path):
    path = _packed(tmp_path, n_tokens=512)
    a = MicroBatchStream(_curriculum(), _factory(path), 0)
    _take(a, 3)
    state = a.state_dict()

    other = LengthCurriculum(stages=(
        Stage(index=0, until_frac=0.5, seq_len=8, batch_size=4, steps=6),
        Stage(index=1, until_frac=1.0, seq_len=32, batch_size=2, steps=40),   # 32 != 16
    ), grad_accum=1)
    b = MicroBatchStream(other, _factory(path), 0)
    with pytest.raises(ValueError, match="does not match this run's configuration"):
        b.load_state_dict(state)


def test_fingerprint_mismatch_on_extra_fingerprint_raises(tmp_path):
    """scripts/train.py passes {train_path, n_tokens} — swapping the corpus must be loud."""
    path = _packed(tmp_path, n_tokens=512)
    c = _curriculum()
    a = MicroBatchStream(c, _factory(path), 0, extra_fingerprint={"n_tokens": 512})
    _take(a, 3)
    state = a.state_dict()

    b = MicroBatchStream(c, _factory(path), 0, extra_fingerprint={"n_tokens": 999})
    with pytest.raises(ValueError, match="does not match this run's configuration"):
        b.load_state_dict(state)


def test_fingerprint_error_names_the_escape_hatch(tmp_path):
    path = _packed(tmp_path, n_tokens=512)
    a = MicroBatchStream(_curriculum(), _factory(path), 1)
    _take(a, 3)
    b = MicroBatchStream(_curriculum(), _factory(path), 2)
    with pytest.raises(ValueError, match="--ignore-data-state"):
        b.load_state_dict(a.state_dict())


def test_version_mismatch_raises_even_when_not_strict(tmp_path):
    """A state we cannot parse is not a state we can approximate."""
    path = _packed(tmp_path, n_tokens=512)
    a = MicroBatchStream(_curriculum(), _factory(path), 0)
    _take(a, 3)
    state = a.state_dict()
    state["version"] = STATE_VERSION + 1

    b = MicroBatchStream(_curriculum(), _factory(path), 0)
    for strict in (True, False):
        with pytest.raises(ValueError, match="dataloader state version"):
            b.load_state_dict(state, strict=strict)


def test_non_strict_skips_the_fingerprint_check(tmp_path):
    path = _packed(tmp_path, n_tokens=512)
    a = MicroBatchStream(_curriculum(), _factory(path), 0)
    _take(a, 3)
    state = a.state_dict()
    state["fingerprint"] = {"totally": "different"}

    b = MicroBatchStream(_curriculum(), _factory(path), 0)
    b.load_state_dict(state, strict=False)          # no raise
    assert b.global_micro == 3


def test_rejects_non_dict_state(tmp_path):
    path = _packed(tmp_path, n_tokens=512)
    s = MicroBatchStream(_curriculum(), _factory(path), 0)
    with pytest.raises(ValueError, match="must be a dict"):
        s.load_state_dict("nope")


def test_out_of_range_stage_idx_raises(tmp_path):
    path = _packed(tmp_path, n_tokens=512)
    a = MicroBatchStream(_curriculum(), _factory(path), 0)
    _take(a, 3)
    state = a.state_dict()
    state["stage_idx"] = 5

    b = MicroBatchStream(_curriculum(), _factory(path), 0)
    with pytest.raises(ValueError, match="outside this curriculum's"):
        b.load_state_dict(state)


def test_rng_tripwire_fires_on_a_wrong_epoch_pairing(tmp_path):
    """The tripwire is coarse but catches the exact silent failure: a position restored
    against a shuffle it did not come from."""
    path = _packed(tmp_path, n_tokens=512)
    c = _curriculum()
    a = MicroBatchStream(c, _factory(path), 0)
    _take(a, 3)
    state = a.state_dict()
    assert state["rng_state"] is not None, "test is vacuous: no rng state was captured"
    state["epoch_idx"] = state["epoch_idx"] + 7      # position no longer matches the rng

    b = MicroBatchStream(c, _factory(path), 0)
    with pytest.raises(ValueError, match="RNG tripwire failed"):
        b.load_state_dict(state)


def test_stage_with_no_batches_per_epoch_raises_clearly(tmp_path):
    """Edge case 2: at long seq_len a small corpus drops below one batch per epoch.

    512 tokens at seq_len 200 give 2 chunks — enough for `PackedLoader` to construct, but
    fewer than one batch of 8, so `epoch()` would yield nothing FOREVER rather than fail.
    """
    path = _packed(tmp_path, n_tokens=512)
    c = LengthCurriculum(stages=(
        Stage(index=0, until_frac=1.0, seq_len=200, batch_size=8, steps=1),),
        grad_accum=1)
    s = MicroBatchStream(c, _factory(path), 0)
    with pytest.raises(ValueError, match="yields no batches per epoch"):
        next(s)


# --- loaders without an rng (edge case 9) -----------------------------------
def test_loader_without_rng_skips_the_tripwire():
    c = LengthCurriculum(stages=(
        Stage(index=0, until_frac=1.0, seq_len=4, batch_size=2, steps=100),),
        grad_accum=1)
    a = MicroBatchStream(c, lambda sl, bs: FakeLoader(5), 0)
    for _ in range(7):
        next(a)
    state = a.state_dict()
    assert state["rng_state"] is None

    b = MicroBatchStream(c, lambda sl, bs: FakeLoader(5), 0)
    b.load_state_dict(state)                     # no raise, no tripwire
    assert b.global_micro == 7
    # ...and the restored position really is position 7: epoch 1 (5 batches/epoch), index 2.
    assert np.array_equal(next(b)[0], np.array([1, 2]))


def test_single_batch_per_epoch_still_advances():
    """Edge case 3: len(loader) == 1 — the epoch advance must still work."""
    c = LengthCurriculum(stages=(
        Stage(index=0, until_frac=1.0, seq_len=4, batch_size=2, steps=100),),
        grad_accum=1)
    s = MicroBatchStream(c, lambda sl, bs: FakeLoader(1), 0)
    reseeds = [int(next(s)[0][0]) for _ in range(5)]
    assert reseeds == [0, 1, 2, 3, 4], "each epoch must reseed with seed + epoch_idx"


# --- seek_micro / seek_step (the no-explicit-state fallbacks) ---------------
def test_seek_micro_refuses_a_multi_stage_curriculum(tmp_path):
    path = _packed(tmp_path, n_tokens=512)
    s = MicroBatchStream(_curriculum(), _factory(path), 0)
    with pytest.raises(ValueError, match="cannot reconstruct a position across a "
                                         "multi-stage curriculum"):
        s.seek_micro(10)


def test_seek_micro_zero_is_a_noop_on_a_multi_stage_curriculum(tmp_path):
    path = _packed(tmp_path, n_tokens=512)
    s = MicroBatchStream(_curriculum(), _factory(path), 0)
    s.seek_micro(0)                              # a fresh run must not raise
    assert s.global_micro == 0


def test_seek_step_reconstructs_the_position_across_stages(tmp_path):
    """The `--ignore-data-state` / pre-#216-bundle fallback. It replays the same counter
    arithmetic the live stream performs, so its tail must match too."""
    path = _packed(tmp_path, n_tokens=512)
    seed, total = 5, 40
    c = _curriculum(steps_a=6, steps_b=40)
    uninterrupted = _take(MicroBatchStream(c, _factory(path, seed), seed), total)

    for k in (0, 3, 6, 7, 20, 33):
        s = MicroBatchStream(c, _factory(path, seed), seed)
        s.seek_step(k)                           # grad_accum == 1, so step == micro
        assert _take(s, total - k) == uninterrupted[k:], f"seek_step({k}) diverged"


def test_seek_step_agrees_with_the_saved_state(tmp_path):
    """Both routes to the same position must produce the same counters — otherwise the
    fallback path silently disagrees with the explicit one."""
    path = _packed(tmp_path, n_tokens=512)
    seed = 5
    c = _curriculum(steps_a=6, steps_b=40)
    for k in (3, 6, 7, 20):
        a = MicroBatchStream(c, _factory(path, seed), seed)
        _take(a, k)
        b = MicroBatchStream(c, _factory(path, seed), seed)
        b.seek_step(k)
        for field in ("stage_idx", "micro_in_stage", "global_micro",
                      "epoch_idx", "batches_into_epoch"):
            assert a.state_dict()[field] == b.state_dict()[field], f"{field} at k={k}"


def test_seek_step_rejects_negative(tmp_path):
    path = _packed(tmp_path, n_tokens=512)
    s = MicroBatchStream(_curriculum(), _factory(path), 0)
    with pytest.raises(ValueError, match="seek_step needs step >= 0"):
        s.seek_step(-1)


# --- accessors --------------------------------------------------------------
def test_active_stage_accessors_track_the_boundary(tmp_path):
    path = _packed(tmp_path, n_tokens=512)
    c = _curriculum(steps_a=3, steps_b=3, grad_accum=2)
    s = MicroBatchStream(c, _factory(path), 0)
    assert (s.seq_len, s.batch_size, s.tokens_per_step) == (8, 4, 8 * 4 * 2)
    _take(s, 3 * 2)
    assert (s.seq_len, s.batch_size, s.tokens_per_step) == (16, 2, 16 * 2 * 2)
