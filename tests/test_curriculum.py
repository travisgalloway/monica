"""Length-curriculum spec parsing, batch derivation, step allocation, and lookup (#216).

Pure stdlib math — no backend, no data. The validation tests matter as much as the happy
path: `until_frac` is CUMULATIVE, which is easy to confuse with "share of the run", and a
silently-misread spec would produce a plausible-looking but wrong stage allocation.
"""

import pytest

from src.train.curriculum import (
    LengthCurriculum, Stage, build_curriculum, parse_curriculum_spec,
)


# --- spec parsing -----------------------------------------------------------
def test_parse_happy_path_without_batch_size():
    assert parse_curriculum_spec("0.25:2048,0.5:4096,1.0:16384") == [
        (0.25, 2048, None), (0.5, 4096, None), (1.0, 16384, None)]


def test_parse_happy_path_with_explicit_batch_size_and_whitespace():
    assert parse_curriculum_spec(" 0.5:1024:32 , 1.0:4096:8 ") == [
        (0.5, 1024, 32), (1.0, 4096, 8)]


def test_parse_single_stage():
    assert parse_curriculum_spec("1.0:512") == [(1.0, 512, None)]


@pytest.mark.parametrize("spec, match", [
    ("", "empty"),
    ("   ", "empty"),
    ("0.5:1024,,1.0:2048", "stage 1 is empty"),
    ("0.5,1.0:2048", "expected"),                       # too few fields
    ("0.5:1024:8:2,1.0:2048", "expected"),              # too many fields
    ("x:1024,1.0:2048", "malformed"),
    ("0.5:abc,1.0:2048", "malformed"),
    ("0.6:1024,0.5:2048,1.0:4096", "strictly increase"),
    ("0.5:1024,0.5:2048,1.0:4096", "strictly increase"),
    ("0.5:1024,0.9:2048", "last stage must end at until_frac 1.0"),
    ("0.0:1024,1.0:2048", r"until_frac must be in \(0, 1\]"),
    ("-0.5:1024,1.0:2048", r"until_frac must be in \(0, 1\]"),
    ("1.5:1024", r"until_frac must be in \(0, 1\]"),
    ("0.5:0,1.0:2048", "seq_len must be >= 1"),
    ("0.5:-8,1.0:2048", "seq_len must be >= 1"),
    ("0.5:1024:0,1.0:2048", "batch_size must be >= 1"),
])
def test_parse_rejects_bad_specs(spec, match):
    with pytest.raises(ValueError, match=match):
        parse_curriculum_spec(spec)


def test_parse_error_echoes_the_whole_spec():
    """An error naming only the field is useless when the spec came from a shell script."""
    spec = "0.6:1024,0.5:2048,1.0:4096"
    with pytest.raises(ValueError, match=r"0\.6:1024,0\.5:2048,1\.0:4096"):
        parse_curriculum_spec(spec)


# --- batch derivation -------------------------------------------------------
def test_derived_batch_holds_tokens_per_step_roughly_constant():
    c = build_curriculum("0.5:1024,1.0:4096", base_seq_len=2048, base_batch_size=32,
                         grad_accum=1, total_steps=100)
    # reference tokens = 32 * 2048 = 65536
    assert [s.batch_size for s in c.stages] == [64, 16]
    assert [s.tokens_per_step(1) for s in c.stages] == [65536, 65536]


def test_derived_batch_floors_so_a_stage_is_never_more_expensive():
    """Floor, not round: a derived stage can only get cheaper than the reference, so a
    long-context stage can never surprise-OOM by rounding its batch UP."""
    c = build_curriculum("1.0:3000", base_seq_len=1024, base_batch_size=8,
                         grad_accum=1, total_steps=10)
    assert c.stages[0].batch_size == 8192 // 3000 == 2          # not round(2.73) == 3
    assert c.stages[0].tokens_per_step(1) <= 8 * 1024


def test_derived_batch_never_drops_below_one():
    c = build_curriculum("1.0:65536", base_seq_len=128, base_batch_size=2,
                         grad_accum=1, total_steps=10)
    assert c.stages[0].batch_size == 1


def test_explicit_batch_size_overrides_derivation():
    c = build_curriculum("0.5:1024,1.0:4096:3", base_seq_len=2048, base_batch_size=32,
                         grad_accum=1, total_steps=100)
    assert [s.batch_size for s in c.stages] == [64, 3]


# --- step allocation --------------------------------------------------------
def test_allocation_from_total_steps_sums_and_splits_by_cumulative_frac():
    c = build_curriculum("0.25:64,0.75:128,1.0:256", base_seq_len=64, base_batch_size=4,
                         grad_accum=1, total_steps=100)
    assert [s.steps for s in c.stages] == [25, 50, 25]
    assert c.total_steps == 100
    assert c.first_steps == (0, 25, 75)


def test_allocation_from_total_tokens():
    # 1000 tokens; stage 0 gets 500 at 10 tokens/step -> 50 steps, stage 1 gets 500 at
    # 20 tokens/step -> 25 steps.
    c = build_curriculum("0.5:10:1,1.0:20:1", base_seq_len=10, base_batch_size=1,
                         grad_accum=1, total_tokens=1000)
    assert [s.steps for s in c.stages] == [50, 25]
    assert c.total_steps == 75


def test_allocation_gives_every_stage_at_least_one_step():
    """A degenerate spec must not produce a zero-step stage the stream can never leave."""
    c = build_curriculum("0.001:64,1.0:128", base_seq_len=64, base_batch_size=4,
                         grad_accum=1, total_steps=3)
    assert all(s.steps >= 1 for s in c.stages)
    assert c.total_steps >= 3          # max(1, ...) may push it past the requested S


def test_build_requires_exactly_one_budget():
    for kwargs in ({}, {"total_tokens": 100, "total_steps": 10}):
        with pytest.raises(ValueError, match="exactly one of total_tokens/total_steps"):
            build_curriculum("1.0:64", base_seq_len=64, base_batch_size=4,
                             grad_accum=1, **kwargs)


def test_grad_accum_multiplies_tokens_per_step_and_shrinks_step_count():
    c = build_curriculum("1.0:10:1", base_seq_len=10, base_batch_size=1,
                         grad_accum=4, total_tokens=1000)
    assert c.stages[0].tokens_per_step(4) == 40
    assert c.stages[0].steps == 25


# --- lookup -----------------------------------------------------------------
def _c3():
    return build_curriculum("0.25:64,0.75:128,1.0:256", base_seq_len=64,
                            base_batch_size=4, grad_accum=1, total_steps=100)


@pytest.mark.parametrize("step, want", [
    (0, 0), (24, 0),            # last step of stage 0
    (25, 1), (74, 1),           # first / last step of stage 1
    (75, 2), (99, 2),           # first / last step of stage 2
    (100, 2), (10_000, 2),      # past the end: clamped to the last stage
])
def test_stage_index_at(step, want):
    assert _c3().stage_index_at(step) == want


def test_stage_index_at_rejects_negative():
    with pytest.raises(ValueError, match="step must be >= 0"):
        _c3().stage_index_at(-1)


def test_tokens_at_and_tokens_before():
    c = _c3()                    # 25 steps @ 64*4, 50 @ 128*2, 25 @ 256*1
    assert [s.batch_size for s in c.stages] == [4, 2, 1]
    assert c.tokens_at(0) == 256 and c.tokens_at(25) == 256 and c.tokens_at(75) == 256
    assert c.tokens_before(0) == 0
    assert c.tokens_before(25) == 25 * 256
    assert c.tokens_before(100) == 100 * 256
    # Past the end charges the last stage's rate — the stream may sit there.
    assert c.tokens_before(101) == 101 * 256


def test_tokens_before_with_uneven_stage_rates():
    c = build_curriculum("0.5:10:4,1.0:20:1", base_seq_len=10, base_batch_size=4,
                         grad_accum=1, total_steps=20)
    assert [s.steps for s in c.stages] == [10, 10]
    assert c.tokens_before(10) == 10 * 40
    assert c.tokens_before(15) == 10 * 40 + 5 * 20


# --- fingerprint / identity -------------------------------------------------
def test_fingerprint_tracks_shape_but_not_steps():
    """Extending a run with a bigger budget must stay legal: the persisted position is in
    counters, not steps, so `steps` is deliberately out of the fingerprint."""
    a = build_curriculum("0.5:64,1.0:128", base_seq_len=64, base_batch_size=4,
                         grad_accum=1, total_steps=100)
    b = build_curriculum("0.5:64,1.0:128", base_seq_len=64, base_batch_size=4,
                         grad_accum=1, total_steps=400)
    assert a.total_steps != b.total_steps
    assert a.fingerprint() == b.fingerprint()

    c = build_curriculum("0.5:64,1.0:256", base_seq_len=64, base_batch_size=4,
                         grad_accum=1, total_steps=100)
    assert a.fingerprint() != c.fingerprint()


def test_single_is_the_degenerate_one_stage_curriculum():
    c = LengthCurriculum.single(seq_len=128, batch_size=8, steps=42, grad_accum=2)
    assert len(c.stages) == 1 and c.total_steps == 42
    assert c.stages[0] == Stage(index=0, until_frac=1.0, seq_len=128,
                                batch_size=8, steps=42)
    assert c.tokens_at(0) == 128 * 8 * 2
    assert c.stage_index_at(1_000_000) == 0


def test_curriculum_rejects_empty_stages_and_bad_grad_accum():
    with pytest.raises(ValueError, match="at least one stage"):
        LengthCurriculum(stages=(), grad_accum=1)
    with pytest.raises(ValueError, match="grad_accum must be >= 1"):
        LengthCurriculum.single(seq_len=8, batch_size=1, steps=1, grad_accum=0)


def test_describe_reports_every_stage():
    lines = _c3().describe()
    assert len(lines) == 3
    assert "seq_len=64" in lines[0] and "first_step=75" in lines[2]
