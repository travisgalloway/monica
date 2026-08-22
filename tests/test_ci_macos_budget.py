"""The macOS CI wall-clock ceiling, as an executable contract (#315).

Why this file exists
--------------------
#303 gave the macOS job both backends so the five MLX↔torch comparisons in
``tests/test_backend_parity.py`` would actually execute. They did — and the job's wall
clock went from **9m16s to 37m48s** against its own ``timeout-minutes: 45``, because the
same install un-skipped a large torch-gated set. Nothing in the merge path observed that,
because a job seven minutes short of its timeout still shows a green check.

The failure that sets up is specific and expensive. When the suite finally crosses 45
minutes, GitHub kills the job with no explanation — and an unexplained kill on *the one
job carrying the MLX↔CUDA parity guarantee* reads as infrastructure flake, not as a
parity regression. That is the same illegible shape as CONF-2's original defect, where a
green check hid five silent skips. ``timeout-minutes`` is a stop, not a signal.

So #315 does two things, and this module pins both:

1. **Splits the job.** The 14 seconds of gate (parity + smoke) moved to ``parity-macos``;
   ``full-macos`` is now only the suite. Suite growth can no longer starve the gate.
2. **Puts a budget guard inside the suite's pytest step**, which measures its own wall
   clock and exits non-zero with an ``::error::`` naming the cause and the remedy. The
   budget sits strictly below the timeout, so the *legible* guard always fires first.

That ordering is the whole design, and it is worth nothing unless something checks it —
a budget that drifts above its timeout is a guard that never runs, which is exactly the
BLIND monitor this repo's standing rules warn about. Hence
``test_budget_fires_before_the_timeout``.

Deliberate design notes
-----------------------
* The job ids, the ``timeout-minutes`` values and the budget literal are written out
  **literally**. This is a contract, not a mirror of the YAML: silently raising the
  budget to fit a slower suite must be a test edit, not a one-line workflow change.
* Every lookup asserts it found its structure, following
  ``tests/test_ci_backend_matrix.py`` and ``tests/test_workflow_triggers.py``. An
  unparseable step is a **hard failure, never a default** — otherwise "I could not find
  the guard" degrades into "no job is over budget, therefore pass".
* Portable (``yaml`` + stdlib, no backend import), so it runs on the Linux ``portable``
  job, where no macOS runner and neither backend exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"

CI = "ci.yml"

# The macOS Python jobs #315 split apart, with their recorded timeouts. Literal on
# purpose (see above). `swift-macos`/`swift-engine` are macOS too but carry no Python
# suite and no budget; they are out of this contract's scope.
SUITE_JOB = "full-macos"
PARITY_JOB = "parity-macos"

EXPECTED_TIMEOUT_MINUTES = {
    SUITE_JOB: 45,
    PARITY_JOB: 15,
}

# The budget the suite step measures itself against, in seconds. Phase 1 (#315): the
# recorded baseline is 37m48s (2268s), so the budget starts at that baseline plus a
# ~2-minute buffer — 2400s (40 min) — rather than the eventual 20-minute target, which
# requires the suite itself to get faster first. Tighten this once it does.
BUDGET_VAR = "MACOS_SUITE_BUDGET_SECONDS"
EXPECTED_BUDGET_SECONDS = 2400

# How much room the budget must leave below the timeout. Five minutes: enough that the
# guard's own failure path (printing, exiting, uploading the log) completes, and enough
# that ordinary runner variance around the budget does not race the timeout.
MIN_MARGIN_SECONDS = 300


def _ci_jobs() -> dict[str, Any]:
    path = WORKFLOWS / CI
    assert path.is_file(), f"missing workflow file: {path}"
    doc = yaml.safe_load(path.read_text())
    assert isinstance(doc, dict) and doc, f"{CI} did not parse into a non-empty mapping"
    jobs = doc.get("jobs")
    assert isinstance(jobs, dict) and jobs, f"{CI} has no `jobs:` mapping"
    return jobs


def _job(job_id: str) -> dict[str, Any]:
    jobs = _ci_jobs()
    assert job_id in jobs, (
        f"{CI} has no job {job_id!r} (it has {sorted(jobs)}) — #315 split the macOS "
        f"Python work into {SUITE_JOB} and {PARITY_JOB}; removing or renaming one of "
        f"them must be a deliberate edit here, not a silent loss of the budget guard"
    )
    job = jobs[job_id]
    assert isinstance(job, dict), f"{CI}: job {job_id!r} is not a mapping"
    return job


def _budget_step(job: dict[str, Any]) -> dict[str, Any]:
    """The one step of ``job`` declaring the budget variable.

    Asserts exactly one, for the reason ``test_ci_backend_matrix`` asserts exactly one
    flag-declaring job: two budgets is no budget, and zero is a guard that vanished.
    """
    steps = job.get("steps") or []
    assert steps, f"{CI}: {SUITE_JOB} has no steps"
    declaring = [
        step
        for step in steps
        if isinstance(step, dict) and BUDGET_VAR in (step.get("env") or {})
    ]
    assert len(declaring) == 1, (
        f"{CI}: expected exactly one step of {SUITE_JOB} to declare {BUDGET_VAR}, found "
        f"{len(declaring)} ({[s.get('name') for s in declaring] or 'none'}) — without it "
        f"the macOS suite has no ceiling but the illegible `timeout-minutes` kill (#315)"
    )
    return declaring[0]


# ── DoD-4: the budget literal is pinned, so raising it is a deliberate edit ────
def test_the_suite_step_declares_the_recorded_budget() -> None:
    """A budget raised to fit a slower suite is a decision; it must look like one.

    Pinning the literal here means "just bump the number in ci.yml" fails the `portable`
    job, forcing whoever bumps it to say so in a diff a reviewer reads.
    """
    step = _budget_step(_job(SUITE_JOB))
    value = str(step["env"][BUDGET_VAR])
    assert value.isdigit(), (
        f"{CI}: {BUDGET_VAR}={value!r} is not an integer number of seconds; the guard "
        f"compares it with `[ \"$ELAPSED\" -le … ]`, which errors on a non-number"
    )
    assert int(value) == EXPECTED_BUDGET_SECONDS, (
        f"{CI}: {BUDGET_VAR} is {value}s but #315 recorded {EXPECTED_BUDGET_SECONDS}s. "
        f"If the change is intended, re-profile from a run's --durations=25 table and "
        f"update BOTH ci.yml and EXPECTED_BUDGET_SECONDS here"
    )


# ── DoD-4: the variable is actually read — anti-BLIND ─────────────────────────
def test_the_budget_is_actually_compared_against_the_measured_elapsed_time() -> None:
    """A budget no step reads is a monitor that cannot observe its target.

    The most expensive failure shape there is: the env var is present, the workflow
    parses, the check is green, and nothing is being measured. So assert the script
    really measures elapsed time and really compares it.
    """
    script = str(_budget_step(_job(SUITE_JOB)).get("run", ""))
    assert script.strip(), f"{CI}: the {BUDGET_VAR} step has an empty `run:` block"
    for fragment, why in (
        ("$SECONDS", "nothing measures the step's wall clock"),
        (f"-le \"${BUDGET_VAR}\"", "the measured time is never compared to the budget"),
        ("::error::", "a breach would not annotate the run, so it reads as flake"),
        ("#315", "the failure message does not name the issue that explains it"),
    ):
        assert fragment in script, (
            f"{CI}: the budget guard no longer contains {fragment!r} — {why}. A guard "
            f"that cannot observe its target must fail loudly, not read healthy"
        )


def test_the_budget_guard_preserves_a_failing_suite_as_a_failure() -> None:
    """A red suite must report red, never "over budget".

    The guard wraps pytest in `set +e` to capture its status; if it ever stops
    re-raising that status, a genuine test failure would surface as a timing complaint
    (or, worse, as a pass whenever the run happened to be quick).
    """
    script = str(_budget_step(_job(SUITE_JOB)).get("run", ""))
    assert 'exit "$STATUS"' in script, (
        f"{CI}: the budget guard no longer re-raises pytest's exit status; a failing "
        f"suite would be reported as a timing result instead of a test failure (#315)"
    )
    assert "MACOS_SUITE_WALL_CLOCK_SECONDS=" in script, (
        f"{CI}: the guard no longer echoes the measured wall clock; that line is the "
        f"only record of what the suite actually cost on the runner (#315 DoD-1)"
    )
    assert "--durations=25" in script, (
        f"{CI}: the suite step dropped --durations=25; the runner's own profile is the "
        f"only thing that can answer 'what got slower' when the budget is breached"
    )


# ── DoD-4: the legible guard must always fire before the illegible one ────────
@pytest.mark.parametrize("job_id", sorted(EXPECTED_TIMEOUT_MINUTES))
def test_macos_job_timeouts_are_the_recorded_literals(job_id: str) -> None:
    """Both timeouts are pinned. ``parity-macos``'s 15 minutes against a ~2-minute job is
    the headroom the split exists to buy; ``full-macos``'s is the outer stop."""
    timeout = _job(job_id).get("timeout-minutes")
    assert timeout == EXPECTED_TIMEOUT_MINUTES[job_id], (
        f"{CI}: {job_id} timeout-minutes is {timeout!r}, #315 recorded "
        f"{EXPECTED_TIMEOUT_MINUTES[job_id]}"
    )


def test_budget_fires_before_the_timeout() -> None:
    """The ordering invariant, asserted rather than assumed.

    If the budget ever drifts above (or too close to) the timeout, GitHub's unexplained
    kill wins the race and the guard never gets to print its message — the ceiling would
    be back to being illegible, which is the entire defect #315 was filed about.
    """
    timeout_seconds = EXPECTED_TIMEOUT_MINUTES[SUITE_JOB] * 60
    assert EXPECTED_BUDGET_SECONDS + MIN_MARGIN_SECONDS <= timeout_seconds, (
        f"budget {EXPECTED_BUDGET_SECONDS}s + {MIN_MARGIN_SECONDS}s margin exceeds "
        f"{SUITE_JOB}'s {timeout_seconds}s timeout — the illegible ETIMEDOUT would fire "
        f"before the legible budget error (#315)"
    )


# ── DoD-3: both macOS Python jobs stay on macOS, and stay separate ────────────
def test_the_split_survived() -> None:
    """The two jobs are distinct, both on macOS, and the suite job no longer runs the
    parity or smoke steps that moved to ``parity-macos``.

    Folding them back together would silently re-couple the 9-second parity gate to a
    35-minute suite — the coupling #315 removed — and no other test would notice.
    """
    for job_id in (SUITE_JOB, PARITY_JOB):
        runs_on = _job(job_id).get("runs-on")
        assert isinstance(runs_on, str) and runs_on.startswith("macos-"), (
            f"{CI}: {job_id} runs-on={runs_on!r}; both halves of the #315 split need the "
            f"macOS arm64 runner (mlx has no Linux wheel)"
        )

    suite_script = "\n".join(
        str(step.get("run", "")) for step in _job(SUITE_JOB).get("steps") or []
    )
    assert "smoke_test.py" not in suite_script, (
        f"{CI}: {SUITE_JOB} runs the smoke gate again; #315 moved it to {PARITY_JOB} so "
        f"a 5-second gate is not hostage to a 35-minute suite"
    )
    assert "--ignore=tests/test_backend_parity.py" in suite_script, (
        f"{CI}: {SUITE_JOB} no longer ignores tests/test_backend_parity.py. Un-ignoring "
        f"it runs the file a second time WITHOUT MONICA_REQUIRE_BOTH_BACKENDS — a run "
        f"that looks like coverage while five comparisons skip (#303)"
    )

    parity_script = "\n".join(
        str(step.get("run", "")) for step in _job(PARITY_JOB).get("steps") or []
    )
    assert "smoke_test.py --backend mlx" in parity_script, (
        f"{CI}: {PARITY_JOB} does not run the MLX smoke gate; it moved here in #315 and "
        f"is now covered by no job at all"
    )
    assert "src.data.split" in parity_script, (
        f"{CI}: {PARITY_JOB} no longer builds a fresh toy split — scripts/smoke_test.py "
        f"needs one, and must never be pointed at data/split (the real OLMo corpus)"
    )
