"""The cross-backend CI gate, as an executable contract (#303).

Why this file exists
--------------------
``tests/test_backend_parity.py`` guards what CLAUDE.md calls a core invariant: the same
portable weights and the same input must give the same logits on MLX and on the
CUDA/torch backend. Five of its six tests need **both** backends — and for the whole M12
build no CI job installed both, so those five skipped in every job while the checkmark
stayed green. A check that cannot observe its target is indistinguishable from the good
outcome; that is the most expensive failure shape there is.

#303 fixes it by giving a macOS job both backends (`.[dev,data,mlx,cuda]` — the macOS
arm64 torch wheel is CPU-only, which is exactly the surface the CUDA backend is compared
on) and a dedicated step carrying ``MONICA_REQUIRE_BOTH_BACKENDS=1``, under which
``requires_both_backends`` attaches no marker and a missing backend errors.

#315 moved that step off `full-macos` onto its own `parity-macos` job. The reason is this
module's own subject matter: coupled to a 35-minute suite, the 9-second parity gate died as
an `ETIMEDOUT` that reads as infrastructure flake — CONF-2's original failure shape wearing
a different hat. Splitting it gives the gate ~13 minutes of headroom on a ~2-minute job.
``DESIGNATED_JOB`` below is the literal that had to be edited to allow it, which is exactly
the behaviour this contract was designed for.

That flag protects the job that sets it — but nothing stops someone deleting the flag,
the step, or the `cuda` extra, which would return the gate to silent skipping without a
single red mark. **This module is the layer that catches that**, and it is deliberately
portable (``yaml`` + stdlib, no backend import) so it runs on the Linux `portable` job,
where neither backend exists and the parity tests themselves cannot speak.

Deliberate design notes
-----------------------
* The expected job id, workflow file and extras are written out **literally**. This is a
  contract, not a mirror of whatever the YAML happens to say.
* Every lookup asserts it actually found its structure. Following
  ``tests/test_workflow_triggers.py``: an unparseable or missing structure is a **hard
  failure, never a default** — otherwise "I could not find the env var" degrades into
  "no job declares it wrongly, therefore pass".
* Job topology is *not* this module's business; ``test_workflow_triggers.py`` owns the
  trigger×job matrix — #315 adds a job, so that file's ``PR_PUSH_JOBS`` grew to 9 in the
  same commit. The macOS wall-clock ceiling is likewise not this module's business;
  ``test_ci_macos_budget.py`` owns it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"

CI = "ci.yml"

# The one job designated to carry both backends. Literal on purpose (see above).
DESIGNATED_JOB = "parity-macos"
FLAG = "MONICA_REQUIRE_BOTH_BACKENDS"

# The extras that job's install step must request. `mlx` brings the MLX backend; `cuda`
# brings torch — pyproject's own comment says the cuda extra "also runs CPU-only for
# conformance on a Mac", which is this job.
REQUIRED_EXTRAS = ("mlx", "cuda")


def _ci_jobs() -> dict[str, Any]:
    path = WORKFLOWS / CI
    assert path.is_file(), f"missing workflow file: {path}"
    doc = yaml.safe_load(path.read_text())
    assert isinstance(doc, dict) and doc, f"{CI} did not parse into a non-empty mapping"
    jobs = doc.get("jobs")
    assert isinstance(jobs, dict) and jobs, f"{CI} has no `jobs:` mapping"
    return jobs


def _declaring_jobs(jobs: dict[str, Any]) -> dict[str, list[str]]:
    """Map job id -> where it declares FLAG ("job env" or a step name).

    Looks at both the job-level ``env:`` and every step's ``env:``, so moving the flag
    from one to the other does not slip past this contract.
    """
    found: dict[str, list[str]] = {}
    for job_id, job in jobs.items():
        assert isinstance(job, dict), f"{CI}: job {job_id!r} is not a mapping"
        sites: list[str] = []
        if FLAG in (job.get("env") or {}):
            sites.append("job env")
        for step in job.get("steps") or []:
            assert isinstance(step, dict), f"{CI}: a step of {job_id!r} is not a mapping"
            if FLAG in (step.get("env") or {}):
                sites.append(f"step {step.get('name', '<unnamed>')!r}")
        if sites:
            found[job_id] = sites
    return found


# ── DoD-4, layer 3: exactly one job declares the flag, and it is `parity-macos` ──
def test_exactly_one_job_declares_the_both_backends_flag() -> None:
    """One gate, named. Zero means the parity tests are back to skipping silently in
    every job; two means the "designated job" is no longer a single observable thing."""
    declaring = _declaring_jobs(_ci_jobs())
    assert set(declaring) == {DESIGNATED_JOB}, (
        f"{CI} must declare {FLAG} on exactly one job ({DESIGNATED_JOB}); found "
        f"{ {k: v for k, v in declaring.items()} or 'no job at all'} — without it the five "
        f"cross-backend comparisons in tests/test_backend_parity.py skip in every CI job "
        f"while the checkmark stays green (#303)"
    )


def test_the_flag_is_set_to_one() -> None:
    """``requires_both_backends`` compares against the literal string "1"; anything else
    (``true``, ``yes``, an empty value) is a flag that reads as set but does nothing."""
    jobs = _ci_jobs()
    job = jobs[DESIGNATED_JOB]
    envs = [job.get("env") or {}] + [step.get("env") or {} for step in job.get("steps") or []]
    values = [str(env[FLAG]) for env in envs if FLAG in env]
    assert values, f"{CI}: {DESIGNATED_JOB} does not declare {FLAG} anywhere"
    for value in values:
        assert value == "1", (
            f"{CI}: {DESIGNATED_JOB} sets {FLAG}={value!r}; tests/test_backend_parity.py "
            f'compares against the literal "1", so any other value silently restores the '
            f"skip markers"
        )


# ── DoD-1: the designated job actually installs both backends ─────────────────
# Matches the extras list of an editable install: pip install -e ".[dev,data,mlx,cuda]".
# Parsing the extras rather than substring-searching the whole script is deliberate — a
# comment mentioning "cuda" must not satisfy this contract.
_EXTRAS = re.compile(r"pip install\s+(?:-e\s+)?[\"']?\.\[([a-z0-9,\-_ ]+)\]")


def test_designated_job_installs_both_backends() -> None:
    """The flag is only meaningful if the wheels are there. Assert the install step asks
    for both extras — otherwise the job errors on every run instead of gating."""
    job = _ci_jobs()[DESIGNATED_JOB]
    runs = "\n".join(str(step.get("run", "")) for step in job.get("steps") or [])
    installed: set[str] = set()
    for match in _EXTRAS.finditer(runs):
        installed |= {e.strip() for e in match.group(1).split(",")}
    assert installed, (
        f"{CI}: found no `pip install -e '.[...]'` line in {DESIGNATED_JOB}'s steps — "
        f"either the job stopped installing the package or the install line changed "
        f"shape. This test must not pass without observing the extras it checks"
    )
    missing = [e for e in REQUIRED_EXTRAS if e not in installed]
    assert not missing, (
        f"{CI}: {DESIGNATED_JOB} no longer installs {missing} (it installs {sorted(installed)}); "
        f"it is the designated cross-backend parity job and needs mlx AND torch present"
    )


def test_designated_job_runs_on_macos() -> None:
    """mlx has no Linux wheel, so this gate can only live on a macOS runner. If someone
    moves it to ubuntu the parity tests would error rather than run — catch it here."""
    runs_on = _ci_jobs()[DESIGNATED_JOB].get("runs-on")
    assert isinstance(runs_on, str) and runs_on.startswith("macos-"), (
        f"{CI}: {DESIGNATED_JOB} runs-on={runs_on!r}; mlx installs on macOS arm64 only, "
        f"so the both-backends gate cannot move off a macOS runner"
    )


def test_the_parity_step_targets_the_parity_test_file() -> None:
    """The flag must actually unlock a step that runs the file it protects. A flag set on
    (or over) a step that runs something else is a gate pointed at nothing.

    ``_declaring_jobs``/``test_exactly_one_job_declares_the_both_backends_flag`` both
    accept the flag at job level OR step level, so this test must accept both shapes too:
    job-level only requires *some* step in the job to run the parity file; step-level
    requires *every* flagged step to run it.
    """
    job = _ci_jobs()[DESIGNATED_JOB]
    steps = job.get("steps") or []
    job_level = FLAG in (job.get("env") or {})
    flagged = [step for step in steps if FLAG in (step.get("env") or {})]
    assert job_level or flagged, f"{CI}: no step of {DESIGNATED_JOB} carries {FLAG}"

    if job_level:
        ran = any("tests/test_backend_parity.py" in str(step.get("run", "")) for step in steps)
        assert ran, (
            f"{CI}: {FLAG} is declared at job level on {DESIGNATED_JOB} but no step runs "
            f"tests/test_backend_parity.py; the flag would be set for a job that never "
            f"runs the file it unlocks"
        )
    else:
        for step in flagged:
            assert "tests/test_backend_parity.py" in str(step.get("run", "")), (
                f"{CI}: the step carrying {FLAG} does not run tests/test_backend_parity.py; "
                f"the flag would be set for a command it does not affect"
            )
