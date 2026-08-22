"""The trigger×job matrix of the GitHub Actions workflows, as an executable contract (#302).

Why this file exists
--------------------
CLAUDE.md states a *hard* constraint: the poc-scale parity jobs must run only on
``workflow_dispatch`` or ``schedule`` and **never** on ``pull_request``/``push``, so that a
green PR run never implies poc scale was exercised. Before #302 that constraint was carried
by an ``if: github.event_name == …`` expression on each heavy job — one typo away from
either silently skipping a PR gate or silently running a 90-minute macOS job on every PR.

#302 made it *structural* instead: the four heavy jobs live in
``.github/workflows/scheduled-parity.yml``, whose ``on:`` block has no ``pull_request`` and
no ``push`` key at all. There is nothing to skip under, so there is nothing to get wrong in
an expression.

That is only a guarantee if something checks it. GitHub itself will not tell us until the
Monday after a bad merge, so this module parses both workflow files and asserts the full
matrix: which jobs would run, for each event, in each file.

#312 and the assertion that changed shape
-----------------------------------------
``ci.yml`` now has a ``schedule:`` of its own — a monthly cron that runs its 9 gates against
unchanged ``main`` to catch environmental drift (a runner-image, Xcode or toolchain move that
per-PR CI reads as flake). This module used to assert ``"schedule" not in _triggers(CI)``, and
deleting that line is **not** a weakening: trigger absence was never what protected the heavy
jobs, it was an artifact of how #302 happened to be implemented. What protects them is the file
split, and that property is asserted directly — a ``schedule`` event in ``ci.yml`` fires exactly
``PR_PUSH_JOBS`` and intersects ``HEAVY_JOBS`` in nothing (DoD-1), and no file holding a heavy job
declares ``pull_request``/``push`` at all (DoD-5). The cron literal and ``ci.yml``'s concurrency
group are pinned too, because a cadence that can be silently retimed or silently cancelled is the
BLIND monitor this coverage exists to avoid.

Deliberate design notes
-----------------------
* The id sets (``ALL_JOBS``/``PR_PUSH_JOBS``/``HEAVY_JOBS``) are written out **literally**.
  This is a contract, not a mirror of whatever the YAML happens to say — adding or removing
  a CI job must force a deliberate edit here.
* ``_evaluates_true`` raises on any expression form it does not recognise. An unparsed
  guard is *unknown*, never *"the job runs"*: a check that cannot observe its target must
  fail loudly rather than read green.
* ``_triggers`` handles the YAML 1.1 ``on:``/``True`` key hazard explicitly (see below) and
  asserts it found a trigger block, so a future PyYAML change cannot degrade into
  "this workflow has no triggers, therefore nothing runs, therefore everything passes".
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"

CI = "ci.yml"
SCHEDULED = "scheduled-parity.yml"

# The 9 jobs that gate every pull request and every push to main.
PR_PUSH_JOBS = {
    "portable",
    "smoke-linux",
    "cuda-cpu",
    "full-macos",
    "parity-macos",
    "swift-macos",
    "swift-linux",
    "swift-parity",
    "swift-engine",
}

# The 4 heavy dispatch/schedule-only jobs (#195's training-fixture oracle pair and
# #267's poc-scale parity pair). These are the ones CLAUDE.md says must never run on
# pull_request/push.
HEAVY_JOBS = {
    "train-fixture-oracle",
    "train-fixture-oracle-verify",
    "poc-fixture-oracle",
    "poc-parity",
}

ALL_JOBS = PR_PUSH_JOBS | HEAVY_JOBS


def _load(name: str) -> dict[str, Any]:
    path = WORKFLOWS / name
    assert path.is_file(), f"missing workflow file: {path}"
    doc = yaml.safe_load(path.read_text())
    assert isinstance(doc, dict) and doc, f"{name} did not parse into a non-empty mapping"
    return doc


def _triggers(workflow: dict[str, Any], name: str) -> dict[str, Any]:
    """Return the workflow's ``on:`` block.

    The hazard: PyYAML's ``safe_load`` implements YAML 1.1, where a bare ``on`` key is the
    **boolean ``True``**, not the string ``"on"``. ``workflow["on"]`` therefore raises
    KeyError on a perfectly valid Actions file. Look under both spellings, and assert one
    of them hit — an empty trigger block would make every "does not run on X" assertion
    below pass vacuously.
    """
    block = workflow.get("on", workflow.get(True))
    assert block is not None, (
        f"{name}: could not find the `on:` block under either the string key 'on' or the "
        f"YAML-1.1 boolean key True (top-level keys: {sorted(map(str, workflow))})"
    )
    if isinstance(block, list):  # `on: [push, pull_request]` shorthand
        block = {event: None for event in block}
    assert isinstance(block, dict), f"{name}: unexpected `on:` shape {type(block)!r}"
    return block


_COMPARISON = re.compile(
    r"^github\.event_name\s*(==|!=)\s*'([a-z_]+)'$"
)


def _evaluates_true(expr: str, event: str) -> bool:
    """Evaluate the narrow subset of GitHub expressions our workflows use.

    Supports ``github.event_name ==/!= '<event>'`` atoms joined by ``||`` and ``&&``
    (``&&`` binds tighter, as in the Actions expression grammar). Anything else raises:
    an ``if:`` we cannot interpret must fail the test, never default to "the job runs".
    """
    expr = expr.strip()
    if expr.startswith("${{") and expr.endswith("}}"):
        expr = expr[3:-2].strip()

    def atom(text: str) -> bool:
        match = _COMPARISON.match(text.strip())
        if match is None:
            raise ValueError(f"unhandled `if:` expression fragment: {text!r}")
        operator, wanted = match.groups()
        return (event == wanted) if operator == "==" else (event != wanted)

    return any(
        all(atom(a) for a in conjunction.split("&&"))
        for conjunction in expr.split("||")
    )


def _jobs_for(name: str, event: str) -> set[str]:
    """The set of job ids that would run in workflow ``name`` for ``event``."""
    workflow = _load(name)
    if event not in _triggers(workflow, name):
        return set()
    return {
        job_id
        for job_id, job in workflow["jobs"].items()
        if "if" not in job or _evaluates_true(str(job["if"]), event)
    }


# ── DoD-1 ─────────────────────────────────────────────────────────────────────
def test_schedule_fires_the_right_job_set_in_each_workflow() -> None:
    """Each file's cron fires exactly its own jobs, and ci.yml's can never reach a heavy one.

    Each workflow now carries a cron: scheduled-parity.yml's weekly one for the 4 heavy
    gates, ci.yml's monthly one for #312's drift coverage over the 9 PR/push gates. #302's
    guarantee is the third assertion — a `schedule` in ci.yml cannot fire a heavy job,
    because the heavy jobs are in another file. That is the *property* #302 protected;
    "ci.yml has no `schedule:`" was merely the *mechanism* it happened to use, and asserting
    the property directly is strictly stronger than asserting the mechanism.
    """
    assert _jobs_for(SCHEDULED, "schedule") == HEAVY_JOBS
    assert _jobs_for(CI, "schedule") == PR_PUSH_JOBS, (
        "ci.yml's monthly drift cron (#312) must fire exactly the 9 PR/push gates"
    )
    assert HEAVY_JOBS & _jobs_for(CI, "schedule") == set(), (
        "a heavy parity job became reachable from ci.yml's cron — #302's split is the only "
        "thing keeping it off pull requests too"
    )


# ── DoD-1b (#312) ─────────────────────────────────────────────────────────────
def test_ci_drift_cadence_is_the_recorded_monthly_cron() -> None:
    """ci.yml's drift cadence is monthly, and retiming it takes a deliberate edit here.

    The literal is written out for the same reason the job-id sets are: this is a contract,
    not a mirror. Turning "43 9 3 * *" into "* * * * *" — 9 jobs, four of them macOS, every
    minute — must fail a test rather than merge as a one-character YAML change.
    """
    schedule = _triggers(_load(CI), CI)["schedule"]
    assert isinstance(schedule, list) and len(schedule) == 1, (
        f"ci.yml should declare exactly one cron entry, found {schedule!r}"
    )
    cron = schedule[0]["cron"]
    assert cron == "43 9 3 * *", f"ci.yml's drift cadence changed to {cron!r} (#312 recorded 43 9 3 * *)"
    day_of_month = cron.split()[2]
    assert day_of_month != "*", (
        f"ci.yml's cron day-of-month is {day_of_month!r} — the cadence is no longer monthly"
    )


# ── DoD-5b (#312) ─────────────────────────────────────────────────────────────
def test_scheduled_ci_run_cannot_be_cancelled_by_a_push() -> None:
    """The monthly run gets its own concurrency group, so a push cannot silently eat it.

    GitHub keeps at most one in-progress plus one pending run per concurrency group, and
    queuing a third **cancels the older pending one**. With the group keyed only on workflow
    and ref, a drift run queued behind an in-flight push-to-main run is cancellable by the
    next push: no failure, no signal, a monitor that reads healthy while blind. Keying the
    group on `github.event_name` separates the two populations.
    """
    concurrency = _load(CI)["concurrency"]
    assert "github.event_name" in concurrency["group"], (
        f"ci.yml's concurrency group is {concurrency['group']!r} — without github.event_name "
        f"a scheduled drift run shares a group with push-to-main runs and can be cancelled "
        f"pending, leaving no signal (#312)"
    )
    # PR-supersede behaviour is unchanged: every run on a PR ref carries the same event name,
    # so they still share a group and still cancel in progress.
    assert _evaluates_true(str(concurrency["cancel-in-progress"]), "pull_request")
    for event in ("push", "schedule", "workflow_dispatch"):
        assert not _evaluates_true(str(concurrency["cancel-in-progress"]), event), (
            f"ci.yml would now cancel an in-progress {event} run"
        )


# ── DoD-2 ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("event", ["pull_request", "push"])
def test_pull_request_and_push_job_sets_unchanged(event: str) -> None:
    """PR/push behaviour is what it was before #302, plus #315's split: 9 jobs, no heavy job."""
    assert _jobs_for(CI, event) == PR_PUSH_JOBS
    assert _jobs_for(SCHEDULED, event) == set()


# ── DoD-3 ─────────────────────────────────────────────────────────────────────
def test_both_workflows_are_manually_dispatchable() -> None:
    """Every job stays manually runnable — now as two `gh workflow run` commands.

    #302 splits one dispatch into two (`ci.yml` and `scheduled-parity.yml`). The
    *capability* — every job dispatchable on demand, with no inputs — is preserved, and
    that is what this asserts.
    """
    for name in (CI, SCHEDULED):
        triggers = _triggers(_load(name), name)
        assert "workflow_dispatch" in triggers, f"{name} is not manually dispatchable"
        assert not (triggers["workflow_dispatch"] or {}).get("inputs"), (
            f"{name} declares workflow_dispatch inputs; a bare `gh workflow run` must "
            f"stay sufficient"
        )
    dispatched = _jobs_for(CI, "workflow_dispatch") | _jobs_for(SCHEDULED, "workflow_dispatch")
    assert dispatched == ALL_JOBS


# ── DoD-4 ─────────────────────────────────────────────────────────────────────
def test_no_ci_job_can_be_skipped_on_pull_request() -> None:
    """No job in ci.yml carries an `if:` at all.

    #302 removed four `if:` expressions and added none. Every `if:` on a PR gate is a
    chance to typo that gate into permanent silence — a job that stops running without
    anyone noticing.
    """
    with_guards = sorted(j for j, body in _load(CI)["jobs"].items() if "if" in body)
    assert with_guards == [], f"ci.yml jobs gained an `if:` guard: {with_guards}"


# ── DoD-5 ─────────────────────────────────────────────────────────────────────
def test_heavy_jobs_are_structurally_unreachable_from_pr_and_push() -> None:
    """CLAUDE.md's hard constraint, proved by trigger *absence* rather than by expression.

    For each heavy job, every workflow that contains it must have neither `pull_request`
    nor `push` in its `on:` block. This is stronger than checking the job's `if:`: there
    is no PR/push event to be skipped under in the first place.
    """
    found: set[str] = set()
    for path in sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml")):
        workflow = _load(path.name)
        contained = HEAVY_JOBS & set(workflow.get("jobs") or {})
        if not contained:
            continue
        found |= contained
        triggers = _triggers(workflow, path.name)
        for event in ("pull_request", "push"):
            assert event not in triggers, (
                f"{path.name} holds heavy job(s) {sorted(contained)} and declares a "
                f"`{event}:` trigger — a green PR run must never imply poc scale was "
                f"exercised (CLAUDE.md)"
            )
    assert found == HEAVY_JOBS, f"heavy jobs not found in any workflow: {sorted(HEAVY_JOBS - found)}"


# ── DoD-6 ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", [CI, SCHEDULED])
def test_needs_edges_resolve_within_their_workflow(name: str) -> None:
    """`needs:` cannot cross workflow files — a dangling edge is a hard Actions error."""
    jobs = _load(name)["jobs"]
    for job_id, body in jobs.items():
        needs = body.get("needs") or []
        if isinstance(needs, str):
            needs = [needs]
        missing = [n for n in needs if n not in jobs]
        assert not missing, f"{name}: job {job_id} needs {missing}, absent from this workflow"


def test_every_job_is_accounted_for() -> None:
    """Anti-BLIND: nothing was lost when #302 moved four jobs between files."""
    assert len(ALL_JOBS) == 13
    ci_jobs = set(_load(CI)["jobs"])
    scheduled_jobs = set(_load(SCHEDULED)["jobs"])
    assert ci_jobs and scheduled_jobs, "one of the workflows parsed with no jobs"
    assert ci_jobs & scheduled_jobs == set(), "a job id is duplicated across both workflows"
    assert ci_jobs | scheduled_jobs == ALL_JOBS


# ── the parser's own guard rails ──────────────────────────────────────────────
def test_expression_evaluator_rejects_what_it_cannot_parse() -> None:
    """An unrecognised `if:` must raise, not silently read as "this job runs"."""
    assert _evaluates_true("github.event_name == 'schedule'", "schedule")
    assert not _evaluates_true("github.event_name == 'schedule'", "push")
    assert _evaluates_true(
        "github.event_name == 'workflow_dispatch' || github.event_name == 'schedule'",
        "schedule",
    )
    assert not _evaluates_true("github.event_name != 'push'", "push")
    with pytest.raises(ValueError):
        _evaluates_true("success() && github.ref == 'refs/heads/main'", "push")
