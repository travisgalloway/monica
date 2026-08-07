"""Revision-pinned loaders + normalizing adapters for the named external code suites (#221).

ABOVE THE SEAM. Pure stdlib at import; `datasets` is imported *inside* `load_external` so
importing this module never requires the optional dependency (the
`src/lsp/ts_boundaries.py` / `src/eval/olmes_adapter.py` precedent).

Scope, decided deliberately
---------------------------
#221 names MultiPL-E (`humaneval-ts`/`mbpp-ts`), SAFIM, Real-FIM-Eval, CrossCodeEval,
RepoBench and MCEval. What ships here is the **loader + adapter layer**, validated against
small checked-in fixtures, with **no live dataset pulls by default**. Those suites need
network dataset access that cannot be meaningfully validated in this environment, and a
loader that has only ever been exercised against a fixture should say so rather than
pretend it has been run against the real thing.

Two consequences, both fail-loud rather than fail-quiet:

1. **`revision` is `None` for every entry** — a Hugging Face commit SHA cannot be resolved
   without network access, and inventing one is strictly worse than not having one: a wrong
   pin either silently loads a different revision or errors far from its cause. So
   `load_external(..., fixture_only=False)` **raises** while the pin is missing, naming the
   set and the one-line fix. Filling the SHAs is a one-line-each follow-up for whoever
   first runs a live pull.
2. **`repo_verified`** records whether the `hf_repo`/`config` identifiers themselves were
   confirmed against something real. Only the MultiPL-E entries are `True` (they match
   `scripts/build_humaneval_ts_set.py`, which has been run against the live hub); the rest
   are best-known identifiers that must be confirmed at pin time. The loader refuses live
   pulls regardless, so an unconfirmed identifier can never be silently hit.

Fixtures are **synthetic**
--------------------------
`eval_sets/external/<name>/fixture.jsonl` holds 5-8 hand-authored rows whose *field names*
match the upstream schema. No third-party rows are checked in, so no upstream licence is
redistributed and there is no benchmark data in this repo to leak into training. They test
the adapter — which is the only thing that can be tested offline.

The normalized shape
--------------------
Every adapter maps an upstream row to
``{"id", "kind", "prompt", "suffix", "answer", "meta"}`` with
``kind in {"completion", "infill", "cross_file"}``. `suffix` is non-`None` only for
`infill` sets; cross-file context lands in `meta`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .code_suite import read_jsonl

#: Repo root, so fixture paths resolve regardless of the caller's cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = _REPO_ROOT / "eval_sets" / "external"

VALID_KINDS = ("completion", "infill", "cross_file")


@dataclass(frozen=True)
class ExternalSet:
    """One external suite: where it lives, how it is pinned, and how a row normalizes."""

    name: str
    hf_repo: str
    config: Optional[str]
    split: str
    revision: Optional[str]          # HF commit SHA; None means UNPINNED -> no live pull
    kind: str
    fixture: Path
    normalize: Callable[[dict], dict]
    repo_verified: bool = False
    note: str = ""


def _norm(row: dict, *, id: str, kind: str, prompt: str, suffix: Optional[str],
          answer: Optional[str], meta: dict) -> dict:
    if kind not in VALID_KINDS:
        raise ValueError(f"unknown kind {kind!r} (want one of {VALID_KINDS})")
    return {"id": str(id), "kind": kind, "prompt": prompt, "suffix": suffix,
            "answer": answer, "meta": meta}


def _require(row: dict, name: str, *fields: str) -> None:
    """Fail loudly on a schema drift rather than emitting a row with empty fields — a
    silently-empty prompt scores as a trivially easy instance."""
    missing = [f for f in fields if f not in row]
    if missing:
        raise ValueError(f"{name}: row is missing {missing} (have {sorted(row)}); the upstream "
                         "schema has changed, or the wrong config/split was loaded")


def normalize_multipl_e(row: dict) -> dict:
    """MultiPL-E: `{name, prompt, tests, stop_tokens}` — a signature + docstring with an
    open body. No gold body ships, so `answer` is `None`."""
    _require(row, "multipl-e", "name", "prompt")
    return _norm(row, id=row["name"], kind="completion", prompt=row["prompt"], suffix=None,
                 answer=None,
                 meta={"tests": row.get("tests"), "stop_tokens": row.get("stop_tokens")})


def normalize_safim(row: dict) -> dict:
    """SAFIM: syntax-aware FIM — `{task_id, prompt, suffix, ground_truth}`."""
    _require(row, "safim", "task_id", "prompt", "suffix", "ground_truth")
    return _norm(row, id=row["task_id"], kind="infill", prompt=row["prompt"],
                 suffix=row["suffix"], answer=row["ground_truth"],
                 meta={"lang": row.get("lang"), "unit": row.get("unit")})


def normalize_real_fim(row: dict) -> dict:
    """Real-FIM-Eval: real commit-derived infills — `{id, prefix, suffix, middle}`."""
    _require(row, "real-fim-eval", "id", "prefix", "suffix", "middle")
    return _norm(row, id=row["id"], kind="infill", prompt=row["prefix"], suffix=row["suffix"],
                 answer=row["middle"],
                 meta={"lang": row.get("lang"), "repo": row.get("repo")})


def normalize_crosscodeeval(row: dict) -> dict:
    """CrossCodeEval: `{task_id, prompt, groundtruth, crossfile_context}` — the cross-file
    context is the whole point, so it rides in `meta` rather than being concatenated here
    (how it is laid into the context window is the caller's experimental choice)."""
    _require(row, "crosscodeeval", "task_id", "prompt", "groundtruth")
    return _norm(row, id=row["task_id"], kind="cross_file", prompt=row["prompt"], suffix=None,
                 answer=row["groundtruth"],
                 meta={"crossfile_context": row.get("crossfile_context"),
                       "lang": row.get("lang"), "repo": row.get("repository")})


def normalize_repobench(row: dict) -> dict:
    """RepoBench-C: `{repo_name, file_path, context, import_statement, next_line}`."""
    _require(row, "repobench", "repo_name", "file_path", "context", "next_line")
    return _norm(row, id=f"{row['repo_name']}::{row['file_path']}", kind="cross_file",
                 prompt=row["context"], suffix=None, answer=row["next_line"],
                 meta={"repo_name": row["repo_name"], "file_path": row["file_path"],
                       "import_statement": row.get("import_statement")})


def normalize_mceval(row: dict) -> dict:
    """MCEval: multilingual generation — `{task_id, language, prompt, canonical_solution}`."""
    _require(row, "mceval", "task_id", "language", "prompt")
    return _norm(row, id=str(row["task_id"]), kind="completion", prompt=row["prompt"],
                 suffix=None, answer=row.get("canonical_solution"),
                 meta={"language": row["language"], "tests": row.get("test")})


_UNPINNED = None      # every entry: see the module docstring's point (1)

EXTERNAL_SETS: Dict[str, ExternalSet] = {
    spec.name: spec for spec in (
        ExternalSet(
            name="multipl-e-humaneval-ts", hf_repo="nuprl/MultiPL-E", config="humaneval-ts",
            split="test", revision=_UNPINNED,          # TODO(pin): resolve the HF commit SHA
            kind="completion", fixture=FIXTURE_ROOT / "multipl-e-humaneval-ts/fixture.jsonl",
            normalize=normalize_multipl_e, repo_verified=True,
            note="already built into eval_sets/humaneval_ts/ by scripts/build_humaneval_ts_set.py",
        ),
        ExternalSet(
            name="multipl-e-mbpp-ts", hf_repo="nuprl/MultiPL-E", config="mbpp-ts",
            split="test", revision=_UNPINNED,          # TODO(pin): resolve the HF commit SHA
            kind="completion", fixture=FIXTURE_ROOT / "multipl-e-mbpp-ts/fixture.jsonl",
            normalize=normalize_multipl_e, repo_verified=True,
        ),
        ExternalSet(
            name="safim", hf_repo="gonglinyuan/safim", config=None, split="test",
            revision=_UNPINNED,                        # TODO(pin): resolve the HF commit SHA
            kind="infill", fixture=FIXTURE_ROOT / "safim/fixture.jsonl",
            normalize=normalize_safim,
            note="repo id NOT verified offline — confirm before the first live pull",
        ),
        ExternalSet(
            name="real-fim-eval", hf_repo="JetBrains-Research/real-fim-eval", config=None,
            split="test", revision=_UNPINNED,          # TODO(pin): resolve the HF commit SHA
            kind="infill", fixture=FIXTURE_ROOT / "real-fim-eval/fixture.jsonl",
            normalize=normalize_real_fim,
            note="repo id NOT verified offline — confirm before the first live pull",
        ),
        ExternalSet(
            name="crosscodeeval", hf_repo="Salesforce/CrossCodeEval", config=None, split="test",
            revision=_UNPINNED,                        # TODO(pin): resolve the HF commit SHA
            kind="cross_file", fixture=FIXTURE_ROOT / "crosscodeeval/fixture.jsonl",
            normalize=normalize_crosscodeeval,
            note="repo id NOT verified offline (upstream is amazon-science/cceval on GitHub) "
                 "— confirm before the first live pull",
        ),
        ExternalSet(
            name="repobench", hf_repo="tianyang/repobench-c", config=None, split="test",
            revision=_UNPINNED,                        # TODO(pin): resolve the HF commit SHA
            kind="cross_file", fixture=FIXTURE_ROOT / "repobench/fixture.jsonl",
            normalize=normalize_repobench,
            note="repo id NOT verified offline — confirm before the first live pull",
        ),
        ExternalSet(
            name="mceval", hf_repo="Multilingual-Multimodal-NLP/McEval", config=None,
            split="test", revision=_UNPINNED,          # TODO(pin): resolve the HF commit SHA
            kind="completion", fixture=FIXTURE_ROOT / "mceval/fixture.jsonl",
            normalize=normalize_mceval,
            note="repo id NOT verified offline — confirm before the first live pull",
        ),
    )
}


def get_external_set(name: str) -> ExternalSet:
    if name not in EXTERNAL_SETS:
        raise SystemExit(f"unknown external set {name!r}; known: {sorted(EXTERNAL_SETS)}")
    return EXTERNAL_SETS[name]


def revision_for(name: str) -> Optional[str]:
    """The pinned HF commit SHA, or `None` while the set is unpinned."""
    return get_external_set(name).revision


def load_external(name: str, *, fixture_only: bool = True,
                  limit: Optional[int] = None) -> List[dict]:
    """Normalized rows for one external suite.

    `fixture_only=True` (**the default**) reads the checked-in synthetic fixture and never
    touches the network. `fixture_only=False` does a live `datasets.load_dataset` — and
    raises `SystemExit` while `revision is None`, because an unpinned pull is not a
    reproducible measurement.
    """
    spec = get_external_set(name)
    if fixture_only:
        rows = read_jsonl(spec.fixture)
    else:
        if spec.revision is None:
            raise SystemExit(
                f"external set {name!r} has no pinned revision, so a live pull would not be "
                f"reproducible. Fill `revision=` for {name!r} in src/eval/external_sets.py "
                f"with the {spec.hf_repo} commit SHA you intend to measure against, then "
                "re-run. (Fixture mode — the default — needs no pin.)"
            )
        try:
            from datasets import load_dataset          # local: optional `[data]` extra
        except ModuleNotFoundError as e:
            raise SystemExit(
                "the `datasets` package is required for a live external-set pull — install "
                "the '[data]' extra, or use the default fixture_only=True path"
            ) from e
        ds = load_dataset(spec.hf_repo, spec.config, split=spec.split, revision=spec.revision)
        rows = [dict(r) for r in ds]

    if limit is not None:
        rows = rows[:limit]
    return [spec.normalize(r) for r in rows]


def external_sets_manifest(*, fixture_only: bool = True) -> Dict[str, dict]:
    """`{name: {hf_repo, config, split, revision, kind, fixture, n_fixture_rows, ...}}` for
    the driver to echo into its results JSON — including the `None` revisions, so an
    unpinned run is visible in its own output rather than only in this file."""
    out: Dict[str, dict] = {}
    for name in sorted(EXTERNAL_SETS):
        spec = EXTERNAL_SETS[name]
        entry = {
            "hf_repo": spec.hf_repo,
            "config": spec.config,
            "split": spec.split,
            "revision": spec.revision,
            "pinned": spec.revision is not None,
            "repo_verified": spec.repo_verified,
            "kind": spec.kind,
            "fixture": str(spec.fixture.relative_to(_REPO_ROOT))
            if spec.fixture.is_relative_to(_REPO_ROOT) else str(spec.fixture),
            "note": spec.note,
        }
        if fixture_only:
            entry["n_fixture_rows"] = len(read_jsonl(spec.fixture)) if spec.fixture.exists() else 0
        out[name] = entry
    return out
