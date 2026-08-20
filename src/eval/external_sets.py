"""Revision-pinned loaders + normalizing adapters for the named external code suites (#221, #304).

ABOVE THE SEAM. Pure stdlib at import; `datasets` is imported *inside* `load_external` so
importing this module never requires the optional dependency (the
`src/lsp/ts_boundaries.py` / `src/eval/olmes_adapter.py` precedent).

What is pinned
--------------
All seven entries carry a real 40-hex Hugging Face commit SHA, read from the live hub on
**2026-08-20** with `HF_TOKEN` unset. None of the seven is gated or private, so a live pull
needs no credential and no token is committed anywhere in this repo. Every `note` below
records how that identifier was confirmed and when.

`load_external(..., fixture_only=False)` therefore works — but the guard that refuses an
*unpinned* pull is still here and still fires (`test_a_live_pull_without_a_pin_raises...`),
because a future entry added without a SHA must not silently become a non-reproducible
measurement.

Four identifiers had to be corrected before anything could load (#304)
---------------------------------------------------------------------
The table shipped under #221 was a best-guess written offline, and five of its seven rows
were wrong the first time they met real data:

* `JetBrains-Research/real-fim-eval` returns HTTP **401** — it does not exist. The dataset
  that does is `gonglinyuan/real_fim_eval` (same author as SAFIM), configs `add`/`edit`.
  We take **`add`**: pure insertion, exactly the `(prefix, suffix, middle)` shape. `edit`
  carries a `to_remove` column, making it a *replacement* task that does not map onto the
  normalized infill row without inventing semantics.
* `Salesforce/CrossCodeEval` returns HTTP **401** — CrossCodeEval has no official HF
  dataset (upstream is `amazon-science/cceval`, released via GitHub/Drive). We pin the
  third-party mirror `Vincentvmt/CrossCodeEval`; see `CROSSCODEEVAL_MIRROR_NOTE`.
* `tianyang/repobench-c` exists but **cannot be loaded**: its only files are
  `.gitattributes`, `README.md` and `repobench-c.py`, i.e. it is a loading-script dataset,
  and `datasets>=4` removed script execution. Pinning it would have produced a pin that
  fails at the first live pull. `tianyang/repobench_python_v1.1` is the maintained parquet
  re-release carrying the schema this adapter wants.
* `safim` and `mceval` both had `config=None`, which is invalid — neither declares a default
  config. `block` is SAFIM's headline block-completion config; `generation` is McEval's
  completion-from-docstring config.

Two measurement caveats, recorded so no reader mistakes these for clean instruments
-----------------------------------------------------------------------------------
1. **CrossCodeEval is a third-party mirror.** It was checked, not trusted: its file tree is
   a verbatim copy of the official `crosscodeeval_data/` release layout, and the pinned
   TypeScript file holds **3356** rows, matching the published CrossCodeEval TypeScript
   instance count. That is the whole of the verification — no upstream checksum exists.
2. **RepoBench v1.1 is Python/Java only.** There is no TypeScript split, so this row
   measures cross-file recall in *Python*, not in M12's target language.

Fixtures are **synthetic**
--------------------------
`eval_sets/external/<name>/fixture.jsonl` holds 5-8 hand-authored rows whose *field names*
match the upstream schema as observed live. No third-party rows are checked in, so no
upstream licence is redistributed and there is no benchmark data in this repo to leak into
training. They test the adapter offline; they are **not** evidence that a live pull works —
that is what the `MONICA_EXTERNAL_LIVE=1` test and the recorded row counts in
`eval_sets/external/README.md` are for.

The normalized shape
--------------------
Every adapter maps an upstream row to
``{"id", "kind", "prompt", "suffix", "answer", "meta"}`` with
``kind in {"completion", "infill", "cross_file"}``. `suffix` is non-`None` only for
`infill` sets; cross-file context lands in `meta`, never concatenated into `prompt`.
`_norm` enforces every part of that — including that `prompt` is a `str`, which is what
stops a struct-valued upstream column (RepoBench's `context` is a *list of structs*) from
sailing through a presence-only check and emitting a non-string prompt, and that it is
non-empty unless the row is an infill whose suffix carries the context.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .code_suite import read_jsonl

#: Repo root, so fixture paths resolve regardless of the caller's cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = _REPO_ROOT / "eval_sets" / "external"

VALID_KINDS = ("completion", "infill", "cross_file")

#: SAFIM's `eval_prompt` is a single string with the hole marked by this literal. Splitting
#: on it is the only way to recover (prefix, suffix) — there is no `suffix` column.
SAFIM_PLACEHOLDER = "{{completion}}"

CROSSCODEEVAL_MIRROR_NOTE = (
    "third-party mirror: CrossCodeEval has no official HF dataset (upstream "
    "amazon-science/cceval ships via GitHub/Drive) and Salesforce/CrossCodeEval is 401. "
    "Mirror checked 2026-08-20 by (a) file-tree match against the official "
    "crosscodeeval_data/ release layout (LICENSES/, README, "
    "{python,java,csharp,typescript}/line_completion*.jsonl) and (b) row count of the "
    "pinned TypeScript file = 3356, the published CCEval TypeScript instance count."
)


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
    #: Explicit file glob inside the repo, for repos that declare no configs (the
    #: CrossCodeEval mirror). Threaded into `load_dataset` only when set.
    data_files: Optional[str] = None


def _norm(name: str, *, id: Any, kind: str, prompt: Any, suffix: Any,
          answer: Any, meta: dict) -> dict:
    """Build the normalized row, rejecting anything that violates its documented shape.

    This is the fail-loud front door, not `_require`: a presence check only proves a column
    *exists*. RepoBench's `context` exists and is a list of `{identifier, path, snippet}`
    structs, so `prompt=row["context"]` passes a presence check and emits a non-string
    prompt — a guard that accepts bad data instead of rejecting it. The type checks below
    are what actually close that hole.
    """
    if kind not in VALID_KINDS:
        raise ValueError(f"{name}: unknown kind {kind!r} (want one of {VALID_KINDS})")
    if not isinstance(prompt, str):
        raise ValueError(
            f"{name}: prompt must be a str, got {type(prompt).__name__} ({prompt!r:.120}); "
            "the upstream column is not a string — pick the column that holds the prompt "
            "text, or serialize it deliberately"
        )
    if suffix is not None and not isinstance(suffix, str):
        raise ValueError(f"{name}: suffix must be a str or None, got {type(suffix).__name__}")
    # An empty prompt would score as a trivially easy instance — except for an infill row,
    # where an empty *prefix* is a real top-of-file insertion (56 of Real-FIM-Eval `add`'s
    # 17879 rows, every one with a substantial suffix). There, the pair must not both be
    # empty; everywhere else the prompt itself must carry something.
    if not prompt and not (kind == "infill" and suffix):
        raise ValueError(f"{name}: prompt is empty, which would score as a trivially easy "
                         "instance; the upstream schema has changed or the wrong column was used")
    if suffix is not None and kind != "infill":
        raise ValueError(f"{name}: suffix is non-None but kind is {kind!r}; only infill sets "
                         "carry a suffix")
    if suffix is None and kind == "infill":
        raise ValueError(f"{name}: kind is 'infill' but suffix is None")
    if answer is not None and not isinstance(answer, str):
        raise ValueError(f"{name}: answer must be a str or None, got {type(answer).__name__}")
    if not str(id):
        raise ValueError(f"{name}: row has an empty id")
    return {"id": str(id), "kind": kind, "prompt": prompt, "suffix": suffix,
            "answer": answer, "meta": meta}


def _require(row: dict, name: str, *fields: str) -> None:
    """Fail loudly on a schema drift rather than emitting a row with empty fields — a
    silently-empty prompt scores as a trivially easy instance. Presence only; the *type*
    checks live in `_norm`."""
    missing = [f for f in fields if f not in row]
    if missing:
        raise ValueError(f"{name}: row is missing {missing} (have {sorted(row)}); the upstream "
                         "schema has changed, or the wrong config/split was loaded")


def normalize_multipl_e(row: dict) -> dict:
    """MultiPL-E: `{name, language, prompt, doctests, original, prompt_terminology, tests,
    stop_tokens}` — a signature + docstring with an open body. No gold body ships, so
    `answer` is `None`."""
    _require(row, "multipl-e", "name", "prompt")
    return _norm("multipl-e", id=row["name"], kind="completion", prompt=row["prompt"],
                 suffix=None, answer=None,
                 meta={"tests": row.get("tests"), "stop_tokens": row.get("stop_tokens"),
                       "language": row.get("language")})


def normalize_safim(row: dict) -> dict:
    """SAFIM (`block`): `{lang, prompt, eval_prompt, ground_truth, unit_tests, task_id}`.

    There is **no `suffix` column**, and `prompt` is the natural-language problem statement,
    not code. The FIM prefix/suffix live in `eval_prompt` around the literal
    `{{completion}}` placeholder. If the placeholder is ever absent this raises rather than
    partitioning to an empty suffix — a silent miss would turn an infill instance into a
    trivially easy completion.
    """
    _require(row, "safim", "task_id", "eval_prompt", "ground_truth")
    eval_prompt = row["eval_prompt"]
    if not isinstance(eval_prompt, str) or SAFIM_PLACEHOLDER not in eval_prompt:
        raise ValueError(
            f"safim: eval_prompt for task {row['task_id']!r} does not contain the "
            f"{SAFIM_PLACEHOLDER!r} placeholder, so the prefix/suffix split is undefined; "
            "the upstream format has changed"
        )
    prefix, _, suffix = eval_prompt.partition(SAFIM_PLACEHOLDER)
    return _norm("safim", id=row["task_id"], kind="infill", prompt=prefix, suffix=suffix,
                 answer=row["ground_truth"],
                 meta={"lang": row.get("lang"), "unit_tests": row.get("unit_tests"),
                       # the upstream `prompt` column is the NL problem statement; it must
                       # not shadow the normalized (code) prompt.
                       "problem_statement": row.get("prompt")})


def normalize_real_fim(row: dict) -> dict:
    """Real-FIM-Eval (`add`): `{repo, ref, path, prompt, suffix, canonical_solution, lang,
    timestamp}` — real commit-derived insertions. There is no `id` column, so one is built
    from `repo@ref:path`."""
    _require(row, "real-fim-eval", "repo", "path", "prompt", "suffix", "canonical_solution")
    ts = row.get("timestamp")
    return _norm("real-fim-eval", id=f"{row['repo']}@{row.get('ref') or ''}:{row['path']}",
                 kind="infill", prompt=row["prompt"], suffix=row["suffix"],
                 answer=row["canonical_solution"],
                 meta={"lang": row.get("lang"), "repo": row["repo"], "ref": row.get("ref"),
                       "path": row["path"],
                       "timestamp": None if ts is None else str(ts)})


def normalize_crosscodeeval(row: dict) -> dict:
    """CrossCodeEval (mirror, TypeScript, RG-1/BM25 file): `{prompt, groundtruth,
    right_context, metadata{task_id,repository,file,...}, crossfile_context{text,list}}`.

    The cross-file context is the whole point, so it rides in `meta` rather than being
    concatenated here — how it is laid into the context window is the caller's experimental
    choice.
    """
    _require(row, "crosscodeeval", "prompt", "groundtruth", "metadata", "crossfile_context")
    md = row["metadata"]
    if not isinstance(md, dict):
        raise ValueError(f"crosscodeeval: metadata must be a mapping, got {type(md).__name__}")
    task_id = md.get("task_id") or f"{md.get('repository')}::{md.get('file')}"
    ctx = row["crossfile_context"]
    ctx_text = ctx.get("text") if isinstance(ctx, dict) else ctx
    ctx_list = ctx.get("list") if isinstance(ctx, dict) else None
    return _norm("crosscodeeval", id=task_id, kind="cross_file", prompt=row["prompt"],
                 suffix=None, answer=row["groundtruth"],
                 meta={"crossfile_context": ctx_text, "crossfile_context_list": ctx_list,
                       "right_context": row.get("right_context"), "lang": "typescript",
                       "repo": md.get("repository"), "file": md.get("file")})


def normalize_repobench(row: dict) -> dict:
    """RepoBench v1.1 (`cross_file_first`): `{repo_name, file_path, context, import_statement,
    token_num, cropped_code, all_code, next_line, gold_snippet_index, created_at, level}`.

    **`context` is a list of `{identifier, path, snippet}` structs, not a string** — the
    prompt is `cropped_code`; the retrieved snippets belong in `meta`.
    """
    _require(row, "repobench", "repo_name", "file_path", "cropped_code", "next_line")
    return _norm("repobench", id=f"{row['repo_name']}::{row['file_path']}", kind="cross_file",
                 prompt=row["cropped_code"], suffix=None, answer=row["next_line"],
                 meta={"repo_name": row["repo_name"], "file_path": row["file_path"],
                       "import_statement": row.get("import_statement"),
                       "crossfile_snippets": row.get("context"),
                       "gold_snippet_index": row.get("gold_snippet_index"),
                       "token_num": row.get("token_num"), "level": row.get("level")})


def normalize_mceval(row: dict) -> dict:
    """McEval (`generation`): `{task_id, prompt, canonical_solution, instruction, level, test,
    entry_point, signature, docstring}`.

    There is **no `language` column** — `task_id` is `"<Language>/<n>"` (e.g. `AWK/1`), so
    the language is derived from it, raising if the separator is ever absent.
    """
    _require(row, "mceval", "task_id", "prompt")
    task_id = str(row["task_id"])
    if "/" not in task_id:
        raise ValueError(f"mceval: task_id {task_id!r} carries no '<Language>/<n>' separator, "
                         "so the language cannot be derived; the upstream format has changed")
    return _norm("mceval", id=task_id, kind="completion", prompt=row["prompt"], suffix=None,
                 answer=row.get("canonical_solution"),
                 meta={"language": task_id.split("/", 1)[0], "tests": row.get("test"),
                       "entry_point": row.get("entry_point"), "level": row.get("level")})


#: How every SHA below was obtained, echoed into the notes so the method travels with the pin.
_CONFIRMED = "confirmed live on huggingface.co/api/datasets (unauthenticated) 2026-08-20"

EXTERNAL_SETS: Dict[str, ExternalSet] = {
    spec.name: spec for spec in (
        ExternalSet(
            name="multipl-e-humaneval-ts", hf_repo="nuprl/MultiPL-E", config="humaneval-ts",
            split="test", revision="28441b6024e71d4a1c1c0f6bf171c935cd5a43f2",
            kind="completion", fixture=FIXTURE_ROOT / "multipl-e-humaneval-ts/fixture.jsonl",
            normalize=normalize_multipl_e, repo_verified=True,
            note=f"{_CONFIRMED}; also already built into eval_sets/humaneval_ts/ by "
                 "scripts/build_humaneval_ts_set.py. Identifier unchanged since #221.",
        ),
        ExternalSet(
            name="multipl-e-mbpp-ts", hf_repo="nuprl/MultiPL-E", config="mbpp-ts",
            split="test", revision="28441b6024e71d4a1c1c0f6bf171c935cd5a43f2",
            kind="completion", fixture=FIXTURE_ROOT / "multipl-e-mbpp-ts/fixture.jsonl",
            normalize=normalize_multipl_e, repo_verified=True,
            note=f"{_CONFIRMED}; same repo/commit as multipl-e-humaneval-ts, different "
                 "config. Identifier unchanged since #221.",
        ),
        ExternalSet(
            name="safim", hf_repo="gonglinyuan/safim", config="block", split="test",
            revision="be132cc15372e90b6f03a608e77f2d940e384edb",
            kind="infill", fixture=FIXTURE_ROOT / "safim/fixture.jsonl",
            normalize=normalize_safim, repo_verified=True,
            note=f"{_CONFIRMED}. CONFIG CORRECTED #304: was None, which is invalid — SAFIM "
                 "declares block/control/api/block_v2/control_fixed with no default, so a "
                 "live pull would have failed. `block` is the headline block-completion "
                 "config. Adapter rewritten: there is no `suffix` column.",
        ),
        ExternalSet(
            name="real-fim-eval", hf_repo="gonglinyuan/real_fim_eval", config="add",
            split="test", revision="a36062544c7ed6c4e5ffb8dad8536fc7777e1f36",
            kind="infill", fixture=FIXTURE_ROOT / "real-fim-eval/fixture.jsonl",
            normalize=normalize_real_fim, repo_verified=True,
            note=f"{_CONFIRMED}. REPO CORRECTED #304: JetBrains-Research/real-fim-eval is "
                 "HTTP 401 (does not exist); the dataset that exists is "
                 "gonglinyuan/real_fim_eval. Config `add` (pure insertion) chosen over "
                 "`edit`, whose `to_remove` column makes it a replacement task that does "
                 "not map onto (prefix, suffix, middle).",
        ),
        ExternalSet(
            name="crosscodeeval", hf_repo="Vincentvmt/CrossCodeEval", config=None,
            split="train", revision="41f916e35cc48bcca5dc369664f931afd9ffa22f",
            data_files="crosscodeeval_data/typescript/line_completion_rg1_bm25.jsonl",
            kind="cross_file", fixture=FIXTURE_ROOT / "crosscodeeval/fixture.jsonl",
            normalize=normalize_crosscodeeval, repo_verified=True,
            note=f"{_CONFIRMED}. REPO CORRECTED #304: Salesforce/CrossCodeEval is HTTP 401 "
                 f"and no official HF dataset exists. {CROSSCODEEVAL_MIRROR_NOTE} The repo "
                 "declares no configs, so the TypeScript file is named explicitly via "
                 "data_files and lands in the default `train` split. The RG-1/BM25 variant "
                 "is pinned rather than plain line_completion.jsonl because only it carries "
                 "the retrieved `crossfile_context` this suite exists to measure; both files "
                 "hold the same 3356 instances.",
        ),
        ExternalSet(
            name="repobench", hf_repo="tianyang/repobench_python_v1.1", config="default",
            split="cross_file_first", revision="8a7cf0c8942cc1aa066bf261839650ac55a2ff79",
            kind="cross_file", fixture=FIXTURE_ROOT / "repobench/fixture.jsonl",
            normalize=normalize_repobench, repo_verified=True,
            note=f"{_CONFIRMED}. REPO+SPLIT CORRECTED #304: tianyang/repobench-c exists but "
                 "is a loading-script dataset, and datasets>=4 removed script execution, so "
                 "it cannot be loaded at all. This is the maintained parquet re-release; it "
                 "has no `test` split (cross_file_first/cross_file_random/in_file). CAVEAT: "
                 "RepoBench v1.1 is Python/Java only — no TypeScript — so this row measures "
                 "cross-file recall in Python, not in M12's target language.",
        ),
        ExternalSet(
            name="mceval", hf_repo="Multilingual-Multimodal-NLP/McEval", config="generation",
            split="test", revision="c4767fec950de7decd487a40a4dab2795df4a094",
            kind="completion", fixture=FIXTURE_ROOT / "mceval/fixture.jsonl",
            normalize=normalize_mceval, repo_verified=True,
            note=f"{_CONFIRMED}. CONFIG CORRECTED #304: was None, which is invalid — McEval "
                 "declares generation/explanation/completion with no default. `generation` "
                 "is the completion-from-docstring config; `completion` is a separate "
                 "FIM-style config with a different schema. Adapter rewritten: there is no "
                 "`language` column, so it is derived from task_id.",
        ),
    )
}


def get_external_set(name: str) -> ExternalSet:
    if name not in EXTERNAL_SETS:
        raise SystemExit(f"unknown external set {name!r}; known: {sorted(EXTERNAL_SETS)}")
    return EXTERNAL_SETS[name]


def revision_for(name: str) -> Optional[str]:
    """The pinned HF commit SHA. `None` would mean the entry is unpinned — no shipped entry
    is, and `load_external(..., fixture_only=False)` refuses any that becomes so."""
    return get_external_set(name).revision


def load_external(name: str, *, fixture_only: bool = True,
                  limit: Optional[int] = None) -> List[dict]:
    """Normalized rows for one external suite.

    `fixture_only=True` (**the default**) reads the checked-in synthetic fixture and never
    touches the network. `fixture_only=False` does a live `datasets.load_dataset` at the
    pinned revision — and raises `SystemExit` if `revision is None`, because an unpinned
    pull is not a reproducible measurement.

    A pin that has gone stale (revision withdrawn, repo renamed or gated) also raises
    `SystemExit`, naming the set and the whole pin, rather than surfacing as an opaque
    `datasets` traceback.
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
        kwargs = {"data_files": spec.data_files} if spec.data_files else {}
        try:
            ds = load_dataset(spec.hf_repo, spec.config, split=spec.split,
                              revision=spec.revision, **kwargs)
        except Exception as e:
            raise SystemExit(
                f"external set {name!r} failed to load at its pin: repo={spec.hf_repo!r} "
                f"config={spec.config!r} data_files={spec.data_files!r} split={spec.split!r} "
                f"revision={spec.revision!r}. The pin may be stale (revision withdrawn, repo "
                f"renamed, or newly gated) — re-resolve it against the hub and update "
                f"src/eval/external_sets.py. Underlying error: {type(e).__name__}: {e}"
            ) from e
        rows = [dict(r) for r in ds]

    if limit is not None:
        rows = rows[:limit]
    return [spec.normalize(r) for r in rows]


def external_sets_manifest(*, fixture_only: bool = True) -> Dict[str, dict]:
    """`{name: {hf_repo, config, split, revision, kind, fixture, n_fixture_rows, ...}}` for
    the driver to echo into its results JSON — so the exact pin a run measured against is
    visible in that run's own output rather than only in this file."""
    out: Dict[str, dict] = {}
    for name in sorted(EXTERNAL_SETS):
        spec = EXTERNAL_SETS[name]
        entry = {
            "hf_repo": spec.hf_repo,
            "config": spec.config,
            "data_files": spec.data_files,
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
