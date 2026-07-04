"""Multi-domain distillation corpus extension sources (Phase A', #65).

The A' corpus extension blends new pretrain domains onto the existing FineWeb-derived
distillation corpus so the student distils the Qwen3-Thinking teacher's code/math/docs/
conversational/reasoning/knowledge behavior at pretrain, not only at SFT:

  - **code**: `bigcode/the-stack-dedup` (`iter_the_stack_dedup`), with `the-stack-smol`
    (`iter_the_stack_smol`) as a fallback if the gated dedup dataset isn't accessible.
  - **math**: `open-web-math/open-web-math` (`iter_open_web_math`).
  - **docs**: `SivilTaram/starcoder2-documentation` (`iter_starcoder2_documentation`), filtered
    to the code languages in play — replaces `code-rag-bench/library-documentation`
    (`iter_library_documentation`, kept as a legacy choice), which is Python-only.
  - **wiki** (knowledge): `wikimedia/structured-wikipedia`, English config only
    (`iter_structured_wikipedia`).
  - **conversation**: UltraChat + OASST1, flattened turns -> pretrain text (`iter_conversation`).
  - **reasoning**: Mixture-of-Thoughts + OpenThoughts2-1M CoT traces, flattened the same way
    (`iter_reasoning`) — OpenThoughts2-1M supersets the older OpenThoughts-114k.
  - **code_problems**: competitive-programming / multilingual problem+solution sources —
    `nvidia/OpenCodeReasoning`, `christopher/rosetta-code`, `Multilingual-Multimodal-NLP/
    McEval-Instruct` (`iter_code_problems`).
  - **code_instruct**: instruction -> code-solution sources — `nvidia/OpenCodeInstruct`,
    `m-a-p/CodeFeedback-Filtered-Instruction` (`iter_code_instruct`).

Every loader yields `Record` (`corpus.py`); `messages_to_text` is the pure, deterministic
bridge from the existing chat-row loaders (`sft_sources`, `reasoning_traces`) to plain pretrain
text, and `_qa_records` is the analogous bridge for flat prompt/response-field sources (the new
code_problems/code_instruct loaders). `char_budget_cap` applies a soft per-stream token budget
(for balanced domain/ecosystem coverage) and tallies provenance; `build_extension_records` chains
everything from one config dict into a single `Record` stream + the provenance counters.

ABOVE THE SEAM — no `mlx`/`torch`; `datasets` is imported LAZILY inside the HF loaders, so
importing this module stays cheap (see `tests/test_import_guard.py`, `PORTABLE_MODULES`).
"""

from __future__ import annotations

import itertools
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from .corpus import Record

#: Heuristic chars-per-token used to convert a token budget into a soft character cap
#: (matches the estimate used elsewhere in the distillation planning docs).
CHARS_PER_TOKEN = 3.5

#: The user-curated ~30-ecosystem code language set, mapped to `the-stack-dedup` `data/<lang>`
#: directory names (see `.claude/plans/issue-65.md`, "Code languages"). Verify against the live
#: dataset card before a real build — unresolved dirs error early (HF `load_dataset` 404s), not
#: silently.
DEFAULT_CODE_LANGS: List[str] = [
    "javascript", "typescript", "html", "css", "json", "python", "toml", "yaml", "rust",
    "c-sharp", "xml", "java", "kotlin", "swift", "dart", "go", "c", "makefile", "sql", "php",
    "shell", "hcl", "powershell", "ruby", "c++", "cmake", "r", "markdown", "lua", "matlab",
]

#: Raw language-label aliases (lowercased key) -> our `DEFAULT_CODE_LANGS` naming (the-stack
#: `data/<lang>` directory convention). Datasets vary in casing/spelling for the same language
#: (e.g. starcoder2-docs' "C#", rosetta-code's "C sharp"); only non-identity aliases are listed
#: here — a raw label that already lowercases to a `DEFAULT_CODE_LANGS` entry matches without one.
#: Verify the *actual* raw label set against each live dataset card before a real build (#65);
#: unmapped labels are cleanly excluded (not an error), since language coverage differs per
#: source.
CODE_LANG_ALIASES = {
    "c#": "c-sharp", "csharp": "c-sharp", "c sharp": "c-sharp",
    "cpp": "c++", "c plus plus": "c++",
    "py": "python",
    "yml": "yaml",
    "golang": "go",
    "bash": "shell", "sh": "shell", "shell script": "shell",
    "terraform": "hcl",
    "ps1": "powershell",
    "rb": "ruby",
    "md": "markdown",
}

#: English config name for `wikimedia/structured-wikipedia`. Verify against the live dataset
#: card before a real build (candidates seen: `enwiki_namespace_0`, or a date-prefixed
#: `<date>.en`) — an unresolved config 404s early, same caveat the other loaders carry.
STRUCTURED_WIKIPEDIA_EN_CONFIG = "enwiki_namespace_0"

#: Dataset-level (not per-row) licenses for curated single-license sources, confirmed against
#: the live HF dataset cards — these datasets carry no per-row license field, so probing one
#: always yielded "unknown" before this fix. Entries added 2026-07-04 (`starcoder2-docs`,
#: `wikipedia`, `opencodereasoning`, `rosetta-code`, `mceval`, `opencodeinstruct`,
#: `codefeedback`) are best-guesses from the dataset cards — verify at build time, same as the
#: originals.
DATASET_LEVEL_LICENSE = {
    "openwebmath": "odc-by",
    "library-docs": "cc-by-sa-4.0",
    "starcoder2-docs": "apache-2.0",
    "wikipedia": "cc-by-sa-4.0",
    "opencodereasoning": "cc-by-4.0",
    "rosetta-code": "GFDL",
    "mceval": "cc-by-sa-4.0",
    "opencodeinstruct": "cc-by-4.0",
    "codefeedback": "apache-2.0",
}


# --------------------------------------------------------------------------- #
# License-field fallback helper
# --------------------------------------------------------------------------- #
def _first_license(row: dict, fields: Iterable[str]) -> str:
    """Try each field name in order; the-stack-family rows carry the license as a (possibly
    empty) list. Returns `"unknown"` if none of `fields` yields a usable value."""
    for name in fields:
        val = row.get(name)
        if isinstance(val, (list, tuple)):
            if val:
                return str(val[0])
            continue
        if isinstance(val, str) and val:
            return val
    return "unknown"


def _normalize_code_lang(raw: str) -> str:
    """Map a dataset's raw language label to our `DEFAULT_CODE_LANGS` naming via
    `CODE_LANG_ALIASES`; unmapped labels are just lowercased/stripped (won't match our set if
    genuinely unmapped, which is the intended "exclude cleanly" behavior)."""
    key = (raw or "").strip().lower()
    return CODE_LANG_ALIASES.get(key, key)


def _code_lang_matches(raw: str, wanted: set[str]) -> bool:
    """`wanted` must already be a `set` of `DEFAULT_CODE_LANGS`-style names
    (lowercase-hyphenated), built ONCE per stream by the caller — this runs on a per-row
    streaming hot path (docs/wiki/code_problems filters over potentially millions of rows), so
    it must not rebuild `wanted` on every call. `raw` is the source row's own language label,
    normalized via `_normalize_code_lang` before the membership check."""
    return _normalize_code_lang(raw) in wanted


# --------------------------------------------------------------------------- #
# Code — the-stack-dedup (primary) / the-stack-smol (fallback)
# --------------------------------------------------------------------------- #
def iter_the_stack_dedup(langs: Iterable[str]) -> Iterator[Record]:
    """Stream `bigcode/the-stack-dedup` (v1, deduped; gated, needs `HF_TOKEN`), one pass per
    language over `data_dir=f"data/{lang}"` (lazy `datasets`, streaming)."""
    from datasets import load_dataset  # pragma: no cover - network/optional extra

    for lang in langs:
        ds = load_dataset("bigcode/the-stack-dedup", data_dir=f"data/{lang}",  # pragma: no cover
                          split="train", streaming=True)
        for row in ds:  # pragma: no cover
            text = row.get("content")
            if not text:
                continue
            license = _first_license(row, ("max_stars_repo_licenses",
                                           "max_issues_repo_licenses",
                                           "max_forks_repo_licenses"))
            yield Record(text=text, source="the-stack-dedup", lang=lang, license=license,
                        meta={"is_code": True})


def iter_the_stack_smol(langs: Iterable[str]) -> Iterator[Record]:
    """Fallback code source (single streaming config, filtered by `row["lang"]`) for when
    `the-stack-dedup`'s gate/`HF_TOKEN` isn't available. Field names verified against the live
    dataset card (2026-07-03): `content`/`lang`/`licenses` are correct as coded."""
    from datasets import load_dataset  # pragma: no cover - network/optional extra

    wanted = {lang.strip().lower() for lang in langs}
    ds = load_dataset("bigcode/the-stack-smol", split="train", streaming=True)  # pragma: no cover
    for row in ds:  # pragma: no cover
        lang = (row.get("lang") or "").strip().lower()
        if wanted and lang not in wanted:
            continue
        text = row.get("content")
        if not text:
            continue
        license = _first_license(row, ("licenses",))
        yield Record(text=text, source="the-stack-smol", lang=lang, license=license,
                    meta={"is_code": True})


# --------------------------------------------------------------------------- #
# Math — open-web-math
# --------------------------------------------------------------------------- #
def iter_open_web_math() -> Iterator[Record]:
    """Stream `open-web-math/open-web-math` (lazy `datasets`, streaming). Verified against the
    live dataset card (2026-07-03): the `text` field is correct, but the license is
    dataset-level (`odc-by`), not a per-row column — rows without a `license` key fall back to
    `DATASET_LEVEL_LICENSE["openwebmath"]` instead of `"unknown"`."""
    from datasets import load_dataset  # pragma: no cover - network/optional extra

    ds = load_dataset("open-web-math/open-web-math", split="train",  # pragma: no cover
                      streaming=True)
    for row in ds:  # pragma: no cover
        text = row.get("text")
        if not text:
            continue
        yield Record(text=text, source="openwebmath", lang="en",
                    license=row.get("license") or DATASET_LEVEL_LICENSE["openwebmath"])


# --------------------------------------------------------------------------- #
# Docs — code-rag-bench/library-documentation
# --------------------------------------------------------------------------- #
def iter_library_documentation() -> Iterator[Record]:
    """Stream `code-rag-bench/library-documentation` (~62 MB; lazy `datasets`, streaming).
    Verified against the live dataset card (2026-07-03): the `doc_content` field is correct,
    but there is no per-row `license` column — license is dataset-level (`cc-by-sa-4.0`); rows
    without a `license` key fall back to `DATASET_LEVEL_LICENSE["library-docs"]` instead of
    `"unknown"`."""
    from datasets import load_dataset  # pragma: no cover - network/optional extra

    ds = load_dataset("code-rag-bench/library-documentation",  # pragma: no cover
                      split="train", streaming=True)
    for row in ds:  # pragma: no cover
        text = row.get("doc_content")
        if not text:
            continue
        yield Record(text=text, source="library-docs", lang="en",
                    license=row.get("license") or DATASET_LEVEL_LICENSE["library-docs"])


def iter_starcoder2_documentation(langs: Iterable[str]) -> Iterator[Record]:
    """Stream `SivilTaram/starcoder2-documentation` (single `train` split, no config; ~59.7k
    rows), filtered to `langs` via `_code_lang_matches` against the row's `language` field (48
    raw language classes, mixed-case — e.g. `"C#"`, `"JavaScript"` — see `CODE_LANG_ALIASES`).
    Replaces `code-rag-bench/library-documentation` (`iter_library_documentation`, kept above as
    a legacy choice), which is Python-only. Field names (`project`/`source`/`language`/
    `content`) and the license are best-guesses from the live dataset card (2026-07-04) — verify
    before a real build."""
    from datasets import load_dataset  # pragma: no cover - network/optional extra

    wanted = {lang.strip().lower() for lang in langs}
    ds = load_dataset("SivilTaram/starcoder2-documentation",  # pragma: no cover
                      split="train", streaming=True)
    for row in ds:  # pragma: no cover
        raw_lang = row.get("language") or ""
        if not _code_lang_matches(raw_lang, wanted):
            continue
        text = row.get("content")
        if not text:
            continue
        yield Record(text=text, source="starcoder2-docs", lang=_normalize_code_lang(raw_lang),
                    license=DATASET_LEVEL_LICENSE["starcoder2-docs"], meta={"is_code": True})


# --------------------------------------------------------------------------- #
# Knowledge — wikimedia/structured-wikipedia (English only)
# --------------------------------------------------------------------------- #
def render_wikipedia_sections(sections) -> str:
    """Recursively walk a decoded `sections` structure into plain text. Structured Wikipedia
    rows have **no flat `text` field** — the body lives in nested section objects (each with a
    `name` and either a `value`/`text` leaf or a further `has_parts`/`sections`/`paragraphs`
    list). This is a best-guess shape from the dataset card (2026-07-04) — verify against real
    rows before a real build; unrecognized shapes are simply skipped, not an error. Pure,
    offline-testable (no network)."""
    out: List[str] = []

    def _walk(node) -> None:
        if node is None:
            return
        if isinstance(node, (list, tuple)):
            for item in node:
                _walk(item)
            return
        if isinstance(node, dict):
            name = node.get("name")
            if isinstance(name, str) and name.strip():
                out.append(name.strip())
            value = node.get("value") if "value" in node else node.get("text")
            if isinstance(value, str) and value.strip():
                out.append(value.strip())
            for key in ("has_parts", "sections", "paragraphs"):
                if key in node:
                    _walk(node[key])
            return
        if isinstance(node, str) and node.strip():
            out.append(node.strip())

    _walk(sections)
    return "\n\n".join(out)


def iter_structured_wikipedia() -> Iterator[Record]:
    """Stream `wikimedia/structured-wikipedia`, **English config only**
    (`STRUCTURED_WIKIPEDIA_EN_CONFIG`). Rows have no flat `text` field: the rendered text is
    `abstract` (the lead section) followed by `render_wikipedia_sections(json.loads(row
    ["sections"]))` — infoboxes/tables/references are skipped. License is a per-row field on the
    real dataset; falls back to `DATASET_LEVEL_LICENSE["wikipedia"]` when absent."""
    import json

    from datasets import load_dataset  # pragma: no cover - network/optional extra

    ds = load_dataset("wikimedia/structured-wikipedia",  # pragma: no cover
                      STRUCTURED_WIKIPEDIA_EN_CONFIG, split="train", streaming=True)
    for row in ds:  # pragma: no cover
        parts = []
        abstract = (row.get("abstract") or "").strip()
        if abstract:
            parts.append(abstract)
        raw_sections = row.get("sections")
        if raw_sections:
            try:
                sections = (json.loads(raw_sections) if isinstance(raw_sections, str)
                           else raw_sections)
            except (TypeError, ValueError):
                sections = None
            if sections:
                rendered = render_wikipedia_sections(sections)
                if rendered:
                    parts.append(rendered)
        text = "\n\n".join(parts)
        if not text:
            continue
        license = row.get("license") or DATASET_LEVEL_LICENSE["wikipedia"]
        yield Record(text=text, source="wikipedia", lang="en", license=license)


# --------------------------------------------------------------------------- #
# Chat -> plain pretrain text
# --------------------------------------------------------------------------- #
_ROLE_LABELS = {"system": "System", "user": "User", "assistant": "Assistant"}


def messages_to_text(messages: List[dict]) -> str:
    """Render a `{role, content}` chat thread into plain pretrain text: each turn becomes
    `"{Role}: {content}\\n\\n"`, concatenated in order (e.g. two turns ->
    `"User: ...\\n\\nAssistant: ...\\n\\n"`). Pure and deterministic — empty/blank turns are
    skipped."""
    out: List[str] = []
    for m in messages:
        role = _ROLE_LABELS.get(m.get("role"), "User")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        out.append(f"{role}: {content}\n\n")
    return "".join(out)


def _messages_records(rows: Iterator[dict], fallback_source: str) -> Iterator[Record]:
    for row in rows:
        text = messages_to_text(row.get("messages") or [])
        if not text:
            continue
        yield Record(text=text, source=row.get("source") or fallback_source, lang="en",
                    license=row.get("license", "unknown"))


# --------------------------------------------------------------------------- #
# Conversation — UltraChat + OASST1
# --------------------------------------------------------------------------- #
def iter_conversation(sources: Iterable[str],
                      max_per_source: Optional[int] = None) -> Iterator[Record]:
    """Wrap `sft_sources.load_ultrachat`/`load_oasst1` message-dict outputs through
    `messages_to_text` into `Record`s."""
    from . import sft_sources

    loaders: Dict[str, callable] = {
        "ultrachat": lambda n: sft_sources.load_ultrachat(max_examples=n),
        "oasst1": lambda n: sft_sources.load_oasst1(max_examples=n),
    }
    for name in sources:
        if name not in loaders:
            raise ValueError(f"unknown conversation source {name!r} (have {sorted(loaders)})")
        yield from _messages_records(loaders[name](max_per_source), name)


# --------------------------------------------------------------------------- #
# Reasoning — Mixture-of-Thoughts + OpenThoughts2 (supersets OpenThoughts-114k)
# --------------------------------------------------------------------------- #
def iter_reasoning(sources: Iterable[str],
                   max_per_source: Optional[int] = None) -> Iterator[Record]:
    """Wrap `reasoning_traces.load_mixture_of_thoughts`/`load_openthoughts`/`load_openthoughts2`
    message-dict outputs (already `<think>...</think>` formatted) through `messages_to_text` into
    `Record`s. **Use `openthoughts2` (OpenThoughts2-1M), not `openthoughts` (OpenThoughts-114k)**
    for a real build — 2-1M supersets 114k, so blending both would duplicate traces (#65,
    2026-07-04). `openthoughts` is kept for back-compat / manual use."""
    from . import reasoning_traces

    loaders: Dict[str, callable] = {
        "mot": lambda n: reasoning_traces.load_mixture_of_thoughts(max_examples=n),
        "openthoughts": lambda n: reasoning_traces.load_openthoughts(max_examples=n),
        "openthoughts2": lambda n: reasoning_traces.load_openthoughts2(max_examples=n),
    }
    for name in sources:
        if name not in loaders:
            raise ValueError(f"unknown reasoning source {name!r} (have {sorted(loaders)})")
        yield from _messages_records(loaders[name](max_per_source), name)


# --------------------------------------------------------------------------- #
# Flat prompt/response fields -> plain pretrain text (the code_problems / code_instruct bridge)
# --------------------------------------------------------------------------- #
def _qa_records(rows: Iterable[dict], q_field: str, a_field: str, source: str, license: str, *,
                lang_field: Optional[str] = None, keep_langs: Optional[Iterable[str]] = None,
                license_field: Optional[str] = None, is_code: bool = False) -> Iterator[Record]:
    """Render `{q_field: ..., a_field: ...}` rows into plain Q/A pretrain text
    (`"{question}\\n\\n{answer}\\n\\n"`), optionally filtering on `lang_field` (via
    `_code_lang_matches` against `keep_langs`) and/or resolving a per-row `license_field`
    (falling back to `license` when absent/empty). Pure, offline-testable — the analogue of
    `messages_to_text`/`_messages_records` for sources shaped as a flat prompt/response pair
    rather than a chat thread.

    `is_code` sets `meta.is_code` explicitly for `filters.is_code_record()` — callers whose rows
    are code (all current code_problems/code_instruct sources) MUST pass `is_code=True` rather
    than relying on `is_code_record`'s source-name substring fallback (`"code" in source`),
    which is easy to evade by accident (e.g. a source named "mceval" doesn't contain "code" and
    would otherwise silently skip the permissive-license gate + minified/autogen filters, #182
    review)."""
    wanted = set(keep_langs) if keep_langs is not None else None
    for row in rows:
        if lang_field is not None and wanted is not None:
            if not _code_lang_matches(row.get(lang_field) or "", wanted):
                continue
        question = (row.get(q_field) or "").strip()
        answer = (row.get(a_field) or "").strip()
        if not question or not answer:
            continue
        row_license = row.get(license_field) if license_field else None
        yield Record(text=f"{question}\n\n{answer}\n\n", source=source, lang="en",
                    license=row_license or license,
                    meta={"is_code": True} if is_code else {})


# --------------------------------------------------------------------------- #
# Code problems — OpenCodeReasoning (Python) + rosetta-code + McEval-Instruct (multilingual)
# --------------------------------------------------------------------------- #
def iter_opencodereasoning() -> Iterator[Record]:
    """Stream `nvidia/OpenCodeReasoning` (`split_0`, ~585k rows carrying `input`).
    Competitive-programming problems (`input`) + full R1 reasoning responses (`output`);
    Python-only, no language filter. License: `cc-by-4.0` (dataset-level, per the card)."""
    from datasets import load_dataset  # pragma: no cover - network/optional extra

    ds = load_dataset("nvidia/OpenCodeReasoning", "split_0",  # pragma: no cover
                      split="train", streaming=True)
    yield from _qa_records(ds, "input", "output", "opencodereasoning",
                           DATASET_LEVEL_LICENSE["opencodereasoning"], is_code=True)


def iter_rosetta_code(langs: Iterable[str]) -> Iterator[Record]:
    """Stream `christopher/rosetta-code`, filtered to `langs` via the `language_name` field
    (883+ raw language names — see `CODE_LANG_ALIASES`; unmapped names are cleanly excluded).
    `task_description` = question, `code` = answer. License: `GFDL` (per the card, not an SPDX
    id — Rosetta Code's own source-material license)."""
    from datasets import load_dataset  # pragma: no cover - network/optional extra

    wanted = {lang.strip().lower() for lang in langs}
    ds = load_dataset("christopher/rosetta-code", split="train",  # pragma: no cover
                      streaming=True)
    yield from _qa_records(ds, "task_description", "code", "rosetta-code",
                           DATASET_LEVEL_LICENSE["rosetta-code"],
                           lang_field="language_name", keep_langs=wanted, is_code=True)


def iter_mceval_instruct(langs: Iterable[str]) -> Iterator[Record]:
    """Stream `Multilingual-Multimodal-NLP/McEval-Instruct` (~35.9k rows), filtered to `langs`
    via the `language` field (69 raw language values). `instruction` = question, `output` =
    answer. License: `cc-by-sa-4.0` (dataset-level)."""
    from datasets import load_dataset  # pragma: no cover - network/optional extra

    wanted = {lang.strip().lower() for lang in langs}
    ds = load_dataset("Multilingual-Multimodal-NLP/McEval-Instruct",  # pragma: no cover
                      split="train", streaming=True)
    yield from _qa_records(ds, "instruction", "output", "mceval",
                           DATASET_LEVEL_LICENSE["mceval"],
                           lang_field="language", keep_langs=wanted, is_code=True)


def iter_code_problems(sources: Iterable[str], langs: Optional[Iterable[str]] = None,
                       max_per_source: Optional[int] = None) -> Iterator[Record]:
    """Chain code-problem sources (competitive-programming / multilingual problem+solution).
    `opencodereasoning` is Python-only (no language filter applies); `rosetta-code`/`mceval` are
    multilingual, filtered to `langs` (default `DEFAULT_CODE_LANGS`). **`kodcode` (KodCode-V1)
    is deliberately NOT registered here** — its CC BY-NC 4.0 license is non-commercial and was
    excluded by user decision (#65, 2026-07-04)."""
    langs = list(langs) if langs is not None else DEFAULT_CODE_LANGS
    loaders: Dict[str, callable] = {
        "opencodereasoning": lambda: iter_opencodereasoning(),
        "rosetta-code": lambda: iter_rosetta_code(langs),
        "mceval": lambda: iter_mceval_instruct(langs),
    }
    for name in sources:
        if name not in loaders:
            raise ValueError(f"unknown code-problem source {name!r} (have {sorted(loaders)})")
        stream = loaders[name]()
        if max_per_source is not None:
            stream = itertools.islice(stream, max_per_source)
        yield from stream


# --------------------------------------------------------------------------- #
# Code instruct — OpenCodeInstruct + CodeFeedback (instruction -> code solution)
# --------------------------------------------------------------------------- #
def iter_opencodeinstruct() -> Iterator[Record]:
    """Stream `nvidia/OpenCodeInstruct` (~5M rows — the caller MUST cap via `code_instruct.
    tokens` in `build_extension_records`; this loader itself is uncapped). `input` = question,
    `output` = answer. License: `cc-by-4.0` (dataset-level)."""
    from datasets import load_dataset  # pragma: no cover - network/optional extra

    ds = load_dataset("nvidia/OpenCodeInstruct", split="train",  # pragma: no cover
                      streaming=True)
    yield from _qa_records(ds, "input", "output", "opencodeinstruct",
                           DATASET_LEVEL_LICENSE["opencodeinstruct"], is_code=True)


def iter_codefeedback() -> Iterator[Record]:
    """Stream `m-a-p/CodeFeedback-Filtered-Instruction` (~157k rows). `query` = question,
    `answer` = answer. Note: per the dataset card this includes some OpenAI-generated content
    (recorded in provenance, not filtered — consistent with the project's existing MoT/
    OpenThoughts use). License: `apache-2.0` (dataset-level)."""
    from datasets import load_dataset  # pragma: no cover - network/optional extra

    ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction",  # pragma: no cover
                      split="train", streaming=True)
    yield from _qa_records(ds, "query", "answer", "codefeedback",
                           DATASET_LEVEL_LICENSE["codefeedback"], is_code=True)


def iter_code_instruct(sources: Iterable[str],
                       max_per_source: Optional[int] = None) -> Iterator[Record]:
    """Chain code-instruction sources (instruction -> code solution)."""
    loaders: Dict[str, callable] = {
        "opencodeinstruct": lambda: iter_opencodeinstruct(),
        "codefeedback": lambda: iter_codefeedback(),
    }
    for name in sources:
        if name not in loaders:
            raise ValueError(f"unknown code-instruct source {name!r} (have {sorted(loaders)})")
        stream = loaders[name]()
        if max_per_source is not None:
            stream = itertools.islice(stream, max_per_source)
        yield from stream


# --------------------------------------------------------------------------- #
# Soft per-stream token budget + provenance
# --------------------------------------------------------------------------- #
def char_budget_cap(records: Iterable[Record], max_chars: Optional[int],
                    counter: dict) -> Iterator[Record]:
    """Soft per-stream character cap (~`CHARS_PER_TOKEN` chars/token): stops yielding once the
    running character total for this call passes `max_chars` (`None` == uncapped). Tallies
    `{docs, chars, approx_tokens}` into `counter[record.source]` for every record yielded, so
    provenance is exact even when `max_chars` is `None`."""
    total_chars = 0
    for r in records:
        if max_chars is not None and total_chars >= max_chars:
            break
        n = len(r.text)
        total_chars += n
        entry = counter.setdefault(r.source, {"docs": 0, "chars": 0, "approx_tokens": 0})
        entry["docs"] += 1
        entry["chars"] += n
        entry["approx_tokens"] = int(entry["chars"] / CHARS_PER_TOKEN)
        yield r


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def _chars_for(tokens: Optional[int]) -> Optional[int]:
    return int(tokens * CHARS_PER_TOKEN) if tokens is not None else None


def build_extension_records(cfg: dict) -> Tuple[Iterator[Record], dict]:
    """Chain all enabled, capped source generators into one `Record` stream + return the
    provenance counters (shared across every enabled domain).

    `cfg` shape (all keys optional; a missing/falsy domain is skipped):
        {"code": {"source": "the-stack-dedup"|"the-stack-smol", "langs": [...],
                  "tokens_per_lang": int|None, "tokens": int|None},
         "math": {"source": "open-web-math", "tokens": int|None},
         "docs": {"source": "starcoder2-documentation"|"library-documentation",
                  "langs": [...], "tokens": int|None},
         "wiki": {"source": "structured-wikipedia", "tokens": int|None},
         "conversation": {"sources": [...], "tokens": int|None},
         "reasoning": {"sources": [...], "tokens": int|None},
         "code_problems": {"sources": [...], "langs": [...], "tokens": int|None},
         "code_instruct": {"sources": [...], "tokens": int|None}}

    `code.tokens_per_lang` (equal cap/lang, one capped stream per language) takes precedence
    over the pooled `code.tokens` when both are set. `docs.langs` defaults to
    `DEFAULT_CODE_LANGS` (relevant only for `starcoder2-documentation`, which is multilingual;
    `library-documentation` ignores it). `code_problems.langs` likewise defaults to
    `DEFAULT_CODE_LANGS` (relevant only to its multilingual sources, `rosetta-code`/`mceval`;
    `opencodereasoning` is Python-only and ignores it).
    """
    counter: dict = {}
    streams: List[Iterator[Record]] = []

    code_cfg = cfg.get("code") or {}
    if code_cfg.get("source") in ("the-stack-dedup", "the-stack-smol"):
        loader = (iter_the_stack_dedup if code_cfg["source"] == "the-stack-dedup"
                 else iter_the_stack_smol)
        langs = code_cfg.get("langs") or DEFAULT_CODE_LANGS
        tokens_per_lang = code_cfg.get("tokens_per_lang")
        if tokens_per_lang is not None:
            max_chars = _chars_for(tokens_per_lang)
            for lang in langs:
                streams.append(char_budget_cap(loader([lang]), max_chars, counter))
        else:
            streams.append(char_budget_cap(loader(langs), _chars_for(code_cfg.get("tokens")),
                                           counter))

    math_cfg = cfg.get("math") or {}
    if math_cfg.get("source") == "open-web-math":
        streams.append(char_budget_cap(iter_open_web_math(), _chars_for(math_cfg.get("tokens")),
                                       counter))

    docs_cfg = cfg.get("docs") or {}
    if docs_cfg.get("source") == "starcoder2-documentation":
        docs_langs = docs_cfg.get("langs") or DEFAULT_CODE_LANGS
        streams.append(char_budget_cap(iter_starcoder2_documentation(docs_langs),
                                       _chars_for(docs_cfg.get("tokens")), counter))
    elif docs_cfg.get("source") == "library-documentation":
        streams.append(char_budget_cap(iter_library_documentation(),
                                       _chars_for(docs_cfg.get("tokens")), counter))

    wiki_cfg = cfg.get("wiki") or {}
    if wiki_cfg.get("source") == "structured-wikipedia":
        streams.append(char_budget_cap(iter_structured_wikipedia(),
                                       _chars_for(wiki_cfg.get("tokens")), counter))

    conv_cfg = cfg.get("conversation") or {}
    if conv_cfg.get("sources"):
        streams.append(char_budget_cap(iter_conversation(conv_cfg["sources"]),
                                       _chars_for(conv_cfg.get("tokens")), counter))

    reasoning_cfg = cfg.get("reasoning") or {}
    if reasoning_cfg.get("sources"):
        streams.append(char_budget_cap(iter_reasoning(reasoning_cfg["sources"]),
                                       _chars_for(reasoning_cfg.get("tokens")), counter))

    code_problems_cfg = cfg.get("code_problems") or {}
    if code_problems_cfg.get("sources"):
        cp_langs = code_problems_cfg.get("langs") or DEFAULT_CODE_LANGS
        streams.append(char_budget_cap(
            iter_code_problems(code_problems_cfg["sources"], langs=cp_langs),
            _chars_for(code_problems_cfg.get("tokens")), counter))

    code_instruct_cfg = cfg.get("code_instruct") or {}
    if code_instruct_cfg.get("sources"):
        streams.append(char_budget_cap(
            iter_code_instruct(code_instruct_cfg["sources"]),
            _chars_for(code_instruct_cfg.get("tokens")), counter))

    def _chain() -> Iterator[Record]:
        for s in streams:
            yield from s

    return _chain(), counter
