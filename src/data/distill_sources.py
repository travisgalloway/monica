"""Multi-domain distillation corpus extension sources (Phase A', #65).

The A' corpus extension blends five new pretrain domains onto the existing FineWeb-derived
distillation corpus so the student distils the Qwen3-Thinking teacher's code/math/docs/
conversational/reasoning behavior at pretrain, not only at SFT:

  - **code**: `bigcode/the-stack-dedup` (`iter_the_stack_dedup`), with `the-stack-smol`
    (`iter_the_stack_smol`) as a fallback if the gated dedup dataset isn't accessible.
  - **math**: `open-web-math/open-web-math` (`iter_open_web_math`).
  - **docs**: `code-rag-bench/library-documentation` (`iter_library_documentation`).
  - **conversation**: UltraChat + OASST1, flattened turns -> pretrain text (`iter_conversation`).
  - **reasoning**: Mixture-of-Thoughts + OpenThoughts CoT traces, flattened the same way
    (`iter_reasoning`).

Every loader yields `Record` (`corpus.py`); `messages_to_text` is the pure, deterministic
bridge from the existing chat-row loaders (`sft_sources`, `reasoning_traces`) to plain pretrain
text. `char_budget_cap` applies a soft per-stream token budget (for balanced domain/ecosystem
coverage) and tallies provenance; `build_extension_records` chains everything from one config
dict into a single `Record` stream + the provenance counters.

ABOVE THE SEAM — no `mlx`/`torch`; `datasets` is imported LAZILY inside the HF loaders, so
importing this module stays cheap (see `tests/test_import_guard.py`, `PORTABLE_MODULES`).
"""

from __future__ import annotations

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

#: Dataset-level (not per-row) licenses for curated single-license sources, confirmed against
#: the live HF dataset cards (2026-07-03) — these datasets carry no per-row license field, so
#: probing one always yielded "unknown" before this fix.
DATASET_LEVEL_LICENSE = {
    "openwebmath": "odc-by",
    "library-docs": "cc-by-sa-4.0",
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
# Reasoning — Mixture-of-Thoughts + OpenThoughts
# --------------------------------------------------------------------------- #
def iter_reasoning(sources: Iterable[str],
                   max_per_source: Optional[int] = None) -> Iterator[Record]:
    """Wrap `reasoning_traces.load_mixture_of_thoughts`/`load_openthoughts` message-dict
    outputs (already `<think>...</think>` formatted) through `messages_to_text` into `Record`s."""
    from . import reasoning_traces

    loaders: Dict[str, callable] = {
        "mot": lambda n: reasoning_traces.load_mixture_of_thoughts(max_examples=n),
        "openthoughts": lambda n: reasoning_traces.load_openthoughts(max_examples=n),
    }
    for name in sources:
        if name not in loaders:
            raise ValueError(f"unknown reasoning source {name!r} (have {sorted(loaders)})")
        yield from _messages_records(loaders[name](max_per_source), name)


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
         "docs": {"source": "library-documentation", "tokens": int|None},
         "conversation": {"sources": [...], "tokens": int|None},
         "reasoning": {"sources": [...], "tokens": int|None}}

    `code.tokens_per_lang` (equal cap/lang, one capped stream per language) takes precedence
    over the pooled `code.tokens` when both are set.
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
    if docs_cfg.get("source") == "library-documentation":
        streams.append(char_budget_cap(iter_library_documentation(),
                                       _chars_for(docs_cfg.get("tokens")), counter))

    conv_cfg = cfg.get("conversation") or {}
    if conv_cfg.get("sources"):
        streams.append(char_budget_cap(iter_conversation(conv_cfg["sources"]),
                                       _chars_for(conv_cfg.get("tokens")), counter))

    reasoning_cfg = cfg.get("reasoning") or {}
    if reasoning_cfg.get("sources"):
        streams.append(char_budget_cap(iter_reasoning(reasoning_cfg["sources"]),
                                       _chars_for(reasoning_cfg.get("tokens")), counter))

    def _chain() -> Iterator[Record]:
        for s in streams:
            yield from s

    return _chain(), counter
