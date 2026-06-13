"""Offline tests for the Wikipedia + instruction text extractors.

The extractors take an injected iterable of records, so the parsing logic is exercised
with no network or `datasets` dependency. Covers: structured-section prose extraction,
JSON-encoded `sections`, the abstract-only fallback, and Dolly formatting + oversample.
"""

from __future__ import annotations

import json

import pytest

from src.data.download import (
    iter_instruct_texts,
    iter_wikipedia_texts,
    wikipedia_doc_text,
)
from src.data.instruct_format import RESPONSE_MARKER, format_example


def _record(name, abstract, sections, encode_sections=False):
    return {
        "name": name,
        "abstract": abstract,
        "sections": json.dumps(sections) if encode_sections else sections,
    }


def test_wikipedia_extracts_title_abstract_and_paragraph_prose():
    sections = [
        {"type": "section", "name": "History", "has_parts": [
            {"type": "paragraph", "value": "First paragraph."},
            {"type": "table", "value": "SHOULD NOT APPEAR"},
            {"type": "section", "has_parts": [
                {"type": "paragraph", "value": "Nested paragraph."},
            ]},
        ]},
    ]
    doc = wikipedia_doc_text(_record("Cats", "A small feline.", sections))
    assert doc == "Cats A small feline. First paragraph. Nested paragraph."
    assert "SHOULD NOT APPEAR" not in doc


def test_wikipedia_handles_json_encoded_sections():
    sections = [{"type": "paragraph", "value": "Encoded body."}]
    doc = wikipedia_doc_text(_record("T", "Abs.", sections, encode_sections=True))
    assert doc == "T Abs. Encoded body."


def test_wikipedia_falls_back_to_abstract_on_bad_sections():
    doc = wikipedia_doc_text({"name": "T", "abstract": "Lead only.",
                              "sections": "{not valid json"})
    assert doc == "T Lead only."


def test_wikipedia_collapses_internal_whitespace_to_one_line():
    sections = [{"type": "paragraph", "value": "line one\nline two\t\ttabbed"}]
    doc = wikipedia_doc_text(_record("N", "ab\nstract", sections))
    assert "\n" not in doc and "\t" not in doc
    assert doc == "N ab stract line one line two tabbed"


def test_iter_wikipedia_skips_empty_docs():
    records = [
        {"name": "", "abstract": "", "sections": None},   # yields nothing -> skipped
        {"name": "Real", "abstract": "Has text.", "sections": None},
    ]
    assert list(iter_wikipedia_texts(records)) == ["Real Has text."]


def test_instruct_formats_with_shared_template():
    records = [{"instruction": "Say hi", "response": "Hi!", "context": ""}]
    (doc,) = list(iter_instruct_texts(records))
    # Matches the shared template (normalized to one line).
    assert doc == " ".join(format_example("Say hi", "Hi!").split())
    assert RESPONSE_MARKER.strip() in doc


def test_instruct_repeat_oversamples():
    records = [{"instruction": "Q", "response": "A"}]
    out = list(iter_instruct_texts(records, repeat=3))
    assert len(out) == 3 and len(set(out)) == 1


def test_instruct_skips_incomplete_pairs():
    records = [
        {"instruction": "no response", "response": ""},
        {"instruction": "", "response": "no instruction"},
        {"instruction": "good", "response": "pair"},
    ]
    out = list(iter_instruct_texts(records))
    assert len(out) == 1


def test_instruct_repeat_below_one_raises():
    with pytest.raises(ValueError):
        list(iter_instruct_texts([{"instruction": "a", "response": "b"}], repeat=0))
