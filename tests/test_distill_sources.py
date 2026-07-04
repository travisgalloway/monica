"""Phase A' multi-domain distillation corpus extension sources (#65).

Hermetic — no network. `datasets` calls in `distill_sources.py` stay `# pragma: no cover`;
this only exercises the pure mapper/render/cap/orchestration functions with fabricated rows.
"""

import json

from src.data.corpus import Record
from src.data.distill_sources import (_code_lang_matches, _normalize_code_lang, _qa_records,
                                       build_extension_records, char_budget_cap,
                                       iter_code_instruct, iter_code_problems,
                                       iter_library_documentation, iter_mceval_instruct,
                                       iter_open_web_math, iter_opencodeinstruct,
                                       iter_opencodereasoning, iter_rosetta_code,
                                       iter_starcoder2_documentation, iter_structured_wikipedia,
                                       iter_the_stack_smol, messages_to_text,
                                       render_wikipedia_sections)


def test_messages_to_text_renders_role_prefixed_turns():
    out = messages_to_text([{"role": "user", "content": "hi"},
                            {"role": "assistant", "content": "hello"}])
    assert out == "User: hi\n\nAssistant: hello\n\n"


def test_messages_to_text_skips_blank_turns():
    out = messages_to_text([{"role": "user", "content": "  "},
                            {"role": "assistant", "content": "ok"}])
    assert out == "Assistant: ok\n\n"


def test_messages_to_text_empty_list():
    assert messages_to_text([]) == ""


def test_messages_to_text_unknown_role_defaults_to_user():
    out = messages_to_text([{"role": "tool", "content": "result"}])
    assert out == "User: result\n\n"


def _records(n: int, source: str = "src", chars_each: int = 100):
    for i in range(n):
        yield Record(text="x" * chars_each, source=source, lang="en", license="mit")


def test_char_budget_cap_stops_at_budget():
    counter: dict = {}
    # 5 records of 100 chars each; cap at 250 chars -> should stop after the 3rd (>= 250 at 300,
    # but the cap check happens BEFORE accumulating record n, so records already yielded push
    # the running total to/over budget and the next call breaks).
    out = list(char_budget_cap(_records(5), 250, counter))
    assert len(out) == 3
    assert counter["src"]["docs"] == 3
    assert counter["src"]["chars"] == 300
    assert counter["src"]["approx_tokens"] == int(300 / 3.5)


def test_char_budget_cap_uncapped_when_none():
    counter: dict = {}
    out = list(char_budget_cap(_records(5), None, counter))
    assert len(out) == 5
    assert counter["src"]["docs"] == 5
    assert counter["src"]["chars"] == 500


def test_char_budget_cap_tallies_multiple_sources_in_one_counter():
    counter: dict = {}
    import itertools
    combined = itertools.chain(_records(2, source="a"), _records(2, source="b"))
    out = list(char_budget_cap(combined, None, counter))
    assert len(out) == 4
    assert set(counter) == {"a", "b"}
    assert counter["a"]["docs"] == 2 and counter["b"]["docs"] == 2


def test_build_extension_records_empty_cfg_yields_nothing():
    stream, counter = build_extension_records({})
    assert list(stream) == []
    assert counter == {}


def test_build_extension_records_conversation_domain(monkeypatch):
    def fake_load_oasst1(max_examples=None):
        yield {"messages": [{"role": "user", "content": "hi"},
                            {"role": "assistant", "content": "hello"}],
              "source": "oasst1", "license": "apache-2.0"}

    import src.data.sft_sources as sft_sources
    monkeypatch.setattr(sft_sources, "load_oasst1", fake_load_oasst1)

    cfg = {"conversation": {"sources": ["oasst1"], "tokens": None}}
    stream, counter = build_extension_records(cfg)
    records = list(stream)
    assert len(records) == 1
    assert records[0].text == "User: hi\n\nAssistant: hello\n\n"
    assert records[0].source == "oasst1"
    assert counter["oasst1"]["docs"] == 1


def test_build_extension_records_reasoning_domain(monkeypatch):
    def fake_load_openthoughts(max_examples=None):
        yield {"messages": [{"role": "user", "content": "2+2?"},
                            {"role": "assistant", "content": "<think>add</think>4"}],
              "source": "openthoughts", "license": "apache-2.0"}

    import src.data.reasoning_traces as reasoning_traces
    monkeypatch.setattr(reasoning_traces, "load_openthoughts", fake_load_openthoughts)

    cfg = {"reasoning": {"sources": ["openthoughts"], "tokens": None}}
    stream, counter = build_extension_records(cfg)
    records = list(stream)
    assert len(records) == 1
    assert records[0].source == "openthoughts"
    assert "4" in records[0].text
    assert counter["openthoughts"]["docs"] == 1


def test_build_extension_records_code_per_language_cap(monkeypatch):
    def fake_iter_the_stack_dedup(langs):
        (lang,) = list(langs)
        for i in range(10):
            yield Record(text="c" * 50, source="the-stack-dedup", lang=lang, license="mit",
                        meta={"is_code": True})

    import src.data.distill_sources as distill_sources
    monkeypatch.setattr(distill_sources, "iter_the_stack_dedup", fake_iter_the_stack_dedup)

    cfg = {"code": {"source": "the-stack-dedup", "langs": ["python", "rust"],
                    "tokens_per_lang": 100}}   # -> 350 char cap/lang -> 7 docs/lang @ 50 chars
    stream, counter = distill_sources.build_extension_records(cfg)
    records = list(stream)
    assert len(records) == 14
    assert counter["the-stack-dedup"]["docs"] == 14


def test_build_extension_records_unknown_source_raises():
    import pytest

    stream, _counter = build_extension_records({"conversation": {"sources": ["nope"]}})
    with pytest.raises(ValueError):
        list(stream)


# --------------------------------------------------------------------------- #
# Dataset-level license fallback (#176) — open-web-math / library-documentation have no
# per-row `license` field; the-stack-smol's `repository_name` fallback was bogus (not a
# license field). These monkeypatch the lazy `datasets.load_dataset` import site.
# --------------------------------------------------------------------------- #
def test_iter_open_web_math_falls_back_to_dataset_level_license(monkeypatch):
    def fake_load_dataset(*args, **kwargs):
        return [{"text": "some math text"}]  # no "license" key, as on the real dataset

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    records = list(iter_open_web_math())
    assert len(records) == 1
    assert records[0].license == "odc-by"


def test_iter_library_documentation_falls_back_to_dataset_level_license(monkeypatch):
    def fake_load_dataset(*args, **kwargs):
        return [{"doc_content": "some docs text"}]  # no "license" key, as on the real dataset

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    records = list(iter_library_documentation())
    assert len(records) == 1
    assert records[0].license == "cc-by-sa-4.0"


def test_iter_the_stack_smol_uses_real_licenses_field_not_repository_name(monkeypatch):
    def fake_load_dataset(*args, **kwargs):
        return [{"content": "print('hi')", "lang": "python", "licenses": ["mit"],
                 "repository_name": "foo/bar"}]

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    records = list(iter_the_stack_smol(["python"]))
    assert len(records) == 1
    assert records[0].license == "mit"


# --------------------------------------------------------------------------- #
# Code-language normalization/matching (#65, 2026-07-04) — the shared filter used by
# starcoder2-docs / rosetta-code / mceval, whose raw language labels don't match our
# DEFAULT_CODE_LANGS naming 1:1.
# --------------------------------------------------------------------------- #
def test_normalize_code_lang_maps_known_aliases():
    assert _normalize_code_lang("C#") == "c-sharp"
    assert _normalize_code_lang("  CSharp ") == "c-sharp"
    assert _normalize_code_lang("Cpp") == "c++"
    assert _normalize_code_lang("Bash") == "shell"
    assert _normalize_code_lang("YML") == "yaml"


def test_normalize_code_lang_identity_when_already_canonical():
    assert _normalize_code_lang("Python") == "python"
    assert _normalize_code_lang("javascript") == "javascript"


def test_normalize_code_lang_unmapped_label_passes_through_lowercased():
    # Not an error -- an unrecognized/unmapped label just won't match any `wanted` set.
    assert _normalize_code_lang("Brainfuck") == "brainfuck"


def test_code_lang_matches_true_for_alias_and_canonical():
    wanted = {"c-sharp", "python"}
    assert _code_lang_matches("C#", wanted) is True
    assert _code_lang_matches("python", wanted) is True


def test_code_lang_matches_false_for_unwanted_language():
    assert _code_lang_matches("Haskell", {"python", "rust"}) is False


# --------------------------------------------------------------------------- #
# _qa_records — the flat prompt/response bridge shared by the code_problems/code_instruct
# loaders (the analogue of messages_to_text/_messages_records for non-chat sources).
# --------------------------------------------------------------------------- #
def test_qa_records_renders_question_answer_pair():
    rows = [{"q": "What is 2+2?", "a": "4"}]
    out = list(_qa_records(rows, "q", "a", "src", "mit"))
    assert len(out) == 1
    assert out[0].text == "What is 2+2?\n\n4\n\n"
    assert out[0].source == "src"
    assert out[0].license == "mit"


def test_qa_records_skips_rows_missing_either_field():
    rows = [{"q": "", "a": "4"}, {"q": "q?", "a": ""}, {"q": "q2?", "a": "a2"}]
    out = list(_qa_records(rows, "q", "a", "src", "mit"))
    assert len(out) == 1
    assert out[0].text == "q2?\n\na2\n\n"


def test_qa_records_filters_by_language_field():
    rows = [{"q": "q1", "a": "a1", "lang": "Python"},
           {"q": "q2", "a": "a2", "lang": "Haskell"}]
    out = list(_qa_records(rows, "q", "a", "src", "mit",
                           lang_field="lang", keep_langs={"python"}))
    assert len(out) == 1
    assert out[0].text.startswith("q1")


def test_qa_records_per_row_license_overrides_fallback():
    rows = [{"q": "q", "a": "a", "lic": "apache-2.0"}]
    out = list(_qa_records(rows, "q", "a", "src", "mit", license_field="lic"))
    assert out[0].license == "apache-2.0"


def test_qa_records_falls_back_when_license_field_absent():
    rows = [{"q": "q", "a": "a"}]
    out = list(_qa_records(rows, "q", "a", "src", "mit", license_field="lic"))
    assert out[0].license == "mit"


# --------------------------------------------------------------------------- #
# Docs (CHANGED, #65) — starcoder2-documentation replaces library-documentation
# (Python-only). Multilingual: filter by the row's `language` field.
# --------------------------------------------------------------------------- #
def test_iter_starcoder2_documentation_filters_to_selected_langs(monkeypatch):
    def fake_load_dataset(*args, **kwargs):
        return [{"content": "def f(): pass", "language": "Python"},
               {"content": "console.log(1)", "language": "JavaScript"},
               {"content": "class Foo", "language": "C#"}]

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    records = list(iter_starcoder2_documentation(["python", "c-sharp"]))
    assert {r.text for r in records} == {"def f(): pass", "class Foo"}
    assert all(r.license == "apache-2.0" for r in records)
    assert all(r.source == "starcoder2-docs" for r in records)


def test_iter_starcoder2_documentation_skips_rows_without_content(monkeypatch):
    def fake_load_dataset(*args, **kwargs):
        return [{"language": "Python"}]  # no "content" key

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    assert list(iter_starcoder2_documentation(["python"])) == []


# --------------------------------------------------------------------------- #
# Knowledge — wikimedia/structured-wikipedia (English only, #65). Rows have no flat `text`
# field; render_wikipedia_sections walks the decoded `sections` structure.
# --------------------------------------------------------------------------- #
def test_render_wikipedia_sections_walks_nested_has_parts():
    sections = [
        {"name": "History", "has_parts": [
            {"value": "Founded in 1900."},
            {"name": "Early years", "has_parts": [{"text": "It grew quickly."}]},
        ]},
    ]
    out = render_wikipedia_sections(sections)
    assert "History" in out
    assert "Founded in 1900." in out
    assert "Early years" in out
    assert "It grew quickly." in out


def test_render_wikipedia_sections_handles_empty_and_none():
    assert render_wikipedia_sections(None) == ""
    assert render_wikipedia_sections([]) == ""


def test_iter_structured_wikipedia_concatenates_abstract_and_sections(monkeypatch):
    def fake_load_dataset(*args, **kwargs):
        return [{"abstract": "Lead paragraph.",
                "sections": json.dumps([{"name": "Body", "has_parts": [{"value": "More."}]}]),
                "license": "CC-BY-SA-4.0"}]

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    records = list(iter_structured_wikipedia())
    assert len(records) == 1
    assert "Lead paragraph." in records[0].text
    assert "Body" in records[0].text and "More." in records[0].text
    assert records[0].license == "CC-BY-SA-4.0"
    assert records[0].lang == "en"


def test_iter_structured_wikipedia_falls_back_to_dataset_level_license(monkeypatch):
    def fake_load_dataset(*args, **kwargs):
        return [{"abstract": "Lead only.", "sections": None}]  # no "license" key

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    records = list(iter_structured_wikipedia())
    assert len(records) == 1
    assert records[0].license == "cc-by-sa-4.0"


def test_iter_structured_wikipedia_skips_rows_with_no_renderable_text(monkeypatch):
    def fake_load_dataset(*args, **kwargs):
        return [{"abstract": "", "sections": None}]

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    assert list(iter_structured_wikipedia()) == []


# --------------------------------------------------------------------------- #
# Code problems — OpenCodeReasoning (Python-only) / rosetta-code / McEval-Instruct
# (multilingual, #65). KodCode-V1 deliberately excluded (CC BY-NC 4.0).
# --------------------------------------------------------------------------- #
def test_iter_opencodereasoning_maps_input_output(monkeypatch):
    def fake_load_dataset(*args, **kwargs):
        return [{"input": "Sum two numbers.", "output": "def s(a,b): return a+b"}]

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    records = list(iter_opencodereasoning())
    assert len(records) == 1
    assert records[0].text == "Sum two numbers.\n\ndef s(a,b): return a+b\n\n"
    assert records[0].license == "cc-by-4.0"
    assert records[0].source == "opencodereasoning"


def test_iter_rosetta_code_filters_by_language_name(monkeypatch):
    def fake_load_dataset(*args, **kwargs):
        return [{"task_description": "Print hello", "code": "print('hi')",
                "language_name": "Python"},
               {"task_description": "Print hello", "code": "puts 'hi'",
                "language_name": "Ruby"}]

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    records = list(iter_rosetta_code(["python"]))
    assert len(records) == 1
    assert "print('hi')" in records[0].text
    assert records[0].license == "GFDL"


def test_iter_mceval_instruct_filters_by_language(monkeypatch):
    def fake_load_dataset(*args, **kwargs):
        return [{"instruction": "Write a loop", "output": "for i in range(10): pass",
                "language": "Python"},
               {"instruction": "Write a loop", "output": "for(int i=0;i<10;i++){}",
                "language": "Java"}]

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    records = list(iter_mceval_instruct(["python"]))
    assert len(records) == 1
    assert "range(10)" in records[0].text
    assert records[0].license == "cc-by-sa-4.0"


def test_iter_code_problems_chains_multiple_sources(monkeypatch):
    import src.data.distill_sources as distill_sources

    monkeypatch.setattr(distill_sources, "iter_opencodereasoning",
                        lambda: iter([Record(text="a", source="opencodereasoning")]))
    monkeypatch.setattr(distill_sources, "iter_rosetta_code",
                        lambda langs: iter([Record(text="b", source="rosetta-code")]))

    records = list(distill_sources.iter_code_problems(["opencodereasoning", "rosetta-code"]))
    assert {r.text for r in records} == {"a", "b"}


def test_iter_code_problems_unknown_source_raises():
    import pytest

    with pytest.raises(ValueError):
        list(iter_code_problems(["kodcode"]))  # deliberately excluded, not a valid source


# --------------------------------------------------------------------------- #
# Code instruct — OpenCodeInstruct / CodeFeedback (instruction -> code solution, #65).
# --------------------------------------------------------------------------- #
def test_iter_opencodeinstruct_maps_input_output(monkeypatch):
    def fake_load_dataset(*args, **kwargs):
        return [{"input": "Reverse a string.", "output": "def r(s): return s[::-1]"}]

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    records = list(iter_opencodeinstruct())
    assert len(records) == 1
    assert records[0].license == "cc-by-4.0"
    assert records[0].source == "opencodeinstruct"


def test_iter_code_instruct_chains_multiple_sources(monkeypatch):
    import src.data.distill_sources as distill_sources

    monkeypatch.setattr(distill_sources, "iter_opencodeinstruct",
                        lambda: iter([Record(text="a", source="opencodeinstruct")]))
    monkeypatch.setattr(distill_sources, "iter_codefeedback",
                        lambda: iter([Record(text="b", source="codefeedback")]))

    records = list(distill_sources.iter_code_instruct(["opencodeinstruct", "codefeedback"]))
    assert {r.text for r in records} == {"a", "b"}


def test_iter_code_instruct_unknown_source_raises():
    import pytest

    with pytest.raises(ValueError):
        list(iter_code_instruct(["nope"]))


# --------------------------------------------------------------------------- #
# build_extension_records wiring — the new/changed domains (docs, wiki, code_problems,
# code_instruct) and the reasoning openthoughts2 swap (#65, 2026-07-04).
# --------------------------------------------------------------------------- #
def test_build_extension_records_docs_domain_starcoder2(monkeypatch):
    import src.data.distill_sources as distill_sources

    def fake_iter_starcoder2(langs):
        yield Record(text="doc text", source="starcoder2-docs", license="apache-2.0")

    monkeypatch.setattr(distill_sources, "iter_starcoder2_documentation", fake_iter_starcoder2)

    cfg = {"docs": {"source": "starcoder2-documentation", "langs": ["python"], "tokens": None}}
    stream, counter = distill_sources.build_extension_records(cfg)
    records = list(stream)
    assert len(records) == 1
    assert records[0].source == "starcoder2-docs"
    assert counter["starcoder2-docs"]["docs"] == 1


def test_build_extension_records_wiki_domain(monkeypatch):
    import src.data.distill_sources as distill_sources

    def fake_iter_wiki():
        yield Record(text="wiki text", source="wikipedia", license="cc-by-sa-4.0")

    monkeypatch.setattr(distill_sources, "iter_structured_wikipedia", fake_iter_wiki)

    cfg = {"wiki": {"source": "structured-wikipedia", "tokens": None}}
    stream, counter = distill_sources.build_extension_records(cfg)
    records = list(stream)
    assert len(records) == 1
    assert records[0].source == "wikipedia"
    assert counter["wikipedia"]["docs"] == 1


def test_build_extension_records_code_problems_domain(monkeypatch):
    import src.data.distill_sources as distill_sources

    def fake_iter_code_problems(sources, langs=None, max_per_source=None):
        yield Record(text="problem", source="opencodereasoning", license="cc-by-4.0")

    monkeypatch.setattr(distill_sources, "iter_code_problems", fake_iter_code_problems)

    cfg = {"code_problems": {"sources": ["opencodereasoning"], "tokens": None}}
    stream, counter = distill_sources.build_extension_records(cfg)
    records = list(stream)
    assert len(records) == 1
    assert counter["opencodereasoning"]["docs"] == 1


def test_build_extension_records_code_instruct_domain(monkeypatch):
    import src.data.distill_sources as distill_sources

    def fake_iter_code_instruct(sources, max_per_source=None):
        yield Record(text="instr", source="codefeedback", license="apache-2.0")

    monkeypatch.setattr(distill_sources, "iter_code_instruct", fake_iter_code_instruct)

    cfg = {"code_instruct": {"sources": ["codefeedback"], "tokens": None}}
    stream, counter = distill_sources.build_extension_records(cfg)
    records = list(stream)
    assert len(records) == 1
    assert counter["codefeedback"]["docs"] == 1


def test_build_extension_records_reasoning_domain_openthoughts2(monkeypatch):
    def fake_load_openthoughts2(max_examples=None):
        yield {"messages": [{"role": "user", "content": "2+2?"},
                            {"role": "assistant", "content": "<think>add</think>4"}],
              "source": "openthoughts2", "license": "apache-2.0"}

    import src.data.reasoning_traces as reasoning_traces
    monkeypatch.setattr(reasoning_traces, "load_openthoughts2", fake_load_openthoughts2)

    cfg = {"reasoning": {"sources": ["openthoughts2"], "tokens": None}}
    stream, counter = build_extension_records(cfg)
    records = list(stream)
    assert len(records) == 1
    assert records[0].source == "openthoughts2"
    assert counter["openthoughts2"]["docs"] == 1
