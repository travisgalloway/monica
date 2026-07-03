"""Phase A' multi-domain distillation corpus extension sources (#65).

Hermetic — no network. `datasets` calls in `distill_sources.py` stay `# pragma: no cover`;
this only exercises the pure mapper/render/cap/orchestration functions with fabricated rows.
"""

from src.data.corpus import Record
from src.data.distill_sources import (build_extension_records, char_budget_cap,
                                       iter_library_documentation, iter_open_web_math,
                                       iter_the_stack_smol, messages_to_text)


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
