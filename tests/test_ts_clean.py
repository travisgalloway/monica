"""Tests for `src/data/ts_clean.py` (#193 Stage 4 -- the LSP-clean filter).

The clean-rate math and MODULE_RESOLUTION_CODES-ignore behavior are exercised with a stub
tsc runner (canned `.codes()`), no toolchain required. A small real-`tsc` block is guarded
by `resolve_tsc() is None` (mirrors `tests/test_lsp_tsc.py`).
"""

from __future__ import annotations

import pytest

from src.data.corpus import Record
from src.data.ts_clean import CleanRateStats, tsc_clean
from src.lsp.diagnostics import MODULE_RESOLUTION_CODES
from src.lsp.tsc import TscRunner, resolve_tsc


class _StubTsc:
    """Canned `.codes()` keyed by the record's text."""

    def __init__(self, codes_by_text: dict, raise_on: frozenset = frozenset()):
        self.codes_by_text = codes_by_text
        self.raise_on = raise_on
        self.calls = []

    def codes(self, source: str):
        self.calls.append(source)
        if source in self.raise_on:
            raise RuntimeError("tsc blew up")
        return list(self.codes_by_text.get(source, []))


def _records(texts):
    return [Record(text=t, source="stack-v2", lang="typescript", license="mit") for t in texts]


# --- clean-rate math -------------------------------------------------------------------
def test_tsc_clean_keeps_only_zero_diagnostic_records():
    texts = ["clean one", "dirty one", "clean two"]
    stub = _StubTsc({"dirty one": ["TS2339"]})
    stats = CleanRateStats()
    out = list(tsc_clean(_records(texts), tsc_runner=stub, stats=stats))
    assert [r.text for r in out] == ["clean one", "clean two"]
    assert stats.as_dict() == {"n_seen": 3, "n_clean": 2, "n_dirty": 1, "n_error": 0}


def test_tsc_clean_ignores_module_resolution_codes_by_default():
    # A file with only an unresolved-import diagnostic is still "clean" -- the
    # load-bearing lesson from build_clean_prefix_set.py.
    code = next(iter(MODULE_RESOLUTION_CODES))
    stub = _StubTsc({"imports stuff": [code]})
    stats = CleanRateStats()
    out = list(tsc_clean(_records(["imports stuff"]), tsc_runner=stub, stats=stats))
    assert len(out) == 1
    assert stats.n_clean == 1 and stats.n_dirty == 0


def test_tsc_clean_module_resolution_ignore_is_toggleable():
    code = next(iter(MODULE_RESOLUTION_CODES))
    stub = _StubTsc({"imports stuff": [code]})
    stats = CleanRateStats()
    out = list(tsc_clean(_records(["imports stuff"]), tsc_runner=stub,
                         ignore_module_resolution=False, stats=stats))
    assert out == []
    assert stats.n_dirty == 1 and stats.n_clean == 0


def test_tsc_clean_mixed_codes_kept_only_if_all_ignorable():
    code = next(iter(MODULE_RESOLUTION_CODES))
    stub = _StubTsc({"mixed": [code, "TS2339"]})   # one ignorable, one real
    out = list(tsc_clean(_records(["mixed"]), tsc_runner=stub))
    assert out == []   # TS2339 survives the ignore filter -> still dirty


def test_tsc_clean_counts_exceptions_as_errors_and_skips_them():
    stub = _StubTsc({}, raise_on=frozenset({"boom"}))
    stats = CleanRateStats()
    out = list(tsc_clean(_records(["boom", "fine"]), tsc_runner=stub, stats=stats))
    assert [r.text for r in out] == ["fine"]
    assert stats.as_dict() == {"n_seen": 2, "n_clean": 1, "n_dirty": 0, "n_error": 1}


def test_tsc_clean_works_without_stats():
    stub = _StubTsc({"dirty": ["TS2339"]})
    out = list(tsc_clean(_records(["clean", "dirty"]), tsc_runner=stub))
    assert [r.text for r in out] == ["clean"]


def test_clean_rate_stats_as_dict():
    stats = CleanRateStats(n_seen=10, n_clean=7, n_dirty=2, n_error=1)
    assert stats.as_dict() == {"n_seen": 10, "n_clean": 7, "n_dirty": 2, "n_error": 1}


# --- real tsc (skipped on a node-less host) --------------------------------------------
pytestmark_real = pytest.mark.skipif(resolve_tsc() is None,
                                     reason="no node/tsc toolchain on this host")


@pytestmark_real
def test_tsc_clean_with_real_tsc_runner():
    with TscRunner() as runner:
        recs = _records([
            "export function add(a: number, b: number): number { return a + b; }\n",
            "export function bad(): number { return gorblak; }\n",
        ])
        stats = CleanRateStats()
        out = list(tsc_clean(recs, tsc_runner=runner, stats=stats))
        assert len(out) == 1
        assert stats.n_seen == 2 and stats.n_clean == 1 and stats.n_dirty == 1
