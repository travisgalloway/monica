"""Blend at natural size with per-source epoch counts (#74).

Pure-Python; needs pyarrow only for the shard IO the blend reads from.
"""

import pytest

pytest.importorskip("pyarrow")

from src.data.blend import BlendSpec, blend
from src.data.corpus import Record, write_shards


def _write(tmp_path, name, texts, lang="en"):
    uri = tmp_path / name
    write_shards([Record(t, name, lang=lang) for t in texts], uri)
    return str(uri)


def test_blendspec_passes_clamped():
    spec = BlendSpec(passes={"web": 1, "wiki": 3, "greedy": 99}, default_passes=2,
                     max_passes=4)
    assert spec.passes_for("web") == 1
    assert spec.passes_for("wiki") == 3
    assert spec.passes_for("greedy") == 4          # clamped
    assert spec.passes_for("unknown") == 2         # default


def test_blend_honors_epoch_counts(tmp_path):
    web = _write(tmp_path, "web", ["w0", "w1"])
    wiki = _write(tmp_path, "wiki", ["k0"])
    spec = BlendSpec(passes={"web": 1, "wiki": 3})
    out = list(blend({"web": web, "wiki": wiki}, spec, seed=0))
    texts = [r.text for r in out]
    assert texts.count("w0") == 1 and texts.count("w1") == 1   # web once
    assert texts.count("k0") == 3                              # wiki thrice


def test_blend_priority_language_oversample(tmp_path):
    code = _write(tmp_path, "stack", ["ts_a", "py_b"], lang="typescript")
    # all docs are typescript here; 1 base pass + 2 priority passes = 3 total
    spec = BlendSpec(passes={"stack": 1}, priority_langs={"typescript": 2})
    out = [r.text for r in blend({"stack": code}, spec, seed=1)]
    assert out.count("ts_a") == 3 and out.count("py_b") == 3


def test_blend_deterministic(tmp_path):
    a = _write(tmp_path, "a", [f"a{i}" for i in range(20)])
    b = _write(tmp_path, "b", [f"b{i}" for i in range(20)])
    spec = BlendSpec(passes={"a": 2, "b": 1})
    r1 = [r.text for r in blend({"a": a, "b": b}, spec, seed=7, buffer_size=8)]
    r2 = [r.text for r in blend({"a": a, "b": b}, spec, seed=7, buffer_size=8)]
    assert r1 == r2 and sorted(r1) == sorted(r2)
    assert len(r1) == 20 * 2 + 20            # 2 passes of a, 1 of b
