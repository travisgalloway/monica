"""Tests for `src/lsp/prettier.py` (#193 Stage 3).

Pure-logic behavior (resolve, graceful-fallback-on-failure) is exercised offline with no
real `prettier` binary. A small block of real-toolchain tests is guarded by
`resolve_prettier() is None` (mirrors `tests/test_lsp_tsc.py`'s pattern for `tsc`).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

import src.lsp.prettier as prettier_mod
from src.lsp.prettier import LOCAL_PRETTIER, PrettierRunner, format_source, resolve_prettier


# --- offline: resolve_prettier graceful-skip idiom ------------------------------------
def test_resolve_prettier_none_when_local_binary_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(prettier_mod, "LOCAL_PRETTIER", tmp_path / "no" / "prettier")
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/node")
    assert resolve_prettier() is None


def test_resolve_prettier_none_when_node_missing(tmp_path, monkeypatch):
    fake_bin = tmp_path / "prettier"
    fake_bin.write_text("#!/usr/bin/env node\n")
    monkeypatch.setattr(prettier_mod, "LOCAL_PRETTIER", fake_bin)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert resolve_prettier() is None


def test_resolve_prettier_returns_argv_when_available(tmp_path, monkeypatch):
    fake_bin = tmp_path / "prettier"
    fake_bin.write_text("#!/usr/bin/env node\n")
    monkeypatch.setattr(prettier_mod, "LOCAL_PRETTIER", fake_bin)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/node")
    assert resolve_prettier() == [str(fake_bin)]


def test_local_prettier_path_is_under_set_dir():
    from src.lsp.tsc import SET_DIR
    assert SET_DIR in LOCAL_PRETTIER.parents
    assert LOCAL_PRETTIER.parts[-3:] == ("node_modules", ".bin", "prettier")


# --- offline: format_source never drops/corrupts on failure ---------------------------
def test_format_source_returns_original_on_missing_binary():
    src = "const x=1"
    out = format_source(src, ["/definitely/not/a/real/prettier/binary"])
    assert out == src


def test_format_source_returns_original_on_timeout():
    # A python one-liner that sleeps: extra argv (`--parser typescript`, appended by
    # format_source) lands in sys.argv, which the script never reads, so it reliably
    # blocks past the near-zero timeout instead of erroring on unexpected args.
    src = "const x = 1;"
    out = format_source(src, [sys.executable, "-c", "import time; time.sleep(5)"],
                        timeout=0.05)
    assert out == src


def test_format_source_returns_original_on_nonzero_exit():
    # `false` always exits 1 and writes nothing to stdout.
    src = "const x = 1;"
    out = format_source(src, ["false"])
    assert out == src


def test_prettier_runner_requires_a_resolvable_toolchain(monkeypatch):
    monkeypatch.setattr(prettier_mod, "resolve_prettier", lambda: None)
    with pytest.raises(RuntimeError):
        PrettierRunner()


def test_prettier_runner_uses_injected_argv():
    runner = PrettierRunner(["/definitely/not/a/real/prettier/binary"])
    src = "const x=1"
    assert runner.format(src) == src   # falls back to unchanged on failure


# --- real prettier (skipped on a host with no local toolchain) ------------------------
pytestmark_real = pytest.mark.skipif(resolve_prettier() is None,
                                     reason="no prettier toolchain on this host "
                                            "(run `npm install` in eval_sets/ts_error_injection)")


@pytestmark_real
def test_real_prettier_formats_source():
    argv = resolve_prettier()
    out = format_source("const x=1", argv)
    assert out.strip() == "const x = 1;"


@pytestmark_real
def test_real_prettier_runner_end_to_end():
    runner = PrettierRunner(resolve_prettier())
    out = runner.format("function f(a,b){return a+b}")
    assert "function f(a, b)" in out
