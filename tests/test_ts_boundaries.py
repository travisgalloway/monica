"""Tests for the tree-sitter true-top-level-boundary extractor (#201).

These prove confound (b) is fixed: a cut at a returned boundary can never land inside
an unclosed multi-line construct. Skipped wholesale without the optional tree-sitter
toolchain (installed via the `[eval]` extra).
"""

from __future__ import annotations

import pytest

from src.lsp.ts_boundaries import (
    first_boundary_in_range, top_level_boundaries, tree_sitter_available,
)

pytestmark = pytest.mark.skipif(not tree_sitter_available(),
                                 reason="tree-sitter / tree-sitter-typescript not installed")


def test_boundaries_land_after_complete_top_level_statements():
    src = ('import x from "y";\n'
           'const a = 1;\n'
           'function f(n: number): number {\n'
           '  return n + 1;\n'
           '}\n')
    bounds = top_level_boundaries(src)
    # A boundary sits right after each of the three complete statements.
    assert src[: bounds[0]] == 'import x from "y";'
    assert src[: bounds[1]] == 'import x from "y";\nconst a = 1;'
    assert src[: bounds[2]].endswith('return n + 1;\n}')
    # Every boundary is the end of a whole statement -- the char just before is never
    # inside an open brace with no matching close in the prefix.
    for b in bounds:
        prefix = src[:b]
        assert prefix.count("{") == prefix.count("}"), f"unbalanced braces at cut {b}: {prefix!r}"


def test_no_boundary_inside_an_unclosed_interface():
    # The interface is never closed -> its node is an error -> no boundary inside it.
    src = ('const a = 1;\n'
           'interface Foo {\n'
           '  bar: number;\n'
           '  baz: string;\n')
    bounds = top_level_boundaries(src)
    # Only the complete `const a = 1;` yields a boundary; nothing inside the interface.
    assert bounds == [len('const a = 1;')]
    for b in bounds:
        assert "interface" not in src[:b]


def test_no_boundary_inside_a_multiline_object_or_function_body():
    # A depth-0 newline exists after `bar: 1,` but it is INSIDE the object literal;
    # the naive char-scanner would have cut there. tree-sitter must not.
    src = ('const cfg = {\n'
           '  bar: 1,\n'
           '  baz: 2,\n'
           '};\n'
           'const done = true;\n')
    bounds = top_level_boundaries(src)
    # First boundary is the END of the whole object declaration, not mid-object.
    assert src[: bounds[0]] == 'const cfg = {\n  bar: 1,\n  baz: 2,\n};'
    # None of the boundaries land between the object's braces.
    for b in bounds:
        prefix = src[:b]
        assert prefix.count("{") == prefix.count("}")


def test_first_boundary_in_range_respects_bounds():
    src = ('const a = 1;\n'          # boundary at 12
           'const bb = 22;\n'        # boundary at 27
           'const ccc = 333;\n')     # boundary at 44
    # Below-range boundaries are skipped; the first in-range one is returned.
    assert first_boundary_in_range(src, min_chars=15, max_chars=40) == 27
    # Nothing in range -> None (all boundaries past max).
    assert first_boundary_in_range(src, min_chars=100, max_chars=200) is None


def test_pure_comment_and_empty_source_yield_no_boundary():
    assert top_level_boundaries("") == []
    assert top_level_boundaries("// just a comment\n") == []
