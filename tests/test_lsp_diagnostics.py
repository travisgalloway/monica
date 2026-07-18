"""Tests for `src/lsp/diagnostics.py` — pure string/parsing logic, no `tsc` needed.

Real-compiler format pinning lives in `test_lsp_tsc.py`; this file only exercises
`diagnostics.py`'s own primitives against hand-built inputs.
"""

from __future__ import annotations

from src.lsp.diagnostics import (Diagnostic, MODULE_RESOLUTION_CODES, SUPPRESSION_RE,
                                  close_open_delimiters, drop_codes, filter_diagnostics,
                                  is_incomplete, is_source_balanced, line_col_to_offset,
                                  parse_tsc_output, statement_boundary, strip_suggestion)


# --------------------------------------------------------------------------- #
# parse_tsc_output
# --------------------------------------------------------------------------- #

def test_parse_parenthesized_format():
    source = "const x: number = 'hi';\n"
    output = "snippet.ts(1,7): error TS2322: Type 'string' is not assignable to type 'number'.\n"
    diags = parse_tsc_output(output, source)
    assert len(diags) == 1
    d = diags[0]
    assert d.code == "TS2322"
    assert d.line == 1 and d.col == 7
    assert d.offset == line_col_to_offset(source, 1, 7)
    assert "not assignable" in d.message


def test_parse_pretty_fallback_format():
    source = "const x: number = 'hi';\n"
    output = "snippet.ts:1:7 - error TS2322: Type 'string' is not assignable to type 'number'.\n"
    diags = parse_tsc_output(output, source)
    assert len(diags) == 1
    assert diags[0].code == "TS2322"


def test_parse_ignores_non_diagnostic_lines():
    source = "const x = 1;\n"
    output = "\nsnippet.ts(1,1): error TS9999: bogus\nFound 1 error.\n"
    diags = parse_tsc_output(output, source)
    assert len(diags) == 1
    assert diags[0].code == "TS9999"


def test_parse_multiple_diagnostics_in_order():
    source = "a\nb\n"
    output = ("snippet.ts(1,1): error TS1001: first\n"
              "snippet.ts(2,1): error TS1002: second\n")
    diags = parse_tsc_output(output, source)
    assert [d.code for d in diags] == ["TS1001", "TS1002"]


def test_line_col_to_offset_multiline():
    source = "abc\ndef\nghi"
    assert line_col_to_offset(source, 1, 1) == 0
    assert line_col_to_offset(source, 2, 1) == 4
    assert line_col_to_offset(source, 3, 3) == 10


def test_line_col_to_offset_bmp_non_ascii():
    # 'é' is a single UTF-16 code unit (BMP), so this behaves like plain ASCII.
    source = "const é = 1;\nconst y = 2;"
    # 'y' is at index 6 within line 2 ("const y..."), so 1-indexed col 7.
    assert line_col_to_offset(source, 2, 7) == source.index("y = 2")


def test_line_col_to_offset_astral_surrogate_pair():
    # An emoji outside the BMP is 1 Python str character but 2 UTF-16 code units --
    # a flat col-1 mapping would land one character too far right after it.
    source = "const s = \"\U0001F600\"; const after = 1;"
    after_char_idx = source.index("after")
    # tsc's column count: everything before "after" in UTF-16 units, +1 (1-indexed).
    utf16_units_before = len(source[:after_char_idx].encode("utf-16-le")) // 2
    assert line_col_to_offset(source, 1, utf16_units_before + 1) == after_char_idx


# --------------------------------------------------------------------------- #
# is_incomplete (TS1xxx)
# --------------------------------------------------------------------------- #

def test_is_incomplete_ts1xxx():
    assert is_incomplete("TS1003")
    assert is_incomplete("TS1005")
    assert is_incomplete("TS1109")


def test_is_incomplete_false_for_others():
    assert not is_incomplete("TS2339")
    assert not is_incomplete("TS2304")
    assert not is_incomplete("TS2554")


# --------------------------------------------------------------------------- #
# filter_diagnostics: TS1xxx drop, frontier drop, generation_start clamp
# --------------------------------------------------------------------------- #

def _diag(code, offset, message="msg"):
    return Diagnostic(code=code, line=1, col=offset + 1, message=message, offset=offset)


def test_filter_drops_ts1xxx():
    diags = [_diag("TS1003", 5), _diag("TS2339", 5)]
    out = filter_diagnostics(diags, frontier=100, generation_start=0)
    assert [d.code for d in out] == ["TS2339"]


def test_filter_drops_at_or_past_frontier():
    diags = [_diag("TS2339", 10)]
    assert filter_diagnostics(diags, frontier=10, generation_start=0) == []  # >= frontier dropped
    kept = filter_diagnostics(diags, frontier=11, generation_start=0)
    assert len(kept) == 1 and kept[0].offset == 10  # < frontier kept


def test_filter_clamps_offset_up_to_generation_start():
    # The arity-005 regression: a real diagnostic anchored inside the PROMPT region
    # (offset < generation_start) must clamp up to generation_start, not be dropped
    # and not go negative.
    diags = [_diag("TS2554", 5)]
    out = filter_diagnostics(diags, frontier=100, generation_start=20)
    assert len(out) == 1
    assert out[0].offset == 20
    assert out[0].code == "TS2554"


def test_filter_clamp_is_noop_when_offset_already_past_generation_start():
    diags = [_diag("TS2339", 30)]
    out = filter_diagnostics(diags, frontier=100, generation_start=20)
    assert out[0].offset == 30


def test_filter_never_produces_negative_offset():
    diags = [_diag("TS2554", 0)]
    out = filter_diagnostics(diags, frontier=100, generation_start=0)
    assert out[0].offset == 0
    assert out[0].offset >= 0


# --------------------------------------------------------------------------- #
# is_source_balanced + the control-flow-completeness gate (#199 Stage A)
#
# Real finding, confirmed against the live typescript-language-server binary on
# HumanEval-TS block-budget generation: a not-yet-finished function body (more
# segments still legitimately coming) triggers TS2355/TS2366 ("function lacks a
# return on all paths") from a language server's error-recovery parsing, which
# `tsc`'s batch compile never gets far enough to emit on the same unclosed text.
# Before this gate, an ts_lsp-backed slow-hard run measured 100% over-repair on
# otherwise-correct real code (`--limit 5` smoke, eval_lsp_humaneval.py).
# --------------------------------------------------------------------------- #

def test_is_source_balanced_true_for_complete_code():
    assert is_source_balanced("function f(): boolean {\n  return true;\n}\n")


def test_is_source_balanced_false_for_open_function_body():
    # The for-loop is closed but the enclosing function is not -- exactly the
    # HumanEval-TS block-budget mid-generation state that triggered the bug.
    src = "function f(arr: number[]): boolean {\n  for (let i = 0; i < arr.length; i++) {\n  }\n"
    assert not is_source_balanced(src)


def test_filter_drops_control_flow_completeness_when_source_unbalanced():
    # TS2366 anchored well before the frontier -- would pass every OTHER gate --
    # but the source is still missing its closing brace, so it must be dropped.
    src = "function f(): boolean {\n  for (let i = 0; i < 1; i++) {\n  }\n"
    diags = [_diag("TS2366", 10, message="Function lacks ending return statement")]
    out = filter_diagnostics(diags, frontier=len(src), generation_start=0, source=src)
    assert out == []


def test_filter_keeps_control_flow_completeness_when_source_balanced():
    # Same code, but the function IS fully closed -- now it's a real defect.
    src = "function f(): boolean {\n  for (let i = 0; i < 1; i++) {\n  }\n}\n"
    diags = [_diag("TS2366", 10, message="Function lacks ending return statement")]
    out = filter_diagnostics(diags, frontier=len(src), generation_start=0, source=src)
    assert [d.code for d in out] == ["TS2366"]


def test_filter_control_flow_gate_is_noop_without_source():
    # Every pre-#199-Stage-A caller (e.g. test_lsp_tsc.py) never passes `source` --
    # the new gate must not activate and change their behavior.
    diags = [_diag("TS2366", 10)]
    out = filter_diagnostics(diags, frontier=100, generation_start=0)
    assert [d.code for d in out] == ["TS2366"]


def test_filter_unaffected_codes_pass_through_regardless_of_balance():
    src = "function f(arr: number[]): boolean {\n  for (let i = 0; i < arr.length; i++) {\n  }\n"
    diags = [_diag("TS2339", 5)]
    out = filter_diagnostics(diags, frontier=len(src), generation_start=0, source=src)
    assert [d.code for d in out] == ["TS2339"]


# --------------------------------------------------------------------------- #
# strip_suggestion
# --------------------------------------------------------------------------- #

def test_strip_suggestion_removes_did_you_mean():
    d = _diag("TS2304", 0, message="Cannot find name 'visitorName'. Did you mean 'userName'?")
    stripped = strip_suggestion(d)
    assert "Did you mean" not in stripped.message
    assert stripped.message.startswith("Cannot find name 'visitorName'")


def test_strip_suggestion_noop_without_suggestion():
    d = _diag("TS2339", 0, message="Property 'gorblak' does not exist on type 'User'.")
    assert strip_suggestion(d).message == d.message


def test_suppression_re_matches_ts_ignore_and_as_any():
    assert SUPPRESSION_RE.search("// @ts-ignore\nconsole.log(u.gorblak);")
    assert SUPPRESSION_RE.search("const v = u as any;")
    assert not SUPPRESSION_RE.search("console.log(u.name);")


# --------------------------------------------------------------------------- #
# close_open_delimiters — string/template/comment aware
# --------------------------------------------------------------------------- #

def test_close_open_delimiters_noop_on_balanced():
    src = "console.log(u.name);\n"
    assert close_open_delimiters(src) == src


def test_close_open_delimiters_brace_in_string_is_not_a_real_delimiter():
    src = 'const t = ("{ not real'
    closed = close_open_delimiters(src)
    # The `{` inside the (unterminated) string must NOT be tracked as a real open
    # bracket — only the string itself (innermost) and the paren need closing.
    assert closed == src + '")'


def test_close_open_delimiters_brace_in_line_comment_ignored():
    src = "foo({ // comment with a } brace"
    closed = close_open_delimiters(src)
    assert closed == src + "\n})"


def test_close_open_delimiters_brace_in_block_comment_ignored():
    src = "foo({ /* a } b */ bar("
    closed = close_open_delimiters(src)
    # Nesting outer->inner is paren1 > brace1 > paren2 (the comment's `}` doesn't
    # count); closers come out innermost-first: paren2, brace1, paren1.
    assert closed == src + ")})"


def test_close_open_delimiters_unterminated_block_comment():
    src = "foo(/* never closed"
    closed = close_open_delimiters(src)
    assert closed == src + "*/)"


def test_close_open_delimiters_template_literal_expr():
    src = "const s = `hello ${user."
    closed = close_open_delimiters(src)
    assert closed == src + "}`"


def test_close_open_delimiters_string_inside_template_expr():
    src = "const s = `${'a"
    closed = close_open_delimiters(src)
    assert closed == src + "'}`"


def test_close_open_delimiters_nested_brackets():
    src = "const arr = [1, {a: (2"
    closed = close_open_delimiters(src)
    assert closed == src + ")}]"


# --------------------------------------------------------------------------- #
# statement_boundary
# --------------------------------------------------------------------------- #

def test_statement_boundary_semicolon_at_depth_zero():
    text = "name;\n"
    assert statement_boundary(text) == text.index(";") + 1


def test_statement_boundary_none_when_absent():
    assert statement_boundary("console.log(u.name") is None


def test_statement_boundary_ignores_semicolon_inside_call():
    # A `;` cannot appear inside a call in TS, but do check bracket depth isn't
    # fooled by a newline inside an open paren.
    text = "foo(\n  1,\n  2\n);\n"
    boundary = statement_boundary(text)
    assert boundary == text.index(");") + 2


def test_statement_boundary_ignores_semicolon_in_string():
    text = 'const s = "a;b";\n'
    boundary = statement_boundary(text)
    assert boundary == text.index('";') + 2  # the real terminator, not the one in the string


def test_statement_boundary_newline_counts_at_depth_zero():
    text = "const x = 1\nconst y = 2;\n"
    assert statement_boundary(text) == text.index("\n") + 1


# --- module-resolution filter (#201 confound (a)) --------------------------- #

def test_module_resolution_codes_membership():
    # The unresolved-import family the over-repair probe must ignore everywhere.
    assert "TS2307" in MODULE_RESOLUTION_CODES        # cannot find module
    assert "TS2305" in MODULE_RESOLUTION_CODES        # no exported member
    # A genuine type error is NOT in the ignore set.
    assert "TS2339" not in MODULE_RESOLUTION_CODES     # property does not exist


def test_drop_codes_filters_only_the_named_codes():
    diags = [
        Diagnostic(code="TS2307", line=1, col=1, message="cannot find module", offset=0),
        Diagnostic(code="TS2339", line=2, col=5, message="no such property", offset=20),
        Diagnostic(code="TS2305", line=3, col=1, message="no exported member", offset=40),
    ]
    wrapped = drop_codes(lambda src: list(diags), MODULE_RESOLUTION_CODES)
    kept = wrapped("irrelevant source")
    assert [d.code for d in kept] == ["TS2339"]   # only the real error survives


def test_drop_codes_passes_everything_through_when_no_match():
    diags = [Diagnostic(code="TS2339", line=1, col=1, message="x", offset=0)]
    wrapped = drop_codes(lambda src: list(diags), MODULE_RESOLUTION_CODES)
    assert [d.code for d in wrapped("s")] == ["TS2339"]
