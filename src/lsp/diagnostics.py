"""`tsc` diagnostic parsing and the string-level primitives the repair loop needs.

The pinned `typescript@5.9.3` under `--pretty false` (non-TTY) emits one line per
diagnostic in the **parenthesized** form:

    snippet.ts(3,15): error TS2339: Property 'gorblak' does not exist on type 'User'.

Not `file:line:col: message` — that format (what issue #199's text originally
specified) never matches this compiler's real output; a parser built to it silently
reports zero diagnostics for everything. See `docs/design/12-lsp-in-the-loop.md`.

`offset` is a **character** offset into the Python `str` source. `tsc` (and LSP
servers generally) report columns as UTF-16 code units; `line_col_to_offset` handles
the general case (`_utf16_col_to_char_index` accounts for astral-plane characters
that are one Python `str` character but a 2-unit UTF-16 surrogate pair) rather than
asserting the source is ASCII-only — the #194 eval set happens to be 100% ASCII, but
a model's free-running completion, or the #199 Stage A HumanEval-TS bodies, are not
guaranteed to stay that way.

ABOVE THE SEAM — stdlib only. No `mlx`/`torch` import anywhere in this module
(guarded by `tests/test_import_guard.py`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Iterator, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

# `<file>(<line>,<col>): error <code>: <message>` — tsc's --pretty false format.
_DIAG_RE = re.compile(
    r"^.+\((?P<line>\d+),(?P<col>\d+)\): error (?P<code>TS\d+): (?P<message>.*)$"
)

# Fallback: the human ("pretty") form, `file:line:col - error TSxxxx: message`,
# kept only so a diagnostics-consuming test/caller run without `--pretty false`
# doesn't silently see zero diagnostics too. Not exercised by the harness itself
# (`TscRunner` always passes `--pretty false`).
_DIAG_RE_PRETTY = re.compile(
    r"^.+:(?P<line>\d+):(?P<col>\d+) - error (?P<code>TS\d+): (?P<message>.*)$"
)

# Reward-hacking suppressions: these make `tsc` "clean" without fixing anything.
SUPPRESSION_RE = re.compile(r"@ts-ignore|@ts-expect-error|\bas\s+any\b")

# The TS1xxx family: syntax-incompleteness codes (e.g. TS1003 "Identifier
# expected"). An in-progress autoregressive prefix can trigger any of ~200 of
# these for the trivial reason that generation hasn't finished the statement yet,
# not because of a real bug — they must never count as a "real" diagnostic.
_TS1XXX_RE = re.compile(r"^TS1\d{3}$")

# "Did you mean 'x'?" — tsc's spelling-correction suggestion, stripped by the
# --strip-suggestions ablation so the harness can't just copy the compiler's answer.
_SUGGESTION_RE = re.compile(r"\s*Did you mean\b[^.]*\.?\s*$")


@dataclass(frozen=True)
class Diagnostic:
    code: str
    line: int
    col: int
    message: str
    offset: int
    # #199 Stage A: which oracle arm produced this diagnostic ("ts" | "opengrep"),
    # and its LSP severity (1=Error, 2=Warning, 3=Information, 4=Hint). Defaulted so
    # every existing 5-positional-arg call site (tsc.py, the #194 test suite) keeps
    # working unchanged -- both new fields describe LSP-sourced diagnostics only.
    source: str = "ts"
    severity: int = 1


def _line_start_offsets(source: str) -> List[int]:
    """Character offset of the start of each 1-indexed source line."""
    starts = [0]
    for i, ch in enumerate(source):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def _utf16_col_to_char_index(line: str, col: int) -> int:
    """1-indexed UTF-16-code-unit column -> 0-indexed Python `str` character index
    within `line`. Most characters are 1 UTF-16 unit; astral-plane characters
    (outside the Basic Multilingual Plane, e.g. many emoji) encode as a 2-unit
    surrogate pair in UTF-16 but are a single Python `str` character, so a flat
    `col - 1` offset silently mis-maps once one appears anywhere on the line
    (`tsc`'s columns are always UTF-16 units, regardless of what generated the
    source). The #194 eval set is 100% ASCII by construction, but a model's
    free-running block-budget completion is not guaranteed to stay ASCII, so this
    module must handle the general case rather than assert it away.
    """
    units = 0
    for i, ch in enumerate(line):
        if units >= col - 1:
            return i
        units += 2 if ord(ch) > 0xFFFF else 1
    return len(line)


def line_col_to_offset(source: str, line: int, col: int) -> int:
    """1-indexed (line, col) -> 0-indexed character offset into `source`."""
    starts = _line_start_offsets(source)
    line_text = source.split("\n")[line - 1]
    return starts[line - 1] + _utf16_col_to_char_index(line_text, col)


def parse_tsc_output(output: str, source: str) -> List[Diagnostic]:
    """Parse `tsc --pretty false` output into `Diagnostic`s, offsets resolved
    against `source` (the exact text that was compiled). Falls back to the
    "pretty" `file:line:col - error TSxxxx:` form for a line that doesn't match
    the parenthesized form. Non-diagnostic lines (blank lines, the trailing
    "Found N errors." summary, wrapped message continuations) are ignored.
    """
    diags: List[Diagnostic] = []
    for line in output.splitlines():
        m = _DIAG_RE.match(line) or _DIAG_RE_PRETTY.match(line)
        if not m:
            continue
        ln = int(m.group("line"))
        col = int(m.group("col"))
        diags.append(Diagnostic(
            code=m.group("code"), line=ln, col=col,
            message=m.group("message"), offset=line_col_to_offset(source, ln, col),
        ))
    return diags


def is_incomplete(code: str) -> bool:
    """True for the TS1xxx family (syntax-incompleteness codes)."""
    return bool(_TS1XXX_RE.match(code))


# --------------------------------------------------------------------------- #
# Frontier filtering + prompt-region clamping
# --------------------------------------------------------------------------- #

def filter_diagnostics(diags: Sequence[Diagnostic], *, frontier: int,
                        generation_start: int) -> List[Diagnostic]:
    """The repair loop's diagnostic gate.

    Drops:
      * every TS1xxx diagnostic (in-progress-generation syntax noise, never real).
      * every diagnostic at or past `frontier` (the char offset of the START of the
        last emitted token) — "an error is only real once the model has committed to
        it by emitting at least one more token" (see the design doc's frontier
        section). One token of lag is the theoretical minimum detection latency.

    Clamps the offset of every surviving diagnostic up to `generation_start` (never
    below it, never negative): some diagnostics anchor inside the *prompt* even
    though they're caused by the completion (e.g. `subtract(` + `10);` anchors
    TS2554 at the call expression, inside the prompt) — real, but not attributable
    to any generated token, so rollback must not walk past the start of generation.
    """
    out: List[Diagnostic] = []
    for d in diags:
        if is_incomplete(d.code):
            continue
        if d.offset >= frontier:
            continue
        clamped = max(d.offset, generation_start)
        out.append(d if clamped == d.offset else replace(d, offset=clamped))
    return out


def strip_suggestion(diag: Diagnostic) -> Diagnostic:
    """Return `diag` with a trailing "Did you mean 'x'?" clause removed from its
    message (the `--strip-suggestions` ablation — see the design doc's risks
    section on `tsc` leaking the answer)."""
    stripped = _SUGGESTION_RE.sub("", diag.message).rstrip()
    return diag if stripped == diag.message else replace(diag, message=stripped)


# --------------------------------------------------------------------------- #
# String/template/comment-aware delimiter scanner
# --------------------------------------------------------------------------- #

_PAIRS = {"(": ")", "[": "]", "{": "}"}
_CLOSERS = (")", "]", "}")
_QUOTES = ("'", '"', "`")


def _scan(text: str) -> Iterator[Tuple[int, str, List[str]]]:
    """Yield `(index, char, stack)` for each character of `text`, where `stack` is
    the open-delimiter stack *after* consuming that character (top-of-stack is one
    of: a bracket closer `)`/`]`/`}`; a quote char `'`/`"`/`` ` `` for an open
    string/template; `'/*'`/`'//'` for an open comment). Shared by
    `close_open_delimiters` and `statement_boundary` so both agree on what counts
    as "inside a string/template/comment" vs. "real code at depth 0".

    A two-character token (`\\x` escape, `${`, `*/`, `//`) only yields its first
    index — the second character is consumed without its own yield, which is
    exactly what keeps e.g. a backslash-escaped newline inside a string from being
    mistaken for a statement-terminating newline.
    """
    stack: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        top = stack[-1] if stack else None
        consumed = 1

        if top in ("'", '"'):
            if c == "\\" and i + 1 < n:
                consumed = 2
            elif c == top:
                stack.pop()
        elif top == "`":
            if c == "\\" and i + 1 < n:
                consumed = 2
            elif c == "`":
                stack.pop()
            elif c == "$" and i + 1 < n and text[i + 1] == "{":
                stack.append("}")
                consumed = 2
        elif top == "/*":
            if c == "*" and i + 1 < n and text[i + 1] == "/":
                stack.pop()
                consumed = 2
        elif top == "//":
            if c == "\n":
                stack.pop()
        else:
            # "Real code" context (stack empty or top is a bracket closer).
            if c == "/" and i + 1 < n and text[i + 1] == "/":
                stack.append("//")
                consumed = 2
            elif c == "/" and i + 1 < n and text[i + 1] == "*":
                stack.append("/*")
                consumed = 2
            elif c in _QUOTES:
                stack.append(c)
            elif c in _PAIRS:
                stack.append(_PAIRS[c])
            elif c in _CLOSERS:
                if stack and stack[-1] == c:
                    stack.pop()
                # else: unmatched closer in source — best-effort, leave stack as is.

        yield i, c, list(stack)
        i += consumed


def close_open_delimiters(source: str) -> str:
    """Append the minimal closing suffix that resolves every delimiter still open
    at the end of `source` — brackets, an unterminated string/template, or an
    unterminated comment — string/template/comment aware (a brace inside a string
    or comment is not a real open delimiter).

    Closers are appended in LIFO order (innermost first): an open line comment
    closes with `\\n` (so anything after it isn't swallowed), a block comment with
    `*/`, a string/template with its own quote char, and a bracket with its match.
    A no-op on already-balanced source.
    """
    stack: List[str] = []
    for _, _, stack in _scan(source):
        pass
    parts = []
    for item in reversed(stack):
        if item == "//":
            parts.append("\n")
        elif item == "/*":
            parts.append("*/")
        else:
            parts.append(item)
    return source + "".join(parts)


def statement_boundary(text: str) -> Optional[int]:
    """Index just past the first statement-terminating token (`;` or `\\n`) found
    at bracket depth 0 outside any string/template/comment, or `None` if `text`
    contains no such boundary yet.
    """
    for i, c, stack in _scan(text):
        if c not in (";", "\n"):
            continue
        in_special = bool(stack) and stack[-1] in ("'", '"', "`", "/*", "//")
        bracket_depth = sum(1 for s in stack if s in _CLOSERS)
        if not in_special and bracket_depth == 0:
            return i + 1
    return None
