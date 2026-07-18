"""True top-level statement boundaries via tree-sitter-typescript (#201).

The character-scanner `statement_boundary` in `diagnostics.py` returns the first
brace-depth-0 `;`/newline in a text — which is fine for the incremental slow loop
but *wrong* for cutting a real file into over-repair probe prefixes: it can't tell
that a depth-0 newline sits inside an unclosed multi-line construct, so a cut there
yields an un-compilable fragment (the E4 confound (b), see
`docs/design/12-lsp-in-the-loop.md`). A real parser can.

`top_level_boundaries(source)` returns the offset one-past-the-end of each *complete*
top-level statement, so a cut at any of them is always a valid top-level boundary,
never inside an unclosed `interface`/object/function.

ABOVE THE SEAM -- stdlib only *at import*. The `tree_sitter` import is deferred into
the parser factory (like `datasets` in the corpus scripts), so importing this module
never requires the optional dependency and portable hosts / the import guard stay
happy. Install it via the `[eval]` extra (`tree-sitter`, `tree-sitter-typescript`).
"""

from __future__ import annotations

from typing import List, Optional

_PARSER = None


def _parser():
    """Lazily build and cache one TypeScript parser. The tree-sitter import is
    deferred here so module import stays dependency-free."""
    global _PARSER
    if _PARSER is None:
        from tree_sitter import Language, Parser        # local: optional dependency
        import tree_sitter_typescript as tsts
        # Plain TypeScript grammar (NOT tsx) -- see the plan's grammar-quirk risk.
        _PARSER = Parser(Language(tsts.language_typescript()))
    return _PARSER


def tree_sitter_available() -> bool:
    """True if the optional tree-sitter toolchain imports (mirrors `resolve_*`)."""
    try:
        import tree_sitter        # noqa: F401
        import tree_sitter_typescript  # noqa: F401
        return True
    except ImportError:
        return False


def top_level_boundaries(source: str) -> List[int]:
    """Character offsets one-past-the-end of each complete top-level statement in
    `source`. A cut at any returned offset ends exactly after a whole top-level
    statement -- never inside an unclosed construct.

    Nodes whose subtree contains a parse error (an incomplete tail, the classic
    truncated `interface`) are skipped -- fail closed, never emit a bad boundary.
    Pure `comment` nodes are skipped too (a boundary right after a comment is not a
    statement boundary worth cutting on).
    """
    parser = _parser()
    data = source.encode("utf-8")
    tree = parser.parse(data)
    out: List[int] = []
    for node in tree.root_node.named_children:
        if node.has_error or node.is_missing or node.type == "comment":
            continue
        # tree-sitter offsets are in bytes; map to a char index so callers can slice
        # the original `str` directly (exact for non-ASCII, trivial for ASCII).
        out.append(len(data[: node.end_byte].decode("utf-8")))
    return out


def first_boundary_in_range(source: str, min_chars: int, max_chars: int) -> Optional[int]:
    """The first true top-level boundary whose offset is in `[min_chars, max_chars]`,
    or `None`. Convenience for the probe builder's one-prefix-per-file cut."""
    for b in top_level_boundaries(source):
        if b < min_chars:
            continue
        if b > max_chars:
            break
        return b
    return None
