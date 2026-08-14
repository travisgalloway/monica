"""Portable, LSP-agnostic prefix-constraint mechanism for #226 completion-list
logit masking (constrained decode).

Three pieces, in the order a caller uses them:

  - `build_vocab_table(decode, vocab_size, anchor_ids)` — probe every vocab id's text
    IN CONTEXT (byte-level BPE makes `decode([i])` alone unsound — see
    `src/lsp/lm.py`'s `offset_map` docstring for the exact trap) and cache the result.
  - `VocabTrie` — a char-trie over the probeable token strings, built once from a
    vocab table.
  - `allowed_extensions(trie, labels, prefix, exit_ids=...)` — given the labels valid
    at the current cursor and the identifier text generated so far, return every
    vocab id that keeps SOME label reachable.

Pure stdlib + numpy (numpy only for the `VocabTable` typing convenience — no array
math happens here). No `mlx`/`torch` import anywhere in this module (guarded by
`tests/test_import_guard.py`).
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence

#: Vocab id -> token string, or `None` for an id that could not be probed cleanly
#: (empty decode, or decodes to the Unicode replacement character `�` — a
#: byte-level-BPE id that only makes sense as part of a multi-byte sequence with
#: ITS OWN continuation, not this one).
VocabTable = List[Optional[str]]

_REPLACEMENT_CHAR = "�"


def build_vocab_table(
    decode: Callable[[Sequence[int]], str],
    vocab_size: int,
    *,
    anchor_ids: Sequence[int],
) -> VocabTable:
    """Probe every id in `[0, vocab_size)` IN CONTEXT: `decode(anchor_ids + [i])` minus
    the `decode(anchor_ids)` prefix. An id whose probed text is empty or contains the
    Unicode replacement character is recorded as `None` and excluded from every
    allowed set — exclusion is the conservative direction (masking is a restriction;
    a token we cannot reason about is one we must not permit).
    """
    anchor_text = decode(anchor_ids)
    anchor_len = len(anchor_text)
    table: VocabTable = [None] * vocab_size
    for i in range(vocab_size):
        probed = decode(list(anchor_ids) + [i])
        if not probed.startswith(anchor_text):
            # Non-prefix-consistent decode for this id in this context — cannot
            # reason about its suffix at all.
            continue
        piece = probed[anchor_len:]
        if not piece or _REPLACEMENT_CHAR in piece:
            continue
        table[i] = piece
    return table


class VocabTrie:
    """Char-trie over a `VocabTable`'s probeable (non-`None`) token strings.

    `prefix_matches(s)` walks `s` once (O(len(s))) and returns every token id whose
    string is a prefix of `s` — i.e. every token that could legally come next if the
    remaining text to produce is exactly `s`.
    """

    __slots__ = ("_root",)

    def __init__(self, vocab: VocabTable) -> None:
        # node: {"children": {char: node}, "ids": [token_id, ...]}
        # `ids` on a node holds every token id whose string equals the path to that
        # node — normally one, but two ids can share a string in a real BPE vocab.
        self._root: dict = {"children": {}, "ids": []}
        for token_id, piece in enumerate(vocab):
            if piece is None:
                continue
            self._insert(piece, token_id)

    def _insert(self, piece: str, token_id: int) -> None:
        node = self._root
        for ch in piece:
            node = node["children"].setdefault(ch, {"children": {}, "ids": []})
        node["ids"].append(token_id)

    def prefix_matches(self, s: str) -> List[int]:
        """Every token id whose string is a prefix of `s` (including the empty walk
        matching zero-length — no token has an empty string, so that contributes
        nothing, but a walk that never advances still returns the root's ids, which
        is always empty by construction)."""
        node = self._root
        out: List[int] = list(node["ids"])
        for ch in s:
            node = node["children"].get(ch)
            if node is None:
                break
            out.extend(node["ids"])
        return out


def allowed_extensions(
    trie: VocabTrie,
    labels: Iterable[str],
    prefix: str,
    *,
    exit_ids: Sequence[int] = (),
) -> List[int]:
    """Every vocab id that keeps some `label` in `labels` reachable, given the
    identifier text generated so far (`prefix`).

    For each label still consistent with `prefix` (i.e. `label.startswith(prefix)`),
    the remaining suffix is `label[len(prefix):]`; `trie.prefix_matches(suffix)` is
    unioned in. If `prefix` is ITSELF a complete label, `exit_ids` (tokens that begin
    with a character which cannot continue an identifier) are unioned in too — without
    this the model can never leave the span once it has spelled a valid name.
    """
    out: set = set()
    prefix_is_complete_label = False
    for label in labels:
        if not label.startswith(prefix):
            continue
        suffix = label[len(prefix):]
        if not suffix:
            prefix_is_complete_label = True
            continue
        out.update(trie.prefix_matches(suffix))
    if prefix_is_complete_label:
        out.update(exit_ids)
    return sorted(out)
