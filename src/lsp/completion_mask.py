"""#226 completion-list logit masking — the SSI glue between a completion-list source
(the real LSP oracle, a reference-derived ceiling, or a null "pay the cost, apply no
mask" control) and the pure `src/serve/constrained.py` trie mechanism.

Three pieces:

  - `LabelSource` (a `Protocol`) + its three implementations `LspLabels` / `OracleLabels`
    / `NullLabels` — where the valid-identifier list for the current cursor comes from.
  - The identifier-span state machine (`CompletionMasker._advance`) — a cheap left-to-
    right scanner over the generated text (string/template/comment-aware via
    `diagnostics.mask_strings_and_comments`, so a `.` inside a string never opens a
    span) that tracks whether generation is currently inside a maskable span and issues
    **exactly one** completion-list query per span, not per token — the load-bearing
    performance decision (see `docs/design/12-lsp-in-the-loop.md`'s #220 bench: masking
    per-token would put an `update()`/`didChange` on every decode step).
  - `CompletionMasker` — owns the scanner state, a `LabelSource`, and a lazily-built
    `VocabTrie`; `mask_for(generated_text)` returns the allowed id set for the NEXT
    token, or `None` when no mask applies this step (outside any span, a null arm, or a
    step whose allowed set came out empty — see `n_mask_bypass`).

ABOVE THE SEAM — stdlib only (+ `src.serve.constrained`, itself portable). No
`mlx`/`torch` import anywhere in this module (guarded by `tests/test_import_guard.py`).
"""

from __future__ import annotations

import re
import time
from typing import Callable, Iterable, List, Optional, Protocol, Sequence, runtime_checkable

from ..serve.constrained import (VocabTable, VocabTrie, allowed_extensions,
                                  build_vocab_table)
from .diagnostics import mask_strings_and_comments

_IDENT_CHAR_RE = re.compile(r"[A-Za-z0-9_$]")
_IDENT_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")


# --------------------------------------------------------------------------- #
# label sources
# --------------------------------------------------------------------------- #

@runtime_checkable
class LabelSource(Protocol):
    """`query` is called exactly once per span, with the text UP TO AND INCLUDING
    the anchor (e.g. the member-access `.`) and the offset generation should
    resume from. Returns the label list valid at that cursor, or `None` to mean
    "no mask for this span" (the M4 null contract — see `NullLabels`)."""

    def query(self, path: str, text: str, anchor_offset: int) -> Optional[List[str]]:
        ...


class LspLabels:
    """Labels from a live `TsLspService`: `service.update(path, text)` then
    `service.completions(path, anchor_offset)`. Records
    `service.last_completion_incomplete` into `n_incomplete_lists` — a truncated
    list silently treated as complete would be a BLIND failure."""

    def __init__(self, service, path: str) -> None:
        self.service = service
        self.path = path
        self.n_incomplete_lists = 0

    def query(self, path: str, text: str, anchor_offset: int) -> Optional[List[str]]:
        self.service.update(path, text)
        items = self.service.completions(path, anchor_offset)
        if getattr(self.service, "last_completion_incomplete", False):
            self.n_incomplete_lists += 1
        return [it.label for it in items]


class OracleLabels:
    """The ceiling arm: labels are identifiers extracted from a record's OWN
    reference/gold text, not the language server. A blank reference is a hard
    error — the driver is expected to catch it, count `n_oracle_unavailable`, and
    exclude that record from every arm in the paired comparison."""

    def __init__(self, reference_text: str) -> None:
        if not reference_text or not reference_text.strip():
            raise ValueError(
                "OracleLabels requires a non-blank reference text — the driver "
                "must catch this, count it, and exclude the record from every "
                "arm in the paired comparison")
        self.reference_text = reference_text
        self._labels = sorted(set(_IDENT_RE.findall(reference_text)))

    def query(self, path: str, text: str, anchor_offset: int) -> Optional[List[str]]:
        return list(self._labels)


class NullLabels:
    """The M4 null: delegates to `inner` so its query, latency, and counters ALL
    still happen (the span is real, the cost is real), then returns `None` — no
    mask is ever applied. This is what makes "the mask helped" separable from
    "the arm ran a different code path"."""

    def __init__(self, inner: LabelSource) -> None:
        self.inner = inner

    def query(self, path: str, text: str, anchor_offset: int) -> Optional[List[str]]:
        self.inner.query(path, text, anchor_offset)
        return None


# --------------------------------------------------------------------------- #
# the masker
# --------------------------------------------------------------------------- #

class CompletionMasker:
    """Owns the identifier-span state machine and a lazily-built `VocabTrie`.

    `mask_scope`:
      - `"member"` (default) — spans open on `.` / `?.`, exactly the shape of the
        #194 injection set and the home of the TS2339/TS2551 "names-wrong" failure
        this arm targets.
      - `"identifier"` — additionally opens on any bare identifier at a word
        boundary. Available, not default: it risks masking keywords/locals and
        widens the variable under test.

    Never arms inside a string, template literal, or comment (scanned via
    `diagnostics.mask_strings_and_comments`, the same string/comment-aware walk
    `close_open_delimiters`/`statement_boundary` use). Regex literals are NOT
    separately detected — a residual imprecision, documented rather than hidden
    (mirrors `mask_strings_and_comments`'s own documented residual gaps): a `.`
    inside a regex literal can open a spurious span. In practice this is rare at
    the member-access positions this arm targets and is visible in the results
    JSON via an elevated `n_mask_bypass` if it happens (a regex-interior span
    has no real completions and every step bypasses).
    """

    def __init__(self, label_source: LabelSource, path: str, decode: Callable[[Sequence[int]], str],
                 *, mask_scope: str = "member") -> None:
        if mask_scope not in ("member", "identifier"):
            raise ValueError(f"unknown mask_scope {mask_scope!r}")
        self.label_source = label_source
        self.path = path
        self.decode = decode
        self.mask_scope = mask_scope

        self._vocab: Optional[VocabTable] = None
        self._trie: Optional[VocabTrie] = None
        self._exit_ids: List[int] = []

        self._processed_len = 0
        self._span_open = False
        self._anchor_offset: Optional[int] = None
        self._labels: Optional[List[str]] = None

        self.n_mask_steps = 0
        self.n_mask_bypass = 0
        self.n_completion_calls = 0
        self.mask_wall_s = 0.0

    # ----------------------------------------------------------------- #
    # vocab (lazy — built once, on the first call that supplies vocab_size)
    # ----------------------------------------------------------------- #

    def _ensure_vocab(self, vocab_size: int) -> None:
        if self._trie is not None:
            return
        self._vocab = build_vocab_table(self.decode, vocab_size, anchor_ids=())
        self._trie = VocabTrie(self._vocab)
        self._exit_ids = [i for i, piece in enumerate(self._vocab)
                          if piece and not _IDENT_CHAR_RE.match(piece[0])]

    # ----------------------------------------------------------------- #
    # span state machine
    # ----------------------------------------------------------------- #

    def _open_span(self, text: str, anchor_offset: int) -> None:
        self._span_open = True
        self._anchor_offset = anchor_offset
        t0 = time.monotonic()
        labels = self.label_source.query(self.path, text[:anchor_offset], anchor_offset)
        self.mask_wall_s += time.monotonic() - t0
        self.n_completion_calls += 1
        self._labels = labels

    def _close_span(self) -> None:
        self._span_open = False
        self._anchor_offset = None
        self._labels = None

    def _advance(self, text: str) -> None:
        if len(text) < self._processed_len:
            # Text shrank — shouldn't happen for append-only generation, but never
            # trust stale state over a scan that can't have kept up.
            self._processed_len = 0
            self._close_span()

        masked = mask_strings_and_comments(text)
        for i in range(self._processed_len, len(masked)):
            ch = masked[i]
            if self._span_open:
                if not _IDENT_CHAR_RE.match(ch):
                    self._close_span()
                continue
            if ch == "." and self.mask_scope in ("member", "identifier"):
                self._open_span(text, i + 1)
            elif self.mask_scope == "identifier" and _IDENT_CHAR_RE.match(ch):
                prev = masked[i - 1] if i > 0 else ""
                if not _IDENT_CHAR_RE.match(prev):
                    self._open_span(text, i)
        self._processed_len = len(masked)

    # ----------------------------------------------------------------- #
    # the per-step call
    # ----------------------------------------------------------------- #

    def mask_for(self, generated_text: str, *, vocab_size: Optional[int] = None) -> Optional[List[int]]:
        """Advance the scanner over `generated_text` (the full text generated so
        far) and return the allowed id set for the NEXT token, or `None` if no
        mask applies this step. `vocab_size`, when given, primes the (once-built,
        cached) `VocabTrie` — passed by the decode loop from its own already-live
        `logits.size`, so no new `LMAdapter` protocol method is needed."""
        self._advance(generated_text)
        if not self._span_open or self._labels is None:
            return None
        if vocab_size is not None:
            self._ensure_vocab(vocab_size)
        if self._trie is None:
            return None  # vocab not primed yet -- nothing we can constrain against
        prefix = generated_text[self._anchor_offset:]
        allowed = allowed_extensions(self._trie, self._labels, prefix, exit_ids=self._exit_ids)
        self.n_mask_steps += 1
        if not allowed:
            self.n_mask_bypass += 1
            return None
        return allowed
