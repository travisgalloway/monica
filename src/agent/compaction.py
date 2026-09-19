"""Staged context compaction (T4) with soft elision and hard summarization (#350).

Empirical findings from arXiv:2609.20804 demonstrate that staged two-tier compaction
delivers the lowest token costs across benchmarks while matching full-context accuracy.
It combines:
  1. Soft Threshold (~0.60 usable context): Rule-based elision (M1) replacing bulky
     middle-region observation outputs (compiler logs, test runs, file reads) with
     `[Output elided: N chars]`. Zero compute/API cost.
  2. Hard Threshold (~0.85 usable context): LLM-based summarization (M3) compressing
     the oldest middle-region events into a concise narrative.

Protected Window Invariant:
  - The system preamble (instructions, task prompt) and the most recent >= 2 turns
    ({recent} ~0.30 usable context) are NEVER elided or summarized. They are strictly
    preserved verbatim.

No Lossless Recall Machinery:
  - As proven in arXiv:2609.20804, making elided content recoverable via tool calls
    (`recall_event`) adds complex machinery that models virtually never invoke
    (-0.36% net delta; median calls = 0) while introducing meta-cognitive distraction
    and prompt overhead. Lossless recall machinery is explicitly omitted.

ABOVE THE SEAM — pure Python standard library only. No hardware backends (mlx/torch)
are imported anywhere in this module (enforced by tests/test_import_guard.py).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..data.tool_sources import TOOL_RESPONSE_CLOSE, TOOL_RESPONSE_OPEN
from ..eval.bfcl_adapter import parse_tool_calls

DEFAULT_MAX_CONTEXT_TOKENS = 8192
DEFAULT_SOFT_THRESHOLD = 0.60
DEFAULT_HARD_THRESHOLD = 0.85
DEFAULT_PROTECT_RECENT_TURNS = 2
DEFAULT_ELISION_CHAR_THRESHOLD = 200
DEFAULT_SUMMARY_ROLE = "user"

_TOOL_RESPONSE_REGEX = re.compile(
    rf"({re.escape(TOOL_RESPONSE_OPEN)})(.*?)({re.escape(TOOL_RESPONSE_CLOSE)})",
    re.DOTALL,
)


# --------------------------------------------------------------------------- #
# Telemetry and Configuration Dataclasses
# --------------------------------------------------------------------------- #

@dataclass
class CompactionConfig:
    """Configuration for staged two-tier context compaction."""

    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
    soft_threshold: float = DEFAULT_SOFT_THRESHOLD
    hard_threshold: float = DEFAULT_HARD_THRESHOLD
    protect_recent_turns: int = DEFAULT_PROTECT_RECENT_TURNS
    elision_char_threshold: int = DEFAULT_ELISION_CHAR_THRESHOLD
    summary_role: str = DEFAULT_SUMMARY_ROLE


@dataclass
class CompactionReport:
    """Detailed report generated after evaluating and executing compaction."""

    compacted: bool = False
    elision_applied: bool = False
    summarization_applied: bool = False
    initial_tokens: int = 0
    final_tokens: int = 0
    elided_observations: int = 0
    elided_chars_total: int = 0
    summarized_turns: int = 0
    summary_text: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "compacted": self.compacted,
            "elision_applied": self.elision_applied,
            "summarization_applied": self.summarization_applied,
            "initial_tokens": self.initial_tokens,
            "final_tokens": self.final_tokens,
            "elided_observations": self.elided_observations,
            "elided_chars_total": self.elided_chars_total,
            "summarized_turns": self.summarized_turns,
            "summary_text": self.summary_text,
            "events": list(self.events),
        }


# CompactionResult alias for backwards compatibility
CompactionResult = CompactionReport


# --------------------------------------------------------------------------- #
# Token Estimation and Partitioning Helpers
# --------------------------------------------------------------------------- #

def estimate_tokens(
    content: str | list[dict[str, Any]],
    token_counter: Callable[[str | list[dict[str, Any]]], int] | None = None,
) -> int:
    """Estimate token count for a string or message history.

    If token_counter is provided, delegates directly to it. Otherwise, computes
    a standard character/word heuristic (max(words, (chars + 3) // 4) + per-message overhead).
    """
    if token_counter is not None:
        return int(token_counter(content))

    if isinstance(content, str):
        if not content:
            return 0
        words = len(content.split())
        chars = len(content)
        return max(words, (chars + 3) // 4)

    # Conversation history: list[dict[str, Any]]
    total = 0
    for msg in content:
        # Per-message formatting overhead (role delimiters and metadata)
        total += 4
        text = str(msg.get("content", ""))
        words = len(text.split())
        chars = len(text)
        total += max(words, (chars + 3) // 4)
    return total


def partition_conversation(
    messages: list[dict[str, Any]],
    protect_recent_turns: int = DEFAULT_PROTECT_RECENT_TURNS,
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
    """Partition messages into (preamble, middle_turns, recent_turns).

    Protected Window Invariant:
      - Preamble: Initial system message(s) and the first user message (task prompt).
        Strictly preserved verbatim.
      - Recent turns: The most recent `protect_recent_turns` turns (minimum 2).
        Strictly preserved verbatim.
      - Middle turns: Turns strictly between preamble and recent turns. Subject to
        staged compaction (soft elision and hard summarization).
    """
    recent_k = max(2, int(protect_recent_turns))

    if not messages:
        return [], [], []

    # 1. Identify preamble boundary:
    # All leading system messages and the first user message (the initial task prompt).
    first_user_idx: int | None = None
    for idx, msg in enumerate(messages):
        if msg.get("role") == "user":
            first_user_idx = idx
            break

    if first_user_idx is None:
        # No user task prompt found; entire conversation treated as preamble
        return [dict(m) for m in messages], [], []

    preamble = [dict(m) for m in messages[: first_user_idx + 1]]
    remainder = messages[first_user_idx + 1 :]

    if not remainder:
        return preamble, [], []

    # 2. Group remainder into turns.
    # A turn begins with an assistant message (or a leading injected summary message).
    turns: list[list[dict[str, Any]]] = []
    current_turn: list[dict[str, Any]] = []

    for msg in remainder:
        role = msg.get("role")
        if role == "assistant":
            if current_turn:
                turns.append(current_turn)
            current_turn = [dict(msg)]
        else:
            if not current_turn:
                # Leading non-assistant message in remainder (e.g. prior summary message)
                turns.append([dict(msg)])
            else:
                current_turn.append(dict(msg))

    if current_turn:
        turns.append(current_turn)

    # 3. Split into middle_turns and recent_turns based on recent_k
    if len(turns) <= recent_k:
        return preamble, [], turns

    middle_turns = turns[: len(turns) - recent_k]
    recent_turns = turns[len(turns) - recent_k :]
    return preamble, middle_turns, recent_turns


def elide_observation_content(
    content: str,
    elision_char_threshold: int = DEFAULT_ELISION_CHAR_THRESHOLD,
) -> tuple[str, int, int]:
    """Elide bulky tool observation outputs within content.

    Replaces bulky outputs with `[Output elided: N chars]` where N is the
    character count of the elided observation body. Zero compute/API cost.

    Returns:
        (elided_content, elided_blocks_count, elided_chars_count)
    """
    if "[Output elided:" in content:
        # Already elided; avoid nested or repetitive elisions
        return content, 0, 0

    elided_blocks = 0
    elided_chars = 0

    if TOOL_RESPONSE_OPEN in content:
        def _replace_block(match: re.Match[str]) -> str:
            nonlocal elided_blocks, elided_chars
            prefix, body, suffix = match.group(1), match.group(2), match.group(3)
            stripped_body = body.strip()
            if len(stripped_body) > elision_char_threshold:
                n_chars = len(stripped_body)
                elided_blocks += 1
                elided_chars += n_chars
                return f"{prefix}\n[Output elided: {n_chars} chars]\n{suffix}"
            return match.group(0)

        new_content = _TOOL_RESPONSE_REGEX.sub(_replace_block, content)
        return new_content, elided_blocks, elided_chars

    # Content without <tool_response> tags
    stripped = content.strip()
    if len(stripped) > elision_char_threshold:
        n_chars = len(stripped)
        elided_blocks += 1
        elided_chars += n_chars
        return f"[Output elided: {n_chars} chars]", elided_blocks, elided_chars

    return content, 0, 0


# --------------------------------------------------------------------------- #
# Turn Formatting and Summarization Helpers
# --------------------------------------------------------------------------- #

def _format_turns_for_summary(turns: list[list[dict[str, Any]]]) -> str:
    """Format turns into structured text for LLM-based summarization."""
    lines: list[str] = []
    for t_idx, turn in enumerate(turns, 1):
        lines.append(f"Turn {t_idx}:")
        for msg in turn:
            role = msg.get("role", "")
            content = str(msg.get("content", "")).strip()
            if role == "assistant":
                # Extract calls if present
                calls = parse_tool_calls(content)
                if calls:
                    call_strs = [
                        f"{c.get('name', 'tool')}({json.dumps(c.get('arguments', {}))[:80]})"
                        for c in calls
                    ]
                    lines.append(f"  Actions: {', '.join(call_strs)}")
                else:
                    lines.append(f"  Response: {content[:150]}")
            elif role in ("user", "tool"):
                if "[Output elided:" in content:
                    lines.append(f"  Observation: {content[:120]}")
                elif TOOL_RESPONSE_OPEN in content:
                    matches = _TOOL_RESPONSE_REGEX.findall(content)
                    obs_snippets = [m[1].strip()[:80] for m in matches]
                    lines.append(f"  Observation: {'; '.join(obs_snippets)}")
                else:
                    lines.append(f"  Observation: {content[:100]}")
    return "\n".join(lines)


def _deterministic_narrative_summary(turns: list[list[dict[str, Any]]]) -> str:
    """Deterministic fallback narrative summary when no LLM summarizer is supplied."""
    actions: list[str] = []
    observations: list[str] = []

    for turn in turns:
        for msg in turn:
            role = msg.get("role")
            content = str(msg.get("content", ""))
            if role == "assistant":
                calls = parse_tool_calls(content)
                for call in calls:
                    name = call.get("name")
                    args = call.get("arguments", {})
                    if name == "view_file" and "path" in args:
                        actions.append(f"inspected {args['path']}")
                    elif name == "edit_file" and "path" in args:
                        actions.append(f"edited {args['path']}")
                    elif name == "execute_bash" and "command" in args:
                        cmd = str(args["command"]).strip()
                        actions.append(f"ran `{cmd[:30]}`")
                    elif name:
                        actions.append(f"called {name}")
            elif role in ("user", "tool"):
                if "[Output elided:" in content:
                    m = re.search(r"\[Output elided: (\d+) chars\]", content)
                    if m:
                        observations.append(f"observed {m.group(1)} chars elided output")
                elif "error" in content.lower():
                    observations.append("encountered error")

    act_summary = ", ".join(actions) if actions else "executed multi-step operations"
    obs_summary = "; ".join(observations) if observations else "completed successfully"
    return (
        f"Prior turns compressed into narrative. Executed: {act_summary}. "
        f"Observations: {obs_summary}."
    )


# --------------------------------------------------------------------------- #
# Staged Context Compactor (T4)
# --------------------------------------------------------------------------- #

class ContextCompactor:
    """Staged two-tier context compactor (T4) combining soft elision and hard summarization.

    Implements:
      - Soft Threshold (soft_threshold ~ 0.60): Rule-based elision (M1) replacing bulky
        middle-region observation bodies with `[Output elided: N chars]`. Zero compute/API cost.
      - Hard Threshold (hard_threshold ~ 0.85): LLM-based summarization (M3) compressing
        the oldest middle-region events into a concise narrative.
      - Protected Window Invariant: System preamble (instructions, task prompt) and
        most recent turns (>= 2 turns, ~0.30 usable context) are strictly preserved verbatim.
      - No Lossless Recall Machinery: Explicitly omits recall_event external vector/disk caches,
        preventing meta-cognitive distraction and prompt overhead (arXiv:2609.20804).
    """

    def __init__(
        self,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        soft_threshold: float = DEFAULT_SOFT_THRESHOLD,
        hard_threshold: float = DEFAULT_HARD_THRESHOLD,
        protect_recent_turns: int = DEFAULT_PROTECT_RECENT_TURNS,
        elision_char_threshold: int = DEFAULT_ELISION_CHAR_THRESHOLD,
        summarizer: Callable[[str | list[dict[str, Any]]], str] | Any | None = None,
        token_counter: Callable[[str | list[dict[str, Any]]], int] | None = None,
        summary_role: str = DEFAULT_SUMMARY_ROLE,
    ) -> None:
        self.max_context_tokens = int(max_context_tokens)
        self.soft_threshold = float(soft_threshold)
        self.hard_threshold = float(hard_threshold)
        # Enforce protected window invariant: at least 2 most recent turns protected verbatim
        self.protect_recent_turns = max(2, int(protect_recent_turns))
        self.elision_char_threshold = int(elision_char_threshold)
        self.summarizer = summarizer
        self.token_counter = token_counter
        self.summary_role = summary_role

    def _estimate_tokens(self, content: str | list[dict[str, Any]]) -> int:
        return estimate_tokens(content, self.token_counter)

    def _call_summarizer(self, prompt: str, turns: list[list[dict[str, Any]]]) -> str:
        """Invoke configured summarizer (callable or generative object), or fallback."""
        if self.summarizer is not None:
            if callable(self.summarizer):
                try:
                    res = self.summarizer(prompt)
                except TypeError:
                    res = self.summarizer([{"role": "user", "content": prompt}])
            elif hasattr(self.summarizer, "generate") and callable(self.summarizer.generate):
                try:
                    res = self.summarizer.generate(prompt)
                except TypeError:
                    res = self.summarizer.generate([{"role": "user", "content": prompt}])
            else:
                res = None

            if isinstance(res, dict):
                text = str(res.get("content", "")).strip()
            elif res is not None:
                text = str(res).strip()
            else:
                text = ""

            if text:
                return text

        return _deterministic_narrative_summary(turns)

    def compact_with_report(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], CompactionReport]:
        """Execute staged compaction on conversation history, returning compacted messages and report.

        Protected Window Invariant:
          The preamble and recent turns are guaranteed strictly preserved verbatim.
        """
        initial_tokens = self._estimate_tokens(messages)
        soft_token_budget = int(self.soft_threshold * self.max_context_tokens)
        hard_token_budget = int(self.hard_threshold * self.max_context_tokens)

        # If below soft threshold, no compaction required
        if initial_tokens <= soft_token_budget:
            report = CompactionReport(
                compacted=False,
                elision_applied=False,
                summarization_applied=False,
                initial_tokens=initial_tokens,
                final_tokens=initial_tokens,
            )
            return [dict(m) for m in messages], report

        # Partition conversation into preamble, middle turns, and recent turns
        preamble, middle_turns, recent_turns = partition_conversation(
            messages, protect_recent_turns=self.protect_recent_turns
        )

        # If no middle turns exist, cannot compact without violating protected window invariant
        if not middle_turns:
            report = CompactionReport(
                compacted=False,
                elision_applied=False,
                summarization_applied=False,
                initial_tokens=initial_tokens,
                final_tokens=initial_tokens,
            )
            return [dict(m) for m in messages], report

        # ------------------------------------------------------------------- #
        # Tier 1: Soft Threshold Rule-Based Elision (M1)
        # ------------------------------------------------------------------- #
        elision_applied = False
        total_elided_obs = 0
        total_elided_chars = 0

        elided_middle_turns: list[list[dict[str, Any]]] = []
        for turn in middle_turns:
            new_turn: list[dict[str, Any]] = []
            for msg in turn:
                role = msg.get("role")
                content = str(msg.get("content", ""))
                if role in ("user", "tool"):
                    new_content, n_obs, n_chars = elide_observation_content(
                        content, elision_char_threshold=self.elision_char_threshold
                    )
                    if n_obs > 0:
                        elision_applied = True
                        total_elided_obs += n_obs
                        total_elided_chars += n_chars
                        new_msg = dict(msg)
                        new_msg["content"] = new_content
                        new_turn.append(new_msg)
                        continue
                new_turn.append(dict(msg))
            elided_middle_turns.append(new_turn)

        # Assemble candidate messages after soft elision
        candidate_messages: list[dict[str, Any]] = []
        candidate_messages.extend(preamble)
        for t in elided_middle_turns:
            candidate_messages.extend(t)
        for t in recent_turns:
            candidate_messages.extend(t)

        tokens_after_elision = self._estimate_tokens(candidate_messages)

        # If soft elision brought context below or equal to hard threshold, stop here
        if tokens_after_elision <= hard_token_budget:
            report = CompactionReport(
                compacted=elision_applied,
                elision_applied=elision_applied,
                summarization_applied=False,
                initial_tokens=initial_tokens,
                final_tokens=tokens_after_elision,
                elided_observations=total_elided_obs,
                elided_chars_total=total_elided_chars,
                summarized_turns=0,
            )
            return candidate_messages, report

        # ------------------------------------------------------------------- #
        # Tier 2: Hard Threshold LLM-Based Summarization (M3)
        # ------------------------------------------------------------------- #
        # Context still exceeds hard threshold. Compress oldest middle turns into a concise narrative.
        summarization_applied = True
        best_messages: list[dict[str, Any]] = candidate_messages
        best_tokens = tokens_after_elision
        best_summary_text: str | None = None
        summarized_count = 0

        # Progressively summarize oldest middle turns until below hard threshold
        for k in range(1, len(elided_middle_turns) + 1):
            turns_to_summarize = elided_middle_turns[:k]
            remaining_middle_turns = elided_middle_turns[k:]

            steps_text = _format_turns_for_summary(turns_to_summarize)
            summary_prompt = (
                "You are Monica's autonomous context summarizer. Provide a concise narrative summary "
                "of the following prior execution turns. Detail what actions were executed, files inspected "
                "or modified, commands run, and key findings or outcomes. Omit raw logs.\n\n"
                f"Turns to summarize:\n{steps_text}\n\n"
                "Concise Narrative Summary:"
            )

            summary_text = self._call_summarizer(summary_prompt, turns_to_summarize)
            summary_msg = {
                "role": self.summary_role,
                "content": f"[Context Summary: {summary_text}]",
            }

            trial_messages: list[dict[str, Any]] = []
            trial_messages.extend(preamble)
            trial_messages.append(summary_msg)
            for t in remaining_middle_turns:
                trial_messages.extend(t)
            for t in recent_turns:
                trial_messages.extend(t)

            trial_tokens = self._estimate_tokens(trial_messages)
            best_messages = trial_messages
            best_tokens = trial_tokens
            best_summary_text = summary_text
            summarized_count = k

            if trial_tokens <= hard_token_budget:
                break

        report = CompactionReport(
            compacted=True,
            elision_applied=elision_applied,
            summarization_applied=summarization_applied,
            initial_tokens=initial_tokens,
            final_tokens=best_tokens,
            elided_observations=total_elided_obs,
            elided_chars_total=total_elided_chars,
            summarized_turns=summarized_count,
            summary_text=best_summary_text,
        )
        return best_messages, report

    def compact(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Compact conversation history and return compacted messages list."""
        compacted_messages, _ = self.compact_with_report(messages)
        return compacted_messages
