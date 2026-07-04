"""BFCL-style function-calling scorer (#102) — the tool-use eval harness.

Berkeley Function-Calling Leaderboard (BFCL) scores whether a model calls the right
tool(s) with the right arguments, or correctly abstains when no tool applies. This
module is the pure scoring core (parse / match / score / aggregate), mirroring
`olmes_adapter.py`'s split: heavy tokenizer/generation happens below the seam
(injected as a `generate_fn` callable), scoring logic is pure stdlib and testable
anywhere.

Categories (BFCL naming) drive the scoring rule in `score_example`:
  * "simple"              — exactly one gold call; the prediction must be exactly
    that one call.
  * "parallel"/"multiple" — a set of gold calls; the prediction must match as a
    multiset (order-insensitive, but count-sensitive — duplicates matter).
  * "abstention"/"relevance" — no tool applies; correct iff the prediction makes
    NO call (a spurious call is wrong; so is a missing call when one IS expected —
    scored symmetrically by simply comparing `pred_calls == []`).

Argument matching (`match_call`) is deliberately lenient: BFCL's "possible answers"
sometimes list several acceptable values for one argument (represented here as a
Python list on the gold side); an exact-dict match would be too strict and report
near-zero accuracy even for a correct model. Optional prediction args absent from
gold are ignored (BFCL scores required/expected args, not every key a model chooses
to add).

`parse_tool_calls` reuses the exact `<tool_call>{json}</tool_call>` tag convention
from `src.data.tool_sources` (the SFT corpus renders calls the same way — see
`tool_sources.format_tool_call`), so the harness scores generations in the same
format the student was trained to emit.

ABOVE THE SEAM — stdlib only (+ `src.data.tool_sources`, itself portable). No
`mlx`/`torch` import anywhere in this module (guarded by
`tests/test_import_guard.py`). `evaluate_bfcl` takes `generate_fn` as an injected
seam callable; the MLX-backed generation lives in `scripts/eval_bfcl.py`.
"""

from __future__ import annotations

import json
from typing import Callable, Dict, List, Sequence

from ..data.tool_sources import (TOOL_CALL_CLOSE, TOOL_CALL_OPEN, render_tool_system,
                                 validate_call_against_tools)

_ABSTENTION_CATEGORIES = {"abstention", "relevance"}
_PARALLEL_CATEGORIES = {"parallel", "multiple"}


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def parse_tool_calls(text: str) -> List[dict]:
    """Extract every `<tool_call>{json}</tool_call>` block from a decoded string.

    Tolerates malformed JSON (or a block missing "name") by skipping it — a model
    that emits broken JSON should score as "no valid call there", not crash the
    harness. Blocks are returned in the order they appear."""
    calls: List[dict] = []
    pos = 0
    while True:
        s = text.find(TOOL_CALL_OPEN, pos)
        if s == -1:
            break
        e = text.find(TOOL_CALL_CLOSE, s)
        if e == -1:
            break
        block = text[s + len(TOOL_CALL_OPEN):e].strip()
        pos = e + len(TOOL_CALL_CLOSE)
        try:
            call = json.loads(block)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(call, dict) and "name" in call:
            calls.append(call)
    return calls


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #

def match_call(pred: dict, gold: dict) -> bool:
    """True iff `pred` matches `gold`: names equal, and every gold argument is
    satisfied. A gold value that is a `list` means "any of these is an acceptable
    answer" (BFCL's multi-possible-answer convention); any other gold value must
    match exactly. Predicted arguments not present in `gold` are ignored (BFCL
    scores the expected args, not every extra key a model happens to add)."""
    if pred.get("name") != gold.get("name"):
        return False
    pred_args = pred.get("arguments") or {}
    gold_args = gold.get("arguments") or {}
    for key, gold_val in gold_args.items():
        if key not in pred_args:
            return False
        pred_val = pred_args[key]
        if isinstance(gold_val, list):
            if pred_val not in gold_val:
                return False
        elif pred_val != gold_val:
            return False
    return True


def _match_multiset(pred_calls: Sequence[dict], gold_calls: Sequence[dict]) -> bool:
    """Order-insensitive, count-sensitive match between two call lists: every gold
    call must be matched (via `match_call`) by exactly one distinct predicted call,
    with none left over on either side (so counts/duplicates matter)."""
    if len(pred_calls) != len(gold_calls):
        return False
    remaining = list(gold_calls)
    for p in pred_calls:
        idx = next((i for i, g in enumerate(remaining) if match_call(p, g)), None)
        if idx is None:
            return False
        remaining.pop(idx)
    return True


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def score_example(pred_calls: List[dict], gold_example: dict) -> dict:
    """Score one example's predicted calls against its gold, dispatching on
    `gold_example["category"]` (BFCL naming — see module docstring for the rule
    per category; unknown categories fall back to "simple"'s single-call rule).

    Also reports `schema_valid`: True iff every predicted call validates against
    the example's declared `tools` (`validate_call_against_tools`) — schema
    validity and correctness are independent axes (a call can be schema-valid but
    the wrong call, or vice versa in principle). Vacuously True when no calls were
    predicted.

    Returns `{"correct": bool, "category": str, "schema_valid": bool}`.
    """
    category = gold_example.get("category", "simple")
    gold_calls = gold_example.get("gold", [])
    tools = gold_example.get("tools", [])

    if category in _ABSTENTION_CATEGORIES:
        correct = pred_calls == []
    elif category in _PARALLEL_CATEGORIES:
        correct = _match_multiset(pred_calls, gold_calls)
    else:  # "simple" (default)
        correct = (len(pred_calls) == 1 and len(gold_calls) == 1
                   and match_call(pred_calls[0], gold_calls[0]))

    schema_valid = all(validate_call_against_tools(c, tools) for c in pred_calls)
    return {"correct": correct, "category": category, "schema_valid": schema_valid}


# --------------------------------------------------------------------------- #
# Aggregation + the seam
# --------------------------------------------------------------------------- #

def evaluate_bfcl(examples: Sequence[dict], generate_fn: Callable) -> dict:
    """Run BFCL-style scoring over `examples`, each a dict with at least
    `messages` (or `prompt`), `gold` (list of gold calls), `category`, `tools`.

    `generate_fn(prompt_or_messages) -> decoded_text` is the seam callable:
    generation happens below the seam, exactly like `olmes_adapter`'s injected
    `generate_until_texts` — this function only parses and aggregates. Each
    example's `messages` (if present) is passed to `generate_fn` in preference to
    `prompt`, so a caller free to choose either representation.

    Returns a summary dict: overall `accuracy`, `per_category_accuracy`, the
    `schema_valid_rate` (fraction of examples whose predicted calls are all
    schema-valid), and the per-example `results` for debugging.
    """
    results = []
    for ex in examples:
        prompt_or_messages = ex.get("messages", ex.get("prompt"))
        text = generate_fn(prompt_or_messages)
        pred_calls = parse_tool_calls(text)
        results.append(score_example(pred_calls, ex))

    n = len(results)
    by_category: Dict[str, List[dict]] = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r)
    per_category_accuracy = {
        cat: sum(1 for r in rows if r["correct"]) / len(rows)
        for cat, rows in by_category.items()
    }

    return {
        "n_examples": n,
        "accuracy": (sum(1 for r in results if r["correct"]) / n) if n else float("nan"),
        "per_category_accuracy": per_category_accuracy,
        "schema_valid_rate": (sum(1 for r in results if r["schema_valid"]) / n) if n else float("nan"),
        "results": results,
    }


# --------------------------------------------------------------------------- #
# Offline fixture — hand-authored, CC0, covers simple/parallel/abstention
# --------------------------------------------------------------------------- #

_WEATHER_TOOL = {"name": "get_weather", "description": "Get current weather for a city",
                 "parameters": {"type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"]}}
_TIMER_TOOL = {"name": "set_timer", "description": "Set a countdown timer",
               "parameters": {"type": "object",
                              "properties": {"seconds": {"type": "integer"}},
                              "required": ["seconds"]}}

BFCL_FIXTURE: List[dict] = [
    {
        "category": "simple",
        "tools": [_WEATHER_TOOL],
        "messages": [
            {"role": "system", "content": render_tool_system([_WEATHER_TOOL])},
            {"role": "user", "content": "What's the weather in Boston?"},
        ],
        # "Boston" or "Boston, MA" both count as the right city (BFCL possible-answers style).
        "gold": [{"name": "get_weather", "arguments": {"city": ["Boston", "Boston, MA"]}}],
    },
    {
        "category": "parallel",
        "tools": [_WEATHER_TOOL, _TIMER_TOOL],
        "messages": [
            {"role": "user", "content": "What's the weather in Tokyo, and set a 5 minute timer?"},
        ],
        "gold": [
            {"name": "get_weather", "arguments": {"city": "Tokyo"}},
            {"name": "set_timer", "arguments": {"seconds": 300}},
        ],
    },
    {
        "category": "abstention",
        "tools": [_WEATHER_TOOL],
        "messages": [
            {"role": "user", "content": "Translate 'hello' into French."},
        ],
        "gold": [],
    },
]
