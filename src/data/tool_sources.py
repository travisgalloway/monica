"""Tool-use SFT sources + formatting helpers (#102) — the tool-calling skill's data.

Tool calls are **assistant** content (`<tool_call>{json}</tool_call>` blocks), tool results
come back as a **user** turn (`<tool_response>{json}</tool_response>`), available tools
(including distractors) live in the **system** turn (`<tools>[...]</tools>`). This is the
exact `<think>`/`<answer>` precedent from `reasoning_traces.py` — specialized assistant
content flows through `response_spans` unchanged. No new chat roles are introduced.

Primary sources:
- `handauthored` — CC0 checked-in set (always offline, 3 rows covering positive call,
  abstention, and multi-turn ReAct)
- `xlam` — Salesforce/xlam-function-calling-60k (CC-BY-4.0)
- `toolace` — Team-ACE/ToolACE (Apache-2.0)
- `when2call` — the restraint/abstention source (CC-BY-4.0)
- `glaive` — glaiveai/glaive-function-calling-v2 (flagged: partly GPT-distilled)

BFCL is eval-only (shared/eval/), intentionally absent from _LOADERS.

Calls are schema-validated against each row's declared tools via
`validate_call_against_tools` (name known + required args present), wired into
`tool_sft.build_tool_sft`; invalid rows are dropped and counted (`n_schema_invalid`).
Mappers themselves still just return None to skip malformed/unparseable rows.

ABOVE THE SEAM — stdlib only; `datasets` is imported lazily inside the HF loaders.
Offline path: `handauthored_tool_records()` + `--byte-fallback`.
"""

from __future__ import annotations

import json
import random
from typing import Iterable, Iterator, List, Optional, Sequence, Set

TOOL_CALL_OPEN, TOOL_CALL_CLOSE = "<tool_call>", "</tool_call>"
TOOL_RESPONSE_OPEN, TOOL_RESPONSE_CLOSE = "<tool_response>", "</tool_response>"
TOOLS_OPEN, TOOLS_CLOSE = "<tools>", "</tools>"

# Source -> license. Design-time best-effort; reconcile against dataset cards before a real run.
# Glaive is partly GPT-distilled (conflicts with the no-commercial-distillation stance) -> flagged.
SOURCE_LICENSES = {
    "glaive": "apache-2.0-flagged-distilled",
    "xlam": "cc-by-4.0",
    "toolace": "apache-2.0",
    "when2call": "cc-by-4.0",
    "handauthored": "cc0",
}


def _tag(messages: List[dict], source: str) -> dict:
    return {"messages": messages, "source": source,
            "license": SOURCE_LICENSES.get(source, "unknown")}


# --------------------------------------------------------------------------- #
# Formatting helpers (the tool analog of `format_trace`)
# --------------------------------------------------------------------------- #

def render_tool_system(tools: List[dict], *, preamble: Optional[str] = None) -> str:
    """System content listing callable tools: optional preamble + `<tools>[{schema}, ...]</tools>`.
    Tools are JSON-serialized verbatim (sort_keys=False, no whitespace collapse)."""
    body = f"{TOOLS_OPEN}\n{json.dumps(list(tools))}\n{TOOLS_CLOSE}"
    return f"{preamble.strip()}\n{body}" if preamble else body


def format_tool_call(calls: List[dict]) -> str:
    """Assistant content for one or more tool calls. Each call is a dict {name, arguments};
    rendered as stacked `<tool_call>\\n{json}\\n</tool_call>` blocks (parallel calls = multiple
    blocks in ONE assistant turn, so span boundaries stay put)."""
    blocks = []
    for c in calls:
        payload = json.dumps({"name": c["name"], "arguments": c.get("arguments", {})})
        blocks.append(f"{TOOL_CALL_OPEN}\n{payload}\n{TOOL_CALL_CLOSE}")
    return "\n".join(blocks)


def format_tool_response(results: Sequence) -> str:
    """User content for one or more tool results: stacked `<tool_response>\\n{json}\\n</tool_response>`.
    A result that is not already a str is JSON-serialized."""
    blocks = []
    for r in results:
        payload = r if isinstance(r, str) else json.dumps(r)
        blocks.append(f"{TOOL_RESPONSE_OPEN}\n{payload}\n{TOOL_RESPONSE_CLOSE}")
    return "\n".join(blocks)


# --------------------------------------------------------------------------- #
# Schema validation (closes the "calls schema-validated" claim from #102 box #1)
# --------------------------------------------------------------------------- #

def validate_call_against_tools(call: dict, tools: List[dict]) -> bool:
    """True iff `call` ({"name", "arguments"}) is a legal call against `tools`:
    `call["name"]` must name a tool in `tools`, and every name in that tool's
    `parameters["required"]` must be a key of `call["arguments"]`. Lenient by
    design — JSON-level presence only, no type/value checking — so this catches
    hallucinated tool names and missing required args (the two failure modes that
    actually corrupt training signal) without rejecting a correct call over a
    string-vs-int nit. Reused by both the SFT builder (`tool_sft.py`) and the
    BFCL eval harness (`src/eval/bfcl_adapter.py`)."""
    name = call.get("name")
    by_name = {t.get("name"): t for t in tools if isinstance(t, dict)}
    tool = by_name.get(name)
    if tool is None:
        return False
    required = (tool.get("parameters") or {}).get("required") or []
    arguments = call.get("arguments") or {}
    return all(r in arguments for r in required)


# --------------------------------------------------------------------------- #
# Distractor pool + deterministic sampler
# --------------------------------------------------------------------------- #

_DISTRACTOR_POOL: List[dict] = [
    {"name": "send_email", "description": "Send an email to a recipient",
     "parameters": {"type": "object",
                    "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
                    "required": ["to", "body"]}},
    {"name": "create_calendar_event", "description": "Create a calendar event",
     "parameters": {"type": "object",
                    "properties": {"title": {"type": "string"}, "date": {"type": "string"}},
                    "required": ["title", "date"]}},
    {"name": "play_music", "description": "Play a song or playlist",
     "parameters": {"type": "object",
                    "properties": {"track": {"type": "string"}}, "required": ["track"]}},
    {"name": "set_timer", "description": "Set a countdown timer",
     "parameters": {"type": "object",
                    "properties": {"seconds": {"type": "integer"}}, "required": ["seconds"]}},
]


def sample_distractors(real_names: Set[str], k: int, *, rng: random.Random) -> List[dict]:
    """k distractor schemas drawn deterministically from `_DISTRACTOR_POOL`, never overlapping a
    real tool name (a distractor must never be a valid answer)."""
    pool = [t for t in _DISTRACTOR_POOL if t["name"] not in real_names]
    rng.shuffle(pool)
    return pool[:k]


# --------------------------------------------------------------------------- #
# Row builders
# --------------------------------------------------------------------------- #

def build_tool_messages(tools: List[dict], user: str, calls: List[dict], *,
                        results: Optional[Sequence] = None, final: Optional[str] = None,
                        system_preamble: Optional[str] = None,
                        source: str = "handauthored") -> dict:
    """Assemble a tagged row:
        system(render_tool_system(tools)) -> user(user) -> assistant(format_tool_call(calls))
        [-> user(format_tool_response(results)) -> assistant(final)]      # ReAct, if results+final given
    `tools` already includes any distractors. Returns {messages, source, license}."""
    messages = [
        {"role": "system", "content": render_tool_system(tools, preamble=system_preamble)},
        {"role": "user", "content": user.strip()},
        {"role": "assistant", "content": format_tool_call(calls)},
    ]
    if results is not None and final is not None:
        messages.append({"role": "user", "content": format_tool_response(results)})
        messages.append({"role": "assistant", "content": final.strip()})
    return _tag(messages, source)


def build_abstention_messages(tools: List[dict], user: str, response: str, *,
                              system_preamble: Optional[str] = None,
                              source: str = "handauthored") -> dict:
    """system(tools) -> user -> assistant(plain no-call answer). `tools` must list plausible tools
    so the model learns when NOT to call. The assistant content contains NO <tool_call> block."""
    messages = [
        {"role": "system", "content": render_tool_system(tools, preamble=system_preamble)},
        {"role": "user", "content": user.strip()},
        {"role": "assistant", "content": response.strip()},
    ]
    return _tag(messages, source)


# --------------------------------------------------------------------------- #
# Row mappers (pure, network-free, unit-tested) + lazy HF loaders
# --------------------------------------------------------------------------- #

def glaive_row_to_messages(row: dict, source: str = "glaive") -> Optional[dict]:
    """glaiveai/glaive-function-calling-v2: parse the 'system' tool-schema block + the 'chat'
    transcript, mapping FUNCTION CALL turns -> assistant <tool_call>, FUNCTION RESPONSE turns ->
    user <tool_response>. Returns None on malformed rows (drop, never corrupt)."""
    try:
        system_text = (row.get("system") or "").strip()
        chat_text = (row.get("chat") or "").strip()
        if not chat_text:
            return None

        # Parse tools from the system field (after "SYSTEM: " prefix if present)
        tools: List[dict] = []
        if system_text:
            # Try to extract a JSON array of tool definitions
            start = system_text.find("[")
            end = system_text.rfind("]")
            if start != -1 and end != -1 and end > start:
                try:
                    tools = json.loads(system_text[start:end + 1])
                except (json.JSONDecodeError, ValueError):
                    tools = []

        # Parse the chat transcript: lines alternate between USER/ASSISTANT/FUNCTION CALL/etc.
        messages: List[dict] = []
        if tools:
            messages.append({"role": "system",
                              "content": render_tool_system(tools)})

        # Simple line-by-line parse of the glaive chat format
        current_role: Optional[str] = None
        current_lines: List[str] = []

        def flush():
            nonlocal current_role, current_lines
            if current_role and current_lines:
                content = "\n".join(current_lines).strip()
                if content:
                    messages.append({"role": current_role, "content": content})
            current_role = None
            current_lines = []

        for line in chat_text.splitlines():
            if line.startswith("USER:"):
                flush()
                current_role = "user"
                current_lines = [line[5:].strip()]
            elif line.startswith("ASSISTANT:"):
                flush()
                current_role = "assistant"
                current_lines = [line[10:].strip()]
            elif line.startswith("FUNCTION CALL:") or line.startswith("FUNCTION_CALL:"):
                # Tool call from the assistant
                flush()
                current_role = "assistant"
                raw = line.split(":", 1)[1].strip()
                try:
                    call_data = json.loads(raw)
                    formatted = format_tool_call([call_data]) if isinstance(call_data, dict) else raw
                except (json.JSONDecodeError, ValueError):
                    formatted = raw
                current_lines = [formatted]
            elif line.startswith("FUNCTION RESPONSE:") or line.startswith("FUNCTION_RESPONSE:"):
                flush()
                current_role = "user"
                raw = line.split(":", 1)[1].strip()
                current_lines = [f"{TOOL_RESPONSE_OPEN}\n{raw}\n{TOOL_RESPONSE_CLOSE}"]
            else:
                if current_role:
                    current_lines.append(line)
        flush()

        # Validate: must end in assistant, have at least user+assistant
        if not messages or messages[-1]["role"] != "assistant":
            return None
        has_user = any(m["role"] == "user" for m in messages)
        if not has_user:
            return None

        return _tag(messages, source)
    except Exception:
        return None


def xlam_row_to_messages(row: dict, source: str = "xlam") -> Optional[dict]:
    """Salesforce/xlam-function-calling-60k: rows carry {query, tools, answers}. tools -> system
    <tools>; query -> user; answers (list of {name, arguments}) -> one assistant turn via
    format_tool_call (parallel calls stacked). 'tools'/'answers' may be JSON strings -> json.loads."""
    try:
        query = (row.get("query") or "").strip()
        if not query:
            return None

        tools_raw = row.get("tools", [])
        if isinstance(tools_raw, str):
            tools_raw = json.loads(tools_raw)
        tools = list(tools_raw) if tools_raw else []

        answers_raw = row.get("answers", [])
        if isinstance(answers_raw, str):
            answers_raw = json.loads(answers_raw)
        answers = list(answers_raw) if answers_raw else []

        if not answers:
            return None

        # Validate calls are JSON-serializable
        for call in answers:
            json.dumps(call)

        messages = [
            {"role": "system", "content": render_tool_system(tools)},
            {"role": "user", "content": query},
            {"role": "assistant", "content": format_tool_call(answers)},
        ]
        return _tag(messages, source)
    except Exception:
        return None


def toolace_row_to_messages(row: dict, source: str = "toolace") -> Optional[dict]:
    """Team-ACE/ToolACE: multi-turn conversation + tool defs; map roles, tool calls -> assistant
    <tool_call>, tool outputs -> user <tool_response>."""
    try:
        conversations = row.get("conversations") or row.get("messages") or []
        if not conversations:
            return None

        tools_raw = row.get("tools") or row.get("functions") or []
        if isinstance(tools_raw, str):
            tools_raw = json.loads(tools_raw)
        tools = list(tools_raw)

        messages: List[dict] = []
        if tools:
            messages.append({"role": "system", "content": render_tool_system(tools)})

        for turn in conversations:
            role = (turn.get("role") or turn.get("from") or "").lower()
            content = (turn.get("content") or turn.get("value") or "").strip()
            if not content:
                continue

            if role in ("human", "user"):
                messages.append({"role": "user", "content": content})
            elif role in ("gpt", "assistant", "bot"):
                # Check if this is a tool call
                if TOOL_CALL_OPEN in content or '"name"' in content:
                    try:
                        # Try parsing as tool call JSON
                        call_data = json.loads(content)
                        if isinstance(call_data, dict) and "name" in call_data:
                            messages.append({"role": "assistant",
                                             "content": format_tool_call([call_data])})
                            continue
                        elif isinstance(call_data, list):
                            messages.append({"role": "assistant",
                                             "content": format_tool_call(call_data)})
                            continue
                    except (json.JSONDecodeError, ValueError):
                        pass
                messages.append({"role": "assistant", "content": content})
            elif role in ("tool", "function", "observation"):
                # Tool result -> user turn
                try:
                    payload = json.loads(content)
                    result_str = json.dumps(payload)
                except (json.JSONDecodeError, ValueError):
                    result_str = content
                messages.append({"role": "user",
                                 "content": f"{TOOL_RESPONSE_OPEN}\n{result_str}\n{TOOL_RESPONSE_CLOSE}"})
            elif role == "system":
                # Skip system turns if we already set one from tools
                if not any(m["role"] == "system" for m in messages):
                    messages.append({"role": "system", "content": content})

        if not messages or messages[-1]["role"] != "assistant":
            return None
        if not any(m["role"] == "user" for m in messages):
            return None

        return _tag(messages, source)
    except Exception:
        return None


def when2call_row_to_messages(row: dict, source: str = "when2call") -> Optional[dict]:
    """When2Call (the restraint source): rows where correct behavior is NOT to call ->
    build_abstention_messages(tools=..., user=..., response=<refusal/clarification>, source=source)."""
    try:
        query = (row.get("query") or row.get("user") or "").strip()
        response = (row.get("response") or row.get("answer") or "").strip()
        if not query or not response:
            return None

        tools_raw = row.get("tools") or []
        if isinstance(tools_raw, str):
            tools_raw = json.loads(tools_raw)
        tools = list(tools_raw)

        return build_abstention_messages(tools, query, response, source=source)
    except Exception:
        return None


def load_glaive(split: str = "train", max_examples: Optional[int] = None) -> Iterator[dict]:
    """Stream glaiveai/glaive-function-calling-v2 and map rows to tagged tool rows (lazy datasets)."""
    from datasets import load_dataset  # pragma: no cover - network/optional extra

    ds = load_dataset("glaiveai/glaive-function-calling-v2", split=split, streaming=True)
    n = 0
    for row in ds:
        rec = glaive_row_to_messages(row, source="glaive")
        if rec is None:
            continue
        yield rec
        n += 1
        if max_examples is not None and n >= max_examples:
            break


def load_xlam(split: str = "train", max_examples: Optional[int] = None) -> Iterator[dict]:
    """Stream Salesforce/xlam-function-calling-60k and map rows to tagged tool rows (lazy datasets)."""
    from datasets import load_dataset  # pragma: no cover - network/optional extra

    ds = load_dataset("Salesforce/xlam-function-calling-60k", split=split, streaming=True)
    n = 0
    for row in ds:
        rec = xlam_row_to_messages(row, source="xlam")
        if rec is None:
            continue
        yield rec
        n += 1
        if max_examples is not None and n >= max_examples:
            break


def load_toolace(split: str = "train", max_examples: Optional[int] = None) -> Iterator[dict]:
    """Stream Team-ACE/ToolACE and map rows to tagged tool rows (lazy datasets)."""
    from datasets import load_dataset  # pragma: no cover - network/optional extra

    ds = load_dataset("Team-ACE/ToolACE", split=split, streaming=True)
    n = 0
    for row in ds:
        rec = toolace_row_to_messages(row, source="toolace")
        if rec is None:
            continue
        yield rec
        n += 1
        if max_examples is not None and n >= max_examples:
            break


def load_when2call(split: str = "train", max_examples: Optional[int] = None) -> Iterator[dict]:
    """Stream When2Call and map rows to tagged abstention rows (lazy datasets)."""
    from datasets import load_dataset  # pragma: no cover - network/optional extra

    ds = load_dataset("when2call/When2Call", split=split, streaming=True)
    n = 0
    for row in ds:
        rec = when2call_row_to_messages(row, source="when2call")
        if rec is None:
            continue
        yield rec
        n += 1
        if max_examples is not None and n >= max_examples:
            break


# --------------------------------------------------------------------------- #
# Handauthored offline set (checked in, CC0) — offline coverage
# --------------------------------------------------------------------------- #

def handauthored_tool_records() -> Iterator[dict]:
    """Checked-in CC0 set covering all three behaviors offline:
    (1) positive call WITH >=1 distractor in the system tools;
    (2) abstention (asked task has no matching tool -> plain no-<tool_call> answer, distractors present);
    (3) multi-turn ReAct: system -> user -> assistant(<tool_call>) -> user(<tool_response>) -> assistant(final)."""
    rng = random.Random(0)
    weather = {"name": "get_weather", "description": "Get current weather for a city",
               "parameters": {"type": "object",
                              "properties": {"city": {"type": "string"}}, "required": ["city"]}}
    # (1) positive call with distractors
    tools1 = [weather] + sample_distractors({"get_weather"}, 2, rng=rng)
    yield build_tool_messages(
        tools1, "What's the weather in Paris?",
        [{"name": "get_weather", "arguments": {"city": "Paris"}}], source="handauthored")
    # (2) abstention — no tool matches the request
    tools2 = sample_distractors(set(), 3, rng=rng)
    yield build_abstention_messages(
        tools2, "Translate 'hello' into French.",
        "I don't have a tool for translation, but 'hello' in French is 'bonjour'.",
        source="handauthored")
    # (3) ReAct
    tools3 = [weather] + sample_distractors({"get_weather"}, 1, rng=rng)
    yield build_tool_messages(
        tools3, "Is it raining in Tokyo right now?",
        [{"name": "get_weather", "arguments": {"city": "Tokyo"}}],
        results=[{"city": "Tokyo", "condition": "rain", "temp_c": 18}],
        final="Yes, it's currently raining in Tokyo, around 18 degrees Celsius.",
        source="handauthored")


# --------------------------------------------------------------------------- #
# Aggregator
# --------------------------------------------------------------------------- #

_LOADERS = {
    "handauthored": lambda n: handauthored_tool_records(),
    "glaive": lambda n: load_glaive(max_examples=n),
    "xlam": lambda n: load_xlam(max_examples=n),
    "toolace": lambda n: load_toolace(max_examples=n),
    "when2call": lambda n: load_when2call(max_examples=n),
}
# NOTE: BFCL is eval-only (shared/eval/), intentionally NOT a loader here.


def iter_tool_sft(sources: Iterable[str], max_per_source: Optional[int] = None) -> Iterator[dict]:
    """Concatenate tagged tool-use rows from the named sources."""
    for name in sources:
        if name not in _LOADERS:
            raise ValueError(f"unknown tool source {name!r} (have {sorted(_LOADERS)})")
        yield from _LOADERS[name](max_per_source)
