# Agent Harness: Autonomous Execution Loop, Anti-Spin Circuit Breakers, Staged Context Compaction, and Native Web Tools (#349, #350, #366, #367)

This document describes how the Monica agent harness implements autonomous multi-turn ReAct execution loops (`src/agent/runtime.py`, #349), anti-spin circuit breakers (#349), staged context compaction (`src/agent/compaction.py`, #350), and native web tools (`src/agent/search.py`, `src/agent/fetcher.py`, #366, #367) above the hardware seam.

## 1. Autonomous Multi-Turn Execution Loop (`src/agent/runtime.py`)

The agent runtime drives multi-turn ReAct loops (`Thought` -> `Action` -> `Tool Observation`) over repository workspaces and instruction-tuned language models.

```
+-------------------------------------------------------------------+
|                        AgentRuntime Loop                          |
|                                                                   |
|   1. System & Task Prompt Formulation (<tools>[...]</tools>)      |
|   2. Language Model Turn Generation (LMAdapter / Callable)        |
|   3. Parse ReAct Response (<think>, <tool_call>, completion)      |
|        |                                                          |
|        +---> No tool calls: Completion Turn -> Return Result      |
|        |                                                          |
|        +---> Tool calls:                                          |
|                a. Circuit breaker call inspection                 |
|                b. Execute tool (WorkspaceToolExecutor)            |
|                c. Circuit breaker result inspection               |
|                d. Format tool observation (<tool_response>)       |
|                e. Check termination or inject redirection reminder|
|   4. Repeat until completion, breaker trip, or turn budget limit  |
+-------------------------------------------------------------------+
```

### Seam Isolation and Portability
The runtime resides strictly Above the Seam. It imports only standard library modules (`os`, `sys`, `time`, `json`, `re`, `pathlib`, `subprocess`, `fnmatch`, `dataclasses`) and NumPy. Hardware backends (`mlx`, `torch`, `bitsandbytes`) are forbidden and verified by `tests/test_import_guard.py`.

### Pluggable Backend Adapters
The runtime accepts multiple model backend interfaces:
1. **Callable Functions**: High-level test doubles or HTTP inference wrappers taking conversation messages and returning assistant strings.
2. **`LMAdapter` Protocol**: Stepwise generation models implementing `reset(context)`, `step(token_id)`, and `decode(token_ids)`. If the adapter provides `render_chat`, the runtime uses tokenizer-specific formatting; otherwise, it applies standard ChatML rendering via `src.data.chat_template.render`.
3. **Generative Objects**: Objects exposing `generate(messages)` or `chat(messages)`.

### Workspace Tool Execution
`WorkspaceToolExecutor` provides repository manipulation routines for tools defined in `CODING_AGENT_TOOLS`:
- `execute_bash`: Runs shell commands within the repository root with process timeouts and output capture.
- `view_file`: Reads files and slices specified line ranges with line numbers.
- `edit_file`: Replaces unique string patterns in workspace files, rejecting missing or ambiguous targets.
- `grep_search`: Searches regex and literal patterns across files in the workspace.
- `find_files`: Resolves glob patterns recursively within repository paths.
- Path containment validation verifies that relative and absolute file paths do not escape the workspace root.

## 2. Anti-Spin Circuit Breakers

Autonomous agents can become stuck in repetitive loops, such as calling identical tools with identical failing arguments or repeating actions without making progress. Anti-spin circuit breakers track tool invocations in real time to prevent context and turn budget exhaustion.

### Canonical Tool Call Signatures
Tool invocations are normalized to canonical byte sequences via `canonical_call_bytes`. Dictionary arguments are serialized with sorted keys and compact separators (`json.dumps(args, sort_keys=True, separators=(',', ':'))`). This ensures that identical calls with differing dictionary key order or whitespace produce identical byte representations.

### Detection and Escalation
`AntiSpinCircuitBreaker` tracks two counters for consecutive identical invocations:
1. `consecutive_identical_calls`: Number of uninterrupted invocations with identical tool name and canonical arguments.
2. `consecutive_identical_failures`: Number of uninterrupted identical invocations whose execution resulted in an error or non-zero exit code.

Counters reset whenever the tool name or arguments change, or when a failing call succeeds.

Escalation proceeds in two stages:
- **Redirection Reminder (5 repeats)**: When either counter reaches 5 consecutive repeats, the runtime injects a corrective notice into the observation context:
  `[CIRCUIT BREAKER REDIRECTION] Detected 5 consecutive identical tool calls for '<tool>'. Stop repeating identical calls. Adjust arguments, use a different tool, or provide your final answer.`
- **Early Termination (8 repeats)**: When either counter reaches 8 consecutive repeats, execution terminates immediately. The run status is set to `circuit_breaker_tripped`, and a dedicated failure reason is recorded (e.g. `anti_spin_circuit_breaker: 8 consecutive identical failures for tool '<tool>'`). The agent halts without exhausting the remaining turn budget.

## 3. Trajectory Telemetry and Logging

The runtime records structured telemetry for every turn:
- **Turn Telemetry**: Prompt token counts, completion token counts, model generation wall time, cumulative tool execution time, total turn duration, and phase transitions (`thought`, `action`, `observation`, `completion`).
- **Telemetry Events**: Explicit records for `run_start`, `turn_start`, `thought_generated`, `tool_call`, `tool_executed`, `circuit_breaker_redirection`, `circuit_breaker_terminated`, and `run_complete`.
- **Pluggable Loggers**: Sinks implement the `TrajectoryLogger` protocol. `InMemoryTrajectoryLogger` retains turns and results in memory for evaluation. `JsonlTrajectoryLogger` streams turn and trajectory records to disk.

## 4. Two-Stage Web Access (`web_search` and `fetch_web_page`)

Coding agents often require technical documentation, API specifications, and error messages that postdate model training cutoffs. Providing this capability through unconstrained webpage retrieval risks context exhaustion. Raw search engine results pages contain navigation links, advertisements, and repetitive markup that consume tokens without providing technical signal.

Monica separates web access into two discrete stages:
1. **Search Discovery (`web_search`)**: High-level query execution returning distilled, token-budgeted summary records (`title`, `url`, `snippet`).
2. **Deep Article Extraction (`fetch_web_page`)**: Targeted markdown extraction for specific URLs selected by the model after reviewing search results.

This division ensures that search discovery consumes fewer than 400 tokens per call, while deep page extraction enforces a strict 12,000-character ceiling (~3,000 tokens).

### Tool Schema Specifications

Tool schemas are declared in `src/data/tool_sources.py` and verified by `validate_call_against_tools`.

#### `web_search` Schema

```json
{
  "name": "web_search",
  "description": "Search the web for technical documentation, API references, and error signatures",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Search query keywords or error signature"
      },
      "count": {
        "type": "integer",
        "description": "Number of search results to return (default: 5, max: 10)",
        "default": 5,
        "maximum": 10
      }
    },
    "required": ["query"]
  }
}
```

The `query` parameter is required. The optional `count` parameter specifies the number of search results to return, defaulting to 5 and clamped to a maximum of 10.

#### `fetch_web_page` Schema

```json
{
  "name": "fetch_web_page",
  "description": "Fetch and extract readable markdown content from a web page URL",
  "parameters": {
    "type": "object",
    "properties": {
      "url": {
        "type": "string",
        "description": "The web page URL to fetch and extract content from"
      }
    },
    "required": ["url"]
  }
}
```

The `url` parameter is required and accepts HTTP and HTTPS web addresses.

### Brave Search Provider Implementation

`BraveSearchClient` queries the Brave Search API endpoint:
`https://api.search.brave.com/res/v1/web/search`

Authentication uses an API key configured via the `BRAVE_SEARCH_API_KEY` environment variable or passed directly to `BraveSearchClient(api_key=...)`.

#### Token-Optimized Distillation

Raw responses from search providers contain nested structures and metadata. `distill_search_response` strips extraneous attributes and produces records with three keys:
1. `title`: Page title with HTML markup stripped and entities unescaped.
2. `url`: Direct destination URL.
3. `snippet`: Extracted page description, sanitized of HTML tags, normalized of excess whitespace, and capped at 300 characters.

A payload of 5 distilled results averages 250 to 350 tokens, remaining below the 400-token ceiling.

### Page Extraction Implementation (`fetch_web_page`)

`PageFetcherClient` provides secure webpage retrieval and clean markdown extraction in `src/agent/fetcher.py`.

#### SSRF Ingress Defense

Uncontrolled agent network fetches introduce Server-Side Request Forgery vulnerabilities against local host services and cloud metadata endpoints. `validate_url_for_ssrf` inspects target URLs prior to socket creation:
1. **Scheme Validation**: Only `http` and `https` schemes are permitted. Schemes such as `file`, `ftp`, and `gopher` are rejected.
2. **Reserved Hostname Blocking**: Hostnames matching `localhost` or ending with `.localhost`, `.local`, `.internal`, `.arpa`, `.corp`, `.lan`, and `.home` are rejected.
3. **Restricted IP Detection**: Destination IP addresses are checked against private RFC 1918 ranges (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), loopbacks (`127.0.0.0/8`, `::1`), link-local subnets (`169.254.0.0/16`, `fe80::/10`), cloud metadata addresses (`169.254.169.254`, `fd00:ec2::254`), and IPv4-mapped IPv6 equivalents.
4. **Pre-Connect DNS Resolution**: Hostnames are resolved to IP addresses via `socket.getaddrinfo` before connection initiation. If any resolved IP falls within a restricted range, the request is aborted.
5. **Redirect Verification**: `SSRFSafeRedirectHandler` validates every HTTP redirect destination URL against SSRF criteria.

#### Clean Markdown Extraction

The client supports two extraction strategies:
1. **Off-Box Jina Reader Proxy**: Requests route to `https://r.jina.ai/<target_url>` with header `Accept: text/markdown` and an optional `JINA_API_KEY`. Headless JavaScript rendering and HTML-to-markdown conversion occur off-box, returning clean markdown without local browser automation overhead.
2. **Offline Fallback Parser**: If the proxy is unavailable or disabled (`use_jina=False`), the client executes a direct HTTP GET request and passes HTML to `HTMLToMarkdownParser`. This parser uses the standard library `html.parser.HTMLParser` to convert structural tags into clean markdown while discarding `<script>`, `<style>`, `<head>`, and `<svg>` elements.

#### Token Ceiling and Truncation Guardrail

Large web pages can exhaust agent context windows. `PageFetcherClient` enforces bounds:
1. **12,000-Character Ceiling**: Extracted content is clamped to 12,000 characters (~3,000 tokens).
2. **Truncation Notice**: Excess text is cut cleanly and appended with notice `\n\n[Content truncated at 12000 chars to preserve context window...]`. The result object sets `truncated=True`.
3. **8-Second Network Timeout**: All network operations enforce an 8.0-second hard timeout.

### Deduplication and Resilience

The agent harness operates in automated ReAct loops where network interruptions or repetitive model behaviors must not abort the overall session:
1. **Hard Request Timeouts**: Default 5.0 seconds for `web_search` and 8.0 seconds for `fetch_web_page`.
2. **In-Turn Deduplication**: Both clients maintain in-memory session caches to serve repeated identical requests instantly without repeating network traffic.
3. **Structured Error Reporting**: Network errors, HTTP status errors, and timeouts return structured result objects with explanatory error messages rather than unhandled exceptions.


## 5. Staged Context Compaction (T4: Soft Elision and Hard Summarization, #350)

Empirical findings from arXiv:2609.20804 show that staged two-tier context compaction (T4) delivers the lowest token consumption across benchmarks while matching full-context accuracy. The compaction pipeline combines rule-based observation elision at a soft threshold with language-model summarization at a hard threshold.

```
+-----------------------------------------------------------------------------+
|                     Staged Context Compaction Pipeline                      |
|                                                                             |
|  1. Ingestion: Partition history into Preamble, Middle Turns, Recent Window  |
|                                                                             |
|  2. Check Soft Threshold (~0.60 usable context budget):                     |
|       - If context <= soft threshold: retain history unchanged.             |
|       - If context > soft threshold: trigger Soft Elision (M1).             |
|         Replace bulky middle observations with [Output elided: N chars].    |
|                                                                             |
|  3. Check Hard Threshold (~0.85 usable context budget):                     |
|       - If context <= hard threshold after elision: complete compaction.    |
|       - If context > hard threshold: trigger Hard Summarization (M3).       |
|         Compress oldest middle turns into a concise narrative summary.      |
|                                                                             |
|  4. Invariant: System preamble and recent turns (>= 2) preserved verbatim.  |
+-----------------------------------------------------------------------------+
```

### Two-Tier Compaction Stages

1. **Soft Threshold Elision (M1, ~0.60 Usable Context)**:
   When conversation token usage exceeds the configured soft threshold (default 0.60 of `max_context_tokens`), the compactor inspects observation messages in the middle region. Bulky observation bodies, such as compiler build logs, test outputs, and large file reads exceeding `elision_char_threshold` (default 200 characters), are replaced with `[Output elided: N chars]`, where N records the original character count. This operation incurs zero compute or API cost.

2. **Hard Threshold Summarization (M3, ~0.85 Usable Context)**:
   If conversation token count continues to exceed the hard threshold (default 0.85 of `max_context_tokens`) after soft elision, the compactor invokes language-model summarization. It extracts actions, tool invocations, and key findings from the oldest middle turns and compresses them into a concise narrative summary message (`[Context Summary: <narrative>]`). If no external language model is supplied, the compactor generates a deterministic extractive narrative summary.

### Protected Window Invariant

The compactor enforces a strict window preservation invariant:
1. **System Preamble**: The initial system instructions and the first user message containing the task prompt are never elided or summarized. They are strictly preserved verbatim.
2. **Recent Window**: The most recent >= 2 turns (default `protect_recent_turns = 2`, approximately 0.30 usable context) are never elided or summarized. Even if an observation in the recent window exceeds the elision threshold, it remains intact to ensure immediate conversational context is available to the model.

### Omission of Lossless Recall Machinery

Evaluation in arXiv:2609.20804 confirmed that models virtually never invoke external retrieval tools (`recall_event`) when offered (-0.36% net delta; median calls = 0). Maintaining vector indices, disk caches, and additional tool definitions introduces cognitive distraction in model decision-making and adds prompt token overhead. The implementation explicitly omits recall tools and storage caches.

### Telemetry and Integration

`AgentRuntime` evaluates `ContextCompactor` at the start of each turn. When compaction occurs, the runtime appends a `compaction` phase transition to the active turn and logs a `context_compacted` telemetry event containing pre-compaction and post-compaction token counts, elided observation counts, elided character totals, and summarized turn counts.
