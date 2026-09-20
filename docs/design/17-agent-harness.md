# Agent Harness: Autonomous Execution Loop, Anti-Spin Circuit Breakers, Staged Context Compaction, Native Web Tools, Safety Gates, Planning Scaffolding, and Tool SFT Integration (#349, #350, #351, #352, #366, #367, #368)

This document describes how the Monica agent harness implements autonomous multi-turn ReAct execution loops (`src/agent/runtime.py`, #349), anti-spin circuit breakers (#349), staged context compaction (`src/agent/compaction.py`, #350), native web tools (`src/agent/search.py`, `src/agent/fetcher.py`, #366, #367), deterministic safety gates (`src/agent/safety.py`, #351), capability-adaptive planning scaffolding with out-of-history plan injection (`src/agent/planning.py`, #352), and web tool runtime dispatch with SFT distractor integration (#368) above the hardware seam.

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

## 3. Trajectory Telemetry and Logging (`TrajectoryTelemetry`, #349, #368)

The runtime records comprehensive structured telemetry across each turn and for the complete execution trajectory:
- **Turn Telemetry**: Prompt token counts, completion token counts, model generation wall time, cumulative tool execution time, observation token counts, total turn duration, and phase transitions (`thought`, `action`, `observation`, `completion`).
- **Telemetry Events**: Explicit structured records for `run_start`, `turn_start`, `thought_generated`, `tool_call`, `tool_executed`, `network_event`, `circuit_breaker_redirection`, `circuit_breaker_terminated`, and `run_complete`.
- **`TrajectoryTelemetry` Structure**: Aggregate trajectory metrics captured on `AgentRunResult.telemetry`, tracking total wall time, total tool wall time, prompt and completion token counts, `web_search_count`, `web_fetch_count`, and all `network_events` with request targets, durations, token counts, and error statuses.
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

### Runtime Dispatch Integration and Default Tool Registry (#368)

`WorkspaceToolExecutor` maintains an explicit default tool registry mapping tool names to execution handlers:
- `web_search` and `fetch_web_page` handlers are registered in the default tool registry upon initialization.
- Custom tools can be dynamically registered via `executor.register_tool(name, handler)` or inspected via the `executor.registry` mapping.
- Handlers verify `enable_web_tools` and gracefully return informative errors if web capabilities are disabled.
- Execution timings (`duration_s`), observation token counts (`token_count`), and structured network events (`network_event`) flow automatically into turn events and `AgentRunResult.telemetry`.

### Tool SFT & Evaluation Distractors (#368)

`web_search` and `fetch_web_page` are registered in `CODING_AGENT_TOOLS` in `src/data/tool_sources.py`:
- They are available by default in multi-turn coding agent runs without manual tool concatenation.
- For tasks where external web access is unnecessary (e.g. local syntax fixes or repository refactoring), they serve as negative distractor candidates in tool-use SFT and abstention evaluation pipelines.


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

1. **Soft Threshold Elision (M1, ~0.60 Usable Context, #350, #368)**:
   When conversation token usage exceeds the configured soft threshold (default 0.60 of `max_context_tokens`), the compactor inspects observation messages in the middle region:
   - **Repository & Tool Outputs**: Bulky observation bodies, such as compiler build logs, test outputs, and large file reads exceeding `elision_char_threshold` (default 200 characters), are replaced with `[Output elided: N chars]`.
   - **Web Page Extractions (#368)**: Bulky `fetch_web_page` observation outputs exceeding `elision_char_threshold` are recognized from preceding assistant tool calls and replaced with `[Web page content elided: N chars]`.
   - This operation incurs zero compute or API cost while pruning thousands of tokens of external documentation that has already been synthesized by the model.

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

## 6. Deterministic Safety Gates and Immediate Diagnostic Feedback (#351)

In arXiv:2609.20804, orthogonal deterministic substrate guards proved crucial for harness reliability by surfacing errors immediately and preventing destructive actions:
1. **Immediate Post-Edit Diagnostic Feedback**: In `edit_file` and `write_file` tool handlers, the harness automatically runs fast static analysis and linter diagnostics immediately upon file mutation. Diagnostic findings are appended directly to the tool observation returned to the agent, catching syntax errors and undefined symbols before the agent attempts expensive test runs.
2. **Read-Before-Write Safety Gate**: A session-scoped `FileReadRegistry` maintains records of all files read during the active session along with their SHA-256 content hashes. Edits or writes targeting existing files that have not been read in the current session are rejected with an informative error message prompting the agent to view the file first.
3. **Workspace Containment**: Strict path containment checks prevent relative traversal escapes (`..`), absolute path escapes outside the workspace boundary, and symlink jailbreaks (including traversing through symlinks pointing outside the repository root).

### Read-Before-Write Safety Registry (`FileReadRegistry`)

The registry maintains in-memory records of accessed files:
- `register_read(path, content)`: Computes SHA-256 digest of file content and registers read timestamp and size.
- `register_write(path, content)`: Updates the recorded SHA-256 digest following a successful mutation, allowing subsequent mutations in the same session without redundant reads.
- `is_read(path)`: Verifies whether the target path was accessed during the current session.
- `reset()`: Invoked at the start of each `AgentRuntime.run()` session to ensure read records are strictly session-scoped.

When an edit or overwrite targets an unread existing file, the tool handler rejects the action:

```json
{
  "error": "Read-before-write safety violation: file 'module.py' has not been read in this session. Use view_file to inspect the file before modifying it.",
  "is_error": true,
  "safety_violation": "unread_file"
}
```

### Immediate Post-Mutation Diagnostic Feedback (`run_file_diagnostics`)

Upon writing or editing a file on disk, `WorkspaceToolExecutor` automatically invokes fast static analysis:
- **Python (`.py`, `.pyi`)**: Executes `ruff check --output-format=json --stdin-filename <path> -` over stdin. If `ruff` is unavailable, falls back to `pyflakes` or `ast.parse` syntax checking. Syntax errors and linter issues are parsed into structured items (`line`, `column`, `message`, `code`, `severity`, `source`).
- **TypeScript / JavaScript (`.ts`, `.tsx`, `.js`, `.jsx`)**: Dispatches to `TsLspService` (or custom diagnostic providers), invoking `service.update(path, content)` and querying `service.diagnostics(path)`.
- **JSON (`.json`)**: Validates JSON syntax via standard library `json.loads`.
- Findings are attached to the tool observation payload in `diagnostics` and summarized in `message`.

```json
{
  "status": "ok",
  "path": "app.py",
  "message": "Successfully edited app.py. Static analysis: 1 issue(s) detected. Line 1: [invalid-syntax] Expected a parameter or the end of the parameter list",
  "diagnostics": [
    {
      "line": 1,
      "column": 12,
      "message": "Expected a parameter or the end of the parameter list",
      "code": "invalid-syntax",
      "severity": "error",
      "source": "ruff"
    }
  ]
}
```

### Workspace Containment (`resolve_safe_workspace_path`)

All path-accepting tools (`view_file`, `edit_file`, `write_file`, `find_files`, `grep_search`) resolve paths via `resolve_safe_workspace_path(path_str, workspace_root)`:
- `Path.resolve()` canonicalizes paths, fully expanding symlinks and normalizing `..`.
- Rejects paths where `candidate.is_relative_to(workspace_root)` is `False`.
- Rejects non-existent files targeted through symlinked external parent directories.
- Rejects empty strings and paths containing null bytes (`\x00`).
- Rejections return structured error objects with `"safety_violation": "path_jailbreak"`.

## 7. Capability-Adaptive Planning Scaffolding (`src/agent/planning.py`, #352)

Empirical findings in arXiv:2609.20804 demonstrate an inverse relationship between model scale and planning utility. Planning functions as an accuracy scaffold for sub-frontier models (<100B), but operates as a cost-cutting termination gate for frontier models.

```
+-----------------------------------------------------------------------------+
|               Capability-Adaptive Planning Scaffolding                      |
|                                                                             |
|  1. Out-of-History Plan State:                                              |
|     - Harness maintains active checklist in external PlanManager state.     |
|     - Injects clean {PLAN} block into turn prompt conditioning.             |
|     - Prevents conversational turn growth and context contamination.        |
|                                                                             |
|  2. Policy Selection (PlanningPolicy):                                      |
|     - Sub-frontier (<100B): Enforce mandatory turn 1 plan generation;       |
|       prevent premature task aborts when checklist items remain pending.    |
|     - Frontier (>=100B): Configure plan as completion exit gate;            |
|       terminate execution immediately upon checklist verification.          |
|                                                                             |
|  3. Plan State Mutation:                                                    |
|     - Agents invoke update_plan tool to update checklist items ([ ] -> [x]) |
|       or initialize structured plans.                                       |
+-----------------------------------------------------------------------------+
```

### Out-of-History Plan Injection
To prevent conversational history bloat, `PlanManager` maintains the active plan outside the dialogue message list. For each turn, the runtime generates conditioned messages by injecting the active `{PLAN}` block into the prompt template or system message. The persistent conversation history retains only tool executions and observations, avoiding duplicated planning dialogue turns.

### Capability-Adaptive Scaffolding Policies
`PlanManager` supports two operational policies:
1. **Sub-Frontier Scaffolding (`PlanningPolicy.SUB_FRONTIER`)**: Sub-frontier models (<100B parameters) tend to abandon complex code localization tasks early. This policy enforces plan generation on turn 1. When a model attempts completion before all checklist items are marked complete, the runtime rejects the early abort and injects a scaffolding redirection reminder.
2. **Frontier Exit Gate (`PlanningPolicy.FRONTIER`)**: Frontier models frequently engage in repetitive post-edit verification loops. This policy instructs the model to exit once verification items are completed. When all checklist items are marked complete via `update_plan`, the runtime triggers an immediate clean completion, reducing API token costs.
3. **Adaptive Auto-Resolution (`PlanningPolicy.ADAPTIVE`)**: Selects `SUB_FRONTIER` for models under 100B parameters, and `FRONTIER` for frontier models based on model parameter count or model identifiers.

### Plan Mutation Tool (`update_plan`)
The `update_plan` tool allows models to mutate checklist states:
- `step`: 1-based index of the target step.
- `completed`: Boolean status (`True` for `[x]`, `False` for `[ ]`).
- `plan`: Markdown checklist string for initializing or replacing the active plan.
- `steps`: Array of string step descriptions.
- `updates`: Batch update list specifying step indices and completion statuses.


## 8. Benchmark POC & MVP Runs (`src/agent/benchmark.py`, `scripts/run_agent_benchmark.py`, #371)

Milestone 12 tracks the operational execution runs of the autonomous ReAct coding agent harness against synthetic and real-world software engineering benchmarks across two stages:

```
+-----------------------------------------------------------------------------------+
|               Autonomous Coding Agent Benchmark Architecture                      |
|                                                                                   |
|  Stage 1: POC Run (Operational Stress Scenarios)                                  |
|    1. Anti-Spin Circuit Breaker: 5-repeat redirect, 8-repeat early termination     |
|    2. Staged Context Compaction: 0.60 soft elision, 0.85 hard summarization       |
|    3. Post-Edit Diagnostics & Safety: instant syntax feedback, read-before-write   |
|    4. Out-of-History Plan Injection: prompt conditioning & clean exit gate        |
|    5. Native Web Access: compact search (<400 tok), SSRF defense, 12k char limit  |
|                                                                                   |
|  Stage 2: MVP Run (Full Benchmark Evaluation)                                     |
|    - SWE-bench-lite: multi-file repository bugfixes & regression suites           |
|    - HumanEval: TypeScript function synthesis & test suites (eval_sets/)          |
|    - Repo Refactoring: RLVR behavioral invariance & interface decoupling          |
|                                                                                   |
|  Telemetry Profiles: pass@k, token consumption, trajectory length, latency profiles|
|  Cloud Specifications: RunPod hardware tiers, GPU-hours, and cost modeling        |
+-----------------------------------------------------------------------------------+
```

### Stage 1: POC Run (Synthetic & Sandbox Validation)
Stage 1 tests the agent harness under operational stress in isolated sandbox workspaces:
1. **Anti-Spin Circuit Breakers (#349)**: Monitors consecutive identical tool calls or failures. Injects corrective redirection warnings at 5 repeats and terminates execution early at 8 repeats (`status="circuit_breaker_tripped"`) without wasting the max turn budget.
2. **Staged Context Compaction (#350)**: Applies rule-based soft elision to bulky middle observations at 60% usable context, and hard LLM summarization at 85% usable context. Preamble and the most recent turns are strictly preserved verbatim without context window blowup.
3. **Post-Edit Diagnostic Feedback & Safety Gates (#351)**: Attaches immediate AST/LSP syntax diagnostics directly to file mutation tool observations. Rejects edits to unread files under read-before-write safety, and catches path traversal jailbreaks.
4. **Out-of-History Plan Injection (#352)**: Maintains structured plans in external state, injecting active checklist blocks into prompt conditioning without accumulating noisy conversational turns. Triggers clean exit gate upon checklist completion.
5. **Native Web Search & Page Fetch (#366-#368)**: Enforces compact search result schemas (<400 tokens), blocks loopback and RFC 1918 addresses via SSRF defense, and caps page fetch content at 12,000 characters.

### Stage 2: MVP Run (Full Benchmark Evaluation)
Stage 2 evaluates the harness against repository-level benchmarks:
- **SWE-bench-lite**: Multi-file repository tasks requiring multi-turn code exploration, localization, editing, and bash test verification.
- **HumanEval**: TypeScript function generation and assertion testing drawn from `eval_sets/humaneval_ts/humaneval_ts.jsonl`.
- **Repo Refactoring Suites**: Multi-file refactoring tasks (e.g. monolithic code to Strategy and Dependency Injection patterns) interfacing directly with deterministic RLVR verifiers (`RefactoringVerifier` and `ExecutionVerifier`) to verify 100% test pass rates, interface mockability, zero real socket/DB leakage, and zero escape hatches.

### Telemetry Profiling & Output Persistence
The benchmark pipeline records comprehensive operational metrics:
- **Pass@k**: Evaluates solution pass rates (`pass@1`).
- **Token Consumption**: Measures prompt tokens, completion tokens, total tokens, and per-task averages.
- **Trajectory Profiles**: Analyzes turn distributions, mean trajectory length, min, and max turns per task.
- **Latency Profiles**: Breaks down model generation wall time, tool execution wall time, and total task wall time.
- **Transcripts**: Streams structured JSON results (`--output`) and per-instance JSONL execution transcripts (`--transcript`).

### Cloud Compute & RunPod Execution Specifications
For full-scale evaluation across hundreds of benchmark tasks with local models:
- **Hardware Tiers**:
  - `a40` (Recommended): NVIDIA A40 (48GB VRAM) at ~$0.40/hr (Community). Ideal for sub-frontier models and multi-agent harness sweeps.
  - `rtx4090`: NVIDIA GeForce RTX 4090 (24GB VRAM) at ~$0.44/hr.
  - `a100`: NVIDIA A100-PCIE-80GB (80GB VRAM) at ~$1.89/hr (Secure). High memory bandwidth for concurrent large-context agent runs.
  - `h100`: NVIDIA H100-SXM-80GB (80GB VRAM) at ~$3.29/hr (Secure).
- **Execution Safety Policy**: Autonomous subagents do not provision remote cloud GPU instances directly. The CLI provides `--cloud-spec` and `--dry-run` modes to output cloud compute parameters, runtime estimates, and launch commands for human review before deployment.
