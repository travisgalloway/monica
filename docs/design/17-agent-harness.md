# Agent Harness: Native Web Search Discovery & Brave Provider

Why the Monica agent harness includes native web search discovery (`web_search`), how the Brave Search API provider is integrated above the hardware seam, and how search payloads stay within compact token budgets (#366).

## Architectural Context: Two-Stage Web Access

Coding agents often need access to technical documentation, API specifications, and error messages that postdate model training cutoffs. Providing this capability through full webpage retrieval risks context exhaustion. Raw search engine results pages (SERPs) contain navigation links, advertisements, and repetitive boilerplate markup that consume thousands of tokens without providing technical signal.

Monica separates web access into two discrete stages:
1. **Search Discovery (`web_search`)**: High-level query execution returning distilled, token-budgeted summary records (`title`, `url`, `snippet`).
2. **Deep Article Extraction (`fetch_web_page`)**: Targeted markdown extraction for specific URLs selected by the model after reviewing search results.

This division ensures that search discovery consumes fewer than 400 tokens per call, preserving context space for codebase reasoning and diff synthesis.

## Hardware Seam Isolation

The search client in `src/agent/search.py` is strictly Above the Seam. It uses only Python standard library modules (`urllib.request`, `urllib.parse`, `urllib.error`, `json`, `html`, `dataclasses`).

No hardware backends (`mlx`, `torch`, `bitsandbytes`) are imported. Module portability is enforced continuously by `tests/test_import_guard.py`.

## Tool Schema Specification

The `web_search` tool schema is defined in `src/data/tool_sources.py` and validated by `validate_call_against_tools`:

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

## Brave Search Provider Implementation

`BraveSearchClient` queries the Brave Search API endpoint:
`https://api.search.brave.com/res/v1/web/search`

Authentication uses an API key configured via the `BRAVE_SEARCH_API_KEY` environment variable or passed directly to `BraveSearchClient(api_key=...)`.

### Token-Optimized Distillation

Raw responses from search providers contain nested structures and metadata (such as page age, family friendly flags, and thumbnails). `distill_search_response` strips all extraneous attributes and produces records with three keys:

1. `title`: Page title with HTML markup stripped and entities unescaped.
2. `url`: Direct destination URL.
3. `snippet`: Extracted page description or snippet, sanitized of HTML tags, normalized of excess whitespace, and capped at 300 characters.

A payload of 5 distilled results averages 250 to 350 tokens, remaining comfortably below the 400-token ceiling.

## Resilience and Deduplication

The agent harness operates in automated ReAct loops where network flakiness or repetitive model behaviors must not abort the overall session.

1. **Hard Request Timeout**: All network calls enforce a default 5.0-second timeout. If the remote service does not respond within this window, the client aborts and returns a descriptive error message.
2. **In-Turn Query Caching**: LLMs occasionally issue identical or case-variant search queries across consecutive reasoning turns. `BraveSearchClient` maintains an in-memory session cache keyed on `(query.strip().lower(), count)`. Repeated queries return cached results immediately without making external network calls.
3. **Graceful Failure Reporting**: The provider catches `urllib.error.HTTPError` (including 401 Unauthorized and 429 Rate Limit), `urllib.error.URLError`, timeouts, and malformed JSON. Rather than allowing exceptions to escape and crash the harness loop, the client returns structured `SearchResult` objects with descriptive error strings.
