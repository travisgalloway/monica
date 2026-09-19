# Agent Harness: Native Web Tools (Search Discovery & Page Extraction)

How the Monica agent harness implements native web search discovery (`web_search`, #366) and targeted webpage extraction (`fetch_web_page`, #367) above the hardware seam.

## Architectural Context: Two-Stage Web Access

Coding agents often require access to technical documentation, API specifications, and error messages that postdate model training cutoffs. Providing this capability through unconstrained webpage retrieval risks context exhaustion. Raw search engine results pages (SERPs) contain navigation links, advertisements, and repetitive boilerplate markup that consume thousands of tokens without providing technical signal.

Monica separates web access into two discrete stages:
1. **Search Discovery (`web_search`)**: High-level query execution returning distilled, token-budgeted summary records (`title`, `url`, `snippet`).
2. **Deep Article Extraction (`fetch_web_page`)**: Targeted markdown extraction for specific URLs selected by the model after reviewing search results.

This division ensures that search discovery consumes fewer than 400 tokens per call, while deep page extraction enforces a strict 12,000-character ceiling (~3,000 tokens).

## Hardware Seam Isolation

Both `src/agent/search.py` and `src/agent/fetcher.py` reside strictly Above the Seam. They use Python standard library modules exclusively (`urllib.request`, `urllib.parse`, `urllib.error`, `socket`, `ipaddress`, `html.parser`, `dataclasses`, `json`, `re`, `os`).

No hardware backends (`mlx`, `torch`, `bitsandbytes`) are imported. Module portability is enforced continuously by `tests/test_import_guard.py`.

## Tool Schema Specifications

Tool schemas are declared in `src/data/tool_sources.py` and verified by `validate_call_against_tools`.

### `web_search` Schema

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

### `fetch_web_page` Schema

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

## Page Extraction Implementation (`fetch_web_page`)

`PageFetcherClient` provides secure webpage retrieval and clean markdown extraction in `src/agent/fetcher.py`.

### SSRF Ingress Defense

Uncontrolled agent network fetches introduce Server-Side Request Forgery (SSRF) vulnerabilities against local host services and cloud metadata endpoints. `validate_url_for_ssrf` inspects target URLs prior to socket creation:

1. **Scheme Validation**: Only `http` and `https` schemes are permitted. Schemes such as `file`, `ftp`, and `gopher` are rejected.
2. **Reserved Hostname Blocking**: Hostnames matching `localhost` or ending with `.localhost`, `.local`, `.internal`, `.arpa`, `.corp`, `.lan`, and `.home` are rejected.
3. **Restricted IP Detection**: Destination IP addresses are checked against private RFC 1918 ranges (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), loopbacks (`127.0.0.0/8`, `::1`), link-local subnets (`169.254.0.0/16`, `fe80::/10`), cloud metadata addresses (`169.254.169.254`, `fd00:ec2::254`), and IPv4-mapped IPv6 equivalents.
4. **Pre-Connect DNS Resolution**: Hostnames are resolved to IP addresses via `socket.getaddrinfo` before connection initiation. If any resolved IP falls within a restricted range, the request is aborted.
5. **Redirect Verification**: `SSRFSafeRedirectHandler` validates every HTTP redirect destination URL against SSRF criteria, preventing redirect bypass attacks.

### Clean Markdown Extraction

The client supports two extraction strategies:

1. **Off-Box Jina Reader Proxy**: Requests route to `https://r.jina.ai/<target_url>` with header `Accept: text/markdown` and an optional `JINA_API_KEY`. Headless JavaScript rendering and HTML-to-markdown conversion occur off-box, returning clean markdown without local browser automation overhead.
2. **Offline Fallback Parser**: If the proxy is unavailable or disabled (`use_jina=False`), the client executes a direct HTTP GET request and passes HTML to `HTMLToMarkdownParser`. This parser uses the standard library `html.parser.HTMLParser` to convert structural tags (`<h1>`-`<h6>`, `<p>`, `<a>`, `<li>`, `<pre>`, `<code>`) into clean markdown while discarding `<script>`, `<style>`, `<head>`, and `<svg>` elements.

### Token Ceiling and Truncation Guardrail

Massive web pages can exhaust agent context windows. `PageFetcherClient` enforces two bounds:

1. **12,000-Character Ceiling**: Extracted content is clamped to 12,000 characters (~3,000 tokens).
2. **Truncation Notice**: Excess text is cut cleanly and appended with notice `\n\n[Content truncated at 12000 chars to preserve context window...]`. The result object sets `truncated=True`.
3. **8-Second Network Timeout**: All network operations enforce an 8.0-second hard timeout.

## Resilience and Deduplication

The agent harness operates in automated ReAct loops where network interruptions or repetitive model behaviors must not abort the overall session:

1. **Hard Request Timeouts**: Default 5.0 seconds for `web_search` and 8.0 seconds for `fetch_web_page`.
2. **In-Turn Deduplication**: Both clients maintain in-memory session caches to serve repeated identical requests instantly without repeating network traffic.
3. **Graceful Failure Reporting**: Network errors, HTTP status errors, and timeouts return structured result objects with explanatory error messages rather than unhandled exceptions.
