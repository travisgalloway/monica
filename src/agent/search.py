"""Native web search discovery tool with Brave Search provider (#366).

Coding agents require web search capabilities to discover technical documentation,
API references, and error signatures without bloating model context with raw SERP markup.
This module implements the discovery stage of a two-stage web architecture (web_search
discovery -> fetch_web_page deep article extraction).

ABOVE THE SEAM — pure Python standard library (urllib, json) only.
No hardware backend or third-party HTTP dependencies.
"""

from __future__ import annotations

import html
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

BRAVE_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
DEFAULT_TIMEOUT_S = 5.0
DEFAULT_COUNT = 5
MAX_COUNT = 10
MAX_SNIPPET_CHARS = 300


def _clean_text(text: str | None) -> str:
    """Strip HTML tags, unescape HTML entities, and normalize whitespace."""
    if not text:
        return ""
    # Strip HTML tags
    cleaned = re.sub(r"<[^>]+>", "", text)
    # Unescape HTML entities
    cleaned = html.unescape(cleaned)
    # Normalize multiple whitespace characters to single spaces
    return " ".join(cleaned.split())


def distill_search_response(raw_response: Any, count: int = DEFAULT_COUNT) -> list[dict[str, str]]:
    """Distill raw Brave Search API response into compact, token-optimized JSON records.

    Strictly limits records to title, url, and snippet fields. Target payload < 400 tokens.
    """
    if not isinstance(raw_response, dict):
        return []

    web_obj = raw_response.get("web")
    if not isinstance(web_obj, dict):
        return []

    raw_results = web_obj.get("results")
    if not isinstance(raw_results, list):
        return []

    try:
        limit = max(1, min(int(count), MAX_COUNT))
    except (ValueError, TypeError):
        limit = DEFAULT_COUNT

    distilled: list[dict[str, str]] = []
    for item in raw_results[:limit]:
        if not isinstance(item, dict):
            continue

        title = _clean_text(item.get("title"))
        url = str(item.get("url") or "").strip()
        raw_snippet = item.get("description") or item.get("snippet") or ""
        snippet = _clean_text(raw_snippet)

        if len(snippet) > MAX_SNIPPET_CHARS:
            snippet = snippet[: MAX_SNIPPET_CHARS - 3].rstrip() + "..."

        distilled.append({
            "title": title,
            "url": url,
            "snippet": snippet,
        })

    return distilled


@dataclass
class SearchResult:
    """Structured search response supporting iteration, slicing, and string/json formatting."""

    records: list[dict[str, str]]
    error: str | None = None
    cached: bool = False

    @property
    def is_success(self) -> bool:
        """True if the search succeeded without error."""
        return self.error is None

    def to_json(self, indent: int | None = None) -> str:
        """Serialize search records or error to JSON string."""
        if self.error is not None:
            return json.dumps({"error": self.error})
        return json.dumps(self.records, indent=indent)

    def __str__(self) -> str:
        """Human- and LLM-readable representation of search results or error."""
        if self.error is not None:
            if self.error.startswith("Error:"):
                return self.error
            return f"Error: {self.error}"
        return json.dumps(self.records, indent=2)

    def __iter__(self) -> Iterator[dict[str, str]]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.records[index]

    def __bool__(self) -> bool:
        return self.error is None and len(self.records) > 0


class BraveSearchClient:
    """Client for Brave Search API with in-turn query caching, timeouts, and distillation."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        endpoint: str = BRAVE_SEARCH_ENDPOINT,
    ) -> None:
        raw_key = api_key if api_key is not None else os.environ.get("BRAVE_SEARCH_API_KEY", "")
        self.api_key = raw_key.strip()
        self.timeout_s = float(timeout_s)
        self.endpoint = endpoint
        self._cache: dict[tuple[str, int], SearchResult] = {}

    def clear_cache(self) -> None:
        """Clear query cache within the execution turn/session."""
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        """Number of cached queries in this session."""
        return len(self._cache)

    def search(self, query: str, count: int = DEFAULT_COUNT) -> SearchResult:
        """Execute web search query against Brave Search API.

        Catches all network and serialization errors, returning descriptive
        error strings rather than raising unhandled exceptions.
        """
        if not query or not query.strip():
            return SearchResult(records=[], error="Search query cannot be empty.")

        q = query.strip()
        try:
            clamped_count = max(1, min(int(count), MAX_COUNT))
        except (ValueError, TypeError):
            clamped_count = DEFAULT_COUNT

        cache_key = (q.lower(), clamped_count)
        if cache_key in self._cache:
            cached_res = self._cache[cache_key]
            return SearchResult(
                records=list(cached_res.records),
                error=cached_res.error,
                cached=True,
            )

        if not self.api_key:
            return SearchResult(
                records=[],
                error="BRAVE_SEARCH_API_KEY is not set. Web search requires a Brave Search API key.",
            )

        params = urllib.parse.urlencode({"q": q, "count": clamped_count})
        url = f"{self.endpoint}?{params}"
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key,
            "User-Agent": "monica-agent/1.0",
        }

        req = urllib.request.Request(url, headers=headers, method="GET")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
                raw_bytes = response.read()
                try:
                    payload = json.loads(raw_bytes.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    return SearchResult(
                        records=[],
                        error=f"Failed to decode Brave search API response as JSON: {e}",
                    )

                records = distill_search_response(payload, count=clamped_count)
                result = SearchResult(records=records, error=None, cached=False)
                self._cache[cache_key] = result
                return result

        except urllib.error.HTTPError as e:
            if e.code == 401:
                detail = "authentication failed (HTTP 401): Invalid or unauthorized API key"
            elif e.code == 429:
                detail = "rate limit exceeded (HTTP 429): Too many search requests"
            else:
                detail = f"HTTP request failed with status code {e.code}: {e.reason}"
            return SearchResult(records=[], error=f"Brave search API {detail}")

        except TimeoutError:
            return SearchResult(
                records=[],
                error=f"Brave search request timed out after {self.timeout_s}s for query: {q!r}",
            )

        except urllib.error.URLError as e:
            if "timed out" in str(e.reason).lower():
                return SearchResult(
                    records=[],
                    error=f"Brave search request timed out after {self.timeout_s}s for query: {q!r}",
                )
            return SearchResult(
                records=[],
                error=f"Brave search network error: {e.reason}",
            )

        except OSError as e:
            return SearchResult(
                records=[],
                error=f"Unexpected search OS/network error: {e}",
            )


def web_search(
    query: str,
    count: int = DEFAULT_COUNT,
    *,
    client: BraveSearchClient | None = None,
    api_key: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> str:
    """Execute a web search discovery query and return distilled JSON or error string.

    Designed for direct registration as an agent harness tool handler.
    """
    c = client or BraveSearchClient(api_key=api_key, timeout_s=timeout_s)
    res = c.search(query, count=count)
    return str(res)
