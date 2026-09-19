"""Tests for native agent web search discovery tool and Brave provider (#366)."""

from __future__ import annotations

import io
import json
import urllib.error
from unittest.mock import MagicMock, patch

from src.agent.search import (
    BRAVE_SEARCH_ENDPOINT,
    MAX_SNIPPET_CHARS,
    BraveSearchClient,
    distill_search_response,
    web_search,
)
from src.data.tool_sources import WEB_SEARCH_TOOL, validate_call_against_tools

# --------------------------------------------------------------------------- #
# Schema Definition & Validation
# --------------------------------------------------------------------------- #

def test_web_search_schema_structure():
    """Verify web_search schema definition in tool_sources.py."""
    assert WEB_SEARCH_TOOL["name"] == "web_search"
    assert "description" in WEB_SEARCH_TOOL
    params = WEB_SEARCH_TOOL["parameters"]
    assert params["type"] == "object"
    assert "query" in params["required"]
    assert "count" not in params["required"]

    props = params["properties"]
    assert props["query"]["type"] == "string"
    assert props["count"]["type"] == "integer"
    assert props["count"]["default"] == 5
    assert props["count"]["maximum"] == 10


def test_web_search_schema_validation():
    """Verify tool call validation against WEB_SEARCH_TOOL."""
    tools = [WEB_SEARCH_TOOL]

    # Valid call with only required query
    assert validate_call_against_tools(
        {"name": "web_search", "arguments": {"query": "python asyncio"}},
        tools,
    )

    # Valid call with optional count
    assert validate_call_against_tools(
        {"name": "web_search", "arguments": {"query": "mlx mamba hybrid", "count": 3}},
        tools,
    )

    # Invalid: missing required query parameter
    assert not validate_call_against_tools(
        {"name": "web_search", "arguments": {"count": 5}},
        tools,
    )

    # Invalid: wrong tool name
    assert not validate_call_against_tools(
        {"name": "unknown_tool", "arguments": {"query": "test"}},
        tools,
    )

    # Invalid: non-dict arguments
    assert not validate_call_against_tools(
        {"name": "web_search", "arguments": "invalid"},
        tools,
    )


# --------------------------------------------------------------------------- #
# Distilled Output Formatting & Token Optimization
# --------------------------------------------------------------------------- #

def _make_mock_brave_payload():
    return {
        "query": {"original": "python asyncio tutorial"},
        "web": {
            "results": [
                {
                    "title": "Async IO in <b>Python</b>: A Complete Walkthrough",
                    "url": "https://example.com/async-io",
                    "description": "An in-depth &amp; practical look at the <b>asyncio</b> library.",
                    "age": "2 years ago",
                    "page_age": "2024-01-01",
                    "family_friendly": True,
                    "extra_snippets": ["extra context that should be stripped"],
                },
                {
                    "title": "Python Documentation: asyncio",
                    "url": "https://docs.python.org/3/library/asyncio.html",
                    "description": "asyncio is a library to write concurrent code using the async/await syntax.",
                    "thumbnail": {"src": "https://example.com/thumb.png"},
                    "profile": {"name": "Python Docs"},
                },
                {
                    "title": "Long Snippet Entry",
                    "url": "https://example.com/long",
                    "description": "Word " * 100,  # exceeds MAX_SNIPPET_CHARS
                },
            ]
        },
    }


def test_distill_search_response_strict_keys():
    """Distilled records strictly contain ONLY title, url, and snippet."""
    payload = _make_mock_brave_payload()
    records = distill_search_response(payload, count=5)

    assert len(records) == 3
    for r in records:
        assert set(r.keys()) == {"title", "url", "snippet"}
        assert "age" not in r
        assert "page_age" not in r
        assert "family_friendly" not in r
        assert "extra_snippets" not in r
        assert "thumbnail" not in r
        assert "profile" not in r


def test_distill_search_response_html_stripping_and_unescaping():
    """HTML tags are stripped and entities unescaped in title and snippet."""
    payload = _make_mock_brave_payload()
    records = distill_search_response(payload, count=5)

    first = records[0]
    assert first["title"] == "Async IO in Python: A Complete Walkthrough"
    assert "<b>" not in first["title"]
    assert first["snippet"] == "An in-depth & practical look at the asyncio library."
    assert "&amp;" not in first["snippet"]
    assert "<b>" not in first["snippet"]


def test_distill_search_response_snippet_truncation():
    """Overly long snippets are truncated with trailing ellipsis."""
    payload = _make_mock_brave_payload()
    records = distill_search_response(payload, count=5)

    long_entry = records[2]
    assert len(long_entry["snippet"]) <= MAX_SNIPPET_CHARS
    assert long_entry["snippet"].endswith("...")


def test_distill_search_response_token_budget():
    """Distilled payload for 5 results comfortably fits under 400 tokens."""
    # 5 representative results
    items = []
    for i in range(5):
        items.append({
            "title": f"Technical Document {i}: Deep Architecture Overview",
            "url": f"https://example.org/docs/arch-{i}",
            "description": f"Detailed documentation describing component {i} and its interface specifications.",
        })
    payload = {"web": {"results": items}}
    records = distill_search_response(payload, count=5)
    json_str = json.dumps(records)

    # 1 token ~= 4 chars rule of thumb; target payload < 400 tokens (< ~1600 chars)
    assert len(json_str) < 1200
    # Word count estimation: ~1.3 tokens per word
    total_words = sum(len(r["title"].split()) + len(r["snippet"].split()) for r in records)
    assert total_words < 250


def test_distill_search_response_edge_cases():
    """Handles malformed or empty payloads gracefully."""
    assert distill_search_response(None) == []
    assert distill_search_response({}) == []
    assert distill_search_response({"web": {}}) == []
    assert distill_search_response({"web": {"results": "not_a_list"}}) == []
    assert distill_search_response({"web": {"results": []}}) == []
    assert distill_search_response("raw_string") == []


# --------------------------------------------------------------------------- #
# BraveSearchClient & Mock Network Calls
# --------------------------------------------------------------------------- #

def test_missing_api_key_handling(monkeypatch):
    """Missing BRAVE_SEARCH_API_KEY returns descriptive error without exceptions."""
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    client = BraveSearchClient(api_key=None)

    result = client.search("python")
    assert not result.is_success
    assert "BRAVE_SEARCH_API_KEY is not set" in result.error
    assert len(result.records) == 0
    assert "Error:" in str(result)


def test_empty_query_handling():
    """Empty or whitespace-only query returns descriptive error without network call."""
    client = BraveSearchClient(api_key="test-key")
    with patch("urllib.request.urlopen") as mock_urlopen:
        res1 = client.search("")
        res2 = client.search("   ")
        mock_urlopen.assert_not_called()

    assert not res1.is_success
    assert "Search query cannot be empty" in res1.error
    assert not res2.is_success


def test_successful_search_request():
    """Successful search executes HTTP GET with auth header and distill results."""
    mock_payload = _make_mock_brave_payload()
    json_bytes = json.dumps(mock_payload).encode("utf-8")

    mock_response = MagicMock()
    mock_response.read.return_value = json_bytes
    mock_response.__enter__.return_value = mock_response

    client = BraveSearchClient(api_key="test-secret-key")

    with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
        result = client.search("python asyncio", count=2)

        assert mock_urlopen.call_count == 1
        req = mock_urlopen.call_args[0][0]
        assert req.get_header("X-subscription-token") == "test-secret-key"
        assert req.get_header("Accept") == "application/json"
        assert req.full_url.startswith(BRAVE_SEARCH_ENDPOINT)
        assert "q=python+asyncio" in req.full_url
        assert "count=2" in req.full_url

    assert result.is_success
    assert len(result) == 2
    assert result[0]["title"] == "Async IO in Python: A Complete Walkthrough"
    assert result[0]["url"] == "https://example.com/async-io"
    assert not result.cached


def test_in_turn_caching_and_deduplication():
    """Identical queries within a session are served from cache without network calls."""
    mock_payload = _make_mock_brave_payload()
    json_bytes = json.dumps(mock_payload).encode("utf-8")

    mock_response = MagicMock()
    mock_response.read.return_value = json_bytes
    mock_response.__enter__.return_value = mock_response

    client = BraveSearchClient(api_key="test-key")

    with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
        # First query
        res1 = client.search("Python AsyncIO", count=3)
        assert mock_urlopen.call_count == 1
        assert not res1.cached

        # Repeated identical query (normalized case)
        res2 = client.search("python asyncio", count=3)
        assert mock_urlopen.call_count == 1  # No additional network call
        assert res2.cached
        assert len(res2) == len(res1)
        assert res2[0]["title"] == res1[0]["title"]

        # Different query executes network call
        res3 = client.search("python multiprocessing", count=3)
        assert mock_urlopen.call_count == 2
        assert not res3.cached

        # Clear cache and retry first query
        client.clear_cache()
        assert client.cache_size == 0
        res4 = client.search("python asyncio", count=3)
        assert mock_urlopen.call_count == 3
        assert not res4.cached


def test_timeout_abort_handling():
    """Request timeout aborts gracefully and returns descriptive error string."""
    client = BraveSearchClient(api_key="test-key", timeout_s=5.0)

    with patch("urllib.request.urlopen", side_effect=TimeoutError("The read operation timed out")):
        result = client.search("slow query")

    assert not result.is_success
    assert "timed out after 5.0s" in result.error
    assert len(result) == 0
    assert "Error:" in str(result)


def test_url_error_timeout_handling():
    """URLError wrapping a timeout also returns descriptive error string."""
    client = BraveSearchClient(api_key="test-key", timeout_s=5.0)

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timed out")):
        result = client.search("slow query")

    assert not result.is_success
    assert "timed out after 5.0s" in result.error


def test_http_401_unauthorized_handling():
    """HTTP 401 returns descriptive authentication error."""
    client = BraveSearchClient(api_key="bad-key")
    http_err = urllib.error.HTTPError(
        url="https://api.search.brave.com/res/v1/web/search",
        code=401,
        msg="Unauthorized",
        hdrs={},
        fp=io.BytesIO(b""),
    )

    with patch("urllib.request.urlopen", side_effect=http_err):
        result = client.search("query")

    assert not result.is_success
    assert "401" in result.error
    assert "authentication failed" in result.error


def test_http_429_rate_limit_handling():
    """HTTP 429 returns descriptive rate limit error."""
    client = BraveSearchClient(api_key="test-key")
    http_err = urllib.error.HTTPError(
        url="https://api.search.brave.com/res/v1/web/search",
        code=429,
        msg="Too Many Requests",
        hdrs={},
        fp=io.BytesIO(b""),
    )

    with patch("urllib.request.urlopen", side_effect=http_err):
        result = client.search("query")

    assert not result.is_success
    assert "429" in result.error
    assert "rate limit exceeded" in result.error


def test_malformed_json_response_handling():
    """Non-JSON response bytes return descriptive decode error."""
    client = BraveSearchClient(api_key="test-key")
    mock_response = MagicMock()
    mock_response.read.return_value = b"<!DOCTYPE html><html>502 Bad Gateway</html>"
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        result = client.search("query")

    assert not result.is_success
    assert "Failed to decode Brave search API response as JSON" in result.error


def test_count_clamping():
    """Count argument is clamped between 1 and MAX_COUNT (10)."""
    mock_payload = _make_mock_brave_payload()
    json_bytes = json.dumps(mock_payload).encode("utf-8")
    mock_response = MagicMock()
    mock_response.read.return_value = json_bytes
    mock_response.__enter__.return_value = mock_response

    client = BraveSearchClient(api_key="test-key")

    with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
        # Zero or negative clamped to 1
        client.search("test", count=0)
        req = mock_urlopen.call_args[0][0]
        assert "count=1" in req.full_url

        # Greater than 10 clamped to 10
        client.search("test2", count=25)
        req = mock_urlopen.call_args[0][0]
        assert "count=10" in req.full_url


# --------------------------------------------------------------------------- #
# Tool Wrapper (web_search)
# --------------------------------------------------------------------------- #

def test_web_search_helper_success():
    """web_search helper returns formatted JSON string on success."""
    mock_payload = _make_mock_brave_payload()
    json_bytes = json.dumps(mock_payload).encode("utf-8")
    mock_response = MagicMock()
    mock_response.read.return_value = json_bytes
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        output = web_search("python", count=1, api_key="test-key")

    assert isinstance(output, str)
    parsed = json.loads(output)
    assert isinstance(parsed, list)
    assert len(parsed) == 1
    assert parsed[0]["title"] == "Async IO in Python: A Complete Walkthrough"


def test_web_search_helper_error():
    """web_search helper returns descriptive error string on failure."""
    output = web_search("python", api_key="")
    assert isinstance(output, str)
    assert output.startswith("Error:")
    assert "BRAVE_SEARCH_API_KEY is not set" in output
