"""Tests for native agent page extraction tool, SSRF ingress guards, and truncation (#367)."""

from __future__ import annotations

import io
import ipaddress
import socket
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from src.agent.fetcher import (
    DEFAULT_MAX_CHARS,
    DEFAULT_TIMEOUT_S,
    JINA_READER_BASE_URL,
    FetchResult,
    HTMLToMarkdownParser,
    PageFetcherClient,
    SSRFSafeRedirectHandler,
    extract_markdown_from_html,
    fetch_web_page,
    is_disallowed_ip,
    truncate_content,
    validate_url_for_ssrf,
)
from src.data.tool_sources import FETCH_WEB_PAGE_TOOL, validate_call_against_tools

# --------------------------------------------------------------------------- #
# Tool Schema Definition & Validation
# --------------------------------------------------------------------------- #

def test_fetch_web_page_schema_structure():
    """Verify fetch_web_page schema definition in tool_sources.py."""
    assert FETCH_WEB_PAGE_TOOL["name"] == "fetch_web_page"
    assert "description" in FETCH_WEB_PAGE_TOOL
    params = FETCH_WEB_PAGE_TOOL["parameters"]
    assert params["type"] == "object"
    assert "url" in params["required"]

    props = params["properties"]
    assert "url" in props
    assert props["url"]["type"] == "string"


def test_fetch_web_page_schema_validation():
    """Verify tool call validation against FETCH_WEB_PAGE_TOOL."""
    tools = [FETCH_WEB_PAGE_TOOL]

    # Valid call with required url
    assert validate_call_against_tools(
        {"name": "fetch_web_page", "arguments": {"url": "https://docs.python.org/3/"}},
        tools,
    )

    # Invalid: missing required url parameter
    assert not validate_call_against_tools(
        {"name": "fetch_web_page", "arguments": {}},
        tools,
    )

    # Invalid: wrong tool name
    assert not validate_call_against_tools(
        {"name": "unknown_tool", "arguments": {"url": "https://example.com"}},
        tools,
    )

    # Invalid: non-dict arguments
    assert not validate_call_against_tools(
        {"name": "fetch_web_page", "arguments": "https://example.com"},
        tools,
    )


# --------------------------------------------------------------------------- #
# SSRF Ingress Defense
# --------------------------------------------------------------------------- #

def test_is_disallowed_ip_helper():
    """Verify ip address range classifier identifies loopback, private, and reserved IPs."""
    assert is_disallowed_ip(ipaddress.ip_address("127.0.0.1"))
    assert is_disallowed_ip(ipaddress.ip_address("10.0.0.1"))
    assert is_disallowed_ip(ipaddress.ip_address("172.16.0.1"))
    assert is_disallowed_ip(ipaddress.ip_address("192.168.1.1"))
    assert is_disallowed_ip(ipaddress.ip_address("169.254.169.254"))
    assert is_disallowed_ip(ipaddress.ip_address("::1"))
    assert is_disallowed_ip(ipaddress.ip_address("fe80::1"))
    assert is_disallowed_ip(ipaddress.ip_address("fc00::1"))
    assert is_disallowed_ip(ipaddress.ip_address("::ffff:127.0.0.1"))

    # Public IPs are allowed
    assert not is_disallowed_ip(ipaddress.ip_address("8.8.8.8"))
    assert not is_disallowed_ip(ipaddress.ip_address("93.184.216.34"))


@pytest.mark.parametrize(
    "blocked_url",
    [
        # Localhost and loopbacks
        "http://localhost",
        "http://localhost:8080/admin",
        "https://localhost/status",
        "http://foo.localhost/resource",
        "http://service.local/api",
        "http://app.internal/config",
        "http://127.0.0.1",
        "http://127.0.0.1:8000/keys",
        "http://127.0.0.2:9000",
        "http://127.255.255.254",
        # RFC 1918 Private Ranges
        "http://10.0.0.1",
        "http://10.255.255.255/secret",
        "http://172.16.0.1/private",
        "http://172.31.255.254",
        "http://192.168.0.1",
        "http://192.168.1.254/router",
        # Cloud metadata
        "http://169.254.169.254/latest/meta-data/",
        "http://169.254.1.1",
        # IPv6 loopback and private/link-local
        "http://[::1]/",
        "http://[::1]:8080",
        "http://[fe80::1]/",
        "http://[fc00::1]/",
        "http://[fd00::1]/",
        # IPv4-mapped IPv6 loopback / private
        "http://[::ffff:127.0.0.1]/",
        "http://[::ffff:10.0.0.1]/",
        "http://[::ffff:169.254.169.254]/",
        # Non-HTTP schemes
        "file:///etc/passwd",
        "ftp://example.com/file.txt",
        "gopher://example.com",
    ],
)
def test_ssrf_blocks_disallowed_destinations(blocked_url):
    """SSRF guard rejects loopback, RFC 1918, link-local, cloud metadata, and invalid schemes."""
    is_safe, reason = validate_url_for_ssrf(blocked_url)
    assert not is_safe
    assert reason is not None

    # Verify client returns 403 SSRF blocked error without socket connection
    client = PageFetcherClient()
    with patch("urllib.request.urlopen") as mock_urlopen:
        res = client.fetch(blocked_url)
        mock_urlopen.assert_not_called()

    assert not res.is_success
    assert res.status_code == 403
    assert "SSRF blocked" in res.error or "Unsupported URL scheme" in res.error


def test_ssrf_blocks_dns_rebinding_to_private_ip():
    """SSRF guard blocks hostnames that resolve to private or loopback IP addresses."""
    # Simulate a domain that resolves to 10.0.0.1
    mock_addrinfo = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 0))]

    with patch("socket.getaddrinfo", return_value=mock_addrinfo):
        is_safe, reason = validate_url_for_ssrf("https://attacker-rebind.com/internal")

    assert not is_safe
    assert "resolved to restricted IP 10.0.0.1" in reason


def test_ssrf_allows_public_addresses():
    """SSRF guard allows public internet IP addresses and resolved hostnames."""
    # Public IP literal
    is_safe_ip, reason_ip = validate_url_for_ssrf("https://93.184.216.34/index.html")
    assert is_safe_ip
    assert reason_ip is None

    # Public hostname resolving to public IP
    mock_addrinfo = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]
    with patch("socket.getaddrinfo", return_value=mock_addrinfo):
        is_safe, reason = validate_url_for_ssrf("https://example.com/docs")

    assert is_safe
    assert reason is None


def test_ssrf_blocks_redirect_to_private_ip():
    """SSRFSafeRedirectHandler blocks redirects targeting private addresses."""
    handler = SSRFSafeRedirectHandler()
    req = urllib.request.Request("https://example.com/redirect")

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        handler.redirect_request(
            req=req,
            fp=None,
            code=302,
            msg="Found",
            headers={},
            newurl="http://127.0.0.1:8000/secret",
        )

    assert exc_info.value.code == 403
    assert "SSRF blocked" in exc_info.value.reason


# --------------------------------------------------------------------------- #
# Clean Markdown Extraction & Jina Reader Integration
# --------------------------------------------------------------------------- #

def test_jina_reader_fetch_success():
    """Successful fetch queries Jina Reader proxy with required headers and returns markdown."""
    target_url = "https://example.com/tech-article"
    markdown_content = "# Tech Article\n\nThis is distilled documentation content."

    mock_resp = MagicMock()
    mock_resp.read.return_value = markdown_content.encode("utf-8")
    mock_resp.status = 200
    mock_resp.__enter__.return_value = mock_resp

    client = PageFetcherClient(api_key="jina_test_key_123")

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen, \
         patch("src.agent.fetcher.validate_url_for_ssrf", return_value=(True, None)):
        result = client.fetch(target_url)

        assert mock_urlopen.call_count == 1
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == f"{JINA_READER_BASE_URL}/{target_url}"
        assert req.get_header("Accept") == "text/markdown"
        assert req.get_header("Authorization") == "Bearer jina_test_key_123"

    assert result.is_success
    assert result.source == "jina"
    assert result.status_code == 200
    assert not result.truncated
    assert result.content == markdown_content
    assert str(result) == markdown_content


def test_html_parser_components():
    """Verify HTMLToMarkdownParser handles headings, links, lists, code, and ignores scripts."""
    html_markup = """
    <html>
      <head><title>Title</title></head>
      <body>
        <h1>Heading 1</h1>
        <h2>Heading 2</h2>
        <p>Text with <a href="https://example.com">Example</a> and <code>var x = 1;</code>.</p>
        <pre><code>line 1\nline 2</code></pre>
        <ul>
          <li>Point A</li>
          <li>Point B</li>
        </ul>
        <script>var secret = 42;</script>
        <style>body { margin: 0; }</style>
      </body>
    </html>
    """
    md = extract_markdown_from_html(html_markup)
    assert "# Heading 1" in md
    assert "## Heading 2" in md
    assert "[Example](https://example.com)" in md
    assert "`var x = 1;`" in md
    assert "```" in md
    assert "- Point A" in md
    assert "secret" not in md

    # Direct parser instantiation test
    parser = HTMLToMarkdownParser()
    parser.feed("<p>Hello</p>")
    parser.close()
    assert "".join(parser.pieces).strip() == "Hello"


def test_fallback_parser_when_jina_proxy_fails():
    """Fallback parser engages automatically when Jina Reader returns an error."""
    target_url = "https://example.com/api-docs"
    html_page = """<!DOCTYPE html>
    <html>
      <head><title>Docs</title><style>body { color: black; }</style></head>
      <body>
        <h1>API Documentation</h1>
        <p>Welcome to the <code>API</code> reference with <a href="https://example.com/link">link</a>.</p>
        <script>console.log("tracker");</script>
        <ul>
          <li>Endpoint A</li>
          <li>Endpoint B</li>
        </ul>
      </body>
    </html>"""

    # Jina Reader returns 502 Bad Gateway
    jina_err = urllib.error.HTTPError(
        url="https://r.jina.ai/test",
        code=502,
        msg="Bad Gateway",
        hdrs={},
        fp=io.BytesIO(b""),
    )

    mock_direct_resp = MagicMock()
    mock_direct_resp.read.return_value = html_page.encode("utf-8")
    mock_direct_resp.status = 200
    mock_direct_resp.headers = {"Content-Type": "text/html; charset=utf-8"}
    mock_direct_resp.__enter__.return_value = mock_direct_resp

    client = PageFetcherClient(fallback_on_error=True)

    with patch("urllib.request.urlopen", side_effect=jina_err), \
         patch.object(client._opener, "open", return_value=mock_direct_resp), \
         patch("src.agent.fetcher.validate_url_for_ssrf", return_value=(True, None)):
        result = client.fetch(target_url)

    assert result.is_success
    assert result.source == "fallback"
    assert "# API Documentation" in result.content
    assert "[link](https://example.com/link)" in result.content
    assert "`API`" in result.content
    assert "- Endpoint A" in result.content
    assert "tracker" not in result.content  # Script content stripped


def test_direct_fallback_mode_skips_jina():
    """When use_jina=False, fetch directly uses local html parser."""
    target_url = "https://example.com/direct"
    html_page = "<h1>Offline Page</h1><p>Parsed directly without proxy.</p>"

    mock_direct_resp = MagicMock()
    mock_direct_resp.read.return_value = html_page.encode("utf-8")
    mock_direct_resp.status = 200
    mock_direct_resp.headers = {"Content-Type": "text/html"}
    mock_direct_resp.__enter__.return_value = mock_direct_resp

    client = PageFetcherClient(use_jina=False)

    with patch("urllib.request.urlopen") as mock_jina_urlopen, \
         patch.object(client._opener, "open", return_value=mock_direct_resp), \
         patch("src.agent.fetcher.validate_url_for_ssrf", return_value=(True, None)):
        result = client.fetch(target_url)
        mock_jina_urlopen.assert_not_called()

    assert result.is_success
    assert result.source == "fallback"
    assert "# Offline Page" in result.content
    assert "Parsed directly without proxy." in result.content


# --------------------------------------------------------------------------- #
# Token Ceiling & Truncation Guardrail
# --------------------------------------------------------------------------- #

def test_truncation_ceiling_attached_cleanly():
    """Content exceeding ceiling is truncated at max_chars with clean notice attached."""
    # Under limit: content untouched
    short_text = "A" * 500
    truncated, was_trunc = truncate_content(short_text, max_chars=12000)
    assert not was_trunc
    assert truncated == short_text
    assert "[Content truncated" not in truncated

    # Exactly at limit: untouched
    exact_text = "B" * 12000
    truncated, was_trunc = truncate_content(exact_text, max_chars=12000)
    assert not was_trunc
    assert truncated == exact_text

    # Over limit: cleanly truncated with explicit notice
    long_text = "C" * 15000
    truncated, was_trunc = truncate_content(long_text, max_chars=12000)
    assert was_trunc
    assert truncated.startswith("C" * 12000)
    assert "[Content truncated at 12000 chars to preserve context window...]" in truncated


def test_fetch_enforces_truncation_ceiling():
    """PageFetcherClient truncates oversized markdown payloads and sets truncated flag."""
    target_url = "https://example.com/massive-document"
    oversized_markdown = "Word " * 5000  # 25,000 characters

    mock_resp = MagicMock()
    mock_resp.read.return_value = oversized_markdown.encode("utf-8")
    mock_resp.status = 200
    mock_resp.__enter__.return_value = mock_resp

    client = PageFetcherClient(max_chars=DEFAULT_MAX_CHARS)

    with patch("urllib.request.urlopen", return_value=mock_resp), \
         patch("src.agent.fetcher.validate_url_for_ssrf", return_value=(True, None)):
        result = client.fetch(target_url)

    assert result.is_success
    assert result.truncated
    assert "[Content truncated at 12000 chars to preserve context window...]" in result.content
    # Content before notice matches truncated character limit
    content_body = result.content.split("\n\n[Content truncated")[0]
    assert len(content_body) <= DEFAULT_MAX_CHARS


# --------------------------------------------------------------------------- #
# Network Timeout & Error Handling
# --------------------------------------------------------------------------- #

def test_timeout_handling_returns_descriptive_error():
    """Network timeout enforces 8-second hard timeout and returns descriptive error."""
    client = PageFetcherClient(timeout_s=DEFAULT_TIMEOUT_S)
    target_url = "https://example.com/slow-endpoint"

    with patch("urllib.request.urlopen", side_effect=TimeoutError("The read operation timed out")), \
         patch("src.agent.fetcher.validate_url_for_ssrf", return_value=(True, None)):
        result = client.fetch(target_url)

    assert not result.is_success
    assert f"timed out after {DEFAULT_TIMEOUT_S}s" in result.error
    assert str(result).startswith("Error:")
    assert f"timed out after {DEFAULT_TIMEOUT_S}s" in str(result)


def test_url_error_timeout_handling():
    """URLError wrapping a timeout also returns descriptive error message."""
    client = PageFetcherClient(timeout_s=DEFAULT_TIMEOUT_S)
    target_url = "https://example.com/timeout-endpoint"

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timed out")), \
         patch("src.agent.fetcher.validate_url_for_ssrf", return_value=(True, None)):
        result = client.fetch(target_url)

    assert not result.is_success
    assert f"timed out after {DEFAULT_TIMEOUT_S}s" in result.error


def test_in_turn_caching_and_deduplication():
    """Identical URLs are served from cache within the session turn."""
    target_url = "https://example.com/cached-article"
    content = "# Cached Article\n\nContent here."

    mock_resp = MagicMock()
    mock_resp.read.return_value = content.encode("utf-8")
    mock_resp.status = 200
    mock_resp.__enter__.return_value = mock_resp

    client = PageFetcherClient()

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen, \
         patch("src.agent.fetcher.validate_url_for_ssrf", return_value=(True, None)):
        res1 = client.fetch(target_url)
        assert mock_urlopen.call_count == 1
        assert not res1.cached

        # Repeated fetch hits cache
        res2 = client.fetch(target_url)
        assert mock_urlopen.call_count == 1
        assert res2.cached
        assert res2.content == res1.content

        # Clearing cache triggers re-fetch
        client.clear_cache()
        assert client.cache_size == 0
        res3 = client.fetch(target_url)
        assert mock_urlopen.call_count == 2
        assert not res3.cached


# --------------------------------------------------------------------------- #
# Tool Wrapper (fetch_web_page) & FetchResult
# --------------------------------------------------------------------------- #

def test_fetch_result_properties():
    """Verify FetchResult bool, len, and string representation semantics."""
    res_success = FetchResult(url="https://example.com", content="hello world")
    assert bool(res_success)
    assert len(res_success) == 11
    assert str(res_success) == "hello world"

    res_err = FetchResult(url="https://example.com", content="", error="Access denied")
    assert not bool(res_err)
    assert not res_err.is_success
    assert str(res_err) == "Error: Access denied"


def test_fetch_web_page_tool_wrapper_success():
    """fetch_web_page returns extracted markdown string directly for tool calls."""
    target_url = "https://example.com/page"
    content = "# Page Title\n\nPage text."

    mock_resp = MagicMock()
    mock_resp.read.return_value = content.encode("utf-8")
    mock_resp.status = 200
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp), \
         patch("src.agent.fetcher.validate_url_for_ssrf", return_value=(True, None)):
        out = fetch_web_page(target_url)

    assert isinstance(out, str)
    assert out == content


def test_fetch_web_page_tool_wrapper_error():
    """fetch_web_page returns formatted error string on failure."""
    out = fetch_web_page("http://127.0.0.1:8080/internal")
    assert isinstance(out, str)
    assert out.startswith("Error:")
    assert "SSRF blocked" in out


def test_empty_url_handling():
    """Empty or whitespace-only URL returns descriptive error without network calls."""
    client = PageFetcherClient()
    res = client.fetch("")
    assert not res.is_success
    assert "URL cannot be empty" in res.error
