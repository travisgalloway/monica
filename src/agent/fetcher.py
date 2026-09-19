"""Native web page extraction tool with SSRF guards and markdown truncation (#367).

Coding agents require targeted page extraction to read documentation and API
references surfaced during web search discovery. This module implements the second stage
of the two-stage web architecture (web_search discovery -> fetch_web_page extraction).

ABOVE THE SEAM — pure Python standard library only.
No hardware backend or third-party HTTP dependencies.
"""

from __future__ import annotations

import html
import ipaddress
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any

JINA_READER_BASE_URL = "https://r.jina.ai"
DEFAULT_TIMEOUT_S = 8.0
DEFAULT_MAX_CHARS = 12000
MAX_CONTENT_CHARS = 12000

DISALLOWED_HOSTNAME_SUFFIXES = (
    ".localhost",
    ".local",
    ".internal",
    ".arpa",
    ".corp",
    ".lan",
    ".home",
)


def is_disallowed_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Determine whether an IP address belongs to a private, loopback, or reserved range."""
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None and is_disallowed_ip(mapped):
        return True

    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def validate_url_for_ssrf(url: str) -> tuple[bool, str | None]:
    """Validate URL target against SSRF attacks before opening network connections.

    Rejects non-HTTP schemes, loopback addresses, RFC 1918 private subnets,
    link-local addresses, cloud metadata endpoints, and reserved hostnames.
    """
    if not url or not isinstance(url, str):
        return False, "URL must be a non-empty string."

    stripped = url.strip()
    try:
        parsed = urllib.parse.urlsplit(stripped)
    except ValueError as e:
        return False, f"Malformed URL: {e}"

    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        return False, f"Unsupported URL scheme {scheme!r}. Only http and https are allowed."

    hostname = parsed.hostname
    if not hostname:
        return False, "URL must contain a valid hostname."

    norm_host = hostname.lower().strip(".")
    if norm_host == "localhost" or norm_host.endswith(DISALLOWED_HOSTNAME_SUFFIXES):
        return False, f"SSRF blocked: Hostname {hostname!r} is a reserved local or internal address."

    # Check IP literal directly if provided
    ip_literal = norm_host.strip("[]")
    try:
        ip = ipaddress.ip_address(ip_literal)
        if is_disallowed_ip(ip):
            return False, f"SSRF blocked: IP {ip} is in a restricted range (loopback, private, or link-local)."
        return True, None
    except ValueError:
        pass

    # Resolve hostname to verify destination IP addresses
    try:
        addr_infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as e:
        return False, f"Could not resolve hostname {hostname!r}: {e}"
    except OSError as e:
        return False, f"DNS resolution error for {hostname!r}: {e}"

    if not addr_infos:
        return False, f"Could not resolve any IP addresses for hostname {hostname!r}."

    for addr_info in addr_infos:
        sockaddr = addr_info[4]
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
            if is_disallowed_ip(ip):
                return False, f"SSRF blocked: Hostname {hostname!r} resolved to restricted IP {ip}.\""
        except ValueError:
            return False, f"Resolved invalid IP address {ip_str!r} for hostname {hostname!r}."

    return True, None


class SSRFSafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Redirect handler that validates destination URLs against SSRF rules."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        is_safe, reason = validate_url_for_ssrf(newurl)
        if not is_safe:
            raise urllib.error.HTTPError(
                newurl,
                403,
                f"SSRF blocked on redirect: {reason}",
                headers,
                fp,
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class HTMLToMarkdownParser(HTMLParser):
    """Extract readable markdown text from HTML markup using only standard library."""

    def __init__(self) -> None:
        super().__init__()
        self.pieces: list[str] = []
        self.ignore_tags = {"script", "style", "noscript", "head", "svg", "iframe"}
        self.ignore_depth = 0
        self.current_href: str | None = None
        self.link_text: list[str] = []
        self.in_pre = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self.ignore_tags:
            self.ignore_depth += 1
            return
        if self.ignore_depth > 0:
            return

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(tag[1])
            self.pieces.append("\n\n" + "#" * level + " ")
        elif tag in ("p", "div", "article", "section", "blockquote"):
            self.pieces.append("\n\n")
        elif tag == "br":
            self.pieces.append("\n")
        elif tag == "li":
            self.pieces.append("\n- ")
        elif tag == "hr":
            self.pieces.append("\n\n---\n\n")
        elif tag == "a":
            attr_dict = dict(attrs)
            self.current_href = attr_dict.get("href")
            self.link_text = []
        elif tag == "pre":
            self.in_pre = True
            self.pieces.append("\n\n```\n")
        elif tag == "code" and not self.in_pre:
            self.pieces.append("`")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.ignore_tags:
            self.ignore_depth = max(0, self.ignore_depth - 1)
            return
        if self.ignore_depth > 0:
            return

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6", "p", "div", "article", "section", "blockquote"):
            self.pieces.append("\n")
        elif tag == "a":
            if self.current_href:
                text = "".join(self.link_text).strip()
                href = self.current_href.strip()
                if text and href:
                    self.pieces.append(f"[{text}]({href})")
                elif text:
                    self.pieces.append(text)
                elif href and href.startswith(("http://", "https://")):
                    self.pieces.append(f"<{href}>")
            self.current_href = None
            self.link_text = []
        elif tag == "pre":
            self.in_pre = False
            self.pieces.append("\n```\n\n")
        elif tag == "code" and not self.in_pre:
            self.pieces.append("`")

    def handle_data(self, data: str) -> None:
        if self.ignore_depth > 0:
            return
        if self.current_href is not None:
            self.link_text.append(data)
        else:
            self.pieces.append(data)


def extract_markdown_from_html(html_text: str) -> str:
    """Convert HTML string to clean markdown text using HTMLToMarkdownParser."""
    if not html_text:
        return ""
    parser = HTMLToMarkdownParser()
    try:
        parser.feed(html_text)
        parser.close()
    except (ValueError, RuntimeError):
        # Fallback to simple regex tag stripping if parsing fails
        cleaned = re.sub(r"<[^>]+>", " ", html_text)
        return html.unescape(" ".join(cleaned.split()))

    raw = "".join(parser.pieces)
    # Unescape HTML entities
    unescaped = html.unescape(raw)
    # Normalize excessive newlines and whitespace
    lines = [line.rstrip() for line in unescaped.splitlines()]
    compact = "\n".join(lines)
    compact = re.sub(r"\n{3,}", "\n\n", compact)
    return compact.strip()


def truncate_content(content: str, max_chars: int = DEFAULT_MAX_CHARS) -> tuple[str, bool]:
    """Enforce character ceiling and attach clean truncation notice if exceeded."""
    if len(content) <= max_chars:
        return content, False
    truncated = content[:max_chars].rstrip()
    notice = f"\n\n[Content truncated at {max_chars} chars to preserve context window...]"
    return truncated + notice, True


@dataclass
class FetchResult:
    """Structured webpage extraction result supporting inspection and string conversion."""

    url: str
    content: str
    error: str | None = None
    truncated: bool = False
    source: str = ""
    status_code: int | None = None
    cached: bool = False

    @property
    def is_success(self) -> bool:
        """True if page content was fetched without error."""
        return self.error is None

    def __str__(self) -> str:
        """Representation formatted for direct LLM and tool harness consumption."""
        if self.error is not None:
            if self.error.startswith("Error:"):
                return self.error
            return f"Error: {self.error}"
        return self.content

    def __bool__(self) -> bool:
        return self.is_success and len(self.content) > 0

    def __len__(self) -> int:
        return len(self.content)


class PageFetcherClient:
    """Web page extraction client with SSRF guards, Jina Reader proxy, and offline fallback."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_chars: int = DEFAULT_MAX_CHARS,
        jina_base_url: str = JINA_READER_BASE_URL,
        use_jina: bool = True,
        fallback_on_error: bool = True,
    ) -> None:
        raw_key = api_key if api_key is not None else os.environ.get("JINA_API_KEY", "")
        self.api_key = raw_key.strip()
        self.timeout_s = float(timeout_s)
        self.max_chars = int(max_chars)
        self.jina_base_url = jina_base_url.rstrip("/")
        self.use_jina = bool(use_jina)
        self.fallback_on_error = bool(fallback_on_error)
        self._cache: dict[str, FetchResult] = {}
        self._opener = urllib.request.build_opener(SSRFSafeRedirectHandler)

    def clear_cache(self) -> None:
        """Clear in-turn page cache for the active session."""
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        """Number of cached page extraction results."""
        return len(self._cache)

    def _fetch_direct(self, url: str) -> tuple[str, int | None, str | None]:
        """Perform direct HTTP fetch using SSRF-safe opener."""
        headers = {
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
            "User-Agent": "monica-agent/1.0",
        }
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with self._opener.open(req, timeout=self.timeout_s) as response:
                status_code = getattr(response, "status", 200)
                raw_bytes = response.read()
                # Determine charset if provided
                charset = "utf-8"
                content_type = response.headers.get("Content-Type", "")
                if "charset=" in content_type:
                    charset = content_type.split("charset=")[-1].split(";")[0].strip()
                try:
                    text = raw_bytes.decode(charset, errors="replace")
                except (LookupError, UnicodeDecodeError):
                    text = raw_bytes.decode("utf-8", errors="replace")
                return text, status_code, None
        except urllib.error.HTTPError as e:
            return "", e.code, f"HTTP {e.code}: {e.reason}"
        except TimeoutError:
            return "", None, f"Page fetch request timed out after {self.timeout_s}s for URL: {url!r}"
        except urllib.error.URLError as e:
            if "timed out" in str(e.reason).lower():
                return "", None, f"Page fetch request timed out after {self.timeout_s}s for URL: {url!r}"
            return "", None, f"Network error fetching URL: {e.reason}"
        except OSError as e:
            return "", None, f"Network OS error fetching URL: {e}"

    def _fetch_jina(self, url: str) -> tuple[str, int | None, str | None]:
        """Fetch markdown from Jina Reader proxy off-box."""
        jina_url = f"{self.jina_base_url}/{url}"
        headers = {
            "Accept": "text/markdown",
            "User-Agent": "monica-agent/1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(jina_url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
                status_code = getattr(response, "status", 200)
                raw_bytes = response.read()
                text = raw_bytes.decode("utf-8", errors="replace")
                return text, status_code, None
        except urllib.error.HTTPError as e:
            return "", e.code, f"Jina Reader HTTP {e.code}: {e.reason}"
        except TimeoutError:
            return "", None, f"Page fetch request timed out after {self.timeout_s}s for URL: {url!r}"
        except urllib.error.URLError as e:
            if "timed out" in str(e.reason).lower():
                return "", None, f"Page fetch request timed out after {self.timeout_s}s for URL: {url!r}"
            return "", None, f"Jina Reader network error: {e.reason}"
        except OSError as e:
            return "", None, f"Jina Reader OS error: {e}"

    def fetch(self, url: str) -> FetchResult:
        """Extract page markdown with SSRF verification, caching, and fallback handling."""
        if not url or not url.strip():
            return FetchResult(url="", content="", error="URL cannot be empty.")

        target_url = url.strip()
        cache_key = target_url
        if cache_key in self._cache:
            cached_res = self._cache[cache_key]
            return FetchResult(
                url=cached_res.url,
                content=cached_res.content,
                error=cached_res.error,
                truncated=cached_res.truncated,
                source=cached_res.source,
                status_code=cached_res.status_code,
                cached=True,
            )

        # SSRF validation before network connections
        is_safe, ssrf_error = validate_url_for_ssrf(target_url)
        if not is_safe:
            result = FetchResult(
                url=target_url,
                content="",
                error=ssrf_error,
                source="ssrf_guard",
                status_code=403,
            )
            return result

        # Attempt extraction via Jina Reader proxy if enabled
        if self.use_jina:
            raw_text, status_code, err = self._fetch_jina(target_url)
            if err is None and raw_text:
                truncated_text, is_truncated = truncate_content(raw_text, max_chars=self.max_chars)
                result = FetchResult(
                    url=target_url,
                    content=truncated_text,
                    error=None,
                    truncated=is_truncated,
                    source="jina",
                    status_code=status_code,
                    cached=False,
                )
                self._cache[cache_key] = result
                return result

            # If Jina timed out and fallback is disabled, return timeout immediately
            if not self.fallback_on_error or (err and "timed out" in err.lower()):
                result = FetchResult(
                    url=target_url,
                    content="",
                    error=err or "Failed to retrieve content from Jina Reader.",
                    source="jina",
                    status_code=status_code,
                )
                return result

        # Offline / in-process fallback using direct HTTP fetch and HTML parser
        raw_html, status_code, err = self._fetch_direct(target_url)
        if err is not None:
            result = FetchResult(
                url=target_url,
                content="",
                error=err,
                source="fallback",
                status_code=status_code,
            )
            return result

        markdown = extract_markdown_from_html(raw_html)
        if not markdown.strip() and raw_html.strip():
            # If html parser returned empty, use normalized raw text
            markdown = " ".join(raw_html.split())

        truncated_text, is_truncated = truncate_content(markdown, max_chars=self.max_chars)
        result = FetchResult(
            url=target_url,
            content=truncated_text,
            error=None,
            truncated=is_truncated,
            source="fallback",
            status_code=status_code,
            cached=False,
        )
        self._cache[cache_key] = result
        return result


def fetch_web_page(
    url: str,
    *,
    client: PageFetcherClient | None = None,
    api_key: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """Extract readable markdown content from a web page URL.

    Designed for direct registration as an agent harness tool handler.
    """
    c = client or PageFetcherClient(
        api_key=api_key,
        timeout_s=timeout_s,
        max_chars=max_chars,
    )
    res = c.fetch(url)
    return str(res)
