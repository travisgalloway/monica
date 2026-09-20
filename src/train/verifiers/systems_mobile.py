"""#341 -- Systems, Native & Mobile Stack Verifiers (Rust, C/C++, Swift, Kotlin).

Part of the M12 post-training and verifiable code reward track (#198, #230, #341).
Expands static compiler verifiers across systems, embedded, and mobile application
platforms WITHOUT running untrusted binaries:

1. Rust (Systems, WASM, CLI):
   - Toolchains: rustc/cargo check (--target wasm32-unknown-unknown), clippy
   - Checks: Borrow checker, lifetime bounds, Send/Sync invariants, no-std compatibility,
     trait implementations
   - Anti-Goodhart: Reject unsafe, .unwrap(), todo!(), unimplemented!(), #[allow(clippy::all)]

2. C / C++ (C17, C++20, C++23):
   - Toolchains: clang++ -fsyntax-only -std=c++20 -Wall -Werror, clang-tidy
   - Checks: Template instantiation, constexpr evaluation, memory ownership semantics,
     strict const-correctness
   - Anti-Goodhart: Reject reinterpret_cast, raw pointer ownership transfer, implicit narrowing

3. Swift (Apple Ecosystem: iOS, macOS, SwiftUI):
   - Toolchain: swiftc -parse -typecheck -strict-concurrency=complete
   - Checks: Swift 6 strict concurrency, Sendable conformance, SwiftUI View body type
     resolution, @Observable macros
   - Anti-Goodhart: Reject @unchecked Sendable, fatalError(), force unwrap !

4. Kotlin (Android, Jetpack Compose, Ktor):
   - Toolchain: kotlinc -Xno-optimize -Werror
   - Checks: Coroutine context dispatch, Nullable type system, Jetpack Compose @Composable
     invariants, sealed interface completeness
   - Anti-Goodhart: Reject @Suppress, !! force assertions, non-exhaustive when branches

Pure reward-shaping core matches the LspVerifier pattern:
- Diagnostics error code parsing
- Severity weights & diagnostics_to_reward integration
- Empty/whitespace/comment-only degenerate output guards
- Anti-Goodhart escape-hatch detection with superset/directives/none modes
- Thread-safe, injectable oracle seam for CI testing without compiler toolchains
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from src.lsp.diagnostics import Diagnostic, mask_strings_and_comments
from src.train.verifiers import (
    DEFAULT_SEVERITY_WEIGHTS,
    diagnostics_to_reward,
    is_degenerate_output,
    severity_weight,
)

_HATCH_MODES = ("superset", "directives", "none")


# --------------------------------------------------------------------------- #
# Toolchain Resolvers (Probe availability without executing untrusted code)
# --------------------------------------------------------------------------- #

def resolve_rust_toolchain() -> Optional[str]:
    """Locate rustc or cargo executable on PATH."""
    return shutil.which("rustc") or shutil.which("cargo")


def resolve_cpp_toolchain() -> Optional[str]:
    """Locate clang++ or g++ executable on PATH."""
    return shutil.which("clang++") or shutil.which("g++")


def resolve_swift_toolchain() -> Optional[str]:
    """Locate swiftc executable on PATH."""
    return shutil.which("swiftc")


def resolve_kotlin_toolchain() -> Optional[str]:
    """Locate kotlinc executable on PATH."""
    return shutil.which("kotlinc")


# --------------------------------------------------------------------------- #
# Diagnostic Error Parsers (Convert compiler/linter outputs to Diagnostics)
# --------------------------------------------------------------------------- #

_RUST_HEAD_RE = re.compile(
    r"^(?P<severity>error|warning|note|help)(?:\[(?P<code>[A-Za-z0-9_:]+)\])?:\s*(?P<message>.*)$"
)
_RUST_SPAN_RE = re.compile(
    r"^\s*-->\s*(?P<file>[^:]+):(?P<line>\d+):(?P<col>\d+)"
)


def parse_rust_diagnostics(output: str, source: str = "") -> List[Diagnostic]:
    """Parse rustc / cargo check / clippy output (both text and JSON format)."""
    diags: List[Diagnostic] = []
    lines = output.splitlines()
    pending: Optional[Dict[str, Any]] = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # JSON format (--message-format=json / --error-format=json)
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                data = json.loads(stripped)
                msg_obj = data["message"] if isinstance(data.get("message"), dict) else data
                if isinstance(msg_obj, dict) and "level" in msg_obj:
                    level = msg_obj.get("level", "error")
                    if level not in ("error", "warning", "note", "help"):
                        continue
                    msg_text = msg_obj.get("message", "")
                    if msg_text.startswith("aborting due to") or "try `rustc --explain" in msg_text:
                        continue
                    sev = 1 if level == "error" else (2 if level == "warning" else 3)
                    code_val = msg_obj.get("code")
                    code_str = ""
                    if isinstance(code_val, dict):
                        code_str = code_val.get("code", "")
                    elif code_val:
                        code_str = str(code_val)
                    if not code_str and "clippy::" in msg_text:
                        m_clip = re.search(r"\b(clippy::[a-zA-Z0-9_]+)\b", msg_text)
                        if m_clip:
                            code_str = m_clip.group(1)

                    spans = msg_obj.get("spans", [])
                    line_num = spans[0]["line_start"] if spans else 1
                    col_num = spans[0]["column_start"] if spans else 1
                    diags.append(
                        Diagnostic(
                            code=code_str or ("RUST_WARN" if sev == 2 else "RUST_ERR"),
                            line=line_num,
                            col=col_num,
                            message=msg_text,
                            offset=0,
                            source="rust",
                            severity=sev,
                        )
                    )
                    continue
            except (json.JSONDecodeError, KeyError):
                pass

        # Text format
        m_head = _RUST_HEAD_RE.match(line)
        if m_head:
            sev_str = m_head.group("severity")
            sev = 1 if sev_str == "error" else (2 if sev_str == "warning" else 3)
            code = m_head.group("code")
            msg = m_head.group("message").strip()
            if not code and "clippy::" in msg:
                m_clip = re.search(r"\b(clippy::[a-zA-Z0-9_]+)\b", msg)
                if m_clip:
                    code = m_clip.group(1)
            pending = {
                "sev": sev,
                "code": code or ("RUST_WARN" if sev == 2 else "RUST_ERR"),
                "message": msg,
                "line": 1,
                "col": 1,
            }
            diags.append(
                Diagnostic(
                    code=pending["code"],
                    line=pending["line"],
                    col=pending["col"],
                    message=pending["message"],
                    offset=0,
                    source="rust",
                    severity=pending["sev"],
                )
            )
            continue

        if pending is not None:
            m_span = _RUST_SPAN_RE.match(line)
            if m_span:
                last = diags[-1]
                diags[-1] = Diagnostic(
                    code=last.code,
                    line=int(m_span.group("line")),
                    col=int(m_span.group("col")),
                    message=last.message,
                    offset=last.offset,
                    source=last.source,
                    severity=last.severity,
                )
                pending = None

    return diags


_CPP_DIAG_RE = re.compile(
    r"^(?P<file>[^:\n\r]+):(?P<line>\d+):(?P<col>\d+):\s*(?:fatal\s+)?(?P<severity>error|warning|note):\s*(?P<message>.*?)(?:\s*\[(?P<code>-[W\w-]+|[a-zA-Z0-9_.-]+)\])?$"
)


def parse_cpp_diagnostics(output: str, source: str = "") -> List[Diagnostic]:
    """Parse clang++ / clang-tidy diagnostics."""
    diags: List[Diagnostic] = []
    for line in output.splitlines():
        m = _CPP_DIAG_RE.match(line.strip())
        if not m:
            continue
        sev_str = m.group("severity")
        sev = 1 if sev_str == "error" else (2 if sev_str == "warning" else 3)
        code = m.group("code")
        msg = m.group("message").strip()
        diags.append(
            Diagnostic(
                code=code or ("CPP_WARN" if sev == 2 else "CPP_ERR"),
                line=int(m.group("line")),
                col=int(m.group("col")),
                message=msg,
                offset=0,
                source="cpp",
                severity=sev,
            )
        )
    return diags


_SWIFT_DIAG_RE = re.compile(
    r"^(?P<file>[^:\n\r]+):(?P<line>\d+):(?P<col>\d+):\s*(?P<severity>error|warning|note):\s*(?P<message>.*?)(?:\s*\[(?P<code>[\w.-]+)\])?$"
)


def parse_swift_diagnostics(output: str, source: str = "") -> List[Diagnostic]:
    """Parse swiftc compiler diagnostics."""
    diags: List[Diagnostic] = []
    for line in output.splitlines():
        m = _SWIFT_DIAG_RE.match(line.strip())
        if not m:
            continue
        sev_str = m.group("severity")
        sev = 1 if sev_str == "error" else (2 if sev_str == "warning" else 3)
        code = m.group("code")
        msg = m.group("message").strip()
        if not code:
            if "sendable" in msg.lower() or "concurren" in msg.lower():
                code = "SWIFT_CONCURRENCY"
            elif "type" in msg.lower():
                code = "SWIFT_TYPECHECK"
            else:
                code = "SWIFT_WARN" if sev == 2 else "SWIFT_ERR"
        diags.append(
            Diagnostic(
                code=code,
                line=int(m.group("line")),
                col=int(m.group("col")),
                message=msg,
                offset=0,
                source="swift",
                severity=sev,
            )
        )
    return diags


_KT_DIAG_RE = re.compile(
    r"^(?P<file>[^:\n\r]+):(?P<line>\d+):(?P<col>\d+):\s*(?P<severity>error|warning|info):\s*(?P<message>.*)$"
)


def parse_kotlin_diagnostics(output: str, source: str = "") -> List[Diagnostic]:
    """Parse kotlinc compiler diagnostics."""
    diags: List[Diagnostic] = []
    for line in output.splitlines():
        m = _KT_DIAG_RE.match(line.strip())
        if not m:
            continue
        sev_str = m.group("severity")
        sev = 1 if sev_str == "error" else (2 if sev_str == "warning" else 3)
        msg = m.group("message").strip()
        code = "KT_WARN" if sev == 2 else "KT_ERR"
        if "type mismatch" in msg.lower():
            code = "KT_TYPE_MISMATCH"
        elif "when" in msg.lower() and "exhaustive" in msg.lower():
            code = "KT_NON_EXHAUSTIVE_WHEN"
        elif "null" in msg.lower():
            code = "KT_NULLABILITY"
        diags.append(
            Diagnostic(
                code=code,
                line=int(m.group("line")),
                col=int(m.group("col")),
                message=msg,
                offset=0,
                source="kotlin",
                severity=sev,
            )
        )
    return diags


# --------------------------------------------------------------------------- #
# Anti-Goodhart Escape Hatch Definitions
# --------------------------------------------------------------------------- #

# Rust Anti-Goodhart Hatches:
# Reject unsafe, .unwrap(), todo!(), unimplemented!(), #[allow(clippy::all)]
RUST_ESCAPE_HATCH_PATTERNS: Dict[str, re.Pattern] = {
    "unsafe": re.compile(r"\bunsafe\b"),
    "unwrap": re.compile(r"\.\s*unwrap\s*\("),
    "todo": re.compile(r"\btodo!\s*[\(\[]"),
    "unimplemented": re.compile(r"\bunimplemented!\s*[\(\[]"),
    "allow_clippy_all": re.compile(r"#!?\[\s*allow\s*\("),
    "panic": re.compile(r"\bpanic!\s*[\(\[]"),
}
RUST_DIRECTIVE_HATCHES = frozenset({"allow_clippy_all"})

# C / C++ Anti-Goodhart Hatches:
# Reject reinterpret_cast, raw pointer ownership transfer, implicit narrowing
CPP_ESCAPE_HATCH_PATTERNS: Dict[str, re.Pattern] = {
    "reinterpret_cast": re.compile(r"\breinterpret_cast\s*<"),
    "raw_pointer_ownership": re.compile(
        r"\bdelete\b|\bdelete\s*\[\s*\]|\bfree\s*\(|\.\s*release\s*\(\s*\)|\bnew\s+(?!std::)"
    ),
    "implicit_narrowing": re.compile(
        r"#pragma\s+(?:clang|GCC)\s+diagnostic\s+ignored|\(\s*void\s*\*?\s*\)"
    ),
}
CPP_DIRECTIVE_HATCHES = frozenset({"implicit_narrowing"})

# Swift Anti-Goodhart Hatches:
# Reject @unchecked Sendable, fatalError(), force unwrap !
SWIFT_ESCAPE_HATCH_PATTERNS: Dict[str, re.Pattern] = {
    "unchecked_sendable": re.compile(r"@unchecked\s+Sendable\b"),
    "fatal_error": re.compile(r"\bfatalError\s*\("),
    "force_unwrap": re.compile(r"(?<=[\w\)\]\"'`])!(?!=)"),
    "force_cast": re.compile(r"\bas!\s+"),
}
SWIFT_DIRECTIVE_HATCHES = frozenset({"unchecked_sendable"})

# Kotlin Anti-Goodhart Hatches:
# Reject @Suppress, !! force assertions, non-exhaustive when branches
KOTLIN_ESCAPE_HATCH_PATTERNS: Dict[str, re.Pattern] = {
    "suppress": re.compile(r"@(?:file:)?Suppress\s*\("),
    "force_assertion": re.compile(r"(?<!\!)!!(?!\!)"),
    "todo": re.compile(r"\bTODO\s*\("),
    "non_exhaustive_when": re.compile(
        r"else\s*->\s*(?:\{\s*\}|\(\)|\bUnit\b|\bTODO\b|null)"
    ),
}
KOTLIN_DIRECTIVE_HATCHES = frozenset({"suppress"})


def find_escape_hatches_for_lang(
    text: str,
    patterns: Mapping[str, re.Pattern],
    directive_hatches: frozenset[str],
) -> List[str]:
    """Find escape hatches in code text using string and comment masking."""
    masked = mask_strings_and_comments(text)
    found = set()
    for name, pattern in patterns.items():
        haystack = text if name in directive_hatches else masked
        if pattern.search(haystack):
            found.add(name)
    return sorted(found)


def find_rust_escape_hatches(text: str) -> List[str]:
    return find_escape_hatches_for_lang(text, RUST_ESCAPE_HATCH_PATTERNS, RUST_DIRECTIVE_HATCHES)


def find_cpp_escape_hatches(text: str) -> List[str]:
    return find_escape_hatches_for_lang(text, CPP_ESCAPE_HATCH_PATTERNS, CPP_DIRECTIVE_HATCHES)


def find_swift_escape_hatches(text: str) -> List[str]:
    return find_escape_hatches_for_lang(text, SWIFT_ESCAPE_HATCH_PATTERNS, SWIFT_DIRECTIVE_HATCHES)


def find_kotlin_escape_hatches(text: str) -> List[str]:
    return find_escape_hatches_for_lang(text, KOTLIN_ESCAPE_HATCH_PATTERNS, KOTLIN_DIRECTIVE_HATCHES)


# --------------------------------------------------------------------------- #
# Static Compiler Oracles (Non-executing toolchain runners)
# --------------------------------------------------------------------------- #

class RustCompilerOracle:
    """Invokes rustc in syntax/typecheck-only metadata mode without binary execution."""

    def __init__(self, *, target: Optional[str] = None, timeout_s: float = 10.0):
        self.target = target
        self.timeout_s = timeout_s
        self.n_calls = 0
        self.wall_s = 0.0

    def diagnostics(self, source: str) -> List[Diagnostic]:
        rustc = resolve_rust_toolchain()
        if not rustc:
            raise RuntimeError("rustc / cargo toolchain not found on PATH")

        t0 = time.monotonic()
        self.n_calls += 1
        with tempfile.TemporaryDirectory() as td:
            src_path = os.path.join(td, "lib.rs")
            rmeta_path = os.path.join(td, "lib.rmeta")
            with open(src_path, "w", encoding="utf-8") as f:
                f.write(source)

            cmd = [
                rustc,
                "--crate-type=lib",
                "--emit=metadata",
                "--error-format=json",
                "-o",
                rmeta_path,
                src_path,
            ]
            if self.target:
                cmd.extend(["--target", self.target])

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                )
                output = proc.stderr + "\n" + proc.stdout
                return parse_rust_diagnostics(output, source=source)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"rustc verification timed out after {self.timeout_s}s") from exc
            finally:
                self.wall_s += time.monotonic() - t0

    def close(self) -> None:
        pass


class CppCompilerOracle:
    """Invokes clang++ syntax-only checking (-fsyntax-only) without binary execution."""

    def __init__(self, *, std: str = "c++20", timeout_s: float = 10.0):
        self.std = std
        self.timeout_s = timeout_s
        self.n_calls = 0
        self.wall_s = 0.0

    def diagnostics(self, source: str) -> List[Diagnostic]:
        clang = resolve_cpp_toolchain()
        if not clang:
            raise RuntimeError("clang++ / g++ compiler not found on PATH")

        t0 = time.monotonic()
        self.n_calls += 1
        with tempfile.TemporaryDirectory() as td:
            src_path = os.path.join(td, "snippet.cpp")
            with open(src_path, "w", encoding="utf-8") as f:
                f.write(source)

            cmd = [
                clang,
                "-fsyntax-only",
                f"-std={self.std}",
                "-Wall",
                "-Werror",
                "-x",
                "c++",
                src_path,
            ]

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                )
                return parse_cpp_diagnostics(proc.stderr, source=source)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"clang++ verification timed out after {self.timeout_s}s") from exc
            finally:
                self.wall_s += time.monotonic() - t0

    def close(self) -> None:
        pass


class SwiftCompilerOracle:
    """Invokes swiftc typecheck-only checking without binary execution."""

    def __init__(self, *, strict_concurrency: str = "complete", timeout_s: float = 10.0):
        self.strict_concurrency = strict_concurrency
        self.timeout_s = timeout_s
        self.n_calls = 0
        self.wall_s = 0.0

    def diagnostics(self, source: str) -> List[Diagnostic]:
        swiftc = resolve_swift_toolchain()
        if not swiftc:
            raise RuntimeError("swiftc toolchain not found on PATH")

        t0 = time.monotonic()
        self.n_calls += 1
        with tempfile.TemporaryDirectory() as td:
            src_path = os.path.join(td, "snippet.swift")
            with open(src_path, "w", encoding="utf-8") as f:
                f.write(source)

            cmd = [
                swiftc,
                "-typecheck",
                f"-strict-concurrency={self.strict_concurrency}",
                src_path,
            ]

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                )
                return parse_swift_diagnostics(proc.stderr, source=source)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"swiftc verification timed out after {self.timeout_s}s") from exc
            finally:
                self.wall_s += time.monotonic() - t0

    def close(self) -> None:
        pass


class KotlinCompilerOracle:
    """Invokes kotlinc compiler with -Werror without binary execution."""

    def __init__(self, *, timeout_s: float = 10.0):
        self.timeout_s = timeout_s
        self.n_calls = 0
        self.wall_s = 0.0

    def diagnostics(self, source: str) -> List[Diagnostic]:
        kotlinc = resolve_kotlin_toolchain()
        if not kotlinc:
            raise RuntimeError("kotlinc toolchain not found on PATH")

        t0 = time.monotonic()
        self.n_calls += 1
        with tempfile.TemporaryDirectory() as td:
            src_path = os.path.join(td, "snippet.kt")
            with open(src_path, "w", encoding="utf-8") as f:
                f.write(source)

            cmd = [
                kotlinc,
                "-Xno-optimize",
                "-Werror",
                src_path,
            ]

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                )
                return parse_kotlin_diagnostics(proc.stderr, source=source)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"kotlinc verification timed out after {self.timeout_s}s") from exc
            finally:
                self.wall_s += time.monotonic() - t0

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# Static Compiler Verifiers (RLVR Reward Functions matching LspVerifier)
# --------------------------------------------------------------------------- #

class SystemsMobileVerifier:
    """Base static compiler verifier matching LspVerifier reward shaping.

    Evaluates completions on static diagnostic cleanliness and anti-Goodhart invariants.
    Pure reward-shaping core:
      - Diagnostic errors/warnings reduce reward from 1.0 (base clean)
      - Degenerate output (empty, whitespace, comment-only) short-circuits to -1.0
      - Language-specific escape hatches short-circuit to -1.0
    """

    def __init__(
        self,
        *,
        language: str,
        hatch_patterns: Mapping[str, re.Pattern],
        directive_hatches: frozenset[str],
        oracle_factory: Callable[[], Any],
        timeout_s: float = 10.0,
        hatches: str = "superset",
        oracle: Any = None,
        on_error: str = "raise",
        fail_fast: bool = True,
        **reward_kwargs: Any,
    ) -> None:
        if hatches not in _HATCH_MODES:
            raise ValueError(f"unknown hatches mode {hatches!r} (want one of {_HATCH_MODES})")
        if on_error not in ("raise", "skip"):
            raise ValueError(f"on_error must be 'raise' or 'skip', got {on_error!r}")

        self.language = language
        self.hatch_patterns = hatch_patterns
        self.directive_hatches = directive_hatches
        self.oracle_factory = oracle_factory
        self.timeout_s = timeout_s
        self.hatches = hatches
        self.oracle = oracle
        self.on_error = on_error
        self.fail_fast = fail_fast
        self.reward_kwargs = reward_kwargs

        self._closed = False
        self._lock = threading.Lock()

        self._n_samples = 0
        self._n_clean = 0
        self._n_hacked = 0
        self._n_degenerate = 0
        self._n_oracle_errors = 0
        self._hatch_counts: Dict[str, int] = {}
        self._degenerate_reasons: Dict[str, int] = {}
        self._diag_total = 0
        self._reward_total = 0.0

    def _ensure_oracle(self) -> Any:
        if self.oracle is None:
            self.oracle = self.oracle_factory()
        return self.oracle

    def find_escape_hatches(self, text: str) -> List[str]:
        return find_escape_hatches_for_lang(text, self.hatch_patterns, self.directive_hatches)

    def _select_hatches(self, all_hatches: Sequence[str]) -> List[str]:
        if self.hatches == "superset":
            return list(all_hatches)
        if self.hatches == "directives":
            return [h for h in all_hatches if h in self.directive_hatches]
        if self.hatches == "none":
            return []
        raise ValueError(f"unknown hatches mode {self.hatches!r}")

    def reward(
        self,
        completion: str,
        reference: Optional[str] = None,
        *,
        prompt: str = "",
        **kwargs: Any,
    ) -> Optional[float]:
        """Score one completion against static compiler rules.

        Matches scripts/rlvr.py reward_fn(decoded, ref, *, prompt=...) contract.
        """
        completion = "" if completion is None else str(completion)
        self._n_samples += 1

        reason = is_degenerate_output(completion)
        degenerate = reason is not None
        if degenerate:
            self._n_degenerate += 1
            self._degenerate_reasons[reason] = self._degenerate_reasons.get(reason, 0) + 1

        all_hatches = self.find_escape_hatches(completion)
        for h in all_hatches:
            self._hatch_counts[h] = self._hatch_counts.get(h, 0) + 1
        hacked = bool(self._select_hatches(all_hatches))
        if hacked:
            self._n_hacked += 1

        if self.fail_fast and (hacked or degenerate):
            diags: Sequence[Diagnostic] = []
        else:
            with self._lock:
                oracle = self._ensure_oracle()
                artifact = f"{prompt}{completion}"
                try:
                    diags = oracle.diagnostics(artifact)
                except Exception:
                    self._n_oracle_errors += 1
                    if self.on_error == "skip":
                        return None
                    raise

        self._diag_total += len(diags)
        if not diags and not hacked and not degenerate:
            self._n_clean += 1

        r = diagnostics_to_reward(
            diags,
            hacked=hacked,
            degenerate=degenerate,
            **self.reward_kwargs,
        )
        self._reward_total += r
        return r

    def telemetry(self) -> dict:
        n = self._n_samples
        oracle = self.oracle
        return {
            "language": self.language,
            "n_samples": n,
            "n_clean": self._n_clean,
            "n_hacked": self._n_hacked,
            "n_degenerate": self._n_degenerate,
            "n_oracle_errors": self._n_oracle_errors,
            "hatch_counts": dict(self._hatch_counts),
            "degenerate_reasons": dict(self._degenerate_reasons),
            "mean_diagnostics": (self._diag_total / n) if n else 0.0,
            "mean_reward": (self._reward_total / n) if n else 0.0,
            "n_calls": oracle.n_calls if oracle is not None and hasattr(oracle, "n_calls") else 0,
            "wall_s": oracle.wall_s if oracle is not None and hasattr(oracle, "wall_s") else 0.0,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.oracle is not None and hasattr(self.oracle, "close"):
            self.oracle.close()

    def __enter__(self) -> "SystemsMobileVerifier":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


class RustVerifier(SystemsMobileVerifier):
    """Static verifier for Rust code (Systems, WASM, CLI)."""

    def __init__(
        self,
        *,
        target: Optional[str] = None,
        timeout_s: float = 10.0,
        hatches: str = "superset",
        oracle: Any = None,
        on_error: str = "raise",
        fail_fast: bool = True,
        **reward_kwargs: Any,
    ) -> None:
        super().__init__(
            language="rust",
            hatch_patterns=RUST_ESCAPE_HATCH_PATTERNS,
            directive_hatches=RUST_DIRECTIVE_HATCHES,
            oracle_factory=lambda: RustCompilerOracle(target=target, timeout_s=timeout_s),
            timeout_s=timeout_s,
            hatches=hatches,
            oracle=oracle,
            on_error=on_error,
            fail_fast=fail_fast,
            **reward_kwargs,
        )


class CppVerifier(SystemsMobileVerifier):
    """Static verifier for C / C++ code (C17, C++20, C++23)."""

    def __init__(
        self,
        *,
        std: str = "c++20",
        timeout_s: float = 10.0,
        hatches: str = "superset",
        oracle: Any = None,
        on_error: str = "raise",
        fail_fast: bool = True,
        **reward_kwargs: Any,
    ) -> None:
        super().__init__(
            language="cpp",
            hatch_patterns=CPP_ESCAPE_HATCH_PATTERNS,
            directive_hatches=CPP_DIRECTIVE_HATCHES,
            oracle_factory=lambda: CppCompilerOracle(std=std, timeout_s=timeout_s),
            timeout_s=timeout_s,
            hatches=hatches,
            oracle=oracle,
            on_error=on_error,
            fail_fast=fail_fast,
            **reward_kwargs,
        )


class SwiftVerifier(SystemsMobileVerifier):
    """Static verifier for Swift code (iOS, macOS, SwiftUI)."""

    def __init__(
        self,
        *,
        strict_concurrency: str = "complete",
        timeout_s: float = 10.0,
        hatches: str = "superset",
        oracle: Any = None,
        on_error: str = "raise",
        fail_fast: bool = True,
        **reward_kwargs: Any,
    ) -> None:
        super().__init__(
            language="swift",
            hatch_patterns=SWIFT_ESCAPE_HATCH_PATTERNS,
            directive_hatches=SWIFT_DIRECTIVE_HATCHES,
            oracle_factory=lambda: SwiftCompilerOracle(
                strict_concurrency=strict_concurrency, timeout_s=timeout_s
            ),
            timeout_s=timeout_s,
            hatches=hatches,
            oracle=oracle,
            on_error=on_error,
            fail_fast=fail_fast,
            **reward_kwargs,
        )


class KotlinVerifier(SystemsMobileVerifier):
    """Static verifier for Kotlin code (Android, Jetpack Compose, Ktor)."""

    def __init__(
        self,
        *,
        timeout_s: float = 10.0,
        hatches: str = "superset",
        oracle: Any = None,
        on_error: str = "raise",
        fail_fast: bool = True,
        **reward_kwargs: Any,
    ) -> None:
        super().__init__(
            language="kotlin",
            hatch_patterns=KOTLIN_ESCAPE_HATCH_PATTERNS,
            directive_hatches=KOTLIN_DIRECTIVE_HATCHES,
            oracle_factory=lambda: KotlinCompilerOracle(timeout_s=timeout_s),
            timeout_s=timeout_s,
            hatches=hatches,
            oracle=oracle,
            on_error=on_error,
            fail_fast=fail_fast,
            **reward_kwargs,
        )


# Convenience aliases
RustStaticVerifier = RustVerifier
CppStaticVerifier = CppVerifier
SwiftStaticVerifier = SwiftVerifier
KotlinStaticVerifier = KotlinVerifier
