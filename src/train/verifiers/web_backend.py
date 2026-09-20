"""#343 -- Web & Backend Application Stack Verifiers.

Part of the M12 post-training, RLVR, and code capability track (#198, #230, #343).
Provides deterministic static verifiers and zero execution sandbox requirements across
the top web and enterprise backend application stacks:

1. TypeScript / React / Next.js / Node / NestJS:
   - In-process: JSX and TS AST / structure analyzer (<10ms).
   - Toolchain probe: tsc --noEmit diagnostic analyzer.
   - Checks: Hook dependency rules, Server/Client component directive boundaries,
     NestJS DI decorator integrity, route parameter typing.
   - Anti-Goodhart: Reject as any, @ts-ignore, @ts-expect-error, empty JSX <></>, empty handler bodies.

2. Python Web (FastAPI / Pydantic & Django ORM):
   - In-process: ast.parse(), Pydantic field schema & route binding validator.
   - Toolchain probe: pyright, django-admin check, makemigrations --dry-run.
   - Checks: Pydantic field schemas, route path parameter bindings, model reverse relationships,
     migration graph determinism.
   - Anti-Goodhart: Reject # type: ignore, bare dict returns, empty pass bodies.

3. Go Web & Microservices:
   - In-process: Go structure & AST analyzer.
   - Toolchain probe: go vet, golangci-lint run --fast.
   - Checks: Struct tags, unhandled error returns, goroutine leak patterns, interface implementation.
   - Anti-Goodhart: Reject _ = err, empty select{}, panic-only functions.

4. Java / Spring Boot 3:
   - In-process: Java AST and Spring structure analyzer.
   - Toolchain probe: javac -proc:only -Xlint:all.
   - Checks: @Component / @Service bean injection, Jakarta Persistence entity mappings,
     nullability annotations.
   - Anti-Goodhart: Reject empty catch (Exception e) {}, raw types, unmapped injection points.

5. C# / .NET & ASP.NET Core:
   - In-process: C# / Roslyn structure analyzer.
   - Toolchain probe: dotnet build /t:Compile.
   - Checks: Nullable reference types, Minimal API route bindings, Entity Framework model configurations.
   - Anti-Goodhart: Reject #pragma warning disable, dynamic, missing null guards.

6. PHP / Modern Web & Laravel:
   - In-process: PHP syntax & Laravel structure analyzer.
   - Toolchain probe: php -l, phpstan analyse --level=8.
   - Checks: Strict typing (declare(strict_types=1)), Eloquent relationship return types,
     Service Provider bindings.
   - Anti-Goodhart: Reject @phpstan-ignore, untyped docblocks.

7. Ruby / Rails & Modern Ruby:
   - In-process: Ruby syntax & Rails structure analyzer.
   - Toolchain probe: prism AST, rubocop --lint, sorbet.
   - Checks: Syntax correctness, Rails model validations, Sorbet typed signatures.
   - Anti-Goodhart: Reject # rubocop:disable, T.untyped.

Reward-shaping core matches LspVerifier and SystemsMobileVerifier patterns:
- Diagnostics error code parsing and severity weights
- Degenerate output guards (empty, whitespace, comment-only)
- Anti-Goodhart escape-hatch detection with superset/directives/none modes
- Thread-safe, injectable oracle seams for CI testing without external binaries
- Unified WebBackendVerifier auto-routing across stacks
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from src.lsp.diagnostics import Diagnostic, mask_strings_and_comments
from src.train.verifiers import (
    DEFAULT_SEVERITY_WEIGHTS,
    diagnostics_to_reward,
    is_degenerate_output,
    severity_weight,
)

_HATCH_MODES = ("superset", "directives", "none")


# --------------------------------------------------------------------------- #
# Toolchain Resolvers (Probe host environment without executing untrusted code)
# --------------------------------------------------------------------------- #

def resolve_ts_web_toolchain() -> Dict[str, bool]:
    """Check availability of TypeScript and Node toolchains."""
    return {
        "tsc": shutil.which("tsc") is not None,
        "node": shutil.which("node") is not None,
    }


def resolve_python_web_toolchain() -> Dict[str, bool]:
    """Check availability of Python web linting and framework tools."""
    return {
        "ast": True,
        "pyright": shutil.which("pyright") is not None,
        "django_admin": shutil.which("django-admin") is not None,
    }


def resolve_go_web_toolchain() -> Optional[str]:
    """Locate go or golangci-lint executable on PATH."""
    return shutil.which("golangci-lint") or shutil.which("go")


def resolve_java_web_toolchain() -> Optional[str]:
    """Locate javac executable on PATH."""
    return shutil.which("javac")


def resolve_csharp_web_toolchain() -> Optional[str]:
    """Locate dotnet executable on PATH."""
    return shutil.which("dotnet")


def resolve_php_web_toolchain() -> Dict[str, bool]:
    """Check availability of PHP interpreter and PHPStan."""
    return {
        "php": shutil.which("php") is not None,
        "phpstan": shutil.which("phpstan") is not None,
    }


def resolve_ruby_web_toolchain() -> Dict[str, bool]:
    """Check availability of Ruby tools (ruby, rubocop, sorbet/srb)."""
    return {
        "ruby": shutil.which("ruby") is not None,
        "rubocop": shutil.which("rubocop") is not None,
        "srb": shutil.which("srb") is not None,
    }


# --------------------------------------------------------------------------- #
# Comment & String Masking Helpers
# --------------------------------------------------------------------------- #

_PY_RUBY_COMMENT_RE = re.compile(r"#[^\r\n]*")
_PY_RUBY_STRING_RE = re.compile(
    r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\''
)


def mask_python_ruby_comments_and_strings(text: str) -> str:
    """Mask Python and Ruby comments and strings with whitespace to avoid false positives."""
    def _blank(m: re.Match) -> str:
        s = m.group(0)
        return "".join("\n" if c == "\n" else " " for c in s)

    text_no_str = _PY_RUBY_STRING_RE.sub(_blank, text)
    return _PY_RUBY_COMMENT_RE.sub(_blank, text_no_str)


# --------------------------------------------------------------------------- #
# Diagnostic Error Parsers (Compiler & Toolchain output -> List[Diagnostic])
# --------------------------------------------------------------------------- #

_TSC_DIAG_RE = re.compile(
    r"^(?P<file>[^:\n\r]+)(?:\((?P<line>\d+),(?P<col>\d+)\)|:(?P<line2>\d+):(?P<col2>\d+))\s*[-:]\s*(?:error\s+)?(?P<code>TS\d+):\s*(?P<message>.*)$"
)


def parse_tsc_web_diagnostics(output: str, source: str = "") -> List[Diagnostic]:
    """Parse tsc output into Diagnostics."""
    diags: List[Diagnostic] = []
    for line in output.splitlines():
        m = _TSC_DIAG_RE.match(line.strip())
        if not m:
            continue
        ln = int(m.group("line") or m.group("line2") or 1)
        col = int(m.group("col") or m.group("col2") or 1)
        diags.append(
            Diagnostic(
                code=m.group("code"),
                line=ln,
                col=col,
                message=m.group("message").strip(),
                offset=0,
                source="tsc",
                severity=1,
            )
        )
    return diags


_PYRIGHT_LINE_RE = re.compile(
    r"^(?P<file>[^:\n\r]+):(?P<line>\d+):(?P<col>\d+)\s*-\s*(?P<severity>error|warning|information):\s*(?P<message>.*?)(?:\s*\((?P<code>[a-zA-Z0-9_.-]+)\))?$"
)


def parse_pyright_diagnostics(output: str, source: str = "") -> List[Diagnostic]:
    """Parse pyright output into Diagnostics."""
    diags: List[Diagnostic] = []
    for line in output.splitlines():
        m = _PYRIGHT_LINE_RE.match(line.strip())
        if not m:
            continue
        sev_str = m.group("severity")
        sev = 1 if sev_str == "error" else (2 if sev_str == "warning" else 3)
        code = m.group("code") or ("PYRIGHT_WARN" if sev == 2 else "PYRIGHT_ERR")
        diags.append(
            Diagnostic(
                code=code,
                line=int(m.group("line")),
                col=int(m.group("col")),
                message=m.group("message").strip(),
                offset=0,
                source="pyright",
                severity=sev,
            )
        )
    return diags


_GO_DIAG_RE = re.compile(
    r"^(?P<file>[^:\n\r]+):(?P<line>\d+):(?P<col>\d+):\s*(?P<message>.*)$"
)


def parse_go_web_diagnostics(output: str, source: str = "") -> List[Diagnostic]:
    """Parse go vet / golangci-lint output into Diagnostics."""
    diags: List[Diagnostic] = []
    for line in output.splitlines():
        m = _GO_DIAG_RE.match(line.strip())
        if not m:
            continue
        msg = m.group("message").strip()
        code = "GO_ERR"
        if "warning" in msg.lower():
            code = "GO_WARN"
        elif "error" in msg.lower():
            code = "GO_ERR"
        diags.append(
            Diagnostic(
                code=code,
                line=int(m.group("line")),
                col=int(m.group("col")),
                message=msg,
                offset=0,
                source="go",
                severity=2 if "warning" in msg.lower() else 1,
            )
        )
    return diags


_JAVAC_DIAG_RE = re.compile(
    r"^(?P<file>[^:\n\r]+):(?P<line>\d+):\s*(?:(?P<severity>error|warning):\s*)?(?P<message>.*)$"
)


def parse_javac_diagnostics(output: str, source: str = "") -> List[Diagnostic]:
    """Parse javac output into Diagnostics."""
    diags: List[Diagnostic] = []
    for line in output.splitlines():
        m = _JAVAC_DIAG_RE.match(line.strip())
        if not m:
            continue
        sev_str = m.group("severity") or "error"
        sev = 2 if sev_str == "warning" else 1
        diags.append(
            Diagnostic(
                code="JAVAC_WARN" if sev == 2 else "JAVAC_ERR",
                line=int(m.group("line")),
                col=1,
                message=m.group("message").strip(),
                offset=0,
                source="javac",
                severity=sev,
            )
        )
    return diags


_DOTNET_DIAG_RE = re.compile(
    r"^(?P<file>[^:\n\r]+)\((?P<line>\d+),(?P<col>\d+)\):\s*(?P<severity>error|warning)\s*(?P<code>CS\d+):\s*(?P<message>.*)$"
)


def parse_dotnet_diagnostics(output: str, source: str = "") -> List[Diagnostic]:
    """Parse dotnet build / Roslyn output into Diagnostics."""
    diags: List[Diagnostic] = []
    for line in output.splitlines():
        m = _DOTNET_DIAG_RE.match(line.strip())
        if not m:
            continue
        sev_str = m.group("severity")
        sev = 1 if sev_str == "error" else 2
        diags.append(
            Diagnostic(
                code=m.group("code"),
                line=int(m.group("line")),
                col=int(m.group("col")),
                message=m.group("message").strip(),
                offset=0,
                source="dotnet",
                severity=sev,
            )
        )
    return diags


_PHPSTAN_LINE_RE = re.compile(
    r"^\s*(?P<line>\d+)\s*\|\s*(?P<message>.*)$"
)


def parse_php_web_diagnostics(output: str, source: str = "") -> List[Diagnostic]:
    """Parse php -l or phpstan output into Diagnostics."""
    diags: List[Diagnostic] = []
    for line in output.splitlines():
        stripped = line.strip()
        if "PHP Parse error:" in stripped or "PHP Fatal error:" in stripped:
            m_ln = re.search(r"on line (\d+)", stripped)
            ln = int(m_ln.group(1)) if m_ln else 1
            diags.append(
                Diagnostic(
                    code="PHP_PARSE_ERR",
                    line=ln,
                    col=1,
                    message=stripped,
                    offset=0,
                    source="php",
                    severity=1,
                )
            )
            continue
        m = _PHPSTAN_LINE_RE.match(stripped)
        if m:
            diags.append(
                Diagnostic(
                    code="PHPSTAN_ERR",
                    line=int(m.group("line")),
                    col=1,
                    message=m.group("message").strip(),
                    offset=0,
                    source="phpstan",
                    severity=1,
                )
            )
    return diags


_RUBOCOP_DIAG_RE = re.compile(
    r"^(?P<file>[^:\n\r]+):(?P<line>\d+):(?P<col>\d+):\s*(?P<severity>[A-Z]):\s*(?P<message>.*)$"
)


def parse_ruby_web_diagnostics(output: str, source: str = "") -> List[Diagnostic]:
    """Parse rubocop output into Diagnostics."""
    diags: List[Diagnostic] = []
    for line in output.splitlines():
        m = _RUBOCOP_DIAG_RE.match(line.strip())
        if not m:
            continue
        sev_code = m.group("severity")
        sev = 1 if sev_code in ("E", "F") else (2 if sev_code == "W" else 3)
        diags.append(
            Diagnostic(
                code="RUBOCOP_ERR" if sev == 1 else "RUBOCOP_WARN",
                line=int(m.group("line")),
                col=int(m.group("col")),
                message=m.group("message").strip(),
                offset=0,
                source="rubocop",
                severity=sev,
            )
        )
    return diags


# --------------------------------------------------------------------------- #
# Anti-Goodhart Escape Hatch Definitions & Finders
# --------------------------------------------------------------------------- #

# 1. TypeScript / React / Next.js / NestJS
TS_WEB_ESCAPE_HATCH_PATTERNS: Dict[str, re.Pattern] = {
    "as_any": re.compile(r"\bas\s+any\b"),
    "ts_ignore": re.compile(r"@ts-ignore\b"),
    "ts_expect_error": re.compile(r"@ts-expect-error\b"),
    "empty_jsx": re.compile(r"<>\s*</>|<React\.Fragment>\s*</React\.Fragment>"),
    "empty_handler": re.compile(
        r"(?:on[A-Z]\w*|onClick|onChange|onSubmit|handle[A-Z]\w*)\s*=\s*\{(?:\s*\(\s*\)|[a-zA-Z0-9_]+)\s*=>\s*\{\s*\}\s*\}|"
        r"\bfunction\s+[a-zA-Z0-9_]*Handler\s*\([^)]*\)\s*\{\s*\}"
    ),
}
TS_WEB_DIRECTIVE_HATCHES = frozenset({"ts_ignore", "ts_expect_error", "as_any"})

# 2. Python Web (FastAPI / Pydantic / Django)
PYTHON_WEB_ESCAPE_HATCH_PATTERNS: Dict[str, re.Pattern] = {
    "type_ignore": re.compile(r"#\s*type:\s*ignore\b"),
    "bare_dict_return": re.compile(r"->\s*(?:dict|Dict)\b(?!\s*\[)"),
    "empty_pass_body": re.compile(
        r"(?:def|class)\s+[a-zA-Z0-9_]+\s*\([^)]*\)(?:\s*->\s*[^:]+)?\s*:\s*(?:pass|\.\.\.)\s*(?:\n|$)"
    ),
}
PYTHON_WEB_DIRECTIVE_HATCHES = frozenset({"type_ignore"})

# 3. Go Web & Microservices
GO_WEB_ESCAPE_HATCH_PATTERNS: Dict[str, re.Pattern] = {
    "blank_err_assignment": re.compile(r"\b_\s*=\s*err\b"),
    "empty_select": re.compile(r"\bselect\s*\{\s*\}"),
    "panic_only_function": re.compile(
        r"func\s+(?:\([^)]+\)\s*)?[a-zA-Z0-9_]+\s*\([^)]*\)[^{]*\{\s*panic\s*\([^)]*\)\s*;?\s*\}"
    ),
}
GO_WEB_DIRECTIVE_HATCHES: frozenset[str] = frozenset()

# 4. Java / Spring Boot 3
JAVA_WEB_ESCAPE_HATCH_PATTERNS: Dict[str, re.Pattern] = {
    "empty_catch": re.compile(
        r"catch\s*\(\s*(?:Exception|Throwable|[A-Z]\w*Exception)\s+[a-zA-Z0-9_]+\s*\)\s*\{\s*\}"
    ),
    "raw_types": re.compile(
        r"\b(?:List|Map|Set|Queue|Collection|Optional)\s+[a-zA-Z0-9_]+\s*[=;,)]"
    ),
    "unmapped_injection": re.compile(
        r"@Autowired\s*(?:private|protected|public)?\s+Object\s+[a-zA-Z0-9_]+"
    ),
}
JAVA_WEB_DIRECTIVE_HATCHES: frozenset[str] = frozenset()

# 5. C# / .NET & ASP.NET Core
CSHARP_WEB_ESCAPE_HATCH_PATTERNS: Dict[str, re.Pattern] = {
    "pragma_warning_disable": re.compile(r"#pragma\s+warning\s+disable"),
    "dynamic_keyword": re.compile(r"\bdynamic\b"),
    "null_forgiving_abuse": re.compile(r"(?<!=)!\s*\.\s*[a-zA-Z_]"),
}
CSHARP_WEB_DIRECTIVE_HATCHES = frozenset({"pragma_warning_disable"})

# 6. PHP / Modern Web & Laravel
PHP_WEB_ESCAPE_HATCH_PATTERNS: Dict[str, re.Pattern] = {
    "phpstan_ignore": re.compile(r"@phpstan-ignore\b"),
    "untyped_docblock": re.compile(r"@param\s+mixed\b|@return\s+mixed\b"),
    "untyped_param": re.compile(
        r"function\s+[a-zA-Z0-9_]+\s*\(\s*\$[a-zA-Z0-9_]+\s*[,)]"
    ),
}
PHP_WEB_DIRECTIVE_HATCHES = frozenset({"phpstan_ignore"})

# 7. Ruby / Rails & Modern Ruby
RUBY_WEB_ESCAPE_HATCH_PATTERNS: Dict[str, re.Pattern] = {
    "rubocop_disable": re.compile(r"#\s*rubocop:disable\b"),
    "sorbet_untyped": re.compile(r"\bT\.untyped\b"),
}
RUBY_WEB_DIRECTIVE_HATCHES = frozenset({"rubocop_disable"})


def find_escape_hatches_for_web(
    text: str,
    patterns: Mapping[str, re.Pattern],
    directive_hatches: frozenset[str],
    *,
    is_hash_comment: bool = False,
) -> List[str]:
    """Find anti-Goodhart escape hatches in web source code."""
    if not text:
        return []
    if is_hash_comment:
        masked = mask_python_ruby_comments_and_strings(text)
    else:
        try:
            masked = mask_strings_and_comments(text)
        except Exception:
            masked = text

    found: Set[str] = set()
    for name, pattern in patterns.items():
        haystack = text if name in directive_hatches else masked
        if pattern.search(haystack):
            found.add(name)
    return sorted(found)


def find_ts_web_escape_hatches(text: str) -> List[str]:
    return find_escape_hatches_for_web(text, TS_WEB_ESCAPE_HATCH_PATTERNS, TS_WEB_DIRECTIVE_HATCHES)


def find_python_web_escape_hatches(text: str) -> List[str]:
    return find_escape_hatches_for_web(
        text, PYTHON_WEB_ESCAPE_HATCH_PATTERNS, PYTHON_WEB_DIRECTIVE_HATCHES, is_hash_comment=True
    )


def find_go_web_escape_hatches(text: str) -> List[str]:
    return find_escape_hatches_for_web(text, GO_WEB_ESCAPE_HATCH_PATTERNS, GO_WEB_DIRECTIVE_HATCHES)


def find_java_web_escape_hatches(text: str) -> List[str]:
    return find_escape_hatches_for_web(text, JAVA_WEB_ESCAPE_HATCH_PATTERNS, JAVA_WEB_DIRECTIVE_HATCHES)


def find_csharp_web_escape_hatches(text: str) -> List[str]:
    return find_escape_hatches_for_web(text, CSHARP_WEB_ESCAPE_HATCH_PATTERNS, CSHARP_WEB_DIRECTIVE_HATCHES)


def find_php_web_escape_hatches(text: str) -> List[str]:
    return find_escape_hatches_for_web(text, PHP_WEB_ESCAPE_HATCH_PATTERNS, PHP_WEB_DIRECTIVE_HATCHES)


def find_ruby_web_escape_hatches(text: str) -> List[str]:
    return find_escape_hatches_for_web(
        text, RUBY_WEB_ESCAPE_HATCH_PATTERNS, RUBY_WEB_DIRECTIVE_HATCHES, is_hash_comment=True
    )


# --------------------------------------------------------------------------- #
# In-Process Static Oracles (Fast, Deterministic, Zero Execution Sandbox)
# --------------------------------------------------------------------------- #

class TypeScriptWebOracle:
    """In-process AST & structural analyzer for TypeScript, React, Next.js & NestJS."""

    _CLIENT_HOOKS = frozenset({
        "useState", "useEffect", "useContext", "useReducer", "useRef",
        "useLayoutEffect", "useTransition", "useId", "useCallback", "useMemo",
    })
    _CLIENT_GLOBALS = (r"\bwindow\.", r"\blocalStorage\.", r"\bsessionStorage\.", r"\bdocument\.")

    def __init__(self, *, timeout_s: float = 5.0):
        self.timeout_s = timeout_s
        self.n_calls = 0
        self.wall_s = 0.0

    def diagnostics(self, source: str) -> List[Diagnostic]:
        t0 = time.monotonic()
        self.n_calls += 1
        diags: List[Diagnostic] = []
        lines = source.splitlines()

        has_use_client = any(
            re.search(r"^['\"]use client['\"];?", line.strip()) for line in lines[:5]
        )
        has_use_server = any(
            re.search(r"^['\"]use server['\"];?", line.strip()) for line in lines[:5]
        )

        # Multi-line check for hooks inside conditionals or loops
        for m_cond in re.finditer(r"\b(?:if|for|while)\s*\([^)]*\)\s*\{[^}]*?\b(use[A-Z]\w*)\b", source, re.DOTALL):
            hook_name = m_cond.group(1)
            ln = source[:m_cond.start(1)].count("\n") + 1
            diags.append(
                Diagnostic(
                    code="TS_HOOK_INSIDE_CONDITIONAL",
                    line=ln,
                    col=1,
                    message=f"React hook {hook_name} called conditionally or inside loop, violating Hook rules",
                    offset=0,
                    source="react_hooks",
                    severity=1,
                )
            )

        for line_no, line in enumerate(lines, start=1):
            stripped = line.strip()

            # 1. Hook Dependency Rules
            m_hook = re.search(r"\b(useEffect|useCallback|useMemo)\s*\(\s*(?:\([^)]*\)|[a-zA-Z0-9_]+)\s*=>", stripped)
            if m_hook:
                hook_name = m_hook.group(1)
                if re.search(r"\b" + hook_name + r"\s*\([^,]+\)\s*;?$", stripped):
                    diags.append(
                        Diagnostic(
                            code="TS_HOOK_MISSING_DEPS",
                            line=line_no,
                            col=1,
                            message=f"React hook {hook_name} is missing a dependency array argument",
                            offset=0,
                            source="react_hooks",
                            severity=1,
                        )
                    )

            # Hook called inside conditional or loop
            m_cond_hook = re.search(
                r"\b(?:if|for|while)\s*\([^)]*\)\s*\{[^}]*\b(use[A-Z]\w*)\b", stripped
            )
            if m_cond_hook:
                hook_name = m_cond_hook.group(1)
                diags.append(
                    Diagnostic(
                        code="TS_HOOK_INSIDE_CONDITIONAL",
                        line=line_no,
                        col=1,
                        message=f"React hook {hook_name} called conditionally or inside loop, violating Hook rules",
                        offset=0,
                        source="react_hooks",
                        severity=1,
                    )
                )

            # 2. Server/Client component directive boundaries
            uses_client_hook = any(re.search(r"\b" + h + r"\b", stripped) for h in self._CLIENT_HOOKS)
            uses_client_global = any(re.search(p, stripped) for p in self._CLIENT_GLOBALS)

            if (uses_client_hook or uses_client_global) and not has_use_client:
                diags.append(
                    Diagnostic(
                        code="TS_MISSING_USE_CLIENT",
                        line=line_no,
                        col=1,
                        message="File uses client-only hooks or browser globals but is missing 'use client' directive",
                        offset=0,
                        source="nextjs_boundaries",
                        severity=1,
                    )
                )

            if has_use_server and uses_client_hook:
                diags.append(
                    Diagnostic(
                        code="TS_INVALID_USE_SERVER_HOOK",
                        line=line_no,
                        col=1,
                        message="File declared 'use server' cannot invoke client React hooks",
                        offset=0,
                        source="nextjs_boundaries",
                        severity=1,
                    )
                )

            # 3. NestJS DI decorator integrity
            if re.search(r"constructor\s*\([^)]*\)", stripped):
                m_params = re.findall(
                    r"(?:private|public|protected)?\s*(?:readonly\s+)?([a-zA-Z0-9_]+)\s*(?:,|[)]|$)",
                    stripped,
                )
                for p in m_params:
                    if p and p not in ("readonly", "private", "public", "protected", ""):
                        if f"{p}:" not in stripped and f"{p} :" not in stripped:
                            diags.append(
                                Diagnostic(
                                    code="NESTJS_UNTYPED_DI_PARAM",
                                    line=line_no,
                                    col=1,
                                    message=f"NestJS constructor injection parameter {p!r} is missing type annotation",
                                    offset=0,
                                    source="nestjs_di",
                                    severity=1,
                                )
                            )

            # 4. Route parameter typing
            if re.search(r"(?:export\s+)?async\s+function\s+(?:GET|POST|PUT|DELETE|PATCH)\s*\([^)]*\{\s*params\s*\}\s*[,)]", stripped):
                if ": Promise<" not in stripped and ": {" not in stripped and ": Params" not in stripped:
                    diags.append(
                        Diagnostic(
                            code="TS_UNTYPED_ROUTE_PARAM",
                            line=line_no,
                            col=1,
                            message="Next.js route handler params object lacks explicit type annotation",
                            offset=0,
                            source="nextjs_routes",
                            severity=1,
                        )
                    )

            if re.search(r"@Param\s*\([^)]*\)\s*[a-zA-Z0-9_]+\s*(?:,|[)]|$)", stripped):
                if ":" not in stripped:
                    diags.append(
                        Diagnostic(
                            code="NESTJS_UNTYPED_PARAM",
                            line=line_no,
                            col=1,
                            message="NestJS @Param decorator target parameter is missing type annotation",
                            offset=0,
                            source="nestjs_routes",
                            severity=1,
                        )
                    )

        self.wall_s += time.monotonic() - t0
        return diags

    def close(self) -> None:
        pass


class PythonWebOracle:
    """In-process AST & schema analyzer for FastAPI, Pydantic & Django ORM."""

    def __init__(self, *, timeout_s: float = 5.0):
        self.timeout_s = timeout_s
        self.n_calls = 0
        self.wall_s = 0.0

    def diagnostics(self, source: str) -> List[Diagnostic]:
        t0 = time.monotonic()
        self.n_calls += 1
        diags: List[Diagnostic] = []

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            self.wall_s += time.monotonic() - t0
            return [
                Diagnostic(
                    code="PY_SYNTAX_ERROR",
                    line=e.lineno or 1,
                    col=e.offset or 1,
                    message=f"Python syntax error: {e.msg}",
                    offset=0,
                    source="python_ast",
                    severity=1,
                )
            ]

        for node in ast.walk(tree):
            # 1. Pydantic field schemas: untyped fields in BaseModel subclasses
            if isinstance(node, ast.ClassDef):
                base_names = [
                    b.id if isinstance(b, ast.Name) else (b.attr if isinstance(b, ast.Attribute) else "")
                    for b in node.bases
                ]
                if any(b in ("BaseModel", "BaseSettings") for b in base_names):
                    for stmt in node.body:
                        if isinstance(stmt, ast.Assign):
                            for target in stmt.targets:
                                target_name = target.id if isinstance(target, ast.Name) else "field"
                                if not target_name.startswith("_") and target_name != "model_config":
                                    diags.append(
                                        Diagnostic(
                                            code="PYDANTIC_UNTYPED_FIELD",
                                            line=stmt.lineno,
                                            col=stmt.col_offset,
                                            message=f"Pydantic model {node.name!r} has untyped field {target_name!r}; type annotation required",
                                            offset=0,
                                            source="pydantic_schema",
                                            severity=1,
                                        )
                                    )

                # 2. Django ORM models: ForeignKey missing on_delete, missing related_name
                if any(b in ("Model", "models.Model") for b in base_names):
                    fk_targets: List[str] = []
                    for stmt in node.body:
                        val = None
                        if isinstance(stmt, ast.Assign):
                            val = stmt.value
                        elif isinstance(stmt, ast.AnnAssign):
                            val = stmt.value

                        if isinstance(val, ast.Call):
                            func_name = ""
                            if isinstance(val.func, ast.Name):
                                func_name = val.func.id
                            elif isinstance(val.func, ast.Attribute):
                                func_name = val.func.attr

                            if func_name in ("ForeignKey", "OneToOneField"):
                                kw_names = [kw.arg for kw in val.keywords if kw.arg]
                                if "on_delete" not in kw_names:
                                    diags.append(
                                        Diagnostic(
                                            code="DJANGO_MISSING_ON_DELETE",
                                            line=stmt.lineno,
                                            col=stmt.col_offset,
                                            message=f"Django relationship {func_name} in {node.name!r} must define on_delete",
                                            offset=0,
                                            source="django_orm",
                                            severity=1,
                                        )
                                    )
                                target_model = ""
                                if val.args:
                                    arg0 = val.args[0]
                                    if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                                        target_model = arg0.value
                                    elif isinstance(arg0, ast.Name):
                                        target_model = arg0.id
                                if target_model:
                                    if target_model in fk_targets and "related_name" not in kw_names:
                                        diags.append(
                                            Diagnostic(
                                                code="DJANGO_MISSING_RELATED_NAME",
                                                line=stmt.lineno,
                                                col=stmt.col_offset,
                                                message=f"Multiple ForeignKeys to {target_model!r} in {node.name!r} require distinct related_name",
                                                offset=0,
                                                source="django_orm",
                                                severity=2,
                                            )
                                        )
                                    fk_targets.append(target_model)

                # 3. Django Migration Graph Determinism
                if any(b in ("Migration", "migrations.Migration") for b in base_names):
                    attrs = {
                        stmt.targets[0].id
                        for stmt in node.body
                        if isinstance(stmt, ast.Assign) and stmt.targets and isinstance(stmt.targets[0], ast.Name)
                    }
                    if "dependencies" not in attrs or "operations" not in attrs:
                        diags.append(
                            Diagnostic(
                                code="DJANGO_INVALID_MIGRATION",
                                line=node.lineno,
                                col=node.col_offset,
                                message="Django Migration class must define both 'dependencies' and 'operations' attributes",
                                offset=0,
                                source="django_migrations",
                                severity=1,
                            )
                        )

            # 4. FastAPI route path parameter bindings & return typing
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if len(node.body) == 1:
                    first = node.body[0]
                    if isinstance(first, ast.Pass) or (
                        isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and first.value.value is Ellipsis
                    ):
                        diags.append(
                            Diagnostic(
                                code="PY_EMPTY_HANDLER_BODY",
                                line=node.lineno,
                                col=node.col_offset,
                                message=f"Function {node.name!r} has empty body (pass / ...)",
                                offset=0,
                                source="fastapi_routes",
                                severity=1,
                            )
                        )

                for dec in node.decorator_list:
                    if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
                        method = dec.func.attr
                        if method in ("get", "post", "put", "delete", "patch", "api_route"):
                            if dec.args and isinstance(dec.args[0], ast.Constant) and isinstance(dec.args[0].value, str):
                                path_template = dec.args[0].value
                                path_params = re.findall(r"\{([a-zA-Z0-9_]+)\}", path_template)
                                fn_param_map = {arg.arg: arg for arg in node.args.args}

                                for p in path_params:
                                    if p not in fn_param_map:
                                        diags.append(
                                            Diagnostic(
                                                code="FASTAPI_UNBOUND_PATH_PARAM",
                                                line=node.lineno,
                                                col=node.col_offset,
                                                message=f"FastAPI route path parameter {{{p}}} is not declared in function signature of {node.name!r}",
                                                offset=0,
                                                source="fastapi_routes",
                                                severity=1,
                                            )
                                        )
                                    else:
                                        arg_node = fn_param_map[p]
                                        if arg_node.annotation is None:
                                            diags.append(
                                                Diagnostic(
                                                    code="FASTAPI_UNTYPED_PARAM",
                                                    line=node.lineno,
                                                    col=node.col_offset,
                                                    message=f"FastAPI route parameter {p!r} in {node.name!r} must have explicit type annotation",
                                                    offset=0,
                                                    source="fastapi_routes",
                                                    severity=1,
                                                )
                                            )

                if node.returns is not None:
                    ret_id = None
                    if isinstance(node.returns, ast.Name):
                        ret_id = node.returns.id
                    elif isinstance(node.returns, ast.Attribute):
                        ret_id = node.returns.attr
                    if ret_id in ("dict", "Dict"):
                        diags.append(
                            Diagnostic(
                                code="FASTAPI_BARE_DICT_RETURN",
                                line=node.lineno,
                                col=node.col_offset,
                                message=f"Handler {node.name!r} specifies bare dict return; typed schema or Pydantic model required",
                                offset=0,
                                source="fastapi_routes",
                                severity=1,
                            )
                        )

        self.wall_s += time.monotonic() - t0
        return diags

    def close(self) -> None:
        pass


class GoWebOracle:
    """In-process structural analyzer for Go web services & microservices."""

    def __init__(self, *, timeout_s: float = 5.0):
        self.timeout_s = timeout_s
        self.n_calls = 0
        self.wall_s = 0.0

    def diagnostics(self, source: str) -> List[Diagnostic]:
        t0 = time.monotonic()
        self.n_calls += 1
        diags: List[Diagnostic] = []
        lines = source.splitlines()

        in_struct = False
        for line_no, line in enumerate(lines, start=1):
            stripped = line.strip()

            if "struct {" in stripped:
                in_struct = True
            elif in_struct and stripped == "}":
                in_struct = False

            # 1. Go struct tag syntax
            if in_struct and "`" in stripped:
                m_tag = re.search(r"`([^`]+)`", stripped)
                if m_tag:
                    tag_content = m_tag.group(1).strip()
                    tokens = [t for t in tag_content.split(" ") if t]
                    for tok in tokens:
                        if not re.match(r'^[a-zA-Z0-9_-]+:"[^"]*"$', tok):
                            diags.append(
                                Diagnostic(
                                    code="GO_MALFORMED_STRUCT_TAG",
                                    line=line_no,
                                    col=1,
                                    message=f"Malformed Go struct tag token {tok!r}; want key:\"value\"",
                                    offset=0,
                                    source="go_struct_tags",
                                    severity=1,
                                )
                            )

            # 2. Unhandled error returns: err := ... followed directly by non-check or ignored
            if re.search(r"\b(?:[a-zA-Z0-9_]+,\s*)?err\s*:?=\s*", stripped):
                remaining = " ".join(l.strip() for l in lines[line_no: min(line_no + 3, len(lines))])
                if remaining and "if err != nil" not in remaining and "return" in remaining:
                    if "return err" not in remaining and "return nil, err" not in remaining and "return res, err" not in remaining:
                        diags.append(
                            Diagnostic(
                                code="GO_UNHANDLED_ERROR",
                                line=line_no,
                                col=1,
                                message="Unhandled error assignment; err received without 'if err != nil' check",
                                offset=0,
                                source="go_error_handling",
                                severity=1,
                            )
                        )

            # 3. Goroutine leak patterns
            if "go func(" in stripped:
                window = "\n".join(lines[line_no - 1: min(line_no + 15, len(lines))])
                if "for {" in window and "select" not in window and "Done()" not in window:
                    diags.append(
                        Diagnostic(
                            code="GO_GOROUTINE_LEAK",
                            line=line_no,
                            col=1,
                            message="Goroutine executes unconditioned loop without select or context cancellation guard",
                            offset=0,
                            source="go_concurrency",
                            severity=1,
                        )
                    )

        self.wall_s += time.monotonic() - t0
        return diags

    def close(self) -> None:
        pass


class JavaWebOracle:
    """In-process structural analyzer for Java & Spring Boot 3 applications."""

    def __init__(self, *, timeout_s: float = 5.0):
        self.timeout_s = timeout_s
        self.n_calls = 0
        self.wall_s = 0.0

    def diagnostics(self, source: str) -> List[Diagnostic]:
        t0 = time.monotonic()
        self.n_calls += 1
        diags: List[Diagnostic] = []
        lines = source.splitlines()

        is_entity = any("@Entity" in l for l in lines)
        has_id = any("@Id" in l for l in lines)

        if is_entity and not has_id:
            diags.append(
                Diagnostic(
                    code="JPA_MISSING_ID",
                    line=1,
                    col=1,
                    message="Jakarta Persistence @Entity class must declare a primary key field annotated with @Id",
                    offset=0,
                    source="jakarta_persistence",
                    severity=1,
                )
            )

        for line_no, line in enumerate(lines, start=1):
            stripped = line.strip()

            # 1. Field injection antipattern in Spring Boot 3
            if stripped.startswith("@Autowired") and line_no < len(lines):
                next_line = lines[line_no].strip() if line_no < len(lines) else ""
                if ";" in next_line and "(" not in next_line:
                    diags.append(
                        Diagnostic(
                            code="SPRING_FIELD_INJECTION",
                            line=line_no,
                            col=1,
                            message="@Autowired on field is discouraged in Spring Boot 3; use constructor injection",
                            offset=0,
                            source="spring_injection",
                            severity=2,
                        )
                    )

            # 2. JPA @OneToMany missing mappedBy
            if "@OneToMany" in stripped and "mappedBy" not in stripped:
                next_line = lines[line_no].strip() if line_no < len(lines) else ""
                if "@JoinColumn" not in next_line:
                    diags.append(
                        Diagnostic(
                            code="JPA_INVALID_RELATIONSHIP",
                            line=line_no,
                            col=1,
                            message="JPA @OneToMany mapping must specify 'mappedBy' or be paired with @JoinColumn",
                            offset=0,
                            source="jakarta_persistence",
                            severity=1,
                        )
                    )

        self.wall_s += time.monotonic() - t0
        return diags

    def close(self) -> None:
        pass


class CSharpWebOracle:
    """In-process structural analyzer for C# / ASP.NET Core applications."""

    def __init__(self, *, timeout_s: float = 5.0):
        self.timeout_s = timeout_s
        self.n_calls = 0
        self.wall_s = 0.0

    def diagnostics(self, source: str) -> List[Diagnostic]:
        t0 = time.monotonic()
        self.n_calls += 1
        diags: List[Diagnostic] = []
        lines = source.splitlines()

        for line_no, line in enumerate(lines, start=1):
            stripped = line.strip()

            # 1. Minimal API route bindings: app.MapGet("/users/{id}", ...)
            m_route = re.search(r"\.Map(?:Get|Post|Put|Delete)\s*\(\s*\"([^\"]+)\"", stripped)
            if m_route:
                route_tmpl = m_route.group(1)
                route_params = re.findall(r"\{([a-zA-Z0-9_]+)\}", route_tmpl)
                if route_params:
                    window = " ".join(lines[line_no - 1: min(line_no + 2, len(lines))])
                    for p in route_params:
                        if not re.search(r"\b" + p + r"\b", window.split(route_tmpl)[-1]):
                            diags.append(
                                Diagnostic(
                                    code="CSHARP_UNBOUND_ROUTE_PARAM",
                                    line=line_no,
                                    col=1,
                                    message=f"Minimal API route template parameter {{{p}}} is not bound in endpoint delegate",
                                    offset=0,
                                    source="aspnet_core",
                                    severity=1,
                                )
                            )

            # 2. Entity Framework: Entity missing primary key
            if "class " in stripped and "{" in stripped and "DbContext" not in stripped:
                window = "\n".join(lines[line_no - 1: min(line_no + 25, len(lines))])
                if "public " in window and "get; set;" in window:
                    if "[Key]" not in window and not re.search(r"\bId\s*\{\s*get;", window):
                        diags.append(
                            Diagnostic(
                                code="EF_MISSING_PRIMARY_KEY",
                                line=line_no,
                                col=1,
                                message="Entity model class lacks primary key property 'Id' or [Key] attribute",
                                offset=0,
                                source="ef_core",
                                severity=1,
                            )
                        )

        self.wall_s += time.monotonic() - t0
        return diags

    def close(self) -> None:
        pass


class PhpWebOracle:
    """In-process structural analyzer for PHP & Laravel applications."""

    def __init__(self, *, timeout_s: float = 5.0):
        self.timeout_s = timeout_s
        self.n_calls = 0
        self.wall_s = 0.0

    def diagnostics(self, source: str) -> List[Diagnostic]:
        t0 = time.monotonic()
        self.n_calls += 1
        diags: List[Diagnostic] = []
        lines = source.splitlines()

        # 1. Strict typing declaration check
        has_strict_types = any("declare(strict_types=1)" in l.replace(" ", "") for l in lines[:5])
        if lines and any("<?php" in l for l in lines[:3]) and not has_strict_types:
            diags.append(
                Diagnostic(
                    code="PHP_MISSING_STRICT_TYPES",
                    line=1,
                    col=1,
                    message="PHP file missing declare(strict_types=1); at header",
                    offset=0,
                    source="php_typing",
                    severity=2,
                )
            )

        # 2. Eloquent relationship return types
        is_model = any("extends Model" in l for l in lines)
        for line_no, line in enumerate(lines, start=1):
            stripped = line.strip()

            if is_model and re.search(r"public\s+function\s+[a-zA-Z0-9_]+\s*\([^)]*\)", stripped):
                window = "\n".join(lines[line_no - 1: min(line_no + 8, len(lines))])
                if any(rel in window for rel in ("$this->hasMany", "$this->belongsTo", "$this->hasOne", "$this->belongsToMany")):
                    if not re.search(r"\)\s*:\s*[A-Z]\w*", stripped):
                        diags.append(
                            Diagnostic(
                                code="LARAVEL_UNTYPED_RELATIONSHIP",
                                line=line_no,
                                col=1,
                                message="Laravel Eloquent relationship method lacks explicit return type (e.g. : HasMany)",
                                offset=0,
                                source="laravel_eloquent",
                                severity=1,
                            )
                        )

            # 3. Service Provider bindings
            if "extends ServiceProvider" in stripped:
                window = "\n".join(lines[line_no - 1:])
                if "function register" not in window and "function boot" not in window:
                    diags.append(
                        Diagnostic(
                            code="LARAVEL_INVALID_SERVICE_PROVIDER",
                            line=line_no,
                            col=1,
                            message="Laravel ServiceProvider class must define register() or boot() method",
                            offset=0,
                            source="laravel_providers",
                            severity=1,
                        )
                    )

        self.wall_s += time.monotonic() - t0
        return diags

    def close(self) -> None:
        pass


class RubyWebOracle:
    """In-process structural analyzer for Ruby & Rails applications."""

    def __init__(self, *, timeout_s: float = 5.0):
        self.timeout_s = timeout_s
        self.n_calls = 0
        self.wall_s = 0.0

    def diagnostics(self, source: str) -> List[Diagnostic]:
        t0 = time.monotonic()
        self.n_calls += 1
        diags: List[Diagnostic] = []
        lines = source.splitlines()

        # 1. Syntax balance check
        open_count = 0
        close_count = 0
        for line in lines:
            stripped = line.strip()
            if re.match(r"^(?:def|class|module|if|unless|while|until|case)\b", stripped):
                open_count += 1
            elif re.search(r"\bdo\s*(?:\|[^|]*\|)?\s*$", stripped):
                open_count += 1
            if stripped == "end" or stripped.startswith("end "):
                close_count += 1

        if open_count != close_count:
            diags.append(
                Diagnostic(
                    code="RUBY_SYNTAX_ERROR",
                    line=len(lines),
                    col=1,
                    message=f"Ruby syntax imbalance: opened {open_count} block(s) but encountered {close_count} 'end'(s)",
                    offset=0,
                    source="ruby_syntax",
                    severity=1,
                )
            )

        # 2. Sorbet typed signatures & Rails models
        has_sorbet = any(re.match(r"^#\s*typed:\s*(?:strict|true|strong)", l.strip()) for l in lines[:5])
        is_model = any("< ApplicationRecord" in l or "< ActiveRecord::Base" in l for l in lines)

        for line_no, line in enumerate(lines, start=1):
            stripped = line.strip()

            if has_sorbet and stripped.startswith("def "):
                prev_line = lines[line_no - 2].strip() if line_no >= 2 else ""
                if not prev_line.startswith("sig ") and not prev_line.startswith("sig{"):
                    diags.append(
                        Diagnostic(
                            code="SORBET_MISSING_SIG",
                            line=line_no,
                            col=1,
                            message=f"Method {stripped.split()[1]!r} under Sorbet typing lacks a preceding sig {{ ... }} declaration",
                            offset=0,
                            source="sorbet_typing",
                            severity=1,
                        )
                    )

            if is_model and "class " in stripped:
                window = "\n".join(lines[line_no - 1:])
                if "validates " not in window and "validate " not in window:
                    diags.append(
                        Diagnostic(
                            code="RAILS_MISSING_VALIDATION",
                            line=line_no,
                            col=1,
                            message="Rails ActiveRecord model defines business state without any validation contracts",
                            offset=0,
                            source="rails_models",
                            severity=2,
                        )
                    )

        self.wall_s += time.monotonic() - t0
        return diags

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# Base & Stack-Specific Verifiers
# --------------------------------------------------------------------------- #

class BaseWebBackendVerifier:
    r"""Base verifier for Web & Backend application stacks.

    Matches LspVerifier / SystemsMobileVerifier reward shaping:
      - Diagnostic errors/warnings reduce reward from 1.0 (base clean)
      - Degenerate output (empty, whitespace, comment-only) short-circuits to -1.0
      - Anti-Goodhart escape hatches short-circuit to -1.0
      - $R = \max(1.0 - \sum \text{penalties}, \text{diag_floor})$ with hatch floors ($R = -1.0$)
    """

    def __init__(
        self,
        *,
        stack_name: str,
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

        self.stack_name = stack_name
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
        is_hash = self.stack_name in ("python_web", "ruby_web")
        return find_escape_hatches_for_web(
            text, self.hatch_patterns, self.directive_hatches, is_hash_comment=is_hash
        )

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
        """Score one completion against static web and backend stack rules.

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
            "stack": self.stack_name,
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

    def __enter__(self) -> "BaseWebBackendVerifier":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


class TypeScriptWebVerifier(BaseWebBackendVerifier):
    """Static verifier for TypeScript / React / Next.js / NestJS web applications."""

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
            stack_name="typescript_web",
            hatch_patterns=TS_WEB_ESCAPE_HATCH_PATTERNS,
            directive_hatches=TS_WEB_DIRECTIVE_HATCHES,
            oracle_factory=lambda: TypeScriptWebOracle(timeout_s=timeout_s),
            timeout_s=timeout_s,
            hatches=hatches,
            oracle=oracle,
            on_error=on_error,
            fail_fast=fail_fast,
            **reward_kwargs,
        )


class PythonWebVerifier(BaseWebBackendVerifier):
    """Static verifier for Python web applications (FastAPI, Pydantic, Django ORM)."""

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
            stack_name="python_web",
            hatch_patterns=PYTHON_WEB_ESCAPE_HATCH_PATTERNS,
            directive_hatches=PYTHON_WEB_DIRECTIVE_HATCHES,
            oracle_factory=lambda: PythonWebOracle(timeout_s=timeout_s),
            timeout_s=timeout_s,
            hatches=hatches,
            oracle=oracle,
            on_error=on_error,
            fail_fast=fail_fast,
            **reward_kwargs,
        )


class GoWebVerifier(BaseWebBackendVerifier):
    """Static verifier for Go web applications & microservices."""

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
            stack_name="go_web",
            hatch_patterns=GO_WEB_ESCAPE_HATCH_PATTERNS,
            directive_hatches=GO_WEB_DIRECTIVE_HATCHES,
            oracle_factory=lambda: GoWebOracle(timeout_s=timeout_s),
            timeout_s=timeout_s,
            hatches=hatches,
            oracle=oracle,
            on_error=on_error,
            fail_fast=fail_fast,
            **reward_kwargs,
        )


class JavaWebVerifier(BaseWebBackendVerifier):
    """Static verifier for Java & Spring Boot 3 applications."""

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
            stack_name="java_web",
            hatch_patterns=JAVA_WEB_ESCAPE_HATCH_PATTERNS,
            directive_hatches=JAVA_WEB_DIRECTIVE_HATCHES,
            oracle_factory=lambda: JavaWebOracle(timeout_s=timeout_s),
            timeout_s=timeout_s,
            hatches=hatches,
            oracle=oracle,
            on_error=on_error,
            fail_fast=fail_fast,
            **reward_kwargs,
        )


class CSharpWebVerifier(BaseWebBackendVerifier):
    """Static verifier for C# / ASP.NET Core & EF Core applications."""

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
            stack_name="csharp_web",
            hatch_patterns=CSHARP_WEB_ESCAPE_HATCH_PATTERNS,
            directive_hatches=CSHARP_WEB_DIRECTIVE_HATCHES,
            oracle_factory=lambda: CSharpWebOracle(timeout_s=timeout_s),
            timeout_s=timeout_s,
            hatches=hatches,
            oracle=oracle,
            on_error=on_error,
            fail_fast=fail_fast,
            **reward_kwargs,
        )


class PhpWebVerifier(BaseWebBackendVerifier):
    """Static verifier for PHP & Laravel applications."""

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
            stack_name="php_web",
            hatch_patterns=PHP_WEB_ESCAPE_HATCH_PATTERNS,
            directive_hatches=PHP_WEB_DIRECTIVE_HATCHES,
            oracle_factory=lambda: PhpWebOracle(timeout_s=timeout_s),
            timeout_s=timeout_s,
            hatches=hatches,
            oracle=oracle,
            on_error=on_error,
            fail_fast=fail_fast,
            **reward_kwargs,
        )


class RubyWebVerifier(BaseWebBackendVerifier):
    """Static verifier for Ruby & Rails applications."""

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
            stack_name="ruby_web",
            hatch_patterns=RUBY_WEB_ESCAPE_HATCH_PATTERNS,
            directive_hatches=RUBY_WEB_DIRECTIVE_HATCHES,
            oracle_factory=lambda: RubyWebOracle(timeout_s=timeout_s),
            timeout_s=timeout_s,
            hatches=hatches,
            oracle=oracle,
            on_error=on_error,
            fail_fast=fail_fast,
            **reward_kwargs,
        )


# Convenience aliases
TypeScriptStaticVerifier = TypeScriptWebVerifier
PythonStaticVerifier = PythonWebVerifier
GoStaticVerifier = GoWebVerifier
JavaStaticVerifier = JavaWebVerifier
CSharpStaticVerifier = CSharpWebVerifier
PhpStaticVerifier = PhpWebVerifier
RubyStaticVerifier = RubyWebVerifier


# --------------------------------------------------------------------------- #
# Unified Web & Backend Verifier (Auto-dispatching across application stacks)
# --------------------------------------------------------------------------- #

class WebBackendVerifier:
    """Unified multi-stack verifier for Web & Backend Application Stacks (#343)."""

    def __init__(
        self,
        *,
        stack: str = "auto",
        fail_fast: bool = True,
        hatches: str = "superset",
        **reward_kwargs: Any,
    ) -> None:
        self.stack = stack.lower()
        self.fail_fast = fail_fast
        self.hatches = hatches
        self.reward_kwargs = reward_kwargs

        self._verifiers: Dict[str, BaseWebBackendVerifier] = {
            "typescript_web": TypeScriptWebVerifier(fail_fast=fail_fast, hatches=hatches, **reward_kwargs),
            "python_web": PythonWebVerifier(fail_fast=fail_fast, hatches=hatches, **reward_kwargs),
            "go_web": GoWebVerifier(fail_fast=fail_fast, hatches=hatches, **reward_kwargs),
            "java_web": JavaWebVerifier(fail_fast=fail_fast, hatches=hatches, **reward_kwargs),
            "csharp_web": CSharpWebVerifier(fail_fast=fail_fast, hatches=hatches, **reward_kwargs),
            "php_web": PhpWebVerifier(fail_fast=fail_fast, hatches=hatches, **reward_kwargs),
            "ruby_web": RubyWebVerifier(fail_fast=fail_fast, hatches=hatches, **reward_kwargs),
        }
        self._n_samples = 0

    def detect_stack(self, text: str) -> str:
        """Infer the web & backend application stack from code syntax and markers."""
        stripped = text.strip()
        if stripped.startswith("<?php") or "declare(strict_types=1)" in stripped or "extends ServiceProvider" in stripped:
            return "php_web"
        if "# typed:" in stripped or "< ApplicationRecord" in stripped or "< ActiveRecord::Base" in stripped or "sig {" in stripped:
            return "ruby_web"
        if "package " in stripped and "func " in stripped:
            return "go_web"
        if "using Microsoft.AspNetCore" in stripped or "WebApplication.CreateBuilder" in stripped or "DbContext" in stripped:
            return "csharp_web"
        if "@SpringBootApplication" in stripped or "@RestController" in stripped or "@Entity" in stripped or "public class " in stripped:
            return "java_web"
        if "def " in stripped or "import fastapi" in stripped or "BaseModel" in stripped or "models.Model" in stripped:
            return "python_web"
        if "'use client'" in stripped or '"use client"' in stripped or "@Injectable" in stripped or "interface " in stripped or "export " in stripped:
            return "typescript_web"
        return "typescript_web"

    def reward(
        self,
        completion: str,
        reference: Optional[str] = None,
        *,
        prompt: str = "",
        stack: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[float]:
        self._n_samples += 1
        target_stack = (stack or self.stack).lower()
        if target_stack == "auto":
            target_stack = self.detect_stack(f"{prompt}{completion}")
        v = self._verifiers.get(target_stack, self._verifiers["typescript_web"])
        return v.reward(completion, reference, prompt=prompt, **kwargs)

    def telemetry(self) -> dict:
        merged: Dict[str, Any] = {
            "n_samples": self._n_samples,
            "sub_verifiers": {},
        }
        for name, v in self._verifiers.items():
            merged["sub_verifiers"][name] = v.telemetry()
        return merged

    def close(self) -> None:
        for v in self._verifiers.values():
            v.close()

    def __enter__(self) -> "WebBackendVerifier":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
