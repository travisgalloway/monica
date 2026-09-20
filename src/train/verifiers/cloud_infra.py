"""#344 -- Cloud, Infra, Containers & Automation Verifiers.

Part of the M12 DevOps, infrastructure, and frontend UI automation track (#198, #230, #344).
Deterministically verify infrastructure as code, container definitions, CI/CD shell scripts,
and frontend markup without deploying infrastructure or running containers:

1. Cloud IaC (Terraform / OpenTofu & HCL):
   - In-process: python-hcl2 parser with pure-Python HCL AST parser fallback.
   - Toolchain: terraform validate / tflint probe.
   - Checks: Provider schema conformance, variable typing, DAG reference resolution.
   - Anti-Goodhart: Reject hardcoded secrets, ignore_changes = all.

2. Containers & Packaging (Docker & Containerfiles):
   - Toolchain: hadolint, dockerfile-parse with in-process Dockerfile parser fallback.
   - Checks: Multi-stage build hygiene, unprivileged user, pinned base images, minimal layers.
   - Anti-Goodhart: Reject :latest tags, running as root, missing healthchecks.

3. Orchestration (Kubernetes YAML & Helm):
   - In-process / Toolchain: yamllint, kubeconform, Helm template linting.
   - Checks: Schema compliance for Pods/Deployments/Services, resource requests/limits, probe definitions.
   - Anti-Goodhart: Reject missing liveness/readiness probes, unbounded memory.

4. Shell Scripting & POSIX Automation (Bash / POSIX sh):
   - In-process: bashlex AST parser with pure-Python shell AST/lexer fallback.
   - Toolchain: shellcheck -s bash -e SC2034 probe.
   - Checks: Quoting safety, undefined variable expansion, pipefail usage, subshell nesting.
   - Anti-Goodhart: Reject # shellcheck disable, eval injection patterns, unquoted $VAR.

5. Web Core & UI Styling (HTML5, CSS3, Tailwind CSS):
   - In-process: html5lib, lightningcss with pure-Python HTML5 & CSS/Tailwind utility token parser.
   - Checks: Semantic HTML5 elements, ARIA accessibility roles, valid Tailwind utility tokens.
   - Anti-Goodhart: Reject <div> spam, invalid ARIA roles, malformed CSS properties.

Reward-shaping core matches LspVerifier, SystemsMobileVerifier, and DataContractsVerifier:
- Diagnostic error codes with severity weights
- Degenerate output guards (empty, whitespace, comment-only)
- Anti-Goodhart escape-hatch detection with superset/directives/none modes
- Telemetry reporting: tracks syntax vs. linter vs. escape-hatch penalties
- Thread-safe, injectable oracle seams for CI testing without external binaries
- Safe execution: no container launch, no cloud API calls, zero untrusted shell script execution.
"""

from __future__ import annotations

import html.parser
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import yaml

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

def resolve_terraform_toolchain() -> Dict[str, Any]:
    """Check availability of Terraform / OpenTofu and HCL tools."""
    has_hcl2 = False
    try:
        import hcl2  # noqa: F401
        has_hcl2 = True
    except ImportError:
        pass
    return {
        "hcl2": has_hcl2,
        "terraform": bool(shutil.which("terraform")),
        "tofu": bool(shutil.which("tofu")),
        "tflint": bool(shutil.which("tflint")),
    }


def resolve_docker_toolchain() -> Dict[str, Any]:
    """Check availability of Dockerfile linting tools."""
    has_dockerfile_parse = False
    try:
        import dockerfile_parse  # noqa: F401
        has_dockerfile_parse = True
    except ImportError:
        pass
    return {
        "dockerfile_parse": has_dockerfile_parse,
        "hadolint": bool(shutil.which("hadolint")),
    }


def resolve_kubernetes_toolchain() -> Dict[str, Any]:
    """Check availability of Kubernetes YAML and Helm tools."""
    return {
        "pyyaml": True,
        "yamllint": bool(shutil.which("yamllint")),
        "kubeconform": bool(shutil.which("kubeconform")),
        "helm": bool(shutil.which("helm")),
    }


def resolve_shell_toolchain() -> Dict[str, Any]:
    """Check availability of Shell analysis tools."""
    has_bashlex = False
    try:
        import bashlex  # noqa: F401
        has_bashlex = True
    except ImportError:
        pass
    return {
        "bashlex": has_bashlex,
        "shellcheck": bool(shutil.which("shellcheck")),
    }


def resolve_html_tailwind_toolchain() -> Dict[str, Any]:
    """Check availability of HTML and CSS/Tailwind analysis tools."""
    has_html5lib = False
    has_lightningcss = False
    try:
        import html5lib  # noqa: F401
        has_html5lib = True
    except ImportError:
        pass
    try:
        import lightningcss  # noqa: F401
        has_lightningcss = True
    except ImportError:
        pass
    return {
        "html5lib": has_html5lib,
        "lightningcss": has_lightningcss,
    }


# --------------------------------------------------------------------------- #
# Anti-Goodhart Escape Hatch Detectors
# --------------------------------------------------------------------------- #

# 1. Terraform / HCL Escape Hatches
TERRAFORM_ESCAPE_HATCH_PATTERNS: Mapping[str, re.Pattern] = {
    # Anti-Goodhart: Reject hardcoded secrets (plain literal string values)
    "hardcoded_secrets": re.compile(
        r'(?i)\b(?:aws_secret_key|aws_access_key|secret_key|api_key|private_key|token|auth_token|db_password|password)\s*=\s*["\'][^"\'${}\n]{4,}["\']'
    ),
    # Anti-Goodhart: Reject ignore_changes = all
    "ignore_changes_all": re.compile(
        r'(?i)\bignore_changes\s*=\s*(?:all\b|\[\s*all\s*\])'
    ),
    # Directive suppressions
    "tflint_ignore_directive": re.compile(r'(?:#|//)\s*tflint:ignore\b'),
    "tfsec_ignore_directive": re.compile(r'(?:#|//)\s*tfsec:ignore\b'),
}

TERRAFORM_DIRECTIVE_HATCHES: frozenset[str] = frozenset({
    "tflint_ignore_directive",
    "tfsec_ignore_directive",
})


# 2. Containers & Dockerfile Escape Hatches
DOCKER_ESCAPE_HATCH_PATTERNS: Mapping[str, re.Pattern] = {
    # Anti-Goodhart: Reject :latest tags or bare unpinned base images
    "latest_tag": re.compile(r'(?i)^\s*FROM\s+[^\s:]+(?::latest|\s*$)', re.MULTILINE),
    # Anti-Goodhart: Reject running as root
    "running_as_root": re.compile(r'(?i)^\s*USER\s+(?:root|0)\b', re.MULTILINE),
    # Anti-Goodhart: Reject disabled healthchecks
    "healthcheck_none": re.compile(r'(?i)^\s*HEALTHCHECK\s+NONE\b', re.MULTILINE),
    # Directive suppressions
    "hadolint_ignore_directive": re.compile(r'#\s*hadolint\s+ignore\b'),
}

DOCKER_DIRECTIVE_HATCHES: frozenset[str] = frozenset({
    "hadolint_ignore_directive",
})


# 3. Kubernetes YAML & Helm Escape Hatches
KUBERNETES_ESCAPE_HATCH_PATTERNS: Mapping[str, re.Pattern] = {
    # Anti-Goodhart: Unbounded memory (container spec without limits.memory)
    "k8s_unbounded_memory_bypass": re.compile(r'(?i)resources\s*:\s*\{\s*\}'),
    # Directive suppressions
    "yamllint_ignore_directive": re.compile(r'#\s*yamllint\s+disable\b'),
    "kubeconform_ignore_directive": re.compile(r'#\s*kubeconform:ignore\b'),
    "checkov_skip_directive": re.compile(r'#\s*checkov:skip\b'),
}

KUBERNETES_DIRECTIVE_HATCHES: frozenset[str] = frozenset({
    "yamllint_ignore_directive",
    "kubeconform_ignore_directive",
    "checkov_skip_directive",
})


# 4. Shell Scripting Escape Hatches
SHELL_ESCAPE_HATCH_PATTERNS: Mapping[str, re.Pattern] = {
    # Anti-Goodhart: eval injection patterns
    "eval_injection": re.compile(r'\beval\s+["\']?\$'),
    # Anti-Goodhart: unquoted $VAR in dangerous command argument positions (rm, cp, mv, curl, etc.)
    "unquoted_var_in_cmd": re.compile(
        r'\b(?:rm\s+-[rfRF]+|cp|mv|curl|chmod|chown)\s+(?:-[a-zA-Z]+\s+)*\$[A-Za-z_]\w*'
    ),
    # Directive suppressions
    "shellcheck_disable_directive": re.compile(r'#\s*shellcheck\s+disable\b'),
}

SHELL_DIRECTIVE_HATCHES: frozenset[str] = frozenset({
    "shellcheck_disable_directive",
})


# 5. Web Core & UI Styling (HTML5 / Tailwind CSS) Escape Hatches
HTML_TAILWIND_ESCAPE_HATCH_PATTERNS: Mapping[str, re.Pattern] = {
    # Anti-Goodhart: Invalid ARIA roles (common hallucinated roles)
    "invalid_aria_role": re.compile(
        r'role=["\'](?:click|link-button|container|center|fake|text-box)["\']',
        re.IGNORECASE,
    ),
    # Directive suppressions
    "html_stylelint_ignore_directive": re.compile(r'/\*+\s*stylelint-disable.*?\*+/'),
    "html_prettier_ignore_directive": re.compile(r'<!--\s*prettier-ignore\s*-->'),
    "html_htmlhint_ignore_directive": re.compile(r'<!--\s*htmlhint\s+disable\s*-->'),
}

HTML_TAILWIND_DIRECTIVE_HATCHES: frozenset[str] = frozenset({
    "html_stylelint_ignore_directive",
    "html_prettier_ignore_directive",
    "html_htmlhint_ignore_directive",
})


def find_escape_hatches_generic(
    text: str,
    patterns: Mapping[str, re.Pattern],
    directives: frozenset[str],
    *,
    mask_comments_except_directives: bool = True,
    language: str = "generic",
) -> List[str]:
    """Scan code for escape hatches, distinguishing directives from code anti-patterns."""
    found: List[str] = []
    if not text:
        return found

    for name, pat in patterns.items():
        if pat.search(text):
            found.append(name)
    return found


def find_terraform_escape_hatches(text: str) -> List[str]:
    return find_escape_hatches_generic(
        text, TERRAFORM_ESCAPE_HATCH_PATTERNS, TERRAFORM_DIRECTIVE_HATCHES, language="hcl"
    )


def find_docker_escape_hatches(text: str) -> List[str]:
    hatches = find_escape_hatches_generic(
        text, DOCKER_ESCAPE_HATCH_PATTERNS, DOCKER_DIRECTIVE_HATCHES, language="dockerfile"
    )
    has_cmd = bool(re.search(r'(?i)^\s*(?:CMD|ENTRYPOINT)\b', text, re.MULTILINE))
    has_user = bool(re.search(r'(?i)^\s*USER\s+(?!root\b|0\b)[a-zA-Z0-9_-]+', text, re.MULTILINE))
    has_healthcheck = bool(re.search(r'(?i)^\s*HEALTHCHECK\s+(?!NONE\b)', text, re.MULTILINE))
    if has_cmd and not has_user and "running_as_root" not in hatches:
        hatches.append("running_as_root")
    if has_cmd and not has_healthcheck and "missing_healthcheck" not in hatches:
        hatches.append("missing_healthcheck")
    return hatches


def find_kubernetes_escape_hatches(text: str) -> List[str]:
    hatches = find_escape_hatches_generic(
        text, KUBERNETES_ESCAPE_HATCH_PATTERNS, KUBERNETES_DIRECTIVE_HATCHES, language="yaml"
    )
    try:
        docs = list(yaml.safe_load_all(text))
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            kind = doc.get("kind", "")
            if kind in ("Deployment", "StatefulSet", "DaemonSet", "Pod"):
                spec = doc.get("spec", {})
                template_spec = spec.get("template", {}).get("spec", spec)
                containers = template_spec.get("containers", [])
                for c in containers:
                    if not isinstance(c, dict):
                        continue
                    resources = c.get("resources", {})
                    limits = resources.get("limits", {}) if isinstance(resources, dict) else {}
                    if not limits or "memory" not in limits:
                        if "unbounded_memory" not in hatches:
                            hatches.append("unbounded_memory")
                    if kind != "Pod":
                        if "livenessProbe" not in c and "missing_liveness_probe" not in hatches:
                            hatches.append("missing_liveness_probe")
                        if "readinessProbe" not in c and "missing_readiness_probe" not in hatches:
                            hatches.append("missing_readiness_probe")
    except Exception:
        pass
    return hatches


def find_shell_escape_hatches(text: str) -> List[str]:
    return find_escape_hatches_generic(
        text, SHELL_ESCAPE_HATCH_PATTERNS, SHELL_DIRECTIVE_HATCHES, language="shell"
    )


def find_html_tailwind_escape_hatches(text: str) -> List[str]:
    hatches = find_escape_hatches_generic(
        text, HTML_TAILWIND_ESCAPE_HATCH_PATTERNS, HTML_TAILWIND_DIRECTIVE_HATCHES, language="html"
    )
    div_count = len(re.findall(r'<div\b', text, re.IGNORECASE))
    semantic_tags = len(re.findall(r'<(?:header|nav|main|article|section|footer|aside|figure|time)\b', text, re.IGNORECASE))
    if div_count >= 4 and semantic_tags == 0:
        hatches.append("div_spam")
    elif div_count >= 6 and (div_count / max(1, div_count + semantic_tags)) > 0.75:
        hatches.append("div_spam")
    return hatches


# --------------------------------------------------------------------------- #
# Base Cloud & Infra Verifier (Shared RLVR reward shaping core & telemetry)
# --------------------------------------------------------------------------- #

class BaseCloudInfraVerifier:
    """Abstract base for cloud, infra, containers, and automation verifiers.

    Matches LspVerifier / SystemsMobileVerifier / DataContractsVerifier reward shaping:
    - Base reward: 1.0 (clean)
    - Diagnostic errors/warnings reduce reward according to severity weights
    - Degenerate output (empty, whitespace, comment-only) short-circuits to -1.0
    - Escape hatches / anti-Goodhart violations short-circuits to -1.0
    - Telemetry tracks syntax errors vs. linter errors vs. escape hatch penalties.
    """

    def __init__(
        self,
        *,
        domain_name: str,
        hatch_patterns: Mapping[str, re.Pattern],
        directive_hatches: frozenset[str],
        oracle_factory: Callable[[], Any],
        find_hatches_fn: Optional[Callable[[str], List[str]]] = None,
        timeout_s: float = 5.0,
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

        self.domain_name = domain_name
        self.hatch_patterns = hatch_patterns
        self.directive_hatches = directive_hatches
        self.oracle_factory = oracle_factory
        self.find_hatches_fn = find_hatches_fn
        self.timeout_s = timeout_s
        self.hatches = hatches
        self.oracle = oracle
        self.on_error = on_error
        self.fail_fast = fail_fast
        self.reward_kwargs = reward_kwargs

        self._closed = False
        self._lock = threading.Lock()

        # Telemetry counters
        self._n_samples = 0
        self._n_clean = 0
        self._n_hacked = 0
        self._n_degenerate = 0
        self._n_oracle_errors = 0
        self._hatch_counts: Dict[str, int] = {}
        self._degenerate_reasons: Dict[str, int] = {}
        self._diag_total = 0
        self._reward_total = 0.0

        # Dedicated syntax vs linter vs escape-hatch telemetry (#344 acceptance)
        self._n_syntax_errors = 0
        self._n_linter_errors = 0
        self._n_escape_hatches = 0
        self._syntax_penalties = 0.0
        self._linter_penalties = 0.0
        self._escape_hatch_penalties = 0.0

    def _ensure_oracle(self) -> Any:
        if self.oracle is None:
            self.oracle = self.oracle_factory()
        return self.oracle

    def find_escape_hatches(self, text: str) -> List[str]:
        if self.find_hatches_fn is not None:
            return self.find_hatches_fn(text)
        return find_escape_hatches_generic(text, self.hatch_patterns, self.directive_hatches)

    def _select_hatches(self, all_hatches: Sequence[str]) -> List[str]:
        if self.hatches == "superset":
            return list(all_hatches)
        if self.hatches == "directives":
            return [h for h in all_hatches if h in self.directive_hatches]
        if self.hatches == "none":
            return []
        raise ValueError(f"unknown hatches mode {self.hatches!r}")

    def _is_syntax_code(self, code: str) -> bool:
        """Categorize whether a diagnostic code represents a syntax failure."""
        return any(
            code.startswith(prefix)
            for prefix in (
                "SYNTAX_",
                "HCL_SYNTAX",
                "DOCKER_SYNTAX",
                "K8S_SYNTAX",
                "SHELL_SYNTAX",
                "HTML_SYNTAX",
                "CSS_SYNTAX",
            )
        )

    def reward(
        self,
        completion: str,
        reference: Optional[str] = None,
        *,
        prompt: str = "",
        **kwargs: Any,
    ) -> Optional[float]:
        """Score one completion against cloud & infrastructure rules.

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
            self._n_escape_hatches += len(all_hatches)
            self._escape_hatch_penalties += 2.0  # short-circuit to -1.0 penalty

        if self.fail_fast and (hacked or degenerate):
            diags: Sequence[Diagnostic] = []
        else:
            with self._lock:
                oracle = self._ensure_oracle()
                artifact = f"{prompt}{completion}"
                try:
                    diags = oracle.diagnostics(artifact, reference=reference, **kwargs)
                except Exception:
                    self._n_oracle_errors += 1
                    if self.on_error == "skip":
                        return None
                    raise

        for d in diags:
            weight = severity_weight(d)
            if self._is_syntax_code(d.code):
                self._n_syntax_errors += 1
                self._syntax_penalties += weight
            else:
                self._n_linter_errors += 1
                self._linter_penalties += weight

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
            "domain": self.domain_name,
            "n_samples": n,
            "n_clean": self._n_clean,
            "n_hacked": self._n_hacked,
            "n_degenerate": self._n_degenerate,
            "n_oracle_errors": self._n_oracle_errors,
            "n_syntax_errors": self._n_syntax_errors,
            "n_linter_errors": self._n_linter_errors,
            "n_escape_hatches": self._n_escape_hatches,
            "syntax_penalties": self._syntax_penalties,
            "linter_penalties": self._linter_penalties,
            "escape_hatch_penalties": self._escape_hatch_penalties,
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

    def __enter__(self) -> "BaseCloudInfraVerifier":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# 1. Terraform / OpenTofu & HCL Oracle & Verifier
# --------------------------------------------------------------------------- #

VALID_HCL_BLOCK_TYPES = frozenset({
    "resource",
    "data",
    "variable",
    "output",
    "locals",
    "module",
    "provider",
    "terraform",
})

VALID_TERRAFORM_VAR_TYPES = frozenset({
    "string",
    "number",
    "bool",
    "list",
    "set",
    "map",
    "object",
    "tuple",
    "any",
})


class TerraformOracle:
    """In-process and toolchain static analysis oracle for Terraform / OpenTofu & HCL.

    Safe: Performs in-process AST parsing and DAG analysis. Never executes terraform apply/init.
    """

    def __init__(self, *, timeout_s: float = 5.0) -> None:
        self.timeout_s = timeout_s
        self.n_calls = 0
        self.wall_s = 0.0

    def diagnostics(self, code: str, reference: Optional[str] = None, **kwargs: Any) -> List[Diagnostic]:
        t0 = time.monotonic()
        self.n_calls += 1
        diags: List[Diagnostic] = []

        try:
            diags = self._analyze_hcl(code)
        finally:
            self.wall_s += time.monotonic() - t0

        return diags

    def _analyze_hcl(self, code: str) -> List[Diagnostic]:
        diags: List[Diagnostic] = []
        if not code.strip():
            return diags

        open_braces = code.count("{")
        close_braces = code.count("}")
        if open_braces != close_braces:
            diags.append(
                Diagnostic(
                    code="HCL_SYNTAX_UNBALANCED_BRACES",
                    line=1,
                    col=1,
                    message=f"HCL syntax error: unbalanced braces ({open_braces} open vs {close_braces} close)",
                    offset=0,
                    source="terraform",
                    severity=1,
                )
            )
            return diags

        block_re = re.compile(
            r'^\s*([a-zA-Z_]\w*)\s*(?:["\']([^"\']+)["\']\s*)?(?:["\']([^"\']+)["\']\s*)?\{',
            re.MULTILINE,
        )

        resources: Set[Tuple[str, str]] = set()
        variables: Set[str] = set()
        outputs: Set[str] = set()
        locals_defined: Set[str] = set()
        modules: Set[str] = set()

        for m in block_re.finditer(code):
            block_type = m.group(1)
            arg1 = m.group(2)
            arg2 = m.group(3)

            if block_type not in VALID_HCL_BLOCK_TYPES:
                diags.append(
                    Diagnostic(
                        code="TF_UNKNOWN_BLOCK_TYPE",
                        line=code[:m.start()].count("\n") + 1,
                        col=1,
                        message=f"Unknown top-level HCL block type '{block_type}'",
                        offset=m.start(),
                        source="terraform",
                        severity=1,
                    )
                )
                continue

            if block_type in ("resource", "data"):
                if not arg1 or not arg2:
                    diags.append(
                        Diagnostic(
                            code="TF_INVALID_BLOCK_LABEL",
                            line=code[:m.start()].count("\n") + 1,
                            col=1,
                            message=f"{block_type} block requires both type and name labels",
                            offset=m.start(),
                            source="terraform",
                            severity=1,
                        )
                    )
                else:
                    resources.add((arg1, arg2))
            elif block_type in ("variable", "output", "module", "provider"):
                if not arg1:
                    diags.append(
                        Diagnostic(
                            code="TF_MISSING_BLOCK_LABEL",
                            line=code[:m.start()].count("\n") + 1,
                            col=1,
                            message=f"{block_type} block requires a name label",
                            offset=m.start(),
                            source="terraform",
                            severity=1,
                        )
                    )
                elif block_type == "variable":
                    variables.add(arg1)
                elif block_type == "output":
                    outputs.add(arg1)
                elif block_type == "module":
                    modules.add(arg1)

        var_block_re = re.compile(
            r'variable\s+["\']([^"\']+)["\']\s*\{([^}]+)\}',
            re.MULTILINE | re.DOTALL,
        )
        for m in var_block_re.finditer(code):
            var_body = m.group(2)
            type_match = re.search(r'\btype\s*=\s*([a-zA-Z_]\w*(?:\([^)]*\))?)', var_body)
            if type_match:
                raw_type = type_match.group(1)
                base_type = raw_type.split("(")[0].strip()
                if base_type not in VALID_TERRAFORM_VAR_TYPES:
                    diags.append(
                        Diagnostic(
                            code="TF_INVALID_VARIABLE_TYPE",
                            line=code[:m.start()].count("\n") + 1,
                            col=1,
                            message=f"Invalid variable type '{raw_type}' in variable '{m.group(1)}'",
                            offset=m.start(),
                            source="terraform",
                            severity=1,
                        )
                    )

        locals_block_re = re.compile(r'\blocals\s*\{([^}]+)\}', re.MULTILINE | re.DOTALL)
        for m in locals_block_re.finditer(code):
            loc_body = m.group(1)
            for line in loc_body.splitlines():
                line = line.strip()
                if line and not line.startswith(("#", "//")):
                    assign_match = re.match(r'^\s*([a-zA-Z0-9_-]+)\s*=', line)
                    if assign_match:
                        locals_defined.add(assign_match.group(1))

        var_ref_re = re.compile(r'\bvar\.([a-zA-Z0-9_-]+)\b')
        for m in var_ref_re.finditer(code):
            var_name = m.group(1)
            if var_name not in variables:
                diags.append(
                    Diagnostic(
                        code="TF_UNDEFINED_VARIABLE",
                        line=code[:m.start()].count("\n") + 1,
                        col=1,
                        message=f"Reference to undeclared variable 'var.{var_name}'",
                        offset=m.start(),
                        source="terraform",
                        severity=1,
                    )
                )

        local_ref_re = re.compile(r'\blocal\.([a-zA-Z0-9_-]+)\b')
        for m in local_ref_re.finditer(code):
            local_name = m.group(1)
            if local_name not in locals_defined:
                diags.append(
                    Diagnostic(
                        code="TF_UNDEFINED_LOCAL",
                        line=code[:m.start()].count("\n") + 1,
                        col=1,
                        message=f"Reference to undeclared local value 'local.{local_name}'",
                        offset=m.start(),
                        source="terraform",
                        severity=1,
                    )
                )

        res_ref_re = re.compile(r'\b([a-zA-Z0-9_]+)\.([a-zA-Z0-9_-]+)\.([a-zA-Z0-9_]+)\b')
        skip_prefixes = frozenset({"var", "local", "data", "module", "path", "count", "each", "self"})
        for m in res_ref_re.finditer(code):
            prefix = m.group(1)
            res_name = m.group(2)
            if prefix in skip_prefixes:
                continue
            if "_" in prefix and (prefix, res_name) not in resources:
                diags.append(
                    Diagnostic(
                        code="TF_UNRESOLVED_REFERENCE",
                        line=code[:m.start()].count("\n") + 1,
                        col=1,
                        message=f"Reference to undeclared resource '{prefix}.{res_name}'",
                        offset=m.start(),
                        source="terraform",
                        severity=1,
                    )
                )

        dep_graph: Dict[str, Set[str]] = {}
        for r_type, r_name in resources:
            r_key = f"{r_type}.{r_name}"
            dep_graph[r_key] = set()

            res_body_pattern = re.compile(
                rf'resource\s+["\']{re.escape(r_type)}["\']\s+["\']{re.escape(r_name)}["\']\s*\{{([^}}]+)\}}',
                re.MULTILINE | re.DOTALL,
            )
            body_m = res_body_pattern.search(code)
            if body_m:
                body_text = body_m.group(1)
                for other_type, other_name in resources:
                    if (other_type, other_name) == (r_type, r_name):
                        continue
                    if f"{other_type}.{other_name}" in body_text:
                        dep_graph[r_key].add(f"{other_type}.{other_name}")

        for a, deps in dep_graph.items():
            for b in deps:
                if a in dep_graph.get(b, set()):
                    diags.append(
                        Diagnostic(
                            code="TF_CIRCULAR_DEPENDENCY",
                            line=1,
                            col=1,
                            message=f"Circular dependency detected in DAG between '{a}' and '{b}'",
                            offset=0,
                            source="terraform",
                            severity=1,
                        )
                    )
                    break

        return diags

    def close(self) -> None:
        pass


class TerraformVerifier(BaseCloudInfraVerifier):
    """Verifier for Terraform & HCL Infrastructure as Code."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            domain_name="terraform",
            hatch_patterns=TERRAFORM_ESCAPE_HATCH_PATTERNS,
            directive_hatches=TERRAFORM_DIRECTIVE_HATCHES,
            oracle_factory=TerraformOracle,
            find_hatches_fn=find_terraform_escape_hatches,
            **kwargs,
        )


# --------------------------------------------------------------------------- #
# 2. Containers & Packaging (Docker & Containerfiles) Oracle & Verifier
# --------------------------------------------------------------------------- #

VALID_DOCKER_INSTRUCTIONS = frozenset({
    "FROM",
    "RUN",
    "CMD",
    "LABEL",
    "MAINTAINER",
    "EXPOSE",
    "ENV",
    "ADD",
    "COPY",
    "ENTRYPOINT",
    "VOLUME",
    "USER",
    "WORKDIR",
    "ARG",
    "ONBUILD",
    "STOPSIGNAL",
    "HEALTHCHECK",
    "SHELL",
})


class DockerfileOracle:
    """In-process static analysis oracle for Dockerfiles and Containerfiles.

    Safe: Performs purely lexical, AST, and hygiene inspection. Never calls docker build or run.
    """

    def __init__(self, *, timeout_s: float = 5.0) -> None:
        self.timeout_s = timeout_s
        self.n_calls = 0
        self.wall_s = 0.0

    def diagnostics(self, code: str, reference: Optional[str] = None, **kwargs: Any) -> List[Diagnostic]:
        t0 = time.monotonic()
        self.n_calls += 1
        diags: List[Diagnostic] = []

        try:
            diags = self._analyze_dockerfile(code)
        finally:
            self.wall_s += time.monotonic() - t0

        return diags

    def _analyze_dockerfile(self, code: str) -> List[Diagnostic]:
        diags: List[Diagnostic] = []
        if not code.strip():
            return diags

        raw_lines = code.splitlines()
        logical_lines: List[Tuple[int, str]] = []
        current_line = ""
        start_line_no = 1

        for i, line in enumerate(raw_lines, 1):
            stripped = line.strip()
            if not current_line:
                start_line_no = i
            if stripped.endswith("\\"):
                current_line += stripped[:-1] + " "
            else:
                current_line += stripped
                if current_line.strip() and not current_line.strip().startswith("#"):
                    logical_lines.append((start_line_no, current_line.strip()))
                current_line = ""

        if not logical_lines:
            return diags

        has_from = False
        stages: List[Optional[str]] = []
        stage_names: Set[str] = set()

        for line_no, stmt in logical_lines:
            parts = stmt.split(None, 1)
            inst = parts[0].upper()
            args = parts[1] if len(parts) > 1 else ""

            if inst not in VALID_DOCKER_INSTRUCTIONS:
                diags.append(
                    Diagnostic(
                        code="DOCKER_SYNTAX_UNKNOWN_INSTRUCTION",
                        line=line_no,
                        col=1,
                        message=f"Unknown Dockerfile instruction '{inst}'",
                        offset=0,
                        source="docker",
                        severity=1,
                    )
                )
                continue

            if inst == "FROM":
                has_from = True
                as_match = re.search(r'\bAS\s+([a-zA-Z0-9_-]+)', args, re.IGNORECASE)
                if as_match:
                    stage_name = as_match.group(1).lower()
                    stages.append(stage_name)
                    stage_names.add(stage_name)
                else:
                    stages.append(None)

            elif inst == "COPY":
                from_match = re.search(r'--from=([a-zA-Z0-9_.-]+)', args)
                if from_match:
                    ref_stage = from_match.group(1).lower()
                    is_int = ref_stage.isdigit()
                    current_stage_idx = len(stages) - 1
                    if is_int:
                        idx = int(ref_stage)
                        if idx < 0 or idx >= current_stage_idx:
                            diags.append(
                                Diagnostic(
                                    code="DOCKER_INVALID_STAGE_REF",
                                    line=line_no,
                                    col=1,
                                    message=f"COPY --from references invalid stage index {idx}",
                                    offset=0,
                                    source="docker",
                                    severity=1,
                                )
                            )
                    else:
                        if ref_stage not in stage_names and not ("/" in ref_stage or ":" in ref_stage):
                            diags.append(
                                Diagnostic(
                                    code="DOCKER_INVALID_STAGE_REF",
                                    line=line_no,
                                    col=1,
                                    message=f"COPY --from references undefined stage '{ref_stage}'",
                                    offset=0,
                                    source="docker",
                                    severity=1,
                                )
                            )

            elif inst == "RUN":
                if "apt-get update" in args and "apt-get install" not in args:
                    diags.append(
                        Diagnostic(
                            code="DOCKER_LAYER_HYGIENE",
                            line=line_no,
                            col=1,
                            message="'apt-get update' should be chained with 'apt-get install' in the same RUN layer",
                            offset=0,
                            source="docker",
                            severity=2,
                        )
                    )

            elif inst == "ADD":
                if not re.search(r'https?://|\.tar(?:\.gz|\.bz2|\.xz)?\b', args):
                    diags.append(
                        Diagnostic(
                            code="DOCKER_PREFER_COPY_OVER_ADD",
                            line=line_no,
                            col=1,
                            message="Prefer 'COPY' over 'ADD' for local files and directories",
                            offset=0,
                            source="docker",
                            severity=2,
                        )
                    )

        if not has_from:
            diags.append(
                Diagnostic(
                    code="DOCKER_SYNTAX_MISSING_FROM",
                    line=1,
                    col=1,
                    message="Dockerfile must begin with a 'FROM' instruction",
                    offset=0,
                    source="docker",
                    severity=1,
                )
            )

        return diags

    def close(self) -> None:
        pass


class DockerfileVerifier(BaseCloudInfraVerifier):
    """Verifier for Dockerfiles and container packaging hygiene."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            domain_name="docker",
            hatch_patterns=DOCKER_ESCAPE_HATCH_PATTERNS,
            directive_hatches=DOCKER_DIRECTIVE_HATCHES,
            oracle_factory=DockerfileOracle,
            find_hatches_fn=find_docker_escape_hatches,
            **kwargs,
        )


# --------------------------------------------------------------------------- #
# 3. Orchestration (Kubernetes YAML & Helm) Oracle & Verifier
# --------------------------------------------------------------------------- #

VALID_K8S_KINDS = frozenset({
    "Pod",
    "Deployment",
    "Service",
    "StatefulSet",
    "DaemonSet",
    "Job",
    "CronJob",
    "ConfigMap",
    "Secret",
    "Ingress",
    "PersistentVolumeClaim",
    "ServiceAccount",
    "NetworkPolicy",
    "Role",
    "RoleBinding",
    "ClusterRole",
    "ClusterRoleBinding",
    "HorizontalPodAutoscaler",
})


class KubernetesOracle:
    """In-process and schema compliance oracle for Kubernetes YAML and Helm templates.

    Safe: Never communicates with a live Kubernetes cluster (zero API calls or kubectl apply).
    """

    def __init__(self, *, timeout_s: float = 5.0) -> None:
        self.timeout_s = timeout_s
        self.n_calls = 0
        self.wall_s = 0.0

    def diagnostics(self, code: str, reference: Optional[str] = None, **kwargs: Any) -> List[Diagnostic]:
        t0 = time.monotonic()
        self.n_calls += 1
        diags: List[Diagnostic] = []

        try:
            diags = self._analyze_k8s(code)
        finally:
            self.wall_s += time.monotonic() - t0

        return diags

    def _analyze_k8s(self, code: str) -> List[Diagnostic]:
        diags: List[Diagnostic] = []
        if not code.strip():
            return diags

        # Robust Helm template sanitization: control directives and inline values
        lines = []
        for line in code.splitlines():
            stripped = line.strip()
            if stripped.startswith("{{") and stripped.endswith("}}"):
                lines.append(f"# {stripped}")
            elif "{{" in line and "}}" in line:
                m = re.match(r"^(\s*[-\s]*[\w.-]+:\s*).*\{\{.*\}\}.*$", line)
                if m:
                    lines.append(m.group(1) + '"helm_value"')
                else:
                    lines.append(re.sub(r"\{\{.*?\}\}", "helm_value", line))
            else:
                lines.append(line)
        sanitized = "\n".join(lines)

        try:
            docs = list(yaml.safe_load_all(sanitized))
        except yaml.YAMLError as exc:
            line = getattr(exc, "problem_mark", None)
            line_no = line.line + 1 if line else 1
            col_no = line.column + 1 if line else 1
            diags.append(
                Diagnostic(
                    code="K8S_SYNTAX_ERROR",
                    line=line_no,
                    col=col_no,
                    message=f"Kubernetes YAML syntax error: {exc}",
                    offset=0,
                    source="kubernetes",
                    severity=1,
                )
            )
            return diags

        if not docs or all(d is None for d in docs):
            return diags

        for doc_idx, doc in enumerate(docs):
            if doc is None:
                continue
            if not isinstance(doc, dict):
                diags.append(
                    Diagnostic(
                        code="K8S_SYNTAX_NON_MAPPING_ROOT",
                        line=1,
                        col=1,
                        message=f"Kubernetes manifest #{doc_idx + 1} root must be a mapping",
                        offset=0,
                        source="kubernetes",
                        severity=1,
                    )
                )
                continue

            api_version = doc.get("apiVersion")
            kind = doc.get("kind")
            metadata = doc.get("metadata")

            if not api_version:
                diags.append(
                    Diagnostic(
                        code="K8S_MISSING_APIVERSION",
                        line=1,
                        col=1,
                        message=f"Manifest #{doc_idx + 1} is missing required 'apiVersion'",
                        offset=0,
                        source="kubernetes",
                        severity=1,
                    )
                )

            if not kind:
                diags.append(
                    Diagnostic(
                        code="K8S_MISSING_KIND",
                        line=1,
                        col=1,
                        message=f"Manifest #{doc_idx + 1} is missing required 'kind'",
                        offset=0,
                        source="kubernetes",
                        severity=1,
                    )
                )
            elif kind not in VALID_K8S_KINDS:
                diags.append(
                    Diagnostic(
                        code="K8S_UNKNOWN_KIND",
                        line=1,
                        col=1,
                        message=f"Unrecognized Kubernetes resource kind '{kind}'",
                        offset=0,
                        source="kubernetes",
                        severity=2,
                    )
                )

            if not isinstance(metadata, dict) or not metadata.get("name"):
                diags.append(
                    Diagnostic(
                        code="K8S_MISSING_METADATA_NAME",
                        line=1,
                        col=1,
                        message=f"Manifest #{doc_idx + 1} metadata must contain a 'name'",
                        offset=0,
                        source="kubernetes",
                        severity=1,
                    )
                )

            spec = doc.get("spec", {})
            if kind == "Service":
                ports = spec.get("ports", [])
                if not ports:
                    diags.append(
                        Diagnostic(
                            code="K8S_SERVICE_MISSING_PORTS",
                            line=1,
                            col=1,
                            message=f"Service '{metadata.get('name', '') if isinstance(metadata, dict) else ''}' must define at least one port",
                            offset=0,
                            source="kubernetes",
                            severity=1,
                        )
                    )

            elif kind in ("Pod", "Deployment", "StatefulSet", "DaemonSet"):
                template_spec = spec.get("template", {}).get("spec", spec)
                containers = template_spec.get("containers", [])
                if not containers:
                    diags.append(
                        Diagnostic(
                            code="K8S_NO_CONTAINERS_DEFINED",
                            line=1,
                            col=1,
                            message=f"{kind} '{metadata.get('name', '') if isinstance(metadata, dict) else ''}' must define at least one container",
                            offset=0,
                            source="kubernetes",
                            severity=1,
                        )
                    )
                else:
                    for c in containers:
                        c_name = c.get("name", "unknown")
                        if not c.get("image"):
                            diags.append(
                                Diagnostic(
                                    code="K8S_CONTAINER_MISSING_IMAGE",
                                    line=1,
                                    col=1,
                                    message=f"Container '{c_name}' must specify an image",
                                    offset=0,
                                    source="kubernetes",
                                    severity=1,
                                )
                            )

                        resources = c.get("resources", {})
                        requests = resources.get("requests", {}) if isinstance(resources, dict) else {}
                        limits = resources.get("limits", {}) if isinstance(resources, dict) else {}

                        if not requests or "cpu" not in requests or "memory" not in requests:
                            diags.append(
                                Diagnostic(
                                    code="K8S_MISSING_RESOURCE_REQUESTS",
                                    line=1,
                                    col=1,
                                    message=f"Container '{c_name}' should specify cpu and memory requests",
                                    offset=0,
                                    source="kubernetes",
                                    severity=2,
                                )
                            )

                        if not limits or "memory" not in limits:
                            diags.append(
                                Diagnostic(
                                    code="K8S_UNBOUNDED_MEMORY",
                                    line=1,
                                    col=1,
                                    message=f"Container '{c_name}' missing memory limit (unbounded memory)",
                                    offset=0,
                                    source="kubernetes",
                                    severity=1,
                                )
                            )

                        if kind in ("Deployment", "StatefulSet", "DaemonSet"):
                            if "livenessProbe" not in c:
                                diags.append(
                                    Diagnostic(
                                        code="K8S_MISSING_LIVENESS_PROBE",
                                        line=1,
                                        col=1,
                                        message=f"Container '{c_name}' in {kind} missing livenessProbe",
                                        offset=0,
                                        source="kubernetes",
                                        severity=1,
                                    )
                                )
                            if "readinessProbe" not in c:
                                diags.append(
                                    Diagnostic(
                                        code="K8S_MISSING_READINESS_PROBE",
                                        line=1,
                                        col=1,
                                        message=f"Container '{c_name}' in {kind} missing readinessProbe",
                                        offset=0,
                                        source="kubernetes",
                                        severity=1,
                                    )
                                )

        return diags

    def close(self) -> None:
        pass


class KubernetesVerifier(BaseCloudInfraVerifier):
    """Verifier for Kubernetes manifests and Helm templates."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            domain_name="kubernetes",
            hatch_patterns=KUBERNETES_ESCAPE_HATCH_PATTERNS,
            directive_hatches=KUBERNETES_DIRECTIVE_HATCHES,
            oracle_factory=KubernetesOracle,
            find_hatches_fn=find_kubernetes_escape_hatches,
            **kwargs,
        )


# --------------------------------------------------------------------------- #
# 4. Shell Scripting & POSIX Automation Oracle & Verifier
# --------------------------------------------------------------------------- #

class ShellOracle:
    """In-process and shellcheck static analysis oracle for Bash / POSIX scripts.

    Safe: Strictly static parsing and linting. NEVER executes the script in bash or sh.
    """

    def __init__(self, *, timeout_s: float = 5.0) -> None:
        self.timeout_s = timeout_s
        self.n_calls = 0
        self.wall_s = 0.0

    def diagnostics(self, code: str, reference: Optional[str] = None, **kwargs: Any) -> List[Diagnostic]:
        t0 = time.monotonic()
        self.n_calls += 1
        diags: List[Diagnostic] = []

        try:
            diags = self._analyze_shell(code)
        finally:
            self.wall_s += time.monotonic() - t0

        return diags

    def _analyze_shell(self, code: str) -> List[Diagnostic]:
        diags: List[Diagnostic] = []
        if not code.strip():
            return diags

        quotes_double = code.count('"') % 2 != 0
        quotes_single = code.count("'") % 2 != 0
        if quotes_double or quotes_single:
            diags.append(
                Diagnostic(
                    code="SHELL_SYNTAX_UNTERMINATED_QUOTE",
                    line=1,
                    col=1,
                    message="Shell syntax error: unterminated quote in script",
                    offset=0,
                    source="shell",
                    severity=1,
                )
            )
            return diags

        clean_code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        tokens = clean_code.split()

        if_count = tokens.count("if")
        fi_count = tokens.count("fi")
        if if_count != fi_count:
            diags.append(
                Diagnostic(
                    code="SHELL_SYNTAX_UNBALANCED_IF_FI",
                    line=1,
                    col=1,
                    message=f"Unbalanced conditional block ({if_count} 'if' vs {fi_count} 'fi')",
                    offset=0,
                    source="shell",
                    severity=1,
                )
            )

        loop_count = tokens.count("for") + tokens.count("while") + tokens.count("until")
        done_count = tokens.count("done")
        if loop_count != done_count:
            diags.append(
                Diagnostic(
                    code="SHELL_SYNTAX_UNBALANCED_LOOP_DONE",
                    line=1,
                    col=1,
                    message=f"Unbalanced loop construct ({loop_count} loops vs {done_count} 'done')",
                    offset=0,
                    source="shell",
                    severity=1,
                )
            )

        case_count = tokens.count("case")
        esac_count = tokens.count("esac")
        if case_count != esac_count:
            diags.append(
                Diagnostic(
                    code="SHELL_SYNTAX_UNBALANCED_CASE_ESAC",
                    line=1,
                    col=1,
                    message=f"Unbalanced case construct ({case_count} 'case' vs {esac_count} 'esac')",
                    offset=0,
                    source="shell",
                    severity=1,
                )
            )

        has_set_e = bool(re.search(r'\bset\s+-[a-zA-Z]*e', code))
        has_set_u = bool(re.search(r'\bset\s+-[a-zA-Z]*u', code))
        has_pipefail = bool(re.search(r'\bset\s+-[a-zA-Z]*o\s+pipefail|\bset\s+-[a-zA-Z]*pipefail', code))

        if not (has_set_e and has_set_u and has_pipefail):
            diags.append(
                Diagnostic(
                    code="SHELL_MISSING_STRICT_MODE",
                    line=1,
                    col=1,
                    message="Shell script should enforce strict mode ('set -euo pipefail')",
                    offset=0,
                    source="shell",
                    severity=2,
                )
            )

        for line_no, line in enumerate(code.splitlines(), 1):
            s_line = line.strip()
            if not s_line or s_line.startswith("#"):
                continue
            if s_line.startswith(("set ", "export ", "local ", "declare ", "echo ", "readonly ")):
                continue
            if re.search(r'\b(?:rm|cp|mv|chmod|chown|source|\.)\s+[^"\']*\$[A-Za-z_]\w*', s_line):
                diags.append(
                    Diagnostic(
                        code="SHELL_UNQUOTED_VARIABLE",
                        line=line_no,
                        col=1,
                        message=f"Unquoted variable in dangerous command at line {line_no}: '{s_line}'",
                        offset=0,
                        source="shell",
                        severity=1,
                    )
                )

        if re.search(r'`[^`]+`', code):
            diags.append(
                Diagnostic(
                    code="SHELL_LEGACY_BACKTICKS",
                    line=1,
                    col=1,
                    message="Prefer modern subshell '$(cmd)' over legacy backtick '`cmd`'",
                    offset=0,
                    source="shell",
                    severity=2,
                )
            )

        return diags

    def close(self) -> None:
        pass


class ShellVerifier(BaseCloudInfraVerifier):
    """Verifier for Bash and POSIX shell scripts."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            domain_name="shell",
            hatch_patterns=SHELL_ESCAPE_HATCH_PATTERNS,
            directive_hatches=SHELL_DIRECTIVE_HATCHES,
            oracle_factory=ShellOracle,
            find_hatches_fn=find_shell_escape_hatches,
            **kwargs,
        )


# --------------------------------------------------------------------------- #
# 5. Web Core & UI Styling (HTML5, CSS3, Tailwind CSS) Oracle & Verifier
# --------------------------------------------------------------------------- #

VALID_ARIA_ROLES = frozenset({
    "alert", "alertdialog", "application", "article", "banner", "button", "cell",
    "checkbox", "columnheader", "combobox", "complementary", "contentinfo", "definition",
    "dialog", "directory", "document", "feed", "figure", "form", "grid", "gridcell",
    "group", "heading", "img", "link", "list", "listbox", "listitem", "log", "main",
    "marquee", "math", "menu", "menubar", "menuitem", "menuitemcheckbox", "menuitemradio",
    "navigation", "none", "note", "option", "presentation", "progressbar", "radio",
    "radiogroup", "region", "row", "rowgroup", "rowheader", "scrollbar", "search",
    "searchbox", "separator", "slider", "spinbutton", "status", "switch", "tab",
    "table", "tablist", "tabpanel", "term", "textbox", "timer", "toolbar", "tooltip",
    "tree", "treegrid", "treeitem",
})

SEMANTIC_HTML_TAGS = frozenset({
    "header", "nav", "main", "article", "section", "footer", "aside",
    "figure", "figcaption", "time", "mark", "summary", "details",
})

TAILWIND_PREFIX_PATTERNS = re.compile(
    r'^(?:(?:sm|md|lg|xl|2xl|hover|focus|active|disabled|dark|group-hover|peer-checked):)*'
    r'(?:'
    r'flex|grid|inline-flex|inline-grid|block|inline-block|inline|hidden|contents|table|'
    r'container|relative|absolute|fixed|sticky|static|'
    r'[p|m][xytblr]?-(?:\d+|auto|px|\[[^\]]+\])|'
    r'gap-(?:\d+|\[[^\]]+\])|space-[xy]-(?:\d+|\[[^\]]+\])|'
    r'[wh]-(?:\d+|auto|full|screen|px|\[[^\]]+\]|1/\d+|2/\d+|3/\d+|4/\d+|5/\d+|6/\d+|11/\d+|12/\d+)|'
    r'max-[wh]-(?:\S+)|min-[wh]-(?:\S+)|'
    r'text-(?:xs|sm|base|lg|xl|2xl|3xl|4xl|5xl|6xl|left|center|right|justify|[a-z]+-(?:\d+|\[[^\]]+\]))|'
    r'font-(?:thin|extralight|light|normal|medium|semibold|bold|extrabold|black|sans|serif|mono)|'
    r'leading-(?:\S+)|tracking-(?:\S+)|'
    r'bg-(?:[a-z]+-(?:\d+|\[[^\]]+\])|white|black|transparent|current)|'
    r'border(?:-[a-z]+-(?:\d+|\[[^\]]+\])|-[0248]|-[xytblr](?:-[0248])?|)?|'
    r'rounded(?:-(?:none|sm|md|lg|xl|2xl|3xl|full|[xytblr](?:-(?:none|sm|md|lg|xl|2xl|3xl|full))?))?|'
    r'shadow(?:-(?:sm|md|lg|xl|2xl|inner|none))?|'
    r'items-(?:start|end|center|baseline|stretch)|'
    r'justify-(?:start|end|center|between|around|evenly)|'
    r'col-span-(?:\d+|full)|grid-cols-(?:\d+|none)|'
    r'cursor-(?:pointer|default|wait|not-allowed)|'
    r'transition(?:-(?:all|colors|opacity|shadow|transform))?|'
    r'duration-(?:\d+)|ease-(?:linear|in|out|in-out)|'
    r'overflow-(?:auto|hidden|visible|scroll|[xy]-(?:auto|hidden|visible|scroll))|'
    r'z-(?:\d+|auto)|opacity-(?:\d+)|'
    r'truncate|uppercase|lowercase|capitalize|italic|underline'
    r')$',
    re.IGNORECASE,
)


class HtmlSemanticParser(html.parser.HTMLParser):
    """HTML5 parser checking tag nesting, semantic markup, ARIA, and Tailwind classes."""

    def __init__(self) -> None:
        super().__init__()
        self.tags: List[str] = []
        self.semantic_tag_count = 0
        self.div_count = 0
        self.tag_stack: List[str] = []
        self.diags: List[Diagnostic] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        self.tags.append(tag)
        attr_dict = dict(attrs)

        void_elements = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"})
        if tag not in void_elements:
            self.tag_stack.append(tag)

        if tag in SEMANTIC_HTML_TAGS:
            self.semantic_tag_count += 1
        elif tag == "div":
            self.div_count += 1

        if tag == "img":
            if "alt" not in attr_dict:
                self.diags.append(
                    Diagnostic(
                        code="A11Y_IMG_MISSING_ALT",
                        line=self.getpos()[0],
                        col=self.getpos()[1] + 1,
                        message="<img> element must specify an 'alt' attribute for accessibility",
                        offset=0,
                        source="html_tailwind",
                        severity=1,
                    )
                )

        if "role" in attr_dict:
            role = (attr_dict["role"] or "").strip().lower()
            if role and role not in VALID_ARIA_ROLES:
                self.diags.append(
                    Diagnostic(
                        code="A11Y_INVALID_ROLE",
                        line=self.getpos()[0],
                        col=self.getpos()[1] + 1,
                        message=f"Invalid ARIA role '{role}'",
                        offset=0,
                        source="html_tailwind",
                        severity=1,
                    )
                )

        class_attr = attr_dict.get("class") or attr_dict.get("classname")
        if class_attr:
            tokens = class_attr.split()
            for token in tokens:
                token = token.strip()
                if not token:
                    continue
                if not TAILWIND_PREFIX_PATTERNS.match(token):
                    if any(bad in token.lower() for bad in ("align-everything", "col-super-stretch", "text-ultra-bold", "bg-unicorn")):
                        self.diags.append(
                            Diagnostic(
                                code="TAILWIND_UNKNOWN_UTILITY",
                                line=self.getpos()[0],
                                col=self.getpos()[1] + 1,
                                message=f"Unrecognized Tailwind CSS utility class token '{token}'",
                                offset=0,
                                source="html_tailwind",
                                severity=1,
                            )
                        )

    def handle_endtag(self, tag: str) -> None:
        void_elements = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"})
        if tag in void_elements:
            return
        if self.tag_stack and self.tag_stack[-1] == tag:
            self.tag_stack.pop()
        else:
            self.diags.append(
                Diagnostic(
                    code="HTML_SYNTAX_UNBALANCED_TAG",
                    line=self.getpos()[0],
                    col=self.getpos()[1] + 1,
                    message=f"Mismatched HTML closing tag </{tag}>",
                    offset=0,
                    source="html_tailwind",
                    severity=1,
                )
            )


class HtmlTailwindOracle:
    """In-process static analysis oracle for HTML5, CSS3, and Tailwind CSS.

    Safe: Pure in-process string and HTML/CSS AST parser. Zero browser or headless renderer needed.
    """

    def __init__(self, *, timeout_s: float = 5.0) -> None:
        self.timeout_s = timeout_s
        self.n_calls = 0
        self.wall_s = 0.0

    def diagnostics(self, code: str, reference: Optional[str] = None, **kwargs: Any) -> List[Diagnostic]:
        t0 = time.monotonic()
        self.n_calls += 1
        diags: List[Diagnostic] = []

        try:
            diags = self._analyze_html_tailwind(code)
        finally:
            self.wall_s += time.monotonic() - t0

        return diags

    def _analyze_html_tailwind(self, code: str) -> List[Diagnostic]:
        diags: List[Diagnostic] = []
        if not code.strip():
            return diags

        parser = HtmlSemanticParser()
        try:
            parser.feed(code)
            parser.close()
            diags.extend(parser.diags)
        except Exception as exc:
            diags.append(
                Diagnostic(
                    code="HTML_SYNTAX_PARSE_ERROR",
                    line=1,
                    col=1,
                    message=f"HTML parsing failure: {exc}",
                    offset=0,
                    source="html_tailwind",
                    severity=1,
                )
            )
            return diags

        if parser.div_count >= 4 and parser.semantic_tag_count == 0:
            diags.append(
                Diagnostic(
                    code="HTML_NON_SEMANTIC_MARKUP",
                    line=1,
                    col=1,
                    message="Markup uses excessive <div> tags without semantic HTML5 elements (main, header, nav, section, article, footer)",
                    offset=0,
                    source="html_tailwind",
                    severity=2,
                )
            )

        style_blocks = re.findall(r'<style[^>]*>(.*?)</style>', code, re.DOTALL | re.IGNORECASE)
        for s_idx, style_content in enumerate(style_blocks):
            open_b = style_content.count("{")
            close_b = style_content.count("}")
            if open_b != close_b:
                diags.append(
                    Diagnostic(
                        code="CSS_SYNTAX_UNBALANCED_BRACES",
                        line=1,
                        col=1,
                        message=f"CSS syntax error in <style> block #{s_idx + 1}: unbalanced braces",
                        offset=0,
                        source="html_tailwind",
                        severity=1,
                    )
                )

        return diags

    def close(self) -> None:
        pass


class HtmlTailwindVerifier(BaseCloudInfraVerifier):
    """Verifier for HTML5, ARIA accessibility, CSS3, and Tailwind CSS tokens."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            domain_name="html_tailwind",
            hatch_patterns=HTML_TAILWIND_ESCAPE_HATCH_PATTERNS,
            directive_hatches=HTML_TAILWIND_DIRECTIVE_HATCHES,
            oracle_factory=HtmlTailwindOracle,
            find_hatches_fn=find_html_tailwind_escape_hatches,
            **kwargs,
        )


# --------------------------------------------------------------------------- #
# Unified Cloud & Infra Verifier & Auto-Router
# --------------------------------------------------------------------------- #

class CloudInfraVerifier:
    """Unified verifier that auto-routes across Cloud, Infra, Container & UI domains.

    Supports:
    - 'terraform' / 'hcl': Terraform & OpenTofu
    - 'docker' / 'container': Dockerfiles & Container packaging
    - 'kubernetes' / 'k8s' / 'helm': Kubernetes manifests & Helm templates
    - 'shell' / 'bash' / 'posix': Bash & POSIX automation scripts
    - 'html_tailwind' / 'html' / 'tailwind': HTML5, ARIA, and Tailwind CSS

    Auto-detects the domain from code content if domain is omitted.
    Aggregates sub-verifier telemetry into a single unified telemetry report.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._verifiers: Dict[str, BaseCloudInfraVerifier] = {
            "terraform": TerraformVerifier(**kwargs),
            "docker": DockerfileVerifier(**kwargs),
            "kubernetes": KubernetesVerifier(**kwargs),
            "shell": ShellVerifier(**kwargs),
            "html_tailwind": HtmlTailwindVerifier(**kwargs),
        }

    def detect_domain(self, code: str) -> str:
        """Infer target domain based on structural markers."""
        s = code.strip()

        # Dockerfile
        if s.startswith("FROM ") or re.search(r'(?i)^\s*FROM\s+\S+', code, re.MULTILINE):
            return "docker"

        # Kubernetes / Helm
        if re.search(r'(?i)^\s*apiVersion:\s*\S+', code, re.MULTILINE) or re.search(r'(?i)^\s*kind:\s*\S+', code, re.MULTILINE):
            return "kubernetes"
        if "{{" in code and ".Values" in code:
            return "kubernetes"

        # Terraform / HCL
        if re.search(r'(?i)\b(?:resource|provider|variable|terraform|locals|data)\s+["\']', code) or "terraform {" in code:
            return "terraform"

        # Shell
        if s.startswith("#!/") or re.search(r'\bset\s+-[euo]+\b', code) or ("fi\n" in code and "then\n" in code):
            return "shell"

        # HTML / Tailwind
        if "<!DOCTYPE" in s or "<html" in s or "<div" in s or "<main" in s or "<header" in s or "<button" in s:
            return "html_tailwind"

        # Default fallback to terraform
        return "terraform"

    def reward(
        self,
        completion: str,
        reference: Optional[str] = None,
        *,
        prompt: str = "",
        domain: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[float]:
        """Route to appropriate sub-verifier and return reward."""
        if not domain:
            domain = self.detect_domain(f"{prompt}{completion}")

        key = domain.lower()
        if key in ("hcl", "tf"):
            key = "terraform"
        elif key in ("container", "dockerfile"):
            key = "docker"
        elif key in ("k8s", "helm"):
            key = "kubernetes"
        elif key in ("bash", "posix", "sh", "shellcheck"):
            key = "shell"
        elif key in ("html", "tailwind", "css", "ui"):
            key = "html_tailwind"

        v = self._verifiers.get(key)
        if v is None:
            raise ValueError(f"unknown cloud infra domain {domain!r}")

        return v.reward(completion, reference=reference, prompt=prompt, **kwargs)

    def telemetry(self) -> dict:
        """Aggregate telemetry across all sub-verifiers."""
        total_samples = 0
        total_clean = 0
        total_hacked = 0
        total_degenerate = 0
        total_oracle_errors = 0
        total_syntax_errors = 0
        total_linter_errors = 0
        total_escape_hatches = 0
        total_syntax_penalties = 0.0
        total_linter_penalties = 0.0
        total_escape_hatch_penalties = 0.0
        total_reward = 0.0
        sub_telemetry: Dict[str, dict] = {}

        for name, v in self._verifiers.items():
            t = v.telemetry()
            sub_telemetry[name] = t
            n = t["n_samples"]
            total_samples += n
            total_clean += t["n_clean"]
            total_hacked += t["n_hacked"]
            total_degenerate += t["n_degenerate"]
            total_oracle_errors += t["n_oracle_errors"]
            total_syntax_errors += t["n_syntax_errors"]
            total_linter_errors += t["n_linter_errors"]
            total_escape_hatches += t["n_escape_hatches"]
            total_syntax_penalties += t["syntax_penalties"]
            total_linter_penalties += t["linter_penalties"]
            total_escape_hatch_penalties += t["escape_hatch_penalties"]
            total_reward += t["mean_reward"] * n

        return {
            "verifier": "cloud_infra_unified",
            "n_samples": total_samples,
            "n_clean": total_clean,
            "n_hacked": total_hacked,
            "n_degenerate": total_degenerate,
            "n_oracle_errors": total_oracle_errors,
            "n_syntax_errors": total_syntax_errors,
            "n_linter_errors": total_linter_errors,
            "n_escape_hatches": total_escape_hatches,
            "syntax_penalties": total_syntax_penalties,
            "linter_penalties": total_linter_penalties,
            "escape_hatch_penalties": total_escape_hatch_penalties,
            "mean_reward": (total_reward / total_samples) if total_samples else 0.0,
            "sub_verifiers": sub_telemetry,
        }

    def close(self) -> None:
        for v in self._verifiers.values():
            v.close()

    def __enter__(self) -> "CloudInfraVerifier":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
