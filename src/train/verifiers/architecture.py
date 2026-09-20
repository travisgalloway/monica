"""#346 -- RLVR: Clean Architecture boundary linters & dependency DAG verifiers.

Part of the M12 software architecture and structural reinforcement track (#198, #221, #342, #346).
Deterministically enforce Clean / Hexagonal / Layered Architecture rules and dependency invariants
on multi-file model generations without code execution or external linters:

1. Inward Dependency Flow Verification (ArchRuleVerifier):
   - In-process AST parsers for Python (ast.parse) and TypeScript (<15ms).
   - Constructs directed import dependency graph G = (V, E) across architectural layers:
     Domain (core business rules, entities), Application (use cases, interactors, ports),
     Infrastructure (adapters, repositories, external gateways, DB), Presentation (API, web, UI).
   - Invariant: Domain must NEVER import infrastructure packages (ORMs, database drivers,
     web frameworks, external network clients).
   - Invariant: Dependencies must flow strictly inward: Infrastructure -> Application -> Domain,
     Presentation -> Application -> Domain.

2. Package Cycle Detection & Coupling Metrics:
   - Asserts that the inter-module dependency graph is a strict Directed Acyclic Graph (DAG)
     with zero circular import cycles.
   - Computes Afferent Coupling (Ca), Efferent Coupling (Ce), Fan-Out, and Instability per module.
   - Penalizes excessive architectural coupling and circular dependencies.

3. Reward-shaping and Anti-Goodhart invariants:
   - Fine-grained penalty scoring for layer breaches and circular dependency cycles.
   - Degenerate output guards (empty, whitespace, comment-only) short-circuit to -1.0.
   - Anti-Goodhart escape-hatch guards (arch-ignore, @ts-ignore, dynamic import bypasses)
     short-circuit to -1.0.
   - Thread-safe telemetry and injectable oracle seam for testing.
"""

from __future__ import annotations

import ast
import json
import os
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional, Set, Tuple

from src.lsp.diagnostics import Diagnostic, mask_strings_and_comments
from src.train.verifiers import (
    DEFAULT_SEVERITY_WEIGHTS,
    diagnostics_to_reward,
    is_degenerate_output,
    severity_weight,
)

# --------------------------------------------------------------------------- #
# Constants & Layer Definitions
# --------------------------------------------------------------------------- #

class Layer(str, Enum):
    DOMAIN = "domain"
    APPLICATION = "application"
    INFRASTRUCTURE = "infrastructure"
    PRESENTATION = "presentation"
    SHARED = "shared"
    UNKNOWN = "unknown"


# Inward layer ranking: Domain (0) <- Application (1) <- Infrastructure (2) / Presentation (2)
LAYER_RANKS: Mapping[str, int] = {
    Layer.DOMAIN.value: 0,
    Layer.APPLICATION.value: 1,
    Layer.INFRASTRUCTURE.value: 2,
    Layer.PRESENTATION.value: 2,
    Layer.SHARED.value: 0,
    Layer.UNKNOWN.value: 99,
}

# Third-party infrastructure frameworks strictly forbidden in the Domain layer
FORBIDDEN_DOMAIN_FRAMEWORKS = frozenset({
    # Relational & NoSQL ORMs / database drivers
    "sqlalchemy", "tortoise", "peewee", "motor", "pymongo", "redis", "ioredis",
    "psycopg2", "psycopg", "asyncpg", "sqlite3", "prisma", "@prisma/client",
    "typeorm", "mikro-orm", "mongoose", "sequelize", "pg", "mysql", "mysql2",
    "knex", "objection", "dynamoose", "cassandra-driver",

    # Web & API frameworks
    "fastapi", "flask", "django", "starlette", "tornado", "aiohttp", "bottle",
    "express", "koa", "fastify", "nestjs", "@nestjs/common", "@nestjs/core",
    "next", "hono", "gin", "fiber", "echo",

    # Network clients, cloud SDKs & messaging
    "requests", "httpx", "urllib3", "boto3", "botocore", "google.cloud", "azure",
    "axios", "node-fetch", "got", "superagent", "grpc", "@grpc/grpc-js",
    "celery", "pika", "kafka", "kafkajs", "amqplib",
})

# Third-party presentation & web frameworks forbidden in Application layer
FORBIDDEN_APPLICATION_FRAMEWORKS = frozenset({
    "fastapi", "flask", "django", "starlette", "tornado", "bottle",
    "express", "koa", "fastify", "nestjs", "@nestjs/common", "@nestjs/core", "next", "hono",
})

# Fine-grained penalty overrides for RLVR reward shaping
DEFAULT_ARCH_CODE_OVERRIDES: Mapping[str, float] = {
    "ARCH_LAYER_BREACH": 0.35,
    "ARCH_DOMAIN_INFRASTRUCTURE_IMPORT": 0.35,
    "ARCH_CIRCULAR_DEPENDENCY": 0.40,
    "ARCH_EXCESSIVE_COUPLING": 0.15,
    "ARCH_EXCESSIVE_FAN_OUT": 0.15,
}

# Anti-Goodhart escape-hatch patterns
ARCHITECTURE_ESCAPE_HATCH_PATTERNS: Mapping[str, re.Pattern] = {
    "arch_suppress_directive": re.compile(
        r"(?:#|//)\s*(?:arch-ignore|noqa:\s*ARCH|type:\s*ignore\s*\[arch\])\b",
        re.IGNORECASE,
    ),
    "dynamic_import_bypass": re.compile(
        r"\b(?:__import__|importlib\.(?:import_module|__import__)|sys\.modules\s*\[)"
    ),
    "eval_exec_bypass": re.compile(r"\b(?:eval|exec|Function)\s*\("),
    "ts_ignore_directive": re.compile(r"@ts-(?:ignore|expect-error|nocheck)\b"),
    "type_ignore_directive": re.compile(r"#\s*type:\s*ignore\b"),
    "as_any_escape": re.compile(r"\bas\s+any\b"),
}

ARCHITECTURE_DIRECTIVE_HATCHES: frozenset[str] = frozenset({
    "arch_suppress_directive",
    "ts_ignore_directive",
    "type_ignore_directive",
    "as_any_escape",
})


# --------------------------------------------------------------------------- #
# Path Normalization & Multi-File Parsing
# --------------------------------------------------------------------------- #

def normalize_file_path(path: str) -> str:
    """Normalize file path into a clean relative POSIX path."""
    p = str(path).replace("\\", "/").strip()
    p = p.removeprefix("./")
    parts: List[str] = []
    for part in PurePosixPath(p).parts:
        if part in ("", "."):
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/".join(parts)


def parse_multi_file_source(source: str | Mapping[str, str] | Sequence[dict]) -> dict[str, str]:
    """Parse multi-file model completion or repository representation into {path: content}."""
    if isinstance(source, Mapping):
        return {normalize_file_path(k): str(v) for k, v in source.items()}

    if isinstance(source, Sequence) and not isinstance(source, str):
        files: dict[str, str] = {}
        for item in source:
            if isinstance(item, dict) and "path" in item:
                content = item.get("content") or item.get("text") or ""
                files[normalize_file_path(item["path"])] = str(content)
        return files

    text = str(source).strip()
    files: dict[str, str] = {}

    # 1. JSON-encoded repository manifest
    if text.startswith("{") or text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                if "files" in parsed and isinstance(parsed["files"], (dict, list)):
                    return parse_multi_file_source(parsed["files"])
                if all(isinstance(v, str) for v in parsed.values()):
                    return {normalize_file_path(k): v for k, v in parsed.items()}
            elif isinstance(parsed, list):
                return parse_multi_file_source(parsed)
        except Exception:
            pass

    # 2. XML file tags: <file path="...">...</file>
    xml_re = re.compile(
        r'<file\s+(?:path|name)=["\']([^"\']+)["\']\s*>(.*?)</file>',
        re.DOTALL | re.IGNORECASE,
    )
    xml_matches = list(xml_re.finditer(text))
    if xml_matches:
        for m in xml_matches:
            path = normalize_file_path(m.group(1))
            files[path] = m.group(2).strip("\n")
        return files

    # 3. Markdown code blocks with header preceding the fence:
    #    ### path/to/file.py
    #    ```python
    #    ...
    #    ```
    header_fence_re = re.compile(
        r"""(?:^|\n)(?:#+|//|\*\*|File:?|Filename:?)\s*([a-zA-Z0-9_./\-]+\.[a-zA-Z0-9]+)\*?\*?\s*\n+```(?:[a-zA-Z0-9_\-]+)?\n([\s\S]*?)```""",
        re.MULTILINE,
    )
    for m in header_fence_re.finditer(text):
        p = normalize_file_path(m.group(1))
        files[p] = m.group(2)

    if files:
        return files

    # 4. Markdown code fence with embedded path:
    #    ```python:path/to/file.py
    fence_path_re = re.compile(
        r"""```(?:[a-zA-Z0-9_\-]+)?[:\s]+(?:filename=|title=)?["'`]?([a-zA-Z0-9_./\-]+\.[a-zA-Z0-9]+)["'`]?\s*\n([\s\S]*?)```""",
        re.MULTILINE,
    )
    for m in fence_path_re.finditer(text):
        p = normalize_file_path(m.group(1))
        files[p] = m.group(2)

    if files:
        return files

    # 5. Comment at the very first line of a code fence:
    fence_comment_re = re.compile(
        r"""```(?:[a-zA-Z0-9_\-]+)?\s*\n(?:#|//)\s*(?:file(?:name)?:?)\s*([a-zA-Z0-9_./\-]+\.[a-zA-Z0-9]+)\s*\n([\s\S]*?)```""",
        re.MULTILINE,
    )
    for m in fence_comment_re.finditer(text):
        p = normalize_file_path(m.group(1))
        files[p] = m.group(2)

    if files:
        return files

    return {"file.py": text}


# --------------------------------------------------------------------------- #
# Layer Classification
# --------------------------------------------------------------------------- #

_LAYER_REGEX_MAP = (
    (Layer.DOMAIN.value, re.compile(r"(?:^|/)(?:domain|entities|models?|core)(?:/|$)", re.IGNORECASE)),
    (Layer.APPLICATION.value, re.compile(r"(?:^|/)(?:application|app|use_?cases?|services?|interactors?|ports?)(?:/|$)", re.IGNORECASE)),
    (Layer.INFRASTRUCTURE.value, re.compile(r"(?:^|/)(?:infrastructure|infra|adapters?|repositories?|database|db|clients?|external|gateways?)(?:/|$)", re.IGNORECASE)),
    (Layer.PRESENTATION.value, re.compile(r"(?:^|/)(?:presentation|ui|api|web|controllers?|views?|routes?|handlers?|http)(?:/|$)", re.IGNORECASE)),
)


def classify_file_layer(path: str, layer_map: Optional[Mapping[str, str]] = None) -> str:
    """Classify a file path into its architectural layer."""
    norm = normalize_file_path(path)
    if layer_map:
        if norm in layer_map:
            return layer_map[norm].lower()
        for prefix, lay in layer_map.items():
            if norm.startswith(prefix.strip("/")):
                return lay.lower()

    for layer_name, pattern in _LAYER_REGEX_MAP:
        if pattern.search(norm):
            return layer_name

    return Layer.UNKNOWN.value


# --------------------------------------------------------------------------- #
# In-Process AST Import Extraction
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ImportRecord:
    """Extracted import relationship from a file."""

    source_file: str
    target_specifier: str
    target_file: Optional[str]
    is_external: bool
    line: int
    col: int
    symbols: Tuple[str, ...] = ()


_TS_MODULE_SUFFIXES = (
    "",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mts",
    ".cts",
    "/index.ts",
    "/index.tsx",
    "/index.js",
    "/index.jsx",
)

_PY_MODULE_SUFFIXES = (
    "",
    ".py",
    "/__init__.py",
)

_TS_IMPORT_RE = re.compile(
    r"""
    (?:
        (?:import|export)\s+(?:type\s+)?(?:(?:\{[^}]*\}|\*\s+as\s+[\w$]+|[\w$]+(?:\s*,\s*\{[^}]*\})?)\s+from\s+)?['"](?P<spec1>[^'"]+)['"]
      | import\s*['"](?P<spec2>[^'"]+)['"]
      | require\s*\(\s*['"](?P<spec3>[^'"]+)['"]\s*\)
      | import\s*\(\s*['"](?P<spec4>[^'"]+)['"]\s*\)
    )
    """,
    re.VERBOSE | re.MULTILINE,
)


def _resolve_ts_import(specifier: str, importer_path: str, known_files: Set[str]) -> Optional[str]:
    """Resolve a TypeScript import specifier against known project files."""
    spec = specifier.strip()
    norm_importer = normalize_file_path(importer_path)
    base_dir = PurePosixPath(norm_importer).parent

    if spec.startswith("."):
        target_norm = normalize_file_path(str(base_dir / spec))
        for suffix in _TS_MODULE_SUFFIXES:
            cand = normalize_file_path(target_norm + suffix)
            if cand in known_files:
                return cand
        return None

    trimmed = re.sub(r"^[@~]/", "", spec)
    for cand_prefix in (trimmed, f"src/{trimmed}"):
        for suffix in _TS_MODULE_SUFFIXES:
            cand = normalize_file_path(cand_prefix + suffix)
            if cand in known_files:
                return cand

    for suffix in _TS_MODULE_SUFFIXES:
        cand = normalize_file_path(spec + suffix)
        if cand in known_files:
            return cand

    return None


def parse_typescript_imports(content: str, file_path: str, known_files: Set[str]) -> List[ImportRecord]:
    """Extract all import records from TypeScript / JavaScript code in-process (<2ms)."""
    norm_file = normalize_file_path(file_path)
    records: List[ImportRecord] = []

    for m in _TS_IMPORT_RE.finditer(content):
        spec = m.group("spec1") or m.group("spec2") or m.group("spec3") or m.group("spec4")
        if not spec:
            continue
        spec = spec.strip()
        line = content[:m.start()].count("\n") + 1
        col = m.start() - max(content.rfind("\n", 0, m.start()), 0)

        resolved = _resolve_ts_import(spec, norm_file, known_files)
        is_external = resolved is None

        pkg_root = spec
        if is_external:
            if spec.startswith("@"):
                parts = spec.split("/")
                pkg_root = "/".join(parts[:2]) if len(parts) >= 2 else spec
            else:
                pkg_root = spec.split("/")[0]

        records.append(
            ImportRecord(
                source_file=norm_file,
                target_specifier=pkg_root,
                target_file=resolved,
                is_external=is_external,
                line=line,
                col=col,
            )
        )

    return records


def _resolve_py_import(
    module: str,
    level: int,
    importer_path: str,
    known_files: Set[str],
) -> Optional[str]:
    """Resolve a Python import module specifier against known project files."""
    norm_importer = normalize_file_path(importer_path)
    importer_dir = PurePosixPath(norm_importer).parent

    if level > 0:
        curr = importer_dir
        for _ in range(level - 1):
            curr = curr.parent
        rel_path = "/".join(module.split(".")) if module else ""
        target_norm = normalize_file_path(str(curr / rel_path)) if rel_path else normalize_file_path(str(curr))
    else:
        target_norm = normalize_file_path("/".join(module.split(".")))

    for suffix in _PY_MODULE_SUFFIXES:
        cand = normalize_file_path(target_norm + suffix)
        if cand in known_files:
            return cand

    for suffix in _PY_MODULE_SUFFIXES:
        cand = normalize_file_path("src/" + target_norm + suffix)
        if cand in known_files:
            return cand

    parts = target_norm.split("/")
    if len(parts) > 1:
        for i in range(len(parts) - 1, 0, -1):
            parent_prefix = "/".join(parts[:i])
            for suffix in _PY_MODULE_SUFFIXES:
                cand = normalize_file_path(parent_prefix + suffix)
                if cand in known_files:
                    return cand
                cand_src = normalize_file_path("src/" + parent_prefix + suffix)
                if cand_src in known_files:
                    return cand_src

    return None


def parse_python_imports(content: str, file_path: str, known_files: Set[str]) -> Tuple[List[ImportRecord], Optional[Diagnostic]]:
    """Extract all import records from Python source using standard ast.parse (<2ms)."""
    norm_file = normalize_file_path(file_path)
    records: List[ImportRecord] = []
    syntax_diag: Optional[Diagnostic] = None

    try:
        tree = ast.parse(content, filename=norm_file)
    except SyntaxError as syn:
        syntax_diag = Diagnostic(
            code="PYTHON_SYNTAX_ERROR",
            line=syn.lineno or 1,
            col=syn.offset or 1,
            message=f"Syntax error: {syn.msg}",
            offset=0,
            source="ast",
            severity=1,
        )
        return records, syntax_diag

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                spec = alias.name
                resolved = _resolve_py_import(spec, 0, norm_file, known_files)
                is_ext = resolved is None
                pkg_root = spec.split(".")[0] if is_ext else spec
                records.append(
                    ImportRecord(
                        source_file=norm_file,
                        target_specifier=pkg_root,
                        target_file=resolved,
                        is_external=is_ext,
                        line=node.lineno,
                        col=node.col_offset,
                        symbols=(alias.asname or alias.name,),
                    )
                )
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            level = node.level
            resolved = _resolve_py_import(mod, level, norm_file, known_files)

            if not resolved and level == 0 and mod:
                pkg_root = mod.split(".")[0]
            elif not resolved and level > 0:
                pkg_root = mod
            else:
                pkg_root = mod

            is_ext = resolved is None
            if is_ext:
                for alias in node.names:
                    cand_mod = f"{mod}.{alias.name}" if mod else alias.name
                    sub_res = _resolve_py_import(cand_mod, level, norm_file, known_files)
                    if sub_res:
                        resolved = sub_res
                        is_ext = False
                        break

            records.append(
                ImportRecord(
                    source_file=norm_file,
                    target_specifier=pkg_root,
                    target_file=resolved,
                    is_external=is_ext,
                    line=node.lineno,
                    col=node.col_offset,
                    symbols=tuple(a.name for a in node.names),
                )
            )

    return records, None


# --------------------------------------------------------------------------- #
# Coupling Metrics & Graph Analysis
# --------------------------------------------------------------------------- #

@dataclass
class ModuleCoupling:
    """Coupling metrics for a module/file."""

    file: str
    layer: str
    ca: int
    ce: int
    fan_out: int
    instability: float
    internal_imports: List[str] = field(default_factory=list)
    external_imports: List[str] = field(default_factory=list)


@dataclass
class ArchAnalysisResult:
    """Full architectural analysis report."""

    is_clean: bool
    is_dag: bool
    cycles: List[List[str]]
    coupling: Dict[str, ModuleCoupling]
    diagnostics: List[Diagnostic]
    elapsed_ms: float
    file_layers: Dict[str, str]
    records: List[ImportRecord]


def detect_cycles_tarjan(nodes: Sequence[str], edges: Mapping[str, Set[str]]) -> List[List[str]]:
    """Detect circular import dependency cycles using Tarjan's SCC algorithm."""
    index = 0
    indices: Dict[str, int] = {}
    lowlinks: Dict[str, int] = {}
    stack: List[str] = []
    on_stack: Set[str] = set()
    sccs: List[List[str]] = []

    def strongconnect(v: str) -> None:
        nonlocal index
        indices[v] = index
        lowlinks[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)

        for w in edges.get(v, ()):
            if w not in indices:
                strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif w in on_stack:
                lowlinks[v] = min(lowlinks[v], indices[w])

        if lowlinks[v] == indices[v]:
            scc: List[str] = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                scc.append(w)
                if w == v:
                    break
            sccs.append(scc)

    for node in sorted(nodes):
        if node not in indices:
            strongconnect(node)

    cycles: List[List[str]] = []
    for scc in sccs:
        if len(scc) == 1:
            u = scc[0]
            if u in edges.get(u, ()):
                cycles.append([u, u])
            continue

        scc_set = set(scc)
        for start_node in sorted(scc):
            visited: List[str] = []

            def find_cycle(curr: str) -> Optional[List[str]]:
                visited.append(curr)
                for nxt in sorted(edges.get(curr, ())):
                    if nxt not in scc_set:
                        continue
                    if nxt == start_node and len(visited) > 1:
                        return list(visited)
                    if nxt not in visited:
                        res = find_cycle(nxt)
                        if res is not None:
                            return res
                visited.pop()
                return None

            c = find_cycle(start_node)
            if c is not None:
                min_idx = c.index(min(c))
                norm_c = c[min_idx:] + c[:min_idx]
                if norm_c not in cycles:
                    cycles.append(norm_c)

    cycles.sort(key=lambda c: (len(c), c))
    return cycles


# --------------------------------------------------------------------------- #
# ArchRuleOracle Engine
# --------------------------------------------------------------------------- #

class ArchRuleOracle:
    """In-process Clean Architecture boundary linter and dependency DAG verifier (<15ms)."""

    def __init__(
        self,
        *,
        max_efferent_coupling: int = 10,
        max_fan_out: int = 15,
        strict_presentation_to_infra: bool = False,
        layer_map: Optional[Mapping[str, str]] = None,
        timeout_s: float = 5.0,
    ) -> None:
        self.max_efferent_coupling = max_efferent_coupling
        self.max_fan_out = max_fan_out
        self.strict_presentation_to_infra = strict_presentation_to_infra
        self.layer_map = layer_map
        self.timeout_s = timeout_s

        self.n_calls = 0
        self.wall_s = 0.0

    def analyze(
        self,
        source: str | Mapping[str, str] | Sequence[dict],
        *,
        layer_map: Optional[Mapping[str, str]] = None,
    ) -> ArchAnalysisResult:
        """Run complete architectural analysis across multi-file repository."""
        t0 = time.monotonic()
        self.n_calls += 1

        active_layer_map = layer_map or self.layer_map
        files_dict = parse_multi_file_source(source)
        known_files = set(files_dict.keys())

        diagnostics: List[Diagnostic] = []
        import_records: List[ImportRecord] = []
        file_layers: Dict[str, str] = {
            f: classify_file_layer(f, active_layer_map) for f in known_files
        }

        # 1. Parse AST imports for each file in-process
        for file_path, content in files_dict.items():
            if file_path.endswith((".ts", ".tsx", ".js", ".jsx", ".mts", ".cts")):
                recs = parse_typescript_imports(content, file_path, known_files)
                import_records.extend(recs)
            elif file_path.endswith(".py"):
                recs, syn_diag = parse_python_imports(content, file_path, known_files)
                if syn_diag is not None:
                    diagnostics.append(syn_diag)
                import_records.extend(recs)

        # 2. Build dependency graph G = (V, E)
        internal_edges: Dict[str, Set[str]] = {f: set() for f in known_files}
        external_deps: Dict[str, Set[str]] = {f: set() for f in known_files}
        dependents: Dict[str, Set[str]] = {f: set() for f in known_files}

        for rec in import_records:
            src_layer = file_layers.get(rec.source_file, Layer.UNKNOWN.value)

            if rec.is_external:
                external_deps[rec.source_file].add(rec.target_specifier)
                target_pkg = rec.target_specifier.lower()

                # Invariant: Domain must NEVER import infrastructure packages
                if src_layer == Layer.DOMAIN.value and target_pkg in FORBIDDEN_DOMAIN_FRAMEWORKS:
                    diagnostics.append(
                        Diagnostic(
                            code="ARCH_DOMAIN_INFRASTRUCTURE_IMPORT",
                            line=rec.line,
                            col=rec.col,
                            message=(
                                f"Domain layer ({rec.source_file}) must not import infrastructure "
                                f"framework/driver {rec.target_specifier!r}"
                            ),
                            offset=0,
                            source="clean_architecture",
                            severity=1,
                        )
                    )

                # Invariant: Application must not import presentation/web frameworks
                if src_layer == Layer.APPLICATION.value and target_pkg in FORBIDDEN_APPLICATION_FRAMEWORKS:
                    diagnostics.append(
                        Diagnostic(
                            code="ARCH_LAYER_BREACH",
                            line=rec.line,
                            col=rec.col,
                            message=(
                                f"Application layer ({rec.source_file}) must not import presentation "
                                f"framework {rec.target_specifier!r}"
                            ),
                            offset=0,
                            source="clean_architecture",
                            severity=1,
                        )
                    )
            else:
                tgt = rec.target_file
                if tgt and tgt in known_files and tgt != rec.source_file:
                    internal_edges[rec.source_file].add(tgt)
                    dependents[tgt].add(rec.source_file)

                # Invariant: Dependencies must flow strictly inward
                tgt_layer = file_layers.get(tgt or "", Layer.UNKNOWN.value)
                if tgt and tgt != rec.source_file:
                    tgt_cap = tgt_layer.capitalize()

                    # Domain must not import Application, Infrastructure, Presentation
                    if src_layer == Layer.DOMAIN.value and tgt_layer in (
                        Layer.APPLICATION.value,
                        Layer.INFRASTRUCTURE.value,
                        Layer.PRESENTATION.value,
                    ):
                        diagnostics.append(
                            Diagnostic(
                                code="ARCH_LAYER_BREACH",
                                line=rec.line,
                                col=rec.col,
                                message=(
                                    f"Domain layer ({rec.source_file}) must not import outer "
                                    f"{tgt_cap} layer ({tgt})"
                                ),
                                offset=0,
                                source="clean_architecture",
                                severity=1,
                            )
                        )
                    # Application must not import Infrastructure or Presentation
                    elif src_layer == Layer.APPLICATION.value and tgt_layer in (
                        Layer.INFRASTRUCTURE.value,
                        Layer.PRESENTATION.value,
                    ):
                        diagnostics.append(
                            Diagnostic(
                                code="ARCH_LAYER_BREACH",
                                line=rec.line,
                                col=rec.col,
                                message=(
                                    f"Application layer ({rec.source_file}) must not import outer "
                                    f"{tgt_cap} layer ({tgt})"
                                ),
                                offset=0,
                                source="clean_architecture",
                                severity=1,
                            )
                        )
                    # Infrastructure must not import Presentation
                    elif src_layer == Layer.INFRASTRUCTURE.value and tgt_layer == Layer.PRESENTATION.value:
                        diagnostics.append(
                            Diagnostic(
                                code="ARCH_LAYER_BREACH",
                                line=rec.line,
                                col=rec.col,
                                message=(
                                    f"Infrastructure layer ({rec.source_file}) must not import "
                                    f"Presentation layer ({tgt})"
                                ),
                                offset=0,
                                source="clean_architecture",
                                severity=1,
                            )
                        )
                    # Presentation must not import Infrastructure directly if strict mode is enabled
                    elif self.strict_presentation_to_infra and src_layer == Layer.PRESENTATION.value and tgt_layer == Layer.INFRASTRUCTURE.value:
                        diagnostics.append(
                            Diagnostic(
                                code="ARCH_LAYER_BREACH",
                                line=rec.line,
                                col=rec.col,
                                message=(
                                    f"Presentation layer ({rec.source_file}) must not import "
                                    f"Infrastructure layer ({tgt}) directly"
                                ),
                                offset=0,
                                source="clean_architecture",
                                severity=2,
                            )
                        )

        # 3. Detect Circular Dependency Cycles
        cycles = detect_cycles_tarjan(list(known_files), internal_edges)
        is_dag = len(cycles) == 0

        for cycle in cycles:
            cycle_str = " -> ".join(cycle + [cycle[0]])
            diagnostics.append(
                Diagnostic(
                    code="ARCH_CIRCULAR_DEPENDENCY",
                    line=1,
                    col=1,
                    message=f"Circular dependency cycle detected: {cycle_str}",
                    offset=0,
                    source="dependency_dag",
                    severity=1,
                )
            )

        # 4. Measure Afferent (Ca), Efferent (Ce) Coupling, Fan-Out & Instability
        coupling_map: Dict[str, ModuleCoupling] = {}
        for f in known_files:
            ca = len(dependents[f])
            ce = len(internal_edges[f])
            fan_out = ce + len(external_deps[f])
            instability = (ce / (ca + ce)) if (ca + ce) > 0 else 0.0

            mc = ModuleCoupling(
                file=f,
                layer=file_layers[f],
                ca=ca,
                ce=ce,
                fan_out=fan_out,
                instability=instability,
                internal_imports=sorted(internal_edges[f]),
                external_imports=sorted(external_deps[f]),
            )
            coupling_map[f] = mc

            if ce > self.max_efferent_coupling:
                diagnostics.append(
                    Diagnostic(
                        code="ARCH_EXCESSIVE_COUPLING",
                        line=1,
                        col=1,
                        message=(
                            f"Module {f} has excessive efferent coupling "
                            f"(Ce={ce} > {self.max_efferent_coupling})"
                        ),
                        offset=0,
                        source="coupling_metrics",
                        severity=2,
                    )
                )

            if fan_out > self.max_fan_out:
                diagnostics.append(
                    Diagnostic(
                        code="ARCH_EXCESSIVE_FAN_OUT",
                        line=1,
                        col=1,
                        message=(
                            f"Module {f} has excessive fan-out "
                            f"(FanOut={fan_out} > {self.max_fan_out})"
                        ),
                        offset=0,
                        source="coupling_metrics",
                        severity=2,
                    )
                )

        elapsed = (time.monotonic() - t0) * 1000.0
        self.wall_s += elapsed / 1000.0

        is_clean = len(diagnostics) == 0 and is_dag

        return ArchAnalysisResult(
            is_clean=is_clean,
            is_dag=is_dag,
            cycles=cycles,
            coupling=coupling_map,
            diagnostics=diagnostics,
            elapsed_ms=elapsed,
            file_layers=file_layers,
            records=import_records,
        )

    def diagnostics(
        self,
        source: str | Mapping[str, str] | Sequence[dict],
        **kwargs: Any,
    ) -> List[Diagnostic]:
        """Return list of diagnostics adhering to the standard oracle contract."""
        res = self.analyze(source, **kwargs)
        return res.diagnostics

    def close(self) -> None:
        """Close oracle resources."""
        pass


# --------------------------------------------------------------------------- #
# Escape-Hatch & Degenerate Helpers
# --------------------------------------------------------------------------- #

def find_architecture_escape_hatches(text: str) -> List[str]:
    """Detect anti-Goodhart clean architecture escape hatches."""
    found: List[str] = []
    masked = mask_strings_and_comments(text)

    for name, pattern in ARCHITECTURE_ESCAPE_HATCH_PATTERNS.items():
        if name in ARCHITECTURE_DIRECTIVE_HATCHES:
            if pattern.search(text):
                found.append(name)
        else:
            if pattern.search(masked) or pattern.search(text):
                found.append(name)

    return found


def is_arch_degenerate_output(text: str, *, min_code_chars: int = 2) -> Optional[str]:
    """Check for empty, whitespace-only, comment-only (Python and TS), or too-short code."""
    base_res = is_degenerate_output(text, min_code_chars=min_code_chars)
    if base_res is not None:
        return base_res

    # Strip Python '#' comments and check remaining executable code
    lines: List[str] = []
    for line in text.splitlines():
        code_part = re.sub(r"#.*$", "", line).strip()
        if code_part:
            lines.append(code_part)

    residual = "".join(lines).strip()
    if not residual:
        return "comment_only" if text.strip() else "whitespace_only"
    if len(residual) < min_code_chars:
        return "too_short"

    return None


# --------------------------------------------------------------------------- #
# ArchRuleVerifier Class
# --------------------------------------------------------------------------- #

class ArchRuleVerifier:
    """Verifier for Clean Architecture boundaries, inward dependency flow, and DAG rules.

    Matches RLVR reward contract:
    - Base reward: 1.0 (clean)
    - Diagnostic errors/warnings reduce reward according to severity weights and code overrides
    - Degenerate output (empty, whitespace, comment-only) short-circuits to -1.0
    - Escape hatches / anti-Goodhart violations short-circuit to -1.0
    - Fast in-process AST execution (<15ms)
    """

    def __init__(
        self,
        *,
        max_efferent_coupling: int = 10,
        max_fan_out: int = 15,
        strict_presentation_to_infra: bool = False,
        layer_map: Optional[Mapping[str, str]] = None,
        code_overrides: Optional[Mapping[str, float]] = None,
        severity_weights: Mapping[int, float] = DEFAULT_SEVERITY_WEIGHTS,
        fail_fast: bool = True,
        oracle: Optional[ArchRuleOracle] = None,
        timeout_s: float = 5.0,
    ) -> None:
        self.max_efferent_coupling = max_efferent_coupling
        self.max_fan_out = max_fan_out
        self.strict_presentation_to_infra = strict_presentation_to_infra
        self.layer_map = layer_map
        self.code_overrides = dict(DEFAULT_ARCH_CODE_OVERRIDES)
        if code_overrides:
            self.code_overrides.update(code_overrides)
        self.severity_weights = severity_weights
        self.fail_fast = fail_fast
        self.timeout_s = timeout_s

        self.oracle = oracle or ArchRuleOracle(
            max_efferent_coupling=max_efferent_coupling,
            max_fan_out=max_fan_out,
            strict_presentation_to_infra=strict_presentation_to_infra,
            layer_map=layer_map,
            timeout_s=timeout_s,
        )

        self._lock = threading.Lock()
        self._n_samples = 0
        self._n_clean = 0
        self._n_degenerate = 0
        self._n_hacked = 0
        self._n_layer_breaches = 0
        self._n_circular_cycles = 0
        self._reward_total = 0.0
        self._elapsed_ms_total = 0.0

    def find_escape_hatches(self, text: str) -> List[str]:
        """Detect anti-Goodhart clean architecture escape hatches."""
        return find_architecture_escape_hatches(text)

    def evaluate(
        self,
        completion: str | Mapping[str, str] | Sequence[dict],
        reference: Any = None,
        *,
        prompt: str = "",
        repo: Optional[Mapping[str, str]] = None,
        files: Optional[Mapping[str, str]] = None,
        layer_map: Optional[Mapping[str, str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Evaluate completion and return detailed diagnostics, metrics, and reward."""
        t0 = time.monotonic()

        if isinstance(completion, str):
            raw_text = completion
        elif isinstance(completion, Mapping):
            raw_text = "\n".join(str(v) for v in completion.values())
        else:
            raw_text = str(completion)

        degen_reason = is_arch_degenerate_output(raw_text)
        is_degenerate = degen_reason is not None

        hatches = self.find_escape_hatches(raw_text)
        is_hacked = len(hatches) > 0

        source_input = repo or files or completion

        if self.fail_fast and (is_degenerate or is_hacked):
            reward = -1.0
            with self._lock:
                self._n_samples += 1
                if is_degenerate:
                    self._n_degenerate += 1
                if is_hacked:
                    self._n_hacked += 1
                self._reward_total += reward
                self._elapsed_ms_total += (time.monotonic() - t0) * 1000.0

            return {
                "reward": reward,
                "is_clean": False,
                "is_dag": False,
                "is_degenerate": is_degenerate,
                "degenerate_reason": degen_reason,
                "is_hacked": is_hacked,
                "hatches": hatches,
                "diagnostics": [],
                "cycles": [],
                "coupling": {},
                "elapsed_ms": (time.monotonic() - t0) * 1000.0,
            }

        with self._lock:
            analysis = self.oracle.analyze(source_input, layer_map=layer_map)

        diagnostics = analysis.diagnostics
        reward = diagnostics_to_reward(
            diagnostics,
            hacked=is_hacked,
            degenerate=is_degenerate,
            severity_weights=self.severity_weights,
            code_overrides=self.code_overrides,
        )

        n_breaches = sum(
            1 for d in diagnostics if d.code in ("ARCH_LAYER_BREACH", "ARCH_DOMAIN_INFRASTRUCTURE_IMPORT")
        )
        n_cycles = len(analysis.cycles)
        is_clean = (reward == 1.0) and analysis.is_clean and not is_hacked and not is_degenerate

        elapsed = (time.monotonic() - t0) * 1000.0

        with self._lock:
            self._n_samples += 1
            if is_clean:
                self._n_clean += 1
            if is_hacked:
                self._n_hacked += 1
            if is_degenerate:
                self._n_degenerate += 1
            self._n_layer_breaches += n_breaches
            self._n_circular_cycles += n_cycles
            self._reward_total += reward
            self._elapsed_ms_total += elapsed

        return {
            "reward": reward,
            "is_clean": is_clean,
            "is_dag": analysis.is_dag,
            "is_degenerate": is_degenerate,
            "degenerate_reason": degen_reason,
            "is_hacked": is_hacked,
            "hatches": hatches,
            "diagnostics": diagnostics,
            "cycles": analysis.cycles,
            "coupling": {k: v.__dict__ for k, v in analysis.coupling.items()},
            "file_layers": analysis.file_layers,
            "elapsed_ms": elapsed,
        }

    def reward(
        self,
        completion: str | Mapping[str, str] | Sequence[dict],
        reference: Any = None,
        *,
        prompt: str = "",
        **kwargs: Any,
    ) -> float:
        """Score one completion against Clean Architecture and DAG rules."""
        eval_result = self.evaluate(completion, reference=reference, prompt=prompt, **kwargs)
        return float(eval_result["reward"])

    def telemetry(self) -> Dict[str, Any]:
        """Aggregate telemetric metrics."""
        with self._lock:
            n = self._n_samples
            return {
                "verifier": "ArchRuleVerifier",
                "n_samples": n,
                "n_clean": self._n_clean,
                "n_hacked": self._n_hacked,
                "n_degenerate": self._n_degenerate,
                "n_layer_breaches": self._n_layer_breaches,
                "n_circular_cycles": self._n_circular_cycles,
                "mean_reward": (self._reward_total / n) if n else 0.0,
                "mean_elapsed_ms": (self._elapsed_ms_total / n) if n else 0.0,
                "oracle_calls": self.oracle.n_calls,
                "oracle_wall_s": self.oracle.wall_s,
            }

    def close(self) -> None:
        """Release oracle resources."""
        self.oracle.close()
