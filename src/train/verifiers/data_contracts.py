"""#342 -- Data Engineering, Contracts & Query Verifiers.

Part of the M12 software architecture, relational data, and contract evaluation track (#198, #221, #342).
Provides ultra-fast, sandbox-free verification for data pipelines, API contracts, schema boundaries,
and architectural layers:

1. Relational SQL (Postgres, MySQL, SQLite, DuckDB):
   - In-process: sqlglot multi-dialect AST parsing (<3ms) with pure-Python AST fallback.
   - In-memory execution: SQLite :memory: schema setup and deterministic DML execution (<5ms).
   - Checks: Dialect AST syntax, schema column resolution, index usability, dialect translation equivalence.
   - Anti-Goodhart: Reject SELECT *, cartesian joins without ON clauses.

2. Data & ML Pipelines (Pandas, Polars, PyTorch):
   - In-process: AST visitor + tensor dimension/shape inference.
   - Checks: Tensor shape transformation contracts (B C H W), Polars schema resolution, vectorized operations.
   - Anti-Goodhart: Reject dynamic shape escapes, bare object dtypes, unvectorized row iterations.

3. API Contracts (OpenAPI v3.1 & JSON Schema):
   - In-process: OpenAPI v3.0/v3.1 specification & JSON schema validation.
   - Linter: spectral toolchain probe / built-in contract linter.
   - Checks: Schema validity, path parameter bindings, 2xx response definitions, type safety.
   - Anti-Goodhart: Reject untyped payloads, unconstrained strings.

4. GraphQL Schemas & Operations:
   - In-process: GraphQL SDL and query AST validation.
   - Checks: Schema definition language (SDL) correctness, query document field resolution, variable typing.
   - Anti-Goodhart: Reject circular queries, unconstrained scalar types.

5. Protocol Buffers & gRPC:
   - Toolchain: buf lint / protoc --descriptor_set_out=/dev/null toolchain probe + in-process AST parser.
   - Checks: Field tag uniqueness, reserved tags, backward compatibility, package namespaces.
   - Anti-Goodhart: Reject missing field numbers, breaking tag mutations.

6. Clean Architecture Boundary Linter:
   - Import graph checker: Forbids domain entities from importing infrastructure/web/ORM frameworks;
     enforces layered flow (API -> Application -> Domain).
   - Anti-Goodhart: Reject dynamic import bypasses (__import__, importlib, eval/exec).

Reward-shaping core matches LspVerifier and SystemsMobileVerifier patterns:
- Diagnostics error code parsing and severity weights
- Degenerate output guards (empty, whitespace, comment-only)
- Anti-Goodhart escape-hatch detection with superset/directives/none modes
- Thread-safe, injectable oracle seams for CI testing without external binaries
"""

from __future__ import annotations

import ast
import contextlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

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

def resolve_sql_toolchain() -> Dict[str, bool]:
    """Check availability of SQL tools (sqlite3 is always available in stdlib)."""
    has_sqlglot = False
    try:
        import sqlglot  # noqa: F401
        has_sqlglot = True
    except ImportError:
        pass
    return {"sqlite3": True, "sqlglot": has_sqlglot}


def resolve_protobuf_toolchain() -> Optional[str]:
    """Locate protoc or buf executable on PATH."""
    return shutil.which("buf") or shutil.which("protoc")


def resolve_spectral_toolchain() -> Optional[str]:
    """Locate spectral executable on PATH."""
    return shutil.which("spectral")


def resolve_openapi_toolchain() -> Dict[str, bool]:
    """Check availability of OpenAPI validation packages."""
    has_spec_validator = False
    has_jsonschema = False
    try:
        import openapi_spec_validator  # noqa: F401
        has_spec_validator = True
    except ImportError:
        pass
    try:
        import jsonschema  # noqa: F401
        has_jsonschema = True
    except ImportError:
        pass
    return {
        "openapi_spec_validator": has_spec_validator,
        "jsonschema": has_jsonschema,
        "spectral": bool(resolve_spectral_toolchain()),
    }


def resolve_graphql_toolchain() -> Dict[str, bool]:
    """Check availability of graphql-core package."""
    has_graphql = False
    try:
        import graphql  # noqa: F401
        has_graphql = True
    except ImportError:
        pass
    return {"graphql_core": has_graphql}


# --------------------------------------------------------------------------- #
# Anti-Goodhart Escape Hatch Detectors
# --------------------------------------------------------------------------- #

# 1. Relational SQL Escape Hatches
SQL_ESCAPE_HATCH_PATTERNS: Mapping[str, re.Pattern] = {
    # Anti-Goodhart: Reject SELECT * (model must explicitly project columns)
    "select_star": re.compile(r"\bSELECT\s+(?:DISTINCT\s+)?\*", re.IGNORECASE),
    # Anti-Goodhart: Reject cartesian joins without ON/USING clauses
    "cartesian_cross_join": re.compile(r"\bCROSS\s+JOIN\b", re.IGNORECASE),
    "cartesian_join_without_on": re.compile(
        r"\b(?:INNER\s+|LEFT\s+|RIGHT\s+|FULL\s+)?JOIN\s+[\w.]+\s+(?!ON\b|USING\b)(?:WHERE\b|ORDER\b|GROUP\b|LIMIT\b|;|$)",
        re.IGNORECASE,
    ),
    # Directives / suppression comments
    "sql_suppress_directive": re.compile(
        r"(?:--\s*(?:noqa|nolint|ignore|suppress|type:\s*ignore)|/\*+.*?(?:noqa|nolint|ignore|suppress).*?\*+/)",
        re.IGNORECASE,
    ),
}

SQL_DIRECTIVE_HATCHES: frozenset[str] = frozenset({"sql_suppress_directive"})


# 2. Data & ML Pipelines Escape Hatches
DATA_PIPELINE_ESCAPE_HATCH_PATTERNS: Mapping[str, re.Pattern] = {
    # Anti-Goodhart: Dynamic shape escapes
    "dynamic_shape_view": re.compile(r"\.(?:view|reshape)\s*\(\s*-1\s*\)"),
    "dynamic_shape_size": re.compile(r"\b(?:torch\.)?Size\s*\(\s*\[\s*-1\s*\]\s*\)"),
    "untyped_any_tensor": re.compile(r":\s*(?:typing\.)?Any\b"),
    # Anti-Goodhart: Bare object dtypes
    "bare_object_dtype": re.compile(
        r"(?:dtype\s*=\s*(?:['\"](?:object|O)['\"]|(?:np\.)?object_\b|object\b|pl\.Object\b)|\.astype\s*\(\s*(?:['\"](?:object|O)['\"]|(?:np\.)?object_\b|object\b)\s*\))"
    ),
    "polars_object_dtype": re.compile(r"\bpl\.Object\b"),
    # Anti-Goodhart: Row-wise iterations
    "row_iteration_iterrows": re.compile(r"\.iterrows\s*\(\s*\)"),
    "row_iteration_itertuples": re.compile(r"\.itertuples\s*\(\s*\)"),
    "row_iteration_apply_axis1": re.compile(r"\.apply\s*\([^)]*axis\s*=\s*1[^)]*\)"),
    # Suppression directives
    "python_suppress_directive": re.compile(r"#\s*(?:type:\s*ignore|noqa|pylint:\s*disable)"),
}

DATA_PIPELINE_DIRECTIVE_HATCHES: frozenset[str] = frozenset({"python_suppress_directive"})


# 3. API Contracts (OpenAPI & JSON Schema) Escape Hatches
OPENAPI_ESCAPE_HATCH_PATTERNS: Mapping[str, re.Pattern] = {
    # Anti-Goodhart: Untyped payloads (schema with empty object or type any)
    "untyped_payload_empty_schema": re.compile(r"schema\s*:\s*\{\s*\}"),
    "untyped_payload_any": re.compile(r"type\s*:\s*['\"]?(?:any|unknown)['\"]?", re.IGNORECASE),
    # Directive suppressions
    "spectral_ignore_directive": re.compile(r"#\s*spectral-disable(?:-next-line)?\b"),
}

OPENAPI_DIRECTIVE_HATCHES: frozenset[str] = frozenset({"spectral_ignore_directive"})


# 4. GraphQL Escape Hatches
GRAPHQL_ESCAPE_HATCH_PATTERNS: Mapping[str, re.Pattern] = {
    # Anti-Goodhart: Unconstrained scalars
    "unconstrained_scalar_any": re.compile(r"\bscalar\s+(?:Any|JSON|Object)\b", re.IGNORECASE),
    # Directive suppressions
    "graphql_suppress_directive": re.compile(r"#\s*(?:graphql-disable|noqa|ignore)\b"),
}

GRAPHQL_DIRECTIVE_HATCHES: frozenset[str] = frozenset({"graphql_suppress_directive"})


# 5. Protocol Buffers & gRPC Escape Hatches
PROTOBUF_ESCAPE_HATCH_PATTERNS: Mapping[str, re.Pattern] = {
    # Anti-Goodhart: Missing field numbers in proto message body
    "missing_field_number": re.compile(
        r"(?:^|[\s;{}])(?:optional\s+|repeated\s+|required\s+)?(?:int32|int64|uint32|uint64|sint32|sint64|fixed32|fixed64|sfixed32|sfixed64|bool|string|bytes|double|float|[A-Z]\w*)\s+[a-z_]\w*\s*;(?!\s*=)",
        re.MULTILINE,
    ),
    # Directive suppressions
    "buf_ignore_directive": re.compile(r"//\s*buf:lint:ignore\b"),
}

PROTOBUF_DIRECTIVE_HATCHES: frozenset[str] = frozenset({"buf_ignore_directive"})


# 6. Clean Architecture Boundary Escape Hatches
ARCHITECTURE_ESCAPE_HATCH_PATTERNS: Mapping[str, re.Pattern] = {
    # Anti-Goodhart: Dynamic import bypasses
    "dynamic_import_dunder": re.compile(r"\b__import__\s*\("),
    "dynamic_import_importlib": re.compile(r"\bimportlib\.import_module\s*\("),
    "dynamic_import_sys_modules": re.compile(r"\bsys\.modules\s*\["),
    "dynamic_import_eval_exec": re.compile(r"\b(?:eval|exec)\s*\("),
    # Suppression directives
    "arch_suppress_directive": re.compile(r"#\s*(?:noqa|type:\s*ignore|arch-ignore)\b"),
}

ARCHITECTURE_DIRECTIVE_HATCHES: frozenset[str] = frozenset({"arch_suppress_directive"})


STRING_TARGETED_HATCHES = frozenset({
    "bare_object_dtype",
    "missing_field_number",
    "untyped_payload_empty_schema",
    "untyped_payload_any",
    "unconstrained_string_pattern",
})

def find_escape_hatches_generic(
    text: str,
    patterns: Mapping[str, re.Pattern],
    directives: frozenset[str],
    *,
    mask_comments_except_directives: bool = True,
    language: str = "python",
) -> List[str]:
    """Scan code for escape hatches, distinguishing directives from code anti-patterns."""
    found: List[str] = []
    if not text:
        return found

    masked = text
    if mask_comments_except_directives:
        try:
            masked = mask_strings_and_comments(text)
        except Exception:
            masked = text

    for name, pat in patterns.items():
        haystack = text if (name in directives or name in STRING_TARGETED_HATCHES) else masked
        if pat.search(haystack):
            found.append(name)

    return sorted(set(found))


def _has_cartesian_join(sql: str) -> bool:
    join_splits = re.split(r"\b(?:INNER\s+|LEFT\s+|RIGHT\s+|FULL\s+)?JOIN\b", sql, flags=re.IGNORECASE)
    if len(join_splits) <= 1:
        return False
    for part in join_splits[1:]:
        clause = re.split(r"\b(?:INNER\s+|LEFT\s+|RIGHT\s+|FULL\s+)?JOIN\b|\bWHERE\b|\bGROUP\b|\bORDER\b|\bLIMIT\b|;|$", part, flags=re.IGNORECASE)[0]
        if not re.search(r"\b(?:ON|USING)\b", clause, flags=re.IGNORECASE):
            return True
    return False

def find_sql_escape_hatches(text: str) -> List[str]:
    """Detect anti-Goodhart SQL escape hatches (SELECT *, cartesian joins)."""
    hatches = find_escape_hatches_generic(
        text, SQL_ESCAPE_HATCH_PATTERNS, SQL_DIRECTIVE_HATCHES, language="sql"
    )
    if "cartesian_join_without_on" not in hatches and _has_cartesian_join(text):
        hatches.append("cartesian_join_without_on")
    return hatches


def find_data_pipeline_escape_hatches(text: str) -> List[str]:
    """Detect anti-Goodhart data/ML pipeline escape hatches."""
    return find_escape_hatches_generic(
        text, DATA_PIPELINE_ESCAPE_HATCH_PATTERNS, DATA_PIPELINE_DIRECTIVE_HATCHES, language="python"
    )


def _check_unconstrained_strings_in_spec(spec_data: Any) -> bool:
    """Recursively check if any schema in OpenAPI spec defines an unconstrained string."""
    if isinstance(spec_data, dict):
        t = spec_data.get("type")
        if t == "string":
            constraints = {"format", "pattern", "minLength", "maxLength", "enum"}
            if not any(k in spec_data for k in constraints):
                return True
        for v in spec_data.values():
            if _check_unconstrained_strings_in_spec(v):
                return True
    elif isinstance(spec_data, list):
        for item in spec_data:
            if _check_unconstrained_strings_in_spec(item):
                return True
    return False


def find_openapi_escape_hatches(text: str) -> List[str]:
    """Detect anti-Goodhart OpenAPI escape hatches (untyped payloads, unconstrained strings)."""
    hatches = find_escape_hatches_generic(
        text, OPENAPI_ESCAPE_HATCH_PATTERNS, OPENAPI_DIRECTIVE_HATCHES, mask_comments_except_directives=False
    )
    parsed = None
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
        except Exception:
            pass
    else:
        try:
            import yaml
            parsed = yaml.safe_load(text)
        except Exception:
            pass

    if parsed is not None and _check_unconstrained_strings_in_spec(parsed):
        if "unconstrained_string" not in hatches:
            hatches.append("unconstrained_string")

    return hatches


def find_graphql_escape_hatches(text: str) -> List[str]:
    """Detect anti-Goodhart GraphQL escape hatches."""
    return find_escape_hatches_generic(
        text, GRAPHQL_ESCAPE_HATCH_PATTERNS, GRAPHQL_DIRECTIVE_HATCHES, mask_comments_except_directives=False
    )


def find_protobuf_escape_hatches(text: str) -> List[str]:
    """Detect anti-Goodhart Protobuf escape hatches."""
    return find_escape_hatches_generic(
        text, PROTOBUF_ESCAPE_HATCH_PATTERNS, PROTOBUF_DIRECTIVE_HATCHES, mask_comments_except_directives=False
    )


def find_architecture_escape_hatches(text: str) -> List[str]:
    """Detect anti-Goodhart clean architecture escape hatches."""
    return find_escape_hatches_generic(
        text, ARCHITECTURE_ESCAPE_HATCH_PATTERNS, ARCHITECTURE_DIRECTIVE_HATCHES, language="python"
    )


# --------------------------------------------------------------------------- #
# Base Contract Verifier (Shared RLVR reward shaping core)
# --------------------------------------------------------------------------- #

class BaseContractVerifier:
    """Abstract base for data engineering and contract verifiers.

    Matches LspVerifier / SystemsMobileVerifier reward shaping:
    - Base reward: 1.0 (clean)
    - Diagnostic errors/warnings reduce reward according to severity weights
    - Degenerate output (empty, whitespace, comment-only) short-circuits to -1.0
    - Escape hatches / anti-Goodhart violations short-circuit to -1.0
    """

    def __init__(
        self,
        *,
        contract_name: str,
        hatch_patterns: Mapping[str, re.Pattern],
        directive_hatches: frozenset[str],
        oracle_factory: Callable[[], Any],
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

        self.contract_name = contract_name
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
        return find_escape_hatches_generic(text, self.hatch_patterns, self.directive_hatches)

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
        """Score one completion against contract rules.

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
                    diags = oracle.diagnostics(artifact, reference=reference, **kwargs)
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
            "contract": self.contract_name,
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

    def __enter__(self) -> "BaseContractVerifier":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# 1. Relational SQL Verifier & In-Memory Execution Oracle
# --------------------------------------------------------------------------- #

def validate_pure_sql_syntax(sql: str) -> List[Diagnostic]:
    """Pure-Python syntax and clause parser when sqlglot is absent and schema is not supplied."""
    diags: List[Diagnostic] = []
    stripped = sql.strip().rstrip(";")
    if not stripped:
        return [
            Diagnostic(
                code="SQL_EMPTY_QUERY",
                line=1,
                col=1,
                message="SQL query is empty",
                offset=0,
                source="sql_parser",
                severity=1,
            )
        ]

    # Balanced parentheses check
    stack: List[int] = []
    for idx, ch in enumerate(stripped):
        if ch == "(":
            stack.append(idx)
        elif ch == ")":
            if not stack:
                return [
                    Diagnostic(
                        code="SQL_SYNTAX_ERROR",
                        line=1,
                        col=idx + 1,
                        message="Unmatched ')' in SQL query",
                        offset=idx,
                        source="sql_parser",
                        severity=1,
                    )
                ]
            stack.pop()
    if stack:
        return [
            Diagnostic(
                code="SQL_SYNTAX_ERROR",
                line=1,
                col=stack[-1] + 1,
                message="Unclosed '(' in SQL query",
                offset=stack[-1],
                source="sql_parser",
                severity=1,
            )
        ]

    tokens = re.findall(r"[\w*]+|[(),;=<>!+-]", stripped)
    upper_tokens = [t.upper() for t in tokens]
    if "SELECT" in upper_tokens:
        sel_idx = upper_tokens.index("SELECT")
        if "FROM" in upper_tokens:
            from_idx = upper_tokens.index("FROM")
            proj_tokens = upper_tokens[sel_idx + 1:from_idx]
            if not proj_tokens or proj_tokens == [","]:
                diags.append(
                    Diagnostic(
                        code="SQL_SYNTAX_ERROR",
                        line=1,
                        col=1,
                        message="SELECT statement missing projection list before FROM",
                        offset=0,
                        source="sql_parser",
                        severity=1,
                    )
                )
            elif proj_tokens[-1] == ",":
                diags.append(
                    Diagnostic(
                        code="SQL_SYNTAX_ERROR",
                        line=1,
                        col=1,
                        message="Trailing comma in SELECT projection list",
                        offset=0,
                        source="sql_parser",
                        severity=1,
                    )
                )

    return diags


class SqliteMemoryOracle:
    """In-memory SQLite execution oracle with deterministic DML verification (<5ms)."""

    def __init__(
        self,
        *,
        schema_ddl: Optional[str] = None,
        dialect: str = "sqlite",
        timeout_s: float = 5.0,
        check_indexes: bool = False,
    ):
        self.schema_ddl = schema_ddl
        self.dialect = dialect.lower()
        self.timeout_s = timeout_s
        self.check_indexes = check_indexes
        self.n_calls = 0
        self.wall_s = 0.0

    def diagnostics(
        self,
        query: str,
        *,
        schema_ddl: Optional[str] = None,
        reference: Optional[str] = None,
        dialect: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Diagnostic]:
        t0 = time.monotonic()
        self.n_calls += 1
        diags: List[Diagnostic] = []
        target_dialect = (dialect or self.dialect).lower()
        ddl = schema_ddl or self.schema_ddl

        # 1. Multi-dialect AST parsing (<3ms) via sqlglot if installed
        try:
            import sqlglot
            try:
                sqlglot.parse(query, read=target_dialect if target_dialect != "sqlite" else None)
            except sqlglot.errors.ParseError as err:
                diags.append(
                    Diagnostic(
                        code="SQL_PARSE_ERROR",
                        line=getattr(err, "line", 1) or 1,
                        col=getattr(err, "col", 1) or 1,
                        message=str(err),
                        offset=0,
                        source="sqlglot",
                        severity=1,
                    )
                )
                self.wall_s += time.monotonic() - t0
                return diags
        except ImportError:
            # When sqlglot is absent and no schema is provided, run pure AST parser
            if not ddl:
                pure_diags = validate_pure_sql_syntax(query)
                if pure_diags:
                    self.wall_s += time.monotonic() - t0
                    return pure_diags

        # 2. In-memory SQLite execution (<5ms) when schema DDL is supplied or query is self-contained
        if ddl or not re.search(r"\bFROM\s+[a-zA-Z_]\w*", query, re.IGNORECASE):
            conn = sqlite3.connect(":memory:")
            try:
                cursor = conn.cursor()
                if ddl:
                    cursor.executescript(ddl)

                query_stripped = query.strip().rstrip(";")
                if not query_stripped:
                    diags.append(
                        Diagnostic(
                            code="SQL_EMPTY_QUERY",
                            line=1,
                            col=1,
                            message="Query is empty",
                            offset=0,
                            source="sqlite",
                            severity=1,
                        )
                    )
                    return diags

                try:
                    if query_stripped.upper().startswith("SELECT") or query_stripped.upper().startswith("WITH"):
                        cursor.execute(f"EXPLAIN {query_stripped}")
                        cursor.execute(query_stripped)
                        res = cursor.fetchall()
                        # Equivalence check against reference query if provided
                        if reference:
                            cursor.execute(reference.strip().rstrip(";"))
                            ref_res = cursor.fetchall()
                            if res != ref_res:
                                diags.append(
                                    Diagnostic(
                                        code="SQL_EQUIVALENCE_MISMATCH",
                                        line=1,
                                        col=1,
                                        message="Query result does not match reference query result",
                                        offset=0,
                                        source="sqlite_execution",
                                        severity=1,
                                    )
                                )

                        if self.check_indexes and ddl:
                            cursor.execute(f"EXPLAIN QUERY PLAN {query_stripped}")
                            plan = [str(row) for row in cursor.fetchall()]
                            plan_str = " ".join(plan)
                            if "SCAN" in plan_str and "INDEX" not in plan_str and "INDEX" in ddl.upper():
                                diags.append(
                                    Diagnostic(
                                        code="SQL_UNINDEXED_SCAN",
                                        line=1,
                                        col=1,
                                        message="Query executes table scan without using available index",
                                        offset=0,
                                        source="sqlite_plan",
                                        severity=2,
                                    )
                                )
                    else:
                        cursor.execute(query_stripped)
                except sqlite3.OperationalError as op_err:
                    msg = str(op_err)
                    code = "SQL_COLUMN_ERROR" if "no such column" in msg else (
                        "SQL_TABLE_ERROR" if "no such table" in msg else "SQL_SYNTAX_ERROR"
                    )
                    diags.append(
                        Diagnostic(
                            code=code,
                            line=1,
                            col=1,
                            message=msg,
                            offset=0,
                            source="sqlite",
                            severity=1,
                        )
                    )
                except sqlite3.Error as sq_err:
                    diags.append(
                        Diagnostic(
                            code="SQL_EXECUTION_ERROR",
                            line=1,
                            col=1,
                            message=str(sq_err),
                            offset=0,
                            source="sqlite",
                            severity=1,
                        )
                    )
            finally:
                conn.close()
                self.wall_s += time.monotonic() - t0

        return diags

    def close(self) -> None:
        pass


class SqlVerifier(BaseContractVerifier):
    """Static and in-memory execution verifier for Relational SQL queries."""

    def __init__(
        self,
        *,
        schema_ddl: Optional[str] = None,
        dialect: str = "sqlite",
        timeout_s: float = 5.0,
        hatches: str = "superset",
        oracle: Any = None,
        check_indexes: bool = False,
        on_error: str = "raise",
        fail_fast: bool = True,
        **reward_kwargs: Any,
    ) -> None:
        super().__init__(
            contract_name="sql",
            hatch_patterns=SQL_ESCAPE_HATCH_PATTERNS,
            directive_hatches=SQL_DIRECTIVE_HATCHES,
            oracle_factory=lambda: SqliteMemoryOracle(
                schema_ddl=schema_ddl, dialect=dialect, timeout_s=timeout_s, check_indexes=check_indexes
            ),
            timeout_s=timeout_s,
            hatches=hatches,
            oracle=oracle,
            on_error=on_error,
            fail_fast=fail_fast,
            **reward_kwargs,
        )

    def find_escape_hatches(self, text: str) -> List[str]:
        return find_sql_escape_hatches(text)


SqlStaticVerifier = SqlVerifier


# --------------------------------------------------------------------------- #
# 2. Data & ML Pipelines Verifier (Pandas, Polars, PyTorch)
# --------------------------------------------------------------------------- #

class DataPipelineOracle:
    """In-process AST visitor and dimensional contract verifier for Data & ML pipelines."""

    def __init__(
        self,
        *,
        known_columns: Optional[Sequence[str]] = None,
        timeout_s: float = 5.0,
    ):
        self.known_columns = set(known_columns) if known_columns else None
        self.timeout_s = timeout_s
        self.n_calls = 0
        self.wall_s = 0.0

    def diagnostics(
        self,
        source: str,
        *,
        known_columns: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> List[Diagnostic]:
        t0 = time.monotonic()
        self.n_calls += 1
        diags: List[Diagnostic] = []
        cols = set(known_columns) if known_columns else self.known_columns

        try:
            tree = ast.parse(source)
        except SyntaxError as syn:
            diags.append(
                Diagnostic(
                    code="PYTHON_SYNTAX_ERROR",
                    line=syn.lineno or 1,
                    col=syn.offset or 1,
                    message=f"Syntax error: {syn.msg}",
                    offset=0,
                    source="ast",
                    severity=1,
                )
            )
            self.wall_s += time.monotonic() - t0
            return diags

        class PipelineVisitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.found_diags: List[Diagnostic] = []

            def visit_Call(self, node: ast.Call) -> None:
                func_name = ""
                if isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                    # Polars/Pandas column selection checks
                    if cols and func_name in ("select", "drop", "col", "with_columns"):
                        candidate_col_nodes = []
                        for arg in node.args:
                            if isinstance(arg, (ast.List, ast.Tuple)):
                                candidate_col_nodes.extend(arg.elts)
                            else:
                                candidate_col_nodes.append(arg)
                        for c_node in candidate_col_nodes:
                            if isinstance(c_node, ast.Constant) and isinstance(c_node.value, str):
                                if c_node.value not in cols:
                                    self.found_diags.append(
                                        Diagnostic(
                                            code="DATA_UNKNOWN_COLUMN",
                                            line=node.lineno,
                                            col=node.col_offset,
                                            message=f"Unknown DataFrame column: {c_node.value!r}",
                                            offset=0,
                                            source="pipeline_schema",
                                            severity=1,
                                        )
                                    )

                # Vectorized operations check
                if func_name == "apply":
                    for kw in node.keywords:
                        if kw.arg == "axis" and isinstance(kw.value, ast.Constant) and kw.value.value == 1:
                            self.found_diags.append(
                                Diagnostic(
                                    code="DATA_UNVECTORIZED_ROW_APPLY",
                                    line=node.lineno,
                                    col=node.col_offset,
                                    message="DataFrame.apply(..., axis=1) is unvectorized row-wise execution",
                                    offset=0,
                                    source="pipeline_vectorize",
                                    severity=1,
                                )
                            )

                # Tensor shape transformation contract checks
                if func_name in ("view", "reshape"):
                    for arg in node.args:
                        if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                            if isinstance(arg.operand, ast.Constant) and arg.operand.value == 1 and len(node.args) == 1:
                                self.found_diags.append(
                                    Diagnostic(
                                        code="TENSOR_DYNAMIC_ESCAPE",
                                        line=node.lineno,
                                        col=node.col_offset,
                                        message="Tensor flatten with unconstrained view(-1) violates dimension contract",
                                        offset=0,
                                        source="tensor_contract",
                                        severity=1,
                                    )
                                )

                # Tensor permute duplicate dimension check
                if func_name == "permute":
                    dims = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, int)]
                    if len(dims) > 1 and len(set(dims)) != len(dims):
                        self.found_diags.append(
                            Diagnostic(
                                code="TENSOR_INVALID_PERMUTE",
                                line=node.lineno,
                                col=node.col_offset,
                                message=f"Tensor permute contains duplicate dimension indices: {dims}",
                                offset=0,
                                source="tensor_contract",
                                severity=1,
                            )
                        )

                self.generic_visit(node)

            def visit_For(self, node: ast.For) -> None:
                if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Attribute):
                    if node.iter.func.attr in ("iterrows", "itertuples"):
                        self.found_diags.append(
                            Diagnostic(
                                code="DATA_ROW_ITERATION",
                                line=node.lineno,
                                col=node.col_offset,
                                message=f"Row iteration via {node.iter.func.attr}() violates vectorized pipeline contract",
                                offset=0,
                                source="pipeline_vectorize",
                                severity=1,
                            )
                        )
                self.generic_visit(node)

            def visit_Subscript(self, node: ast.Subscript) -> None:
                if cols and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                    if node.slice.value not in cols and isinstance(node.value, ast.Name) and "df" in node.value.id.lower():
                        self.found_diags.append(
                            Diagnostic(
                                code="DATA_UNKNOWN_COLUMN",
                                line=node.lineno,
                                col=node.col_offset,
                                message=f"Unknown DataFrame column subscript: {node.slice.value!r}",
                                offset=0,
                                source="pipeline_schema",
                                severity=1,
                            )
                        )
                self.generic_visit(node)

        visitor = PipelineVisitor()
        visitor.visit(tree)
        diags.extend(visitor.found_diags)
        self.wall_s += time.monotonic() - t0
        return diags

    def close(self) -> None:
        pass


class DataPipelineVerifier(BaseContractVerifier):
    """Static AST verifier for Data & ML Pipelines (Pandas, Polars, PyTorch)."""

    def __init__(
        self,
        *,
        known_columns: Optional[Sequence[str]] = None,
        timeout_s: float = 5.0,
        hatches: str = "superset",
        oracle: Any = None,
        on_error: str = "raise",
        fail_fast: bool = True,
        **reward_kwargs: Any,
    ) -> None:
        super().__init__(
            contract_name="data_pipeline",
            hatch_patterns=DATA_PIPELINE_ESCAPE_HATCH_PATTERNS,
            directive_hatches=DATA_PIPELINE_DIRECTIVE_HATCHES,
            oracle_factory=lambda: DataPipelineOracle(
                known_columns=known_columns, timeout_s=timeout_s
            ),
            timeout_s=timeout_s,
            hatches=hatches,
            oracle=oracle,
            on_error=on_error,
            fail_fast=fail_fast,
            **reward_kwargs,
        )

    def find_escape_hatches(self, text: str) -> List[str]:
        return find_data_pipeline_escape_hatches(text)


# --------------------------------------------------------------------------- #
# 3. API Contracts (OpenAPI v3.1 & JSON Schema) Verifier
# --------------------------------------------------------------------------- #

class OpenApiContractOracle:
    """In-process OpenAPI v3.0 / v3.1 and JSON Schema contract validator."""

    def __init__(self, *, timeout_s: float = 5.0):
        self.timeout_s = timeout_s
        self.n_calls = 0
        self.wall_s = 0.0

    def diagnostics(self, spec_text: str, **kwargs: Any) -> List[Diagnostic]:
        t0 = time.monotonic()
        self.n_calls += 1
        diags: List[Diagnostic] = []

        spec_data: Optional[Dict[str, Any]] = None
        stripped = spec_text.strip()
        if stripped.startswith("{"):
            try:
                spec_data = json.loads(stripped)
            except json.JSONDecodeError as jde:
                diags.append(
                    Diagnostic(
                        code="OPENAPI_JSON_SYNTAX_ERROR",
                        line=jde.lineno,
                        col=jde.colno,
                        message=f"JSON parse error: {jde.msg}",
                        offset=0,
                        source="openapi",
                        severity=1,
                    )
                )
                self.wall_s += time.monotonic() - t0
                return diags
        else:
            try:
                import yaml
                spec_data = yaml.safe_load(spec_text)
            except Exception as y_err:
                diags.append(
                    Diagnostic(
                        code="OPENAPI_YAML_SYNTAX_ERROR",
                        line=1,
                        col=1,
                        message=f"YAML parse error: {y_err}",
                        offset=0,
                        source="openapi",
                        severity=1,
                    )
                )
                self.wall_s += time.monotonic() - t0
                return diags

        if not isinstance(spec_data, dict):
            diags.append(
                Diagnostic(
                    code="OPENAPI_ROOT_INVALID",
                    line=1,
                    col=1,
                    message="OpenAPI specification root must be an object",
                    offset=0,
                    source="openapi",
                    severity=1,
                )
            )
            self.wall_s += time.monotonic() - t0
            return diags

        openapi_ver = spec_data.get("openapi")
        if not openapi_ver or not str(openapi_ver).startswith("3."):
            diags.append(
                Diagnostic(
                    code="OPENAPI_VERSION_INVALID",
                    line=1,
                    col=1,
                    message=f"Expected OpenAPI v3.x, found: {openapi_ver!r}",
                    offset=0,
                    source="openapi",
                    severity=1,
                )
            )

        info = spec_data.get("info")
        if not isinstance(info, dict) or "title" not in info or "version" not in info:
            diags.append(
                Diagnostic(
                    code="OPENAPI_INFO_MISSING",
                    line=1,
                    col=1,
                    message="OpenAPI spec must contain 'info' with 'title' and 'version'",
                    offset=0,
                    source="openapi",
                    severity=1,
                )
            )

        paths = spec_data.get("paths")
        if not isinstance(paths, dict):
            diags.append(
                Diagnostic(
                    code="OPENAPI_PATHS_MISSING",
                    line=1,
                    col=1,
                    message="OpenAPI spec must contain 'paths' object",
                    offset=0,
                    source="openapi",
                    severity=1,
                )
            )
            self.wall_s += time.monotonic() - t0
            return diags

        path_param_re = re.compile(r"\{([^{}]+)\}")
        valid_http_methods = {"get", "post", "put", "delete", "patch", "options", "head"}

        for path_str, path_item in paths.items():
            if not isinstance(path_item, dict):
                continue

            expected_params = set(path_param_re.findall(path_str))

            path_level_params: Dict[str, dict] = {}
            for p in path_item.get("parameters", []):
                if isinstance(p, dict) and p.get("in") == "path":
                    path_level_params[p.get("name", "")] = p

            for method_name, op in path_item.items():
                if method_name.lower() not in valid_http_methods:
                    continue
                if not isinstance(op, dict):
                    continue

                op_params = dict(path_level_params)
                for p in op.get("parameters", []):
                    if isinstance(p, dict) and p.get("in") == "path":
                        op_params[p.get("name", "")] = p

                for param_name in expected_params:
                    if param_name not in op_params:
                        diags.append(
                            Diagnostic(
                                code="OPENAPI_UNBOUND_PATH_PARAM",
                                line=1,
                                col=1,
                                message=f"Path parameter {param_name!r} in {path_str!r} is not declared in parameters",
                                offset=0,
                                source="openapi",
                                severity=1,
                            )
                        )
                    else:
                        param_obj = op_params[param_name]
                        if not param_obj.get("required", False):
                            diags.append(
                                Diagnostic(
                                    code="OPENAPI_PATH_PARAM_NOT_REQUIRED",
                                    line=1,
                                    col=1,
                                    message=f"Path parameter {param_name!r} in {path_str!r} must have 'required: true'",
                                    offset=0,
                                    source="openapi",
                                    severity=1,
                                )
                            )

                responses = op.get("responses", {})
                if not isinstance(responses, dict):
                    diags.append(
                        Diagnostic(
                            code="OPENAPI_RESPONSES_MISSING",
                            line=1,
                            col=1,
                            message=f"Operation {method_name.upper()} {path_str} has invalid 'responses'",
                            offset=0,
                            source="openapi",
                            severity=1,
                        )
                    )
                else:
                    has_2xx = any(
                        str(code).startswith("2") or str(code).lower() == "default"
                        for code in responses.keys()
                    )
                    if not has_2xx:
                        diags.append(
                            Diagnostic(
                                code="OPENAPI_MISSING_2XX_RESPONSE",
                                line=1,
                                col=1,
                                message=f"Operation {method_name.upper()} {path_str} lacks a 2xx or default response definition",
                                offset=0,
                                source="openapi",
                                severity=1,
                            )
                        )

        self.wall_s += time.monotonic() - t0
        return diags

    def close(self) -> None:
        pass


class OpenApiVerifier(BaseContractVerifier):
    """Contract verifier for OpenAPI v3.1 & JSON Schema."""

    def __init__(
        self,
        *,
        timeout_s: float = 5.0,
        hatches: str = "superset",
        oracle: Any = None,
        on_error: str = "raise",
        fail_fast: bool = True,
        **reward_kwargs: Any,
    ) -> None:
        super().__init__(
            contract_name="openapi",
            hatch_patterns=OPENAPI_ESCAPE_HATCH_PATTERNS,
            directive_hatches=OPENAPI_DIRECTIVE_HATCHES,
            oracle_factory=lambda: OpenApiContractOracle(timeout_s=timeout_s),
            timeout_s=timeout_s,
            hatches=hatches,
            oracle=oracle,
            on_error=on_error,
            fail_fast=fail_fast,
            **reward_kwargs,
        )

    def find_escape_hatches(self, text: str) -> List[str]:
        return find_openapi_escape_hatches(text)


# --------------------------------------------------------------------------- #
# 4. GraphQL Schemas & Operations Verifier
# --------------------------------------------------------------------------- #

class GraphQLContractOracle:
    """In-process GraphQL SDL and Query validation oracle."""

    def __init__(self, *, max_query_depth: int = 5, timeout_s: float = 5.0):
        self.max_query_depth = max_query_depth
        self.timeout_s = timeout_s
        self.n_calls = 0
        self.wall_s = 0.0

    def diagnostics(self, doc_text: str, **kwargs: Any) -> List[Diagnostic]:
        t0 = time.monotonic()
        self.n_calls += 1
        diags: List[Diagnostic] = []
        stripped = doc_text.strip()
        if not stripped:
            diags.append(
                Diagnostic(
                    code="GRAPHQL_EMPTY_DOCUMENT",
                    line=1,
                    col=1,
                    message="GraphQL document is empty",
                    offset=0,
                    source="graphql",
                    severity=1,
                )
            )
            self.wall_s += time.monotonic() - t0
            return diags

        try:
            import graphql
            try:
                if "type " in stripped or "schema " in stripped or "interface " in stripped:
                    schema = graphql.build_schema(doc_text)
                    errors = graphql.validate_schema(schema)
                    for err in errors:
                        diags.append(
                            Diagnostic(
                                code="GRAPHQL_SCHEMA_ERROR",
                                line=getattr(err, "locations", [None])[0].line if getattr(err, "locations", None) else 1,
                                col=getattr(err, "locations", [None])[0].column if getattr(err, "locations", None) else 1,
                                message=str(err),
                                offset=0,
                                source="graphql_core",
                                severity=1,
                            )
                        )
                else:
                    graphql.parse(doc_text)
            except Exception as g_err:
                diags.append(
                    Diagnostic(
                        code="GRAPHQL_SYNTAX_ERROR",
                        line=getattr(g_err, "locations", [None])[0].line if getattr(g_err, "locations", None) else 1,
                        col=getattr(g_err, "locations", [None])[0].column if getattr(g_err, "locations", None) else 1,
                        message=str(g_err),
                        offset=0,
                        source="graphql_core",
                        severity=1,
                    )
                )
                self.wall_s += time.monotonic() - t0
                return diags
        except ImportError:
            pass

        stack: List[Tuple[str, int, int]] = []
        line_num = 1
        col_num = 1
        depth = 0
        max_depth_seen = 0

        for idx, ch in enumerate(doc_text):
            if ch == chr(10):
                line_num += 1
                col_num = 1
                continue
            if ch == "{":
                depth += 1
                if depth > max_depth_seen:
                    max_depth_seen = depth
                stack.append((ch, line_num, col_num))
            elif ch == "}":
                depth = max(0, depth - 1)
                if not stack or stack[-1][0] != "{":
                    diags.append(
                        Diagnostic(
                            code="GRAPHQL_UNMATCHED_RBRACE",
                            line=line_num,
                            col=col_num,
                            message="Unmatched closing brace '}' in GraphQL document",
                            offset=idx,
                            source="graphql_parser",
                            severity=1,
                        )
                    )
                else:
                    stack.pop()
            col_num += 1

        if stack:
            for unclosed, l_n, c_n in stack:
                diags.append(
                    Diagnostic(
                        code="GRAPHQL_UNCLOSED_LBRACE",
                        line=l_n,
                        col=c_n,
                        message="Unclosed brace '{' in GraphQL document",
                        offset=0,
                        source="graphql_parser",
                        severity=1,
                    )
                )

        if max_depth_seen > self.max_query_depth:
            diags.append(
                Diagnostic(
                    code="GRAPHQL_CIRCULAR_QUERY_DEPTH",
                    line=1,
                    col=1,
                    message=f"GraphQL query depth {max_depth_seen} exceeds maximum allowable depth {self.max_query_depth}",
                    offset=0,
                    source="graphql_ast",
                    severity=1,
                )
            )

        self.wall_s += time.monotonic() - t0
        return diags

    def close(self) -> None:
        pass


class GraphQLVerifier(BaseContractVerifier):
    """Contract verifier for GraphQL schemas (SDL) and operations."""

    def __init__(
        self,
        *,
        max_query_depth: int = 5,
        timeout_s: float = 5.0,
        hatches: str = "superset",
        oracle: Any = None,
        on_error: str = "raise",
        fail_fast: bool = True,
        **reward_kwargs: Any,
    ) -> None:
        super().__init__(
            contract_name="graphql",
            hatch_patterns=GRAPHQL_ESCAPE_HATCH_PATTERNS,
            directive_hatches=GRAPHQL_DIRECTIVE_HATCHES,
            oracle_factory=lambda: GraphQLContractOracle(
                max_query_depth=max_query_depth, timeout_s=timeout_s
            ),
            timeout_s=timeout_s,
            hatches=hatches,
            oracle=oracle,
            on_error=on_error,
            fail_fast=fail_fast,
            **reward_kwargs,
        )

    def find_escape_hatches(self, text: str) -> List[str]:
        return find_graphql_escape_hatches(text)


# --------------------------------------------------------------------------- #
# 5. Protocol Buffers & gRPC Verifier
# --------------------------------------------------------------------------- #

class ProtobufContractOracle:
    """In-process Protobuf AST and toolchain validator."""

    def __init__(self, *, timeout_s: float = 5.0):
        self.timeout_s = timeout_s
        self.n_calls = 0
        self.wall_s = 0.0

    def diagnostics(
        self,
        proto_text: str,
        *,
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Diagnostic]:
        t0 = time.monotonic()
        self.n_calls += 1
        diags: List[Diagnostic] = []

        if not re.search(r"\bsyntax\s*=\s*['\"]proto[23]['\"]\s*;", proto_text):
            diags.append(
                Diagnostic(
                    code="PROTO_MISSING_SYNTAX",
                    line=1,
                    col=1,
                    message="Missing syntax = proto3 declaration at top of file",
                    offset=0,
                    source="protobuf",
                    severity=1,
                )
            )

        msg_re = re.compile(r"\bmessage\s+(\w+)\s*\{([^}]+)\}")
        field_re = re.compile(r"(?:optional\s+|repeated\s+|required\s+)?(\w+)\s+(\w+)\s*=\s*(\d+)\s*;")
        reserved_re = re.compile(r"\breserved\s+([^;]+);")

        field_tag_map: Dict[str, Dict[str, int]] = {}
        reserved_tags: Dict[str, Set[int]] = {}

        for m_msg in msg_re.finditer(proto_text):
            msg_name = m_msg.group(1)
            body = m_msg.group(2)
            tags_seen: Dict[int, str] = {}
            res_tags: Set[int] = set()
            res_names: Set[str] = set()
            field_tag_map[msg_name] = {}
            reserved_tags[msg_name] = res_tags

            for m_res in reserved_re.finditer(body):
                for item in m_res.group(1).split(","):
                    item = item.strip().strip('"\x27')
                    if item.isdigit():
                        res_tags.add(int(item))
                    elif "to" in item:
                        parts = item.split("to")
                        if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
                            for t_n in range(int(parts[0].strip()), int(parts[1].strip()) + 1):
                                res_tags.add(t_n)
                    else:
                        res_names.add(item)

            for m_f in field_re.finditer(body):
                f_type = m_f.group(1)
                f_name = m_f.group(2)
                f_tag = int(m_f.group(3))

                if f_tag in tags_seen:
                    diags.append(
                        Diagnostic(
                            code="PROTO_DUPLICATE_TAG",
                            line=1,
                            col=1,
                            message=f"Duplicate field tag {f_tag} in message {msg_name} (shared by {tags_seen[f_tag]!r} and {f_name!r})",
                            offset=0,
                            source="protobuf",
                            severity=1,
                        )
                    )
                else:
                    tags_seen[f_tag] = f_name

                if f_tag in res_tags:
                    diags.append(
                        Diagnostic(
                            code="PROTO_RESERVED_TAG_USED",
                            line=1,
                            col=1,
                            message=f"Field {f_name!r} uses reserved tag {f_tag} in message {msg_name}",
                            offset=0,
                            source="protobuf",
                            severity=1,
                        )
                    )

                if f_name in res_names:
                    diags.append(
                        Diagnostic(
                            code="PROTO_RESERVED_NAME_USED",
                            line=1,
                            col=1,
                            message=f"Field {f_name!r} uses reserved name in message {msg_name}",
                            offset=0,
                            source="protobuf",
                            severity=1,
                        )
                    )

                if f_tag < 1 or f_tag > 536870911 or (19000 <= f_tag <= 19999):
                    diags.append(
                        Diagnostic(
                            code="PROTO_INVALID_TAG_NUMBER",
                            line=1,
                            col=1,
                            message=f"Field tag {f_tag} is out of valid range (1..536870911, excluding 19000..19999)",
                            offset=0,
                            source="protobuf",
                            severity=1,
                        )
                    )

                field_tag_map[msg_name][f_name] = f_tag

        if reference:
            for msg_name, f_tags in ref_oracle_tags(reference).items():
                if msg_name in field_tag_map:
                    current_tags = field_tag_map[msg_name]
                    for ref_fname, ref_ftag in f_tags.items():
                        if ref_fname in current_tags:
                            if current_tags[ref_fname] != ref_ftag:
                                diags.append(
                                    Diagnostic(
                                        code="PROTO_BREAKING_TAG_MUTATION",
                                        line=1,
                                        col=1,
                                        message=f"Breaking mutation: tag for field {ref_fname!r} in {msg_name} changed from {ref_ftag} to {current_tags[ref_fname]}",
                                        offset=0,
                                        source="protobuf_compat",
                                        severity=1,
                                    )
                                )
                        else:
                            if ref_ftag not in reserved_tags.get(msg_name, set()):
                                diags.append(
                                    Diagnostic(
                                        code="PROTO_UNRESERVED_REMOVED_TAG",
                                        line=1,
                                        col=1,
                                        message=f"Field {ref_fname!r} (tag {ref_ftag}) removed from {msg_name} without being marked reserved",
                                        offset=0,
                                        source="protobuf_compat",
                                        severity=1,
                                    )
                                )

        self.wall_s += time.monotonic() - t0
        return diags

    def close(self) -> None:
        pass


def ref_oracle_tags(proto_text: str) -> Dict[str, Dict[str, int]]:
    """Helper to extract {message: {field_name: tag}} from reference proto."""
    msg_re = re.compile(r"\bmessage\s+(\w+)\s*\{([^}]+)\}")
    field_re = re.compile(r"(?:optional\s+|repeated\s+|required\s+)?(\w+)\s+(\w+)\s*=\s*(\d+)\s*;")
    res: Dict[str, Dict[str, int]] = {}
    for m in msg_re.finditer(proto_text):
        msg_name = m.group(1)
        res[msg_name] = {}
        for f in field_re.finditer(m.group(2)):
            res[msg_name][f.group(2)] = int(f.group(3))
    return res


class ProtobufVerifier(BaseContractVerifier):
    """Contract verifier for Protocol Buffers & gRPC definitions."""

    def __init__(
        self,
        *,
        timeout_s: float = 5.0,
        hatches: str = "superset",
        oracle: Any = None,
        on_error: str = "raise",
        fail_fast: bool = True,
        **reward_kwargs: Any,
    ) -> None:
        super().__init__(
            contract_name="protobuf",
            hatch_patterns=PROTOBUF_ESCAPE_HATCH_PATTERNS,
            directive_hatches=PROTOBUF_DIRECTIVE_HATCHES,
            oracle_factory=lambda: ProtobufContractOracle(timeout_s=timeout_s),
            timeout_s=timeout_s,
            hatches=hatches,
            oracle=oracle,
            on_error=on_error,
            fail_fast=fail_fast,
            **reward_kwargs,
        )

    def find_escape_hatches(self, text: str) -> List[str]:
        return find_protobuf_escape_hatches(text)


# --------------------------------------------------------------------------- #
# 6. Clean Architecture Boundary Linter
# --------------------------------------------------------------------------- #

FORBIDDEN_DOMAIN_FRAMEWORKS = frozenset({
    "fastapi", "flask", "django", "starlette", "tornado", "aiohttp",
    "requests", "httpx", "urllib3",
    "sqlalchemy", "tortoise", "peewee", "motor", "pymongo", "redis",
    "celery", "boto3",
})

FORBIDDEN_DOMAIN_LAYERS = frozenset({
    "infrastructure", "infra", "adapters", "presentation", "api", "web",
    "controllers", "application", "services",
})

FORBIDDEN_APPLICATION_LAYERS = frozenset({
    "infrastructure", "infra", "adapters", "presentation", "api", "web",
    "controllers", "fastapi", "flask",
})


class CleanArchitectureOracle:
    """Import graph and architectural boundary checker."""

    def __init__(self, *, layer: str = "domain", timeout_s: float = 5.0):
        self.layer = layer.lower()
        self.timeout_s = timeout_s
        self.n_calls = 0
        self.wall_s = 0.0

    def diagnostics(
        self,
        source: str,
        *,
        layer: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Diagnostic]:
        t0 = time.monotonic()
        self.n_calls += 1
        diags: List[Diagnostic] = []
        target_layer = (layer or self.layer).lower()

        try:
            tree = ast.parse(source)
        except SyntaxError as syn:
            diags.append(
                Diagnostic(
                    code="PYTHON_SYNTAX_ERROR",
                    line=syn.lineno or 1,
                    col=syn.offset or 1,
                    message=f"Syntax error: {syn.msg}",
                    offset=0,
                    source="ast",
                    severity=1,
                )
            )
            self.wall_s += time.monotonic() - t0
            return diags

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    pkg_root = alias.name.split(".")[0].lower()
                    if target_layer == "domain":
                        if pkg_root in FORBIDDEN_DOMAIN_FRAMEWORKS:
                            diags.append(
                                Diagnostic(
                                    code="ARCH_DOMAIN_FRAMEWORK_IMPORT",
                                    line=node.lineno,
                                    col=node.col_offset,
                                    message=f"Domain layer must not import framework {pkg_root!r}",
                                    offset=0,
                                    source="clean_architecture",
                                    severity=1,
                                )
                            )
                        if pkg_root in FORBIDDEN_DOMAIN_LAYERS:
                            diags.append(
                                Diagnostic(
                                    code="ARCH_LAYER_INVERSION",
                                    line=node.lineno,
                                    col=node.col_offset,
                                    message=f"Domain layer must not import outer layer {pkg_root!r}",
                                    offset=0,
                                    source="clean_architecture",
                                    severity=1,
                                )
                            )
                    elif target_layer == "application":
                        if pkg_root in FORBIDDEN_APPLICATION_LAYERS:
                            diags.append(
                                Diagnostic(
                                    code="ARCH_LAYER_INVERSION",
                                    line=node.lineno,
                                    col=node.col_offset,
                                    message=f"Application layer must not import presentation/web framework {pkg_root!r}",
                                    offset=0,
                                    source="clean_architecture",
                                    severity=1,
                                )
                            )

            elif isinstance(node, ast.ImportFrom):
                mod_name = (node.module or "").lower()
                pkg_root = mod_name.split(".")[0] if mod_name else ""
                if target_layer == "domain":
                    if pkg_root in FORBIDDEN_DOMAIN_FRAMEWORKS:
                        diags.append(
                            Diagnostic(
                                code="ARCH_DOMAIN_FRAMEWORK_IMPORT",
                                line=node.lineno,
                                col=node.col_offset,
                                message=f"Domain layer must not import framework {pkg_root!r}",
                                offset=0,
                                source="clean_architecture",
                                severity=1,
                            )
                        )
                    if pkg_root in FORBIDDEN_DOMAIN_LAYERS:
                        diags.append(
                            Diagnostic(
                                code="ARCH_LAYER_INVERSION",
                                line=node.lineno,
                                col=node.col_offset,
                                message=f"Domain layer must not import outer layer {pkg_root!r}",
                                offset=0,
                                source="clean_architecture",
                                severity=1,
                            )
                        )
                elif target_layer == "application":
                    if pkg_root in FORBIDDEN_APPLICATION_LAYERS:
                        diags.append(
                            Diagnostic(
                                code="ARCH_LAYER_INVERSION",
                                line=node.lineno,
                                col=node.col_offset,
                                message=f"Application layer must not import presentation/web layer {pkg_root!r}",
                                offset=0,
                                source="clean_architecture",
                                severity=1,
                            )
                        )

        self.wall_s += time.monotonic() - t0
        return diags

    def close(self) -> None:
        pass


class CleanArchitectureVerifier(BaseContractVerifier):
    """Linter for Clean Architecture layers and dependency inversions."""

    def __init__(
        self,
        *,
        layer: str = "domain",
        timeout_s: float = 5.0,
        hatches: str = "superset",
        oracle: Any = None,
        on_error: str = "raise",
        fail_fast: bool = True,
        **reward_kwargs: Any,
    ) -> None:
        super().__init__(
            contract_name="clean_architecture",
            hatch_patterns=ARCHITECTURE_ESCAPE_HATCH_PATTERNS,
            directive_hatches=ARCHITECTURE_DIRECTIVE_HATCHES,
            oracle_factory=lambda: CleanArchitectureOracle(layer=layer, timeout_s=timeout_s),
            timeout_s=timeout_s,
            hatches=hatches,
            oracle=oracle,
            on_error=on_error,
            fail_fast=fail_fast,
            **reward_kwargs,
        )

    def find_escape_hatches(self, text: str) -> List[str]:
        return find_architecture_escape_hatches(text)


ArchitectureBoundaryVerifier = CleanArchitectureVerifier


# --------------------------------------------------------------------------- #
# Unified Data Contracts Verifier (Auto-dispatching multi-contract verifier)
# --------------------------------------------------------------------------- #

class DataContractsVerifier:
    """Unified multi-contract verifier for Data Engineering, Contracts & Query verification."""

    def __init__(
        self,
        *,
        contract_type: str = "auto",
        fail_fast: bool = True,
        hatches: str = "superset",
        **reward_kwargs: Any,
    ) -> None:
        self.contract_type = contract_type.lower()
        self.fail_fast = fail_fast
        self.hatches = hatches
        self.reward_kwargs = reward_kwargs

        self._verifiers: Dict[str, BaseContractVerifier] = {
            "sql": SqlVerifier(fail_fast=fail_fast, hatches=hatches, **reward_kwargs),
            "data_pipeline": DataPipelineVerifier(fail_fast=fail_fast, hatches=hatches, **reward_kwargs),
            "openapi": OpenApiVerifier(fail_fast=fail_fast, hatches=hatches, **reward_kwargs),
            "graphql": GraphQLVerifier(fail_fast=fail_fast, hatches=hatches, **reward_kwargs),
            "protobuf": ProtobufVerifier(fail_fast=fail_fast, hatches=hatches, **reward_kwargs),
            "clean_architecture": CleanArchitectureVerifier(fail_fast=fail_fast, hatches=hatches, **reward_kwargs),
        }
        self._n_samples = 0

    def detect_contract_type(self, text: str) -> str:
        """Infer the contract type from text keywords and structural markers."""
        stripped = text.strip()
        if "syntax" in stripped and ('"proto3"' in stripped or '"proto2"' in stripped):
            return "protobuf"
        if '"openapi"' in stripped or "openapi:" in stripped or (
            stripped.startswith("{") and '"paths"' in stripped
        ):
            return "openapi"
        if "type Query" in stripped or "type Mutation" in stripped or stripped.startswith("query ") or stripped.startswith("mutation "):
            return "graphql"
        if re.search(r"\b(?:SELECT|INSERT|UPDATE|DELETE|CREATE\s+TABLE)\b", stripped, re.IGNORECASE):
            return "sql"
        if "from domain" in stripped or "import domain" in stripped or "class " in stripped:
            if any(f in stripped for f in ("fastapi", "sqlalchemy", "django", "flask", "infrastructure")):
                return "clean_architecture"
        if "pl.col" in stripped or "torch." in stripped or "pd.DataFrame" in stripped:
            return "data_pipeline"
        return "sql"

    def reward(
        self,
        completion: str,
        reference: Optional[str] = None,
        *,
        prompt: str = "",
        contract: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[float]:
        self._n_samples += 1
        target_kind = (contract or self.contract_type).lower()
        if target_kind == "auto":
            target_kind = self.detect_contract_type(f"{prompt}{completion}")
        v = self._verifiers.get(target_kind, self._verifiers["sql"])
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

    def __enter__(self) -> "DataContractsVerifier":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
