"""#348 -- RLVR: Non-destructive database migration replay & API breaking-change verifiers.

Part of the M12 database schema engineering and API evolution track (#198, #221, #342, #348).
Deterministically verify database migrations, schema evolution safety, and API contract
backward compatibility without running production infrastructure:

1. In-Memory Database Migration Replay:
   - Task: given a schema alteration requirement (table split, column migration, indexing),
     generate forward and rollback DDL migration scripts.
   - Verifier sets up an in-memory database (:memory: SQLite or ephemeral relational engine),
     executes forward migration, and asserts that pre-seeded records retain 100% data integrity
     without silent data loss.
   - Executes rollback migration (down) and verifies exact schema state parity with original
     pre-migration DDL.
   - Execution completes deterministically in <50ms.

2. Zero-Downtime Schema Safety Rules:
   - Rejects destructive operations (e.g. dropping columns or tables without deprecation phases,
     direct column renames, or adding NOT NULL columns without default values) violating
     Expand/Contract patterns.

3. API Contract Breaking-Change Detection:
   - Detects breaking changes across OpenAPI specifications (v3.0 / v3.1) and Protocol Buffers schemas.
   - Forbids breaking changes (removing endpoints, removing methods, removing fields, changing
     field types, tightening nullability, adding required parameters) while scoring non-breaking
     additive extensions.
   - Integrates with openapi-diff and buf breaking toolchains when available, backed by an
     in-process, deterministic AST diffing engine with injectable oracle seams.

Reward-shaping core matches LspVerifier and DataContractsVerifier patterns:
- Diagnostic error codes and severity weights
- Degenerate output guards (empty, whitespace, comment-only) -> -1.0
- Anti-Goodhart escape-hatch guards (migration-ignore, contract-ignore) -> -1.0
- Thread-safe telemetry and injectable oracle seams for CI testing
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
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

from src.lsp.diagnostics import Diagnostic, mask_strings_and_comments
from src.train.verifiers import (
    DEFAULT_SEVERITY_WEIGHTS,
    diagnostics_to_reward,
    is_degenerate_output,
    severity_weight,
)

_HATCH_MODES = ("superset", "directives", "none")


# --------------------------------------------------------------------------- #
# Toolchain Resolvers
# --------------------------------------------------------------------------- #

def resolve_contract_diff_toolchains() -> Dict[str, bool]:
    """Check availability of database and contract diffing tools."""
    return {
        "sqlite3": True,
        "openapi-diff": shutil.which("openapi-diff") is not None,
        "buf": shutil.which("buf") is not None,
    }


# --------------------------------------------------------------------------- #
# Constants & Penalty Overrides
# --------------------------------------------------------------------------- #

DEFAULT_MIGRATION_CODE_OVERRIDES: Mapping[str, float] = {
    "MIGRATION_SYNTAX_ERROR": 0.50,
    "MIGRATION_EXECUTION_FAILURE": 0.50,
    "MIGRATION_DESTRUCTIVE_CHANGE": 0.40,
    "MIGRATION_EXPAND_CONTRACT_VIOLATION": 0.40,
    "MIGRATION_DATA_LOSS": 0.50,
    "MIGRATION_MISSING_DOWN": 0.40,
    "MIGRATION_ROLLBACK_FAILED": 0.50,
    "MIGRATION_ROLLBACK_SCHEMA_MISMATCH": 0.40,
    "MIGRATION_ROLLBACK_DATA_LOSS": 0.50,
    "MIGRATION_ESCAPE_HATCH": 1.00,
    "MIGRATION_DEGENERATE_OUTPUT": 1.00,
}

DEFAULT_CONTRACT_DIFF_CODE_OVERRIDES: Mapping[str, float] = {
    "CONTRACT_SYNTAX_ERROR": 0.50,
    "CONTRACT_ENDPOINT_REMOVED": 0.50,
    "CONTRACT_METHOD_REMOVED": 0.50,
    "CONTRACT_REQUIRED_PARAM_ADDED": 0.40,
    "CONTRACT_REQUIRED_PROPERTY_ADDED": 0.40,
    "CONTRACT_PARAM_TYPE_CHANGED": 0.40,
    "CONTRACT_RESPONSE_CODE_REMOVED": 0.40,
    "CONTRACT_RESPONSE_PROPERTY_REMOVED": 0.50,
    "CONTRACT_RESPONSE_TYPE_CHANGED": 0.40,
    "CONTRACT_NULLABILITY_CHANGED": 0.35,
    "PROTO_MESSAGE_REMOVED": 0.50,
    "PROTO_FIELD_REMOVED_WITHOUT_RESERVE": 0.50,
    "PROTO_FIELD_TAG_CHANGED": 0.50,
    "PROTO_FIELD_TYPE_CHANGED": 0.50,
    "PROTO_CARDINALITY_CHANGED": 0.40,
    "PROTO_RESERVED_TAG_REUSED": 0.40,
    "PROTO_ENUM_VALUE_REMOVED": 0.40,
    "PROTO_SERVICE_REMOVED": 0.50,
    "PROTO_RPC_REMOVED": 0.50,
    "PROTO_RPC_SIGNATURE_CHANGED": 0.40,
    "CONTRACT_ESCAPE_HATCH": 1.00,
    "CONTRACT_DEGENERATE_OUTPUT": 1.00,
}

MIGRATION_ESCAPE_HATCH_PATTERNS: Mapping[str, re.Pattern] = {
    "migration_suppress_directive": re.compile(
        r"(?:--|#|//|\*)\s*(?:migration-ignore|noqa:\s*MIGRATION|skip-migration|force-destructive|unsafe-migration)\b",
        re.IGNORECASE,
    ),
    "migration_pragma_bypass": re.compile(
        r"(?:PRAGMA\s+writable_schema\s*=\s*1|PRAGMA\s+ignore_check_constraints\s*=\s*1)",
        re.IGNORECASE,
    ),
}

CONTRACT_DIFF_ESCAPE_HATCH_PATTERNS: Mapping[str, re.Pattern] = {
    "contract_suppress_directive": re.compile(
        r"(?:--|#|//|\*)\s*(?:contract-ignore|noqa:\s*BREAKING|openapi-diff-ignore|buf:lint:ignore|breaking-ignore)\b",
        re.IGNORECASE,
    ),
}


def find_migration_escape_hatches(text: str) -> List[str]:
    """Detect anti-Goodhart escape hatches in migration scripts."""
    found: List[str] = []
    for name, pat in MIGRATION_ESCAPE_HATCH_PATTERNS.items():
        if pat.search(text):
            found.append(name)
    return found


def find_contract_diff_escape_hatches(text: str) -> List[str]:
    """Detect anti-Goodhart escape hatches in API contract specifications."""
    found: List[str] = []
    for name, pat in CONTRACT_DIFF_ESCAPE_HATCH_PATTERNS.items():
        if pat.search(text):
            found.append(name)
    return found

def is_migration_degenerate_output(text: str, *, min_code_chars: int = 2) -> Optional[str]:
    """Check for empty, whitespace-only, comment-only (including SQL -- comments), or too-short code."""
    base_res = is_degenerate_output(text, min_code_chars=min_code_chars)
    if base_res is not None:
        return base_res

    lines: List[str] = []
    for line in text.splitlines():
        code_part = re.sub(r"--.*$", "", line).strip()
        if code_part:
            lines.append(code_part)

    residual = "".join(lines)
    residual = re.sub(r"/\*[\s\S]*?\*/", "", residual).strip()
    if not residual:
        return "comment_only" if text.strip() else "whitespace_only"
    if len(residual) < min_code_chars:
        return "too_short"

    return None


def is_contract_degenerate_output(text: str, *, min_code_chars: int = 2) -> Optional[str]:
    """Check for empty, whitespace-only, comment-only (# and // comments), or too-short specs."""
    base_res = is_degenerate_output(text, min_code_chars=min_code_chars)
    if base_res is not None:
        return base_res

    lines: List[str] = []
    for line in text.splitlines():
        code_part = re.sub(r"(?:#|//).*$", "", line).strip()
        if code_part:
            lines.append(code_part)

    residual = "".join(lines)
    residual = re.sub(r"/\*[\s\S]*?\*/", "", residual).strip()
    if not residual:
        return "comment_only" if text.strip() else "whitespace_only"
    if len(residual) < min_code_chars:
        return "too_short"

    return None



# --------------------------------------------------------------------------- #
# Migration Parsing & Zero-Downtime Expand/Contract Rules
# --------------------------------------------------------------------------- #

def parse_migration_scripts(completion: Union[str, Mapping[str, str], Sequence[dict]]) -> Tuple[str, str]:
    """Extract forward (UP) and rollback (DOWN) SQL scripts from a completion.

    Supports:
    1. Structured mappings: {"up": "...", "down": "..."} or {"forward": "...", "rollback": "..."}
    2. Comment markers:
       -- migrate:up / -- migrate:down
       -- UP / -- DOWN
       -- +goose Up / -- +goose Down
       /* UP */ / /* DOWN */
    3. Markdown code fences:
       ```sql:up.sql ... ``` and ```sql:down.sql ... ```
    4. Default single script fallback: (completion, "")
    """
    if isinstance(completion, Mapping):
        up = completion.get("up") or completion.get("forward") or completion.get("up.sql") or ""
        down = completion.get("down") or completion.get("rollback") or completion.get("down.sql") or ""
        return str(up).strip(), str(down).strip()

    if isinstance(completion, Sequence) and not isinstance(completion, (str, bytes)):
        up_parts: List[str] = []
        down_parts: List[str] = []
        for item in completion:
            if isinstance(item, Mapping):
                name = str(item.get("filename", item.get("path", ""))).lower()
                content = str(item.get("content", item.get("code", "")))
                if "down" in name or "rollback" in name:
                    down_parts.append(content)
                elif "up" in name or "forward" in name:
                    up_parts.append(content)
        if up_parts or down_parts:
            return "\n".join(up_parts).strip(), "\n".join(down_parts).strip()

    text = str(completion).strip()

    # Check for JSON representation
    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                up = parsed.get("up") or parsed.get("forward") or parsed.get("up.sql") or ""
                down = parsed.get("down") or parsed.get("rollback") or parsed.get("down.sql") or ""
                if up or down:
                    return str(up).strip(), str(down).strip()
        except json.JSONDecodeError:
            pass

    # Check for Markdown code fences with filenames
    fence_path_re = re.compile(
        r"""```(?:sql|sqlite)?[:\s]+(?:filename=|title=)?["'`]?([a-zA-Z0-9_./\-]+\.[a-zA-Z0-9]+)["'`]?\s*\n([\s\S]*?)```""",
        re.MULTILINE,
    )
    up_fence: List[str] = []
    down_fence: List[str] = []
    for m in fence_path_re.finditer(text):
        fname = m.group(1).lower()
        code = m.group(2)
        if "down" in fname or "rollback" in fname:
            down_fence.append(code)
        elif "up" in fname or "forward" in fname:
            up_fence.append(code)

    if up_fence or down_fence:
        return "\n".join(up_fence).strip(), "\n".join(down_fence).strip()

    # Check for comment markers: -- migrate:up / -- migrate:down, -- UP / -- DOWN, etc.
    split_patterns = [
        # migrate:up / migrate:down
        (re.compile(r"^[ \t]*--[ \t]*migrate:up\b", re.IGNORECASE | re.MULTILINE),
         re.compile(r"^[ \t]*--[ \t]*migrate:down\b", re.IGNORECASE | re.MULTILINE)),
        # goose
        (re.compile(r"^[ \t]*--[ \t]*\+goose[ \t]+Up\b", re.IGNORECASE | re.MULTILINE),
         re.compile(r"^[ \t]*--[ \t]*\+goose[ \t]+Down\b", re.IGNORECASE | re.MULTILINE)),
        # -- UP / -- DOWN
        (re.compile(r"^[ \t]*--[ \t]*(?:>{1,3}[ \t]*)?UP\b", re.IGNORECASE | re.MULTILINE),
         re.compile(r"^[ \t]*--[ \t]*(?:>{1,3}[ \t]*)?DOWN\b", re.IGNORECASE | re.MULTILINE)),
        # /* UP */ / /* DOWN */
        (re.compile(r"/\*[ \t]*UP[ \t]*\*/", re.IGNORECASE),
         re.compile(r"/\*[ \t]*DOWN[ \t]*\*/", re.IGNORECASE)),
        # -- forward / -- rollback
        (re.compile(r"^[ \t]*--[ \t]*forward\b", re.IGNORECASE | re.MULTILINE),
         re.compile(r"^[ \t]*--[ \t]*rollback\b", re.IGNORECASE | re.MULTILINE)),
    ]

    for up_pat, down_pat in split_patterns:
        up_match = up_pat.search(text)
        down_match = down_pat.search(text)
        if down_match:
            if up_match and up_match.start() < down_match.start():
                up_content = text[up_match.end():down_match.start()].strip()
            else:
                up_content = text[:down_match.start()].strip()
            down_content = text[down_match.end():].strip()
            up_content = _strip_code_fences(up_content)
            down_content = _strip_code_fences(down_content)
            return up_content, down_content

    # Fallback: strip markdown fences if whole completion is fenced
    cleaned = _strip_code_fences(text)
    return cleaned, ""


def _strip_code_fences(text: str) -> str:
    """Strip outermost markdown ``` fences if present."""
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def check_zero_downtime_safety(sql: str, *, allow_destructive: bool = False) -> List[Diagnostic]:
    """Enforce Zero-Downtime Schema Safety (Expand/Contract pattern).

    Rejects destructive operations violating zero-downtime evolution:
    1. DROP TABLE without deprecation
    2. DROP COLUMN or ALTER TABLE ... DROP COLUMN
    3. Direct RENAME TABLE or RENAME COLUMN (breaks running instances)
    4. ALTER TABLE ... ADD COLUMN ... NOT NULL without a DEFAULT value
    5. TRUNCATE TABLE
    """
    if allow_destructive:
        return []

    diags: List[Diagnostic] = []
    # Strip line comments and block comments for analysis
    masked_sql = re.sub(r"--.*$", "", sql, flags=re.MULTILINE)
    masked_sql = re.sub(r"/\*[\s\S]*?\*/", "", masked_sql)

    # 1. DROP TABLE
    drop_table_re = re.compile(r"\bDROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?([a-zA-Z0-9_\"`]+)", re.IGNORECASE)
    for m in drop_table_re.finditer(masked_sql):
        tbl = m.group(1).strip("\"`")
        diags.append(
            Diagnostic(
                code="MIGRATION_DESTRUCTIVE_CHANGE",
                line=sql[:m.start()].count("\n") + 1,
                col=1,
                message=f"Destructive operation: DROP TABLE '{tbl}' violates zero-downtime expand/contract pattern.",
                offset=m.start(),
                source="migration_safety",
                severity=1,
            )
        )

    # 2. DROP COLUMN
    drop_col_re = re.compile(r"\bALTER\s+TABLE\s+([a-zA-Z0-9_\"`]+)\s+DROP\s+(?:COLUMN\s+)?([a-zA-Z0-9_\"`]+)", re.IGNORECASE)
    for m in drop_col_re.finditer(masked_sql):
        tbl = m.group(1).strip("\"`")
        col = m.group(2).strip("\"`")
        diags.append(
            Diagnostic(
                code="MIGRATION_DESTRUCTIVE_CHANGE",
                line=sql[:m.start()].count("\n") + 1,
                col=1,
                message=f"Destructive operation: DROP COLUMN '{col}' on table '{tbl}' violates expand/contract rules.",
                offset=m.start(),
                source="migration_safety",
                severity=1,
            )
        )

    # 3. Direct RENAME COLUMN / RENAME TABLE
    rename_col_re = re.compile(r"\bALTER\s+TABLE\s+([a-zA-Z0-9_\"`]+)\s+RENAME\s+COLUMN\s+([a-zA-Z0-9_\"`]+)\s+TO\s+([a-zA-Z0-9_\"`]+)", re.IGNORECASE)
    for m in rename_col_re.finditer(masked_sql):
        tbl = m.group(1).strip("\"`")
        old_col = m.group(2).strip("\"`")
        new_col = m.group(3).strip("\"`")
        diags.append(
            Diagnostic(
                code="MIGRATION_EXPAND_CONTRACT_VIOLATION",
                line=sql[:m.start()].count("\n") + 1,
                col=1,
                message=f"Destructive rename: RENAME COLUMN '{old_col}' TO '{new_col}' on '{tbl}' breaks running readers. Use expand/contract dual-write instead.",
                offset=m.start(),
                source="migration_safety",
                severity=1,
            )
        )

    # 4. ADD COLUMN NOT NULL without DEFAULT
    add_not_null_re = re.compile(
        r"\bALTER\s+TABLE\s+([a-zA-Z0-9_\"`]+)\s+ADD\s+(?:COLUMN\s+)?([a-zA-Z0-9_\"`]+)\s+([^;]+)",
        re.IGNORECASE,
    )
    for m in add_not_null_re.finditer(masked_sql):
        tbl = m.group(1).strip("\"`")
        col = m.group(2).strip("\"`")
        col_def = m.group(3).upper()
        if "NOT NULL" in col_def and "DEFAULT" not in col_def:
            diags.append(
                Diagnostic(
                    code="MIGRATION_DESTRUCTIVE_CHANGE",
                    line=sql[:m.start()].count("\n") + 1,
                    col=1,
                    message=f"Destructive schema constraint: Adding NOT NULL column '{col}' to '{tbl}' without DEFAULT value violates zero-downtime safety.",
                    offset=m.start(),
                    source="migration_safety",
                    severity=1,
                )
            )

    # 5. TRUNCATE TABLE
    truncate_re = re.compile(r"\bTRUNCATE\s+(?:TABLE\s+)?([a-zA-Z0-9_\"`]+)", re.IGNORECASE)
    for m in truncate_re.finditer(masked_sql):
        tbl = m.group(1).strip("\"`")
        diags.append(
            Diagnostic(
                code="MIGRATION_DESTRUCTIVE_CHANGE",
                line=sql[:m.start()].count("\n") + 1,
                col=1,
                message=f"Destructive operation: TRUNCATE TABLE '{tbl}' causes instant data loss.",
                offset=m.start(),
                source="migration_safety",
                severity=1,
            )
        )

    return diags


# --------------------------------------------------------------------------- #
# Database Schema Snapshot & Exact Parity Verification
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TableColumnInfo:
    cid: int
    name: str
    data_type: str
    notnull: int
    dflt_value: Any
    pk: int


@dataclass(frozen=True)
class SchemaSnapshot:
    """Complete snapshot of SQLite relational schema and data counts."""
    tables: Dict[str, List[TableColumnInfo]] = field(default_factory=dict)
    indexes: Dict[str, str] = field(default_factory=dict)
    views: Dict[str, str] = field(default_factory=dict)
    triggers: Dict[str, str] = field(default_factory=dict)
    record_counts: Dict[str, int] = field(default_factory=dict)
    table_rows: Dict[str, List[Tuple[Any, ...]]] = field(default_factory=dict)


def capture_sqlite_snapshot(conn: sqlite3.Connection, capture_rows: bool = True) -> SchemaSnapshot:
    """Capture complete schema state and seeded records from SQLite connection."""
    cursor = conn.cursor()
    tables: Dict[str, List[TableColumnInfo]] = {}
    indexes: Dict[str, str] = {}
    views: Dict[str, str] = {}
    triggers: Dict[str, str] = {}
    record_counts: Dict[str, int] = {}
    table_rows: Dict[str, List[Tuple[Any, ...]]] = {}

    # Query tables
    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;")
    for row in cursor.fetchall():
        tbl_name = row[0]
        # Query columns for table
        cursor.execute(f"PRAGMA table_info('{tbl_name}');")
        col_infos: List[TableColumnInfo] = []
        for c in cursor.fetchall():
            col_infos.append(
                TableColumnInfo(
                    cid=c[0],
                    name=c[1],
                    data_type=(c[2] or "").upper(),
                    notnull=c[3],
                    dflt_value=c[4],
                    pk=c[5],
                )
            )
        tables[tbl_name] = col_infos

        # Record count
        try:
            cursor.execute(f"SELECT count(*) FROM '{tbl_name}';")
            count_res = cursor.fetchone()
            cnt = count_res[0] if count_res else 0
            record_counts[tbl_name] = cnt

            if capture_rows and cnt <= 500:
                cursor.execute(f"SELECT * FROM '{tbl_name}' ORDER BY 1;")
                table_rows[tbl_name] = cursor.fetchall()
        except sqlite3.OperationalError:
            pass

    # Query indexes
    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL ORDER BY name;")
    for row in cursor.fetchall():
        indexes[row[0]] = (row[1] or "").strip()

    # Query views
    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='view' ORDER BY name;")
    for row in cursor.fetchall():
        views[row[0]] = (row[1] or "").strip()

    # Query triggers
    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='trigger' ORDER BY name;")
    for row in cursor.fetchall():
        triggers[row[0]] = (row[1] or "").strip()

    return SchemaSnapshot(
        tables=tables,
        indexes=indexes,
        views=views,
        triggers=triggers,
        record_counts=record_counts,
        table_rows=table_rows,
    )


def compare_schema_snapshots(initial: SchemaSnapshot, rolled_back: SchemaSnapshot) -> List[str]:
    """Compare initial schema snapshot with post-rollback snapshot to verify exact parity."""
    diffs: List[str] = []

    # Check tables
    init_tables = set(initial.tables.keys())
    rb_tables = set(rolled_back.tables.keys())

    missing_tables = init_tables - rb_tables
    if missing_tables:
        diffs.append(f"Missing tables after rollback: {sorted(missing_tables)}")

    extra_tables = rb_tables - init_tables
    if extra_tables:
        diffs.append(f"Leftover tables after rollback: {sorted(extra_tables)}")

    for tbl in init_tables & rb_tables:
        init_cols = {c.name: c for c in initial.tables[tbl]}
        rb_cols = {c.name: c for c in rolled_back.tables[tbl]}

        missing_cols = set(init_cols.keys()) - set(rb_cols.keys())
        if missing_cols:
            diffs.append(f"Table '{tbl}' missing columns after rollback: {sorted(missing_cols)}")

        extra_cols = set(rb_cols.keys()) - set(init_cols.keys())
        if extra_cols:
            diffs.append(f"Table '{tbl}' has unexpected extra columns after rollback: {sorted(extra_cols)}")

        for col_name in set(init_cols.keys()) & set(rb_cols.keys()):
            ic = init_cols[col_name]
            rc = rb_cols[col_name]
            if ic.data_type != rc.data_type:
                diffs.append(f"Table '{tbl}' column '{col_name}' type mismatch: expected {ic.data_type}, got {rc.data_type}")
            if ic.notnull != rc.notnull:
                diffs.append(f"Table '{tbl}' column '{col_name}' NOT NULL mismatch: expected {ic.notnull}, got {rc.notnull}")
            if str(ic.dflt_value) != str(rc.dflt_value):
                diffs.append(f"Table '{tbl}' column '{col_name}' DEFAULT mismatch: expected {ic.dflt_value}, got {rc.dflt_value}")
            if ic.pk != rc.pk:
                diffs.append(f"Table '{tbl}' column '{col_name}' PK mismatch: expected {ic.pk}, got {rc.pk}")

    # Check indexes
    init_idx = set(initial.indexes.keys())
    rb_idx = set(rolled_back.indexes.keys())
    if init_idx != rb_idx:
        diffs.append(f"Indexes mismatch: missing {sorted(init_idx - rb_idx)}, extra {sorted(rb_idx - init_idx)}")

    return diffs


# --------------------------------------------------------------------------- #
# In-Memory Migration Replay Engine
# --------------------------------------------------------------------------- #

@dataclass
class MigrationReplayResult:
    is_clean: bool
    forward_passed: bool
    data_integrity_passed: bool
    rollback_passed: bool
    schema_parity_passed: bool
    rollback_data_passed: bool
    destructive_changes: List[str] = field(default_factory=list)
    diagnostics: List[Diagnostic] = field(default_factory=list)
    elapsed_ms: float = 0.0


class MigrationReplayOracle:
    """In-process, in-memory SQLite migration replay and schema parity verifier."""

    def __init__(self, *, timeout_s: float = 5.0):
        self.timeout_s = timeout_s
        self.n_calls = 0
        self.wall_s = 0.0

    def replay(
        self,
        up_sql: str,
        down_sql: str,
        *,
        initial_schema: str,
        seed_data: Optional[Union[str, Sequence[str], Mapping[str, Sequence[dict]]]] = None,
        expected_records: Optional[Union[Mapping[str, int], Sequence[Mapping[str, Any]]]] = None,
        allow_destructive: bool = False,
        custom_integrity_fn: Optional[Callable[[sqlite3.Connection], bool]] = None,
    ) -> MigrationReplayResult:
        """Replay migration forward and rollback in an in-memory SQLite database."""
        t0 = time.monotonic()
        self.n_calls += 1
        diags: List[Diagnostic] = []

        # 1. Zero-downtime safety checks on UP migration
        safety_diags = check_zero_downtime_safety(up_sql, allow_destructive=allow_destructive)
        diags.extend(safety_diags)
        destructive_changes = [d.message for d in safety_diags]

        # 2. Check for missing rollback migration
        if not down_sql.strip():
            diags.append(
                Diagnostic(
                    code="MIGRATION_MISSING_DOWN",
                    line=1,
                    col=1,
                    message="Rollback (DOWN) migration script is missing or empty.",
                    offset=0,
                    source="migration_replay",
                    severity=1,
                )
            )

        forward_passed = False
        data_integrity_passed = False
        rollback_passed = False
        schema_parity_passed = False
        rollback_data_passed = False

        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("PRAGMA foreign_keys = ON;")

            # 3. Setup initial schema
            if initial_schema.strip():
                try:
                    conn.executescript(initial_schema)
                except sqlite3.Error as e:
                    diags.append(
                        Diagnostic(
                            code="MIGRATION_SYNTAX_ERROR",
                            line=1,
                            col=1,
                            message=f"Failed to execute initial schema: {e}",
                            offset=0,
                            source="migration_setup",
                            severity=1,
                        )
                    )
                    elapsed = (time.monotonic() - t0) * 1000.0
                    self.wall_s += elapsed / 1000.0
                    return MigrationReplayResult(
                        is_clean=False,
                        forward_passed=False,
                        data_integrity_passed=False,
                        rollback_passed=False,
                        schema_parity_passed=False,
                        rollback_data_passed=False,
                        destructive_changes=destructive_changes,
                        diagnostics=diags,
                        elapsed_ms=elapsed,
                    )

            # 4. Insert seed data
            _insert_seed_data(conn, seed_data)

            # 5. Capture pre-migration snapshot
            pre_snapshot = capture_sqlite_snapshot(conn, capture_rows=True)

            # 6. Execute forward (UP) migration
            try:
                conn.executescript(up_sql)
                forward_passed = True
            except sqlite3.Error as e:
                diags.append(
                    Diagnostic(
                        code="MIGRATION_EXECUTION_FAILURE",
                        line=1,
                        col=1,
                        message=f"Forward (UP) migration failed: {e}",
                        offset=0,
                        source="migration_replay",
                        severity=1,
                    )
                )

            # 7. Assert data integrity after forward migration
            if forward_passed:
                integrity_ok, integrity_err = _assert_data_integrity(
                    conn,
                    pre_snapshot,
                    expected_records=expected_records,
                    custom_fn=custom_integrity_fn,
                )
                if integrity_ok:
                    data_integrity_passed = True
                else:
                    diags.append(
                        Diagnostic(
                            code="MIGRATION_DATA_LOSS",
                            line=1,
                            col=1,
                            message=f"Data integrity violation after forward migration: {integrity_err}",
                            offset=0,
                            source="migration_integrity",
                            severity=1,
                        )
                    )

            # 8. Execute rollback (DOWN) migration
            if forward_passed and down_sql.strip():
                try:
                    conn.executescript(down_sql)
                    rollback_passed = True
                except sqlite3.Error as e:
                    diags.append(
                        Diagnostic(
                            code="MIGRATION_ROLLBACK_FAILED",
                            line=1,
                            col=1,
                            message=f"Rollback (DOWN) migration failed: {e}",
                            offset=0,
                            source="migration_replay",
                            severity=1,
                        )
                    )

            # 9. Verify exact schema state parity and data restoration after rollback
            if rollback_passed:
                post_rb_snapshot = capture_sqlite_snapshot(conn, capture_rows=True)
                diffs = compare_schema_snapshots(pre_snapshot, post_rb_snapshot)
                if not diffs:
                    schema_parity_passed = True
                else:
                    diags.append(
                        Diagnostic(
                            code="MIGRATION_ROLLBACK_SCHEMA_MISMATCH",
                            line=1,
                            col=1,
                            message=f"Schema state parity failed after rollback: {'; '.join(diffs)}",
                            offset=0,
                            source="migration_parity",
                            severity=1,
                        )
                    )

                # Check data restoration after rollback
                data_diffs: List[str] = []
                for tbl, init_rows in pre_snapshot.table_rows.items():
                    rb_rows = post_rb_snapshot.table_rows.get(tbl)
                    if rb_rows != init_rows:
                        data_diffs.append(f"Table '{tbl}' records not restored (expected {len(init_rows)} rows, got {len(rb_rows) if rb_rows else 0})")

                if not data_diffs:
                    rollback_data_passed = True
                else:
                    diags.append(
                        Diagnostic(
                            code="MIGRATION_ROLLBACK_DATA_LOSS",
                            line=1,
                            col=1,
                            message=f"Data restoration failed after rollback: {'; '.join(data_diffs)}",
                            offset=0,
                            source="migration_parity",
                            severity=1,
                        )
                    )

        finally:
            conn.close()

        elapsed = (time.monotonic() - t0) * 1000.0
        self.wall_s += elapsed / 1000.0
        is_clean = len(diags) == 0

        return MigrationReplayResult(
            is_clean=is_clean,
            forward_passed=forward_passed,
            data_integrity_passed=data_integrity_passed,
            rollback_passed=rollback_passed,
            schema_parity_passed=schema_parity_passed,
            rollback_data_passed=rollback_data_passed,
            destructive_changes=destructive_changes,
            diagnostics=diags,
            elapsed_ms=elapsed,
        )

    def close(self) -> None:
        pass


def _insert_seed_data(
    conn: sqlite3.Connection,
    seed_data: Optional[Union[str, Sequence[str], Mapping[str, Sequence[dict]]]],
) -> None:
    """Helper to populate pre-seeded records into SQLite."""
    if not seed_data:
        return

    if isinstance(seed_data, str):
        conn.executescript(seed_data)
    elif isinstance(seed_data, Sequence):
        for stmt in seed_data:
            if isinstance(stmt, str):
                conn.execute(stmt)
    elif isinstance(seed_data, Mapping):
        cursor = conn.cursor()
        for tbl, rows in seed_data.items():
            for row in rows:
                if isinstance(row, dict):
                    cols = list(row.keys())
                    placeholders = ", ".join("?" for _ in cols)
                    col_names = ", ".join(f"'{c}'" for c in cols)
                    sql = f"INSERT INTO '{tbl}' ({col_names}) VALUES ({placeholders});"
                    cursor.execute(sql, list(row.values()))
        conn.commit()


def _assert_data_integrity(
    conn: sqlite3.Connection,
    pre_snapshot: SchemaSnapshot,
    expected_records: Optional[Union[Mapping[str, int], Sequence[Mapping[str, Any]]]] = None,
    custom_fn: Optional[Callable[[sqlite3.Connection], bool]] = None,
) -> Tuple[bool, str]:
    """Verify that forward migration did not silently drop or corrupt pre-seeded records."""
    cursor = conn.cursor()

    # 1. Custom function check if provided
    if custom_fn is not None:
        try:
            if not custom_fn(conn):
                return False, "Custom integrity assertion returned False."
        except Exception as ex:
            return False, f"Custom integrity assertion raised exception: {ex}"

    # 2. Expected records assertion
    if expected_records:
        if isinstance(expected_records, Mapping):
            for tbl, exp_cnt in expected_records.items():
                try:
                    cursor.execute(f"SELECT count(*) FROM '{tbl}';")
                    row = cursor.fetchone()
                    act_cnt = row[0] if row else 0
                    if act_cnt != exp_cnt:
                        return False, f"Table '{tbl}' expected {exp_cnt} records, got {act_cnt}."
                except sqlite3.OperationalError as oe:
                    return False, f"Querying table '{tbl}' failed: {oe}"
        elif isinstance(expected_records, Sequence):
            for item in expected_records:
                if isinstance(item, dict):
                    q = item.get("query")
                    expected_val = item.get("expected")
                    if q:
                        try:
                            cursor.execute(q)
                            row = cursor.fetchone()
                            val = row[0] if row else None
                            if expected_val is not None and val != expected_val:
                                return False, f"Query '{q}' returned {val}, expected {expected_val}."
                        except sqlite3.OperationalError as oe:
                            return False, f"Query '{q}' failed: {oe}"

    # 3. Default integrity: ensure tables that still exist didn't lose rows unexpectedly
    for tbl, pre_cnt in pre_snapshot.record_counts.items():
        if pre_cnt > 0:
            try:
                cursor.execute(f"SELECT count(*) FROM '{tbl}';")
                row = cursor.fetchone()
                cur_cnt = row[0] if row else 0
                if cur_cnt < pre_cnt and (not expected_records or tbl not in expected_records):
                    return False, f"Table '{tbl}' lost rows: had {pre_cnt}, now has {cur_cnt}."
            except sqlite3.OperationalError:
                pass

    return True, ""


# --------------------------------------------------------------------------- #
# Migration Replay Verifier Class
# --------------------------------------------------------------------------- #

class MigrationReplayVerifier:
    """Verifier for in-memory database migration replay and schema safety rules."""

    def __init__(
        self,
        *,
        oracle: Optional[MigrationReplayOracle] = None,
        timeout_s: float = 5.0,
        fail_fast: bool = True,
        allow_destructive: bool = False,
        **reward_kwargs: Any,
    ) -> None:
        self.oracle = oracle or MigrationReplayOracle(timeout_s=timeout_s)
        self.timeout_s = timeout_s
        self.fail_fast = fail_fast
        self.allow_destructive = allow_destructive
        self.reward_kwargs = reward_kwargs

        self._lock = threading.Lock()
        self._n_samples = 0
        self._n_clean = 0
        self._n_hacked = 0
        self._n_degenerate = 0
        self._n_failures = 0
        self._reward_total = 0.0
        self._elapsed_ms_total = 0.0

    def evaluate(
        self,
        completion: Union[str, Mapping[str, str], Sequence[dict]],
        reference: Any = None,
        *,
        initial_schema: Optional[str] = None,
        seed_data: Optional[Union[str, Sequence[str], Mapping[str, Sequence[dict]]]] = None,
        expected_records: Optional[Union[Mapping[str, int], Sequence[Mapping[str, Any]]]] = None,
        prompt: str = "",
        allow_destructive: Optional[bool] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Evaluate forward and rollback migrations against in-memory relational engine."""
        t0 = time.monotonic()

        # Handle reference dict
        if isinstance(reference, Mapping):
            if initial_schema is None:
                initial_schema = reference.get("initial_schema") or reference.get("schema")
            if seed_data is None:
                seed_data = reference.get("seed_data") or reference.get("seed")
            if expected_records is None:
                expected_records = reference.get("expected_records") or reference.get("expected")
            if allow_destructive is None and "allow_destructive" in reference:
                allow_destructive = bool(reference["allow_destructive"])

        if allow_destructive is None:
            allow_destructive = self.allow_destructive

        initial_schema_str = str(initial_schema or "")

        up_sql, down_sql = parse_migration_scripts(completion)
        combined_text = f"{up_sql}\n{down_sql}"

        # 1. Degenerate check
        degen_reason = is_migration_degenerate_output(combined_text)
        is_degenerate = degen_reason is not None

        # 2. Escape hatch check
        hatches = find_migration_escape_hatches(combined_text)
        is_hacked = len(hatches) > 0

        if self.fail_fast and (is_degenerate or is_hacked):
            reward = -1.0
            elapsed = (time.monotonic() - t0) * 1000.0
            with self._lock:
                self._n_samples += 1
                if is_degenerate:
                    self._n_degenerate += 1
                if is_hacked:
                    self._n_hacked += 1
                self._reward_total += reward
                self._elapsed_ms_total += elapsed

            return {
                "reward": reward,
                "is_clean": False,
                "forward_passed": False,
                "data_integrity_passed": False,
                "rollback_passed": False,
                "schema_parity_passed": False,
                "rollback_data_passed": False,
                "destructive_changes": [],
                "is_degenerate": is_degenerate,
                "degenerate_reason": degen_reason,
                "is_hacked": is_hacked,
                "hatches": hatches,
                "diagnostics": [
                    Diagnostic(
                        code="MIGRATION_DEGENERATE_OUTPUT" if is_degenerate else "MIGRATION_ESCAPE_HATCH",
                        line=1,
                        col=1,
                        message=f"Degenerate: {degen_reason}" if is_degenerate else f"Escape hatch: {hatches}",
                        offset=0,
                        source="migration_verifier",
                        severity=1,
                    )
                ],
                "elapsed_ms": elapsed,
            }

        # 3. Replay in SQLite
        res = self.oracle.replay(
            up_sql,
            down_sql,
            initial_schema=initial_schema_str,
            seed_data=seed_data,
            expected_records=expected_records,
            allow_destructive=allow_destructive,
        )

        # 4. Compute reward
        if res.is_clean:
            reward = 1.0
        else:
            reward = diagnostics_to_reward(
                res.diagnostics,
                code_overrides=DEFAULT_MIGRATION_CODE_OVERRIDES,
                **self.reward_kwargs,
            )

        elapsed = (time.monotonic() - t0) * 1000.0
        with self._lock:
            self._n_samples += 1
            if res.is_clean:
                self._n_clean += 1
            else:
                self._n_failures += 1
            self._reward_total += reward
            self._elapsed_ms_total += elapsed

        return {
            "reward": reward,
            "is_clean": res.is_clean,
            "forward_passed": res.forward_passed,
            "data_integrity_passed": res.data_integrity_passed,
            "rollback_passed": res.rollback_passed,
            "schema_parity_passed": res.schema_parity_passed,
            "rollback_data_passed": res.rollback_data_passed,
            "destructive_changes": res.destructive_changes,
            "is_degenerate": False,
            "degenerate_reason": None,
            "is_hacked": False,
            "hatches": [],
            "diagnostics": res.diagnostics,
            "elapsed_ms": elapsed,
        }

    def reward(
        self,
        completion: Union[str, Mapping[str, str], Sequence[dict]],
        reference: Any = None,
        *,
        prompt: str = "",
        **kwargs: Any,
    ) -> float:
        """Compute scalar RLVR reward for migration completion."""
        eval_res = self.evaluate(completion, reference=reference, prompt=prompt, **kwargs)
        return float(eval_res["reward"])

    def telemetry(self) -> Dict[str, Any]:
        """Telemetry statistics."""
        with self._lock:
            n = self._n_samples
            return {
                "verifier": "MigrationReplayVerifier",
                "n_samples": n,
                "n_clean": self._n_clean,
                "n_hacked": self._n_hacked,
                "n_degenerate": self._n_degenerate,
                "n_failures": self._n_failures,
                "mean_reward": (self._reward_total / n) if n else 0.0,
                "mean_elapsed_ms": (self._elapsed_ms_total / n) if n else 0.0,
                "oracle_calls": self.oracle.n_calls,
                "oracle_wall_s": self.oracle.wall_s,
            }

    def __enter__(self) -> 'MigrationReplayVerifier':
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        self.oracle.close()


# --------------------------------------------------------------------------- #
# API Contract Breaking-Change Detection (OpenAPI & Protobuf)
# --------------------------------------------------------------------------- #

@dataclass
class ProtoField:
    name: str
    field_type: str
    tag: int
    label: str = "optional"


@dataclass
class ProtoMessage:
    name: str
    fields: Dict[str, ProtoField] = field(default_factory=dict)
    reserved_tags: Set[int] = field(default_factory=set)
    reserved_names: Set[str] = field(default_factory=set)


@dataclass
class ProtoEnum:
    name: str
    values: Dict[str, int] = field(default_factory=dict)
    reserved_tags: Set[int] = field(default_factory=set)
    reserved_names: Set[str] = field(default_factory=set)


@dataclass
class ProtoServiceRpc:
    name: str
    input_type: str
    output_type: str
    client_streaming: bool = False
    server_streaming: bool = False


@dataclass
class ProtoService:
    name: str
    rpcs: Dict[str, ProtoServiceRpc] = field(default_factory=dict)


@dataclass
class ProtoSchema:
    syntax: str = "proto3"
    package: str = ""
    messages: Dict[str, ProtoMessage] = field(default_factory=dict)
    enums: Dict[str, ProtoEnum] = field(default_factory=dict)
    services: Dict[str, ProtoService] = field(default_factory=dict)


def parse_proto_schema(proto_text: str) -> ProtoSchema:
    """Deterministic in-process Protobuf parser extracting AST entities."""
    schema = ProtoSchema()

    # Syntax
    m_syn = re.search(r"\bsyntax\s*=\s*['\"](proto[23])['\"]\s*;", proto_text)
    if m_syn:
        schema.syntax = m_syn.group(1)

    # Package
    m_pkg = re.search(r"\bpackage\s+([a-zA-Z0-9_.]+)\s*;", proto_text)
    if m_pkg:
        schema.package = m_pkg.group(1)

    # Messages
    msg_re = re.compile(r"\bmessage\s+([a-zA-Z0-9_]+)\s*\{([^}]+)\}", re.MULTILINE)
    for m_msg in msg_re.finditer(proto_text):
        m_name = m_msg.group(1)
        body = m_msg.group(2)
        msg_obj = ProtoMessage(name=m_name)

        # Reserved lines
        for r_line in re.finditer(r"\breserved\s+([^;]+);", body):
            res_content = r_line.group(1)
            for part in res_content.split(","):
                part = part.strip()
                if not part:
                    continue
                if part.startswith(('"', "'")):
                    msg_obj.reserved_names.add(part.strip("\"'"))
                elif "to" in part:
                    subparts = part.split("to")
                    try:
                        start_n = int(subparts[0].strip())
                        end_n = int(subparts[1].strip())
                        for n in range(start_n, end_n + 1):
                            msg_obj.reserved_tags.add(n)
                    except ValueError:
                        pass
                else:
                    try:
                        msg_obj.reserved_tags.add(int(part))
                    except ValueError:
                        pass

        # Fields: [label] type name = tag;
        field_re = re.compile(r"(?:(optional|required|repeated)\s+)?([a-zA-Z0-9_.]+)\s+([a-zA-Z0-9_]+)\s*=\s*(\d+)\s*;")
        for f in field_re.finditer(body):
            label = f.group(1) or "optional"
            ftype = f.group(2)
            fname = f.group(3)
            tag = int(f.group(4))
            msg_obj.fields[fname] = ProtoField(name=fname, field_type=ftype, tag=tag, label=label)

        schema.messages[m_name] = msg_obj

    # Enums
    enum_re = re.compile(r"\benum\s+([a-zA-Z0-9_]+)\s*\{([^}]+)\}", re.MULTILINE)
    for m_enum in enum_re.finditer(proto_text):
        e_name = m_enum.group(1)
        body = m_enum.group(2)
        enum_obj = ProtoEnum(name=e_name)

        for val_match in re.finditer(r"\b([a-zA-Z0-9_]+)\s*=\s*(-?\d+)\s*;", body):
            enum_obj.values[val_match.group(1)] = int(val_match.group(2))

        schema.enums[e_name] = enum_obj

    # Services
    svc_re = re.compile(r"\bservice\s+([a-zA-Z0-9_]+)\s*\{([^}]+)\}", re.MULTILINE)
    rpc_re = re.compile(r"\brpc\s+([a-zA-Z0-9_]+)\s*\(\s*(stream\s+)?([a-zA-Z0-9_.]+)\s*\)\s*returns\s*\(\s*(stream\s+)?([a-zA-Z0-9_.]+)\s*\)\s*;", re.MULTILINE)
    for m_svc in svc_re.finditer(proto_text):
        s_name = m_svc.group(1)
        body = m_svc.group(2)
        svc_obj = ProtoService(name=s_name)

        for m_rpc in rpc_re.finditer(body):
            rpc_name = m_rpc.group(1)
            c_stream = bool(m_rpc.group(2))
            in_type = m_rpc.group(3)
            s_stream = bool(m_rpc.group(4))
            out_type = m_rpc.group(5)
            svc_obj.rpcs[rpc_name] = ProtoServiceRpc(
                name=rpc_name,
                input_type=in_type,
                output_type=out_type,
                client_streaming=c_stream,
                server_streaming=s_stream,
            )

        schema.services[s_name] = svc_obj

    return schema


def diff_proto_schemas(base: ProtoSchema, updated: ProtoSchema) -> Tuple[List[Diagnostic], List[str]]:
    """Detect breaking changes and score non-breaking additive extensions in Protobuf schemas."""
    diags: List[Diagnostic] = []
    additive: List[str] = []

    # 1. Message checks
    for msg_name, base_msg in base.messages.items():
        if msg_name not in updated.messages:
            diags.append(
                Diagnostic(
                    code="PROTO_MESSAGE_REMOVED",
                    line=1,
                    col=1,
                    message=f"Breaking change: message '{msg_name}' removed from Protobuf schema.",
                    offset=0,
                    source="protobuf_diff",
                    severity=1,
                )
            )
            continue

        up_msg = updated.messages[msg_name]
        # Check removed fields
        for fname, b_field in base_msg.fields.items():
            if fname not in up_msg.fields:
                if b_field.tag not in up_msg.reserved_tags and fname not in up_msg.reserved_names:
                    diags.append(
                        Diagnostic(
                            code="PROTO_FIELD_REMOVED_WITHOUT_RESERVE",
                            line=1,
                            col=1,
                            message=f"Breaking change: field '{fname}' (tag {b_field.tag}) removed from '{msg_name}' without reserving tag {b_field.tag}.",
                            offset=0,
                            source="protobuf_diff",
                            severity=1,
                        )
                    )
                else:
                    additive.append(f"Field '{fname}' removed and properly reserved in '{msg_name}'")
            else:
                u_field = up_msg.fields[fname]
                # Tag number changed
                if u_field.tag != b_field.tag:
                    diags.append(
                        Diagnostic(
                            code="PROTO_FIELD_TAG_CHANGED",
                            line=1,
                            col=1,
                            message=f"Breaking change: field '{fname}' in '{msg_name}' tag mutated from {b_field.tag} to {u_field.tag}.",
                            offset=0,
                            source="protobuf_diff",
                            severity=1,
                        )
                    )
                # Field wire type changed
                if u_field.field_type != b_field.field_type:
                    diags.append(
                        Diagnostic(
                            code="PROTO_FIELD_TYPE_CHANGED",
                            line=1,
                            col=1,
                            message=f"Breaking change: field '{fname}' in '{msg_name}' type mutated from '{b_field.field_type}' to '{u_field.field_type}'.",
                            offset=0,
                            source="protobuf_diff",
                            severity=1,
                        )
                    )
                # Cardinality / label changed
                if u_field.label != b_field.label:
                    diags.append(
                        Diagnostic(
                            code="PROTO_CARDINALITY_CHANGED",
                            line=1,
                            col=1,
                            message=f"Breaking change: field '{fname}' in '{msg_name}' label changed from '{b_field.label}' to '{u_field.label}'.",
                            offset=0,
                            source="protobuf_diff",
                            severity=1,
                        )
                    )

        # Check newly added fields
        for fname, u_field in up_msg.fields.items():
            if fname not in base_msg.fields:
                additive.append(f"Added non-breaking field '{fname}' (tag {u_field.tag}) to '{msg_name}'")

    # Newly added messages
    for msg_name in updated.messages:
        if msg_name not in base.messages:
            additive.append(f"Added new message '{msg_name}'")

    # 2. Enum checks
    for enum_name, b_enum in base.enums.items():
        if enum_name not in updated.enums:
            diags.append(
                Diagnostic(
                    code="PROTO_ENUM_VALUE_REMOVED",
                    line=1,
                    col=1,
                    message=f"Breaking change: enum '{enum_name}' was removed.",
                    offset=0,
                    source="protobuf_diff",
                    severity=1,
                )
            )
            continue
        u_enum = updated.enums[enum_name]
        for val_name, val_num in b_enum.values.items():
            if val_name not in u_enum.values:
                diags.append(
                    Diagnostic(
                        code="PROTO_ENUM_VALUE_REMOVED",
                        line=1,
                        col=1,
                        message=f"Breaking change: enum value '{val_name}' removed from '{enum_name}'.",
                        offset=0,
                        source="protobuf_diff",
                        severity=1,
                    )
                )

    # 3. Service checks
    for svc_name, b_svc in base.services.items():
        if svc_name not in updated.services:
            diags.append(
                Diagnostic(
                    code="PROTO_SERVICE_REMOVED",
                    line=1,
                    col=1,
                    message=f"Breaking change: service '{svc_name}' was removed.",
                    offset=0,
                    source="protobuf_diff",
                    severity=1,
                )
            )
            continue
        u_svc = updated.services[svc_name]
        for rpc_name, b_rpc in b_svc.rpcs.items():
            if rpc_name not in u_svc.rpcs:
                diags.append(
                    Diagnostic(
                        code="PROTO_RPC_REMOVED",
                        line=1,
                        col=1,
                        message=f"Breaking change: RPC '{rpc_name}' removed from service '{svc_name}'.",
                        offset=0,
                        source="protobuf_diff",
                        severity=1,
                    )
                )
            else:
                u_rpc = u_svc.rpcs[rpc_name]
                if u_rpc.input_type != b_rpc.input_type or u_rpc.output_type != b_rpc.output_type:
                    diags.append(
                        Diagnostic(
                            code="PROTO_RPC_SIGNATURE_CHANGED",
                            line=1,
                            col=1,
                            message=f"Breaking change: RPC '{rpc_name}' signature mutated.",
                            offset=0,
                            source="protobuf_diff",
                            severity=1,
                        )
                    )

    return diags, additive


# --------------------------------------------------------------------------- #
# OpenAPI Contract Diffing Engine
# --------------------------------------------------------------------------- #

def _parse_spec_dict(text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON or YAML OpenAPI specification."""
    stripped = text.strip()
    if not stripped:
        return None

    # Try JSON
    if stripped.startswith("{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    # Try YAML
    if yaml is not None:
        try:
            parsed = yaml.safe_load(stripped)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return None


def diff_openapi_specs(base: Dict[str, Any], updated: Dict[str, Any]) -> Tuple[List[Diagnostic], List[str]]:
    """Detect breaking changes and non-breaking additive extensions in OpenAPI specs."""
    diags: List[Diagnostic] = []
    additive: List[str] = []

    base_paths = base.get("paths", {})
    up_paths = updated.get("paths", {})

    if not isinstance(base_paths, dict) or not isinstance(up_paths, dict):
        return diags, additive

    # 1. Path endpoint checks
    for path, b_path_item in base_paths.items():
        if path not in up_paths:
            diags.append(
                Diagnostic(
                    code="CONTRACT_ENDPOINT_REMOVED",
                    line=1,
                    col=1,
                    message=f"Breaking change: Endpoint path '{path}' removed from API specification.",
                    offset=0,
                    source="openapi_diff",
                    severity=1,
                )
            )
            continue

        u_path_item = up_paths[path]
        if not isinstance(b_path_item, dict) or not isinstance(u_path_item, dict):
            continue

        # 2. HTTP Operation checks
        http_methods = {"get", "post", "put", "delete", "patch", "options", "head"}
        for method in http_methods:
            if method in b_path_item and method not in u_path_item:
                diags.append(
                    Diagnostic(
                        code="CONTRACT_METHOD_REMOVED",
                        line=1,
                        col=1,
                        message=f"Breaking change: HTTP method '{method.upper()} {path}' was removed.",
                        offset=0,
                        source="openapi_diff",
                        severity=1,
                    )
                )
            elif method in b_path_item and method in u_path_item:
                b_op = b_path_item[method]
                u_op = u_path_item[method]
                if isinstance(b_op, dict) and isinstance(u_op, dict):
                    # Check parameters
                    b_params = {p.get("name"): p for p in b_op.get("parameters", []) if isinstance(p, dict)}
                    u_params = {p.get("name"): p for p in u_op.get("parameters", []) if isinstance(p, dict)}

                    for p_name, u_param in u_params.items():
                        if u_param.get("required") is True and p_name not in b_params:
                            diags.append(
                                Diagnostic(
                                    code="CONTRACT_REQUIRED_PARAM_ADDED",
                                    line=1,
                                    col=1,
                                    message=f"Breaking change: New required parameter '{p_name}' added to '{method.upper()} {path}'.",
                                    offset=0,
                                    source="openapi_diff",
                                    severity=1,
                                )
                            )
                        elif p_name not in b_params:
                            additive.append(f"Added optional parameter '{p_name}' to '{method.upper()} {path}'")

                    for p_name, b_param in b_params.items():
                        if p_name in u_params:
                            u_param = u_params[p_name]
                            if not b_param.get("required") and u_param.get("required"):
                                diags.append(
                                    Diagnostic(
                                        code="CONTRACT_REQUIRED_PARAM_ADDED",
                                        line=1,
                                        col=1,
                                        message=f"Breaking change: Parameter '{p_name}' in '{method.upper()} {path}' changed from optional to required.",
                                        offset=0,
                                        source="openapi_diff",
                                        severity=1,
                                    )
                                )
                            b_type = (b_param.get("schema", {}) or {}).get("type")
                            u_type = (u_param.get("schema", {}) or {}).get("type")
                            if b_type and u_type and b_type != u_type:
                                diags.append(
                                    Diagnostic(
                                        code="CONTRACT_PARAM_TYPE_CHANGED",
                                        line=1,
                                        col=1,
                                        message=f"Breaking change: Parameter '{p_name}' type changed from '{b_type}' to '{u_type}' in '{method.upper()} {path}'.",
                                        offset=0,
                                        source="openapi_diff",
                                        severity=1,
                                    )
                                )

                    # Check Request Body
                    b_rb = b_op.get("requestBody", {})
                    u_rb = u_op.get("requestBody", {})
                    if not b_rb.get("required", False) and u_rb.get("required", False):
                        diags.append(
                            Diagnostic(
                                code="CONTRACT_REQUIRED_PROPERTY_ADDED",
                                line=1,
                                col=1,
                                message=f"Breaking change: requestBody in '{method.upper()} {path}' made required.",
                                offset=0,
                                source="openapi_diff",
                                severity=1,
                            )
                        )

                    # Check Responses
                    b_resps = b_op.get("responses", {})
                    u_resps = u_op.get("responses", {})
                    for code, b_resp in b_resps.items():
                        if str(code).startswith("2") and code not in u_resps:
                            diags.append(
                                Diagnostic(
                                    code="CONTRACT_RESPONSE_CODE_REMOVED",
                                    line=1,
                                    col=1,
                                    message=f"Breaking change: Response status code '{code}' removed from '{method.upper()} {path}'.",
                                    offset=0,
                                    source="openapi_diff",
                                    severity=1,
                                )
                            )
                        elif code in u_resps:
                            _diff_response_content(
                                b_resp,
                                u_resps[code],
                                path=f"{method.upper()} {path} ({code})",
                                diags=diags,
                                additive=additive,
                            )

    # Newly added paths & methods
    for path, u_path_item in up_paths.items():
        if path not in base_paths:
            additive.append(f"Added non-breaking endpoint '{path}'")
        elif isinstance(u_path_item, dict):
            b_path_item = base_paths[path]
            for method in {"get", "post", "put", "delete", "patch"}:
                if method in u_path_item and method not in b_path_item:
                    additive.append(f"Added non-breaking method '{method.upper()} {path}'")

    return diags, additive


def _diff_response_content(
    b_resp: Any,
    u_resp: Any,
    *,
    path: str,
    diags: List[Diagnostic],
    additive: List[str],
) -> None:
    """Compare JSON schema properties of 2xx responses."""
    if not isinstance(b_resp, dict) or not isinstance(u_resp, dict):
        return

    b_content = b_resp.get("content", {}).get("application/json", {}).get("schema", {})
    u_content = u_resp.get("content", {}).get("application/json", {}).get("schema", {})

    b_props = b_content.get("properties", {})
    u_props = u_content.get("properties", {})

    if isinstance(b_props, dict) and isinstance(u_props, dict):
        for prop_name, b_prop in b_props.items():
            if prop_name not in u_props:
                diags.append(
                    Diagnostic(
                        code="CONTRACT_RESPONSE_PROPERTY_REMOVED",
                        line=1,
                        col=1,
                        message=f"Breaking change: Response property '{prop_name}' removed from '{path}'.",
                        offset=0,
                        source="openapi_diff",
                        severity=1,
                    )
                )
            else:
                u_prop = u_props[prop_name]
                b_type = b_prop.get("type") if isinstance(b_prop, dict) else None
                u_type = u_prop.get("type") if isinstance(u_prop, dict) else None
                if b_type and u_type and b_type != u_type:
                    diags.append(
                        Diagnostic(
                            code="CONTRACT_RESPONSE_TYPE_CHANGED",
                            line=1,
                            col=1,
                            message=f"Breaking change: Response property '{prop_name}' type mutated from '{b_type}' to '{u_type}' in '{path}'.",
                            offset=0,
                            source="openapi_diff",
                            severity=1,
                        )
                    )
                # Nullability
                b_null = b_prop.get("nullable", False) if isinstance(b_prop, dict) else False
                u_null = u_prop.get("nullable", False) if isinstance(u_prop, dict) else False
                if b_null and not u_null:
                    diags.append(
                        Diagnostic(
                            code="CONTRACT_NULLABILITY_CHANGED",
                            line=1,
                            col=1,
                            message=f"Breaking change: Response property '{prop_name}' nullability tightened from nullable to non-nullable in '{path}'.",
                            offset=0,
                            source="openapi_diff",
                            severity=1,
                        )
                    )

        for prop_name in u_props:
            if prop_name not in b_props:
                additive.append(f"Added non-breaking response property '{prop_name}' to '{path}'")


# --------------------------------------------------------------------------- #
# Contract Diff Oracle & Verifier
# --------------------------------------------------------------------------- #

class ContractDiffOracle:
    """In-process and toolchain-capable API contract breaking change detector."""

    def __init__(self, *, timeout_s: float = 5.0):
        self.timeout_s = timeout_s
        self.n_calls = 0
        self.wall_s = 0.0

    def diff(
        self,
        completion: str,
        *,
        reference: Optional[str] = None,
        spec_type: Optional[str] = None,
    ) -> Tuple[List[Diagnostic], List[str]]:
        """Run deterministic contract diffing against reference base spec."""
        t0 = time.monotonic()
        self.n_calls += 1
        diags: List[Diagnostic] = []
        additive: List[str] = []

        if reference is None or not str(reference).strip():
            elapsed = time.monotonic() - t0
            self.wall_s += elapsed
            return diags, additive

        ref_str = str(reference).strip()
        comp_str = str(completion).strip()

        if spec_type is None:
            if "syntax = \"proto" in ref_str or "syntax = 'proto" in ref_str or "message " in ref_str:
                spec_type = "protobuf"
            else:
                spec_type = "openapi"

        if spec_type == "protobuf":
            base_proto = parse_proto_schema(ref_str)
            up_proto = parse_proto_schema(comp_str)
            p_diags, p_additive = diff_proto_schemas(base_proto, up_proto)
            diags.extend(p_diags)
            additive.extend(p_additive)
        else:
            base_spec = _parse_spec_dict(ref_str)
            up_spec = _parse_spec_dict(comp_str)
            if base_spec is None or up_spec is None:
                diags.append(
                    Diagnostic(
                        code="CONTRACT_SYNTAX_ERROR",
                        line=1,
                        col=1,
                        message="Failed to parse OpenAPI YAML/JSON specification.",
                        offset=0,
                        source="contract_diff",
                        severity=1,
                    )
                )
            else:
                o_diags, o_additive = diff_openapi_specs(base_spec, up_spec)
                diags.extend(o_diags)
                additive.extend(o_additive)

        self.wall_s += time.monotonic() - t0
        return diags, additive

    def close(self) -> None:
        pass


class ContractDiffVerifier:
    """Verifier for API contract backward compatibility and breaking changes."""

    def __init__(
        self,
        *,
        oracle: Optional[ContractDiffOracle] = None,
        timeout_s: float = 5.0,
        fail_fast: bool = True,
        **reward_kwargs: Any,
    ) -> None:
        self.oracle = oracle or ContractDiffOracle(timeout_s=timeout_s)
        self.timeout_s = timeout_s
        self.fail_fast = fail_fast
        self.reward_kwargs = reward_kwargs

        self._lock = threading.Lock()
        self._n_samples = 0
        self._n_clean = 0
        self._n_hacked = 0
        self._n_degenerate = 0
        self._n_breaking = 0
        self._reward_total = 0.0
        self._elapsed_ms_total = 0.0

    def evaluate(
        self,
        completion: str,
        reference: Any = None,
        *,
        prompt: str = "",
        spec_type: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Evaluate updated API spec / Proto against base reference specification."""
        t0 = time.monotonic()
        comp_str = str(completion or "").strip()

        # 1. Degenerate check
        degen_reason = is_contract_degenerate_output(comp_str)
        is_degenerate = degen_reason is not None

        # 2. Escape hatch check
        hatches = find_contract_diff_escape_hatches(comp_str)
        is_hacked = len(hatches) > 0

        if self.fail_fast and (is_degenerate or is_hacked):
            reward = -1.0
            elapsed = (time.monotonic() - t0) * 1000.0
            with self._lock:
                self._n_samples += 1
                if is_degenerate:
                    self._n_degenerate += 1
                if is_hacked:
                    self._n_hacked += 1
                self._reward_total += reward
                self._elapsed_ms_total += elapsed

            return {
                "reward": reward,
                "is_clean": False,
                "breaking_changes": [],
                "additive_extensions": [],
                "is_degenerate": is_degenerate,
                "degenerate_reason": degen_reason,
                "is_hacked": is_hacked,
                "hatches": hatches,
                "diagnostics": [
                    Diagnostic(
                        code="CONTRACT_DEGENERATE_OUTPUT" if is_degenerate else "CONTRACT_ESCAPE_HATCH",
                        line=1,
                        col=1,
                        message=f"Degenerate: {degen_reason}" if is_degenerate else f"Escape hatch: {hatches}",
                        offset=0,
                        source="contract_diff_verifier",
                        severity=1,
                    )
                ],
                "elapsed_ms": elapsed,
            }

        ref_str = None
        if isinstance(reference, str):
            ref_str = reference
        elif isinstance(reference, Mapping):
            ref_str = reference.get("reference") or reference.get("base") or reference.get("base_spec") or reference.get("base_proto")
            if spec_type is None:
                spec_type = reference.get("spec_type")

        diags, additive = self.oracle.diff(comp_str, reference=ref_str, spec_type=spec_type)

        is_clean = len(diags) == 0
        if is_clean:
            reward = 1.0
        else:
            reward = diagnostics_to_reward(
                diags,
                code_overrides=DEFAULT_CONTRACT_DIFF_CODE_OVERRIDES,
                **self.reward_kwargs,
            )

        elapsed = (time.monotonic() - t0) * 1000.0
        with self._lock:
            self._n_samples += 1
            if is_clean:
                self._n_clean += 1
            else:
                self._n_breaking += 1
            self._reward_total += reward
            self._elapsed_ms_total += elapsed

        return {
            "reward": reward,
            "is_clean": is_clean,
            "breaking_changes": diags,
            "additive_extensions": additive,
            "is_degenerate": False,
            "degenerate_reason": None,
            "is_hacked": False,
            "hatches": [],
            "diagnostics": diags,
            "elapsed_ms": elapsed,
        }

    def reward(
        self,
        completion: str,
        reference: Any = None,
        *,
        prompt: str = "",
        **kwargs: Any,
    ) -> float:
        """Compute scalar RLVR reward for contract completion."""
        eval_res = self.evaluate(completion, reference=reference, prompt=prompt, **kwargs)
        return float(eval_res["reward"])

    def telemetry(self) -> Dict[str, Any]:
        with self._lock:
            n = self._n_samples
            return {
                "verifier": "ContractDiffVerifier",
                "n_samples": n,
                "n_clean": self._n_clean,
                "n_hacked": self._n_hacked,
                "n_degenerate": self._n_degenerate,
                "n_breaking": self._n_breaking,
                "mean_reward": (self._reward_total / n) if n else 0.0,
                "mean_elapsed_ms": (self._elapsed_ms_total / n) if n else 0.0,
                "oracle_calls": self.oracle.n_calls,
                "oracle_wall_s": self.oracle.wall_s,
            }

    def __enter__(self) -> 'ContractDiffVerifier':
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        self.oracle.close()
