"""Harness safety gates and immediate post-mutation diagnostic feedback (#351).

Implements:
1. Workspace Containment: Path jailbreak validation preventing symlinks and
   relative/absolute traversal outside the workspace repository root.
2. Read-Before-Write Safety Gate: Session-scoped file read registry and SHA-256
   content hash verification preventing blind overwrites and edits to unread files.
3. Immediate Post-Edit Diagnostic Feedback: Fast static analysis (ruff, pyflakes,
   Python AST, TsLspService, tsc) invoked immediately upon file mutation, appending
   findings directly to tool observations.

ABOVE THE SEAM — pure Python standard library only. No hardware backends
(mlx/torch/bitsandbytes) are imported anywhere in this module (enforced by
tests/test_import_guard.py).
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Read-Before-Write Registry & SHA-256 Hash Verification
# --------------------------------------------------------------------------- #

@dataclass
class FileReadRecord:
    """Record of a file accessed during an agent session."""

    path: str
    sha256: str
    read_at: float
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "read_at": self.read_at,
            "size_bytes": self.size_bytes,
        }


class FileReadRegistry:
    """Session-level registry tracking files read and their SHA-256 content hashes (#351).

    Prevents blind overwrites and unread file modifications by enforcing that any file
    targeted for edit or overwrite must have been read during the active session.
    """

    def __init__(self) -> None:
        self._records: dict[str, FileReadRecord] = {}

    def register_read(self, rel_path: str, content: str | bytes) -> FileReadRecord:
        """Register a file read operation with its SHA-256 hash."""
        clean_path = str(rel_path).strip()
        data = content.encode("utf-8") if isinstance(content, str) else bytes(content)
        digest = hashlib.sha256(data).hexdigest()
        rec = FileReadRecord(
            path=clean_path,
            sha256=digest,
            read_at=time.time(),
            size_bytes=len(data),
        )
        self._records[clean_path] = rec
        return rec

    def register_write(self, rel_path: str, content: str | bytes) -> FileReadRecord:
        """Register a file write or edit mutation, updating its recorded SHA-256 hash."""
        clean_path = str(rel_path).strip()
        data = content.encode("utf-8") if isinstance(content, str) else bytes(content)
        digest = hashlib.sha256(data).hexdigest()
        rec = FileReadRecord(
            path=clean_path,
            sha256=digest,
            read_at=time.time(),
            size_bytes=len(data),
        )
        self._records[clean_path] = rec
        return rec

    def is_read(self, rel_path: str) -> bool:
        """Check whether a file path has been read in the current session."""
        return str(rel_path).strip() in self._records

    def get_hash(self, rel_path: str) -> str | None:
        """Get the recorded SHA-256 digest for a path, or None if unread."""
        rec = self._records.get(str(rel_path).strip())
        return rec.sha256 if rec else None

    def get_record(self, rel_path: str) -> FileReadRecord | None:
        """Get the full read record for a path."""
        return self._records.get(str(rel_path).strip())

    def clear(self) -> None:
        """Clear all session read records."""
        self._records.clear()

    def all_records(self) -> dict[str, FileReadRecord]:
        """Return a copy of all active read records."""
        return dict(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, rel_path: str) -> bool:
        return str(rel_path).strip() in self._records


# --------------------------------------------------------------------------- #
# Workspace Containment & Jailbreak Prevention
# --------------------------------------------------------------------------- #

def resolve_safe_workspace_path(
    path_str: str,
    workspace_root: str | Path,
) -> tuple[Path | None, str | None]:
    """Resolve a path safely within workspace_root, preventing jailbreaks.

    Enforces workspace containment checks:
      - Traversal attempts with `..` escaping workspace boundary are rejected.
      - Absolute paths outside workspace boundary are rejected.
      - Symlinks pointing outside the repository root are rejected.
      - Non-existent files under symlinked external directories are rejected.
      - Null bytes and empty paths are rejected.

    Returns:
        tuple (resolved_path, error_message). On success, error_message is None.
    """
    if not path_str or "\x00" in path_str:
        return None, f"Security error: invalid path '{path_str}'"

    try:
        root = Path(workspace_root).resolve()
        raw = Path(path_str)

        # Build candidate path
        if raw.is_absolute():
            candidate = raw.resolve()
        else:
            candidate = (root / raw).resolve()

        # Enforce containment within workspace root
        if not candidate.is_relative_to(root):
            return None, f"Security error: path '{path_str}' escapes workspace boundary"

        return candidate, None
    except (ValueError, TypeError, OSError) as e:
        return None, f"Security error: invalid path '{path_str}': {e}"


# --------------------------------------------------------------------------- #
# Immediate Post-Edit Diagnostic Feedback
# --------------------------------------------------------------------------- #

def _find_ruff_bin() -> str | None:
    """Locate the ruff executable in the active Python environment or PATH."""
    # 1. ruff python package entry point
    try:
        import ruff

        if hasattr(ruff, "find_ruff_bin"):
            bin_path = ruff.find_ruff_bin()
            if bin_path and os.path.isfile(bin_path) and os.access(bin_path, os.X_OK):
                return str(bin_path)
    except (ImportError, Exception):  # noqa: BLE001, S110
        pass

    # 2. Sibling binary in Python environment
    py_bin = Path(sys.executable).parent / "ruff"
    if py_bin.is_file() and os.access(py_bin, os.X_OK):
        return str(py_bin)

    # 3. System PATH
    which_bin = shutil.which("ruff")
    if which_bin:
        return which_bin

    return None


def _check_ruff(path: Path | str, content: str) -> list[dict[str, Any]] | None:
    """Run fast static analysis using ruff over stdin."""
    bin_path = _find_ruff_bin()
    if not bin_path:
        return None

    try:
        proc = subprocess.run(
            [bin_path, "check", "--output-format=json", "--stdin-filename", str(path), "-"],
            input=content,
            text=True,
            capture_output=True,
            timeout=5.0,
            check=False,
        )
        if proc.stdout.strip():
            raw_items = json.loads(proc.stdout)
            diagnostics: list[dict[str, Any]] = []
            for it in raw_items:
                loc = it.get("location") or {}
                diagnostics.append({
                    "line": int(loc.get("row", 1)),
                    "column": int(loc.get("column", 1)),
                    "message": str(it.get("message", "")),
                    "code": str(it.get("code", "E999")),
                    "severity": str(it.get("severity", "error")),
                    "source": "ruff",
                })
            return diagnostics
        elif proc.returncode == 0:
            return []
        return None
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        return None


def _check_pyflakes(path: Path | str, content: str) -> list[dict[str, Any]] | None:
    """Run pyflakes static analysis if available."""
    try:
        import io

        from pyflakes import api, reporter

        warn_buf = io.StringIO()
        err_buf = io.StringIO()
        rep = reporter.Reporter(warn_buf, err_buf)
        api.check(content, filename=str(path), reporter=rep)

        out = warn_buf.getvalue() + err_buf.getvalue()
        if not out.strip():
            return []

        diagnostics: list[dict[str, Any]] = []
        for line in out.strip().splitlines():
            parts = line.split(":", 3)
            if len(parts) >= 3:
                try:
                    lineno = int(parts[1])
                    col = int(parts[2]) if len(parts) > 3 and parts[2].strip().isdigit() else 1
                    msg = parts[-1].strip()
                    diagnostics.append({
                        "line": lineno,
                        "column": col,
                        "message": msg,
                        "code": "pyflakes",
                        "severity": "error" if "syntax" in msg.lower() else "warning",
                        "source": "pyflakes",
                    })
                except ValueError:
                    diagnostics.append({
                        "line": 1,
                        "column": 1,
                        "message": line.strip(),
                        "code": "pyflakes",
                        "severity": "warning",
                        "source": "pyflakes",
                    })
        return diagnostics
    except (ImportError, Exception):  # noqa: BLE001
        return None


def _check_python_ast(path: Path | str, content: str) -> list[dict[str, Any]]:
    """Pure Python AST syntax check fallback."""
    try:
        ast.parse(content, filename=str(path))
        return []
    except SyntaxError as e:
        return [{
            "line": int(e.lineno or 1),
            "column": int(e.offset or 1),
            "message": f"SyntaxError: {e.msg}",
            "code": "SyntaxError",
            "severity": "error",
            "source": "python_ast",
        }]


def _check_json(content: str) -> list[dict[str, Any]]:
    """Validate JSON file syntax."""
    try:
        json.loads(content)
        return []
    except json.JSONDecodeError as e:
        return [{
            "line": int(e.lineno),
            "column": int(e.colno),
            "message": f"JSONDecodeError: {e.msg}",
            "code": "JSONDecodeError",
            "severity": "error",
            "source": "json_parser",
        }]


def _check_ts_lsp(path: Path | str, content: str, service: Any) -> list[dict[str, Any]]:
    """Invoke TypeScript LSP service (TsLspService) for diagnostic feedback."""
    try:
        if hasattr(service, "update") and callable(service.update):
            service.update(str(path), content)
        if hasattr(service, "diagnostics") and callable(service.diagnostics):
            diags = service.diagnostics(str(path))
            results: list[dict[str, Any]] = []
            for d in diags:
                if isinstance(d, dict):
                    results.append(d)
                else:
                    results.append({
                        "line": getattr(d, "line", 1),
                        "column": getattr(d, "col", getattr(d, "column", 1)),
                        "message": getattr(d, "message", str(d)),
                        "code": getattr(d, "code", ""),
                        "severity": "error" if getattr(d, "severity", 1) == 1 else "warning",
                        "source": getattr(d, "source", "ts_lsp"),
                    })
            return results
    except Exception:  # noqa: BLE001, S110
        pass
    return []


def format_diagnostic_summary(diagnostics: list[dict[str, Any]]) -> str:
    """Generate concise, single-line human-readable summary of diagnostic findings."""
    if not diagnostics:
        return ""
    count = len(diagnostics)
    first = diagnostics[0]
    line = first.get("line", 1)
    msg = first.get("message", "diagnostic finding")
    code = first.get("code", "")
    code_str = f" [{code}]" if code else ""
    return f"Static analysis: {count} issue(s) detected. Line {line}:{code_str} {msg}"


def run_file_diagnostics(
    path: str | Path,
    content: str | None = None,
    *,
    custom_provider: Callable[[str, str], list[dict[str, Any]]] | Any | None = None,
    ts_lsp_service: Any | None = None,
) -> list[dict[str, Any]]:
    """Run fast static analysis and linter diagnostics on mutated file content (#351).

    Dispatches based on file extension:
      - Python (.py, .pyi): ruff check -> pyflakes -> ast.parse
      - TypeScript/JavaScript (.ts, .tsx, .js, .jsx): ts_lsp_service -> custom_provider
      - JSON (.json): json.loads validation
      - Custom diagnostic providers take precedence when supplied.

    Returns a list of diagnostic dictionaries:
      [{"line": int, "column": int, "message": str, "code": str, "severity": str, "source": str}, ...]
    """
    path_obj = Path(path)
    ext = path_obj.suffix.lower()

    # Read content from file if not directly provided
    if content is None:
        try:
            content = path_obj.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []

    # 1. Custom diagnostic provider hook
    if custom_provider is not None:
        try:
            if callable(custom_provider):
                res = custom_provider(str(path), content)
            elif hasattr(custom_provider, "diagnostics") and callable(custom_provider.diagnostics):
                if hasattr(custom_provider, "update") and callable(custom_provider.update):
                    custom_provider.update(str(path), content)
                res = custom_provider.diagnostics(str(path))
            else:
                res = None

            if isinstance(res, list):
                norm: list[dict[str, Any]] = []
                for item in res:
                    if isinstance(item, dict):
                        norm.append(item)
                    else:
                        norm.append({
                            "line": getattr(item, "line", 1),
                            "column": getattr(item, "col", getattr(item, "column", 1)),
                            "message": getattr(item, "message", str(item)),
                            "code": getattr(item, "code", ""),
                            "severity": "error" if getattr(item, "severity", 1) == 1 else "warning",
                            "source": getattr(item, "source", "custom"),
                        })
                return norm
        except Exception:  # noqa: BLE001, S110
            pass

    # 2. Python files
    if ext in (".py", ".pyi"):
        # Try ruff
        ruff_diags = _check_ruff(path, content)
        if ruff_diags is not None:
            return ruff_diags

        # Try pyflakes
        pyflakes_diags = _check_pyflakes(path, content)
        if pyflakes_diags is not None:
            return pyflakes_diags

        # Fallback to python AST syntax parser
        return _check_python_ast(path, content)

    # 3. TypeScript / JavaScript files
    if ext in (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"):
        if ts_lsp_service is not None:
            return _check_ts_lsp(path, content, ts_lsp_service)
        return []

    # 4. JSON files
    if ext == ".json":
        return _check_json(content)

    return []
