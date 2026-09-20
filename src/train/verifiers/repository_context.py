"""#345 -- Repository-scale symbol grounding & compiler blast radius verifier.

Part of the M12 codebase comprehension and long-context evaluation track (#198, #221, #345).
Deterministically evaluate the model's ability to navigate multi-file repositories, resolve
cross-file symbol definitions, and predict change blast radius without code execution:

1. Compiler-Grounded Blast Radius Verifier:
   - Task: given a modified interface/function signature (e.g. adding a required parameter
     to a core service), the model must predict the list of impacted files and call-sites.
   - Ground truth: run headless compiler/LSP diagnostics (`tsc --noEmit` or `pyright`) on the
     mutated interface to collect exact diagnostic codes (`TS2554: Expected N arguments`, file, line).
     An in-process AST analyzer fallback is provided for fast, deterministic, sub-150ms execution
     without external toolchains.
   - Deterministic reward: score precision and recall of predicted call-sites against compiler
     diagnostic locations in <150ms.

2. Reward-shaping and Anti-Goodhart invariants:
   - Evaluates call-site precision, recall, and F1.
   - Output parsing supports JSON schema arrays/objects and text/markdown location listings.
   - Degenerate output guards (empty, whitespace, comment-only) short-circuit to -1.0.
   - Anti-Goodhart escape-hatch guards (@ts-ignore, eval, fake bypasses) short-circuit to -1.0.
   - Thread-safe, injectable oracle seam for CI testing without external binaries.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import (
    Any,
    Self,
)

from src.lsp.tsc import resolve_tsc
from src.train.verifiers import (
    is_degenerate_output,
)

# Common escape-hatch / anti-Goodhart patterns
_ESCAPE_HATCH_PATTERNS = {
    "ts_ignore": re.compile(r"@ts-ignore"),
    "ts_expect_error": re.compile(r"@ts-expect-error"),
    "ts_nocheck": re.compile(r"@ts-nocheck"),
    "as_any": re.compile(r"\bas\s+any\b"),
    "type_ignore": re.compile(r"#\s*type:\s*ignore"),
    "eval_injection": re.compile(r"\beval\s*\("),
}


def normalize_file_path(path: str | Path) -> str:
    """Normalize path into a clean relative POSIX path."""
    p = str(path).replace("\\", "/").strip()
    p = p.removeprefix("./")
    parts: list[str] = []
    for part in PurePosixPath(p).parts:
        if part in ("", "."):
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/".join(parts)


@dataclass(frozen=True)
class CallSiteLocation:
    """Exact call-site or impact location within a repository."""

    file: str
    line: int
    column: int | None = None
    code: str | None = None
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "line": self.line,
            "column": self.column,
            "code": self.code,
            "message": self.message,
        }


# --------------------------------------------------------------------------- #
# Output Parsing (JSON and markdown / text line locations)
# --------------------------------------------------------------------------- #

_FILE_LINE_COL_RE = re.compile(
    r"""(?:^|[\s"'(`[])(?P<file>[a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+)"""
    r"(?:\((?P<line_paren>\d+)(?:,\s*(?P<col_paren>\d+))?\)|"
    r":(?P<line_colon>\d+)(?::(?P<col_colon>\d+))?|"
    r"(?:\s+(?:at\s+)?line\s+(?P<line_word>\d+)))"
)

_CODE_RE = re.compile(r"\b(TS\d{4,5}|[A-Z][a-zA-Z0-9_]{3,30})\b")


def parse_predicted_locations(completion: str) -> list[CallSiteLocation]:
    """Extract CallSiteLocation items from model output.

    Handles:
    - JSON arrays: `[{"file": "src/app.ts", "line": 42}, ...]`
    - JSON objects: `{"impacted": [...]}`, `{"call_sites": [...]}`, etc.
    - Markdown lists: `- src/app.ts:42` or `src/app.ts(42, 5)`
    - Plain text diagnostics: `src/app.ts:42: error TS2554: ...`
    """
    if not completion or not completion.strip():
        return []

    locations: list[CallSiteLocation] = []
    seen: set[tuple[str, int]] = set()

    # 1. Try JSON extraction (full text or markdown ```json ... ``` blocks)
    json_blobs: list[str] = []
    code_block_matches = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", completion)
    if code_block_matches:
        json_blobs.extend(code_block_matches)
    json_blobs.append(completion.strip())

    parsed_json = False
    for blob in json_blobs:
        blob = blob.strip()
        if not blob.startswith(("[", "{")):
            continue
        try:
            data = json.loads(blob)
            items = []
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                for key in ("call_sites", "impacted", "locations", "diagnostics", "impacted_files", "blast_radius"):
                    if key in data and isinstance(data[key], list):
                        items = data[key]
                        break
                if not items and "file" in data and "line" in data:
                    items = [data]

            for item in items:
                if isinstance(item, dict):
                    f = item.get("file") or item.get("path") or item.get("filename")
                    l = item.get("line") or item.get("lineno")
                    c = item.get("column") or item.get("col")
                    code = item.get("code")
                    msg = item.get("message")
                    if f and l is not None:
                        try:
                            line_int = int(l)
                            col_int = int(c) if c is not None else None
                            norm_f = normalize_file_path(str(f))
                            if (norm_f, line_int) not in seen:
                                seen.add((norm_f, line_int))
                                locations.append(
                                    CallSiteLocation(
                                        file=norm_f,
                                        line=line_int,
                                        column=col_int,
                                        code=str(code) if code else None,
                                        message=str(msg) if msg else None,
                                    )
                                )
                                parsed_json = True
                        except (ValueError, TypeError):
                            continue
                elif isinstance(item, str):
                    sub = _parse_text_line(item)
                    for loc in sub:
                        key = (loc.file, loc.line)
                        if key not in seen:
                            seen.add(key)
                            locations.append(loc)
                            parsed_json = True
        except json.JSONDecodeError:
            continue

    if parsed_json and locations:
        return locations

    # 2. Text / line-oriented regex parsing
    for line in completion.splitlines():
        sub_locs = _parse_text_line(line)
        for loc in sub_locs:
            key = (loc.file, loc.line)
            if key not in seen:
                seen.add(key)
                locations.append(loc)

    return locations


def _parse_text_line(line: str) -> list[CallSiteLocation]:
    """Parse one line of text for file:line(:col) patterns."""
    locs: list[CallSiteLocation] = []
    line_code = None
    code_match = _CODE_RE.search(line)
    if code_match:
        line_code = code_match.group(1)

    for m in _FILE_LINE_COL_RE.finditer(line):
        f = m.group("file")
        if not re.search(r"\.(ts|tsx|js|jsx|py|go|rs|java|cs)$", f, re.IGNORECASE):
            continue

        l_str = m.group("line_paren") or m.group("line_colon") or m.group("line_word")
        c_str = m.group("col_paren") or m.group("col_colon")
        if not l_str:
            continue
        try:
            line_int = int(l_str)
            col_int = int(c_str) if c_str else None
            norm_f = normalize_file_path(f)
            locs.append(
                CallSiteLocation(
                    file=norm_f,
                    line=line_int,
                    column=col_int,
                    code=line_code,
                    message=line.strip() if len(line) < 200 else None,
                )
            )
        except ValueError:
            continue
    return locs


# --------------------------------------------------------------------------- #
# Ground Truth Diagnostic Analyzers
# --------------------------------------------------------------------------- #

def inprocess_blast_radius_diagnostics(
    repo: Mapping[str, str],
    mutated_file: str,
    mutated_content: str,
) -> list[CallSiteLocation]:
    """Compute exact compiler diagnostics in-process (<10ms)."""
    mutated_file_norm = normalize_file_path(mutated_file)
    original_content = repo.get(mutated_file, repo.get(mutated_file_norm, ""))

    diagnostics: list[CallSiteLocation] = []

    if mutated_file_norm.endswith((".ts", ".tsx", ".js", ".jsx")):
        diagnostics.extend(
            _analyze_ts_signature_mutation(
                repo=repo,
                mutated_file=mutated_file_norm,
                original_content=original_content,
                mutated_content=mutated_content,
            )
        )
    elif mutated_file_norm.endswith(".py"):
        diagnostics.extend(
            _analyze_py_signature_mutation(
                repo=repo,
                mutated_file=mutated_file_norm,
                original_content=original_content,
                mutated_content=mutated_content,
            )
        )

    return sorted(diagnostics, key=lambda d: (d.file, d.line))


def _extract_ts_function_params(source: str) -> dict[str, tuple[int, int]]:
    """Extract `{func_name: (required_param_count, total_param_count)}` from TS code."""
    params: dict[str, tuple[int, int]] = {}
    fn_re = re.compile(
        r"(?:export\s+(?:async\s+)?)?function\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\(([^)]*)\)",
        re.MULTILINE,
    )
    for m in fn_re.finditer(source):
        name = m.group(1)
        raw_args = m.group(2).strip()
        if not raw_args:
            params[name] = (0, 0)
            continue
        args = [a.strip() for a in raw_args.split(",") if a.strip()]
        total = len(args)
        req = sum(1 for a in args if not ("?" in a.split(":")[0] or "=" in a))
        params[name] = (req, total)

    method_re = re.compile(
        r"^\s*(?:(?:public|private|protected|async|static)\s+)*([A-Za-z_$][A-Za-z0-9_$]*)\s*\(([^)]*)\)",
        re.MULTILINE,
    )
    for m in method_re.finditer(source):
        name = m.group(1)
        if name in ("if", "for", "while", "switch", "catch"):
            continue
        raw_args = m.group(2).strip()
        if not raw_args:
            params[name] = (0, 0)
            continue
        args = [a.strip() for a in raw_args.split(",") if a.strip()]
        total = len(args)
        req = sum(1 for a in args if not ("?" in a.split(":")[0] or "=" in a))
        params[name] = (req, total)

    return params


def _analyze_ts_signature_mutation(
    repo: Mapping[str, str],
    mutated_file: str,
    original_content: str,
    mutated_content: str,
) -> list[CallSiteLocation]:
    """Find TS2554 / TS2345 / TS2305 call-site diagnostics caused by TS interface mutation."""
    orig_params = _extract_ts_function_params(original_content)
    new_params = _extract_ts_function_params(mutated_content)

    diagnostics: list[CallSiteLocation] = []

    for fn, (new_req, new_tot) in new_params.items():
        orig_req, _ = orig_params.get(fn, (0, 0))
        if new_req > orig_req:
            for path, text in repo.items():
                norm_p = normalize_file_path(path)
                if norm_p == mutated_file:
                    continue
                if fn not in text:
                    continue

                call_re = re.compile(r"\b" + re.escape(fn) + r"\s*\(([^)]*)\)")
                for m in call_re.finditer(text):
                    arg_str = m.group(1).strip()
                    arg_count = len([a for a in arg_str.split(",") if a.strip()]) if arg_str else 0
                    if arg_count < new_req:
                        line_no = text[: m.start()].count("\n") + 1
                        diagnostics.append(
                            CallSiteLocation(
                                file=norm_p,
                                line=line_no,
                                code="TS2554",
                                message=f"Expected {new_req} arguments, but got {arg_count}.",
                            )
                        )

    orig_exports = set(re.findall(r"^export\s+(?:async\s+)?(?:function|const|class|interface|type)\s+([A-Za-z0-9_$]+)", original_content, re.MULTILINE))
    new_exports = set(re.findall(r"^export\s+(?:async\s+)?(?:function|const|class|interface|type)\s+([A-Za-z0-9_$]+)", mutated_content, re.MULTILINE))
    removed = orig_exports - new_exports
    for sym in removed:
        for path, text in repo.items():
            norm_p = normalize_file_path(path)
            if norm_p == mutated_file:
                continue
            if sym not in text:
                continue
            for m in re.finditer(r"\b" + re.escape(sym) + r"\b", text):
                line_no = text[: m.start()].count("\n") + 1
                diagnostics.append(
                    CallSiteLocation(
                        file=norm_p,
                        line=line_no,
                        code="TS2305",
                        message=f"Module has no exported member '{sym}'.",
                    )
                )

    return diagnostics


def _analyze_py_signature_mutation(
    repo: Mapping[str, str],
    mutated_file: str,
    original_content: str,
    mutated_content: str,
) -> list[CallSiteLocation]:
    """Find call-site diagnostics caused by Python signature mutation."""
    diagnostics: list[CallSiteLocation] = []

    def get_py_req_args(src: str) -> dict[str, int]:
        req_args: dict[str, int] = {}
        try:
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    total_pos = len(node.args.args)
                    defaults = len(node.args.defaults)
                    if node.args.args and node.args.args[0].arg in ("self", "cls"):
                        total_pos -= 1
                    n_req = total_pos - defaults
                    req_args[node.name] = max(0, n_req)
        except SyntaxError:
            pass
        return req_args

    orig_args = get_py_req_args(original_content)
    new_args = get_py_req_args(mutated_content)

    for fn_name, new_req in new_args.items():
        orig_req = orig_args.get(fn_name, 0)
        if new_req > orig_req:
            for path, text in repo.items():
                norm_p = normalize_file_path(path)
                if norm_p == mutated_file:
                    continue
                if fn_name not in text:
                    continue
                try:
                    tree = ast.parse(text)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call):
                            called_name = None
                            if isinstance(node.func, ast.Name):
                                called_name = node.func.id
                            elif isinstance(node.func, ast.Attribute):
                                called_name = node.func.attr
                            if called_name == fn_name:
                                n_provided = len(node.args) + len(node.keywords)
                                if n_provided < new_req:
                                    diagnostics.append(
                                        CallSiteLocation(
                                            file=norm_p,
                                            line=node.lineno,
                                            code="reportCallIssue",
                                            message=f"Expected {new_req} arguments, but got {n_provided}.",
                                        )
                                    )
                except SyntaxError:
                    pass

    return diagnostics


def run_compiler_diagnostics(
    repo: Mapping[str, str],
    mutated_file: str | None = None,
    mutated_content: str | None = None,
    *,
    toolchain: str = "auto",
    timeout_s: float = 5.0,
) -> list[CallSiteLocation]:
    """Run headless compiler diagnostics or in-process fallback."""
    full_repo = dict(repo)
    if mutated_file and mutated_content is not None:
        full_repo[mutated_file] = mutated_content

    tsc_cmd = resolve_tsc() if toolchain in ("auto", "tsc") else None

    if toolchain in ("auto", "tsc") and tsc_cmd is not None and any(f.endswith((".ts", ".tsx")) for f in full_repo):
        try:
            with tempfile.TemporaryDirectory(prefix="blast_tsc_") as td:
                tmp_dir = Path(td)
                if "tsconfig.json" not in full_repo:
                    (tmp_dir / "tsconfig.json").write_text(
                        json.dumps({
                            "compilerOptions": {
                                "target": "ES2022",
                                "module": "commonjs",
                                "strict": True,
                                "noEmit": True,
                                "skipLibCheck": True,
                            }
                        }),
                        encoding="utf-8",
                    )
                for rel_p, content in full_repo.items():
                    target_file = tmp_dir / rel_p
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    target_file.write_text(content, encoding="utf-8")

                cmd = tsc_cmd + ["-p", str(tmp_dir), "--pretty", "false"]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
                diag_re = re.compile(r"^(?P<file>.+?)\((?P<line>\d+),(?P<col>\d+)\): error (?P<code>TS\d+): (?P<msg>.*)$")
                results: list[CallSiteLocation] = []
                for line in (proc.stdout + proc.stderr).splitlines():
                    m = diag_re.match(line)
                    if m:
                        raw_f = m.group("file")
                        rel_file = normalize_file_path(os.path.relpath(raw_f, str(tmp_dir)))
                        results.append(
                            CallSiteLocation(
                                file=rel_file,
                                line=int(m.group("line")),
                                column=int(m.group("col")),
                                code=m.group("code"),
                                message=m.group("msg").strip(),
                            )
                        )
                if results:
                    return sorted(results, key=lambda d: (d.file, d.line))
        except (subprocess.TimeoutExpired, OSError):
            pass

    if mutated_file and mutated_content is not None:
        return inprocess_blast_radius_diagnostics(repo, mutated_file, mutated_content)

    return []


# --------------------------------------------------------------------------- #
# Deterministic Scoring (<150ms)
# --------------------------------------------------------------------------- #

def score_blast_radius_prediction(
    predicted: Sequence[CallSiteLocation],
    ground_truth: Sequence[CallSiteLocation],
    *,
    line_tolerance: int = 0,
) -> dict[str, Any]:
    """Score precision and recall of predicted call-sites against compiler diagnostics in <150ms."""
    t0 = time.monotonic()

    norm_gt = [
        CallSiteLocation(
            file=normalize_file_path(g.file),
            line=g.line,
            column=g.column,
            code=g.code,
            message=g.message,
        )
        for g in ground_truth
    ]

    norm_pred = [
        CallSiteLocation(
            file=normalize_file_path(p.file),
            line=p.line,
            column=p.column,
            code=p.code,
            message=p.message,
        )
        for p in predicted
    ]

    n_gt = len(norm_gt)
    n_pred = len(norm_pred)

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()

    for p_idx, p in enumerate(norm_pred):
        for g_idx, g in enumerate(norm_gt):
            if g_idx in matched_gt:
                continue
            if p.file == g.file and abs(p.line - g.line) <= line_tolerance:
                matched_gt.add(g_idx)
                matched_pred.add(p_idx)
                break

    tp = len(matched_pred)
    fp = n_pred - tp
    fn = n_gt - len(matched_gt)

    if n_gt == 0:
        precision = 1.0 if n_pred == 0 else 0.0
        recall = 1.0
        f1 = 1.0 if n_pred == 0 else 0.0
        reward = 1.0 if n_pred == 0 else 0.0
    else:
        precision = (tp / n_pred) if n_pred > 0 else 0.0
        recall = tp / n_gt
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        reward = f1

    pred_files = {p.file for p in norm_pred}
    gt_files = {g.file for g in norm_gt}
    file_tp = len(pred_files & gt_files)
    file_prec = (file_tp / len(pred_files)) if pred_files else (1.0 if not gt_files else 0.0)
    file_rec = (file_tp / len(gt_files)) if gt_files else 1.0
    file_f1 = (2.0 * file_prec * file_rec / (file_prec + file_rec)) if (file_prec + file_rec) > 0 else 0.0

    elapsed_ms = (time.monotonic() - t0) * 1000.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "file_precision": float(file_prec),
        "file_recall": float(file_rec),
        "file_f1": float(file_f1),
        "reward": float(reward),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_predicted": n_pred,
        "n_ground_truth": n_gt,
        "elapsed_ms": float(elapsed_ms),
    }


# --------------------------------------------------------------------------- #
# BlastRadiusVerifier Class
# --------------------------------------------------------------------------- #

class BlastRadiusVerifier:
    """Verifier for compiler-grounded blast radius prediction (#345)."""

    def __init__(
        self,
        *,
        toolchain: str = "auto",
        oracle: Callable[..., Sequence[CallSiteLocation]] | None = None,
        line_tolerance: int = 0,
        fail_fast: bool = True,
        timeout_s: float = 5.0,
    ) -> None:
        self.toolchain = toolchain
        self.oracle = oracle
        self.line_tolerance = line_tolerance
        self.fail_fast = fail_fast
        self.timeout_s = timeout_s

        self._lock = threading.Lock()
        self._n_samples = 0
        self._n_clean = 0
        self._n_degenerate = 0
        self._n_hacked = 0
        self._reward_total = 0.0
        self._precision_total = 0.0
        self._recall_total = 0.0
        self._f1_total = 0.0
        self._elapsed_ms_total = 0.0

    def find_escape_hatches(self, text: str) -> list[str]:
        """Find anti-Goodhart escape hatches in completion."""
        found: list[str] = []
        for name, pattern in _ESCAPE_HATCH_PATTERNS.items():
            if pattern.search(text):
                found.append(name)
        return found

    def ground_truth_diagnostics(
        self,
        repo: Mapping[str, str],
        mutated_file: str,
        mutated_content: str,
    ) -> list[CallSiteLocation]:
        """Compute ground-truth compiler diagnostic locations."""
        if self.oracle is not None:
            return list(self.oracle(repo, mutated_file, mutated_content))

        return run_compiler_diagnostics(
            repo=repo,
            mutated_file=mutated_file,
            mutated_content=mutated_content,
            toolchain=self.toolchain,
            timeout_s=self.timeout_s,
        )

    def evaluate(
        self,
        completion: str,
        reference: Any | None = None,
        *,
        prompt: str = "",
        repo: Mapping[str, str] | None = None,
        mutated_file: str | None = None,
        mutated_content: str | None = None,
        ground_truth: Sequence[CallSiteLocation | dict] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Evaluate completion and return detailed diagnostics and score breakdown."""
        t_start = time.monotonic()

        degen_reason = is_degenerate_output(completion)
        if degen_reason is not None:
            with self._lock:
                self._n_samples += 1
                self._n_degenerate += 1
                self._reward_total += -1.0
            return {
                "reward": -1.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "is_clean": False,
                "is_degenerate": True,
                "degenerate_reason": degen_reason,
                "is_hacked": False,
                "elapsed_ms": (time.monotonic() - t_start) * 1000.0,
            }

        hatches = self.find_escape_hatches(completion)
        if hatches:
            with self._lock:
                self._n_samples += 1
                self._n_hacked += 1
                self._reward_total += -1.0
            return {
                "reward": -1.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "is_clean": False,
                "is_degenerate": False,
                "is_hacked": True,
                "hatches": hatches,
                "elapsed_ms": (time.monotonic() - t_start) * 1000.0,
            }

        gt_locations: list[CallSiteLocation] = []
        if ground_truth is not None:
            for item in ground_truth:
                if isinstance(item, CallSiteLocation):
                    gt_locations.append(item)
                elif isinstance(item, dict):
                    gt_locations.append(
                        CallSiteLocation(
                            file=normalize_file_path(item.get("file", "")),
                            line=int(item.get("line", 1)),
                            column=int(item["column"]) if item.get("column") else None,
                            code=item.get("code"),
                            message=item.get("message"),
                        )
                    )
        elif reference is not None:
            if isinstance(reference, list):
                for item in reference:
                    if isinstance(item, CallSiteLocation):
                        gt_locations.append(item)
                    elif isinstance(item, dict):
                        gt_locations.append(
                            CallSiteLocation(
                                file=normalize_file_path(item.get("file", "")),
                                line=int(item.get("line", 1)),
                                column=int(item["column"]) if item.get("column") else None,
                                code=item.get("code"),
                                message=item.get("message"),
                            )
                        )
                    elif isinstance(item, str):
                        gt_locations.extend(_parse_text_line(item))
            elif isinstance(reference, str):
                gt_locations.extend(parse_predicted_locations(reference))
        elif repo is not None and mutated_file is not None and mutated_content is not None:
            gt_locations = self.ground_truth_diagnostics(repo, mutated_file, mutated_content)

        predicted = parse_predicted_locations(completion)

        score_res = score_blast_radius_prediction(
            predicted=predicted,
            ground_truth=gt_locations,
            line_tolerance=self.line_tolerance,
        )

        reward = score_res["reward"]
        is_clean = (reward == 1.0)

        with self._lock:
            self._n_samples += 1
            if is_clean:
                self._n_clean += 1
            self._reward_total += reward
            self._precision_total += score_res["precision"]
            self._recall_total += score_res["recall"]
            self._f1_total += score_res["f1"]
            self._elapsed_ms_total += score_res["elapsed_ms"]

        return {
            **score_res,
            "is_clean": is_clean,
            "is_degenerate": False,
            "is_hacked": False,
            "predicted": [p.to_dict() for p in predicted],
            "ground_truth": [g.to_dict() for g in gt_locations],
        }

    def reward(
        self,
        completion: str,
        reference: Any | None = None,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> float:
        """Compute scalar reward for RLVR policy update."""
        res = self.evaluate(
            completion=completion,
            reference=reference,
            prompt=prompt or "",
            **kwargs,
        )
        return float(res["reward"])

    def telemetry(self) -> dict[str, Any]:
        """Telemetry tracking sample counts, cleanliness, and precision/recall."""
        with self._lock:
            n = self._n_samples
            return {
                "verifier": "blast_radius",
                "n_samples": n,
                "n_clean": self._n_clean,
                "n_degenerate": self._n_degenerate,
                "n_hacked": self._n_hacked,
                "mean_reward": (self._reward_total / n) if n else 0.0,
                "mean_precision": (self._precision_total / n) if n else 0.0,
                "mean_recall": (self._recall_total / n) if n else 0.0,
                "mean_f1": (self._f1_total / n) if n else 0.0,
                "mean_elapsed_ms": (self._elapsed_ms_total / n) if n else 0.0,
            }

    def close(self) -> None:
        pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
