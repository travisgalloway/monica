"""Verifiable rewards for RLVR / GRPO (#78).

The cleanest post-training stage on licensing: the reward comes from a **verifier**
(exact-match, compiler, test runner) — not licensed data — so only problems + tests are
needed; the model generates the solutions (docs/design/08-corpus-pipeline.md lines 120-123).
**Math first** (exact-match, no sandbox — the cheapest clean reward loop), then a guarded
code path.

ABOVE THE SEAM — stdlib only, no backend. `exact_match_reward` / `math_reward` are pure and
safe. `CodeVerifier` **executes untrusted model output** and is therefore disabled by
default (`enabled=False`) and never run in CI; real TS/Rust/SQL grading uses an external
sandbox (SandboxFusion), out of scope here.

#230 adds a THIRD verifier, `LspVerifier` — the diagnostic-cleanliness reward for the
RLVR/GRPO arm of #198's SSI axis. Unlike `CodeVerifier`, it never executes model output
(static analysis only via `src.lsp.oracle.CompositeOracle`), so it is **enabled by
default**. Its pure reward-shaping core (`severity_weight`, `is_degenerate_output`,
`diagnostics_to_reward`) has no oracle/process dependency at all and is unit-testable in
CI on synthetic `Diagnostic`s; `CompositeOracle`/`src.lsp.diagnostics` imports are kept
function-local (repo idiom — see `src.lsp.oracle`/`src.eval.ssi_contract`) so this
module's top-level import surface stays stdlib-only.
"""

from __future__ import annotations

from pathlib import Path as _Path
__path__ = [str(_Path(__file__).parent / "verifiers")]

import ast
import json
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from src.lsp.diagnostics import Diagnostic

_WS = re.compile(r"\s+")
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def normalize_text(s: str) -> str:
    """Lowercase, trim, collapse internal whitespace."""
    return _WS.sub(" ", (s or "").strip().lower())


def exact_match_reward(answer: str, gold: str) -> float:
    """1.0 if `answer` matches `gold` after normalization, else 0.0."""
    return 1.0 if normalize_text(answer) == normalize_text(gold) else 0.0


def extract_final_number(s: str) -> Optional[float]:
    """The final numeric answer in `s` (GSM8K-style: prefer text after `####`, else the
    last number). Strips thousands separators. Returns None if there is no number."""
    if not s:
        return None
    txt = s.split("####")[-1] if "####" in s else s
    nums = _NUM.findall(txt.replace(",", ""))
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def math_reward(answer: str, gold: str, *, tol: float = 1e-6, use_sympy: bool = True) -> float:
    """1.0 if the final number in `answer` equals the gold answer within `tol`, or if
    the expressions are symbolically equivalent via SymPy (#340), else 0.0.
    `gold` may be a bare number or a full solution string."""
    a, g = extract_final_number(answer), extract_final_number(gold)
    if a is not None and g is not None and abs(a - g) <= tol:
        return 1.0
    if use_sympy:
        try:
            return sympy_symbolic_reward(answer, gold, tol=tol)
        except Exception:
            return 0.0
    return 0.0


class CodeVerifier:
    """Run candidate code against a test suite, rewarding the **fraction of tests passing**
    (partial credit; use >=5 tests per problem so a thin suite can't be gamed — the main
    RLVR failure mode).

    UNSAFE when run locally: executes untrusted model output in a subprocess. Disabled by
    default; set `enabled=True` to opt in for local execution.
    For safe execution in untrusted rollouts and RLVR sweeps, pass an isolated container sandbox
    (`sandbox=ContainerSandbox(...)` or `sandbox="container"` / `"docker"`).
    """

    def __init__(
        self,
        *,
        timeout: float = 5.0,
        enabled: bool = False,
        sandbox: Any | None = None,
    ) -> None:
        self.timeout = timeout
        self.enabled = enabled
        if isinstance(sandbox, str):
            from src.runtime.sandbox import create_sandbox

            self.sandbox = create_sandbox(sandbox, timeout_s=timeout)
        else:
            self.sandbox = sandbox

    def reward(self, code: str, tests: Sequence[str]) -> float:
        is_local = self.sandbox is None or getattr(self.sandbox, "is_local", False)
        if not self.enabled and is_local:
            raise RuntimeError(
                "CodeVerifier is disabled (it executes untrusted code). "
                "Pass enabled=True to opt in — never in CI."
            )
        if not tests:
            return 0.0
        passed = 0
        for t in tests:
            program = f"{code}\n{t}\n"
            if self.sandbox is not None:
                res = self.sandbox.run_python(program, timeout=self.timeout)
                passed += int(res.exit_code == 0)
            else:
                try:
                    # Discard untrusted stdout/stderr (a candidate can print unbounded data);
                    # only the exit code matters.
                    r = subprocess.run(
                        [sys.executable, "-c", program],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=self.timeout,
                    )
                    passed += int(r.returncode == 0)
                except subprocess.TimeoutExpired:
                    pass
        return passed / len(tests)



# --------------------------------------------------------------------------- #
# #230 -- the LSP/opengrep verifier reward (diagnostic-cleanliness ranking).
# Pure reward-shaping core first (no oracle, no process, CI-testable on
# synthetic Diagnostics); LspVerifier wraps it around a real/fake oracle.
# --------------------------------------------------------------------------- #

#: LSP severity 1=Error .. 4=Hint. Only Errors are load-bearing today (the LSP
#: arm already drops everything below severity 1 -- see `ts_lsp.py`'s Trap B),
#: but the shape is severity-general so an opengrep finding (which can report
#: any severity) or a future arm doesn't need a second reward function.
DEFAULT_SEVERITY_WEIGHTS: Dict[int, float] = {1: 0.30, 2: 0.15, 3: 0.05, 4: 0.05}

_DIRECTIVE_HATCHES = frozenset({"ts_ignore", "ts_expect_error", "ts_nocheck",
                                "as_any", "as_unknown_as"})
_HATCH_MODES = ("superset", "directives", "none")


def severity_weight(d: "Diagnostic", weights: Mapping[int, float] = DEFAULT_SEVERITY_WEIGHTS,
                    code_overrides: Optional[Mapping[str, float]] = None) -> float:
    """The reward-shaping penalty for one diagnostic: `code_overrides[d.code]` if
    given and present, else `weights[d.severity]`, else the Hint weight as a
    conservative default for a severity this table doesn't recognize."""
    if code_overrides is not None and d.code in code_overrides:
        return code_overrides[d.code]
    return weights.get(d.severity, weights.get(4, 0.05))


def is_degenerate_output(text: str, *, min_code_chars: int = 2) -> Optional[str]:
    """The empty-output-trap guard (#230): empty / whitespace-only /
    comment-only / near-empty completions have zero diagnostics and would
    otherwise score a perfect `+1.0`, teaching the policy to emit nothing.
    Returns a reason string (`"empty"`, `"whitespace_only"`, `"comment_only"`,
    `"too_short"`) so telemetry can name the trap rather than report an opaque
    bool, or `None` if `text` looks like real code.

    Masks strings/comments with `src.lsp.diagnostics.mask_strings_and_comments`
    (function-local import, keeping this module's top-level stdlib-only), then
    strips whitespace plus residual `/`/`*` punctuation -- that module's own
    docstring documents that the SECOND character of a two-char comment token
    (`//`, `/*`, `*/`) is never independently visited and survives unmasked,
    which is exactly the stray punctuation a comment-only artifact leaves
    behind; stripping it is what makes `"// only a comment\n"` mask down to
    nothing rather than to one leftover character.
    """
    if not text:
        return "empty"
    if not text.strip():
        return "whitespace_only"

    from src.lsp.diagnostics import mask_strings_and_comments

    masked = mask_strings_and_comments(text)
    residual = masked.translate(str.maketrans("", "", "/*")).strip()
    if not residual:
        return "comment_only" if masked.strip() else "whitespace_only"
    if len(residual) < min_code_chars:
        return "too_short"
    return None


def diagnostics_to_reward(diags: Sequence["Diagnostic"], *, hacked: bool = False,
                          degenerate: bool = False, base_clean: float = 1.0,
                          severity_weights: Mapping[int, float] = DEFAULT_SEVERITY_WEIGHTS,
                          code_overrides: Optional[Mapping[str, float]] = None,
                          diag_floor: float = -0.5, hack_reward: float = -1.0,
                          degenerate_reward: float = -1.0) -> float:
    """`reward = max(base_clean - sum(severity_weight(d) for d in diags), diag_floor)`,
    then `hacked or degenerate` short-circuits to `min(hack_reward, degenerate_reward)`
    (both default to the same -1.0, but the two are configurable separately and the
    dominance still has to hold if a caller ever splits them).

    `diag_floor` (default -0.5) sits strictly ABOVE the hack floor (-1.0) on purpose:
    without it, a many-diagnostic honest attempt could score *below* a suppression
    hack (inverting the intended ranking) and would become a single outlier that
    squashes an entire group's advantages toward zero after standardization. Both
    are silent training failures the floor exists to prevent -- it is load-bearing,
    not cosmetic.

    Pure: no process, no I/O, no oracle -- takes `List[Diagnostic]` plus two bools,
    so the whole reward shape is unit-testable in CI on synthetic diagnostics.
    """
    if hacked or degenerate:
        return min(hack_reward, degenerate_reward)
    penalty = sum(severity_weight(d, severity_weights, code_overrides) for d in diags)
    return max(base_clean - penalty, diag_floor)


def _select_hatches(all_hatches: Sequence[str], mode: str) -> List[str]:
    """Filter `find_escape_hatches`' full result to the subset `mode` treats as
    hack-floor-triggering. `"superset"` (default, the #225 M5 gate) keeps every
    hatch; `"directives"` keeps only the five explicit-suppression hatches
    (`ts_ignore`/`ts_expect_error`/`ts_nocheck`/`as_any`/`as_unknown_as`), for
    the ablation that isolates the two known-noisy superset members
    (`empty_body`, `non_null_assertion` -- see `src.lsp.diagnostics`'s module
    docstring for their documented false positives); `"none"` disables the
    hack floor entirely (control arm only). Always computed FROM the same
    `find_escape_hatches` call -- never a separate regex set -- so "applied
    identically across arms" stays a property of the import."""
    if mode == "superset":
        return list(all_hatches)
    if mode == "directives":
        return [h for h in all_hatches if h in _DIRECTIVE_HATCHES]
    if mode == "none":
        return []
    raise ValueError(f"unknown hatches mode {mode!r} (want one of {_HATCH_MODES})")


class LspVerifier:
    """The #230 diagnostic-cleanliness verifier reward: wraps
    `src.lsp.oracle.CompositeOracle` around `diagnostics_to_reward`. Unlike
    `CodeVerifier`, this verifier never EXECUTES model output -- static
    analysis only -- so it is **enabled by default**, as the issue specifies.

    Scoring: the oracle sees `prompt + completion` (a full, scoreable
    artifact -- a bare completion fragment like `name);\\n` is not standalone
    TypeScript, so scoring it alone would be pure syntax noise). The two
    anti-Goodhart guards (`find_escape_hatches`, `is_degenerate_output`) see
    `completion` ALONE -- the prompt may legitimately contain `as any` or an
    unfinished body, and only the model's own text should be penalized.

    No prompt-baseline subtraction, deliberately: the prompt's own diagnostics
    are constant across all K samples of a GRPO group, and `group_advantages`
    standardizes within the group, so a constant offset cancels exactly.
    Subtracting it would cost one extra oracle call per prompt for no gradient
    change (it would shift `mean_reward` in the logs, though -- noted in the
    design doc so that number isn't misread).

    `.reward(completion, reference=None, *, prompt="")` matches
    `scripts/rlvr.py`'s two-positional-arg `reward_fn(decoded, ref)` call site;
    `prompt` is keyword-only and bound per training step by the driver
    (`functools.partial`) rather than held as hidden per-prompt verifier state.

    Lazy oracle + injectable seam: `oracle=None` (default) constructs a
    `CompositeOracle` on first `.reward()` call (function-local import, so this
    module's top-level surface stays stdlib-only); tests inject a fake exposing
    `.diagnostics()/.close()/.n_calls/.wall_s/.ts_stats/.opengrep_stats`, so the
    whole class is CI-testable with no toolchain.
    """

    def __init__(self, *, kind: str = "ts", timeout_s: float = 10.0,
                ignore_module_resolution: bool = True, hatches: str = "superset",
                oracle=None, on_error: str = "raise", fail_fast: bool = False, **reward_kwargs):
        if hatches not in _HATCH_MODES:
            raise ValueError(f"unknown hatches mode {hatches!r} (want one of {_HATCH_MODES})")
        if on_error not in ("raise", "skip"):
            raise ValueError(f"on_error must be 'raise' or 'skip', got {on_error!r}")
        self.kind = kind
        self.timeout_s = timeout_s
        self.ignore_module_resolution = ignore_module_resolution
        self.hatches = hatches
        self.on_error = on_error
        self.fail_fast = fail_fast
        self.reward_kwargs = reward_kwargs

        self._oracle = oracle
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

    def _ensure_oracle(self):
        if self._oracle is None:
            from src.lsp.oracle import CompositeOracle
            self._oracle = CompositeOracle(self.kind, timeout_s=self.timeout_s)
        return self._oracle

    def reward(self, completion: str, reference: Optional[str] = None, *,
              prompt: str = "") -> Optional[float]:
        """Score one completion. Returns `None` only when `on_error="skip"` and
        the oracle call raised -- the caller must then drop the WHOLE GRPO
        group for that step (a partially-scored group has a meaningless
        baseline). `reference` is accepted (and ignored) only to match the
        two-positional-arg `reward_fn(decoded, ref)` call site every
        `scripts/rlvr.py` reward function shares; Phase 1 has no reference-
        based term (see the design doc's Phase 2 note)."""
        from src.lsp.diagnostics import MODULE_RESOLUTION_CODES, drop_codes, find_escape_hatches

        completion = "" if completion is None else str(completion)
        self._n_samples += 1

        reason = is_degenerate_output(completion)
        degenerate = reason is not None
        if degenerate:
            self._n_degenerate += 1
            self._degenerate_reasons[reason] = self._degenerate_reasons.get(reason, 0) + 1

        all_hatches = find_escape_hatches(completion)
        for h in all_hatches:
            self._hatch_counts[h] = self._hatch_counts.get(h, 0) + 1
        hacked = bool(_select_hatches(all_hatches, self.hatches))
        if hacked:
            self._n_hacked += 1

        if self.fail_fast and (hacked or degenerate):
            diags = []
        else:
            with self._lock:
                oracle = self._ensure_oracle()
                diagnose = oracle.diagnostics
                if self.ignore_module_resolution:
                    diagnose = drop_codes(diagnose, MODULE_RESOLUTION_CODES)

                artifact = f"{prompt}{completion}"
                try:
                    diags = diagnose(artifact)
                except Exception:
                    self._n_oracle_errors += 1
                    if self.on_error == "skip":
                        return None
                    raise

        self._diag_total += len(diags)
        if not diags and not hacked and not degenerate:
            self._n_clean += 1

        r = diagnostics_to_reward(diags, hacked=hacked, degenerate=degenerate,
                                  **self.reward_kwargs)
        self._reward_total += r
        return r

    def telemetry(self) -> dict:
        """A JSON-serializable dict: `n_samples`, `n_clean`, `n_hacked`,
        `n_degenerate`, `n_oracle_errors`, `hatch_counts` (name -> count),
        `degenerate_reasons` (reason -> count), `mean_diagnostics`,
        `mean_reward`, plus the oracle's own `n_calls`/`wall_s`/`ts_stats`/
        `opengrep_stats` (each `None` if no oracle has been constructed yet)."""
        n = self._n_samples
        oracle = self._oracle
        return {
            "n_samples": n,
            "n_clean": self._n_clean,
            "n_hacked": self._n_hacked,
            "n_degenerate": self._n_degenerate,
            "n_oracle_errors": self._n_oracle_errors,
            "hatch_counts": dict(self._hatch_counts),
            "degenerate_reasons": dict(self._degenerate_reasons),
            "mean_diagnostics": (self._diag_total / n) if n else 0.0,
            "mean_reward": (self._reward_total / n) if n else 0.0,
            "n_calls": oracle.n_calls if oracle is not None else 0,
            "wall_s": oracle.wall_s if oracle is not None else 0.0,
            "ts_stats": oracle.ts_stats if oracle is not None else None,
            "opengrep_stats": oracle.opengrep_stats if oracle is not None else None,
        }

    def close(self) -> None:
        """Closes the held oracle (self-constructed or injected) exactly once
        -- a second `close()`/`__exit__` is a no-op, and `telemetry()` stays
        readable afterward (the oracle reference itself isn't dropped)."""
        if self._closed:
            return
        self._closed = True
        if self._oracle is not None:
            self._oracle.close()

    def __enter__(self) -> "LspVerifier":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# #339 -- Tool Schema Validation & When2Call Abstention Verifiers
# --------------------------------------------------------------------------- #

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOLS_OPEN = "<tools>"
TOOLS_CLOSE = "</tools>"


def extract_tools_from_text(text: str) -> Optional[List[dict]]:
    """Extract declared tool schemas from a string containing `<tools>[...]</tools>`."""
    if not text or TOOLS_OPEN not in text or TOOLS_CLOSE not in text:
        return None
    start = text.find(TOOLS_OPEN)
    end = text.find(TOOLS_CLOSE, start)
    if start == -1 or end == -1 or end <= start:
        return None
    payload = text[start + len(TOOLS_OPEN):end].strip()
    try:
        data = json.loads(payload)
        if isinstance(data, list) and all(isinstance(t, dict) for t in data):
            return data
    except (json.JSONDecodeError, ValueError):
        return None
    return None


def check_json_schema_type(val: Any, expected_type: str | Sequence[str]) -> bool:
    """Validate that `val` conforms to the JSON Schema `expected_type`."""
    if isinstance(expected_type, (list, tuple)):
        return any(check_json_schema_type(val, t) for t in expected_type)
    if expected_type == "string":
        return isinstance(val, str)
    if expected_type == "integer":
        return isinstance(val, int) and not isinstance(val, bool)
    if expected_type == "number":
        return isinstance(val, (int, float)) and not isinstance(val, bool)
    if expected_type == "boolean":
        return isinstance(val, bool)
    if expected_type == "array":
        return isinstance(val, list)
    if expected_type == "object":
        return isinstance(val, dict)
    if expected_type == "null":
        return val is None
    return True


def parse_tool_calls_with_diagnostics(text: str) -> Tuple[List[dict], List[str]]:
    """Extract `<tool_call>{json}</tool_call>` blocks with explicit error diagnostics.

    Returns `(parsed_calls, errors)`:
      - `parsed_calls`: list of valid call dicts `{"name": str, "arguments": dict}`
      - `errors`: list of syntax/formatting error descriptions (e.g. unclosed tags,
        malformed JSON, non-object arguments).
    """
    calls: List[dict] = []
    errors: List[str] = []
    if not text:
        return calls, errors

    pos = 0
    while True:
        s = text.find(TOOL_CALL_OPEN, pos)
        if s == -1:
            if TOOL_CALL_CLOSE in text[pos:]:
                errors.append("unmatched_tool_call_close_tag")
            break
        e = text.find(TOOL_CALL_CLOSE, s)
        if e == -1:
            errors.append("unclosed_tool_call_tag")
            break
        block = text[s + len(TOOL_CALL_OPEN):e].strip()
        pos = e + len(TOOL_CALL_CLOSE)

        if not block:
            errors.append("empty_tool_call_block")
            continue

        try:
            call = json.loads(block)
        except (json.JSONDecodeError, ValueError) as exc:
            errors.append(f"json_syntax_error: {exc}")
            continue

        if not isinstance(call, dict):
            errors.append(f"call_payload_not_dict: got {type(call).__name__}")
            continue

        if "name" not in call or not isinstance(call["name"], str) or not call["name"].strip():
            errors.append("missing_or_invalid_tool_name")
            continue

        args = call.get("arguments")
        if args is not None and not isinstance(args, dict):
            errors.append(f"arguments_not_dict: got {type(args).__name__}")
            continue

        calls.append({
            "name": call["name"].strip(),
            "arguments": args if isinstance(args, dict) else {},
        })

    return calls, errors


class When2CallAbstentionVerifier:
    """Verifier for restraint / abstention queries answerable directly without tools (#339).

    For queries where no external tool is needed (or where available tools are distractors),
    rewards direct answers (+1.0) and penalizes redundant/spurious tool invocations (-1.0).
    Empty, degenerate, or whitespace-only answers are also penalized (-1.0).
    """

    def __init__(
        self,
        *,
        direct_answer_reward: float = 1.0,
        spurious_call_reward: float = -1.0,
        degenerate_reward: float = -1.0,
        min_answer_chars: int = 2,
    ) -> None:
        self.direct_answer_reward = direct_answer_reward
        self.spurious_call_reward = spurious_call_reward
        self.degenerate_reward = degenerate_reward
        self.min_answer_chars = min_answer_chars

        self._n_samples = 0
        self._n_abstain_success = 0
        self._n_spurious_calls = 0
        self._n_degenerate = 0
        self._reward_total = 0.0

    def reward(
        self,
        completion: str,
        reference: Optional[str] = None,
        *,
        prompt: str = "",
        **kwargs: Any,
    ) -> float:
        """Score one completion on abstention discipline.

        Returns:
          - `spurious_call_reward` (-1.0) if any tool call or tool tag was emitted
          - `degenerate_reward` (-1.0) if output is empty/whitespace/too short
          - `direct_answer_reward` (+1.0) if a non-tool direct answer was produced
        """
        self._n_samples += 1
        completion_str = "" if completion is None else str(completion)

        if TOOL_CALL_OPEN in completion_str or TOOL_CALL_CLOSE in completion_str:
            self._n_spurious_calls += 1
            self._reward_total += self.spurious_call_reward
            return self.spurious_call_reward

        trimmed = completion_str.strip()
        if not trimmed or len(trimmed) < self.min_answer_chars:
            self._n_degenerate += 1
            self._reward_total += self.degenerate_reward
            return self.degenerate_reward

        self._n_abstain_success += 1
        self._reward_total += self.direct_answer_reward
        return self.direct_answer_reward

    def telemetry(self) -> dict:
        n = self._n_samples
        return {
            "n_samples": n,
            "n_abstain_success": self._n_abstain_success,
            "n_spurious_calls": self._n_spurious_calls,
            "n_degenerate": self._n_degenerate,
            "mean_reward": (self._reward_total / n) if n else 0.0,
        }

    def close(self) -> None:
        pass

    def __enter__(self) -> "When2CallAbstentionVerifier":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


class ToolSchemaVerifier:
    """Deterministic tool-use verifier validating calls against declared JSON Schemas (#339).

    Validates:
      1. Valid JSON syntax in `<tool_call>` blocks.
      2. Known tool name in active toolset (penalizes hallucinations).
      3. All required arguments present.
      4. Parameter types conform to schema definitions.
      5. Handles When2Call abstention queries when direct answer is expected.
    """

    def __init__(
        self,
        tools: Optional[Sequence[dict]] = None,
        *,
        base_reward: float = 1.0,
        syntax_error_reward: float = -1.0,
        hallucination_reward: float = -1.0,
        schema_error_reward: float = -1.0,
        no_call_reward: float = -1.0,
        abstention_reward: float = 1.0,
        spurious_call_reward: float = -1.0,
        partial_credit: bool = False,
    ) -> None:
        self.tools = list(tools) if tools is not None else None
        self.base_reward = base_reward
        self.syntax_error_reward = syntax_error_reward
        self.hallucination_reward = hallucination_reward
        self.schema_error_reward = schema_error_reward
        self.no_call_reward = no_call_reward
        self.abstention_reward = abstention_reward
        self.spurious_call_reward = spurious_call_reward
        self.partial_credit = partial_credit

        self._abstention_verifier = When2CallAbstentionVerifier(
            direct_answer_reward=abstention_reward,
            spurious_call_reward=spurious_call_reward,
            degenerate_reward=syntax_error_reward,
        )

        self._n_samples = 0
        self._n_valid = 0
        self._n_syntax_errors = 0
        self._n_hallucinations = 0
        self._n_missing_args = 0
        self._n_type_errors = 0
        self._n_no_calls = 0
        self._n_abstentions_correct = 0
        self._n_spurious_calls = 0
        self._reward_total = 0.0

    def _resolve_tools(
        self,
        tools_arg: Optional[Sequence[dict]],
        prompt: str,
    ) -> List[dict]:
        if tools_arg is not None:
            return list(tools_arg)
        if self.tools is not None:
            return self.tools
        from_prompt = extract_tools_from_text(prompt)
        if from_prompt is not None:
            return from_prompt
        return []

    def validate_call(
        self,
        call: dict,
        active_tools: Sequence[dict],
    ) -> Tuple[bool, List[str], str]:
        """Validate a single parsed tool call against active tool definitions.

        Returns `(is_valid, error_list, primary_error_category)`:
          - primary_error_category in {"clean", "hallucination", "missing_args", "type_error"}
        """
        name = call.get("name")
        by_name = {t.get("name"): t for t in active_tools if isinstance(t, dict)}
        tool = by_name.get(name)
        if tool is None:
            return False, [f"unknown tool {name!r}"], "hallucination"

        errors: List[str] = []
        category = "clean"

        params = tool.get("parameters") or {}
        required = params.get("required") or []
        arguments = call.get("arguments") or {}

        # 1. Required arguments presence
        for req in required:
            if req not in arguments:
                errors.append(f"missing required argument {req!r} for tool {name!r}")
                if category == "clean":
                    category = "missing_args"

        # 2. Parameter type validation
        properties = params.get("properties") or {}
        for arg_name, arg_val in arguments.items():
            if arg_name in properties:
                prop_schema = properties[arg_name]
                expected_type = prop_schema.get("type")
                if expected_type and not check_json_schema_type(arg_val, expected_type):
                    errors.append(
                        f"argument {arg_name!r}={arg_val!r} failed type check (expected {expected_type!r})"
                    )
                    if category == "clean":
                        category = "type_error"
                if expected_type == "array" and isinstance(arg_val, list):
                    items_schema = prop_schema.get("items") or {}
                    item_type = items_schema.get("type")
                    if item_type:
                        for idx, item in enumerate(arg_val):
                            if not check_json_schema_type(item, item_type):
                                errors.append(
                                    f"item {idx} in {arg_name!r}={item!r} failed type check (expected {item_type!r})"
                                )
                                if category == "clean":
                                    category = "type_error"

        return len(errors) == 0, errors, category

    def reward(
        self,
        completion: str,
        reference: Optional[str] = None,
        *,
        prompt: str = "",
        tools: Optional[Sequence[dict]] = None,
        abstain: Optional[bool] = None,
        category: Optional[str] = None,
        **kwargs: Any,
    ) -> float:
        self._n_samples += 1
        completion_str = "" if completion is None else str(completion)
        active_tools = self._resolve_tools(tools, prompt)

        is_abstain = False
        if abstain is True or category in ("abstention", "relevance"):
            is_abstain = True
        elif abstain is None and reference is not None:
            ref_str = str(reference).strip()
            if ref_str.lower() == "abstain":
                is_abstain = True
            elif ref_str and (TOOL_CALL_OPEN not in ref_str) and not ref_str.startswith("{"):
                is_abstain = True

        if is_abstain:
            r = self._abstention_verifier.reward(completion_str, reference=reference, prompt=prompt)
            if r == self.abstention_reward:
                self._n_abstentions_correct += 1
            else:
                self._n_spurious_calls += 1
            self._reward_total += r
            return r

        calls, syntax_errors = parse_tool_calls_with_diagnostics(completion_str)

        if syntax_errors:
            self._n_syntax_errors += 1
            self._reward_total += self.syntax_error_reward
            return self.syntax_error_reward

        if not calls:
            self._n_no_calls += 1
            self._reward_total += self.no_call_reward
            return self.no_call_reward

        all_valid = True
        primary_issue = None
        n_missing = 0
        n_type = 0

        for call in calls:
            ok, call_errors, cat = self.validate_call(call, active_tools)
            if not ok:
                all_valid = False
                if cat == "hallucination":
                    self._n_hallucinations += 1
                    primary_issue = "hallucination"
                elif cat == "missing_args":
                    self._n_missing_args += 1
                    n_missing += len(call_errors)
                    if primary_issue != "hallucination":
                        primary_issue = "missing_args"
                elif cat == "type_error":
                    self._n_type_errors += 1
                    n_type += len(call_errors)
                    if primary_issue not in ("hallucination", "missing_args"):
                        primary_issue = "type_error"

        if all_valid:
            self._n_valid += 1
            self._reward_total += self.base_reward
            return self.base_reward

        if primary_issue == "hallucination":
            r = self.hallucination_reward
        elif self.partial_credit:
            penalty = 0.25 * n_missing + 0.25 * n_type
            r = max(self.base_reward - penalty, self.schema_error_reward)
        else:
            r = self.schema_error_reward

        self._reward_total += r
        return r

    def telemetry(self) -> dict:
        n = self._n_samples
        return {
            "n_samples": n,
            "n_valid": self._n_valid,
            "n_syntax_errors": self._n_syntax_errors,
            "n_hallucinations": self._n_hallucinations,
            "n_missing_args": self._n_missing_args,
            "n_type_errors": self._n_type_errors,
            "n_no_calls": self._n_no_calls,
            "n_abstentions_correct": self._n_abstentions_correct,
            "n_spurious_calls": self._n_spurious_calls,
            "mean_reward": (self._reward_total / n) if n else 0.0,
        }

    def close(self) -> None:
        pass

    def __enter__(self) -> "ToolSchemaVerifier":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# #340 -- SymPy symbolic equivalence & Z3 constraint solver verifiers
# --------------------------------------------------------------------------- #

def _extract_boxed_content(text: str) -> Optional[str]:
    """Extract contents of the last \boxed{...} tag, handling nested braces."""
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return text[start:].strip()


def extract_math_expression(s: str) -> str:
    """Extract a target mathematical expression from candidate completion or reference.

    Prefers:
      1. Last \boxed{...} content
      2. Text after #### marker (GSM8K format)
      3. Explicit answer prefixes (e.g. 'Final Answer:', 'The answer is')
      4. Last non-empty line
    Preprocesses LaTeX fractions, square roots, multiplications, and strips formatting.
    """
    if not s:
        return ""
    s = s.strip()
    boxed = _extract_boxed_content(s)
    if boxed is not None:
        expr = boxed
    elif "####" in s:
        expr = s.split("####")[-1].strip()
    else:
        found_prefix = False
        for prefix in (
            "final answer:",
            "final answer is:",
            "the answer is:",
            "the answer is",
            "answer is:",
            "answer:",
        ):
            idx = s.lower().rfind(prefix)
            if idx != -1:
                expr = s[idx + len(prefix):].strip()
                found_prefix = True
                break
        if not found_prefix:
            lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
            expr = lines[-1] if lines else s

    expr = expr.rstrip(".!;")
    expr = expr.replace("$", "").strip()

    def frac_repl(match: re.Match) -> str:
        return f"(({match.group(1)})/({match.group(2)}))"

    while re.search(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", expr):
        expr = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", frac_repl, expr)

    expr = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", expr)
    expr = re.sub(r"\\sqrt\[([^{}]+)\]\{([^{}]+)\}", r"((\2)**(1/(\1)))", expr)
    expr = expr.replace(r"\cdot", "*").replace(r"\times", "*").replace(r"\div", "/")
    expr = re.sub(r"(?<=\d),(?=\d)", "", expr)
    return expr.strip()


def sympy_symbolic_reward(
    candidate: str,
    reference: str,
    *,
    tol: float = 1e-6,
    base_reward: float = 1.0,
    failure_reward: float = 0.0,
) -> float:
    """Evaluate symbolic equivalence between candidate and reference expressions with SymPy (#340).

    Handles equivalent fractions, radicals, algebraic polynomials, and equations:
    - simplify(candidate - reference) == 0
    - candidate.equals(reference)
    - proportional linear/algebraic equations
    """
    if candidate is None or reference is None:
        return failure_reward

    c_raw, r_raw = str(candidate).strip(), str(reference).strip()
    if not c_raw or not r_raw:
        return failure_reward

    if normalize_text(c_raw) == normalize_text(r_raw):
        return base_reward

    c_num, r_num = extract_final_number(c_raw), extract_final_number(r_raw)
    if c_num is not None and r_num is not None and abs(c_num - r_num) <= tol:
        return base_reward

    c_clean = extract_math_expression(c_raw)
    r_clean = extract_math_expression(r_raw)

    if not c_clean or not r_clean:
        return failure_reward

    if normalize_text(c_clean) == normalize_text(r_clean):
        return base_reward

    try:
        import sympy
        from sympy.parsing.sympy_parser import (
            convert_xor,
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )
    except ImportError:
        return base_reward if (c_num is not None and r_num is not None and abs(c_num - r_num) <= tol) else failure_reward

    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )

    try:
        if "=" in c_clean and "=" in r_clean:
            cl, cr = c_clean.split("=", 1)
            rl, rr = r_clean.split("=", 1)
            cle = parse_expr(cl, transformations=transformations)
            cre = parse_expr(cr, transformations=transformations)
            rle = parse_expr(rl, transformations=transformations)
            rre = parse_expr(rr, transformations=transformations)
            diff_c = cle - cre
            diff_r = rle - rre
            if sympy.simplify(diff_c - diff_r) == 0:
                return base_reward
            ratio = sympy.simplify(diff_c / diff_r)
            if ratio.is_number and ratio != 0:
                return base_reward

        if "=" in c_clean and "=" not in r_clean:
            _, cr = c_clean.split("=", 1)
            cre = parse_expr(cr, transformations=transformations)
            refe = parse_expr(r_clean, transformations=transformations)
            if sympy.simplify(cre - refe) == 0 or cre.equals(refe):
                return base_reward

        if "=" not in c_clean and "=" in r_clean:
            _, rr = r_clean.split("=", 1)
            cande = parse_expr(c_clean, transformations=transformations)
            rre = parse_expr(rr, transformations=transformations)
            if sympy.simplify(cande - rre) == 0 or cande.equals(rre):
                return base_reward

        c_expr = parse_expr(c_clean, transformations=transformations)
        r_expr = parse_expr(r_clean, transformations=transformations)

        diff = sympy.simplify(c_expr - r_expr)
        if diff == 0:
            return base_reward

        if c_expr.equals(r_expr) or sympy.expand(c_expr) == sympy.expand(r_expr):
            return base_reward

        if not c_expr.free_symbols and not r_expr.free_symbols:
            try:
                c_val = float(sympy.N(c_expr))
                r_val = float(sympy.N(r_expr))
                if abs(c_val - r_val) <= tol:
                    return base_reward
            except Exception:
                pass

        return failure_reward
    except Exception:
        c_num_clean = extract_final_number(c_clean)
        r_num_clean = extract_final_number(r_clean)
        if c_num_clean is not None and r_num_clean is not None and abs(c_num_clean - r_num_clean) <= tol:
            return base_reward
        return failure_reward


class SympyVerifier:
    """In-memory symbolic equivalence verifier using SymPy (#340).

    Replaces/enhances `math_reward` by parsing candidate and reference expressions
    into SymPy trees and verifying `simplify(candidate - reference) == 0`.
    Handles equivalent fractions (e.g. 1/2 vs 2/4), radicals (e.g. sqrt(8) vs 2*sqrt(2)),
    polynomial expansions, and algebraic equation forms. Completely sandbox-free.
    """

    def __init__(
        self,
        *,
        base_reward: float = 1.0,
        failure_reward: float = 0.0,
        syntax_error_reward: float = 0.0,
        tol: float = 1e-6,
    ) -> None:
        self.base_reward = base_reward
        self.failure_reward = failure_reward
        self.syntax_error_reward = syntax_error_reward
        self.tol = tol

        self._n_samples = 0
        self._n_equivalent = 0
        self._n_inequivalent = 0
        self._n_syntax_errors = 0
        self._reward_total = 0.0

    def reward(
        self,
        completion: str,
        reference: Optional[str] = None,
        *,
        prompt: str = "",
        **kwargs: Any,
    ) -> float:
        self._n_samples += 1
        if reference is None:
            self._n_syntax_errors += 1
            self._reward_total += self.syntax_error_reward
            return self.syntax_error_reward

        ref_str = str(reference)
        comp_str = "" if completion is None else str(completion)

        r = sympy_symbolic_reward(
            comp_str,
            ref_str,
            tol=self.tol,
            base_reward=self.base_reward,
            failure_reward=self.failure_reward,
        )

        if r == self.base_reward:
            self._n_equivalent += 1
        elif r == self.failure_reward:
            self._n_inequivalent += 1
        else:
            self._n_syntax_errors += 1

        self._reward_total += r
        return r

    def telemetry(self) -> dict:
        n = self._n_samples
        return {
            "n_samples": n,
            "n_equivalent": self._n_equivalent,
            "n_inequivalent": self._n_inequivalent,
            "n_syntax_errors": self._n_syntax_errors,
            "mean_reward": (self._reward_total / n) if n else 0.0,
        }

    def close(self) -> None:
        pass

    def __enter__(self) -> "SympyVerifier":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def parse_candidate_assignments(text: str) -> Optional[dict[str, Any]]:
    """Extract candidate variable assignments from completion text for Z3 verification.

    Supports:
      1. Tagged blocks: `<solution>{...}</solution>`, `<assignment>...</assignment>`
      2. Markdown JSON code blocks: ```json {...} ```
      3. Raw JSON object: `{...}`
      4. SMT2 define-fun models: `(define-fun x () Int 3)`
      5. Key-value lines: `x = 3`, `y = 7`, `P = True`
    """
    if not text:
        return None
    text = text.strip()

    for tag in ("solution", "assignment", "model", "answer"):
        m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
            break

    m_code = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m_code:
        try:
            d = json.loads(m_code.group(1))
            if isinstance(d, dict):
                return d
        except Exception:
            pass

    m_json = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m_json:
        try:
            d = json.loads(m_json.group(0))
            if isinstance(d, dict):
                return d
        except Exception:
            pass

    smt_matches = re.findall(
        r"\(define-fun\s+([a-zA-Z0-9_]+)\s*\(\)\s*(?:Int|Bool|Real|String)\s+([^)]+)\)",
        text,
    )
    if smt_matches:
        d = {}
        for var, val in smt_matches:
            val = val.strip()
            if val.lower() == "true":
                d[var] = True
            elif val.lower() == "false":
                d[var] = False
            else:
                try:
                    d[var] = int(val)
                except ValueError:
                    try:
                        d[var] = float(val)
                    except ValueError:
                        d[var] = val
        return d

    kv_matches = re.findall(
        r'\b([a-zA-Z_]\w*)\s*(?:=|:)\s*(-?\d+(?:\.\d+)?|true|false|"[^"]*"|\'[^\']*\')',
        text,
        re.IGNORECASE,
    )
    if kv_matches:
        d = {}
        for var, val in kv_matches:
            val = val.strip("\"'")
            if val.lower() == "true":
                d[var] = True
            elif val.lower() == "false":
                d[var] = False
            else:
                try:
                    d[var] = int(val)
                except ValueError:
                    try:
                        d[var] = float(val)
                    except ValueError:
                        d[var] = val
        return d

    return None


def _get_z3_ast_vars(expr: Any) -> set[str]:
    """Extract all free uninterpreted variable names from a Z3 AST expression."""
    import z3

    vars_found: set[str] = set()

    def rec(e: Any) -> None:
        if z3.is_const(e) and e.decl().kind() == z3.Z3_OP_UNINTERPRETED:
            vars_found.add(str(e.decl().name()))
        for ch in e.children():
            rec(ch)

    rec(expr)
    return vars_found


def parse_py_constraint(
    expr_str: str,
    vars_dict: dict[str, Any],
    var_types: Optional[Mapping[str, str]] = None,
) -> Any:
    """Safely parse a Python constraint expression into a Z3 formula without eval."""
    import z3

    var_types = var_types or {}

    def get_var(name: str, bool_context: bool = False) -> Any:
        if name in vars_dict:
            return vars_dict[name]
        vtype = var_types.get(name)
        if vtype == "bool" or (vtype is None and bool_context):
            v = z3.Bool(name)
        elif vtype == "real":
            v = z3.Real(name)
        elif vtype in ("string", "str"):
            v = z3.String(name)
        else:
            v = z3.Int(name)
        vars_dict[name] = v
        return v

    def visit(node: ast.AST, bool_context: bool = False) -> Any:
        if isinstance(node, ast.Expression):
            return visit(node.body, bool_context)
        elif isinstance(node, ast.Name):
            return get_var(node.id, bool_context)
        elif isinstance(node, ast.Constant):
            val = node.value
            if isinstance(val, bool):
                return z3.BoolVal(val)
            elif isinstance(val, int):
                return z3.IntVal(val)
            elif isinstance(val, float):
                return z3.RealVal(val)
            elif isinstance(val, str):
                return z3.StringVal(val)
            return val
        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return z3.Not(visit(node.operand, bool_context=True))
            elif isinstance(node.op, ast.USub):
                return -visit(node.operand)
            elif isinstance(node.op, ast.UAdd):
                return visit(node.operand)
        elif isinstance(node, ast.BinOp):
            l = visit(node.left)
            r = visit(node.right)
            if isinstance(node.op, ast.Add):
                return l + r
            elif isinstance(node.op, ast.Sub):
                return l - r
            elif isinstance(node.op, ast.Mult):
                return l * r
            elif isinstance(node.op, ast.Div):
                return l / r
            elif isinstance(node.op, ast.Mod):
                return l % r
        elif isinstance(node, ast.BoolOp):
            vals = [visit(v, bool_context=True) for v in node.values]
            if isinstance(node.op, ast.And):
                return z3.And(*vals)
            elif isinstance(node.op, ast.Or):
                return z3.Or(*vals)
        elif isinstance(node, ast.Compare):
            l = visit(node.left)
            conj = []
            curr = l
            for op, comp in zip(node.ops, node.comparators):
                r = visit(comp)
                if isinstance(op, ast.Eq):
                    conj.append(curr == r)
                elif isinstance(op, ast.NotEq):
                    conj.append(curr != r)
                elif isinstance(op, ast.Lt):
                    conj.append(curr < r)
                elif isinstance(op, ast.LtE):
                    conj.append(curr <= r)
                elif isinstance(op, ast.Gt):
                    conj.append(curr > r)
                elif isinstance(op, ast.GtE):
                    conj.append(curr >= r)
                curr = r
            return z3.And(*conj) if len(conj) > 1 else conj[0]
        elif isinstance(node, ast.Call):
            fname = node.func.id if isinstance(node.func, ast.Name) else ""
            args = [visit(a) for a in node.args]
            if fname == "Distinct":
                return z3.Distinct(*args)
            elif fname == "And":
                return z3.And(*args)
            elif fname == "Or":
                return z3.Or(*args)
            elif fname == "Not":
                return z3.Not(*args)
            elif fname == "Implies":
                return z3.Implies(*args)
            elif fname == "Xor":
                return z3.Xor(*args)
            elif fname == "If":
                return z3.If(*args)
            elif fname == "Abs":
                arg = args[0]
                return z3.If(arg >= 0, arg, -arg)
        raise ValueError(f"Unsupported AST node in constraint: {ast.dump(node)}")

    tree = ast.parse(expr_str.strip(), mode="eval")
    return visit(tree)


class Z3Verifier:
    """SMT constraint solver verifier for formal logic, scheduling, and SAT puzzles (#340).

    Verifies candidate variable assignments satisfy problem constraints in <5ms.
    Completely sandbox-free and executes in-memory with Z3.
    """

    def __init__(
        self,
        *,
        constraints: Optional[Sequence[str] | str] = None,
        var_types: Optional[Mapping[str, str]] = None,
        timeout_ms: int = 50,
        base_reward: float = 1.0,
        failure_reward: float = -1.0,
        syntax_error_reward: float = -1.0,
        require_complete_assignment: bool = True,
        partial_credit: bool = False,
    ) -> None:
        self.constraints = constraints
        self.var_types = dict(var_types) if var_types else {}
        self.timeout_ms = timeout_ms
        self.base_reward = base_reward
        self.failure_reward = failure_reward
        self.syntax_error_reward = syntax_error_reward
        self.require_complete_assignment = require_complete_assignment
        self.partial_credit = partial_credit

        self._n_samples = 0
        self._n_satisfied = 0
        self._n_unsat = 0
        self._n_syntax_errors = 0
        self._n_missing_vars = 0
        self._n_timeouts = 0
        self._solve_time_total_ms = 0.0
        self._reward_total = 0.0

    def _resolve_constraints(
        self,
        constraints_arg: Optional[Sequence[str] | str],
        reference: Optional[str],
        prompt: str,
    ) -> Tuple[Optional[List[str] | str], Optional[dict]]:
        """Resolve constraints and optional gold assignment from arguments, reference, or prompt."""
        if constraints_arg is not None:
            return constraints_arg, None
        if self.constraints is not None:
            return self.constraints, None

        if "<constraints>" in prompt and "</constraints>" in prompt:
            s = prompt.find("<constraints>") + len("<constraints>")
            e = prompt.find("</constraints>", s)
            block = prompt[s:e].strip()
            try:
                parsed = json.loads(block)
                if isinstance(parsed, (list, str)):
                    return parsed, None
                if isinstance(parsed, dict) and "constraints" in parsed:
                    return parsed["constraints"], parsed.get("solution")
            except Exception:
                lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
                return lines if lines else block, None

        if reference is not None:
            ref_str = str(reference).strip()
            try:
                data = json.loads(ref_str)
                if isinstance(data, list):
                    return data, None
                if isinstance(data, dict):
                    if "constraints" in data:
                        return data["constraints"], data.get("solution") or data.get("assignment")
                    return None, data
            except Exception:
                pass

            if "(assert" in ref_str or "(declare-const" in ref_str:
                return ref_str, None

            lines = [ln.strip() for ln in ref_str.splitlines() if ln.strip()]
            if len(lines) > 1 and any("==" in ln or "<" in ln or ">" in ln for ln in lines):
                return lines, None
            if len(lines) == 1 and any(op in lines[0] for op in ("==", "<", ">", "!=")):
                return lines, None

        return None, None

    def reward(
        self,
        completion: str,
        reference: Optional[str] = None,
        *,
        prompt: str = "",
        constraints: Optional[Sequence[str] | str] = None,
        var_types: Optional[Mapping[str, str]] = None,
        **kwargs: Any,
    ) -> float:
        import time

        try:
            import z3
        except ImportError:
            raise RuntimeError("Z3Verifier requires z3-solver. Install with 'pip install z3-solver'.")

        self._n_samples += 1
        comp_str = "" if completion is None else str(completion)
        candidate = parse_candidate_assignments(comp_str)

        if candidate is None:
            self._n_syntax_errors += 1
            self._reward_total += self.syntax_error_reward
            return self.syntax_error_reward

        resolved_constraints, gold_assignment = self._resolve_constraints(
            constraints, reference, prompt
        )

        if resolved_constraints is None and gold_assignment is not None:
            if candidate == gold_assignment:
                self._n_satisfied += 1
                self._reward_total += self.base_reward
                return self.base_reward
            else:
                self._n_unsat += 1
                self._reward_total += self.failure_reward
                return self.failure_reward

        if resolved_constraints is None:
            self._n_syntax_errors += 1
            self._reward_total += self.syntax_error_reward
            return self.syntax_error_reward

        solver = z3.Solver()
        solver.set("timeout", self.timeout_ms)

        merged_types = dict(self.var_types)
        if var_types:
            merged_types.update(var_types)
        for k, v in candidate.items():
            if k not in merged_types:
                if isinstance(v, bool):
                    merged_types[k] = "bool"
                elif isinstance(v, int):
                    merged_types[k] = "int"
                elif isinstance(v, float):
                    merged_types[k] = "real"
                elif isinstance(v, str):
                    merged_types[k] = "string"

        vars_dict: dict[str, Any] = {}
        required_vars: set[str] = set()
        c_list: List[Any] = []

        try:
            if isinstance(resolved_constraints, str) and (
                "(assert" in resolved_constraints or "(declare-const" in resolved_constraints
            ):
                assertions = z3.parse_smt2_string(resolved_constraints)
                for a in assertions:
                    c_list.append(a)
                    solver.add(a)
                    required_vars.update(_get_z3_ast_vars(a))
            else:
                raw_constraints = (
                    [resolved_constraints]
                    if isinstance(resolved_constraints, str)
                    else list(resolved_constraints)
                )
                for c_str in raw_constraints:
                    ast_expr = parse_py_constraint(c_str, vars_dict, merged_types)
                    c_list.append(ast_expr)
                    solver.add(ast_expr)
                    required_vars.update(_get_z3_ast_vars(ast_expr))
        except Exception:
            self._n_syntax_errors += 1
            self._reward_total += self.syntax_error_reward
            return self.syntax_error_reward

        if self.require_complete_assignment:
            missing = required_vars - set(candidate.keys())
            if missing:
                self._n_missing_vars += 1
                self._reward_total += self.failure_reward
                return self.failure_reward

        t0 = time.perf_counter()
        solver.push()
        for var_name, val in candidate.items():
            if var_name in vars_dict:
                v_ast = vars_dict[var_name]
            else:
                vtype = merged_types.get(var_name, "int")
                if vtype == "bool":
                    v_ast = z3.Bool(var_name)
                elif vtype == "real":
                    v_ast = z3.Real(var_name)
                elif vtype in ("string", "str"):
                    v_ast = z3.String(var_name)
                else:
                    v_ast = z3.Int(var_name)
                vars_dict[var_name] = v_ast

            if isinstance(val, bool):
                solver.add(v_ast == z3.BoolVal(val))
            elif isinstance(val, int):
                solver.add(v_ast == z3.IntVal(val))
            elif isinstance(val, float):
                solver.add(v_ast == z3.RealVal(val))
            elif isinstance(val, str):
                solver.add(v_ast == z3.StringVal(val))

        status = solver.check()
        solver.pop()
        solve_ms = (time.perf_counter() - t0) * 1000.0
        self._solve_time_total_ms += solve_ms

        if status == z3.sat:
            self._n_satisfied += 1
            self._reward_total += self.base_reward
            return self.base_reward
        elif status == z3.unsat:
            self._n_unsat += 1
            if self.partial_credit and c_list:
                passed = 0
                for c_ast in c_list:
                    s_sub = z3.Solver()
                    s_sub.set("timeout", max(self.timeout_ms // len(c_list), 5))
                    s_sub.add(c_ast)
                    for var_name, val in candidate.items():
                        v_ast = vars_dict[var_name]
                        if isinstance(val, bool):
                            s_sub.add(v_ast == z3.BoolVal(val))
                        elif isinstance(val, int):
                            s_sub.add(v_ast == z3.IntVal(val))
                        elif isinstance(val, float):
                            s_sub.add(v_ast == z3.RealVal(val))
                        elif isinstance(val, str):
                            s_sub.add(v_ast == z3.StringVal(val))
                    if s_sub.check() == z3.sat:
                        passed += 1
                frac = passed / len(c_list)
                r = self.failure_reward + frac * (self.base_reward - self.failure_reward)
                self._reward_total += r
                return r
            else:
                self._reward_total += self.failure_reward
                return self.failure_reward
        else:
            self._n_timeouts += 1
            self._reward_total += self.failure_reward
            return self.failure_reward

    def telemetry(self) -> dict:
        n = self._n_samples
        return {
            "n_samples": n,
            "n_satisfied": self._n_satisfied,
            "n_unsat": self._n_unsat,
            "n_syntax_errors": self._n_syntax_errors,
            "n_missing_vars": self._n_missing_vars,
            "n_timeouts": self._n_timeouts,
            "mean_solve_time_ms": (self._solve_time_total_ms / n) if n else 0.0,
            "mean_reward": (self._reward_total / n) if n else 0.0,
        }

    def close(self) -> None:
        pass

    def __enter__(self) -> "Z3Verifier":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


# Concurrency & Memoization (#341-#344 perf improvements)
# --------------------------------------------------------------------------- #

class MemoizedVerifier:
    """Thread-safe memoization cache wrapping any verifier reward function or callable.

    Caches `(prompt, completion, reference) -> reward` in an in-memory dictionary.
    Eliminates redundant oracle, parser, or compiler invocations on identical completions
    (common in multi-sample GRPO rollouts).
    """

    def __init__(self, target: Any, max_size: int = 65536):
        self.target = target
        self.max_size = max_size
        self._cache: Dict[Tuple[str, str, Optional[str]], Optional[float]] = {}
        self._inflight: Dict[Tuple[str, str, Optional[str]], threading.Event] = {}
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    def reward(self, completion: str, reference: Optional[str] = None, *,
               prompt: str = "", **kwargs: Any) -> Optional[float]:
        key = (prompt, str(completion), None if reference is None else str(reference))
        while True:
            with self._lock:
                if key in self._cache:
                    self._hits += 1
                    return self._cache[key]
                if key in self._inflight:
                    ev = self._inflight[key]
                else:
                    ev = threading.Event()
                    self._inflight[key] = ev
                    break
            ev.wait()

        r: Optional[float] = None
        try:
            if hasattr(self.target, "reward") and callable(self.target.reward):
                r = self.target.reward(completion, reference, prompt=prompt, **kwargs)
            elif callable(self.target):
                try:
                    r = self.target(completion, reference, prompt=prompt, **kwargs)
                except TypeError:
                    try:
                        r = self.target(completion, reference, prompt=prompt)
                    except TypeError:
                        r = self.target(completion, reference)
            else:
                raise TypeError(f"cannot score with target {type(self.target).__name__}")
        finally:
            with self._lock:
                self._misses += 1
                if r is not None and len(self._cache) < self.max_size:
                    self._cache[key] = r
                self._inflight.pop(key, None)
                ev.set()
        return r

    def telemetry(self) -> dict:
        base = {}
        if hasattr(self.target, "telemetry") and callable(self.target.telemetry):
            base = self.target.telemetry()
        total = self._hits + self._misses
        hit_rate = (self._hits / total) if total > 0 else 0.0
        return {
            **base,
            "cache_hits": self._hits,
            "cache_misses": self._misses,
            "cache_hit_rate": hit_rate,
            "cache_size": len(self._cache),
        }

    def close(self) -> None:
        if hasattr(self.target, "close") and callable(self.target.close):
            self.target.close()

    def __enter__(self) -> "MemoizedVerifier":
        if hasattr(self.target, "__enter__"):
            self.target.__enter__()
        return self

    def __exit__(self, *exc_info) -> None:
        if hasattr(self.target, "__exit__"):
            self.target.__exit__(*exc_info)
        else:
            self.close()


def score_rollouts(
    reward_fn: Callable[..., Optional[float]],
    completions: Sequence[str],
    reference: Optional[str] = None,
    *,
    max_workers: int = 0,
    executor: Optional[ThreadPoolExecutor] = None,
) -> List[Optional[float]]:
    """Score a sequence of completions either sequentially or concurrently.

    Preserves exact input order. If `executor` is given or `max_workers > 1`, scoring
    runs concurrently across threads, releasing the GIL during subprocess and I/O wait.
    """
    if not completions:
        return []

    if max_workers <= 1 and executor is None:
        return [reward_fn(c, reference) for c in completions]

    if executor is not None:
        futures = [executor.submit(reward_fn, c, reference) for c in completions]
        return [f.result() for f in futures]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(reward_fn, c, reference) for c in completions]
        return [f.result() for f in futures]



# --------------------------------------------------------------------------- #
# #341 -- Systems, native & mobile stack verifiers (Rust, C/C++, Swift, Kotlin)
# --------------------------------------------------------------------------- #
from src.train.verifiers.systems_mobile import (
    CppCompilerOracle,
    CppStaticVerifier,
    CppVerifier,
    KotlinCompilerOracle,
    KotlinStaticVerifier,
    KotlinVerifier,
    RustCompilerOracle,
    RustStaticVerifier,
    RustVerifier,
    SwiftCompilerOracle,
    SwiftStaticVerifier,
    SwiftVerifier,
    SystemsMobileVerifier,
    find_cpp_escape_hatches,
    find_kotlin_escape_hatches,
    find_rust_escape_hatches,
    find_swift_escape_hatches,
    parse_cpp_diagnostics,
    parse_kotlin_diagnostics,
    parse_rust_diagnostics,
    parse_swift_diagnostics,
    resolve_cpp_toolchain,
    resolve_kotlin_toolchain,
    resolve_rust_toolchain,
    resolve_swift_toolchain,
)
