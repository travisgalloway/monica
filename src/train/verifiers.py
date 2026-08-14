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

import re
import subprocess
import sys
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence

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


def math_reward(answer: str, gold: str, *, tol: float = 1e-6) -> float:
    """1.0 if the final number in `answer` equals the gold answer within `tol`, else 0.0.
    `gold` may be a bare number or a full solution string (its final number is used)."""
    a, g = extract_final_number(answer), extract_final_number(gold)
    if a is None or g is None:
        return 0.0
    return 1.0 if abs(a - g) <= tol else 0.0


class CodeVerifier:
    """Run candidate code against a test suite, rewarding the **fraction of tests passing**
    (partial credit; use >=5 tests per problem so a thin suite can't be gamed — the main
    RLVR failure mode).

    UNSAFE: executes untrusted model output in a subprocess. Disabled by default; set
    `enabled=True` to opt in (never in CI). Python only — TS/Rust/SQL belong in an external
    sandbox.
    """

    def __init__(self, *, timeout: float = 5.0, enabled: bool = False):
        self.timeout = timeout
        self.enabled = enabled

    def reward(self, code: str, tests: Sequence[str]) -> float:
        if not self.enabled:
            raise RuntimeError(
                "CodeVerifier is disabled (it executes untrusted code). "
                "Pass enabled=True to opt in — never in CI.")
        if not tests:
            return 0.0
        passed = 0
        for t in tests:
            program = f"{code}\n{t}\n"
            try:
                # Discard untrusted stdout/stderr (a candidate can print unbounded data);
                # only the exit code matters.
                r = subprocess.run([sys.executable, "-c", program],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   timeout=self.timeout)
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
                oracle=None, on_error: str = "raise", **reward_kwargs):
        if hatches not in _HATCH_MODES:
            raise ValueError(f"unknown hatches mode {hatches!r} (want one of {_HATCH_MODES})")
        if on_error not in ("raise", "skip"):
            raise ValueError(f"on_error must be 'raise' or 'skip', got {on_error!r}")
        self.kind = kind
        self.timeout_s = timeout_s
        self.ignore_module_resolution = ignore_module_resolution
        self.hatches = hatches
        self.on_error = on_error
        self.reward_kwargs = reward_kwargs

        self._oracle = oracle
        self._closed = False

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
