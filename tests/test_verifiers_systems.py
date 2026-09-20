"""#341 -- CI-safe unit tests for Systems, Native & Mobile verifiers:
Pure reward-shaping core, diagnostic parsing, anti-Goodhart escape-hatch guards,
and verifiers against a fake oracle (CI-safe with no toolchains installed).
Integration tests with real toolchains run when tools are present on the host.
"""

from __future__ import annotations

import pytest

from src.lsp.diagnostics import Diagnostic
from src.train.verifiers import (
    CppVerifier,
    KotlinVerifier,
    RustVerifier,
    SwiftVerifier,
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


# --------------------------------------------------------------------------- #
# Fake Oracle for Toolchain-Free Testing
# --------------------------------------------------------------------------- #

class FakeCompilerOracle:
    """Stands in for compiler oracles without calling external toolchains."""

    def __init__(self, diags=None, raises: bool = False):
        self._diags = [] if diags is None else diags
        self._raises = raises
        self.calls = []
        self.n_calls = 0
        self.wall_s = 0.0
        self.close_count = 0

    def diagnostics(self, source: str) -> list[Diagnostic]:
        self.calls.append(source)
        self.n_calls += 1
        self.wall_s += 0.01
        if self._raises:
            raise RuntimeError("simulated compiler failure")
        return list(self._diags(source)) if callable(self._diags) else list(self._diags)

    def close(self) -> None:
        self.close_count += 1


def _diag(code: str = "ERR", severity: int = 1, source: str = "test") -> Diagnostic:
    return Diagnostic(code=code, line=1, col=1, message="synthetic test error", offset=0, source=source, severity=severity)


# --------------------------------------------------------------------------- #
# Diagnostic Parser Tests
# --------------------------------------------------------------------------- #

def test_parse_rust_diagnostics_text_format():
    output = """
error[E0382]: borrow of moved value: `x`
 --> src/main.rs:5:10
  |
5 |     drop(x);
  |          - value moved here
warning: unused variable: `y`
 --> src/main.rs:6:5
error: clippy::unwrap_used: used unwrap() on a Result value
 --> src/main.rs:7:9
"""
    diags = parse_rust_diagnostics(output)
    assert len(diags) == 3

    assert diags[0].code == "E0382"
    assert diags[0].line == 5
    assert diags[0].col == 10
    assert diags[0].severity == 1
    assert "borrow of moved value" in diags[0].message

    assert diags[1].code == "RUST_WARN"
    assert diags[1].line == 6
    assert diags[1].col == 5
    assert diags[1].severity == 2

    assert diags[2].code == "clippy::unwrap_used"
    assert diags[2].line == 7
    assert diags[2].col == 9
    assert diags[2].severity == 1


def test_parse_rust_diagnostics_json_format():
    json_line = (
        '{"reason":"compiler-message","message":{"code":{"code":"E0597"},'
        '"level":"error","message":"`val` does not live long enough",'
        '"spans":[{"line_start":12,"column_start":8}]}}'
    )
    diags = parse_rust_diagnostics(json_line)
    assert len(diags) == 1
    assert diags[0].code == "E0597"
    assert diags[0].line == 12
    assert diags[0].col == 8
    assert diags[0].severity == 1
    assert "does not live long enough" in diags[0].message


def test_parse_cpp_diagnostics():
    output = """
main.cpp:10:5: error: static_assert failed due to requirement 'sizeof(int) == 8' [-Werror]
main.cpp:15:12: error: no matching function for call to 'f' [clang-diagnostic-error]
main.cpp:20:9: warning: use of old-style cast [google-readability-casting]
main.cpp:25:3: fatal error: 'missing.h' file not found
"""
    diags = parse_cpp_diagnostics(output)
    assert len(diags) == 4

    assert diags[0].line == 10
    assert diags[0].col == 5
    assert diags[0].severity == 1
    assert diags[0].code == "-Werror"

    assert diags[1].line == 15
    assert diags[1].col == 12
    assert diags[1].severity == 1
    assert diags[1].code == "clang-diagnostic-error"

    assert diags[2].line == 20
    assert diags[2].col == 9
    assert diags[2].severity == 2
    assert diags[2].code == "google-readability-casting"

    assert diags[3].line == 25
    assert diags[3].col == 3
    assert diags[3].severity == 1


def test_parse_swift_diagnostics():
    output = """
main.swift:12:5: error: mutation of captured var in concurrently-executing code
main.swift:15:9: warning: non-sendable type 'NonSendableClass' in concurrent context
main.swift:20:3: note: consider using '@Sendable' closure
"""
    diags = parse_swift_diagnostics(output)
    assert len(diags) == 3

    assert diags[0].line == 12
    assert diags[0].col == 5
    assert diags[0].severity == 1
    assert diags[0].code == "SWIFT_CONCURRENCY"

    assert diags[1].line == 15
    assert diags[1].col == 9
    assert diags[1].severity == 2
    assert diags[1].code == "SWIFT_CONCURRENCY"

    assert diags[2].line == 20
    assert diags[2].col == 3
    assert diags[2].severity == 3


def test_parse_kotlin_diagnostics():
    output = """
main.kt:14:10: error: type mismatch: inferred type is String? but String was expected
main.kt:18:5: error: 'when' expression must be exhaustive
main.kt:22:9: warning: variable 'x' is never used
"""
    diags = parse_kotlin_diagnostics(output)
    assert len(diags) == 3

    assert diags[0].line == 14
    assert diags[0].col == 10
    assert diags[0].severity == 1
    assert diags[0].code == "KT_TYPE_MISMATCH"

    assert diags[1].line == 18
    assert diags[1].col == 5
    assert diags[1].severity == 1
    assert diags[1].code == "KT_NON_EXHAUSTIVE_WHEN"

    assert diags[2].line == 22
    assert diags[2].col == 9
    assert diags[2].severity == 2
    assert diags[2].code == "KT_WARN"


# --------------------------------------------------------------------------- #
# Anti-Goodhart Escape-Hatch Tests
# --------------------------------------------------------------------------- #

def test_rust_escape_hatches_detects_all_targets():
    # Reject unsafe, .unwrap(), todo!(), unimplemented!(), #[allow(clippy::all)]
    code_unsafe = "pub unsafe fn deref(p: *const i32) -> i32 { *p }"
    assert "unsafe" in find_rust_escape_hatches(code_unsafe)

    code_unwrap = "let val = opt.unwrap();"
    assert "unwrap" in find_rust_escape_hatches(code_unwrap)

    code_todo = "fn solve() { todo!() }"
    assert "todo" in find_rust_escape_hatches(code_todo)

    code_unimplemented = "fn run() { unimplemented!(\"later\") }"
    assert "unimplemented" in find_rust_escape_hatches(code_unimplemented)

    code_allow = "#[allow(clippy::all)]\nfn foo() {}"
    assert "allow_clippy_all" in find_rust_escape_hatches(code_allow)

    code_panic = "if !valid { panic!(\"boom\"); }"
    assert "panic" in find_rust_escape_hatches(code_panic)


def test_rust_escape_hatches_masking_prevents_false_positives():
    code_clean = """
    // Do not use unsafe or unwrap here
    pub fn add(a: i32, b: i32) -> i32 {
        let msg = "todo!() and unimplemented!() are avoided";
        a + b
    }
    """
    assert find_rust_escape_hatches(code_clean) == []


def test_cpp_escape_hatches_detects_all_targets():
    # Reject reinterpret_cast, raw pointer ownership transfer, implicit narrowing
    code_reinterpret = "auto p = reinterpret_cast<int*>(ptr);"
    assert "reinterpret_cast" in find_cpp_escape_hatches(code_reinterpret)

    code_delete = "delete ptr;"
    assert "raw_pointer_ownership" in find_cpp_escape_hatches(code_delete)

    code_free = "free(raw_mem);"
    assert "raw_pointer_ownership" in find_cpp_escape_hatches(code_free)

    code_new = "int* p = new int(42);"
    assert "raw_pointer_ownership" in find_cpp_escape_hatches(code_new)

    code_release = "auto raw = smart_p.release();"
    assert "raw_pointer_ownership" in find_cpp_escape_hatches(code_release)

    code_pragma = '#pragma clang diagnostic ignored "-Wnarrowing"'
    assert "implicit_narrowing" in find_cpp_escape_hatches(code_pragma)


def test_cpp_escape_hatches_masking_allows_smart_pointers():
    code_clean = """
    // Avoid delete and free by using std::unique_ptr
    #include <memory>
    std::unique_ptr<int> make_val(int x) {
        auto msg = "reinterpret_cast in string literal";
        return std::make_unique<int>(x);
    }
    """
    assert find_cpp_escape_hatches(code_clean) == []


def test_swift_escape_hatches_detects_all_targets():
    # Reject @unchecked Sendable, fatalError(), force unwrap !
    code_sendable = "final class State: @unchecked Sendable {}"
    assert "unchecked_sendable" in find_swift_escape_hatches(code_sendable)

    code_fatal = "func solve() { fatalError(\"unimplemented\") }"
    assert "fatal_error" in find_swift_escape_hatches(code_fatal)

    code_force_unwrap = "let x = optionalVal!"
    assert "force_unwrap" in find_swift_escape_hatches(code_force_unwrap)

    code_force_cast = "let x = val as! String"
    assert "force_cast" in find_swift_escape_hatches(code_force_cast)


def test_swift_escape_hatches_allows_boolean_not_and_inequality():
    code_clean = """
    // Do not call fatalError
    func check(a: Int, b: Int, flag: Bool) -> Bool {
        if a != b && !flag {
            return true
        }
        return false
    }
    """
    assert find_swift_escape_hatches(code_clean) == []


def test_kotlin_escape_hatches_detects_all_targets():
    # Reject @Suppress, !! force assertions, non-exhaustive when branches
    code_suppress = '@Suppress("UNCHECKED_CAST") fun f() {}'
    assert "suppress" in find_kotlin_escape_hatches(code_suppress)

    code_assertion = "val length = text!!.length"
    assert "force_assertion" in find_kotlin_escape_hatches(code_assertion)

    code_todo = "fun solve(): Int = TODO()"
    assert "todo" in find_kotlin_escape_hatches(code_todo)

    code_when = "when (state) { State.A -> 1; else -> {} }"
    assert "non_exhaustive_when" in find_kotlin_escape_hatches(code_when)


def test_kotlin_escape_hatches_allows_null_safety_and_inequality():
    code_clean = """
    // @Suppress should not trigger from comments
    fun validate(a: String?, b: String): Boolean {
        if (a != b) {
            val len = a?.length ?: 0
            return len > 0
        }
        return false
    }
    """
    assert find_kotlin_escape_hatches(code_clean) == []


# --------------------------------------------------------------------------- #
# Verifier Reward Behavior with Injected Oracles
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "verifier_cls,valid_code",
    [
        (RustVerifier, "pub fn add(a: i32, b: i32) -> i32 { a + b }\n"),
        (CppVerifier, "int add(int a, int b) { return a + b; }\n"),
        (SwiftVerifier, "func add(a: Int, b: Int) -> Int { return a + b }\n"),
        (KotlinVerifier, "fun add(a: Int, b: Int): Int = a + b\n"),
    ],
)
def test_verifier_clean_sample_scores_one(verifier_cls, valid_code):
    oracle = FakeCompilerOracle(diags=[])
    v = verifier_cls(oracle=oracle)
    assert v.reward(valid_code) == 1.0
    assert v.telemetry()["n_clean"] == 1
    assert v.telemetry()["n_samples"] == 1


@pytest.mark.parametrize(
    "verifier_cls,hacked_code,hatch_name",
    [
        (RustVerifier, "pub fn f() { todo!() }", "todo"),
        (CppVerifier, "void f(int* p) { delete p; }", "raw_pointer_ownership"),
        (SwiftVerifier, "func f(x: Int?) -> Int { return x! }", "force_unwrap"),
        (KotlinVerifier, "fun f(x: String?): Int = x!!.length", "force_assertion"),
    ],
)
def test_verifier_escape_hatch_dominates_clean_compiler(verifier_cls, hacked_code, hatch_name):
    oracle = FakeCompilerOracle(diags=[])
    v = verifier_cls(oracle=oracle)
    score = v.reward(hacked_code)
    assert score == -1.0
    t = v.telemetry()
    assert t["n_hacked"] == 1
    assert t["hatch_counts"].get(hatch_name, 0) >= 1


@pytest.mark.parametrize(
    "verifier_cls",
    [RustVerifier, CppVerifier, SwiftVerifier, KotlinVerifier],
)
def test_verifier_empty_and_degenerate_output_penalized(verifier_cls):
    oracle = FakeCompilerOracle(diags=[])
    v = verifier_cls(oracle=oracle)
    assert v.reward("") == -1.0
    assert v.reward("   \n\t") == -1.0
    assert v.reward("// only comments\n/* more comments */") == -1.0
    assert v.telemetry()["n_degenerate"] == 3


@pytest.mark.parametrize(
    "verifier_cls,valid_code",
    [
        (RustVerifier, "pub fn add(a: i32, b: i32) -> i32 { a + b }"),
        (CppVerifier, "int add(int a, int b) { return a + b; }"),
        (SwiftVerifier, "func add(a: Int, b: Int) -> Int { return a + b }"),
        (KotlinVerifier, "fun add(a: Int, b: Int): Int = a + b"),
    ],
)
def test_verifier_diagnostics_penalties(verifier_cls, valid_code):
    # 1 error (weight 0.30) -> reward 0.70
    oracle = FakeCompilerOracle(diags=[_diag(code="E1", severity=1)])
    v = verifier_cls(oracle=oracle)
    assert v.reward(valid_code) == pytest.approx(0.70)

    # 1 warning (weight 0.15) -> reward 0.85
    oracle2 = FakeCompilerOracle(diags=[_diag(code="W1", severity=2)])
    v2 = verifier_cls(oracle=oracle2)
    assert v2.reward(valid_code) == pytest.approx(0.85)

    # 10 errors -> clamped at diag_floor (-0.50)
    oracle10 = FakeCompilerOracle(diags=[_diag(code=f"E{i}", severity=1) for i in range(10)])
    v10 = verifier_cls(oracle=oracle10)
    assert v10.reward(valid_code) == pytest.approx(-0.50)


def test_verifier_fail_fast_skips_oracle():
    oracle = FakeCompilerOracle(diags=[])
    v = RustVerifier(oracle=oracle, fail_fast=True)
    # Hacked sample
    v.reward("pub fn f() { todo!() }")
    # Degenerate sample
    v.reward("")
    assert oracle.n_calls == 0

    # Clean sample calls oracle
    v.reward("pub fn add(a: i32, b: i32) -> i32 { a + b }")
    assert oracle.n_calls == 1


def test_verifier_hatches_directives_and_none_modes():
    # Directives mode ignores non-directive hatches (e.g. unwrap in Rust)
    oracle = FakeCompilerOracle(diags=[])
    v_dir = RustVerifier(oracle=oracle, hatches="directives")
    assert v_dir.reward("fn f(opt: Option<i32>) -> i32 { opt.unwrap() }") == 1.0

    # But catches directive hatches (#[allow(clippy::all)])
    assert v_dir.reward("#[allow(clippy::all)]\nfn f() {}") == -1.0

    # None mode disables hack floor entirely
    v_none = RustVerifier(oracle=oracle, hatches="none")
    assert v_none.reward("#[allow(clippy::all)]\nfn f() { todo!() }") == 1.0


def test_verifier_context_manager_and_close():
    oracle = FakeCompilerOracle(diags=[])
    with RustVerifier(oracle=oracle) as v:
        v.reward("pub fn f() {}")
    v.close()  # idempotent close
    assert oracle.close_count == 1


def test_verifier_on_error_behavior():
    oracle = FakeCompilerOracle(raises=True)
    v_raise = RustVerifier(oracle=oracle, on_error="raise")
    with pytest.raises(RuntimeError):
        v_raise.reward("pub fn f() {}")
    assert v_raise.telemetry()["n_oracle_errors"] == 1

    v_skip = RustVerifier(oracle=oracle, on_error="skip")
    assert v_skip.reward("pub fn f() {}") is None
    assert v_skip.telemetry()["n_oracle_errors"] == 1


# --------------------------------------------------------------------------- #
# Toolchain Integration Tests (Executed only when compilers are present)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(resolve_rust_toolchain() is None, reason="rustc / cargo not installed on host")
def test_rust_verifier_real_toolchain():
    v = RustVerifier()
    clean = v.reward("pub fn add(a: i32, b: i32) -> i32 { a + b }\n")
    broken = v.reward("pub fn broken() -> i32 { let x = true; x }\n")
    assert clean > broken
    assert clean == 1.0
    assert broken < 1.0
    v.close()


@pytest.mark.skipif(resolve_cpp_toolchain() is None, reason="clang++ not installed on host")
def test_cpp_verifier_real_toolchain():
    v = CppVerifier()
    clean = v.reward("int add(int a, int b) { return a + b; }\n")
    broken = v.reward("int broken() { return \"hello\"; }\n")
    assert clean > broken
    assert clean == 1.0
    assert broken < 1.0
    v.close()


@pytest.mark.skipif(resolve_swift_toolchain() is None, reason="swiftc not installed on host")
def test_swift_verifier_real_toolchain():
    v = SwiftVerifier()
    clean = v.reward("func add(a: Int, b: Int) -> Int { return a + b }\n")
    broken = v.reward("func broken() -> Int { return \"hello\" }\n")
    assert clean > broken
    assert clean == 1.0
    assert broken < 1.0
    v.close()


@pytest.mark.skipif(resolve_kotlin_toolchain() is None, reason="kotlinc not installed on host")
def test_kotlin_verifier_real_toolchain():
    v = KotlinVerifier()
    clean = v.reward("fun add(a: Int, b: Int): Int = a + b\n")
    broken = v.reward("fun broken(): Int = \"hello\"\n")
    assert clean > broken
    v.close()
