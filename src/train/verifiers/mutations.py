"""#361 -- RLVR: Mutation testing engine for execution verifier reward.

Provides deterministic synthetic fault injection (inverting conditionals, altering arithmetic
operators, modifying boundary constants) across Python and TypeScript codebases.
Computes mutant kill scores to reward completions that pass test suites and kill synthetic mutants:
    R = R_pass * (1 + gamma * R_mutation)
"""

from __future__ import annotations

import ast
import copy
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Mutant:
    """A synthetically mutated variant of source code."""

    id: str
    mutated_code: str
    operator: str
    description: str
    line_number: Optional[int] = None
    original_snippet: str = ""
    mutated_snippet: str = ""


@dataclass
class MutationScoreResult:
    """Outcome of mutation testing over a test suite."""

    pass_rate: float
    mutants_tested: int
    mutants_killed: int
    mutant_kill_rate: float
    reward: float
    killed_mutant_ids: List[str]
    survived_mutant_ids: List[str]


def score_mutation(
    pass_rate: float,
    mutants_tested: int,
    mutants_killed: int,
    gamma: float = 0.5,
) -> float:
    """Compute mutation-augmented reward according to:

        R = R_pass * (1 + gamma * R_mutation)

    If pass_rate is 0.0, reward is 0.0 regardless of mutants.
    If mutants_tested is 0, R_mutation is 0.0, returning pass_rate.
    """
    if pass_rate <= 0.0:
        return 0.0
    kill_rate = (mutants_killed / mutants_tested) if mutants_tested > 0 else 0.0
    return float(pass_rate * (1.0 + gamma * kill_rate))


def _extract_code_intervals(source: str) -> List[Tuple[int, int]]:
    """Extract character index intervals of source that are not inside strings or comments."""
    comment_inline = r"//[^\n]*"
    comment_hash = r"#[^\n]*"
    comment_block = r"/\*[\s\S]*?\*/"
    str_template = r"`[^`\\]*(?:\\.[^`\\]*)*`"
    str_double = r'"[^"\\]*(?:\\.[^"\\]*)*"'
    str_single = r"'[^'\\]*(?:\\.[^'\\]*)*'"
    regex_pattern = f"{comment_inline}|{comment_hash}|{comment_block}|{str_template}|{str_double}|{str_single}"
    pattern = re.compile(regex_pattern)
    intervals: List[Tuple[int, int]] = []
    last_end = 0
    for match in pattern.finditer(source):
        start, end = match.span()
        if start > last_end:
            intervals.append((last_end, start))
        last_end = end
    if last_end < len(source):
        intervals.append((last_end, len(source)))
    return intervals


class MutationOperator(ABC):
    """Abstract mutation operator interface."""

    name: str = "base"

    @abstractmethod
    def generate_python_mutants(self, tree: ast.AST, code: str) -> List[Mutant]:
        """Generate mutants for Python AST."""
        ...

    @abstractmethod
    def generate_ts_mutants(self, code: str) -> List[Mutant]:
        """Generate mutants for TypeScript / JavaScript code."""
        ...


class InvertConditionals(MutationOperator):
    """Invert conditionals and comparison operators:

    - Comparison: == <-> !=, < <-> >=, <= <-> >, > <-> <=, >= <-> <, is <-> is not, in <-> not in
    - Logical: and <-> or, && <-> ||
    - Booleans: True <-> False, true <-> false
    """

    name: str = "invert_conditionals"

    _OP_INVERSIONS = {
        ast.Eq: ast.NotEq,
        ast.NotEq: ast.Eq,
        ast.Lt: ast.GtE,
        ast.LtE: ast.Gt,
        ast.Gt: ast.LtE,
        ast.GtE: ast.Lt,
        ast.Is: ast.IsNot,
        ast.IsNot: ast.Is,
        ast.In: ast.NotIn,
        ast.NotIn: ast.In,
    }

    _OP_NAMES = {
        ast.Eq: "==",
        ast.NotEq: "!=",
        ast.Lt: "<",
        ast.LtE: "<=",
        ast.Gt: ">",
        ast.GtE: ">=",
        ast.Is: "is",
        ast.IsNot: "is not",
        ast.In: "in",
        ast.NotIn: "not in",
    }

    def generate_python_mutants(self, tree: ast.AST, code: str) -> List[Mutant]:
        mutants: List[Mutant] = []
        mutant_idx = 0

        class Collector(ast.NodeVisitor):
            def __init__(self) -> None:
                self.compare_sites: List[Tuple[ast.Compare, int, Any, Any]] = []
                self.boolop_sites: List[Tuple[ast.BoolOp, Any]] = []
                self.constant_sites: List[Tuple[ast.Constant, bool]] = []

            def visit_Compare(self, node: ast.Compare) -> None:
                for op_idx, op in enumerate(node.ops):
                    op_cls = type(op)
                    if op_cls in InvertConditionals._OP_INVERSIONS:
                        self.compare_sites.append(
                            (node, op_idx, op_cls, InvertConditionals._OP_INVERSIONS[op_cls])
                        )
                self.generic_visit(node)

            def visit_BoolOp(self, node: ast.BoolOp) -> None:
                if isinstance(node.op, ast.And):
                    self.boolop_sites.append((node, ast.Or))
                elif isinstance(node.op, ast.Or):
                    self.boolop_sites.append((node, ast.And))
                self.generic_visit(node)

            def visit_Constant(self, node: ast.Constant) -> None:
                if isinstance(node.value, bool):
                    self.constant_sites.append((node, not node.value))
                self.generic_visit(node)

        collector = Collector()
        collector.visit(tree)

        for node, op_idx, old_op, new_op in collector.compare_sites:
            mutant_idx += 1
            tree_copy = copy.deepcopy(tree)

            class Replacer(ast.NodeTransformer):
                def visit_Compare(self, n: ast.Compare) -> ast.Compare:
                    if (
                        n.lineno == node.lineno
                        and n.col_offset == node.col_offset
                        and len(n.ops) > op_idx
                        and isinstance(n.ops[op_idx], old_op)
                    ):
                        n.ops[op_idx] = new_op()
                    return n

            Replacer().visit(tree_copy)
            ast.fix_missing_locations(tree_copy)
            try:
                mutated_src = ast.unparse(tree_copy)
                old_name = self._OP_NAMES.get(old_op, str(old_op))
                new_name = self._OP_NAMES.get(new_op, str(new_op))
                mutants.append(
                    Mutant(
                        id=f"cond_cmp_{mutant_idx}",
                        mutated_code=mutated_src,
                        operator=self.name,
                        description=f"Invert comparison {old_name} -> {new_name}",
                        line_number=getattr(node, "lineno", None),
                        original_snippet=old_name,
                        mutated_snippet=new_name,
                    )
                )
            except Exception:
                pass

        for node, new_boolop in collector.boolop_sites:
            mutant_idx += 1
            tree_copy = copy.deepcopy(tree)

            class BoolReplacer(ast.NodeTransformer):
                def visit_BoolOp(self, n: ast.BoolOp) -> ast.BoolOp:
                    if n.lineno == node.lineno and n.col_offset == node.col_offset:
                        n.op = new_boolop()
                    return n

            BoolReplacer().visit(tree_copy)
            ast.fix_missing_locations(tree_copy)
            try:
                mutated_src = ast.unparse(tree_copy)
                mutants.append(
                    Mutant(
                        id=f"cond_bool_{mutant_idx}",
                        mutated_code=mutated_src,
                        operator=self.name,
                        description=f"Invert logical operator to {new_boolop.__name__.lower()}",
                        line_number=getattr(node, "lineno", None),
                        original_snippet="and" if new_boolop is ast.Or else "or",
                        mutated_snippet="or" if new_boolop is ast.Or else "and",
                    )
                )
            except Exception:
                pass

        for node, new_val in collector.constant_sites:
            mutant_idx += 1
            tree_copy = copy.deepcopy(tree)

            class ConstReplacer(ast.NodeTransformer):
                def visit_Constant(self, n: ast.Constant) -> ast.Constant:
                    if (
                        n.lineno == node.lineno
                        and n.col_offset == node.col_offset
                        and isinstance(n.value, bool)
                    ):
                        n.value = new_val
                    return n

            ConstReplacer().visit(tree_copy)
            ast.fix_missing_locations(tree_copy)
            try:
                mutated_src = ast.unparse(tree_copy)
                mutants.append(
                    Mutant(
                        id=f"cond_const_{mutant_idx}",
                        mutated_code=mutated_src,
                        operator=self.name,
                        description=f"Invert boolean constant to {new_val}",
                        line_number=getattr(node, "lineno", None),
                        original_snippet=str(not new_val),
                        mutated_snippet=str(new_val),
                    )
                )
            except Exception:
                pass

        return mutants

    def generate_ts_mutants(self, code: str) -> List[Mutant]:
        mutants: List[Mutant] = []
        intervals = _extract_code_intervals(code)
        mutant_idx = 0

        replacements = [
            (re.compile(r"==="), "!=="),
            (re.compile(r"!=="), "==="),
            (re.compile(r"(?<!=)==(?!=)"), "!="),
            (re.compile(r"!="), "=="),
            (re.compile(r"<="), ">"),
            (re.compile(r">="), "<"),
            (re.compile(r"(?<![<=-])<(?![<=])"), ">="),
            (re.compile(r"(?<![>=-])>(?![>=])"), "<="),
            (re.compile(r"&&"), "||"),
            (re.compile(r"\|\|"), "&&"),
            (re.compile(r"\btrue\b"), "false"),
            (re.compile(r"\bfalse\b"), "true"),
        ]

        for pattern, replacement in replacements:
            for s_int, e_int in intervals:
                chunk = code[s_int:e_int]
                for match in pattern.finditer(chunk):
                    mutant_idx += 1
                    m_start = s_int + match.start()
                    m_end = s_int + match.end()
                    mutated = code[:m_start] + replacement + code[m_end:]
                    orig_str = code[m_start:m_end]
                    mutants.append(
                        Mutant(
                            id=f"ts_cond_{mutant_idx}",
                            mutated_code=mutated,
                            operator=self.name,
                            description=f"Invert conditional {orig_str} -> {replacement}",
                            line_number=code[:m_start].count("\n") + 1,
                            original_snippet=orig_str,
                            mutated_snippet=replacement,
                        )
                    )
        return mutants


class AlterArithmetic(MutationOperator):
    """Alter arithmetic operators:

    - Addition/Subtraction: + <-> -
    - Multiplication/Division: * <-> /
    - Modulo: % <-> *
    - Floor division/Power: // <-> *, ** <-> *
    """

    name: str = "alter_arithmetic"

    _BINOP_MAP = {
        ast.Add: ast.Sub,
        ast.Sub: ast.Add,
        ast.Mult: ast.Div,
        ast.Div: ast.Mult,
        ast.FloorDiv: ast.Mult,
        ast.Mod: ast.Mult,
        ast.Pow: ast.Mult,
    }

    _BINOP_NAMES = {
        ast.Add: "+",
        ast.Sub: "-",
        ast.Mult: "*",
        ast.Div: "/",
        ast.FloorDiv: "//",
        ast.Mod: "%",
        ast.Pow: "**",
    }

    def generate_python_mutants(self, tree: ast.AST, code: str) -> List[Mutant]:
        mutants: List[Mutant] = []
        mutant_idx = 0

        class BinOpCollector(ast.NodeVisitor):
            def __init__(self) -> None:
                self.sites: List[Tuple[ast.BinOp, Any, Any]] = []

            def visit_BinOp(self, node: ast.BinOp) -> None:
                op_cls = type(node.op)
                if op_cls in AlterArithmetic._BINOP_MAP:
                    self.sites.append((node, op_cls, AlterArithmetic._BINOP_MAP[op_cls]))
                self.generic_visit(node)

        collector = BinOpCollector()
        collector.visit(tree)

        for node, old_op, new_op in collector.sites:
            mutant_idx += 1
            tree_copy = copy.deepcopy(tree)

            class Replacer(ast.NodeTransformer):
                def visit_BinOp(self, n: ast.BinOp) -> ast.BinOp:
                    if (
                        n.lineno == node.lineno
                        and n.col_offset == node.col_offset
                        and isinstance(n.op, old_op)
                    ):
                        n.op = new_op()
                    return n

            Replacer().visit(tree_copy)
            ast.fix_missing_locations(tree_copy)
            try:
                mutated_src = ast.unparse(tree_copy)
                old_sym = self._BINOP_NAMES.get(old_op, str(old_op))
                new_sym = self._BINOP_NAMES.get(new_op, str(new_op))
                mutants.append(
                    Mutant(
                        id=f"arith_{mutant_idx}",
                        mutated_code=mutated_src,
                        operator=self.name,
                        description=f"Alter arithmetic {old_sym} -> {new_sym}",
                        line_number=getattr(node, "lineno", None),
                        original_snippet=old_sym,
                        mutated_snippet=new_sym,
                    )
                )
            except Exception:
                pass

        return mutants

    def generate_ts_mutants(self, code: str) -> List[Mutant]:
        mutants: List[Mutant] = []
        intervals = _extract_code_intervals(code)
        mutant_idx = 0

        replacements = [
            (re.compile(r"(?<=\s)\+(?=\s)"), "-"),
            (re.compile(r"(?<=\s)-(?=\s)"), "+"),
            (re.compile(r"(?<=\s)\*(?=\s)"), "/"),
            (re.compile(r"(?<=\s)/(?=\s)"), "*"),
            (re.compile(r"(?<=\s)%(?=\s)"), "*"),
        ]

        for pattern, replacement in replacements:
            for s_int, e_int in intervals:
                chunk = code[s_int:e_int]
                for match in pattern.finditer(chunk):
                    mutant_idx += 1
                    m_start = s_int + match.start()
                    m_end = s_int + match.end()
                    mutated = code[:m_start] + replacement + code[m_end:]
                    orig_str = code[m_start:m_end]
                    mutants.append(
                        Mutant(
                            id=f"ts_arith_{mutant_idx}",
                            mutated_code=mutated,
                            operator=self.name,
                            description=f"Alter arithmetic {orig_str} -> {replacement}",
                            line_number=code[:m_start].count("\n") + 1,
                            original_snippet=orig_str,
                            mutated_snippet=replacement,
                        )
                    )
        return mutants


class ModifyBoundaryConstants(MutationOperator):
    """Modify boundary constants:

    - Integer boundaries: 0 -> 1, 1 -> 0, -1 -> 0, N -> N + 1
    - Comparison boundary adjustments: < -> <=, <= -> <, > -> >=, >= -> >
    - Strings / Collections: "" -> "MUTANT", [] -> [0]
    """

    name: str = "modify_boundary_constants"

    _BOUNDARY_ADJUSTMENTS = {
        ast.Lt: ast.LtE,
        ast.LtE: ast.Lt,
        ast.Gt: ast.GtE,
        ast.GtE: ast.Gt,
    }

    _BOUNDARY_NAMES = {
        ast.Lt: "<",
        ast.LtE: "<=",
        ast.Gt: ">",
        ast.GtE: ">=",
    }

    def generate_python_mutants(self, tree: ast.AST, code: str) -> List[Mutant]:
        mutants: List[Mutant] = []
        mutant_idx = 0

        class BoundaryCollector(ast.NodeVisitor):
            def __init__(self) -> None:
                self.const_sites: List[Tuple[ast.Constant, Any]] = []
                self.compare_sites: List[Tuple[ast.Compare, int, Any, Any]] = []

            def visit_Constant(self, node: ast.Constant) -> None:
                if isinstance(node.value, bool):
                    return
                if isinstance(node.value, int):
                    if node.value == 0:
                        new_val = 1
                    elif node.value == 1:
                        new_val = 0
                    elif node.value == -1:
                        new_val = 0
                    else:
                        new_val = node.value + 1
                    self.const_sites.append((node, new_val))
                elif isinstance(node.value, str) and node.value == "":
                    self.const_sites.append((node, "MUTANT"))
                self.generic_visit(node)

            def visit_Compare(self, node: ast.Compare) -> None:
                for op_idx, op in enumerate(node.ops):
                    op_cls = type(op)
                    if op_cls in ModifyBoundaryConstants._BOUNDARY_ADJUSTMENTS:
                        self.compare_sites.append(
                            (
                                node,
                                op_idx,
                                op_cls,
                                ModifyBoundaryConstants._BOUNDARY_ADJUSTMENTS[op_cls],
                            )
                        )
                self.generic_visit(node)

        collector = BoundaryCollector()
        collector.visit(tree)

        for node, new_val in collector.const_sites:
            mutant_idx += 1
            tree_copy = copy.deepcopy(tree)

            class ConstReplacer(ast.NodeTransformer):
                def visit_Constant(self, n: ast.Constant) -> ast.Constant:
                    if (
                        n.lineno == node.lineno
                        and n.col_offset == node.col_offset
                        and type(n.value) is type(node.value)
                        and n.value == node.value
                    ):
                        n.value = new_val
                    return n

            ConstReplacer().visit(tree_copy)
            ast.fix_missing_locations(tree_copy)
            try:
                mutated_src = ast.unparse(tree_copy)
                mutants.append(
                    Mutant(
                        id=f"boundary_const_{mutant_idx}",
                        mutated_code=mutated_src,
                        operator=self.name,
                        description=f"Modify boundary constant {node.value!r} -> {new_val!r}",
                        line_number=getattr(node, "lineno", None),
                        original_snippet=repr(node.value),
                        mutated_snippet=repr(new_val),
                    )
                )
            except Exception:
                pass

        for node, op_idx, old_op, new_op in collector.compare_sites:
            mutant_idx += 1
            tree_copy = copy.deepcopy(tree)

            class CmpReplacer(ast.NodeTransformer):
                def visit_Compare(self, n: ast.Compare) -> ast.Compare:
                    if (
                        n.lineno == node.lineno
                        and n.col_offset == node.col_offset
                        and len(n.ops) > op_idx
                        and isinstance(n.ops[op_idx], old_op)
                    ):
                        n.ops[op_idx] = new_op()
                    return n

            CmpReplacer().visit(tree_copy)
            ast.fix_missing_locations(tree_copy)
            try:
                mutated_src = ast.unparse(tree_copy)
                old_sym = self._BOUNDARY_NAMES.get(old_op, str(old_op))
                new_sym = self._BOUNDARY_NAMES.get(new_op, str(new_op))
                mutants.append(
                    Mutant(
                        id=f"boundary_cmp_{mutant_idx}",
                        mutated_code=mutated_src,
                        operator=self.name,
                        description=f"Modify boundary comparison {old_sym} -> {new_sym}",
                        line_number=getattr(node, "lineno", None),
                        original_snippet=old_sym,
                        mutated_snippet=new_sym,
                    )
                )
            except Exception:
                pass

        return mutants

    def generate_ts_mutants(self, code: str) -> List[Mutant]:
        mutants: List[Mutant] = []
        intervals = _extract_code_intervals(code)
        mutant_idx = 0

        cmp_replacements = [
            (re.compile(r"<="), "<"),
            (re.compile(r">="), ">"),
            (re.compile(r"(?<![<=-])<(?![<=])"), "<="),
            (re.compile(r"(?<![>=-])>(?![>=])"), ">="),
        ]
        for pattern, replacement in cmp_replacements:
            for s_int, e_int in intervals:
                chunk = code[s_int:e_int]
                for match in pattern.finditer(chunk):
                    mutant_idx += 1
                    m_start = s_int + match.start()
                    m_end = s_int + match.end()
                    mutated = code[:m_start] + replacement + code[m_end:]
                    orig_str = code[m_start:m_end]
                    mutants.append(
                        Mutant(
                            id=f"ts_bound_cmp_{mutant_idx}",
                            mutated_code=mutated,
                            operator=self.name,
                            description=f"Modify boundary comparison {orig_str} -> {replacement}",
                            line_number=code[:m_start].count("\n") + 1,
                            original_snippet=orig_str,
                            mutated_snippet=replacement,
                        )
                    )

        num_replacements = [
            (re.compile(r"\b0\b"), "1"),
            (re.compile(r"\b1\b"), "0"),
        ]
        for pattern, replacement in num_replacements:
            for s_int, e_int in intervals:
                chunk = code[s_int:e_int]
                for match in pattern.finditer(chunk):
                    mutant_idx += 1
                    m_start = s_int + match.start()
                    m_end = s_int + match.end()
                    mutated = code[:m_start] + replacement + code[m_end:]
                    orig_str = code[m_start:m_end]
                    mutants.append(
                        Mutant(
                            id=f"ts_bound_num_{mutant_idx}",
                            mutated_code=mutated,
                            operator=self.name,
                            description=f"Modify boundary constant {orig_str} -> {replacement}",
                            line_number=code[:m_start].count("\n") + 1,
                            original_snippet=orig_str,
                            mutated_snippet=replacement,
                        )
                    )

        return mutants


class MutationEngine:
    """Deterministic mutation engine coordinating operators and scoring."""

    def __init__(
        self,
        operators: Optional[Sequence[MutationOperator]] = None,
        *,
        max_mutants: Optional[int] = 10,
    ) -> None:
        self.operators = list(
            operators
            if operators is not None
            else [InvertConditionals(), AlterArithmetic(), ModifyBoundaryConstants()]
        )
        self.max_mutants = max_mutants

    def generate_mutants(self, code: str, language: str = "python") -> List[Mutant]:
        """Generate deterministic mutants for candidate code."""
        lang = language.lower().strip()
        all_mutants: List[Mutant] = []

        if lang in ("python", "py"):
            try:
                tree = ast.parse(code)
                for op in self.operators:
                    all_mutants.extend(op.generate_python_mutants(tree, code))
            except SyntaxError:
                for op in self.operators:
                    all_mutants.extend(op.generate_ts_mutants(code))
        else:
            for op in self.operators:
                all_mutants.extend(op.generate_ts_mutants(code))

        all_mutants.sort(key=lambda m: (m.operator, m.line_number or 0, m.id))

        if self.max_mutants is not None and len(all_mutants) > self.max_mutants:
            all_mutants = all_mutants[: self.max_mutants]

        return all_mutants
