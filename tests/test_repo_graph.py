"""Tests for repository dependency graph and topological context sorting (#359).

Covers:
* TypeScript and Python import/export extraction and relative path resolution
* Dependency DAG construction
* Topological sort with definitions preceding use sites
* Deterministic circular dependency resolution (no infinite loops, zero missing files)
* Fixture repository verification
"""

from pathlib import Path
import pytest

from src.data.repo_graph import (
    RepoGraph,
    build_repo_graph,
    extract_py_imports_exports,
    extract_ts_imports_exports,
    resolve_py_specifier,
    resolve_ts_specifier,
)

FIXTURE_REPO = Path(__file__).resolve().parents[1] / "eval_sets/code_recall/fixture_repo.jsonl"


# --------------------------------------------------------------------------------------- #
# TypeScript AST & Specifier Resolution
# --------------------------------------------------------------------------------------- #

def test_extract_ts_imports_and_exports():
    code = (
        'import { User, Role } from "./types";\n'
        'import defaultHelper from "./helpers";\n'
        'import * as math from "./math";\n'
        'export { sum } from "./calc";\n'
        "export function getUser(): User { return { id: '1' }; }\n"
        "export const DEFAULT_TIMEOUT = 5000;\n"
        "export class AuthService {}\n"
        "export interface Config {}\n"
        "export type ID = string;\n"
        "export enum Status { ACTIVE }\n"
    )
    imported, exported = extract_ts_imports_exports(code)
    assert imported == {"./types", "./helpers", "./math", "./calc"}
    assert exported == {
        "getUser",
        "DEFAULT_TIMEOUT",
        "AuthService",
        "Config",
        "ID",
        "Status",
    }


def test_resolve_ts_specifier():
    known = {
        "src/types.ts",
        "src/utils/index.ts",
        "src/components/Button.tsx",
        "src/service.js",
    }
    assert resolve_ts_specifier("./types", "src/user.ts", known) == "src/types.ts"
    assert resolve_ts_specifier("./utils", "src/main.ts", known) == "src/utils/index.ts"
    assert resolve_ts_specifier("./components/Button", "src/App.tsx", known) == "src/components/Button.tsx"
    assert resolve_ts_specifier("../service", "src/utils/index.ts", known) == "src/service.js"
    # External or non-existent
    assert resolve_ts_specifier("react", "src/App.tsx", known) is None
    assert resolve_ts_specifier("./missing", "src/user.ts", known) is None


# --------------------------------------------------------------------------------------- #
# Python AST & Specifier Resolution
# --------------------------------------------------------------------------------------- #

def test_extract_py_imports_and_exports():
    code = (
        "import os\n"
        "import sys as system\n"
        "from . import base\n"
        "from .helpers import format_text\n"
        "from ..config import Settings\n"
        "from package.module import util\n"
        "\n"
        "__all__ = ['public_api', 'HelperClass']\n"
        "\n"
        "def public_api(): pass\n"
        "def _private_func(): pass\n"
        "class HelperClass: pass\n"
        "CONSTANT = 42\n"
    )
    imported, exported = extract_py_imports_exports(code)
    # Check imports
    assert ("os", 0) in imported
    assert ("sys", 0) in imported
    assert ("base", 1) in imported
    assert ("helpers", 1) in imported
    assert ("config", 2) in imported
    assert ("package.module", 0) in imported

    # Check exports (__all__ overrides or explicit top-level)
    assert "public_api" in exported
    assert "HelperClass" in exported
    assert "_private_func" not in exported


def test_resolve_py_specifier():
    known = {
        "pkg/__init__.py",
        "pkg/models.py",
        "pkg/views.py",
        "pkg/sub/helpers.py",
        "config.py",
    }
    # Relative import in pkg/views.py -> from .models import User
    assert resolve_py_specifier("models", 1, "pkg/views.py", known) == "pkg/models.py"
    # Relative import in pkg/sub/helpers.py -> from ..models import User
    assert resolve_py_specifier("models", 2, "pkg/sub/helpers.py", known) == "pkg/models.py"
    # Absolute import -> from pkg.models import User
    assert resolve_py_specifier("pkg.models", 0, "pkg/views.py", known) == "pkg/models.py"
    # Absolute import to top-level module
    assert resolve_py_specifier("config", 0, "pkg/views.py", known) == "config.py"


# --------------------------------------------------------------------------------------- #
# DAG Construction & Topological Ordering
# --------------------------------------------------------------------------------------- #

def test_ts_dag_ordering_definitions_before_uses():
    files = {
        "src/types.ts": "export interface Token { id: number; }\n",
        "src/tokenizer.ts": (
            'import { Token } from "./types";\n'
            "export function tokenize(s: string): Token[] { return []; }\n"
        ),
        "src/pipeline.ts": (
            'import { tokenize } from "./tokenizer";\n'
            'import { Token } from "./types";\n'
            "export function run(s: string) { return tokenize(s); }\n"
        ),
        "src/standalone.ts": "export const PI = 3.14;\n",
    }
    graph = build_repo_graph(files)
    order = graph.topological_sort()

    assert set(order) == set(files.keys())
    # types.ts must precede tokenizer.ts and pipeline.ts
    assert order.index("src/types.ts") < order.index("src/tokenizer.ts")
    assert order.index("src/types.ts") < order.index("src/pipeline.ts")
    # tokenizer.ts must precede pipeline.ts
    assert order.index("src/tokenizer.ts") < order.index("src/pipeline.ts")


def test_py_dag_ordering_definitions_before_uses():
    files = {
        "app/schema.py": "class UserSchema: pass\n",
        "app/models.py": (
            "from .schema import UserSchema\n"
            "class UserModel: pass\n"
        ),
        "app/controllers.py": (
            "from .models import UserModel\n"
            "def get_user(): return UserModel()\n"
        ),
    }
    graph = build_repo_graph(files)
    order = graph.topological_sort()

    assert set(order) == set(files.keys())
    assert order.index("app/schema.py") < order.index("app/models.py")
    assert order.index("app/models.py") < order.index("app/controllers.py")


# --------------------------------------------------------------------------------------- #
# Circular Dependency Resolution
# --------------------------------------------------------------------------------------- #

def test_circular_dependency_resolves_deterministically_without_infinite_loop():
    # Mutual dependency: A <-> B
    files = {
        "src/a.ts": 'import { b } from "./b";\nexport const a = 1;\n',
        "src/b.ts": 'import { a } from "./a";\nexport const b = 2;\n',
        "src/c.ts": 'import { a } from "./a";\nexport const c = 3;\n',
    }
    graph = build_repo_graph(files)
    order = graph.topological_sort()

    # All files present
    assert set(order) == {"src/a.ts", "src/b.ts", "src/c.ts"}
    assert len(order) == 3

    # Stable deterministic tie-breaking (same order every time)
    for _ in range(5):
        assert graph.topological_sort() == order


def test_multi_cycle_with_depth_tie_breaking():
    # 3-node cycle: A -> B -> C -> A
    # Plus leaf node D at deeper path
    files = {
        "src/core/a.ts": 'import { b } from "./b";\nexport const a = 1;\n',
        "src/core/b.ts": 'import { c } from "./c";\nexport const b = 2;\n',
        "src/core/c.ts": 'import { a } from "./a";\nexport const c = 3;\n',
        "src/core/utils/deep.ts": 'export const deep = 4;\n',
        "src/entry.ts": 'import { a } from "./core/a";\n',
    }
    graph = build_repo_graph(files)
    order = graph.topological_sort()

    assert set(order) == set(files.keys())
    assert len(order) == 5
    # entry.ts depends on core/a.ts, should appear after core/a.ts
    assert order.index("src/core/a.ts") < order.index("src/entry.ts")


# --------------------------------------------------------------------------------------- #
# Fixture Repo Verification
# --------------------------------------------------------------------------------------- #

def test_fixture_repo_topological_sort():
    assert FIXTURE_REPO.exists(), f"fixture repo not found at {FIXTURE_REPO}"
    graph = build_repo_graph(FIXTURE_REPO)
    order = graph.topological_sort()

    assert len(order) == len(graph.files)
    assert set(order) == set(graph.files)

    # Key definer -> user pairs from fixture_repo.jsonl:
    # convert.ts imports units.ts
    assert order.index("src/units.ts") < order.index("src/convert.ts")
    # labels.ts imports strings.ts
    assert order.index("src/strings.ts") < order.index("src/labels.ts")
    # linalg.ts imports matrix.ts
    assert order.index("src/matrix.ts") < order.index("src/linalg.ts")
    # report.ts imports geometry.ts
    assert order.index("src/geometry.ts") < order.index("src/report.ts")
