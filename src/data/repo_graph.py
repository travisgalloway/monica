"""Repository dependency graph extraction and topological context sorting (#359).

ABOVE THE SEAM. Pure portable Python -- never imports mlx or torch.

Extracts import/export relationships across multi-file TypeScript and Python projects,
constructs a directed acyclic graph (DAG), and resolves dependencies into a deterministic
topological file ordering so that interface definitions precede call sites during context
packing.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# TypeScript regex fallback patterns (used when tree-sitter is unavailable)
_TS_IMPORT_RE = re.compile(
    r"(?:import\s+(?:(?P<clause>[\w*\s{},$]*)\s+from\s+)?['\"](?P<spec>[^'\"]+)['\"]|"
    r"require\s*\(\s*['\"](?P<req_spec>[^'\"]+)['\"]\s*\))",
    re.MULTILINE,
)

_TS_EXPORT_RE = re.compile(
    r"^export\s+(?:async\s+)?(?:function|const|let|var|class|interface|type|enum)\s+"
    r"([A-Za-z_$][A-Za-z0-9_$]*)",
    re.MULTILINE,
)

_TS_REEXPORT_RE = re.compile(
    r"^export\s+(?:\{[^}]*\}|\*)\s+from\s+['\"]([^'\"]+)['\"]",
    re.MULTILINE,
)

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


def _normalize_posix(path: str | Path) -> str:
    """Normalize path into a clean relative POSIX path."""
    p = PurePosixPath(str(path).replace("\\", "/"))
    parts: List[str] = []
    for part in p.parts:
        if part in ("", "."):
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/".join(parts)


def _tree_sitter_available() -> bool:
    try:
        import tree_sitter  # noqa: F401
        import tree_sitter_typescript  # noqa: F401
        return True
    except ImportError:
        return False


def _ts_parser():
    from tree_sitter import Language, Parser
    import tree_sitter_typescript as tsts

    return Parser(Language(tsts.language_typescript()))


def extract_ts_imports_exports(source: str) -> Tuple[Set[str], Set[str]]:
    """Extract imported module specifiers and exported symbols from TypeScript code.

    Returns:
        (imported_specifiers, exported_symbols)
    """
    imported_specs: Set[str] = set()
    exported_syms: Set[str] = set()

    if _tree_sitter_available():
        try:
            parser = _ts_parser()
            tree = parser.parse(source.encode("utf-8"))
            for node in tree.root_node.named_children:
                if node.type == "import_statement":
                    source_node = node.child_by_field_name("source")
                    if source_node:
                        spec = source_node.text.decode("utf-8").strip("\"'")
                        imported_specs.add(spec)
                elif node.type == "export_statement":
                    # Check for re-export: export { a } from './b'
                    source_node = node.child_by_field_name("source")
                    if source_node:
                        spec = source_node.text.decode("utf-8").strip("\"'")
                        imported_specs.add(spec)

                    decl = node.child_by_field_name("declaration")
                    if decl:
                        name_node = decl.child_by_field_name("name")
                        if name_node:
                            exported_syms.add(name_node.text.decode("utf-8"))
                        elif decl.type == "lexical_declaration":
                            for child in decl.named_children:
                                if child.type == "variable_declarator":
                                    c_name = child.child_by_field_name("name")
                                    if c_name:
                                        exported_syms.add(c_name.text.decode("utf-8"))
            return imported_specs, exported_syms
        except Exception:
            # Fall through to regex parser on any grammar/runtime issue
            pass

    # Regex fallback
    for m in _TS_IMPORT_RE.finditer(source):
        spec = m.group("spec") or m.group("req_spec")
        if spec:
            imported_specs.add(spec.strip())

    for m in _TS_REEXPORT_RE.finditer(source):
        imported_specs.add(m.group(1).strip())

    for m in _TS_EXPORT_RE.finditer(source):
        exported_syms.add(m.group(1).strip())

    return imported_specs, exported_syms


def extract_py_imports_exports(source: str) -> Tuple[Set[Tuple[str, int]], Set[str]]:
    """Extract imports and exports from Python code using the standard library `ast`.

    Returns:
        (imported_modules_with_levels, exported_symbols)
        where imported_modules_with_levels is a set of (module_str, relative_level).
    """
    imported: Set[Tuple[str, int]] = set()
    exported: Set[str] = set()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Fallback to regex for broken/fragmentary Python files
        for m in re.finditer(r"^\s*(?:from\s+(\.*[\w\.]+)\s+import|import\s+([\w\.]+))", source, re.MULTILINE):
            mod1, mod2 = m.groups()
            if mod1:
                dots = len(mod1) - len(mod1.lstrip("."))
                imported.add((mod1.lstrip("."), dots))
            elif mod2:
                imported.add((mod2, 0))
        for m in re.finditer(r"^\s*(?:def|class)\s+([A-Za-z_]\w*)", source, re.MULTILINE):
            if not m.group(1).startswith("_"):
                exported.add(m.group(1))
        return imported, exported

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add((alias.name, 0))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            imported.add((mod, node.level))
            if node.level > 0 and not mod:
                # `from . import foo` -- foo might be a module or symbol
                for alias in node.names:
                    imported.add((alias.name, node.level))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                exported.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id == "__all__" and isinstance(node.value, (ast.List, ast.Tuple)):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                exported.add(elt.value)
                    elif not target.id.startswith("_"):
                        exported.add(target.id)

    return imported, exported


def resolve_ts_specifier(specifier: str, importer_path: str, known_paths: Set[str]) -> Optional[str]:
    """Resolve a relative TypeScript/JavaScript module specifier against known repo paths."""
    if not specifier.startswith("."):
        # External package (e.g. 'react', 'lodash') -- not part of internal repo DAG
        return None

    base_dir = PurePosixPath(_normalize_posix(importer_path)).parent
    target_norm = _normalize_posix(str(base_dir / specifier))

    for suffix in _TS_MODULE_SUFFIXES:
        cand = target_norm + suffix
        if cand in known_paths:
            return cand
    return None


def resolve_py_specifier(module: str, level: int, importer_path: str, known_paths: Set[str]) -> Optional[str]:
    """Resolve a Python import module specifier against known repo paths."""
    norm_importer = _normalize_posix(importer_path)
    importer_dir = PurePosixPath(norm_importer).parent

    if level > 0:
        # Relative import: level 1 is current dir, level 2 is parent dir, etc.
        curr = importer_dir
        for _ in range(level - 1):
            curr = curr.parent

        rel_path = "/".join(module.split(".")) if module else ""
        target_norm = _normalize_posix(str(curr / rel_path)) if rel_path else _normalize_posix(str(curr))
    else:
        # Absolute import from repository root (or top-level package)
        target_norm = _normalize_posix("/".join(module.split(".")))

    for suffix in _PY_MODULE_SUFFIXES:
        cand = target_norm + suffix
        if cand in known_paths:
            return cand

    # If module was e.g. `src.data.loader.PackedLoader`, check parent modules
    parts = target_norm.split("/")
    if len(parts) > 1:
        prefix = "/".join(parts[:-1])
        for suffix in _PY_MODULE_SUFFIXES:
            cand = prefix + suffix
            if cand in known_paths:
                return cand

    return None


@dataclass
class RepoGraph:
    """Directed dependency graph of files in a repository.

    An edge `A -> B` indicates that file `A` depends on file `B` (i.e., `A` imports `B`).
    Consequently, `B` must precede `A` in topologically sorted context packing.
    """

    files: List[str]
    dependencies: Dict[str, Set[str]] = field(default_factory=dict)
    dependents: Dict[str, Set[str]] = field(default_factory=dict)
    exports: Dict[str, Set[str]] = field(default_factory=dict)

    def topological_sort(self) -> List[str]:
        """Return files sorted topologically so definitions precede call sites.

        Circular dependencies are resolved deterministically using path depth
        and lexicographical tie-breaking, guaranteeing termination with all files
        present and zero dropped nodes.
        """
        all_files = set(self.files)
        # unsatisfied_deps[f] = set of dependencies of f that have not yet been emitted
        unsatisfied_deps: Dict[str, Set[str]] = {
            f: {d for d in self.dependencies.get(f, set()) if d in all_files and d != f}
            for f in all_files
        }

        # dependents_map[d] = set of files waiting on d
        dependents_map: Dict[str, Set[str]] = {f: set() for f in all_files}
        for f, d_set in unsatisfied_deps.items():
            for d in d_set:
                dependents_map[d].add(f)

        remaining = set(all_files)
        order: List[str] = []

        while remaining:
            # Nodes with 0 unsatisfied dependencies can be safely scheduled
            ready = [f for f in remaining if len(unsatisfied_deps[f]) == 0]

            if ready:
                # Prioritize definitions that unblock other files, then tie-break by path depth and name
                ready.sort(
                    key=lambda f: (
                        -len(dependents_map[f]),
                        len(PurePosixPath(f).parts),
                        f,
                    )
                )
                chosen = ready[0]
            else:
                # Cycle detected among the remaining nodes!
                # Break cycle by picking the node with the fewest remaining unsatisfied
                # dependencies, unblocking the most dependents, tie-breaking by path depth
                # and lexicographical order.
                chosen = min(
                    remaining,
                    key=lambda f: (
                        len(unsatisfied_deps[f]),
                        -len(dependents_map[f]),
                        len(PurePosixPath(f).parts),
                        f,
                    ),
                )

            remaining.remove(chosen)
            order.append(chosen)

            # Satisfy chosen for all downstream dependents
            for dep in dependents_map[chosen]:
                if dep in remaining:
                    unsatisfied_deps[dep].discard(chosen)

        return order

    def to_dict(self) -> dict:
        """Export graph and topological sort as a JSON-serializable dictionary."""
        return {
            "files": list(self.files),
            "dependencies": {f: sorted(self.dependencies.get(f, set())) for f in self.files},
            "exports": {f: sorted(self.exports.get(f, set())) for f in self.files},
            "topological_order": self.topological_sort(),
        }


def build_repo_graph(source: str | Path | Dict[str, str] | Sequence[dict]) -> RepoGraph:
    """Build a `RepoGraph` from a repo directory, a dict of {path: content}, or a list of [{path, text}].

    Supports both TypeScript/JavaScript and Python multi-file projects.
    """
    file_contents: Dict[str, str] = {}

    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.is_dir():
            for ext in ("*.ts", "*.tsx", "*.js", "*.jsx", "*.py"):
                for file_path in p.rglob(ext):
                    if any(part.startswith(".") or part == "node_modules" for part in file_path.parts):
                        continue
                    try:
                        rel = str(file_path.relative_to(p).as_posix())
                        file_contents[rel] = file_path.read_text(encoding="utf-8")
                    except (UnicodeDecodeError, OSError):
                        continue
        elif p.is_file():
            # If it's a JSONL or JSON manifest
            if p.suffix == ".jsonl":
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            obj = json.loads(line)
                            if "path" in obj and ("text" in obj or "content" in obj):
                                file_contents[_normalize_posix(obj["path"])] = obj.get("text") or obj.get("content") or ""
            else:
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(data, list):
                        for item in data:
                            if "path" in item:
                                file_contents[_normalize_posix(item["path"])] = item.get("text") or item.get("content") or ""
                    elif isinstance(data, dict) and "files" in data:
                        for item in data["files"]:
                            if "path" in item:
                                file_contents[_normalize_posix(item["path"])] = item.get("text") or item.get("content") or ""
                except Exception:
                    pass
    elif isinstance(source, dict):
        for path_key, text in source.items():
            file_contents[_normalize_posix(path_key)] = str(text)
    elif isinstance(source, (list, tuple)):
        for item in source:
            if isinstance(item, dict) and "path" in item:
                file_contents[_normalize_posix(item["path"])] = item.get("text") or item.get("content") or ""

    known_paths = set(file_contents.keys())
    dependencies: Dict[str, Set[str]] = {p: set() for p in known_paths}
    dependents: Dict[str, Set[str]] = {p: set() for p in known_paths}
    exports: Dict[str, Set[str]] = {p: set() for p in known_paths}

    for path, content in file_contents.items():
        if path.endswith((".ts", ".tsx", ".js", ".jsx", ".mts", ".cts")):
            imported_specs, exported_syms = extract_ts_imports_exports(content)
            exports[path] = exported_syms
            for spec in imported_specs:
                resolved = resolve_ts_specifier(spec, path, known_paths)
                if resolved and resolved != path:
                    dependencies[path].add(resolved)
                    dependents[resolved].add(path)
        elif path.endswith(".py"):
            py_imports, py_exports = extract_py_imports_exports(content)
            exports[path] = py_exports
            for mod, level in py_imports:
                resolved = resolve_py_specifier(mod, level, path, known_paths)
                if resolved and resolved != path:
                    dependencies[path].add(resolved)
                    dependents[resolved].add(path)

    return RepoGraph(
        files=sorted(known_paths),
        dependencies=dependencies,
        dependents=dependents,
        exports=exports,
    )


def topological_sort(source: RepoGraph | str | Path | Dict[str, str] | Sequence[dict]) -> List[str]:
    """Return topologically sorted list of files so definitions precede call sites."""
    if isinstance(source, RepoGraph):
        return source.topological_sort()
    return build_repo_graph(source).topological_sort()


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract repository dependency graph and compute topological file ordering.")
    ap.add_argument("--repo", required=True, type=Path, help="path to repository directory or jsonl fixture")
    ap.add_argument("--out", type=Path, default=None, help="optional path to save output JSON manifest")
    args = ap.parse_args()

    graph = build_repo_graph(args.repo)
    result = graph.to_dict()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote repository graph ({len(graph.files)} files) to {args.out}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
