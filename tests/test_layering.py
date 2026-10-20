"""Import layering of ``src/ccg_diffusion`` (AGENTS.md, "Conventions").

* No module imports ``scripts`` or ``holsd``; every internal import follows the allowed edges.
* Package ``__init__`` files contain no imports, and the module-level import graph is acyclic.
* The light modules import without torch, transformers, diffusers, holccg or spacy (checked in a subprocess).
* The modules shared with the Python 3.11 baseline adapter venv parse as Python 3.11 and avoid heavy imports.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src/ccg_diffusion"
HEAVY = ("torch", "transformers", "diffusers", "holccg", "spacy")
# ``holsd`` is only a recorded schema-ID prefix (docs/provenance.md), never an importable package.
FORBIDDEN_ROOTS = ("scripts", "holsd")

_BASE = ("config", "methods", "provenance", "parsing", "targets", "guidance", "backbones")
_BASELINES = ("config", "provenance", "parsing.ccg", "targets", "guidance", "backbones", "evaluation.tab", "baselines")
# Importer prefix -> importable prefixes (relative to ccg_diffusion); the most specific importer prefix applies.
ALLOWED = {
    "config": (),
    "provenance": (),
    "methods": ("config",),
    "parsing": ("config", "provenance", "parsing"),
    "targets.pd_indices": ("config", "provenance"),
    "targets.tokenizer": ("config", "provenance"),
    "targets.lexicon": ("config", "provenance"),
    "targets.guidance_targets": (
        "config", "provenance", "parsing", "targets.pd_indices", "targets.tokenizer", "targets.lexicon"),
    "guidance": ("provenance", "parsing.ccg", "targets.pd_indices", "targets.lexicon", "targets.tokenizer", "guidance"),
    "backbones.jobs": ("targets.pd_indices",),
    "backbones.sd14": ("config", "methods", "provenance", "targets", "guidance", "backbones.jobs", "backbones.sd14"),
    "backbones.sana": ("config", "provenance", "targets.pd_indices", "guidance", "backbones.sana"),
    "compositional": (*_BASE, "compositional"),
    "single_prompt": (*_BASE, "compositional"),
    "pp_attachment": (*_BASE, "evaluation.tab", "baselines.gold_visual_setup", "pp_attachment"),
    "baselines": _BASELINES,
    "baselines.tab_sana": (*_BASELINES, "pp_attachment.sana"),
    "evaluation": ("config", "methods", "provenance", "backbones.jobs", "evaluation"),
}
FORBIDDEN_EDGES = {
    ("backbones.sd14.records", "backbones.sd14.driver"),
    ("backbones.sd14.inputs", "backbones.sd14.driver"),
}
NO_HEAVY_IMPORTS = (
    "", "config", "methods", "provenance", "parsing.ccg", "parsing.sentences", "parsing.derivation",
    "targets.pd_indices", "targets.guidance_targets", "targets.tokenizer", "targets.lexicon", "backbones.jobs",
    "evaluation.tab", "evaluation.statistics", "evaluation.t2i_compbench.dataset", "evaluation.t2i_compbench.export",
    "evaluation.t2i_compbench.official_evaluator", "evaluation.t2i_compbench.results", "baselines.registry",
    "single_prompt",
)
NO_DIFFUSERS_IMPORT = (
    "backbones.sana.tokens", "compositional.sana", "pp_attachment.sana", "baselines.sana_plans", "baselines.tab_sana",
)
ADAPTER_VENV = (
    "config", "provenance", "targets.tokenizer", "baselines.preprocessing", "baselines.composable_diffusion",
    "baselines.structure_diffusion", "baselines.attend_and_excite", "baselines.syngen", "baselines.registry",
)
ADAPTER_FORBIDDEN = ("backbones", "guidance", "parsing", "compositional", "pp_attachment")


def _module_name(path: Path) -> str:
    parts = path.relative_to(PACKAGE).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


MODULES = {_module_name(path): path for path in sorted(PACKAGE.rglob("*.py"))}
TREES = {name: ast.parse(path.read_text(encoding="utf-8"), filename=str(path)) for name, path in MODULES.items()}


def _resolve(importer: str, node: ast.ImportFrom) -> str:
    """Absolute dotted name of the ``from`` part of an import (``ccg_diffusion`` included)."""

    if not node.level:
        return node.module or ""
    package = importer.split(".") if MODULES[importer].name == "__init__.py" else importer.split(".")[:-1]
    base = ["ccg_diffusion", *[p for p in package if p]][: len(package) + 2 - node.level]
    return ".".join([*base, *([node.module] if node.module else [])])


def _imports(importer: str, nodes: list[ast.stmt] | ast.AST) -> list[str]:
    """Absolute imported module names of ``nodes``; ``from pkg import module`` names the module."""

    names = []
    for node in nodes if isinstance(nodes, list) else ast.walk(nodes):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve(importer, node)
            for alias in node.names:
                candidate = f"{base}.{alias.name}"
                internal = candidate.removeprefix("ccg_diffusion.")
                names.append(candidate if candidate.startswith("ccg_diffusion.") and internal in MODULES else base)
    return names


def _module_level_nodes(tree: ast.Module) -> list[ast.stmt]:
    """Import statements executed at import time (``if TYPE_CHECKING:`` blocks excluded)."""

    found: list[ast.stmt] = []

    def visit(statements: list[ast.stmt]) -> None:
        for statement in statements:
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                found.append(statement)
            elif isinstance(statement, ast.If):
                if not (isinstance(statement.test, ast.Name) and statement.test.id == "TYPE_CHECKING"):
                    visit(statement.body)
                visit(statement.orelse)
            elif isinstance(statement, ast.Try):
                for block in (statement.body, statement.orelse, statement.finalbody):
                    visit(block)
                for handler in statement.handlers:
                    visit(handler.body)
            elif isinstance(statement, ast.With):
                visit(statement.body)

    visit(tree.body)
    return found


def _matches(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def _rule(importer: str) -> tuple[str, ...]:
    keys = [key for key in ALLOWED if _matches(importer, key)]
    assert keys, f"no layering rule for ccg_diffusion.{importer}"
    return ALLOWED[max(keys, key=len)]


def test_no_scripts_or_holsd_imports_and_allowed_edges() -> None:
    violations = []
    for importer, tree in TREES.items():
        if MODULES[importer].name == "__init__.py":
            continue
        allowed = _rule(importer)
        for imported in _imports(importer, tree):
            if imported.split(".")[0] in FORBIDDEN_ROOTS:
                violations.append(f"{importer} imports {imported}")
            elif imported.startswith("ccg_diffusion."):
                target = imported.removeprefix("ccg_diffusion.")
                if (importer, target) in FORBIDDEN_EDGES or not any(_matches(target, prefix) for prefix in allowed):
                    violations.append(f"{importer} imports {target}")
    assert violations == []


def test_package_inits_have_no_imports() -> None:
    for name, path in MODULES.items():
        if path.name == "__init__.py":
            assert not [n for n in ast.walk(TREES[name]) if isinstance(n, (ast.Import, ast.ImportFrom))], name
            assert ast.get_docstring(TREES[name]), name


def test_module_level_import_graph_is_acyclic() -> None:
    graph = {}
    for name, tree in TREES.items():
        targets = {imported.removeprefix("ccg_diffusion.") for imported in _imports(name, _module_level_nodes(tree))
                   if imported.startswith("ccg_diffusion.")}
        graph[name] = {target for target in targets if target in MODULES and target != name}
    state: dict[str, int] = {}

    def visit(node: str, path: list[str]) -> None:
        state[node] = 1
        for target in sorted(graph[node]):
            assert state.get(target) != 1, f"import cycle: {' -> '.join([*path, node, target])}"
            if target not in state:
                visit(target, [*path, node])
        state[node] = 2

    for node in sorted(graph):
        if node not in state:
            visit(node, [])


@pytest.mark.parametrize(("modules", "forbidden"), [(NO_HEAVY_IMPORTS, HEAVY), (NO_DIFFUSERS_IMPORT, ("diffusers",))])
def test_light_modules_import_without_heavy_packages(modules: tuple[str, ...], forbidden: tuple[str, ...]) -> None:
    imports = "; ".join(f"import {'.'.join(filter(None, ('ccg_diffusion', name)))}" for name in modules)
    code = f"import sys; {imports}; print(sorted(set({forbidden!r}) & set(sys.modules)))"
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'src'}{os.pathsep}{ROOT}"}
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]", (modules, result.stdout)


def test_adapter_venv_modules() -> None:
    for name in ADAPTER_VENV:
        source = MODULES[name].read_text(encoding="utf-8")
        ast.parse(source, feature_version=(3, 11))
        for imported in _imports(name, _module_level_nodes(TREES[name])):
            root = imported.split(".")[0]
            internal = imported.removeprefix("ccg_diffusion.")
            assert root not in ("holccg", "transformers"), (name, imported)
            assert not (imported.startswith("ccg_diffusion.") and internal.split(".")[0] in ADAPTER_FORBIDDEN), (
                name, imported)
