"""Frozen datasets, source pins and the command-line conventions of every script."""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys

from ccg_diffusion import config
from ccg_diffusion.evaluation import tab
from ccg_diffusion.evaluation.t2i_compbench import dataset
from ccg_diffusion.provenance import repository_root

ROOT = repository_root()
HEAVY = ("torch", "transformers", "diffusers", "holccg", "spacy")
# Scripts that run outside the main environment (evaluator venv, remote node).
NOT_CLIS = {"scripts/t2i_compbench/official_reference/generate.py", "scripts/setup/qwen38_flash_next/evict_ckpt.py"}
# The Stage 1-2 CLIs also accept the recorded underscore spellings, as aliases of hyphenated flags.
UNDERSCORE_ALIAS_SCRIPTS = {"scripts/ccg/parse.py", "scripts/ccg/generate_dependencies.py"}
RETIRED_FLAGS = {"--fixture", "--allow_download", "--allow-tokenizer-download", "--allow-model-download",
                 "--local-files-only", "--image-ids", "--prompt-ids", "--batch-indices", "--limit-batches", "--pilot"}


def _scripts() -> list[str]:
    return sorted(path.relative_to(ROOT).as_posix() for path in (ROOT / "scripts").rglob("*.py")
                  if path.relative_to(ROOT).as_posix() not in NOT_CLIS)


def test_t2i_compbench_dataset_verifies_against_its_contract() -> None:
    record = dataset.verify_dataset()
    assert record["dataset_id"] == config.T2I_COMPBENCH_DATASET_ID and record["prompts"] == 2400
    assert record["selection_manifest_sha256"].startswith("959de730") and record["files"] == 16
    assert dataset.DATASET_DIR == ROOT / config.T2I_COMPBENCH_DATASET_DIR


def test_tab_dataset_verifies_against_its_contract() -> None:
    tab.verify_dataset()
    assert len(tab.load_pp_items()) == tab.PP_ITEMS == 74


def test_source_pin_matches_the_evaluator_constants() -> None:
    pin = json.loads((ROOT / dataset.SOURCE_PIN_PATH).read_text())
    assert (pin["repo"], pin["commit"]) == (dataset.SOURCE_REPOSITORY, dataset.SOURCE_COMMIT)
    recorded = {w["name"]: w for w in pin["weights"]}
    for weight in dataset.EVALUATOR_WEIGHTS:
        assert {k: recorded[weight["name"]][k] for k in ("url", "bytes", "sha256", "official_source")} == {
            k: weight[k] for k in ("url", "bytes", "sha256", "official_source")}
    contract = json.loads((dataset.DATASET_DIR / "dataset_contract.json").read_text())
    assert contract["source"]["commit"] == dataset.SOURCE_COMMIT


def test_equal_interval_selection_keeps_every_prompt_in_source_order() -> None:
    from scripts.data.build_t2i_compbench import select_line_indices

    selection = json.loads((dataset.DATASET_DIR / "selection_manifest.json").read_text())
    for index, category in enumerate(dataset.CATEGORIES):
        assert select_line_indices(category_index=index) == list(range(300))
        assert [row["line_index"] for row in selection["selected"][category]] == list(range(300))
    assert selection["selection_policy"] == "equal_interval" and selection["sample_seed"] == 20260812


def test_every_cli_help_runs_without_heavy_imports() -> None:
    modules = [path[:-3].replace("/", ".") for path in _scripts()]
    probe = (
        "import contextlib, io, runpy, sys\n"
        f"heavy = {HEAVY!r}\n"
        "report = {}\n"
        f"for module in {modules!r}:\n"
        "    sys.argv = [module, '--help']\n"
        "    out = io.StringIO()\n"
        "    try:\n"
        "        with contextlib.redirect_stdout(out):\n"
        "            runpy.run_module(module, run_name='__main__', alter_sys=True)\n"
        "        code = 0\n"
        "    except SystemExit as exc:\n"
        "        code = exc.code or 0\n"
        "    except Exception as exc:\n"
        "        code = repr(exc)\n"
        "    report[module] = [code, sorted(set(heavy) & set(sys.modules)), 'usage' in out.getvalue()]\n"
        "print(__import__('json').dumps(report))\n"
    )
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'src'}{os.pathsep}{ROOT}"}
    done = subprocess.run([sys.executable, "-c", probe], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    report = json.loads(done.stdout.strip().splitlines()[-1])
    assert {m: r for m, r in report.items() if r != [0, [], True]} == {}


def test_methods_command_line_imports_no_heavy_package() -> None:
    probe = ("import runpy, sys\nsys.argv = ['methods', 'list']\nrunpy.run_module('ccg_diffusion.methods', "
             "run_name='__main__')\nprint(sorted(set(%r) & set(sys.modules)))" % (HEAVY,))
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'src'}{os.pathsep}{ROOT}"}
    done = subprocess.run([sys.executable, "-c", probe], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    assert done.stdout.strip().splitlines()[-1] == "[]"


def test_flag_conventions() -> None:
    problems = []
    for path in _scripts():
        for node in ast.walk(ast.parse((ROOT / path).read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument":
                flags = [arg.value for arg in node.args
                         if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--")]
                problems += [f"{path}: {flag}" for flag in flags if flag in RETIRED_FLAGS]
                if flags and "_" in flags[0]:
                    problems.append(f"{path}: {flags[0]} (flags use hyphens)")
                if any("_" in flag for flag in flags[1:]) and path not in UNDERSCORE_ALIAS_SCRIPTS:
                    problems.append(f"{path}: {flags} (underscore aliases only in the Stage 1-2 CLIs)")
    assert problems == []
