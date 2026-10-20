"""Run the official T2I-CompBench++ evaluators over a generation run (``--dry-run`` prints the plan only).

    python -m scripts.t2i_compbench.evaluate --run-id RUN --arm ARM [--categories ...] [--skip-completed]
    python -m scripts.t2i_compbench.evaluate --run-id RUN                     # official SD1.4 reference layout

With ``--arm`` the images exported by ``scripts.t2i_compbench.export_samples``
(``output/vqa/RUN/t2icb_work/ARM/<category>/samples``) are scored against the
frozen dataset prompts and the curated results go to
``output/vqa/RUN/t2i_compbench/ARM/<category>/``. Without ``--arm`` the samples
of ``output/generation/RUN/<category>/samples`` are scored against the official
prompt files and compared with the published Stable v1-4 row (``--targets
stable-v1-4``). The official clone, its Python 3.10 venv and the metric weights
are installed by ``scripts/setup/setup_t2i_compbench.py``; every external input
is verified before any evaluator starts. Exit status 1 if any category fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    from ccg_diffusion.evaluation.t2i_compbench import official_evaluator as ev
    from ccg_diffusion.evaluation.t2i_compbench.dataset import CATEGORIES, DATASET_DIR, REPO_ROOT

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True, help="shared generation/vqa run id")
    parser.add_argument("--categories", nargs="+", default=["all"], help=f"one or more of {list(CATEGORIES)}, or all")
    parser.add_argument("--clone-dir", type=Path, default=REPO_ROOT / "external/t2i_compbench/T2I-CompBench",
                        help="official T2I-CompBench++ clone (read-only)")
    parser.add_argument("--venv-python", type=Path, default=REPO_ROOT / "external/t2i_compbench/.venv/bin/python",
                        help="python interpreter of the evaluator venv")
    parser.add_argument("--generation-root", type=Path, default=REPO_ROOT / "output/generation")
    parser.add_argument("--vqa-root", type=Path, default=REPO_ROOT / "output/vqa")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--prompt-dir", type=Path, default=None,
                        help="directory of <category>_val.txt (default: the frozen dataset prompts with --arm, "
                             "else the official clone)")
    parser.add_argument("--arm", default=None,
                        help="score the images exported under t2icb_work/<arm>/ (arm = recorded method or ablation "
                             "condition ID; docs/provenance.md)")
    parser.add_argument("--targets", choices=("stable-v1-4", "none"), default=None,
                        help="published-value comparison gate (default: none with --arm, else stable-v1-4)")
    parser.add_argument("--skip-completed", action="store_true",
                        help="leave categories that already have a verified eval_manifest.json alone")
    parser.add_argument("--dry-run", action="store_true", help="print the planned invocations and exit")
    args = parser.parse_args(argv)

    try:
        categories = ev.resolve_categories(args.categories)
    except ev.T2ICompBenchError as exc:
        parser.error(str(exc))
    prompt_dir = args.prompt_dir or (DATASET_DIR / "prompts" if args.arm else None)
    targets = args.targets or ("none" if args.arm else "stable-v1-4")
    published_target_gate = targets != "none"
    if args.arm is not None and published_target_gate:
        parser.error("arm evaluations require --targets none; the published Stable v1-4 row is the official "
                     "SD1.4 reference only")

    try:
        official_preflight = ev.validate_official_environment(
            repo_root=args.repo_root, clone_dir=args.clone_dir, venv_python=args.venv_python)
    except (ev.T2ICompBenchError, OSError, json.JSONDecodeError) as exc:
        print(f"official evaluator preflight FAILED: {exc}", file=sys.stderr, flush=True)
        return 1

    curated_root = ev.curated_root_dir(args.vqa_root, args.run_id, args.arm)
    failures: list[str] = []
    for category in categories:
        completed_path = curated_root / category / "eval_manifest.json"
        if args.skip_completed and completed_path.is_file():
            try:
                work_dir = ev.work_root_dir(args.vqa_root, args.run_id, args.arm) / category
                prompt_file = (prompt_dir or args.clone_dir / "examples" / "dataset") / f"{category}_val.txt"
                prompts = ev.read_prompts(prompt_file)
                stage_reference = (
                    ev.validate_metric_stage_manifest(
                        work_dir=work_dir, run_id=args.run_id, arm=args.arm, category=category,
                        prompt_file=prompt_file, prompts=prompts, repo_root=args.repo_root)
                    if args.arm is not None else None
                )
                ev.validate_completed_eval_manifest(
                    completed_path, run_id=args.run_id, arm=args.arm, category=category,
                    stage_reference=stage_reference, repo_root=args.repo_root,
                    official_preflight=official_preflight)
            except (ev.T2ICompBenchError, OSError, json.JSONDecodeError) as exc:
                print(f"[{category}] FAILED completed-result integrity check: {exc}", file=sys.stderr, flush=True)
                failures.append(category)
                continue
            print(f"[{category}] sealed result already evaluated, skipping", flush=True)
            continue
        print(f"[{category}] starting", flush=True)
        try:
            result = ev.evaluate_category(
                category, run_id=args.run_id, clone_dir=args.clone_dir, venv_python=args.venv_python,
                generation_root=args.generation_root, vqa_root=args.vqa_root, repo_root=args.repo_root,
                dry_run=args.dry_run, prompt_dir=prompt_dir, published_target_gate=published_target_gate,
                arm=args.arm, official_preflight=official_preflight)
        except (ev.T2ICompBenchError, OSError, json.JSONDecodeError) as exc:
            print(f"[{category}] FAILED: {exc}", file=sys.stderr, flush=True)
            failures.append(category)
            continue
        if args.dry_run:
            for command in result["commands"]:
                print(f"[{category}] ({command['name']}) cd {command['cwd']} && {' '.join(command['argv'])}")
            continue
        if result["target"] is None:
            print(f"[{category}] {result['metric']}={result['score']:.4f} (no published-target gate)", flush=True)
        else:
            verdict = "PASS" if result["pass"] else "FAIL"
            print(f"[{category}] {result['metric']}={result['score']:.4f} target={result['target']:.4f} "
                  f"delta={result['delta']:+.4f} {verdict}", flush=True)

    if not args.dry_run:
        try:
            summary = ev.write_summary(args.vqa_root, args.run_id, arm=args.arm)
        except (ev.T2ICompBenchError, OSError, json.JSONDecodeError) as exc:
            print(f"summary FAILED: {exc}", file=sys.stderr, flush=True)
            return 1
        if summary["n_gated"]:
            print(f"summary: {summary['n_pass']}/{summary['n_gated']} within +/-{ev.GATE_TOLERANCE}", flush=True)
        else:
            print(f"summary: {summary['n_evaluated']} categories scored (no published-target gate)", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
