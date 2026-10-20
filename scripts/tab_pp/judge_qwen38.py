"""Judge the TAB PP method-comparison trials with Qwen3.8-Flash-Next served through an OpenAI-compatible endpoint.

    python -m scripts.tab_pp.judge_qwen38 run --base-url URL --run-id RUN
        --astra-run CONDITION=RUN[:SOURCE_CONDITION] ... [--workers 8] [--limit N]
    python -m scripts.tab_pp.judge_qwen38 summarize --run-id RUN | --run-dir DIR

``run`` re-judges the trials of the GPT-6 Astra judgments of the ten TAB PP
method-comparison conditions (``sd14_vanilla``, ..., ``sana_c1r1p0m0``): ``--astra-run`` names,
per condition, the judgment directory under ``output/vqa`` (``RUN`` or
``RUN/astra``) and its condition ID SOURCE_CONDITION (default: CONDITION).
``--run-id`` and one ``--astra-run`` per condition are required in this
repository; the defaults name the paper's judgment runs, which are not
distributed. ``run`` resumes: trials with a stored raw response are not sent
again. The paper run used ``--workers 10`` against the vLLM server of
``scripts/setup/qwen38_flash_next/serve.sh``. ``summarize`` (offline) writes
``predictions.jsonl``, ``summary.json`` and ``scores.csv`` from the stored raw
responses; ``--run-dir`` points it at another directory (e.g. a copy). Astra
trial IDs digest the condition ID, so conditions judged under one condition ID
(``pd_spacy`` of both backbones) are rejected: judge them under their own
condition IDs (``sd14_pd_spacy``, ``sana_pd_spacy``; ``judge_astra prepare --condition``).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

DEFAULT_RUN = "20260927_tab_pp_qwen38fn_xhigh_v1"


def parse_astra_run(spec: str) -> tuple[str, tuple[str, str]]:
    """``CONDITION=RUN[:SOURCE_CONDITION]`` -> ``(condition, (run, source condition))``."""
    condition, _, rest = spec.partition("=")
    run, _, source = rest.partition(":")
    if not condition or not run:
        raise argparse.ArgumentTypeError(f"expected CONDITION=RUN[:SOURCE_CONDITION], got {spec!r}")
    return condition, (run, source or condition)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="judge every pending trial, then summarize")
    run.add_argument("--base-url", required=True, help="the OpenAI-compatible server, e.g. http://HOST:8888")
    run.add_argument("--run-id", help=f"judge directory output/vqa/RUN_ID (required with --astra-run; the default, "
                                      f"the paper run {DEFAULT_RUN}, is not distributed)")
    run.add_argument("--served-model", default="qwen3.8-flash-next")
    run.add_argument("--workers", type=int, default=8)
    run.add_argument("--limit", type=int, help="judge at most N pending trials")
    run.add_argument("--timeout", type=float, default=3600)
    run.add_argument("--retries", type=int, default=4)
    run.add_argument("--astra-run", action="append", type=parse_astra_run, default=[],
                     help="CONDITION=RUN[:SOURCE_CONDITION]: take the Astra trials of a method-comparison condition "
                          "from output/vqa/RUN/predictions_5seed.jsonl (one per condition; the default, the "
                          "paper's runs, is not distributed); the conditions need distinct trial IDs, i.e. "
                          "distinct Astra condition IDs")
    summarize = sub.add_parser("summarize", help="write predictions, summary and scores from the raw responses")
    summarize.add_argument("--run-id", default=DEFAULT_RUN,
                           help="judge directory output/vqa/RUN_ID (the default, the paper run, is not distributed)")
    summarize.add_argument("--run-dir", type=Path, help="judge directory (default: output/vqa/<run-id>)")
    args = parser.parse_args(argv)
    if args.command == "run" and args.run_id is None:
        if args.astra_run:
            parser.error("--run-id is required with --astra-run")
        args.run_id = DEFAULT_RUN

    from ccg_diffusion.evaluation.judges import qwen38

    if args.command == "run":
        qwen38.run(qwen38.REPO_ROOT / "output/vqa" / args.run_id, base_url=args.base_url,
                   served_model=args.served_model, workers=args.workers, limit=args.limit,
                   timeout=args.timeout, retries=args.retries, astra_runs=dict(args.astra_run))
    else:
        qwen38.summarize(args.run_dir or qwen38.REPO_ROOT / "output/vqa" / args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
