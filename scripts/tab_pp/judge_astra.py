"""Judge TAB PP generations with GPT-6 Astra under the frozen protocol (only ``run`` sends paid requests).

    python -m scripts.tab_pp.judge_astra prepare --generation-run RUN
    python -m scripts.tab_pp.judge_astra prepare --run-id JUDGE_RUN --condition ID=RUN:SOURCE_CONDITION[:BACKBONE] ...
    python -m scripts.tab_pp.judge_astra prepare --run-id JUDGE_RUN --conditions-from output/vqa/<run>/preparation.json
    python -m scripts.tab_pp.judge_astra preflight --output DIR
    python -m scripts.tab_pp.judge_astra run --output DIR --budget-usd X [--workers 16] [--max-requests N]
    python -m scripts.tab_pp.judge_astra summarize --output DIR [--seed-counts 4 5] [--bootstrap-seed 20260923]
    python -m scripts.tab_pp.judge_astra replay --output DIR [--seed-count 5] [--expect-sha256 HEX] [--write PATH]

``prepare --generation-run RUN`` judges every condition of the run's own trial
plan into ``output/vqa/RUN/astra``; ``--condition`` and ``--conditions-from``
assemble a judgment run ``output/vqa/JUDGE_RUN`` from several generation runs
(the trial plan of a condition is ``output/vqa/RUN/trial_plan.jsonl``; the
optional BACKBONE is a recorded label, ``SD1.4`` or ``Sana0.6B``). ``run``
needs ``OPENAI_API_KEY`` and a dollar budget. ``replay`` re-derives the
predictions from the stored receipts without network access and prints their
sha256 (the canonical JSONL bytes of ``predictions_<n>seed.jsonl``).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

BACKBONE_LABELS = ("SD1.4", "Sana0.6B")


def parse_condition(spec: str) -> tuple[str, str, str, str | None]:
    """``ID=RUN:SOURCE_CONDITION[:BACKBONE]`` -> ``(id, run, source_condition, backbone)``.

    A source condition may itself contain ``:`` (``<run>:<cell>`` of the SD1.4
    ablation run); only a trailing recorded backbone label is split off.
    """
    condition_id, _, rest = spec.partition("=")
    run, _, source = rest.partition(":")
    backbone = None
    for label in BACKBONE_LABELS:
        if source.endswith(":" + label):
            source, backbone = source[: -len(label) - 1], label
    if not condition_id or not run or not source:
        raise argparse.ArgumentTypeError(f"expected ID=RUN:SOURCE_CONDITION[:BACKBONE], got {spec!r}")
    return condition_id, run, source, backbone


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="seal the trials of one or more generation runs (offline)")
    source = prepare.add_mutually_exclusive_group(required=True)
    source.add_argument("--generation-run", help="judge the run's own trial plan into output/vqa/RUN/astra")
    source.add_argument("--condition", action="append", type=parse_condition,
                        help="ID=RUN:SOURCE_CONDITION[:BACKBONE] (repeatable; needs --run-id)")
    source.add_argument("--conditions-from", type=Path, help="the conditions of a recorded preparation.json")
    prepare.add_argument("--run-id", help="judgment run directory output/vqa/RUN_ID")
    preflight = sub.add_parser("preflight", help="re-verify a prepared judgment run (offline)")
    preflight.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run", help="send the pending paid requests")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--budget-usd", type=float, required=True, help="total spend ceiling of this invocation")
    run.add_argument("--workers", type=int, default=16, help="maximum concurrency (default: 16)")
    run.add_argument("--max-requests", type=int)
    run.add_argument("--launch-interval", type=float, default=.3)
    run.add_argument("--allow-any-snapshot", action="store_true",
                     help="accept a returned model other than gpt-6-astra")
    summarize = sub.add_parser("summarize", help="write predictions, Yes rates and bootstrap CIs (offline)")
    summarize.add_argument("--output", type=Path, required=True)
    summarize.add_argument("--seed-counts", type=int, nargs="+", default=[5])
    summarize.add_argument("--bootstrap-seed", type=int, default=20260923)
    replay = sub.add_parser("replay", help="re-derive the predictions from the stored receipts (offline)")
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--seed-count", type=int, default=5)
    replay.add_argument("--expect-sha256", help="fail unless the replayed predictions have this sha256")
    replay.add_argument("--write", type=Path, help="also write the replayed predictions to this file")
    args = parser.parse_args(argv)

    from ccg_diffusion.evaluation.judges import astra
    from ccg_diffusion.provenance import canonical_json, sha256_bytes

    if args.command == "prepare":
        if args.generation_run:
            run_id = astra.validate_run_id(args.generation_run)
            astra.prepare_run(astra.REPO_ROOT / "output/vqa" / run_id, run_id)
            return 0
        if not args.run_id:
            parser.error("--run-id is required with --condition or --conditions-from")
        run_id = astra.validate_run_id(args.run_id)
        if args.conditions_from:
            conditions = astra.conditions_from(args.conditions_from)
        else:
            conditions = [astra.ConditionSource(id=cid, source_run=source_run, source_condition=condition,
                                                trial_plan=f"output/vqa/{source_run}/trial_plan.jsonl",
                                                backbone=backbone)
                          for cid, source_run, condition, backbone in args.condition]
        with astra.locked(astra.REPO_ROOT / "output/vqa" / run_id):
            astra.prepare(astra.REPO_ROOT / "output/vqa" / run_id, conditions, run_id=run_id)
    elif args.command == "preflight":
        plan, rows = astra.preflight(args.output)
        print(json.dumps({"status": "PASS", "run_id": plan["run_id"], "trials": len(rows),
                          "completed": len(astra.receipts(args.output, rows)), "network_calls": 0}))
    elif args.command == "run":
        if not os.environ.get("OPENAI_API_KEY"):
            parser.error("OPENAI_API_KEY is required for paid API requests")
        result = astra.run(args.output, workers=args.workers, budget_usd=args.budget_usd,
                           launch_interval=args.launch_interval, max_requests=args.max_requests,
                           required_model=None if args.allow_any_snapshot else astra.RETURNED_MODEL)
        print(json.dumps(result, indent=2), flush=True)
    elif args.command == "summarize":
        with astra.locked(args.output):
            summary = astra.summarize(args.output, seed_counts=args.seed_counts, bootstrap_seed=args.bootstrap_seed)
        for score in summary["scores"]:
            print(f"{score['condition']:28} seeds={score['seed_count']} {score['yes']}/{score['trials']} "
                  f"{score['percent']:.1f}%")
    else:
        rows = astra.replay(args.output, seed_count=args.seed_count)
        payload = "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")
        digest = sha256_bytes(payload)
        if args.write:
            args.write.parent.mkdir(parents=True, exist_ok=True)
            args.write.write_bytes(payload)
        print(json.dumps({"output": str(args.output), "predictions": len(rows), "sha256": digest}))
        if args.expect_sha256 and digest != args.expect_sha256:
            print(f"replayed predictions sha256 {digest} differs from {args.expect_sha256}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
